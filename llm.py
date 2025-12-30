import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BertTokenizer # 加上 BertTokenizer
from sentence_transformers import CrossEncoder
import numpy as np
from transformers import LogitsProcessor
import torch.nn.functional as F
import math
import json
import logging
import os

# [新增] 尝试导入 LPRAG，如果环境没装也不影响主程序运行
try:
    from lprag_core import PrivacyPerturbator
    LPRAG_AVAILABLE = True
except ImportError:
    LPRAG_AVAILABLE = False
    print("Warning: LPRAG dependencies not found. LPRAG baseline will not work.")

# === [New] DenPAD Core: Density Analyzer ===
class DensityAnalyzer:
    """
    DenPAD component: Loads pre-calculated density map and computes token sensitivity.
    Strictly follows: Sensitivity = 1.0 - Normalized_Density.
    """
    def __init__(self, density_file):
        self.density_map = []
        if density_file and os.path.exists(density_file):
            logging.info(f"Loading density map from {density_file}...")
            try:
                with open(density_file, 'r') as f:
                    self.density_map = json.load(f)
                logging.info(f"Loaded density map for {len(self.density_map)} tokens.")
            except Exception as e:
                logging.error(f"Failed to load density map: {e}")
        else:
            if density_file:
                logging.warning(f"Density file {density_file} not found! DenPAD will default to 0 sensitivity.")

    def get_token_sensitivity(self, token_id):
        """
        Input: token_id (int)
        Output: sensitivity (float) in [0, 1]
        
        Logic: 
        - Density 1.0 (Dense) -> Common word (e.g., 'the') -> Sensitivity 0.0
        - Density 0.0 (Sparse) -> Rare word (e.g., 'Wanda') -> Sensitivity 1.0
        """
        if not self.density_map or token_id < 0 or token_id >= len(self.density_map):
            return 0.0 # Default to low sensitivity if unknown or map not loaded
        
        density = self.density_map[token_id]
        # Invert density: Rare tokens (Low density) -> High Sensitivity
        return 1.0 - density

# === Standard RDP Accountant (Unchanged) ===
class RDPAccountant:
    """
    Tracks cumulative privacy loss using Rényi Differential Privacy (RDP) composition.
    """
    def __init__(self, alpha=10.0, delta=1e-5):
        self.alpha = alpha
        self.delta = delta
        self.rdp = 0.0
        self.steps_with_noise = 0
        self.total_steps = 0

    def add_gaussian_step(self, sensitivity, sigma, noise_injected=True):
        self.total_steps += 1
        if noise_injected:
            self.steps_with_noise += 1
            if sigma > 0:
                # RDP composition for Gaussian mechanism
                self.rdp += (self.alpha * (sensitivity ** 2)) / (2 * (sigma ** 2))

    def get_epsilon(self):
        return self.rdp + np.log(1 / self.delta) / (self.alpha - 1)
    
    def get_gamma(self):
        if self.total_steps == 0:
            return 1.0
        return self.steps_with_noise / self.total_steps

# === Static Noise Processor (Baseline) ===
class StaticNoiseProcessor(LogitsProcessor):
    """
    Baseline: Static uniform noise.
    """
    def __init__(self, epsilon_base=1.0, alpha=10.0, delta=1e-5, noise_scale=0.1):
        self.noise_scale = noise_scale
        self.epsilon_base = epsilon_base
        self.accountant = RDPAccountant(alpha=alpha, delta=delta)
        self.step_count = 0
        self.sensitivity = 1.0 

    def __call__(self, input_ids, scores):
        self.step_count += 1
        noise = torch.randn_like(scores) * self.noise_scale
        self.accountant.add_gaussian_step(sensitivity=self.sensitivity, sigma=self.noise_scale, noise_injected=True)
        return scores + noise

    def get_total_privacy_loss(self):
        return self.accountant.get_epsilon()
    
    def get_gamma(self):
        return self.accountant.get_gamma()

# === PAD Calibrator (Unchanged) ===
class DataDependentCalibrator:
    """
    Original PAD Calibrator: Entropy + Position + Confidence.
    """
    def __init__(self, entropy_weight=0.3, position_weight=0.2):
        self.entropy_weight = entropy_weight
        self.position_weight = position_weight
    
    def calibrate_noise_scale(self, scores, position, base_scale):
        with torch.no_grad():
            probs = F.softmax(scores, dim=-1)
            
            # Normalized entropy
            log_probs = F.log_softmax(scores, dim=-1)
            token_entropy = -(probs * log_probs).sum().item()
            max_entropy = np.log(probs.numel())
            normalized_entropy = token_entropy / max_entropy
            
            # Position factor
            position_factor = 1.0 / (1.0 + position * 0.1)
            
            # Confidence factor
            top1_prob = probs.max().item()
            confidence_factor = 1.0 - top1_prob
            
            calibration_factor = (
                (1 - self.entropy_weight) * 1.0 +
                self.entropy_weight * normalized_entropy +
                self.position_weight * position_factor +
                confidence_factor * 0.3
            )
            
            calibration_factor = max(0.1, min(2.0, calibration_factor))
            return base_scale * calibration_factor

# === PAD Screening (Unchanged) ===
class ScreeningMechanism:
    """
    Original PAD Screening: Skip noise if confident.
    """
    def __init__(self, confidence_threshold=0.9, margin_threshold=2.0):
        self.confidence_threshold = confidence_threshold
        self.margin_threshold = margin_threshold
    
    def should_skip_noise(self, scores):
        probs = F.softmax(scores, dim=-1)
        top1_prob = probs.max().item()
        
        topk = torch.topk(scores, 2, dim=-1).values
        logit_margin = (topk[..., 0] - topk[..., 1]).mean().item()
        
        return top1_prob > self.confidence_threshold and logit_margin > self.margin_threshold

# === DenPAD Adaptive Processor (The Core) ===
class AdaptiveNoiseProcessor(LogitsProcessor):
    """
    DenPAD Processor: Density-Driven Adaptive Noise.
    Integrates PAD's adaptive noise with Density Analysis.
    """
    def __init__(self, epsilon_base=1.0, alpha=10.0, delta=1e-5, 
                 enable_screening=True, enable_calibration=True,
                 density_map_path=None,  # <--- DenPAD param
                 noise_amplification=2.0, min_sensitivity=0.5):
        
        self.base_scale = 0.01 / max(epsilon_base, 0.01)
        self.epsilon_base = epsilon_base
        self.accountant = RDPAccountant(alpha=alpha, delta=delta)
        self.step_count = 0
        
        self.noise_amplification = noise_amplification
        self.min_sensitivity = min_sensitivity
        self.min_sigma = 0.01
        self.max_sigma = 10.0
        
        self.calibrator = DataDependentCalibrator() if enable_calibration else None
        self.screener = ScreeningMechanism() if enable_screening else None
        
        # DenPAD Initialization
        self.density_analyzer = None
        if density_map_path:
            self.density_analyzer = DensityAnalyzer(density_map_path)
            
        self.log_eps = np.log(epsilon_base)

    def __call__(self, input_ids, scores):
        self.step_count += 1
        
        # --- DenPAD Step 1: Look-ahead Density Check ---
        # Predict the most likely token (Greedy look-ahead)
        # We assume the model "wants" to output the Argmax token.
        top_token_id = torch.argmax(scores, dim=-1).item()
        
        semantic_sensitivity = 0.0
        if self.density_analyzer:
            semantic_sensitivity = self.density_analyzer.get_token_sensitivity(top_token_id)
        
        # Define "High Risk" event based on density
        # If sensitivity > 0.5 (meaning Density < 0.5), it is a Rare/Sensitive word.
        # This threshold is heuristic but effective.
        is_high_risk = semantic_sensitivity > 0.5
        
        # --- Step 2: Screening Logic ---
        skip_noise = False
        if self.screener and self.screener.should_skip_noise(scores):
            skip_noise = True
        
        # DenPAD CRITICAL: Override screening if high risk ("Confidence Trap" Mitigation)
        if is_high_risk:
            skip_noise = False 
            
        if skip_noise:
            minimal_noise = torch.randn_like(scores) * self.min_sigma
            self.accountant.add_gaussian_step(sensitivity=0.0, sigma=self.min_sigma, noise_injected=True)
            return scores + minimal_noise
        
        # --- Step 3: Sensitivity Estimation ---
        with torch.no_grad():
            topk = torch.topk(scores, 2, dim=-1).values
            logit_margin = topk[..., 0] - topk[..., 1]
            margin = logit_margin.mean().item()
            
            # Base sensitivity from PAD (Heuristic based on margin)
            base_sensitivity = max(
                self.min_sensitivity,
                min(1.0 / (1 + np.log(1 + max(margin, 1e-6))), 1.0)
            )
            
            # DenPAD Core: Final sensitivity is MAX of (PAD-Est, Density-Sens)
            # If word is rare (high semantic_sensitivity), sensitivity -> 1.0 (Worst Case Bound)
            sensitivity = max(base_sensitivity, semantic_sensitivity)
        
        # --- Step 4: Calibration ---
        if self.calibrator:
            # Calibrate base noise based on Entropy/Position/Confidence
            sigma = self.calibrator.calibrate_noise_scale(scores, self.step_count, self.base_scale)
        else:
            sigma = self.base_scale
            
        # NOTE: Removed manual sigma boosting here to avoid double amplification.
        # The increased 'sensitivity' in Step 3 will naturally increase the final sigma in Step 5.

        # --- Step 5: Injection ---
        # Calculate final sigma required for DP given the sensitivity
        # sigma_final = sigma * (sensitivity / epsilon) * amplification
        sigma_final = sigma * (sensitivity / self.epsilon_base) * self.noise_amplification
        sigma_final = min(self.max_sigma, max(self.min_sigma, sigma_final))
        
        noise = torch.randn_like(scores) * sigma_final
        self.accountant.add_gaussian_step(sensitivity=sensitivity, sigma=sigma_final, noise_injected=True)
        
        return scores + noise

    def get_total_privacy_loss(self):
        return self.accountant.get_epsilon()
    
    def get_gamma(self):
        return self.accountant.get_gamma()

# === LLM Engine (Wrapper) ===
class LLMEngine:
    """
    Language model engine with DenPAD support.
    """
    def __init__(self, model, tokenizer=None, add_noise=False, epsilon=1.0, 
                 alpha=10.0, delta=1e-5, enable_screening=True, 
                 enable_calibration=True,
                 density_map_path=None, # <--- DenPAD Argument
                 enable_lprag=False, # LPRAG 开关
                 lprag_epsilon=3.0,  # LPRAG 的 epsilon
                 noise_amplification=2.0, min_sensitivity=0.5,
                 noise_type="adaptive", static_noise_scale=0.1, verbose=False):
        
        self.model = model
        self.tokenizer = tokenizer
        self.add_noise = add_noise
        self.epsilon = epsilon
        self.noise_type = noise_type
        self.verbose = verbose

        # [新增 LPRAG 初始化]
        self.enable_lprag = enable_lprag
        self.lprag_perturbator = None
        
        if self.enable_lprag:
            if not LPRAG_AVAILABLE:
                raise ValueError("Cannot enable LPRAG: dependencies missing.")
            print("Initializing LPRAG PrivacyPerturbator (this may take time to load Word2Vec/BERT)...")
            # 初始化 LPRAG 核心类，这个类会自动加载 BERT 和 Word2Vec
            self.lprag_perturbator = PrivacyPerturbator(total_epsilon=lprag_epsilon)
        
        if add_noise:
            if noise_type == "static":
                self.noise_processor = StaticNoiseProcessor(
                    epsilon_base=epsilon, alpha=alpha, delta=delta,
                    noise_scale=static_noise_scale
                )
            else:
                self.noise_processor = AdaptiveNoiseProcessor(
                    epsilon_base=epsilon, alpha=alpha, delta=delta,
                    enable_screening=enable_screening,
                    enable_calibration=enable_calibration,
                    density_map_path=density_map_path, # Passed to processor
                    noise_amplification=noise_amplification,
                    min_sensitivity=min_sensitivity
                )
        else:
            self.noise_processor = None

    def generate(self, prompt: str, **decoding_kwargs) -> str:
        if not self.model or not self.tokenizer:
            raise ValueError("Both model and tokenizer must be provided.")
        final_prompt = prompt

        if self.enable_lprag and self.lprag_perturbator:
            # 调用 LPRAG 的 perturb 方法
            # 注意：LPRAG 的 perturb 比较慢，因为它要跑 NER 和 BERT
            try:
                # === [核弹级修复] 基于 BERT Token 的严格截断 ===
                # 1. 临时初始化一个 BERT Tokenizer (和 LPRAG 内部用的一样)
                #    (为了效率，你也可以在 __init__ 里初始化 self.bert_tokenizer)
                bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
                
                # 2. 编码为 ID，不加特殊 token 以便计算长度
                tokenized_ids = bert_tokenizer.encode(prompt, add_special_tokens=False)
                
                # 3. 严格截断到 510 (预留 [CLS] [SEP] 的位置，LPRAG 内部会加)
                if len(tokenized_ids) > 510:
                    tokenized_ids = tokenized_ids[:510]
                    # 4. 解码回字符串
                    truncated_prompt = bert_tokenizer.decode(tokenized_ids, skip_special_tokens=True)
                else:
                    truncated_prompt = prompt

                # 5. 送入 LPRAG (现在绝对安全了)
                final_prompt = self.lprag_perturbator.perturb(truncated_prompt)
                # [可选] 打印对比，看看 LPRAG 到底改了啥
                # if self.verbose:
                #     print(f"[LPRAG] Original: {prompt[:100]}...")
                #     print(f"[LPRAG] Perturbed: {final_prompt[:100]}...")
            except Exception as e:
                # 如果这样还报错，那就是天意了，打印出来看看
                print(f"Error during LPRAG perturbation: {e}")
                # 回退到原始 prompt (Baseline)
                final_prompt = prompt

        # 1. Tokenize the input final_prompt(使用可能被修改过的 final_prompt)
        inputs = self.tokenizer(final_prompt, return_tensors="pt")
        input_ids = inputs["input_ids"]
        
        # === [FIX] OOM / Context Truncation Logic ===
        # Pythia-6.9B max window is 2048. We must ensure input + output fits.
        # We reserve space for max_new_tokens.
        max_new_tokens = decoding_kwargs.get("max_new_tokens", 256)
        # Use model's config limit or default to 2048 if not found
        model_max_length = getattr(self.model.config, "max_position_embeddings", 2048)
        
        # Calculate safe input length (minus a small buffer of 32 for safety)
        safe_max_input_len = model_max_length - max_new_tokens - 32
        
        current_len = input_ids.shape[1]
        if current_len > safe_max_input_len:
            # logging.warning(f"Input length {current_len} exceeds safe limit {safe_max_input_len}. Truncating.")
            # Truncate from the left (keep the end, which usually contains the Question)
            # RAG Prompt: "Context: ... Question: ..." -> We want to keep Question.
            input_ids = input_ids[:, -safe_max_input_len:]
            
            # Update inputs dict for generation
            inputs["input_ids"] = input_ids
            if "attention_mask" in inputs:
                inputs["attention_mask"] = inputs["attention_mask"][:, -safe_max_input_len:]
        
        # Move to device
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        if "pad_token_id" not in decoding_kwargs:
            decoding_kwargs["pad_token_id"] = self.tokenizer.eos_token_id
            
        # Attach Logits Processor
        if self.add_noise and self.noise_processor:
            if "logits_processor" not in decoding_kwargs:
                decoding_kwargs["logits_processor"] = [self.noise_processor]
            else:
                decoding_kwargs["logits_processor"].append(self.noise_processor)
        
        # Generate
        # Note: We pass **inputs directly which contains the truncated input_ids
        output_ids = self.model.generate(**inputs, **decoding_kwargs)
        
        # Decode only the generated part
        generated_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        if self.add_noise and self.verbose:
            epsilon = self.get_total_privacy_loss()
            gamma = self.get_gamma()
            print(f"[DP Log] Cumulative ε: {epsilon:.4f}")
            if gamma is not None:
                print(f"[DP Log] γ: {gamma:.3f}")
        return response

    def get_total_privacy_loss(self):
        if self.add_noise and hasattr(self.noise_processor, "get_total_privacy_loss"):
            return self.noise_processor.get_total_privacy_loss()
        return None
    
    def get_gamma(self):
        if self.add_noise and hasattr(self.noise_processor, "get_gamma"):
            return self.noise_processor.get_gamma()
        return None

# === Standard RAG Pipeline (Unchanged) ===
class RAGPipeline:
    """
    RAG pipeline with privacy protection.
    """
    def __init__(self, retriever, llm, reranker_model: str = "BAAI/bge-reranker-large", device: str = "auto"):
        self.retriever = retriever
        self.llm = llm
        
        if device == "auto":
            reranker_device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            reranker_device = device
            
        self.reranker = CrossEncoder(
            reranker_model,
            device=reranker_device,
        )

    def rerank_contexts(self, question: str, docs, top_n: int = 3):
        if not docs:
            return []

        pairs = [[question, doc.page_content] for doc in docs]
        scores = self.reranker.predict(pairs, show_progress_bar=False)
        doc_scores = list(zip(docs, scores))
        doc_scores.sort(key=lambda x: x[1], reverse=True)

        return [doc for doc, _ in doc_scores[:top_n]]

    def run(self, question: str, k: int = 6, top_n: int = 3, **decoding_kwargs) -> dict:
        docs = self.retriever.similarity_search(question, k=k)
        top_docs = self.rerank_contexts(question, docs, top_n=top_n)
        
        retrieved_text = "\n\n".join(d.page_content for d in top_docs)

        prompt = (
            f"Context:\n{retrieved_text}\n\nQuestion:{question}\nAnswer:\n"
        )
        answer = self.llm.generate(prompt, **decoding_kwargs)

        result = {
            "question": question,
            "context": retrieved_text,
            "answer": answer,
            "retrieved_docs": [d.page_content for d in top_docs],
        }
            
        return result