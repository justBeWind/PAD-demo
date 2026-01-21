import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BertTokenizer
from sentence_transformers import CrossEncoder
import numpy as np
from transformers import LogitsProcessor
import torch.nn.functional as F
import math
import json
import logging
import os

# 尝试导入 LPRAG
try:
    from lprag_core import PrivacyPerturbator
    LPRAG_AVAILABLE = True
except ImportError:
    LPRAG_AVAILABLE = False
    print("Warning: LPRAG dependencies not found. LPRAG baseline will not work.")

# === DenPAD Core: Density Analyzer ===
class DensityAnalyzer:
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
                logging.warning(f"Density file {density_file} not found! Defaulting to 0 sensitivity.")

    def get_token_sensitivity(self, token_id):
        if not self.density_map or token_id < 0 or token_id >= len(self.density_map):
            return 0.0 
        density = self.density_map[token_id]
        # 纯密度驱动核心公式: Sensitivity = 1 - Density
        return 1.0 - density

# === RDP Accountant ===
class RDPAccountant:
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
                self.rdp += (self.alpha * (sensitivity ** 2)) / (2 * (sigma ** 2))

    def get_epsilon(self):
        return self.rdp + np.log(1 / self.delta) / (self.alpha - 1)
    
    def get_gamma(self):
        if self.total_steps == 0:
            return 1.0
        return self.steps_with_noise / self.total_steps

# === Static Noise Processor ===
class StaticNoiseProcessor(LogitsProcessor):
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

# === PAD Calibrator (保留但仅用于非敏感词) ===
class DataDependentCalibrator:
    def __init__(self, entropy_weight=0.3, position_weight=0.2):
        self.entropy_weight = entropy_weight
        self.position_weight = position_weight
    
    def calibrate_noise_scale(self, scores, position, base_scale):
        with torch.no_grad():
            probs = F.softmax(scores, dim=-1)
            log_probs = F.log_softmax(scores, dim=-1)
            token_entropy = -(probs * log_probs).sum().item()
            max_entropy = np.log(probs.numel())
            normalized_entropy = token_entropy / max_entropy
            
            position_factor = 1.0 / (1.0 + position * 0.1)
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

# === PAD Screening ===
class ScreeningMechanism:
    def __init__(self, confidence_threshold=0.9, margin_threshold=2.0):
        self.confidence_threshold = confidence_threshold
        self.margin_threshold = margin_threshold
    
    def should_skip_noise(self, scores):
        probs = F.softmax(scores, dim=-1)
        top1_prob = probs.max().item()
        topk = torch.topk(scores, 2, dim=-1).values
        logit_margin = (topk[..., 0] - topk[..., 1]).mean().item()
        return top1_prob > self.confidence_threshold and logit_margin > self.margin_threshold

# === DenPAD Adaptive Processor (Pure Density Version) ===
class AdaptiveNoiseProcessor(LogitsProcessor):
    def __init__(self, epsilon_base=1.0, alpha=10.0, delta=1e-5, 
                 enable_screening=True, enable_calibration=True,
                 density_map_path=None,  
                 ablation_mode="full", # 这里我们保留参数接口，但逻辑上已经全面倒向 Density
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
        self.ablation_mode = ablation_mode

        self.density_analyzer = None
        if density_map_path:
            self.density_analyzer = DensityAnalyzer(density_map_path)
            
        self.log_eps = np.log(epsilon_base)

    def __call__(self, input_ids, scores):
        self.step_count += 1
        
        # 1. Look-ahead: 获取预测的 Token 及其密度敏感度
        top_token_id = torch.argmax(scores, dim=-1).item()
        density_sensitivity = 0.0
        if self.density_analyzer:
            density_sensitivity = self.density_analyzer.get_token_sensitivity(top_token_id)
        
        # 定义高风险：如果敏感度很高 (对应密度很低)
        # 这里阈值设为 0.5，意味着如果一个词比 50% 的词都稀疏，就开始重点关照
        is_high_risk = density_sensitivity > 0.5 
        
        # 2. Screening (跳过机制)
        skip_noise = False
        if self.screener and self.screener.should_skip_noise(scores):
            skip_noise = True
        
        # [纯密度驱动核心逻辑 1] 
        # 只要是 Density 判定为高风险，绝对不跳过加噪！哪怕模型 Confidence 是 99.9%
        if is_high_risk:
            skip_noise = False 
            
        if skip_noise:
            # 对于低风险且模型自信的词，加极小噪声或者不加
            minimal_noise = torch.randn_like(scores) * self.min_sigma
            self.accountant.add_gaussian_step(sensitivity=0.0, sigma=self.min_sigma, noise_injected=True)
            return scores + minimal_noise
        
        # 3. 敏感度计算 (纯密度驱动)
        # 我们不再关心 PAD 计算出的 conf_sensitivity (logit margin)
        # 我们只信任 Density Map
        
        final_sensitivity = density_sensitivity
        
        # 兜底逻辑：为了避免完全不加噪导致隐私泄露，我们给一个最小敏感度
        # 如果你想要极端的“纯密度”，可以将 min_sensitivity 设得很低。
        # 但通常保留 0.4-0.5 的底线对 Utility 影响不大，对 Privacy 有好处。
        final_sensitivity = max(final_sensitivity, self.min_sensitivity)
        
        # 4. 噪声校准 (Calibration) - 纯密度驱动的关键
        
        if is_high_risk:
            # [纯密度驱动核心逻辑 2] - Bypass PAD Calibrator
            # 如果是敏感实体，我们完全不相信 PAD 的 Calibrator (因为它会因为模型自信而降低噪声)
            # 我们强制使用 Base Scale，甚至为了安全起见，稍微放大 (x1.2 或 x1.5)
            # 这就是你“纯密度”逻辑起作用的地方：Data-Centric Override Model-Centric
            sigma = self.base_scale * 1.5 
        else:
            # 对于非敏感词 (通用词)，我们可以让 PAD 校准器接管，保证生成流畅性 (Utility)
            if self.calibrator:
                sigma = self.calibrator.calibrate_noise_scale(scores, self.step_count, self.base_scale)
            else:
                sigma = self.base_scale
            
        # 5. 最终注入
        # sigma 已经通过 Bypass 逻辑得到了保障，这里再乘上 sensitivity 和 amplification
        sigma_final = sigma * (final_sensitivity / self.epsilon_base) * self.noise_amplification
        sigma_final = min(self.max_sigma, max(self.min_sigma, sigma_final))
        
        noise = torch.randn_like(scores) * sigma_final
        self.accountant.add_gaussian_step(sensitivity=final_sensitivity, sigma=sigma_final, noise_injected=True)
        
        return scores + noise

    def get_total_privacy_loss(self):
        return self.accountant.get_epsilon()
    
    def get_gamma(self):
        return self.accountant.get_gamma()

# === LLM Engine (Wrapper) ===
class LLMEngine:
    def __init__(self, model, tokenizer=None, add_noise=False, epsilon=1.0, 
                 alpha=10.0, delta=1e-5, enable_screening=True, 
                 enable_calibration=True,
                 density_map_path=None, 
                 enable_lprag=False, 
                 lprag_epsilon=3.0, 
                 ablation_mode="full", 
                 noise_amplification=2.0, min_sensitivity=0.5,
                 noise_type="adaptive", static_noise_scale=0.1, verbose=False):
        
        self.model = model
        self.tokenizer = tokenizer
        self.add_noise = add_noise
        self.epsilon = epsilon
        self.noise_type = noise_type
        self.verbose = verbose

        self.enable_lprag = enable_lprag
        self.lprag_perturbator = None
        
        if self.enable_lprag:
            if not LPRAG_AVAILABLE:
                raise ValueError("Cannot enable LPRAG: dependencies missing.")
            print("Initializing LPRAG PrivacyPerturbator...")
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
                    density_map_path=density_map_path, 
                    ablation_mode=ablation_mode,
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
            try:
                bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
                tokenized_ids = bert_tokenizer.encode(prompt, add_special_tokens=False)
                if len(tokenized_ids) > 510:
                    tokenized_ids = tokenized_ids[:510]
                    truncated_prompt = bert_tokenizer.decode(tokenized_ids, skip_special_tokens=True)
                else:
                    truncated_prompt = prompt
                final_prompt = self.lprag_perturbator.perturb(truncated_prompt)
            except Exception as e:
                print(f"Error during LPRAG perturbation: {e}")
                final_prompt = prompt

        inputs = self.tokenizer(final_prompt, return_tensors="pt")
        input_ids = inputs["input_ids"]
        
        max_new_tokens = decoding_kwargs.get("max_new_tokens", 256)
        model_max_length = getattr(self.model.config, "max_position_embeddings", 2048)
        safe_max_input_len = model_max_length - max_new_tokens - 32
        
        current_len = input_ids.shape[1]
        if current_len > safe_max_input_len:
            input_ids = input_ids[:, -safe_max_input_len:]
            inputs["input_ids"] = input_ids
            if "attention_mask" in inputs:
                inputs["attention_mask"] = inputs["attention_mask"][:, -safe_max_input_len:]
        
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        if "pad_token_id" not in decoding_kwargs:
            decoding_kwargs["pad_token_id"] = self.tokenizer.eos_token_id
            
        if self.add_noise and self.noise_processor:
            if "logits_processor" not in decoding_kwargs:
                decoding_kwargs["logits_processor"] = [self.noise_processor]
            else:
                decoding_kwargs["logits_processor"].append(self.noise_processor)
        
        output_ids = self.model.generate(**inputs, **decoding_kwargs)
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
    def __init__(self, retriever, llm, reranker_model: str = "BAAI/bge-reranker-large", device: str = "auto"):
        self.retriever = retriever
        self.llm = llm
        if device == "auto":
            reranker_device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            reranker_device = device
        self.reranker = CrossEncoder(reranker_model, device=reranker_device)

    def rerank_contexts(self, question: str, docs, top_n: int = 3):
        if not docs: return []
        pairs = [[question, doc.page_content] for doc in docs]
        scores = self.reranker.predict(pairs, show_progress_bar=False)
        doc_scores = list(zip(docs, scores))
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in doc_scores[:top_n]]

    def run(self, question: str, k: int = 6, top_n: int = 3, **decoding_kwargs) -> dict:
        docs = self.retriever.similarity_search(question, k=k)
        top_docs = self.rerank_contexts(question, docs, top_n=top_n)
        retrieved_text = "\n\n".join(d.page_content for d in top_docs)
        prompt = (f"Context:\n{retrieved_text}\n\nQuestion:{question}\nAnswer:\n")
        answer = self.llm.generate(prompt, **decoding_kwargs)
        result = {
            "question": question,
            "context": retrieved_text,
            "answer": answer,
            "retrieved_docs": [d.page_content for d in top_docs],
        }
        return result