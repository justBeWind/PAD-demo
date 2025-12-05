import torch
from sentence_transformers import CrossEncoder
import re
import numpy as np
from transformers import LogitsProcessor, BertTokenizer, BertModel
import torch.nn.functional as F
from scipy.stats import entropy
import math
import spacy
import logging

# === [New] Semantic Analyzer from LPRAG core.py ===
class SemanticAnalyzer:
    """
    Transplanted logic from LPRAG's core.py.
    Uses Spacy for NER and BERT for semantic importance (L2 norm) calculation.
    """
    def __init__(self, device='cpu'):
        self.device = device
        logging.info("Initializing SemanticAnalyzer (LPRAG Logic)...")
        
        # Load Spacy model (as used in LPRAG)
        try:
            self.nlp = spacy.load("en_core_web_trf")
        except OSError:
            logging.warning("Spacy model 'en_core_web_trf' not found. Fallback to 'en_core_web_sm' or please install it.")
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except:
                raise ImportError("Please install a spacy model: python -m spacy download en_core_web_trf")

        # Load BERT model (as used in LPRAG)
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased').to(self.device)
        self.bert_model.eval()
        
        # Sensitive entity labels usually associated with PII
        self.sensitive_labels = {
            "PERSON", "ORG", "GPE", "LOC", "DATE", "TIME", 
            "MONEY", "PERCENT", "FAC", "PRODUCT", "EVENT", 
            "LAW", "NORP"
        }

    # # [优化]：移除 DATE, TIME, PERCENT, MONEY 等高频但低敏感度的标签
    #     # 只保留绝对敏感的 PII 类型
    #     self.sensitive_labels = {
    #         "PERSON",  # 人名 (最关键)
    #         "ORG",     # 组织机构 (医院名等)
    #         "GPE",     # 地缘政治实体 (国家/城市)
    #         "LOC",     # 地点
    #         "FAC",     # 设施
    #         "NORP",    # 民族/宗教/政治团体
    #         # "DATE", "TIME", "MONEY", "PERCENT"  <-- 注释掉这些
    #     }
    

    def get_token_sensitivity(self, text_context, current_token_str):
        """
        Analyze the current token within its context to determine sensitivity.
        
        Returns:
            is_sensitive (bool): Whether the token belongs to a sensitive entity.
            semantic_weight (float): The L2 norm of the token embedding (from LPRAG).
        """
        # 1. NER Analysis via Spacy
        # Note: Running Spacy on full text every step is expensive. 
        # Optimization: We analyze a sliding window of the last ~50 tokens.
        window_size = 200 
        analysis_text = text_context[-window_size:] + current_token_str
        
        doc = self.nlp(analysis_text)
        
        is_sensitive = False
        # Check if the *end* of the text (current token) falls into an entity
        # We align by character offset approximately
        if doc.ents:
            # Check the last entity
            last_ent = doc.ents[-1]
            # If the entity is at the very end of the analyzed text
            if last_ent.end_char >= len(analysis_text) - len(current_token_str.strip()):
                if last_ent.label_ in self.sensitive_labels:
                    is_sensitive = True

        # 2. Semantic Importance via BERT (LPRAG logic)
        # Compute L2 norm of the token embedding
        with torch.no_grad():
            inputs = self.bert_tokenizer(current_token_str, return_tensors="pt").to(self.device)
            # Handle empty or special tokens
            if inputs['input_ids'].size(1) == 0:
                return is_sensitive, 1.0
                
            outputs = self.bert_model(**inputs)
            # Get embedding of the token (excluding [CLS] and [SEP])
            # If multiple subwords, we take the mean as per LPRAG style aggregation
            token_embeddings = outputs.last_hidden_state[0, 1:-1, :] 
            if token_embeddings.size(0) > 0:
                norm = torch.norm(token_embeddings.mean(dim=0)).item()
            else:
                norm = 1.0 # Default fallback

        return is_sensitive, norm


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
            # RDP composition for Gaussian mechanism
            self.rdp += (self.alpha * (sensitivity ** 2)) / (2 * (sigma ** 2))

    def get_epsilon(self):
        return self.rdp + np.log(1 / self.delta) / (self.alpha - 1)
    
    def get_gamma(self):
        if self.total_steps == 0:
            return 1.0
        return self.steps_with_noise / self.total_steps


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


class AdaptiveNoiseProcessor(LogitsProcessor):
    """
    The SE-PAD Processor.
    Integrates PAD's adaptive noise with LPRAG's Semantic Analysis.
    """
    def __init__(self, epsilon_base=1.0, alpha=10.0, delta=1e-5, 
                 enable_screening=True, enable_calibration=True,
                 enable_semantics=False,  # <--- New Flag for SE-PAD
                 noise_amplification=2.0, min_sensitivity=0.5,
                 tokenizer=None, device='cpu'): # Need tokenizer to decode text for NER
        
        self.base_scale = 0.01 / max(epsilon_base, 0.01)
        self.epsilon_base = epsilon_base
        self.accountant = RDPAccountant(alpha=alpha, delta=delta)
        self.step_count = 0
        
        self.noise_amplification = noise_amplification
        self.min_sensitivity = min_sensitivity
        self.min_sigma = 0.01
        self.max_sigma = 10.0
        self.tokenizer = tokenizer
        
        self.calibrator = DataDependentCalibrator() if enable_calibration else None
        self.screener = ScreeningMechanism() if enable_screening else None
        
        # === SE-PAD Integration ===
        self.enable_semantics = enable_semantics
        self.semantic_analyzer = None
        if self.enable_semantics:
            if tokenizer is None:
                raise ValueError("Tokenizer must be provided for Semantic Analysis (SE-PAD).")
            self.semantic_analyzer = SemanticAnalyzer(device=device)
            
        self.log_eps = np.log(epsilon_base)

    def __call__(self, input_ids, scores):
        self.step_count += 1
        
        # --- SE-PAD: Semantic Analysis Step ---
        is_sensitive_entity = False
        semantic_norm = 1.0
        
        if self.enable_semantics and self.semantic_analyzer:
            # Decode context (expensive but necessary for NER)
            # We take the last chunk of tokens to form context
            context_text = self.tokenizer.decode(input_ids[0][-50:], skip_special_tokens=True)
            
            # Predict next token (Top-1) to check what we are about to generate
            # This is a 'lookahead' check: if the model *wants* to generate a sensitive entity, we must intervene.
            top_token_id = torch.argmax(scores, dim=-1).item()
            next_token_str = self.tokenizer.decode([top_token_id])
            
            is_sensitive_entity, semantic_norm = self.semantic_analyzer.get_token_sensitivity(context_text, next_token_str)
            
            if is_sensitive_entity:
                # [SE-PAD Logic]: If sensitive, we define it as "High semantic content"
                # LPRAG uses norm to allocate budget. We use norm to boost noise.
                # Heuristic: Higher norm = more distinct/important word = needs more noise to hide.
                # We normalize norm roughly around 1.0 ~ 5.0 range usually.
                pass

        # --- Step 1: Screening (Modified by SE-PAD) ---
        # "Confidence Trap" Mitigation: 
        # If it IS a sensitive entity, we FORCE noise injection, bypassing the screener.
        skip_noise = False
        if self.screener and self.screener.should_skip_noise(scores):
            skip_noise = True
        
        if is_sensitive_entity:
            skip_noise = False # Force override!
            
        if skip_noise:
            minimal_noise = torch.randn_like(scores) * self.min_sigma
            self.accountant.add_gaussian_step(sensitivity=0.0, sigma=self.min_sigma, noise_injected=True)
            return scores + minimal_noise
        
        # --- Step 2: Sensitivity Estimation (Modified by SE-PAD) ---
        with torch.no_grad():
            topk = torch.topk(scores, 2, dim=-1).values
            logit_margin = topk[..., 0] - topk[..., 1]
            margin = logit_margin.mean().item()
            
            sensitivity = max(
                self.min_sensitivity,
                min(1.0 / (1 + np.log(1 + max(margin, 1e-6))), 1.0)
            )
            
            # [SE-PAD Logic]: If sensitive entity, force max sensitivity
            if is_sensitive_entity:
                sensitivity = 1.0 
        
        # --- Step 3: Calibration (Modified by SE-PAD) ---
        if self.calibrator:
            sigma = self.calibrator.calibrate_noise_scale(
                scores, self.step_count, self.base_scale
            )
        else:
            sigma = self.base_scale
        
        # [SE-PAD Logic]: Boost sigma based on semantic norm from LPRAG
        if self.enable_semantics:
            # If sensitive, we boost noise. 
            # We use log(norm) scaling to avoid exploding noise on outliers
            semantic_boost = 1.0
            if is_sensitive_entity:
                # Example: norm 5.0 -> boost ~2.6x
                semantic_boost = 1.0 + np.log(1.0 + semantic_norm) 
            
            sigma = sigma * semantic_boost

        # --- Step 4: Injection ---
        sigma = sigma * (sensitivity / self.epsilon_base) * self.noise_amplification
        sigma = min(self.max_sigma, max(self.min_sigma, sigma))
        
        noise = torch.randn_like(scores) * sigma
        self.accountant.add_gaussian_step(sensitivity=sensitivity, sigma=sigma, noise_injected=True)
        
        if is_sensitive_entity and self.step_count % 10 == 0:
            logging.info(f"[SE-PAD Trigger] Entity detected. Forced Sensitivity=1.0, Sigma={sigma:.2f}")

        return scores + noise

    def get_total_privacy_loss(self):
        return self.accountant.get_epsilon()
    
    def get_gamma(self):
        return self.accountant.get_gamma()


class LLMEngine:
    """
    Language model engine with SE-PAD support.
    """
    def __init__(self, model, tokenizer=None, add_noise=False, epsilon=1.0, 
                 alpha=10.0, delta=1e-5, enable_screening=True, 
                 enable_calibration=True,
                 enable_semantics=True, # <--- Enable SE-PAD by default if noise is on
                 noise_amplification=2.0, min_sensitivity=0.5,
                 noise_type="adaptive", static_noise_scale=0.1, verbose=False):
        
        self.model = model
        self.tokenizer = tokenizer
        self.add_noise = add_noise
        self.epsilon = epsilon
        self.noise_type = noise_type
        self.verbose = verbose
        
        if add_noise:
            if noise_type == "static":
                self.noise_processor = StaticNoiseProcessor(
                    epsilon_base=epsilon, alpha=alpha, delta=delta,
                    noise_scale=static_noise_scale
                )
            else:
                # Initialize Adaptive Processor with SE-PAD capabilities
                self.noise_processor = AdaptiveNoiseProcessor(
                    epsilon_base=epsilon, alpha=alpha, delta=delta,
                    enable_screening=enable_screening,
                    enable_calibration=enable_calibration,
                    enable_semantics=enable_semantics, # Pass flag
                    noise_amplification=noise_amplification,
                    min_sensitivity=min_sensitivity,
                    tokenizer=tokenizer, # Pass tokenizer for NER
                    device=model.device
                )
        else:
            self.noise_processor = None

    def generate(self, prompt: str, **decoding_kwargs) -> str:
        if not self.model or not self.tokenizer:
            raise ValueError("Both model and tokenizer must be provided.")

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
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


class RAGPipeline:
    """
    Standard RAG pipeline (Unchanged).
    """
    def __init__(self, retriever, llm, reranker_model: str = "BAAI/bge-reranker-large", device: str = "auto"):
        self.retriever = retriever
        self.llm = llm
        
        if device == "auto":
            reranker_device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            reranker_device = device
            
        self.reranker = CrossEncoder(
            "BAAI/bge-reranker-large",
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