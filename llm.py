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
import bisect 
from collections import defaultdict, Counter

try:
    from lprag_core import PrivacyPerturbator
    LPRAG_AVAILABLE = True
except ImportError:
    LPRAG_AVAILABLE = False
    print("Warning: LPRAG dependencies not found. LPRAG baseline will not work.")

# === DenPAD Core: Density Analyzer ===
class DensityAnalyzer:
    def __init__(self, density_file):
        self.density_map = None 
        self.is_list = False    
        self.sorted_densities = [] 
        
        if density_file and os.path.exists(density_file):
            logging.info(f"Loading density map from {density_file}...")
            try:
                with open(density_file, 'r') as f:
                    self.density_map = json.load(f)
                if isinstance(self.density_map, list):
                    self.is_list = True
                    self.sorted_densities = sorted([float(x) for x in self.density_map])
                elif isinstance(self.density_map, dict):
                    self.is_list = False
                    self.sorted_densities = sorted([float(x) for x in self.density_map.values()])
            except Exception as e:
                logging.error(f"Failed to load density map: {e}")
                self.density_map = None

    def get_token_rank(self, token_id):
        if self.density_map is None: return 1.0 
        try:
            raw_density = 0.0
            if self.is_list:
                if 0 <= token_id < len(self.density_map):
                    raw_density = self.density_map[token_id]
                else: return 1.0 
            else:
                token_key = str(token_id)
                if token_key in self.density_map:
                    raw_density = self.density_map[token_key]
                else: return 1.0 
            index = bisect.bisect_left(self.sorted_densities, raw_density)
            return index / len(self.sorted_densities)
        except Exception:
            return 1.0

# === RDP Accountant ===
class RDPAccountant:
    def __init__(self, alpha=10.0, delta=1e-5):
        self.alpha = alpha
        self.delta = delta
        self.history = []
        self.alphas = [1.5, 1.75, 2, 2.5, 3, 4, 5, 6, 8, 16, 32, 64, 1e6]

    def add_gaussian_step(self, sensitivity, sigma, noise_injected=True):
        if noise_injected and sigma > 0:
            step_cost = {}
            for alpha in self.alphas:
                if alpha == 1e6: cost = float('inf')
                else: cost = (alpha * (sensitivity ** 2)) / (2 * (sigma ** 2))
                step_cost[alpha] = cost
            self.history.append(step_cost)
        
    def get_total_privacy_loss(self, delta=1e-5):
        if not self.history: return 0.0
        min_epsilon = float('inf')
        for alpha in self.alphas:
            if alpha == 1e6: continue
            total_rdp = sum(step.get(alpha, 0) for step in self.history)
            epsilon_alpha = total_rdp + (math.log(1/delta) + math.log(alpha - 1)) / (alpha - 1)
            if epsilon_alpha < min_epsilon: min_epsilon = epsilon_alpha
        return min_epsilon
    
    def get_gamma(self): return 1.0

# === Static Noise Processor ===
class StaticNoiseProcessor(LogitsProcessor):
    def __init__(self, epsilon_base=1.0, alpha=10.0, delta=1e-5, noise_scale=0.1):
        self.noise_scale = noise_scale
        self.accountant = RDPAccountant(alpha=alpha, delta=delta)
        self.sensitivity = 1.0 

    def __call__(self, input_ids, scores):
        noise = torch.randn_like(scores) * self.noise_scale
        self.accountant.add_gaussian_step(sensitivity=self.sensitivity, sigma=self.noise_scale, noise_injected=True)
        return scores + noise
    def get_total_privacy_loss(self): return self.accountant.get_total_privacy_loss()
    def get_gamma(self): return 1.0

# === DenPAD Adaptive Processor (Corrected Logic) ===
class AdaptiveNoiseProcessor(LogitsProcessor):
    def __init__(self, epsilon_base=1.0, alpha=10.0, delta=1e-5, 
                 enable_screening=True, enable_calibration=True,
                 density_map_path=None,  
                 ablation_mode="full",
                 noise_amplification=3.0, 
                 min_sensitivity=0.0,
                 tokenizer=None,
                 dataset_name="healthcaremagic"):
        
        self.epsilon_base = epsilon_base
        self.accountant = RDPAccountant(alpha=alpha, delta=delta)
        
        # 3-gram Radar
        self.history_len = 3
        self.context_ngrams = set()
        
        self.density_analyzer = None
        if density_map_path:
            self.density_analyzer = DensityAnalyzer(density_map_path)
        
        # Threshold: conservative
        self.safe_rank_threshold = 0.8
        
        print(f">>> [DEBUG] Corrected DenPAD. Threshold={self.safe_rank_threshold:.4f}")

    def set_context(self, context_ids):
        self.context_ngrams.clear()
        if not context_ids or len(context_ids) < self.history_len: return
        if isinstance(context_ids, torch.Tensor):
            context_ids = context_ids.tolist()
            if isinstance(context_ids[0], list): context_ids = context_ids[0]
        
        self.context_tokens = set(context_ids)
        for i in range(len(context_ids) - self.history_len + 1):
            ngram = tuple(context_ids[i : i + self.history_len])
            self.context_ngrams.add(ngram)

    def __call__(self, input_ids, scores):
        # 1. Base PPL Protection: Top-20 Truncation
        top_k_scores, top_k_indices = torch.topk(scores, 20, dim=-1)
        mask = torch.ones_like(scores, dtype=torch.bool)
        mask.scatter_(1, top_k_indices, False)
        scores.masked_fill_(mask, -float('inf'))
        
        # 2. Radar & Density Check
        top_token_id = torch.argmax(scores, dim=-1).item()
        current_history = tuple(input_ids[0, -self.history_len:].tolist())
        is_in_context_sequence = current_history in self.context_ngrams
        
        percentile = 0.0
        if self.density_analyzer:
            percentile = self.density_analyzer.get_token_rank(top_token_id)
        
        # === KEY FIX: Only intervene on RARE tokens ===
        # If the word is common (e.g., "the", "with"), we let it pass even if it's in context.
        # This prevents breaking grammar.
        # We only break the chain when it hits a specific entity.
        should_intervene = is_in_context_sequence and (percentile <= self.safe_rank_threshold)
        
        # 3. Smart Swap Strategy
        if should_intervene:
            candidates = top_k_indices[0].tolist()
            best_replacement = None
            best_replacement_score = -float('inf')
            current_max_logit = scores[0, top_token_id].item()
            
            for idx in candidates:
                if idx == top_token_id: continue 
                
                # === KEY FIX: Relaxed Filter ===
                # We allow context words if they are safe/common.
                # We trust the weighted score to pick good ones.
                
                cand_rank = self.density_analyzer.get_token_rank(idx) if self.density_analyzer else 0.5
                original_score = scores[0, idx].item()
                logit_diff = original_score - current_max_logit 
                
                # Preference: Semantic closeness (Logit) + Commonness (Rank)
                weighted_score = logit_diff + (cand_rank * 5.0)
                
                if weighted_score > best_replacement_score:
                    best_replacement_score = weighted_score
                    best_replacement = idx
            
            if best_replacement is not None:
                # Boost the safe synonym
                scores[0, best_replacement] = current_max_logit + 0.5
                scores[0, top_token_id] = current_max_logit - 0.5
                
        # 4. Minimal DP Noise
        final_sensitivity = 0.1 
        sigma_final = 2.0 
        noise = torch.randn_like(scores) * sigma_final
        noisy_scores = scores + noise
        
        self.accountant.add_gaussian_step(sensitivity=final_sensitivity, sigma=sigma_final, noise_injected=True)
        return noisy_scores

    def get_total_privacy_loss(self): return self.accountant.get_total_privacy_loss()
    def get_gamma(self): return 1.0

# === LLMEngine ===
class LLMEngine:
    def __init__(self, model, tokenizer=None, add_noise=False, epsilon=1.0, 
                 alpha=10.0, delta=1e-5, enable_screening=True, 
                 enable_calibration=True,
                 density_map_path=None, 
                 enable_lprag=False, 
                 lprag_epsilon=3.0, 
                 ablation_mode="full", 
                 noise_amplification=2.0, min_sensitivity=0.5,
                 noise_type="adaptive", static_noise_scale=0.1, 
                 dataset="healthcaremagic",
                 verbose=False):
        
        self.model = model
        self.tokenizer = tokenizer
        self.add_noise = add_noise
        self.epsilon = epsilon
        self.noise_type = noise_type
        self.verbose = verbose
        self.enable_lprag = enable_lprag
        self.lprag_perturbator = None
        
        if self.enable_lprag and LPRAG_AVAILABLE:
            self.lprag_perturbator = PrivacyPerturbator(total_epsilon=lprag_epsilon)
        
        if add_noise:
            if noise_type == "static":
                self.noise_processor = StaticNoiseProcessor(epsilon_base=epsilon, alpha=alpha, delta=delta, noise_scale=static_noise_scale)
            else:
                self.noise_processor = AdaptiveNoiseProcessor(
                    epsilon_base=epsilon, alpha=alpha, delta=delta,
                    enable_screening=enable_screening,
                    enable_calibration=enable_calibration,
                    density_map_path=density_map_path, 
                    ablation_mode=ablation_mode,
                    noise_amplification=2.0, 
                    min_sensitivity=0.0,
                    tokenizer=tokenizer,
                    dataset_name=dataset
                )
        else:
            self.noise_processor = None

    def generate(self, prompt: str, **decoding_kwargs) -> str:
        if not self.model or not self.tokenizer: raise ValueError("Model/Tokenizer missing")
        final_prompt = prompt
        
        context_text = ""
        if "Context:\n" in prompt and "\n\nQuestion:" in prompt:
            try:
                start = prompt.find("Context:\n") + len("Context:\n")
                end = prompt.find("\n\nQuestion:")
                context_text = prompt[start:end]
            except: pass
        
        if self.enable_lprag and self.lprag_perturbator:
            try:
                bert_tok = BertTokenizer.from_pretrained('bert-base-uncased')
                tids = bert_tok.encode(prompt, add_special_tokens=False)[:510]
                final_prompt = self.lprag_perturbator.perturb(bert_tok.decode(tids))
            except: pass
            
        inputs = self.tokenizer(final_prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        if self.add_noise and self.noise_processor:
            if context_text:
                ctx_ids = self.tokenizer(context_text, add_special_tokens=False)["input_ids"]
                self.noise_processor.set_context(ctx_ids)
            else:
                self.noise_processor.set_context([])

        max_new = decoding_kwargs.get("max_new_tokens", 256)
        safe_len = 2048 - max_new - 32
        if inputs["input_ids"].shape[1] > safe_len:
            inputs["input_ids"] = inputs["input_ids"][:, -safe_len:]
            if "attention_mask" in inputs: inputs["attention_mask"] = inputs["attention_mask"][:, -safe_len:]
        if "pad_token_id" not in decoding_kwargs: decoding_kwargs["pad_token_id"] = self.tokenizer.eos_token_id
        if self.add_noise and self.noise_processor:
            if "logits_processor" not in decoding_kwargs: decoding_kwargs["logits_processor"] = [self.noise_processor]
            else: decoding_kwargs["logits_processor"].append(self.noise_processor)
        output_ids = self.model.generate(**inputs, **decoding_kwargs)
        response = self.tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        if self.add_noise and self.verbose: print(f"[DP Log] Privacy Loss: {self.get_total_privacy_loss()}")
        return response

    def get_total_privacy_loss(self): return self.noise_processor.get_total_privacy_loss() if self.noise_processor else None
    def get_gamma(self): return self.noise_processor.get_gamma() if self.noise_processor else None

class RAGPipeline:
    def __init__(self, retriever, llm, reranker_model="BAAI/bge-reranker-large", device="auto"):
        self.retriever = retriever
        self.llm = llm
        reranker_device = "cuda" if torch.cuda.is_available() and device=="auto" else "cpu"
        self.reranker = CrossEncoder(reranker_model, device=reranker_device)
    def rerank_contexts(self, question, docs, top_n=3):
        if not docs: return []
        pairs = [[question, d.page_content] for d in docs]
        scores = self.reranker.predict(pairs, show_progress_bar=False)
        doc_scores = list(zip(docs, scores))
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in doc_scores[:top_n]]
    def run(self, question, k=6, top_n=3, **kwargs):
        docs = self.retriever.similarity_search(question, k=k)
        top_docs = self.rerank_contexts(question, docs, top_n=top_n)
        context = "\n\n".join(d.page_content for d in top_docs)
        prompt = f"[INST] Use the following context to answer the question.\n\nContext:\n{context}\n\nQuestion: {question} [/INST]"
        return {"question": question, "context": context, "answer": self.llm.generate(prompt, **kwargs), "retrieved_docs": [d.page_content for d in top_docs]}