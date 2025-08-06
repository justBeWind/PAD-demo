import torch
from sentence_transformers import CrossEncoder
import re
import numpy as np
from transformers import LogitsProcessor
import torch.nn.functional as F
from scipy.stats import entropy
import math

class RDPAccountant:
    """
    Tracks cumulative privacy loss using Rényi Differential Privacy (RDP) composition for Gaussian mechanisms.
    Converts to (epsilon, delta)-DP guarantee.
    """
    def __init__(self, alpha=10.0, delta=1e-5):
        self.alpha = alpha
        self.delta = delta
        self.rdp = 0.0
        self.steps_with_noise = 0
        self.total_steps = 0

    def add_gaussian_step(self, sensitivity, sigma, noise_injected=True):
        """
        Add the RDP cost of a single Gaussian mechanism step.
        Args:
            sensitivity (float): L2 sensitivity of the step.
            sigma (float): Standard deviation of the Gaussian noise.
            noise_injected (bool): Whether noise was actually injected.
        """
        self.total_steps += 1
        if noise_injected:
            self.steps_with_noise += 1
            self.rdp += (self.alpha * (sensitivity ** 2)) / (2 * (sigma ** 2))

    def get_epsilon(self):
        """
        Convert accumulated RDP to (epsilon, delta)-DP.
        Returns:
            float: epsilon value for the given delta.
        """
        return self.rdp + np.log(1 / self.delta) / (self.alpha - 1)
    
    def get_gamma(self):
        """
        Get the fraction of steps that received full DP protection.
        Returns:
            float: gamma value (fraction of steps with noise injection).
        """
        if self.total_steps == 0:
            return 1.0
        return self.steps_with_noise / self.total_steps


class StaticNoiseProcessor(LogitsProcessor):
    """
    Static noise processor that injects uniform noise to all tokens.
    This serves as a simple baseline for comparison with adaptive methods.
    """
    def __init__(self, epsilon_base=1.0, alpha=10.0, delta=1e-5, noise_scale=0.1):
        self.noise_scale = noise_scale
        self.epsilon_base = epsilon_base
        self.accountant = RDPAccountant(alpha=alpha, delta=delta)
        self.step_count = 0
        
        # Fixed sensitivity for static baseline
        self.sensitivity = 1.0  # Assume maximum sensitivity for all tokens

    def __call__(self, input_ids, scores):
        self.step_count += 1
        
        # Inject uniform noise to all tokens
        noise = torch.randn_like(scores) * self.noise_scale
        self.accountant.add_gaussian_step(sensitivity=self.sensitivity, sigma=self.noise_scale, noise_injected=True)
        
        return scores + noise

    def get_total_privacy_loss(self):
        return self.accountant.get_epsilon()
    
    def get_gamma(self):
        """Get the fraction of steps that received full DP protection."""
        return self.accountant.get_gamma()


class DataDependentCalibrator:
    """
    Calibrates noise scale based on token entropy, position, and confidence.
    """
    def __init__(self, entropy_weight=0.3, position_weight=0.2):
        """
        Args:
            entropy_weight (float): Weight for entropy-based calibration
            position_weight (float): Weight for position-based calibration
        """
        self.entropy_weight = entropy_weight
        self.position_weight = position_weight
    
    def calibrate_noise_scale(self, scores, position, base_scale):
        """
        Calibrate noise scale based on data-dependent factors.
        
        Args:
            scores (torch.Tensor): Token logits
            position (int): Current generation position
            base_scale (float): Base noise scale
            
        Returns:
            float: Calibrated noise scale
        """
        with torch.no_grad():
            probs = F.softmax(scores, dim=-1)
            
            # Compute normalized entropy
            log_probs = F.log_softmax(scores, dim=-1)
            token_entropy = -(probs * log_probs).sum().item()
            max_entropy = np.log(probs.numel())
            normalized_entropy = token_entropy / max_entropy
            
            # Position factor: later positions get less noise
            position_factor = 1.0 / (1.0 + position * 0.1)
            
            # Confidence factor: lower confidence = more noise
            top1_prob = probs.max().item()
            confidence_factor = 1.0 - top1_prob
            
            # Combine factors
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
    Skips noise injection for high-confidence predictions.
    """
    def __init__(self, confidence_threshold=0.9, margin_threshold=2.0):
        """
        Args:
            confidence_threshold (float): Min probability for "safe" prediction
            margin_threshold (float): Min logit margin between top-1 and top-2
        """
        self.confidence_threshold = confidence_threshold
        self.margin_threshold = margin_threshold
    
    def should_skip_noise(self, scores):
        """
        Check if noise injection should be skipped.
        
        Args:
            scores (torch.Tensor): Token logits
            
        Returns:
            bool: True if noise should be skipped
        """
        probs = F.softmax(scores, dim=-1)
        top1_prob = probs.max().item()
        
        topk = torch.topk(scores, 2, dim=-1).values
        logit_margin = (topk[..., 0] - topk[..., 1]).mean().item()
        
        return top1_prob > self.confidence_threshold and logit_margin > self.margin_threshold


class AdaptiveNoiseProcessor(LogitsProcessor):
    """
    Adaptive noise processor for differential privacy in language models.
    """
    def __init__(self, epsilon_base=1.0, alpha=10.0, delta=1e-5, 
                 enable_screening=True, enable_calibration=True,
                 noise_amplification=2.0, min_sensitivity=0.5):
        """
        Args:
            epsilon_base (float): Privacy budget (ε)
            alpha (float): RDP order parameter
            delta (float): Privacy failure probability
            enable_screening (bool): Enable prediction screening
            enable_calibration (bool): Enable adaptive noise calibration
            noise_amplification (float): Noise amplification factor
            min_sensitivity (float): Minimum sensitivity bound
        """
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
        self.log_eps = np.log(epsilon_base)

    def __call__(self, input_ids, scores):
        """
        Add calibrated noise to logits for differential privacy.
        
        Args:
            input_ids (torch.Tensor): Input token IDs
            scores (torch.Tensor): Token logits
            
        Returns:
            torch.Tensor: Noisy logits
        """
        self.step_count += 1
        
        # Apply screening for high-confidence predictions
        if self.screener and self.screener.should_skip_noise(scores):
            minimal_noise = torch.randn_like(scores) * self.min_sigma
            self.accountant.add_gaussian_step(sensitivity=0.0, sigma=self.min_sigma, noise_injected=True)
            return scores + minimal_noise
        
        # Estimate sensitivity from logit margins
        with torch.no_grad():
            topk = torch.topk(scores, 2, dim=-1).values
            logit_margin = topk[..., 0] - topk[..., 1]
            margin = logit_margin.mean().item()
            
            sensitivity = max(
                self.min_sensitivity,
                min(1.0 / (1 + np.log(1 + max(margin, 1e-6))), 1.0)
            )
        
        # Compute noise scale
        if self.calibrator:
            sigma = self.calibrator.calibrate_noise_scale(
                scores, self.step_count, self.base_scale
            )
        else:
            sigma = self.base_scale
        
        sigma = sigma * (sensitivity / self.epsilon_base) * self.noise_amplification
        sigma = min(self.max_sigma, max(self.min_sigma, sigma))
        
        noise = torch.randn_like(scores) * sigma
        self.accountant.add_gaussian_step(sensitivity=sensitivity, sigma=sigma, noise_injected=True)
        
        return scores + noise

    def get_total_privacy_loss(self):
        """Get cumulative privacy loss (ε)."""
        return self.accountant.get_epsilon()
    
    def get_gamma(self):
        """Get fraction of steps with full DP protection."""
        return self.accountant.get_gamma()


class LLMEngine:
    """
    Language model engine with differential privacy support.
    """
    def __init__(self, model, tokenizer=None, add_noise=False, epsilon=1.0, 
                 alpha=10.0, delta=1e-5, enable_screening=True, 
                 enable_calibration=True,
                 noise_amplification=2.0, min_sensitivity=0.5,
                 noise_type="adaptive", static_noise_scale=0.1, verbose=False):
        """
        Args:
            model: Language model
            tokenizer: Model tokenizer
            add_noise (bool): Enable differential privacy
            epsilon (float): Privacy budget
            alpha (float): RDP order parameter
            delta (float): Privacy failure probability
            enable_screening (bool): Enable prediction screening
            enable_calibration (bool): Enable adaptive noise calibration
            noise_amplification (float): Noise amplification factor
            min_sensitivity (float): Minimum sensitivity bound
            noise_type (str): "adaptive" or "static"
            static_noise_scale (float): Noise scale for static injection
        """
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
                self.noise_processor = AdaptiveNoiseProcessor(
                    epsilon_base=epsilon, alpha=alpha, delta=delta,
                    enable_screening=enable_screening,
                    enable_calibration=enable_calibration,
                    noise_amplification=noise_amplification,
                    min_sensitivity=min_sensitivity
                )
        else:
            self.noise_processor = None

    def generate(self, prompt: str, **decoding_kwargs) -> str:
        """
        Generate text with optional differential privacy.
        
        Args:
            prompt (str): Input prompt
            **decoding_kwargs: Generation arguments
            
        Returns:
            str: Generated text
        """
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
            gamma = self.get_gamma() if hasattr(self.noise_processor, 'get_gamma') else None
            print(f"[DP Log] Cumulative ε: {epsilon:.4f}")
            if gamma is not None:
                print(f"[DP Log] γ: {gamma:.3f}")
        return response

    def get_total_privacy_loss(self):
        """Get total privacy loss."""
        if self.add_noise and hasattr(self.noise_processor, "get_total_privacy_loss"):
            return self.noise_processor.get_total_privacy_loss()
        return None
    
    def get_gamma(self):
        """Get fraction of steps with full DP protection."""
        if self.add_noise and hasattr(self.noise_processor, "get_gamma"):
            return self.noise_processor.get_gamma()
        return None


class RAGPipeline:
    """
    RAG pipeline with privacy protection.
    """
    def __init__(self, retriever, llm, reranker_model: str = "BAAI/bge-reranker-large", device: str = "auto"):
        """
        Args:
            retriever: Document retriever
            llm: Language model engine
            reranker_model (str): Reranker model name
            device (str): Device for reranker
        """
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
        """
        Rerank documents by relevance.
        
        Args:
            question (str): User question
            docs: Retrieved documents
            top_n (int): Number of top documents
            
        Returns:
            list: Top-n most relevant documents
        """
        if not docs:
            return []

        pairs = [[question, doc.page_content] for doc in docs]
        scores = self.reranker.predict(pairs, show_progress_bar=False)
        doc_scores = list(zip(docs, scores))
        doc_scores.sort(key=lambda x: x[1], reverse=True)

        return [doc for doc, _ in doc_scores[:top_n]]

    def run(self, question: str, k: int = 6, top_n: int = 3, **decoding_kwargs) -> dict:
        """
        Execute RAG pipeline.
        
        Args:
            question (str): User question
            k (int): Number of documents to retrieve
            top_n (int): Number of documents for generation
            **decoding_kwargs: Generation arguments
            
        Returns:
            dict: Pipeline results
        """
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

    def run_with_privacy_report(self, question: str, **decoding_kwargs):
        """
        Run pipeline with privacy metrics.
        
        Args:
            question (str): User question
            **decoding_kwargs: Generation arguments
            
        Returns:
            dict: Results with privacy metrics
        """
        result = self.run(question, **decoding_kwargs)
        
        epsilon = self.llm.get_total_privacy_loss()
        result["epsilon"] = epsilon
        
        delta = None
        if hasattr(self.llm, "noise_processor") and hasattr(self.llm.noise_processor, "accountant"):
            delta = self.llm.noise_processor.accountant.delta
        result["delta"] = delta
        
        return result


def log_result(result, save_path="privacy_results.jsonl"):
    """Log results to JSONL file."""
    import json
    with open(save_path, "a") as f:
        json.dump(result, f, ensure_ascii=False)
        f.write("\n")