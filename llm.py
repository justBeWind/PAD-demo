import bisect
import json
import logging
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from sentence_transformers import CrossEncoder
from transformers import LogitsProcessor

from denpad_latent import LatentContextDecoder
from denpad_rf import RiskFusionDecoder


class PADRDPAccountant:
    """
    Original PAD-style RDP accountant with step coverage tracking.
    """

    def __init__(self, alpha=10.0, delta=1e-5):
        self.alpha = alpha
        self.delta = delta
        self.rdp = 0.0
        self.steps_with_noise = 0
        self.total_steps = 0

    def add_gaussian_step(self, sensitivity, sigma, noise_injected=True):
        self.total_steps += 1
        if noise_injected and sigma > 0:
            self.steps_with_noise += 1
            self.rdp += (self.alpha * (sensitivity ** 2)) / (2 * (sigma ** 2))

    def get_total_privacy_loss(self):
        return self.rdp + np.log(1 / self.delta) / max(self.alpha - 1, 1e-9)

    def get_gamma(self):
        if self.total_steps == 0:
            return 1.0
        return self.steps_with_noise / self.total_steps

    def reset(self):
        self.rdp = 0.0
        self.steps_with_noise = 0
        self.total_steps = 0


class StaticNoiseProcessor(LogitsProcessor):
    """
    Static Gaussian noise baseline kept for ablations.
    """

    def __init__(self, epsilon_base=1.0, alpha=10.0, delta=1e-5, noise_scale=0.1):
        self.noise_scale = noise_scale
        self.accountant = PADRDPAccountant(alpha=alpha, delta=delta)
        self.sensitivity = 1.0

    def __call__(self, input_ids, scores):
        noise = torch.randn_like(scores) * self.noise_scale
        self.accountant.add_gaussian_step(
            sensitivity=self.sensitivity,
            sigma=self.noise_scale,
            noise_injected=True,
        )
        return scores + noise

    def get_total_privacy_loss(self):
        return self.accountant.get_total_privacy_loss()

    def get_gamma(self):
        return self.accountant.get_gamma()

    def reset(self):
        self.accountant.reset()


class PADDataDependentCalibrator:
    """
    Original PAD calibrator: entropy + position + confidence.
    """

    def __init__(self, entropy_weight=0.3, position_weight=0.2):
        self.entropy_weight = entropy_weight
        self.position_weight = position_weight

    def calibrate_noise_scale(self, scores, position, base_scale):
        with torch.no_grad():
            probs = F.softmax(scores, dim=-1)
            log_probs = F.log_softmax(scores, dim=-1)
            token_entropy = -(probs * log_probs).sum().item()
            max_entropy = np.log(probs.numel())
            normalized_entropy = token_entropy / max(max_entropy, 1e-9)

            position_factor = 1.0 / (1.0 + position * 0.1)
            top1_prob = probs.max().item()
            confidence_factor = 1.0 - top1_prob

            calibration_factor = (
                (1 - self.entropy_weight) * 1.0
                + self.entropy_weight * normalized_entropy
                + self.position_weight * position_factor
                + confidence_factor * 0.3
            )
            calibration_factor = max(0.1, min(2.0, calibration_factor))
            return base_scale * calibration_factor


class PADScreeningMechanism:
    """
    Original PAD confidence-based screening.
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


class PADAdaptiveNoiseProcessor(LogitsProcessor):
    """
    Original PAD adaptive decoding defense.
    """

    def __init__(
        self,
        epsilon_base=1.0,
        alpha=10.0,
        delta=1e-5,
        enable_screening=True,
        enable_calibration=True,
        noise_amplification=2.0,
        min_sensitivity=0.5,
    ):
        self.base_scale = 0.01 / max(epsilon_base, 0.01)
        self.epsilon_base = epsilon_base
        self.accountant = PADRDPAccountant(alpha=alpha, delta=delta)
        self.step_count = 0
        self.noise_amplification = noise_amplification
        self.min_sensitivity = min_sensitivity
        self.min_sigma = 0.01
        self.max_sigma = 10.0
        self.calibrator = PADDataDependentCalibrator() if enable_calibration else None
        self.screener = PADScreeningMechanism() if enable_screening else None

    def __call__(self, input_ids, scores):
        self.step_count += 1

        if self.screener and self.screener.should_skip_noise(scores):
            minimal_noise = torch.randn_like(scores) * self.min_sigma
            self.accountant.add_gaussian_step(
                sensitivity=0.0,
                sigma=self.min_sigma,
                noise_injected=True,
            )
            return scores + minimal_noise

        with torch.no_grad():
            topk = torch.topk(scores, 2, dim=-1).values
            logit_margin = topk[..., 0] - topk[..., 1]
            margin = logit_margin.mean().item()
            sensitivity = max(
                self.min_sensitivity,
                min(1.0 / (1 + np.log(1 + max(margin, 1e-6))), 1.0),
            )

        if self.calibrator:
            sigma = self.calibrator.calibrate_noise_scale(scores, self.step_count, self.base_scale)
        else:
            sigma = self.base_scale

        sigma = sigma * (sensitivity / self.epsilon_base) * self.noise_amplification
        sigma = min(self.max_sigma, max(self.min_sigma, sigma))

        noise = torch.randn_like(scores) * sigma
        self.accountant.add_gaussian_step(
            sensitivity=sensitivity,
            sigma=sigma,
            noise_injected=True,
        )
        return scores + noise

    def get_total_privacy_loss(self):
        return self.accountant.get_total_privacy_loss()

    def get_gamma(self):
        return self.accountant.get_gamma()

    def reset(self):
        self.step_count = 0
        self.accountant.reset()


class DenPADRDPAccountant:
    """
    Multi-alpha accountant kept for the current DenPAD branch.
    """

    def __init__(self, alpha=10.0, delta=1e-5):
        self.alpha = alpha
        self.delta = delta
        self.history = []
        self.alphas = [1.5, 1.75, 2, 2.5, 3, 4, 5, 6, 8, 16, 32, 64, 1e6]

    def add_gaussian_step(self, sensitivity, sigma, noise_injected=True):
        if noise_injected and sigma > 0:
            step_cost = {}
            for alpha in self.alphas:
                if alpha == 1e6:
                    cost = float("inf")
                else:
                    cost = (alpha * (sensitivity ** 2)) / (2 * (sigma ** 2))
                step_cost[alpha] = cost
            self.history.append(step_cost)

    def get_total_privacy_loss(self, delta=None):
        if not self.history:
            return 0.0

        target_delta = self.delta if delta is None else delta
        min_epsilon = float("inf")
        for alpha in self.alphas:
            if alpha == 1e6:
                continue
            total_rdp = sum(step.get(alpha, 0) for step in self.history)
            epsilon_alpha = total_rdp + (
                math.log(1 / target_delta) + math.log(alpha - 1)
            ) / (alpha - 1)
            if epsilon_alpha < min_epsilon:
                min_epsilon = epsilon_alpha
        return min_epsilon

    def get_gamma(self):
        return 1.0

    def reset(self):
        self.history = []


class DensityAnalyzer:
    """
    Loads a token density map for the DenPAD branch.
    """

    def __init__(self, density_file):
        self.density_map = None
        self.is_list = False
        self.sorted_densities = []
        if density_file and os.path.exists(density_file):
            try:
                with open(density_file, "r", encoding="utf-8") as f:
                    self.density_map = json.load(f)
                if isinstance(self.density_map, list):
                    self.is_list = True
                    self.sorted_densities = sorted(float(x) for x in self.density_map)
                elif isinstance(self.density_map, dict):
                    self.sorted_densities = sorted(float(x) for x in self.density_map.values())
            except Exception as exc:
                logging.warning("Failed to load density map %s: %s", density_file, exc)
                self.density_map = None

    def get_token_rank(self, token_id):
        if self.density_map is None:
            return 1.0

        try:
            if self.is_list:
                if 0 <= token_id < len(self.density_map):
                    raw_density = float(self.density_map[token_id])
                else:
                    return 1.0
            else:
                token_key = str(token_id)
                if token_key not in self.density_map:
                    return 1.0
                raw_density = float(self.density_map[token_key])

            index = bisect.bisect_left(self.sorted_densities, raw_density)
            return index / max(len(self.sorted_densities), 1)
        except Exception:
            return 1.0


class DenPADAdaptiveNoiseProcessor(LogitsProcessor):
    """
    Current DenPAD decoding branch preserved as an independent method.
    """

    def __init__(
        self,
        epsilon_base=1.0,
        alpha=10.0,
        delta=1e-5,
        enable_screening=True,
        enable_calibration=True,
        density_map_path=None,
        ablation_mode="full",
        noise_amplification=3.0,
        min_sensitivity=0.0,
        tokenizer=None,
        dataset_name="healthcaremagic",
    ):
        self.epsilon_base = epsilon_base
        self.accountant = DenPADRDPAccountant(alpha=alpha, delta=delta)
        self.history_len = 3
        self.context_ngrams = set()
        self.context_tokens = set()
        self.density_analyzer = DensityAnalyzer(density_map_path)
        self.ablation_mode = ablation_mode
        self.noise_amplification = noise_amplification
        self.min_sensitivity = min_sensitivity
        self.tokenizer = tokenizer
        self.dataset_name = dataset_name
        self.safe_rank_threshold = 0.8
        self.base_scale = 0.01 / max(epsilon_base, 0.01)
        self.min_sigma = 0.01
        self.max_sigma = 2.0

    def set_context(self, context_ids):
        self.context_ngrams.clear()
        self.context_tokens.clear()
        if not context_ids:
            return

        if isinstance(context_ids, torch.Tensor):
            context_ids = context_ids.tolist()
            if context_ids and isinstance(context_ids[0], list):
                context_ids = context_ids[0]

        if len(context_ids) < self.history_len:
            return

        self.context_tokens = set(context_ids)
        for i in range(len(context_ids) - self.history_len + 1):
            ngram = tuple(context_ids[i : i + self.history_len])
            self.context_ngrams.add(ngram)

    def __call__(self, input_ids, scores):
        top_k_scores, top_k_indices = torch.topk(scores, 20, dim=-1)
        mask = torch.ones_like(scores, dtype=torch.bool)
        mask.scatter_(1, top_k_indices, False)
        scores.masked_fill_(mask, -float("inf"))

        top_token_id = torch.argmax(scores, dim=-1).item()
        current_history = tuple(input_ids[0, -self.history_len :].tolist())
        is_in_context_sequence = current_history in self.context_ngrams

        if self.density_analyzer is not None:
            percentile = self.density_analyzer.get_token_rank(top_token_id)
        else:
            percentile = 1.0

        should_intervene = is_in_context_sequence and (percentile <= self.safe_rank_threshold)
        sensitivity = 0.0

        if should_intervene:
            candidates = top_k_indices[0].tolist()
            best_replacement = None
            best_replacement_score = -float("inf")
            current_max_logit = scores[0, top_token_id].item()

            for idx in candidates:
                if idx == top_token_id:
                    continue

                cand_rank = (
                    self.density_analyzer.get_token_rank(idx)
                    if self.density_analyzer is not None
                    else 0.5
                )
                original_score = scores[0, idx].item()
                logit_diff = original_score - current_max_logit
                weighted_score = logit_diff + (cand_rank * 5.0)

                if weighted_score > best_replacement_score:
                    best_replacement_score = weighted_score
                    best_replacement = idx

            if best_replacement is not None:
                scores[0, best_replacement] = current_max_logit + 0.5
                scores[0, top_token_id] = current_max_logit - 0.5
                sensitivity = max(self.min_sensitivity, 0.1)

        if sensitivity > 0:
            sigma_final = self.base_scale * (sensitivity / max(self.epsilon_base, 0.01))
            sigma_final = sigma_final * self.noise_amplification
            sigma_final = min(self.max_sigma, max(self.min_sigma, sigma_final))
        else:
            sigma_final = self.min_sigma

        noise = torch.randn_like(scores) * sigma_final
        noisy_scores = scores + noise
        self.accountant.add_gaussian_step(
            sensitivity=sensitivity,
            sigma=sigma_final,
            noise_injected=True,
        )
        return noisy_scores

    def get_total_privacy_loss(self):
        return self.accountant.get_total_privacy_loss()

    def get_gamma(self):
        return self.accountant.get_gamma()

    def reset(self):
        self.accountant.reset()
        self.context_ngrams.clear()
        self.context_tokens.clear()


class LLMEngine:
    """
    Unified generation wrapper with mutually exclusive methods.
    """

    def __init__(
        self,
        model,
        tokenizer=None,
        method="baseline",
        epsilon=1.0,
        alpha=10.0,
        delta=1e-5,
        enable_screening=True,
        enable_calibration=True,
        density_map_path=None,
        ablation_mode="full",
        noise_amplification=2.0,
        min_sensitivity=0.5,
        noise_type="adaptive",
        static_noise_scale=0.1,
        dataset="healthcaremagic",
        verbose=False,
        denpad_group_betas=None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.method = method
        self.epsilon = epsilon
        self.noise_type = noise_type
        self.verbose = verbose
        self.fusion_decoder = None
        self.latent_decoder = None
        self.last_privacy_loss = None
        self.last_gamma = None
        self.last_latent_stats = None

        if self.tokenizer is not None and self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if method == "baseline" or method == "lprag":
            self.noise_processor = None
        elif method == "denpad":
            self.noise_processor = None
            self.latent_decoder = LatentContextDecoder(
                model=model,
                tokenizer=tokenizer,
                epsilon=epsilon,
                alpha=alpha,
                delta=delta,
                verbose=verbose,
            )
        elif method == "contextpad":
            self.noise_processor = None
            self.fusion_decoder = RiskFusionDecoder(
                model=model,
                tokenizer=tokenizer,
                alpha=alpha,
                delta=delta,
                group_betas=denpad_group_betas,
                verbose=verbose,
            )
        elif method == "pad":
            if noise_type == "static":
                self.noise_processor = StaticNoiseProcessor(
                    epsilon_base=epsilon,
                    alpha=alpha,
                    delta=delta,
                    noise_scale=static_noise_scale,
                )
            else:
                self.noise_processor = PADAdaptiveNoiseProcessor(
                    epsilon_base=epsilon,
                    alpha=alpha,
                    delta=delta,
                    enable_screening=enable_screening,
                    enable_calibration=enable_calibration,
                    noise_amplification=noise_amplification,
                    min_sensitivity=min_sensitivity,
                )
        else:
            raise ValueError(f"Unsupported method: {method}")

    def generate(self, prompt: str, **decoding_kwargs) -> str:
        if not self.model or not self.tokenizer:
            raise ValueError("Model/Tokenizer missing")
        if self.method in {"denpad", "contextpad"}:
            raise ValueError(f"method={self.method} requires generate_with_views(), not generate(prompt).")

        if self.noise_processor and hasattr(self.noise_processor, "reset"):
            self.noise_processor.reset()

        context_text = ""
        if "Context:\n" in prompt and "\n\nQuestion:" in prompt:
            try:
                start = prompt.find("Context:\n") + len("Context:\n")
                end = prompt.find("\n\nQuestion:")
                context_text = prompt[start:end]
            except Exception:
                context_text = ""

        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        if self.noise_processor and hasattr(self.noise_processor, "set_context"):
            if context_text:
                ctx_ids = self.tokenizer(context_text, add_special_tokens=False)["input_ids"]
                self.noise_processor.set_context(ctx_ids)
            else:
                self.noise_processor.set_context([])

        max_new = decoding_kwargs.get("max_new_tokens", 256)
        safe_len = 2048 - max_new - 32
        if inputs["input_ids"].shape[1] > safe_len:
            inputs["input_ids"] = inputs["input_ids"][:, -safe_len:]
            if "attention_mask" in inputs:
                inputs["attention_mask"] = inputs["attention_mask"][:, -safe_len:]

        if "pad_token_id" not in decoding_kwargs:
            decoding_kwargs["pad_token_id"] = self.tokenizer.eos_token_id

        if self.noise_processor:
            if "logits_processor" not in decoding_kwargs:
                decoding_kwargs["logits_processor"] = [self.noise_processor]
            else:
                decoding_kwargs["logits_processor"].append(self.noise_processor)

        output_ids = self.model.generate(**inputs, **decoding_kwargs)
        response = self.tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )

        if self.noise_processor and self.verbose:
            print(f"[DP Log] Privacy Loss: {self.get_total_privacy_loss()}")
        return response

    def generate_with_views(self, question: str, context_views: dict[str, list[str]], view_summaries: dict[str, dict], **decoding_kwargs) -> str:
        if self.method != "contextpad" or self.fusion_decoder is None:
            raise ValueError("generate_with_views() is only available for method=contextpad.")
        answer, stats = self.fusion_decoder.generate(
            question=question,
            context_views=context_views,
            group_summaries=view_summaries,
            max_new_tokens=decoding_kwargs.get("max_new_tokens", 256),
            temperature=decoding_kwargs.get("temperature", 0.2),
            top_p=decoding_kwargs.get("top_p", 0.9),
            do_sample=decoding_kwargs.get("do_sample", True),
            repetition_penalty=decoding_kwargs.get("repetition_penalty", 1.0),
        )
        self.last_privacy_loss = stats.get("epsilon_global")
        self.last_gamma = stats.get("avg_lambda")
        self.last_fusion_stats = stats
        return answer

    def generate_with_latent(self, question: str, docs: list[str], sanitization_metadata: dict, **decoding_kwargs) -> str:
        if self.method != "denpad" or self.latent_decoder is None:
            raise ValueError("generate_with_latent() is only available for method=denpad.")
        answer, stats = self.latent_decoder.generate(
            question=question,
            docs=docs,
            sanitization_metadata=sanitization_metadata,
            max_new_tokens=decoding_kwargs.get("max_new_tokens", 256),
            temperature=decoding_kwargs.get("temperature", 0.2),
            top_p=decoding_kwargs.get("top_p", 0.9),
            do_sample=decoding_kwargs.get("do_sample", True),
            repetition_penalty=decoding_kwargs.get("repetition_penalty", 1.0),
        )
        self.last_privacy_loss = stats.get("epsilon_global")
        self.last_gamma = None
        self.last_latent_stats = stats
        return answer

    def get_total_privacy_loss(self):
        if self.noise_processor:
            return self.noise_processor.get_total_privacy_loss()
        if self.method in {"denpad", "contextpad"}:
            return self.last_privacy_loss
        return None

    def get_gamma(self):
        if self.noise_processor:
            return self.noise_processor.get_gamma()
        if self.method in {"denpad", "contextpad"}:
            return self.last_gamma
        return None


class RAGPipeline:
    def __init__(
        self,
        retriever,
        llm,
        reranker_model="BAAI/bge-reranker-large",
        device="auto",
        use_reranker=True,
        context_sanitizer=None,
    ):
        self.retriever = retriever
        self.llm = llm
        self.use_reranker = use_reranker
        self.context_sanitizer = context_sanitizer
        self.reranker = None
        if self.use_reranker:
            reranker_device = "cuda" if torch.cuda.is_available() and device == "auto" else "cpu"
            self.reranker = CrossEncoder(reranker_model, device=reranker_device)

    def rerank_contexts(self, question, docs, top_n=3):
        if not docs:
            return []
        if not self.use_reranker or self.reranker is None:
            return docs[:top_n]
        pairs = [[question, d.page_content] for d in docs]
        scores = self.reranker.predict(pairs, show_progress_bar=False)
        doc_scores = list(zip(docs, scores))
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in doc_scores[:top_n]]

    def run(self, question, k=6, top_n=3, **kwargs):
        retrieval_start = time.perf_counter()
        docs = self.retriever.similarity_search(question, k=k)
        retrieval_time = time.perf_counter() - retrieval_start

        rerank_start = time.perf_counter()
        top_docs = self.rerank_contexts(question, docs, top_n=top_n)
        rerank_time = time.perf_counter() - rerank_start

        original_docs = [d.page_content for d in top_docs]
        sanitized_docs = original_docs
        sanitization_metadata = None
        sanitization_time = 0.0
        if self.context_sanitizer is not None:
            sanitization_start = time.perf_counter()
            sanitized_docs, sanitization_metadata = self.context_sanitizer.sanitize_retrieved_docs(
                original_docs,
                query=question,
            )
            sanitization_time = time.perf_counter() - sanitization_start

        context_original = "\n\n".join(original_docs)
        context = "\n\n".join(sanitized_docs)
        generation_start = time.perf_counter()
        if sanitization_metadata is not None and sanitization_metadata.get("mode") == "latent" and hasattr(self.llm, "generate_with_latent"):
            answer = self.llm.generate_with_latent(
                question=question,
                docs=original_docs,
                sanitization_metadata=sanitization_metadata,
                **kwargs,
            )
        elif sanitization_metadata is not None and "context_views" in sanitization_metadata and hasattr(self.llm, "generate_with_views"):
            answer = self.llm.generate_with_views(
                question=question,
                context_views=sanitization_metadata["context_views"],
                view_summaries=sanitization_metadata.get("view_summaries", {}),
                **kwargs,
            )
        else:
            prompt = f"[INST] Use the following context to answer the question.\n\nContext:\n{context}\n\nQuestion: {question} [/INST]"
            answer = self.llm.generate(prompt, **kwargs)
        generation_time = time.perf_counter() - generation_start

        result = {
            "question": question,
            "context": context,
            "context_original": context_original,
            "answer": answer,
            "retrieved_docs": original_docs,
            "retrieved_docs_original": original_docs,
            "retrieved_docs_defended": sanitized_docs,
            "retrieval_time_sec": retrieval_time,
            "rerank_time_sec": rerank_time,
            "sanitization_time_sec": sanitization_time,
            "generation_time_sec": generation_time,
        }
        if sanitization_metadata is not None:
            result["denpad_query_epsilon"] = sanitization_metadata.get("epsilon_query", sanitization_metadata.get("epsilon_doc"))
            result["denpad_num_entities"] = sanitization_metadata.get("num_entities", 0)
            result["denpad_num_perturbed"] = sanitization_metadata.get("num_perturbed", 0)
            result["denpad_disable_age_date"] = sanitization_metadata.get("disable_age_date", False)
            result["denpad_disable_duration_phrase"] = sanitization_metadata.get("disable_duration_phrase", False)
            result["denpad_attack_strong"] = sanitization_metadata.get("attack_strong", False)
            result["denpad_extracted_entities_by_label"] = sanitization_metadata.get("extracted_entities_by_label", {})
            result["denpad_filtered_out_entities_by_reason"] = sanitization_metadata.get("filtered_out_entities_by_reason", {})
            result["denpad_retained_entities_by_label"] = sanitization_metadata.get("retained_entities_by_label", {})
            result["denpad_selected_level_counts"] = sanitization_metadata.get("selected_level_counts", {})
            result["denpad_selected_source_counts"] = sanitization_metadata.get("selected_source_counts", {})
            result["denpad_same_pick_by_label"] = sanitization_metadata.get("same_pick_by_label", {})
            result["denpad_candidate_count_by_label"] = sanitization_metadata.get("candidate_count_by_label", {})
            result["denpad_selected_source_by_label"] = sanitization_metadata.get("selected_source_by_label", {})
            result["denpad_resource_summary"] = sanitization_metadata.get("resource_summary", {})
            result["denpad_resource_manifest"] = sanitization_metadata.get("resource_manifest", {})
            result["denpad_audit"] = sanitization_metadata.get("audit_records", [])
            if sanitization_metadata.get("mode") == "latent":
                result["denpad_latent_metadata"] = {
                    "retained_entities_by_label": sanitization_metadata.get("retained_entities_by_label", {}),
                    "num_entities": sanitization_metadata.get("num_entities", 0),
                    "num_perturbed": sanitization_metadata.get("num_perturbed", 0),
                }
                if getattr(self.llm, "latent_decoder", None) is not None:
                    result["denpad_latent_stats"] = getattr(self.llm, "last_latent_stats", {})
                    result["denpad_audit_runtime"] = result["denpad_latent_stats"].get("audit_records", [])
            else:
                result["denpad_context_views"] = {
                    name: docs
                    for name, docs in sanitization_metadata.get("context_views", {}).items()
                    if name == "PUBLIC"
                }
                result["denpad_view_summaries"] = sanitization_metadata.get("view_summaries", {})
            if getattr(self.llm, "fusion_decoder", None) is not None:
                result["denpad_fusion_stats"] = getattr(self.llm, "last_fusion_stats", {})
        return result
