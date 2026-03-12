from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Optional

import torch

from budget_allocator import AdaptiveBudgetAllocator
from unit_constructor import (
    AdaptiveProtectionUnitConstructor,
    ContextPrivacyExtractor,
    ProtectionUnit,
    SensitiveSpan,
    normalize_space,
)


LOGGER = logging.getLogger(__name__)


def _safe_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


class LatentDPAccountant:
    def __init__(self, alpha: float = 10.0, delta: float = 1e-5) -> None:
        self.alpha = alpha
        self.delta = delta
        self.steps: list[tuple[float, float]] = []

    def add_gaussian_step(self, sensitivity: float, sigma: float) -> None:
        if sigma <= 0:
            return
        self.steps.append((sensitivity, sigma))

    def total_epsilon(self) -> float:
        if not self.steps:
            return 0.0
        total_rdp = 0.0
        for sensitivity, sigma in self.steps:
            total_rdp += (self.alpha * (sensitivity ** 2)) / max(2.0 * (sigma ** 2), 1e-12)
        return total_rdp + math.log(1.0 / self.delta) / max(self.alpha - 1.0, 1e-9)


class MidLayerSuppressionHook:
    def __init__(
        self,
        model,
        prompt_length: int,
        query_repr: torch.Tensor,
        unit_plans: list[dict[str, Any]],
        layer_indices: list[int],
        neighbor_window: int = 8,
    ) -> None:
        self.model = model
        self.prompt_length = prompt_length
        self.query_repr = query_repr
        self.unit_plans = unit_plans
        self.layer_indices = layer_indices
        self.neighbor_window = neighbor_window
        self.handles: list[Any] = []
        self.applied_layers: set[int] = set()

    def __enter__(self) -> "MidLayerSuppressionHook":
        layers = self._decoder_layers()
        if not layers:
            return self
        for layer_idx in self.layer_indices:
            if 0 <= layer_idx < len(layers):
                self.handles.append(layers[layer_idx].register_forward_hook(self._make_hook(layer_idx)))
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def _decoder_layers(self):
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return self.model.model.layers
        if hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            return self.model.transformer.h
        base_model = getattr(self.model, "base_model", None)
        if base_model is not None:
            if hasattr(base_model, "model") and hasattr(base_model.model, "layers"):
                return base_model.model.layers
            if hasattr(base_model, "layers"):
                return base_model.layers
        return []

    def _make_hook(self, layer_idx: int):
        def hook(module, inputs, output):
            if layer_idx in self.applied_layers:
                return output
            hidden_states = output[0] if isinstance(output, tuple) else output
            if not torch.is_tensor(hidden_states) or hidden_states.dim() != 3 or hidden_states.size(1) < self.prompt_length:
                return output

            modified = hidden_states.clone()
            query = self.query_repr.to(modified.device, dtype=modified.dtype)
            query = query / query.norm(p=2).clamp_min(1e-8)

            for plan in self.unit_plans:
                start = int(plan["global_start"])
                end = int(plan["global_end"])
                if end <= start or start < 0 or end > modified.size(1):
                    continue
                token_slice = modified[:, start:end, :]
                if token_slice.numel() == 0:
                    continue

                anchor = self._local_anchor(modified, start, end)
                anchor_parallel = (anchor @ query) * query
                anchor_residual = anchor - anchor_parallel
                parallel = (token_slice @ query).unsqueeze(-1) * query.unsqueeze(0).unsqueeze(0)
                residual = token_slice - parallel
                strength = float(plan["strength"])
                suppressed = parallel + (1.0 - strength) * residual + strength * anchor_residual.unsqueeze(0).unsqueeze(0)
                modified[:, start:end, :] = suppressed

            self.applied_layers.add(layer_idx)
            if isinstance(output, tuple):
                return (modified,) + output[1:]
            return modified

        return hook

    def _local_anchor(self, hidden_states: torch.Tensor, start: int, end: int) -> torch.Tensor:
        left_start = max(0, start - self.neighbor_window)
        left_end = max(0, start)
        right_start = min(hidden_states.size(1), end)
        right_end = min(hidden_states.size(1), end + self.neighbor_window)

        neighbors = []
        if left_end > left_start:
            neighbors.append(hidden_states[:, left_start:left_end, :])
        if right_end > right_start:
            neighbors.append(hidden_states[:, right_start:right_end, :])
        if neighbors:
            return torch.cat(neighbors, dim=1).mean(dim=1).squeeze(0)
        return hidden_states[:, start:end, :].mean(dim=1).squeeze(0)


class EarlyLayerDPPerturbationHook:
    def __init__(
        self,
        model,
        prompt_length: int,
        query_repr: torch.Tensor,
        unit_plans: list[dict[str, Any]],
        layer_indices: list[int],
        sigma_scale: float = 0.6,
        blend_scale: float = 0.85,
    ) -> None:
        self.model = model
        self.prompt_length = prompt_length
        self.query_repr = query_repr
        self.unit_plans = unit_plans
        self.layer_indices = layer_indices
        self.sigma_scale = sigma_scale
        self.blend_scale = blend_scale
        self.handles: list[Any] = []
        self.applied_layers: set[int] = set()

    def __enter__(self) -> "EarlyLayerDPPerturbationHook":
        layers = self._decoder_layers()
        if not layers:
            return self
        for layer_idx in self.layer_indices:
            if 0 <= layer_idx < len(layers):
                self.handles.append(layers[layer_idx].register_forward_hook(self._make_hook(layer_idx)))
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def _decoder_layers(self):
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return self.model.model.layers
        if hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            return self.model.transformer.h
        base_model = getattr(self.model, "base_model", None)
        if base_model is not None:
            if hasattr(base_model, "model") and hasattr(base_model.model, "layers"):
                return base_model.model.layers
            if hasattr(base_model, "layers"):
                return base_model.layers
        return []

    def _make_hook(self, layer_idx: int):
        def hook(module, inputs, output):
            if layer_idx in self.applied_layers:
                return output
            hidden_states = output[0] if isinstance(output, tuple) else output
            if not torch.is_tensor(hidden_states) or hidden_states.dim() != 3 or hidden_states.size(1) < self.prompt_length:
                return output

            modified = hidden_states.clone()
            query = self.query_repr.to(modified.device, dtype=modified.dtype)
            query = query / query.norm(p=2).clamp_min(1e-8)

            for plan in self.unit_plans:
                start = int(plan["global_start"])
                end = int(plan["global_end"])
                if end <= start or start < 0 or end > modified.size(1):
                    continue
                token_slice = modified[:, start:end, :]
                if token_slice.numel() == 0:
                    continue

                sigma = float(plan["sigma"]) * self.sigma_scale
                clip_norm = float(plan["clip_norm"])
                blend = min(1.0, float(plan["blend"]) * self.blend_scale)
                flat_query = query.unsqueeze(0)
                parallel = (token_slice @ query).unsqueeze(-1) * flat_query.unsqueeze(0)
                residual = token_slice - parallel
                residual_norm = residual.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-8)
                clip_factor = torch.clamp(clip_norm / residual_norm, max=1.0)
                clipped_residual = residual * clip_factor
                noise = torch.randn_like(clipped_residual) * (sigma * clip_norm)
                perturbed = parallel + clipped_residual + noise
                modified[:, start:end, :] = (1.0 - blend) * token_slice + blend * perturbed

            self.applied_layers.add(layer_idx)
            if isinstance(output, tuple):
                return (modified,) + output[1:]
            return modified

        return hook


class DenPADLatentSanitizer:
    def __init__(
        self,
        spacy_model: str = "en_core_web_sm",
        disable_age_date: bool = False,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        epsilon: float = 0.2,
    ) -> None:
        self.extractor = ContextPrivacyExtractor(spacy_model=spacy_model, disable_age_date=disable_age_date)
        self.unit_constructor = AdaptiveProtectionUnitConstructor(self.extractor)
        self.budget_allocator = AdaptiveBudgetAllocator(
            epsilon=epsilon,
            embedding_model_name=embedding_model_name,
        )
        self.disable_age_date = disable_age_date

    def sanitize_retrieved_docs(self, docs: list[str], query: Optional[str] = None) -> tuple[list[str], dict[str, Any]]:
        query = query or ""
        spans = self.unit_constructor.propose_spans(docs)
        units = self.unit_constructor.build_units(query=query, docs=docs, spans=spans)
        units = self.budget_allocator.allocate(query=query, docs=docs, spans=spans, units=units)
        retained_units = [unit for unit in units if self.budget_allocator.should_perturb(unit)]

        audit_records = []
        for unit in retained_units:
            audit_records.append(
                {
                    "doc_index": unit.doc_index,
                    "entity": unit.local_text,
                    "label": unit.phi_type,
                    "source_labels": [span.label for span in unit.spans],
                    "risk_score": unit.risk_score,
                    "utility_score": unit.utility_score,
                    "copy_risk": unit.copy_risk,
                    "rarity_score": unit.rarity_score,
                    "semantic_relevance": unit.semantic_relevance,
                    "retrieval_contribution": unit.retrieval_contribution,
                    "identifiability_score": unit.identifiability_score,
                    "sigma": unit.sigma,
                    "clip_norm": unit.clip_norm,
                    "blend": unit.blend,
                    "midlayer_strength": unit.midlayer_strength,
                }
            )

        metadata = {
            "mode": "latent",
            "context_docs": docs,
            "spans": spans,
            "protection_units": retained_units,
            "num_entities": len(spans),
            "num_perturbed": len(retained_units),
            "audit_records": audit_records,
            "disable_age_date": self.disable_age_date,
            "retained_entities_by_label": self._count_labels(retained_units),
            "extracted_entities_by_label": self._count_span_labels(spans),
        }
        return docs, metadata

    def _count_span_labels(self, spans: list[SensitiveSpan]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for span in spans:
            counts[span.label] = counts.get(span.label, 0) + 1
        return counts

    def _count_labels(self, units: list[ProtectionUnit]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for unit in units:
            counts[unit.phi_type] = counts.get(unit.phi_type, 0) + 1
        return counts


class LatentContextDecoder:
    def __init__(
        self,
        model,
        tokenizer,
        epsilon: float = 0.2,
        alpha: float = 10.0,
        delta: float = 1e-5,
        min_sigma: float = 0.004,
        max_sigma: float = 0.04,
        max_input_length: int = 2048,
        enable_midlayer_suppression: bool = True,
        enable_early_layer_dp: bool = True,
        early_layer_dp_fractions: tuple[float, ...] = (0.0, 0.12),
        early_layer_sigma_scale: float = 0.6,
        early_layer_blend_scale: float = 0.85,
        suppression_layer_fractions: tuple[float, ...] = (0.25, 0.5, 0.75),
        suppression_neighbor_window: int = 8,
        verbose: bool = False,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.epsilon = epsilon
        self.alpha = alpha
        self.delta = delta
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.max_input_length = max_input_length
        self.enable_midlayer_suppression = enable_midlayer_suppression
        self.enable_early_layer_dp = enable_early_layer_dp
        self.early_layer_dp_fractions = early_layer_dp_fractions
        self.early_layer_sigma_scale = early_layer_sigma_scale
        self.early_layer_blend_scale = early_layer_blend_scale
        self.suppression_layer_fractions = suppression_layer_fractions
        self.suppression_neighbor_window = suppression_neighbor_window
        self.verbose = verbose
        self.last_stats: dict[str, Any] = {}

    def generate(
        self,
        question: str,
        docs: list[str],
        sanitization_metadata: dict[str, Any],
        max_new_tokens: int = 256,
        temperature: float = 0.2,
        top_p: float = 0.9,
        do_sample: bool = True,
        repetition_penalty: float = 1.0,
    ) -> tuple[str, dict[str, Any]]:
        context = "\n\n".join(docs)
        prefix = "[INST] Use the following context to answer the question.\n\nContext:\n"
        suffix = f"\n\nQuestion: {question} [/INST]"

        bos_ids = [self.tokenizer.bos_token_id] if self.tokenizer.bos_token_id is not None else []
        prefix_ids = self.tokenizer(prefix, add_special_tokens=False)["input_ids"]
        context_tokenized = self.tokenizer(context, add_special_tokens=False, return_offsets_mapping=True)
        context_ids = context_tokenized["input_ids"]
        context_offsets = context_tokenized["offset_mapping"]
        suffix_ids = self.tokenizer(suffix, add_special_tokens=False)["input_ids"]

        input_ids_list = bos_ids + prefix_ids + context_ids + suffix_ids
        safe_len = self.max_input_length - max_new_tokens - 16
        if len(input_ids_list) > safe_len:
            trim = len(input_ids_list) - safe_len
            if trim >= len(prefix_ids) + len(context_ids):
                raise ValueError("Prompt trimming would remove the full context region; reduce max_new_tokens.")
            if trim > 0:
                context_ids = context_ids[trim:]
                context_offsets = context_offsets[trim:]
                prefix_ids = []
                input_ids_list = bos_ids + prefix_ids + context_ids + suffix_ids

        device = self.model.device
        input_ids = torch.tensor([input_ids_list], device=device)
        attention_mask = torch.ones_like(input_ids)
        embedding_layer = self.model.get_input_embeddings()
        inputs_embeds = embedding_layer(input_ids).detach().clone()

        query_ids = self.tokenizer(question, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
        query_embeds = embedding_layer(query_ids).squeeze(0)
        query_repr = query_embeds.mean(dim=0)
        query_repr = query_repr / query_repr.norm(p=2).clamp_min(1e-8)

        protection_units: list[ProtectionUnit] = sanitization_metadata.get("protection_units", [])
        token_units = self._align_units_to_tokens(protection_units, context_offsets, docs)
        prompt_context_start = len(bos_ids) + len(prefix_ids)
        accountant = LatentDPAccountant(alpha=self.alpha, delta=self.delta)
        audit_records = []
        suppression_plans = []

        for unit in token_units:
            if unit.token_start < 0 or unit.token_end <= unit.token_start:
                continue

            global_start = prompt_context_start + unit.token_start
            global_end = prompt_context_start + unit.token_end
            token_embeds = inputs_embeds[0, global_start:global_end]
            if token_embeds.numel() == 0:
                continue

            clip_norm = unit.clip_norm
            sigma = _clip(unit.sigma, self.min_sigma, self.max_sigma)
            blend = unit.blend
            query = query_repr.unsqueeze(0)
            parallel = (token_embeds @ query_repr).unsqueeze(-1) * query
            residual = token_embeds - parallel
            residual_norm = residual.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-8)
            clip_factor = torch.clamp(clip_norm / residual_norm, max=1.0)
            clipped_residual = residual * clip_factor
            noise = torch.randn_like(clipped_residual) * (sigma * clip_norm)
            perturbed_tokens = parallel + clipped_residual + noise
            mixed_tokens = (1.0 - blend) * token_embeds + blend * perturbed_tokens
            inputs_embeds[0, global_start:global_end] = mixed_tokens

            applied_delta = mixed_tokens - token_embeds
            unit.perturb_norm = float(applied_delta.norm(p=2).item())
            token_count = max(global_end - global_start, 1)
            accountant.add_gaussian_step(
                sensitivity=max(math.sqrt(token_count) * clip_norm * max(blend, 1e-4), 1e-4),
                sigma=sigma,
            )
            suppression_plans.append(
                {
                    "global_start": global_start,
                    "global_end": global_end,
                    "sigma": sigma,
                    "clip_norm": clip_norm,
                    "blend": blend,
                    "strength": unit.midlayer_strength,
                    "labels": sorted({span.label for span in unit.spans}),
                }
            )
            audit_records.append(
                {
                    "doc_index": unit.doc_index,
                    "entity": unit.local_text,
                    "label": unit.phi_type,
                    "source_entities": [span.text for span in unit.spans],
                    "source_labels": [span.label for span in unit.spans],
                    "char_start": unit.start_char,
                    "char_end": unit.end_char,
                    "token_start": unit.token_start,
                    "token_end": unit.token_end,
                    "risk_score": unit.risk_score,
                    "utility_score": unit.utility_score,
                    "copy_risk": unit.copy_risk,
                    "rarity_score": unit.rarity_score,
                    "semantic_relevance": unit.semantic_relevance,
                    "retrieval_contribution": unit.retrieval_contribution,
                    "identifiability_score": unit.identifiability_score,
                    "sigma": sigma,
                    "clip_norm": clip_norm,
                    "blend": blend,
                    "midlayer_strength": unit.midlayer_strength,
                    "perturb_norm": unit.perturb_norm,
                }
            )

        generation_kwargs = {
            "input_ids": input_ids,
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": do_sample,
            "repetition_penalty": repetition_penalty,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        early_layer_indices = self._early_dp_layer_indices()
        layer_indices = self._suppression_layer_indices()
        for plan in suppression_plans:
            if early_layer_indices:
                accountant.add_gaussian_step(
                    sensitivity=max(
                        math.sqrt(max(plan["global_end"] - plan["global_start"], 1))
                        * plan["clip_norm"]
                        * max(plan["blend"] * self.early_layer_blend_scale, 1e-4),
                        1e-4,
                    ),
                    sigma=_clip(plan["sigma"] * self.early_layer_sigma_scale, self.min_sigma * 0.5, self.max_sigma),
                )
        if (self.enable_midlayer_suppression and suppression_plans and layer_indices) or (self.enable_early_layer_dp and suppression_plans and early_layer_indices):
            early_ctx = EarlyLayerDPPerturbationHook(
                model=self.model,
                prompt_length=input_ids.shape[1],
                query_repr=query_repr,
                unit_plans=suppression_plans,
                layer_indices=early_layer_indices,
                sigma_scale=self.early_layer_sigma_scale,
                blend_scale=self.early_layer_blend_scale,
            )
            suppress_ctx = MidLayerSuppressionHook(
                model=self.model,
                prompt_length=input_ids.shape[1],
                query_repr=query_repr,
                unit_plans=suppression_plans,
                layer_indices=layer_indices,
                neighbor_window=self.suppression_neighbor_window,
            )
            with early_ctx:
                with suppress_ctx:
                    output_ids = self.model.generate(**generation_kwargs)
        else:
            output_ids = self.model.generate(**generation_kwargs)
        response = self.tokenizer.decode(output_ids[0][input_ids.shape[1] :], skip_special_tokens=True).strip()

        stats = {
            "epsilon_global": accountant.total_epsilon(),
            "num_perturbed_spans": len(audit_records),
            "avg_sigma": _safe_mean([record["sigma"] for record in audit_records]),
            "avg_clip_norm": _safe_mean([record["clip_norm"] for record in audit_records]),
            "avg_midlayer_strength": _safe_mean([record["midlayer_strength"] for record in audit_records]),
            "avg_perturb_norm": _safe_mean([record["perturb_norm"] for record in audit_records]),
            "early_dp_layers": early_layer_indices,
            "midlayer_layers": layer_indices,
            "audit_records": audit_records,
        }
        self.last_stats = stats
        return response, stats

    def _align_units_to_tokens(self, units: list[ProtectionUnit], offset_mapping: list[tuple[int, int]], docs: list[str]) -> list[ProtectionUnit]:
        aligned: list[ProtectionUnit] = []
        doc_starts = self._doc_starts_from_docs(docs)
        for unit in units:
            base_offset = doc_starts.get(unit.doc_index, 0)
            global_start = base_offset + unit.start_char
            global_end = base_offset + unit.end_char
            token_indices = []
            for idx, (start_char, end_char) in enumerate(offset_mapping):
                if end_char <= global_start or start_char >= global_end:
                    continue
                token_indices.append(idx)
            copied = ProtectionUnit(**{**unit.__dict__})
            if token_indices:
                copied.token_start = token_indices[0]
                copied.token_end = token_indices[-1] + 1
            aligned.append(copied)
        return aligned

    def _doc_starts_from_docs(self, docs: list[str]) -> dict[int, int]:
        offsets = {}
        running = 0
        for doc_index, doc in enumerate(docs):
            offsets[doc_index] = running
            running += len(doc) + 2
        return offsets

    def _suppression_layer_indices(self) -> list[int]:
        if not self.enable_midlayer_suppression:
            return []
        return self._fractional_layer_indices(self.suppression_layer_fractions)

    def _early_dp_layer_indices(self) -> list[int]:
        if not self.enable_early_layer_dp:
            return []
        return self._fractional_layer_indices(self.early_layer_dp_fractions)

    def _fractional_layer_indices(self, fractions: tuple[float, ...]) -> list[int]:
        layers = []
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            layers = self.model.model.layers
        elif hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            layers = self.model.transformer.h
        elif hasattr(self.model, "base_model"):
            base_model = self.model.base_model
            if hasattr(base_model, "model") and hasattr(base_model.model, "layers"):
                layers = base_model.model.layers
        total_layers = len(layers)
        if total_layers <= 0:
            return []
        indices = set()
        for fraction in fractions:
            fraction = _clip(float(fraction), 0.0, 1.0)
            idx = min(total_layers - 1, max(0, int(round((total_layers - 1) * fraction))))
            indices.add(idx)
        return sorted(indices)
