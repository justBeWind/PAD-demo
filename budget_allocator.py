from __future__ import annotations

import logging
from collections import Counter
from difflib import SequenceMatcher
from typing import Optional

import numpy as np

from phi_taxonomy import PHI_TYPE_DEFINITIONS, phi_definition
from unit_constructor import ProtectionUnit, SensitiveSpan, tokenize

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


LOGGER = logging.getLogger(__name__)


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _safe_mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


class EmbeddingScorer:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self.model = None
        if SentenceTransformer is None:
            LOGGER.warning("sentence-transformers unavailable; falling back to lexical scoring.")
            return
        try:
            self.model = SentenceTransformer(model_name)
        except Exception as exc:
            LOGGER.warning("Failed to load embedding scorer model %s: %s", model_name, exc)
            self.model = None

    @property
    def enabled(self) -> bool:
        return self.model is not None

    def encode(self, texts: list[str]) -> np.ndarray:
        if not self.enabled:
            raise RuntimeError("Embedding scorer unavailable.")
        return np.asarray(self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False))


class AdaptiveBudgetAllocator:
    def __init__(
        self,
        epsilon: float = 0.2,
        min_sigma: float = 0.004,
        max_sigma: float = 0.04,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        self.epsilon = epsilon
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.delta = 1e-5
        self.min_clip = 0.08
        self.max_clip = 0.28
        self.min_blend = 0.10
        self.max_blend = 0.40
        self.perturb_threshold = 0.18
        self.allocation_smoothing = 0.05
        self.embedder = EmbeddingScorer(model_name=embedding_model_name)

    def allocate(self, query: str, docs: list[str], spans: list[SensitiveSpan], units: list[ProtectionUnit]) -> list[ProtectionUnit]:
        if not units:
            return units

        token_counts = Counter()
        for doc in docs:
            token_counts.update(tokenize(doc))
        total_tokens = max(sum(token_counts.values()), 1)

        query_embedding = None
        unit_embeddings = {}
        local_context_embeddings = {}
        leave_out_embeddings = {}
        if self.embedder.enabled:
            texts = [query] + [unit.local_text for unit in units]
            query_and_units = self.embedder.encode(texts)
            query_embedding = query_and_units[0]
            for unit, emb in zip(units, query_and_units[1:]):
                unit_embeddings[id(unit)] = emb

            local_context_texts = []
            leave_out_texts = []
            for unit in units:
                doc_text = docs[unit.doc_index]
                local_context = doc_text[max(0, unit.start_char - 96) : min(len(doc_text), unit.end_char + 96)]
                local_context_texts.append(local_context)
                leave_out_texts.append(doc_text[: unit.start_char] + " " + doc_text[unit.end_char :])
            encoded = self.embedder.encode(local_context_texts + leave_out_texts)
            for idx, unit in enumerate(units):
                local_context_embeddings[id(unit)] = encoded[idx]
                leave_out_embeddings[id(unit)] = encoded[len(units) + idx]

        for span in spans:
            span_tokens = tokenize(span.text)
            if not span_tokens:
                continue
            span.rarity_score = 1.0 - min(sum(token_counts.get(token, 0) for token in span_tokens) / (total_tokens * max(len(span_tokens), 1)), 1.0)

        feature_rows: list[dict[str, float]] = []
        for unit in units:
            definition = phi_definition(unit.phi_type)
            unit_tokens = tokenize(unit.local_text)
            rarity = 1.0 - min(sum(token_counts.get(token, 0) for token in unit_tokens) / (total_tokens * max(len(unit_tokens), 1)), 1.0)

            semantic_rel = self._semantic_relevance(query, unit.local_text, query_embedding, unit_embeddings.get(id(unit)))
            retrieval_contribution = self._retrieval_contribution(
                query,
                docs[unit.doc_index],
                unit,
                query_embedding,
                local_context_embeddings.get(id(unit)),
                leave_out_embeddings.get(id(unit)),
            )
            linkage_risk = self._linkage_risk(unit)
            copyability = self._copyability(query, unit.local_text, semantic_rel)
            risk = _clip(_safe_mean([definition.base_risk, rarity, linkage_risk, copyability]), 0.0, 1.0)
            utility = _clip(_safe_mean([semantic_rel, retrieval_contribution]), 0.0, 1.0)
            allocation_score = _clip(risk * (1.0 - utility), 0.0, 1.0)
            feature_rows.append(
                {
                    "risk": risk,
                    "utility": utility,
                    "allocation_score": allocation_score,
                    "copyability": copyability,
                    "rarity": rarity,
                    "linkage_risk": linkage_risk,
                    "semantic_rel": semantic_rel,
                    "retrieval_contribution": retrieval_contribution,
                }
            )

        allocation_denominator = sum(row["allocation_score"] + self.allocation_smoothing for row in feature_rows)
        allocation_denominator = max(allocation_denominator, 1e-8)

        for unit, row in zip(units, feature_rows):
            risk = row["risk"]
            utility = row["utility"]
            allocation_score = row["allocation_score"]
            copy_pressure = row["copyability"]

            epsilon_share = (allocation_score + self.allocation_smoothing) / allocation_denominator
            allocated_epsilon = self.epsilon * epsilon_share

            unit.semantic_relevance = row["semantic_rel"]
            unit.retrieval_contribution = row["retrieval_contribution"]
            unit.identifiability_score = row["linkage_risk"]
            unit.risk_score = risk
            unit.utility_score = utility
            unit.copy_risk = row["copyability"]
            unit.rarity_score = row["rarity"]
            unit.allocated_epsilon = allocated_epsilon

            unit.sigma = self._sigma(allocated_epsilon)
            unit.clip_norm = self._clip_norm(risk, utility)
            unit.blend = self._blend(utility, copy_pressure)
            unit.midlayer_strength = 0.0

            for span in unit.spans:
                span.risk_score = max(span.risk_score, risk)
                span.utility_score = utility
                span.copy_risk = row["copyability"]
                span.rarity_score = max(span.rarity_score, row["rarity"])
                span.sigma = unit.sigma
                span.clip_norm = unit.clip_norm
                span.allocated_epsilon = allocated_epsilon

        return units

    def should_perturb(self, unit: ProtectionUnit) -> bool:
        if phi_definition(unit.phi_type).group == "direct":
            return unit.risk_score >= 0.34
        return (unit.risk_score * (1.0 - unit.utility_score)) >= self.perturb_threshold

    def _semantic_relevance(
        self,
        query: str,
        unit_text: str,
        query_embedding: Optional[np.ndarray],
        unit_embedding: Optional[np.ndarray],
    ) -> float:
        if query_embedding is not None and unit_embedding is not None:
            return _clip(float(np.dot(query_embedding, unit_embedding)), 0.0, 1.0)
        query_tokens = set(tokenize(query))
        unit_tokens = set(tokenize(unit_text))
        lexical_overlap = len(query_tokens.intersection(unit_tokens)) / max(len(unit_tokens), 1)
        surface = SequenceMatcher(None, query.lower(), unit_text.lower()).ratio()
        return _clip(0.6 * lexical_overlap + 0.4 * surface, 0.0, 1.0)

    def _retrieval_contribution(
        self,
        query: str,
        doc_text: str,
        unit: ProtectionUnit,
        query_embedding: Optional[np.ndarray],
        local_context_embedding: Optional[np.ndarray],
        leave_out_embedding: Optional[np.ndarray],
    ) -> float:
        if query_embedding is not None and local_context_embedding is not None and leave_out_embedding is not None:
            full_sim = float(np.dot(query_embedding, local_context_embedding))
            leave_out_sim = float(np.dot(query_embedding, leave_out_embedding))
            return _clip(0.5 + 0.5 * (full_sim - leave_out_sim), 0.0, 1.0)

        local_context = doc_text[max(0, unit.start_char - 96) : min(len(doc_text), unit.end_char + 96)]
        removed = doc_text[: unit.start_char] + " " + doc_text[unit.end_char :]
        full_surface = SequenceMatcher(None, query.lower(), local_context.lower()).ratio()
        leave_surface = SequenceMatcher(None, query.lower(), removed.lower()).ratio()
        return _clip(0.5 + (full_surface - leave_surface), 0.0, 1.0)

    def _linkage_risk(self, unit: ProtectionUnit) -> float:
        anchor_density = float(unit.context_scores.get("anchor_density", 0.0))
        anchor_diversity = float(unit.context_scores.get("anchor_diversity", 0.0))
        return _clip(_safe_mean([anchor_density, anchor_diversity]), 0.0, 1.0)

    def _copyability(self, query: str, unit_text: str, semantic_rel: float) -> float:
        query_tokens = set(tokenize(query))
        unit_tokens = set(tokenize(unit_text))
        overlap = len(query_tokens.intersection(unit_tokens)) / max(len(unit_tokens), 1)
        surface = SequenceMatcher(None, query.lower(), unit_text.lower()).ratio()
        return _clip(_safe_mean([semantic_rel, overlap, surface]), 0.0, 1.0)

    def _sigma(self, allocated_epsilon: float) -> float:
        effective_epsilon = max(allocated_epsilon, 1e-3)
        gaussian_multiplier = np.sqrt(2.0 * np.log(1.25 / self.delta)) / effective_epsilon
        normalized = gaussian_multiplier / (gaussian_multiplier + 50.0)
        sigma = self.min_sigma + (self.max_sigma - self.min_sigma) * normalized
        return _clip(sigma, self.min_sigma, self.max_sigma)

    def _clip_norm(self, risk: float, utility: float) -> float:
        preservation = _safe_mean([risk, utility])
        clip_norm = self.min_clip + (self.max_clip - self.min_clip) * preservation
        return _clip(clip_norm, self.min_clip, self.max_clip)

    def _blend(self, utility: float, copy_pressure: float) -> float:
        privacy_dominance = _safe_mean([1.0 - utility, copy_pressure])
        blend = self.min_blend + (self.max_blend - self.min_blend) * privacy_dominance
        return _clip(blend, self.min_blend, self.max_blend)
