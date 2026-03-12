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
        self.embedder = EmbeddingScorer(model_name=embedding_model_name)
        self.prototype_names = list(PHI_TYPE_DEFINITIONS.keys())
        self.prototype_embeddings = None
        if self.embedder.enabled:
            self.prototype_embeddings = self.embedder.encode(
                [PHI_TYPE_DEFINITIONS[name].prototype_text for name in self.prototype_names]
            )

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
            contextual_identifiability = self._context_identifiability(unit, definition, unit_embeddings.get(id(unit)))
            copyability = self._copyability(query, unit.local_text, semantic_rel)

            risk = (
                0.35 * definition.base_risk
                + 0.22 * rarity
                + 0.28 * contextual_identifiability
                + 0.15 * copyability
            )
            utility = (
                0.45 * semantic_rel
                + 0.35 * retrieval_contribution
                + 0.20 * definition.base_utility
            )
            risk = _clip(risk, 0.0, 1.0)
            utility = _clip(utility, 0.0, 1.0)

            unit.semantic_relevance = semantic_rel
            unit.retrieval_contribution = retrieval_contribution
            unit.identifiability_score = contextual_identifiability
            unit.risk_score = risk
            unit.utility_score = utility
            unit.copy_risk = copyability
            unit.rarity_score = rarity

            unit.sigma = self._sigma(unit, definition)
            unit.clip_norm = self._clip_norm(unit, definition)
            unit.blend = self._blend(unit, definition)
            unit.midlayer_strength = self._midlayer_strength(unit, definition)

            for span in unit.spans:
                span.risk_score = max(span.risk_score, risk)
                span.utility_score = utility
                span.copy_risk = copyability
                span.rarity_score = max(span.rarity_score, rarity)
                span.sigma = unit.sigma
                span.clip_norm = unit.clip_norm

        return units

    def should_perturb(self, unit: ProtectionUnit) -> bool:
        definition = phi_definition(unit.phi_type)
        if definition.group == "direct":
            return unit.risk_score >= 0.34
        if definition.group == "narrative":
            return unit.risk_score >= 0.48 and unit.utility_score <= 0.70
        return unit.risk_score >= 0.54 and unit.utility_score <= 0.56

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

    def _context_identifiability(self, unit: ProtectionUnit, definition, unit_embedding: Optional[np.ndarray]) -> float:
        prototype_alignment = definition.base_risk
        if unit_embedding is not None and self.prototype_embeddings is not None:
            target_idx = self.prototype_names.index(unit.phi_type)
            prototype_alignment = float(np.dot(unit_embedding, self.prototype_embeddings[target_idx]))
            prototype_alignment = _clip(0.5 + 0.5 * prototype_alignment, 0.0, 1.0)

        coherence = 0.0
        if len(unit.spans) >= 2:
            coherence += 0.08
        if unit.context_flags.get("first_person"):
            coherence += 0.06
        if unit.context_flags.get("relationship"):
            coherence += 0.06
        if unit.context_flags.get("temporal"):
            coherence += 0.05
        if unit.context_flags.get("clinical"):
            coherence += 0.05
        if definition.group == "direct":
            coherence += 0.10
        elif definition.group == "narrative":
            coherence += 0.08
        elif unit.context_flags.get("measurement"):
            coherence -= 0.08
        return _clip(0.72 * prototype_alignment + 0.28 * _clip(coherence + definition.base_risk, 0.0, 1.0), 0.0, 1.0)

    def _copyability(self, query: str, unit_text: str, semantic_rel: float) -> float:
        query_tokens = set(tokenize(query))
        unit_tokens = set(tokenize(unit_text))
        overlap = len(query_tokens.intersection(unit_tokens)) / max(len(unit_tokens), 1)
        surface = SequenceMatcher(None, query.lower(), unit_text.lower()).ratio()
        return _clip(_safe_mean([semantic_rel, overlap, surface]), 0.0, 1.0)

    def _sigma(self, unit: ProtectionUnit, definition) -> float:
        base_scale = min(1.0 / max(self.epsilon, 1e-3), 5.0)
        privacy_pressure = 0.65 * unit.risk_score + 0.35 * unit.copy_risk
        retention_pressure = unit.utility_score
        raw = 0.004 + base_scale * (0.002 + 0.014 * privacy_pressure - 0.006 * retention_pressure)
        if definition.group == "direct":
            raw *= 1.20
        elif definition.group == "narrative":
            raw *= 1.15
        elif definition.name == "MEASUREMENT":
            raw *= 0.92
        return _clip(raw, self.min_sigma, self.max_sigma)

    def _clip_norm(self, unit: ProtectionUnit, definition) -> float:
        base = 0.10 + 0.19 * unit.risk_score - 0.05 * unit.utility_score + 0.06 * unit.copy_risk
        if definition.group == "direct":
            base += 0.06
        elif definition.group == "narrative":
            base += 0.05
        return _clip(base, 0.06, 0.34)

    def _blend(self, unit: ProtectionUnit, definition) -> float:
        base = 0.10 + 0.30 * unit.risk_score - 0.16 * unit.utility_score + 0.12 * unit.copy_risk
        if definition.group == "direct":
            base += 0.08
        elif definition.group == "narrative":
            base += 0.06
        return _clip(base, 0.08, 0.44)

    def _midlayer_strength(self, unit: ProtectionUnit, definition) -> float:
        base = 0.14 + 0.25 * unit.risk_score - 0.10 * unit.utility_score + 0.14 * unit.copy_risk
        if definition.group == "direct":
            base += 0.12
        elif definition.group == "narrative":
            base += 0.10
        return _clip(base, 0.10, 0.62)
