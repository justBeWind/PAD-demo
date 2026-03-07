from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Iterable, Optional


@dataclass(frozen=True)
class CandidateBundle:
    record_candidates: list[str]
    global_candidates: list[str]
    neighbor_candidates: list[str]


@dataclass(frozen=True)
class CandidateTrace:
    category: str
    original: str
    merged_before_filter: list[str]
    filtered_candidates: list[str]
    candidate_layers: dict[str, str]
    candidate_sources: dict[str, str]
    layer_counts: dict[str, int]
    record_hit: bool


class TypedCandidateIndex:
    """
    Typed public resource base + typed nearest-neighbor expansion.

    This wrapper centralizes candidate construction so the DP mechanism only
    operates on finite, typed, auditable candidate spaces.
    """

    def __init__(
        self,
        resource_registry,
        semantic_reranker=None,
        density_scorer=None,
        top_k: int = 20,
        typer=None,
    ) -> None:
        self.resource_registry = resource_registry
        self.semantic_reranker = semantic_reranker
        self.density_scorer = density_scorer
        self.top_k = top_k
        self.typer = typer

    def build(self, entity) -> CandidateBundle:
        label = "GPE" if entity.label == "LOC" else entity.label
        record_candidates = self._record_level_candidates(label, entity.normalized_text)
        global_candidates = self._typed_global_pool(label, entity.normalized_text)
        neighbor_candidates = self._typed_neighbors(label, entity.normalized_text)
        return CandidateBundle(
            record_candidates=record_candidates,
            global_candidates=global_candidates,
            neighbor_candidates=neighbor_candidates,
        )

    def merge_and_filter(self, entity, return_trace: bool = False):
        bundle = self.build(entity)
        merged_before_filter = []
        candidate_layers = {}
        candidate_sources = {}
        category = "GPE" if entity.label == "LOC" else entity.label

        def _append_with_source(candidate: str, source: str) -> None:
            normalized = " ".join(str(candidate).split()).strip()
            if not normalized:
                return
            merged_before_filter.append(normalized)
            if normalized not in candidate_layers:
                if source == "record":
                    level = self.resource_registry.candidate_level(category, entity.normalized_text, normalized)
                    candidate_layers[normalized] = level
                elif source == "neighbor":
                    candidate_layers[normalized] = "neighbor"
                else:
                    candidate_layers[normalized] = "global"
                candidate_sources[normalized] = source

        _append_with_source(entity.normalized_text, "original")
        for candidate in bundle.record_candidates:
            _append_with_source(candidate, "record")
        if bundle.record_candidates and len(bundle.record_candidates) < 4:
            for candidate in bundle.neighbor_candidates:
                _append_with_source(candidate, "neighbor")
        elif not bundle.record_candidates:
            for candidate in bundle.global_candidates:
                _append_with_source(candidate, "global")
            for candidate in bundle.neighbor_candidates:
                _append_with_source(candidate, "neighbor")

        seen = set()
        filtered = []
        for candidate in merged_before_filter:
            candidate = " ".join(str(candidate).split()).strip()
            if not candidate or candidate in seen:
                continue
            seen.add(candidate)
            if self.typer is not None and entity.label in {"DISEASE", "DRUG", "PERSON", "ORG", "GPE", "LOC"}:
                if not self.typer.candidate_matches_label(candidate, entity.label):
                    continue
            filtered.append(candidate)
        filtered = filtered[: self.top_k]
        if not return_trace:
            return filtered
        layer_counts: dict[str, int] = {}
        for candidate in filtered:
            layer = candidate_layers.get(candidate, "unknown")
            layer_counts[layer] = layer_counts.get(layer, 0) + 1
        trace = CandidateTrace(
            category=category,
            original=entity.normalized_text,
            merged_before_filter=merged_before_filter,
            filtered_candidates=filtered,
            candidate_layers={candidate: candidate_layers.get(candidate, "unknown") for candidate in filtered},
            candidate_sources={candidate: candidate_sources.get(candidate, "unknown") for candidate in filtered},
            layer_counts=layer_counts,
            record_hit=bool(bundle.record_candidates),
        )
        return filtered, trace

    def _record_level_candidates(self, label: str, text: str) -> list[str]:
        record = self.resource_registry.find_record(label, text)
        if record is None:
            return []
        candidates = [record.term, *record.aliases, *record.related, *getattr(record, "generalized", ())]
        return self._rank(text, candidates, top_k=max(6, self.top_k // 2))

    def _typed_global_pool(self, label: str, text: str) -> list[str]:
        pool = list(self.resource_registry.get_candidates(label))
        if not pool:
            return []
        return self._rank(text, pool, top_k=max(self.top_k, 12))

    def _typed_neighbors(self, label: str, text: str) -> list[str]:
        if self.density_scorer is None:
            return []
        # Nearest-neighbor expansion stays inside the typed pool.
        typed_pool = list(self.resource_registry.get_candidates(label))
        if len(typed_pool) <= 1:
            return []
        scored = []
        for candidate in typed_pool:
            if candidate.lower() == text.lower():
                continue
            try:
                score = self._similarity(text, candidate)
            except Exception:
                score = SequenceMatcher(None, text.lower(), candidate.lower()).ratio()
            scored.append((candidate, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return [candidate for candidate, score in scored[: max(4, self.top_k // 3)] if score >= 0.62]

    def _rank(self, original: str, candidates: Iterable[str], top_k: int) -> list[str]:
        scored = []
        for candidate in candidates:
            candidate = " ".join(str(candidate).split()).strip()
            if not candidate:
                continue
            scored.append((candidate, self._similarity(original, candidate)))
        scored.sort(key=lambda item: item[1], reverse=True)
        return [candidate for candidate, _ in scored[:top_k]]

    def _similarity(self, left: str, right: str) -> float:
        if self.semantic_reranker is not None and getattr(self.semantic_reranker, "model", None) is not None:
            return float(self.semantic_reranker.similarity(left, right))
        return SequenceMatcher(None, left.lower(), right.lower()).ratio()
