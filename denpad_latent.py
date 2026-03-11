import logging
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Optional

import numpy as np
import torch

try:
    import spacy
except ImportError:
    spacy = None


LOGGER = logging.getLogger(__name__)

GENERIC_LABELS = {"PERSON", "ORG", "GPE", "LOC", "DATE", "TIME", "MONEY", "PERCENT", "CARDINAL", "QUANTITY"}
LABEL_RISK_PRIOR = {
    "EMAIL": 0.99,
    "PHONE": 0.99,
    "ID": 0.98,
    "PERSON": 0.82,
    "ORG": 0.68,
    "GPE": 0.72,
    "LOC": 0.70,
    "DATE": 0.62,
    "TIME": 0.45,
    "AGE": 0.82,
    "MONEY": 0.60,
    "PERCENT": 0.35,
    "QUANTITY": 0.38,
    "NUMERIC": 0.52,
    "CARDINAL": 0.34,
    "MISC": 0.48,
}
STRUCTURED_LABELS = {"EMAIL", "PHONE", "ID"}
QUERY_LOW_UTILITY_LABELS = {"EMAIL", "PHONE", "ID", "DATE", "TIME", "AGE", "NUMERIC", "CARDINAL", "QUANTITY"}
IDENTITY_CONTEXT_HINTS = {
    "i am",
    "i'm",
    "my name",
    "from ",
    "residing",
    "live in",
    "living in",
    "my husband",
    "my wife",
    "my daughter",
    "my son",
    "my cousin",
    "my father",
    "my mother",
    "my doctor",
}
MEASUREMENT_CONTEXT_HINTS = {
    "mg",
    "ml",
    "dose",
    "dosage",
    "tablet",
    "tablets",
    "capsule",
    "capsules",
    "times a day",
    "per day",
    "twice a day",
    "blood sugar",
    "blood pressure",
    "cm",
    "mm",
    "kg",
}

EMAIL_PATTERN = re.compile(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b")
PHONE_PATTERN = re.compile(r"(?:\+?\d[\d\-\s()]{7,}\d)")
ID_PATTERN = re.compile(r"\b(?:[A-Z]{1,3}-?)?\d(?:[\dA-Z/-]{4,})\b")
DATE_PATTERN = re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b")
AGE_PATTERNS = [
    re.compile(r"\b(\d{1,3})\s*(?:years?\s*old|yrs?\s*old|y/o)\b", re.IGNORECASE),
    re.compile(r"\baged\s+(\d{1,3})\b", re.IGNORECASE),
    re.compile(r"\bage\s*(?:is|:)?\s*(\d{1,3})\b", re.IGNORECASE),
    re.compile(r"\b(\d{1,3})\s*-\s*year\s*-\s*old\b", re.IGNORECASE),
]
NUMBER_PATTERN = re.compile(r"\b\d+(?:\.\d+)?\b")
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
MONTH_WORD_PATTERN = re.compile(
    r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b",
    re.IGNORECASE,
)
ORG_SUFFIX_PATTERN = re.compile(
    r"\b(?:inc|corp|corporation|company|co|llc|ltd|limited|group|hospital|clinic|centre|center|"
    r"university|college|school|bank|ministry|department|association|foundation|institute)\b",
    re.IGNORECASE,
)
LOWERCASE_ENTITY_STOPWORDS = {
    "doesn",
    "don't",
    "dont",
    "didn",
    "didn't",
    "won",
    "won't",
    "cant",
    "can't",
    "im",
    "i'm",
    "ive",
    "i've",
    "ill",
    "i'll",
    "youre",
    "you're",
    "hes",
    "he's",
    "shes",
    "she's",
    "its",
    "it's",
    "std",
}


def _tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def _safe_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _should_perturb_span(span: "SensitiveSpan") -> bool:
    if span.label in STRUCTURED_LABELS:
        return span.risk_score >= 0.40
    if span.label in {"AGE", "DATE", "TIME"}:
        return span.risk_score >= 0.58 and span.utility_score <= 0.30
    if span.label in {"NUMERIC", "CARDINAL", "QUANTITY"}:
        has_multi_digit_value = len(re.sub(r"\D", "", span.text)) >= 2
        return has_multi_digit_value and span.risk_score >= 0.55 and span.utility_score <= 0.18
    return span.risk_score >= 0.62 and span.utility_score <= 0.28


@dataclass
class SensitiveSpan:
    text: str
    label: str
    start_char: int
    end_char: int
    doc_index: int
    evidence_source: str
    risk_score: float = 0.0
    utility_score: float = 0.0
    copy_risk: float = 0.0
    rarity_score: float = 0.0
    token_start: int = -1
    token_end: int = -1
    clip_norm: float = 0.0
    sigma: float = 0.0
    perturb_norm: float = 0.0


@dataclass
class ProtectionUnit:
    doc_index: int
    start_char: int
    end_char: int
    token_start: int = -1
    token_end: int = -1
    spans: list[SensitiveSpan] = field(default_factory=list)
    risk_score: float = 0.0
    utility_score: float = 0.0
    copy_risk: float = 0.0
    rarity_score: float = 0.0
    clip_norm: float = 0.0
    sigma: float = 0.0
    perturb_norm: float = 0.0


class ContextPrivacyExtractor:
    def __init__(self, spacy_model: str = "en_core_web_sm", disable_age_date: bool = False) -> None:
        self.disable_age_date = disable_age_date
        self.nlp = None
        if spacy is None:
            LOGGER.warning("spaCy is unavailable. DenPAD-Latent will rely on regex extraction only.")
            return
        try:
            self.nlp = spacy.load(spacy_model)
        except OSError:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                LOGGER.warning("spaCy model loading failed; falling back to regex extraction only.")
                self.nlp = None

    def extract(self, text: str, doc_index: int) -> list[SensitiveSpan]:
        spans: list[SensitiveSpan] = []
        spans.extend(self._extract_regex(text, doc_index))
        spans.extend(self._extract_spacy(text, doc_index))
        return self._merge_overlaps(spans)

    def _extract_regex(self, text: str, doc_index: int) -> list[SensitiveSpan]:
        spans: list[SensitiveSpan] = []
        for match in EMAIL_PATTERN.finditer(text):
            spans.append(SensitiveSpan(match.group(0), "EMAIL", match.start(), match.end(), doc_index, "regex"))
        for match in PHONE_PATTERN.finditer(text):
            digit_count = len(re.sub(r"\D", "", match.group(0)))
            if digit_count < 8:
                continue
            spans.append(SensitiveSpan(match.group(0), "PHONE", match.start(), match.end(), doc_index, "regex"))
        if not self.disable_age_date:
            for match in DATE_PATTERN.finditer(text):
                spans.append(SensitiveSpan(match.group(0), "DATE", match.start(), match.end(), doc_index, "regex"))
        for match in ID_PATTERN.finditer(text):
            candidate = match.group(0)
            if DATE_PATTERN.fullmatch(candidate):
                continue
            if candidate.count("/") == 0 and len(re.sub(r"\D", "", candidate)) < 5:
                continue
            spans.append(SensitiveSpan(candidate, "ID", match.start(), match.end(), doc_index, "regex"))
        if not self.disable_age_date:
            seen_ages = set()
            for pattern in AGE_PATTERNS:
                for match in pattern.finditer(text):
                    span_key = (match.start(1), match.end(1))
                    if span_key in seen_ages:
                        continue
                    seen_ages.add(span_key)
                    spans.append(SensitiveSpan(match.group(1), "AGE", match.start(1), match.end(1), doc_index, "regex"))
        for match in NUMBER_PATTERN.finditer(text):
            candidate = match.group(0)
            if len(candidate) == 1:
                continue
            if any(not (match.end() <= item.start_char or match.start() >= item.end_char) for item in spans):
                continue
            spans.append(SensitiveSpan(candidate, "NUMERIC", match.start(), match.end(), doc_index, "regex"))
        return spans

    def _extract_spacy(self, text: str, doc_index: int) -> list[SensitiveSpan]:
        if self.nlp is None:
            return []
        doc = self.nlp(text)
        spans: list[SensitiveSpan] = []
        for ent in doc.ents:
            if ent.label_ not in GENERIC_LABELS:
                continue
            label = ent.label_
            if self.disable_age_date and label in {"DATE", "TIME"}:
                continue
            if self._should_skip_spacy_entity(text, ent.text, label, ent.start_char, ent.end_char):
                continue
            spans.append(
                SensitiveSpan(
                    ent.text,
                    label,
                    ent.start_char,
                    ent.end_char,
                    doc_index,
                    "spacy",
                )
            )
        return spans

    def _should_skip_spacy_entity(self, text: str, entity_text: str, label: str, start_char: int, end_char: int) -> bool:
        normalized = entity_text.strip()
        lowered = normalized.lower()
        if not normalized:
            return True

        left_char = text[start_char - 1] if start_char > 0 else " "
        right_char = text[end_char] if end_char < len(text) else " "

        if label in {"PERSON", "ORG", "GPE", "LOC"}:
            if left_char.isalpha() or right_char.isalpha():
                return True
            if lowered in LOWERCASE_ENTITY_STOPWORDS:
                return True
            if normalized.islower():
                return True
            if label == "PERSON" and not any(ch.isupper() for ch in normalized):
                return True
            if label == "ORG":
                alpha_only = re.sub(r"[^A-Za-z]", "", normalized)
                if alpha_only and alpha_only.isupper() and len(alpha_only) <= 4 and not ORG_SUFFIX_PATTERN.search(normalized):
                    return True

        if label == "DATE":
            compact = normalized.replace(" ", "")
            if compact.isdigit():
                return True
            if not any(symbol in normalized for symbol in "/-") and MONTH_WORD_PATTERN.search(normalized) is None:
                return True

        if label == "TIME":
            if ":" not in normalized and lowered not in {"am", "pm"} and not lowered.endswith(("am", "pm")):
                return True

        if label in {"CARDINAL", "QUANTITY"}:
            contains_digit = any(ch.isdigit() for ch in normalized)
            if not contains_digit and MONTH_WORD_PATTERN.search(normalized) is None:
                return True
            if lowered in LOWERCASE_ENTITY_STOPWORDS:
                return True
            if len(normalized) <= 2 and not contains_digit:
                return True
            if len(re.sub(r"\D", "", normalized)) <= 1 and contains_digit:
                return True

        return False

    def _merge_overlaps(self, spans: list[SensitiveSpan]) -> list[SensitiveSpan]:
        priority = {
            "EMAIL": 0,
            "PHONE": 1,
            "ID": 2,
            "AGE": 3,
            "PERSON": 4,
            "ORG": 5,
            "GPE": 6,
            "LOC": 7,
            "DATE": 8,
            "TIME": 9,
            "MONEY": 10,
            "NUMERIC": 11,
            "QUANTITY": 12,
            "CARDINAL": 13,
            "PERCENT": 14,
        }
        merged: list[SensitiveSpan] = []
        occupied: dict[int, list[tuple[int, int]]] = defaultdict(list)
        ordered = sorted(
            spans,
            key=lambda item: (
                item.doc_index,
                item.start_char,
                priority.get(item.label, 999),
                -(item.end_char - item.start_char),
            ),
        )
        for span in ordered:
            overlaps = False
            for start_char, end_char in occupied[span.doc_index]:
                if not (span.end_char <= start_char or span.start_char >= end_char):
                    overlaps = True
                    break
            if overlaps:
                continue
            merged.append(span)
            occupied[span.doc_index].append((span.start_char, span.end_char))
        merged.sort(key=lambda item: (item.doc_index, item.start_char))
        return merged


class RiskUtilityScorer:
    def score(self, query: str, docs: list[str], spans: list[SensitiveSpan]) -> list[SensitiveSpan]:
        if not spans:
            return spans
        query_tokens = set(_tokenize(query))
        token_counts = Counter()
        for doc in docs:
            token_counts.update(_tokenize(doc))
        total_tokens = max(sum(token_counts.values()), 1)

        for span in spans:
            span_tokens = _tokenize(span.text)
            if not span_tokens:
                continue
            local_context = self._local_context(docs[span.doc_index], span.start_char, span.end_char)
            identity_hint = self._has_identity_hint(local_context)
            measurement_hint = self._has_measurement_hint(local_context)
            rarity = 1.0 - min(sum(token_counts.get(token, 0) for token in span_tokens) / (total_tokens * max(len(span_tokens), 1)), 1.0)
            query_overlap = len(query_tokens.intersection(span_tokens)) / max(len(span_tokens), 1)
            surface_similarity = SequenceMatcher(None, query.lower(), span.text.lower()).ratio()
            copy_risk = max(query_overlap, surface_similarity if len(span.text) > 4 else 0.0)
            copy_risk = _clip(max(copy_risk, self._copy_bonus(span.label)), 0.0, 1.0)
            type_prior = LABEL_RISK_PRIOR.get(span.label, LABEL_RISK_PRIOR["MISC"])
            structured_bonus = 0.15 if span.label in STRUCTURED_LABELS else 0.0
            risk = 0.55 * type_prior + 0.20 * rarity + 0.15 * copy_risk + structured_bonus + self._risk_bonus(span.label)
            risk += self._context_risk_bonus(span.label, identity_hint, measurement_hint)
            risk = _clip(risk, 0.0, 1.0)

            utility = 0.65 * query_overlap + 0.20 * (1.0 - min(surface_similarity, 1.0)) + 0.15 * self._label_utility_prior(span.label)
            if span.label in QUERY_LOW_UTILITY_LABELS:
                utility *= 0.55
            utility *= self._utility_multiplier(span.label)
            utility *= self._context_utility_multiplier(span.label, identity_hint, measurement_hint)
            utility = _clip(utility, 0.0, 1.0)

            span.rarity_score = rarity
            span.copy_risk = copy_risk
            span.risk_score = risk
            span.utility_score = utility
        return spans

    def _label_utility_prior(self, label: str) -> float:
        if label in {"PERSON", "ORG", "GPE", "LOC"}:
            return 0.25
        if label in {"MONEY", "PERCENT", "QUANTITY", "NUMERIC", "CARDINAL"}:
            return 0.15
        if label in STRUCTURED_LABELS:
            return 0.05
        return 0.10

    def _copy_bonus(self, label: str) -> float:
        if label in STRUCTURED_LABELS:
            return 0.68
        if label in {"AGE", "DATE", "TIME", "NUMERIC", "CARDINAL", "QUANTITY"}:
            return 0.48
        if label in {"PERSON", "ORG", "GPE", "LOC"}:
            return 0.38
        return 0.18

    def _risk_bonus(self, label: str) -> float:
        if label in STRUCTURED_LABELS:
            return 0.14
        if label in {"AGE", "DATE", "TIME", "NUMERIC", "CARDINAL", "QUANTITY"}:
            return 0.10
        if label in {"PERSON", "ORG", "GPE", "LOC"}:
            return 0.07
        return 0.03

    def _utility_multiplier(self, label: str) -> float:
        if label in STRUCTURED_LABELS:
            return 0.34
        if label in {"AGE", "DATE", "TIME", "NUMERIC", "CARDINAL", "QUANTITY"}:
            return 0.46
        if label in {"PERSON", "ORG", "GPE", "LOC"}:
            return 0.62
        return 0.84

    def _local_context(self, doc_text: str, start_char: int, end_char: int, window: int = 48) -> str:
        left = max(0, start_char - window)
        right = min(len(doc_text), end_char + window)
        return doc_text[left:right].lower()

    def _has_identity_hint(self, context: str) -> bool:
        return any(hint in context for hint in IDENTITY_CONTEXT_HINTS)

    def _has_measurement_hint(self, context: str) -> bool:
        return any(hint in context for hint in MEASUREMENT_CONTEXT_HINTS)

    def _context_risk_bonus(self, label: str, identity_hint: bool, measurement_hint: bool) -> float:
        bonus = 0.0
        if label in {"PERSON", "ORG", "GPE", "LOC"} and identity_hint:
            bonus += 0.08
        if label in {"AGE", "DATE", "TIME"} and identity_hint:
            bonus += 0.07
        if label in {"NUMERIC", "CARDINAL", "QUANTITY"} and identity_hint:
            bonus += 0.10
        if label in {"NUMERIC", "CARDINAL", "QUANTITY"} and measurement_hint:
            bonus -= 0.07
        return bonus

    def _context_utility_multiplier(self, label: str, identity_hint: bool, measurement_hint: bool) -> float:
        if label in {"NUMERIC", "CARDINAL", "QUANTITY"} and measurement_hint:
            return 1.30
        if label in {"PERSON", "ORG", "GPE", "LOC"} and identity_hint:
            return 0.82
        if label in {"AGE", "DATE", "TIME"} and identity_hint:
            return 0.78
        if label in {"NUMERIC", "CARDINAL", "QUANTITY"} and identity_hint:
            return 0.72
        return 1.0


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
        span_plans: list[dict[str, Any]],
        layer_indices: list[int],
        neighbor_window: int = 8,
    ) -> None:
        self.model = model
        self.prompt_length = prompt_length
        self.query_repr = query_repr
        self.span_plans = span_plans
        self.layer_indices = layer_indices
        self.neighbor_window = neighbor_window
        self.handles: list[Any] = []
        self.applied_layers: set[int] = set()

    def __enter__(self) -> "MidLayerSuppressionHook":
        layers = self._decoder_layers()
        if not layers:
            return self
        for layer_idx in self.layer_indices:
            if layer_idx < 0 or layer_idx >= len(layers):
                continue
            handle = layers[layer_idx].register_forward_hook(self._make_hook(layer_idx))
            self.handles.append(handle)
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
            if not torch.is_tensor(hidden_states) or hidden_states.dim() != 3:
                return output
            if hidden_states.size(1) < self.prompt_length:
                return output

            modified = hidden_states.clone()
            query = self.query_repr.to(modified.device, dtype=modified.dtype)
            query = query / query.norm(p=2).clamp_min(1e-8)

            for plan in self.span_plans:
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


class DenPADLatentSanitizer:
    def __init__(
        self,
        spacy_model: str = "en_core_web_sm",
        disable_age_date: bool = False,
    ) -> None:
        self.extractor = ContextPrivacyExtractor(spacy_model=spacy_model, disable_age_date=disable_age_date)
        self.scorer = RiskUtilityScorer()

    def sanitize_retrieved_docs(self, docs: list[str], query: Optional[str] = None) -> tuple[list[str], dict[str, Any]]:
        query = query or ""
        spans: list[SensitiveSpan] = []
        doc_offsets: list[dict[str, int]] = []
        running = 0
        for idx, doc in enumerate(docs):
            doc_offsets.append({"doc_index": idx, "start_char": running, "end_char": running + len(doc)})
            spans.extend(self.extractor.extract(doc, idx))
            running += len(doc)
            if idx != len(docs) - 1:
                running += 2

        spans = self.scorer.score(query, docs, spans)
        audit_records = []
        for span in spans:
            audit_records.append(
                {
                    "doc_index": span.doc_index,
                    "entity": span.text,
                    "label": span.label,
                    "risk_score": span.risk_score,
                    "utility_score": span.utility_score,
                    "copy_risk": span.copy_risk,
                    "rarity_score": span.rarity_score,
                    "evidence_source": span.evidence_source,
                }
            )
        metadata = {
            "mode": "latent",
            "context_docs": docs,
            "doc_offsets": doc_offsets,
            "spans": spans,
            "num_entities": len(spans),
            "num_perturbed": len([span for span in spans if _should_perturb_span(span)]),
            "audit_records": audit_records,
            "retained_entities_by_label": dict(Counter(span.label for span in spans)),
        }
        return docs, metadata


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

        bos_ids = []
        if self.tokenizer.bos_token_id is not None:
            bos_ids = [self.tokenizer.bos_token_id]
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

        token_spans = self._align_spans_to_tokens(sanitization_metadata.get("spans", []), context_offsets, docs)
        protection_units = self._build_protection_units(token_spans, context_offsets, docs)
        prompt_context_start = len(bos_ids) + len(prefix_ids)
        accountant = LatentDPAccountant(alpha=self.alpha, delta=self.delta)
        audit_records = []
        suppression_plans = []

        for unit in protection_units:
            if unit.token_start < 0 or unit.token_end <= unit.token_start:
                continue

            global_start = prompt_context_start + unit.token_start
            global_end = prompt_context_start + unit.token_end
            token_embeds = inputs_embeds[0, global_start:global_end]
            if token_embeds.numel() == 0:
                continue

            original_mean = token_embeds.mean(dim=0)
            parallel = torch.dot(original_mean, query_repr) * query_repr
            residual = original_mean - parallel
            clip_norm = self._clip_norm(original_mean, unit)
            residual_norm = residual.norm(p=2).item()
            clipped_residual = residual * min(1.0, clip_norm / max(residual_norm, 1e-8))
            sigma = self._sigma(unit)
            noise = torch.randn_like(clipped_residual) * (sigma * clip_norm)
            blend = self._perturb_blend(unit)
            perturbed_mean = original_mean + blend * (clipped_residual + noise - residual)
            delta = perturbed_mean - original_mean
            token_scale = 0.35 / math.sqrt(max(global_end - global_start, 1))
            applied_delta = delta * token_scale
            inputs_embeds[0, global_start:global_end] = token_embeds + applied_delta.unsqueeze(0)

            unit.clip_norm = clip_norm
            unit.sigma = sigma
            unit.perturb_norm = float(applied_delta.norm(p=2).item())
            accountant.add_gaussian_step(sensitivity=max(clip_norm * blend * token_scale, 1e-4), sigma=sigma)
            midlayer_strength = self._midlayer_strength(unit)
            suppression_plans.append(
                {
                    "global_start": global_start,
                    "global_end": global_end,
                    "strength": midlayer_strength,
                    "labels": sorted({span.label for span in unit.spans}),
                }
            )
            audit_records.append(
                {
                    "doc_index": unit.doc_index,
                    "entity": self._unit_surface_text(unit, docs),
                    "label": self._unit_primary_label(unit),
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
                    "sigma": sigma,
                    "clip_norm": clip_norm,
                    "midlayer_strength": midlayer_strength,
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
        layer_indices = self._suppression_layer_indices()
        hook_context = MidLayerSuppressionHook(
            model=self.model,
            prompt_length=input_ids.shape[1],
            query_repr=query_repr,
            span_plans=suppression_plans,
            layer_indices=layer_indices,
            neighbor_window=self.suppression_neighbor_window,
        )
        if self.enable_midlayer_suppression and suppression_plans and layer_indices:
            with hook_context:
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
            "midlayer_layers": layer_indices,
            "audit_records": audit_records,
        }
        self.last_stats = stats
        return response, stats

    def _align_spans_to_tokens(self, spans: list[SensitiveSpan], offset_mapping: list[tuple[int, int]], docs: list[str]) -> list[SensitiveSpan]:
        aligned: list[SensitiveSpan] = []
        doc_starts = self._doc_starts_from_docs(docs)
        for span in spans:
            base_offset = doc_starts.get(span.doc_index, 0)
            global_start = base_offset + span.start_char
            global_end = base_offset + span.end_char
            token_indices = []
            for idx, (start_char, end_char) in enumerate(offset_mapping):
                if end_char <= global_start or start_char >= global_end:
                    continue
                token_indices.append(idx)
            new_span = SensitiveSpan(**span.__dict__)
            if token_indices:
                new_span.token_start = token_indices[0]
                new_span.token_end = token_indices[-1] + 1
            aligned.append(new_span)
        return aligned

    def _build_protection_units(
        self,
        spans: list[SensitiveSpan],
        offset_mapping: list[tuple[int, int]],
        docs: list[str],
    ) -> list[ProtectionUnit]:
        units: list[ProtectionUnit] = []
        doc_starts = self._doc_starts_from_docs(docs)
        for span in spans:
            if not _should_perturb_span(span) or span.token_start < 0 or span.token_end <= span.token_start:
                continue
            start_char, end_char = self._expand_char_span(span, docs[span.doc_index])
            base_offset = doc_starts.get(span.doc_index, 0)
            global_start = base_offset + start_char
            global_end = base_offset + end_char
            token_indices = []
            for idx, (tok_start, tok_end) in enumerate(offset_mapping):
                if tok_end <= global_start or tok_start >= global_end:
                    continue
                token_indices.append(idx)
            if not token_indices:
                continue
            units.append(
                ProtectionUnit(
                    doc_index=span.doc_index,
                    start_char=start_char,
                    end_char=end_char,
                    token_start=token_indices[0],
                    token_end=token_indices[-1] + 1,
                    spans=[span],
                    risk_score=span.risk_score,
                    utility_score=span.utility_score,
                    copy_risk=span.copy_risk,
                    rarity_score=span.rarity_score,
                )
            )

        if not units:
            return []

        merged: list[ProtectionUnit] = []
        units.sort(key=lambda unit: (unit.doc_index, unit.start_char, unit.end_char))
        for unit in units:
            if not merged or merged[-1].doc_index != unit.doc_index or merged[-1].end_char < unit.start_char:
                merged.append(unit)
                continue
            current = merged[-1]
            current.start_char = min(current.start_char, unit.start_char)
            current.end_char = max(current.end_char, unit.end_char)
            current.token_start = min(current.token_start, unit.token_start)
            current.token_end = max(current.token_end, unit.token_end)
            current.spans.extend(unit.spans)
            current.risk_score = max(current.risk_score, unit.risk_score)
            current.utility_score = min(current.utility_score, unit.utility_score)
            current.copy_risk = max(current.copy_risk, unit.copy_risk)
            current.rarity_score = max(current.rarity_score, unit.rarity_score)
        return merged

    def _expand_char_span(self, span: SensitiveSpan, doc_text: str) -> tuple[int, int]:
        left_window, right_window = self._span_windows(span)
        sentence_breaks = ".!?\n;"

        start = max(0, span.start_char - left_window)
        end = min(len(doc_text), span.end_char + right_window)

        left_boundary = max(doc_text.rfind(ch, 0, span.start_char) for ch in sentence_breaks)
        if left_boundary != -1 and span.start_char - left_boundary <= left_window * 2:
            start = max(start, left_boundary + 1)

        right_candidates = [doc_text.find(ch, span.end_char) for ch in sentence_breaks if doc_text.find(ch, span.end_char) != -1]
        if right_candidates:
            right_boundary = min(right_candidates)
            if right_boundary - span.end_char <= right_window * 2:
                end = min(end, right_boundary + 1)

        while start > 0 and not doc_text[start - 1].isspace():
            start -= 1
        while end < len(doc_text) and not doc_text[end - 1].isspace():
            end += 1
            if end >= len(doc_text):
                end = len(doc_text)
                break

        return start, end

    def _span_windows(self, span: SensitiveSpan) -> tuple[int, int]:
        if span.label in STRUCTURED_LABELS:
            return 48, 56
        if span.label in {"PERSON", "ORG", "GPE", "LOC"}:
            return 52, 72
        if span.label in {"AGE", "DATE", "TIME"}:
            return 40, 56
        if span.label in {"NUMERIC", "CARDINAL", "QUANTITY"}:
            return 20, 30
        return 24, 32

    def _unit_primary_label(self, unit: ProtectionUnit) -> str:
        if not unit.spans:
            return "MISC"
        return max(unit.spans, key=lambda span: (span.risk_score, len(span.text))).label

    def _unit_surface_text(self, unit: ProtectionUnit, docs: list[str]) -> str:
        doc_text = docs[unit.doc_index]
        return _normalize_space(doc_text[unit.start_char:unit.end_char])

    def _protection_label(self, item: SensitiveSpan | ProtectionUnit) -> str:
        if isinstance(item, ProtectionUnit):
            return self._unit_primary_label(item)
        return item.label

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
        for fraction in self.suppression_layer_fractions:
            fraction = _clip(float(fraction), 0.0, 1.0)
            idx = min(total_layers - 1, max(0, int(round((total_layers - 1) * fraction))))
            indices.add(idx)
        return sorted(indices)

    def _clip_norm(self, embedding: torch.Tensor, span: SensitiveSpan | ProtectionUnit) -> float:
        label = self._protection_label(span)
        base = 0.10 + 0.20 * span.risk_score - 0.08 * span.utility_score + 0.08 * span.copy_risk
        if label in STRUCTURED_LABELS:
            base += 0.05
        elif label in {"PERSON", "ORG", "GPE", "LOC", "AGE", "DATE", "TIME"}:
            base += 0.03
        return _clip(base, 0.06, 0.36)

    def _sigma(self, span: SensitiveSpan | ProtectionUnit) -> float:
        label = self._protection_label(span)
        scale = min(1.0 / max(self.epsilon, 1e-3), 5.0)
        raw = 0.004 + scale * (0.002 + 0.012 * span.risk_score + 0.008 * span.copy_risk - 0.008 * span.utility_score)
        if label in STRUCTURED_LABELS:
            raw *= 1.18
        return _clip(raw, self.min_sigma, self.max_sigma)

    def _perturb_blend(self, span: SensitiveSpan | ProtectionUnit) -> float:
        label = self._protection_label(span)
        blend = 0.10 + 0.34 * span.risk_score - 0.22 * span.utility_score + 0.14 * span.copy_risk
        if label in STRUCTURED_LABELS:
            blend += 0.06
        elif label in {"PERSON", "ORG", "GPE", "LOC", "AGE", "DATE", "TIME"}:
            blend += 0.04
        return _clip(blend, 0.08, 0.44)

    def _midlayer_strength(self, span: SensitiveSpan | ProtectionUnit) -> float:
        label = self._protection_label(span)
        base = 0.12 + 0.30 * span.risk_score - 0.18 * span.utility_score + 0.18 * span.copy_risk
        if label in STRUCTURED_LABELS:
            base += 0.14
        elif label in {"AGE", "DATE", "TIME"}:
            base += 0.12
        elif label in {"NUMERIC", "CARDINAL", "QUANTITY"}:
            base += 0.04
        elif label in {"PERSON", "ORG", "GPE", "LOC"}:
            base += 0.10
        return _clip(base, 0.10, 0.62)
