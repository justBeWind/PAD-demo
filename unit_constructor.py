from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional

try:
    import spacy
except ImportError:
    spacy = None

from phi_taxonomy import (
    CLINICAL_TERMS,
    FIRST_PERSON_TERMS,
    MEASUREMENT_TERMS,
    PHITypeDefinition,
    RELATIONSHIP_TERMS,
    TEMPORAL_TERMS,
    normalize_phi_type,
    phi_definition,
    PHI_TYPE_DEFINITIONS,
)

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

LOGGER = logging.getLogger(__name__)

GENERIC_LABELS = {"PERSON", "ORG", "GPE", "LOC", "DATE", "TIME", "MONEY", "PERCENT", "CARDINAL", "QUANTITY"}
EMAIL_PATTERN = re.compile(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b")
PHONE_PATTERN = re.compile(r"(?:\+?\d[\d\-\s()]{7,}\d)")
ID_PATTERN = re.compile(r"\b(?:[A-Z]{1,3}-?)?\d(?:[\dA-Z/-]{4,})\b")
DATE_PATTERN = re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b")
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
def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


@dataclass
class SensitiveSpan:
    text: str
    label: str
    start_char: int
    end_char: int
    doc_index: int
    evidence_source: str
    phi_type: str = "IDENTIFYING_NARRATIVE"
    phi_group: str = "narrative"
    local_context: str = ""
    context_flags: dict[str, bool] = field(default_factory=dict)
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
    phi_type: str
    phi_group: str
    local_text: str
    context_flags: dict[str, bool]
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
    blend: float = 0.0
    midlayer_strength: float = 0.0
    semantic_relevance: float = 0.0
    retrieval_contribution: float = 0.0
    identifiability_score: float = 0.0


class ContextPrivacyExtractor:
    def __init__(self, spacy_model: str = "en_core_web_sm", disable_age_date: bool = False) -> None:
        self.disable_age_date = disable_age_date
        self.nlp = None
        if spacy is None:
            LOGGER.warning("spaCy unavailable; ContextPrivacyExtractor falls back to regex only.")
            return
        try:
            self.nlp = spacy.load(spacy_model)
        except OSError:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                LOGGER.warning("spaCy model loading failed; regex-only extractor enabled.")
                self.nlp = None

    def extract(self, text: str, doc_index: int) -> list[SensitiveSpan]:
        spans = []
        spans.extend(self._extract_regex(text, doc_index))
        spans.extend(self._extract_spacy(text, doc_index))
        merged = self._merge_overlaps(spans)
        for span in merged:
            local_context = self.local_context(text, span.start_char, span.end_char)
            span.local_context = local_context
            span.context_flags = context_flags(local_context)
            span.phi_type = normalize_phi_type(span.label, local_context, span.text)
            definition = phi_definition(span.phi_type)
            span.phi_group = definition.group
        return merged

    def local_context(self, text: str, start_char: int, end_char: int, window: int = 72) -> str:
        left = max(0, start_char - window)
        right = min(len(text), end_char + window)
        return normalize_space(text[left:right])

    def _extract_regex(self, text: str, doc_index: int) -> list[SensitiveSpan]:
        spans: list[SensitiveSpan] = []
        for match in EMAIL_PATTERN.finditer(text):
            spans.append(SensitiveSpan(match.group(0), "EMAIL", match.start(), match.end(), doc_index, "regex"))
        for match in PHONE_PATTERN.finditer(text):
            if len(re.sub(r"\D", "", match.group(0))) < 8:
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
            spans.append(SensitiveSpan(ent.text, label, ent.start_char, ent.end_char, doc_index, "spacy"))
        for chunk in getattr(doc, "noun_chunks", []):
            if self._should_skip_noun_chunk(chunk.text):
                continue
            spans.append(SensitiveSpan(chunk.text, "MISC", chunk.start_char, chunk.end_char, doc_index, "noun_chunk"))
        return spans

    def _should_skip_noun_chunk(self, chunk_text: str) -> bool:
        normalized = normalize_space(chunk_text)
        lowered = normalized.lower()
        if not normalized:
            return True
        tokens = tokenize(lowered)
        if len(tokens) == 0 or len(tokens) > 10:
            return True
        if lowered in LOWERCASE_ENTITY_STOPWORDS:
            return True
        has_signal = (
            any(term in lowered for term in RELATIONSHIP_TERMS)
            or any(term in lowered for term in TEMPORAL_TERMS)
            or any(term in lowered for term in CLINICAL_TERMS)
            or any(ch.isdigit() for ch in normalized)
        )
        return not has_signal

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
            if lowered in LOWERCASE_ENTITY_STOPWORDS or normalized.islower():
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
            "RELATION": 4,
            "PERSON": 5,
            "ORG": 6,
            "GPE": 7,
            "LOC": 8,
            "DATE": 9,
            "TIME": 10,
            "MONEY": 11,
            "NUMERIC": 12,
            "QUANTITY": 13,
            "CARDINAL": 14,
            "PERCENT": 15,
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


def context_flags(local_context: str) -> dict[str, bool]:
    lowered = local_context.lower()
    tokens = set(tokenize(lowered))
    return {
        "first_person": bool(tokens.intersection(FIRST_PERSON_TERMS)),
        "relationship": bool(tokens.intersection(RELATIONSHIP_TERMS)),
        "temporal": bool(tokens.intersection(TEMPORAL_TERMS)),
        "clinical": bool(tokens.intersection(CLINICAL_TERMS)),
        "measurement": any(term in lowered for term in MEASUREMENT_TERMS),
    }


class AdaptiveProtectionUnitConstructor:
    def __init__(
        self,
        extractor: ContextPrivacyExtractor,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        self.extractor = extractor
        self.proposal_scorer = LearnedPHIProposalScorer(model_name=embedding_model_name)

    def propose_spans(self, docs: list[str]) -> list[SensitiveSpan]:
        spans: list[SensitiveSpan] = []
        for doc_index, doc in enumerate(docs):
            spans.extend(self.extractor.extract(doc, doc_index))
        return spans

    def build_units(self, query: str, docs: list[str], spans: list[SensitiveSpan]) -> list[ProtectionUnit]:
        units: list[ProtectionUnit] = []
        for span in spans:
            doc_text = docs[span.doc_index]
            proposal_context = self.extractor.local_context(doc_text, span.start_char, span.end_char, window=96)
            proposal_flags = context_flags(proposal_context)
            decision = self.proposal_scorer.score(
                query=query,
                base_phi_type=span.phi_type,
                span_text=span.text,
                local_text=proposal_context,
                context_flags=proposal_flags,
                evidence_source=span.evidence_source,
            )
            if not decision["protect"]:
                continue

            phi_type = decision["phi_type"]
            definition = phi_definition(phi_type)
            start_char, end_char = self._expand_span(
                span,
                doc_text,
                definition,
                promote_narrative=bool(decision["promote"]),
            )
            local_text = normalize_space(doc_text[start_char:end_char])
            unit_flags = context_flags(local_text)
            units.append(
                ProtectionUnit(
                    doc_index=span.doc_index,
                    start_char=start_char,
                    end_char=end_char,
                    phi_type=phi_type,
                    phi_group=definition.group,
                    local_text=local_text,
                    context_flags=unit_flags,
                    spans=[span],
                )
            )

        units.sort(key=lambda item: (item.doc_index, item.start_char, item.end_char))
        merged: list[ProtectionUnit] = []
        for unit in units:
            if not merged or merged[-1].doc_index != unit.doc_index or merged[-1].end_char < unit.start_char:
                merged.append(unit)
                continue
            current = merged[-1]
            current.start_char = min(current.start_char, unit.start_char)
            current.end_char = max(current.end_char, unit.end_char)
            current.local_text = normalize_space(docs[current.doc_index][current.start_char:current.end_char])
            current.spans.extend(unit.spans)
            current.context_flags = {key: current.context_flags.get(key, False) or unit.context_flags.get(key, False) for key in set(current.context_flags) | set(unit.context_flags)}
            if current.phi_type != "IDENTIFYING_NARRATIVE" and unit.phi_type == "IDENTIFYING_NARRATIVE":
                current.phi_type = unit.phi_type
                current.phi_group = unit.phi_group
        return merged

    def _expand_span(
        self,
        span: SensitiveSpan,
        doc_text: str,
        definition: PHITypeDefinition,
        promote_narrative: bool = False,
    ) -> tuple[int, int]:
        left_window = definition.left_window
        right_window = definition.right_window
        if promote_narrative or definition.narrative or span.context_flags.get("clinical") or span.context_flags.get("temporal") or span.context_flags.get("relationship"):
            left_window = max(left_window, 56)
            right_window = max(right_window, 84)

        sentence_breaks = ".!?\n;"
        start = max(0, span.start_char - left_window)
        end = min(len(doc_text), span.end_char + right_window)

        left_boundary = max(doc_text.rfind(ch, 0, span.start_char) for ch in sentence_breaks)
        if left_boundary != -1:
            start = max(start, left_boundary + 1)
        right_candidates = [doc_text.find(ch, span.end_char) for ch in sentence_breaks if doc_text.find(ch, span.end_char) != -1]
        if right_candidates:
            end = min(end, min(right_candidates) + 1)

        while start > 0 and not doc_text[start - 1].isspace():
            start -= 1
        while end < len(doc_text) and end > 0 and not doc_text[end - 1].isspace():
            end += 1
            if end >= len(doc_text):
                end = len(doc_text)
                break
        return start, end

class LearnedPHIProposalScorer:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model = None
        self.prototype_names = list(PHI_TYPE_DEFINITIONS.keys())
        self.prototype_embeddings = None
        self.coarse_names = ["direct", "quasi", "narrative", "ignore"]
        self.coarse_embeddings = None
        if SentenceTransformer is None:
            LOGGER.warning("sentence-transformers unavailable; LearnedPHIProposalScorer falls back to base PHI labels.")
            return
        try:
            self.model = SentenceTransformer(model_name)
            prototype_texts = [PHI_TYPE_DEFINITIONS[name].prototype_text for name in self.prototype_names]
            self.prototype_embeddings = self.model.encode(prototype_texts, normalize_embeddings=True, show_progress_bar=False)
            coarse_texts = [
                "direct personal identifier such as an email address, phone number, account id, or exact contact detail",
                "quasi identifier such as age, location, organization, family relation, or time marker linked to a private person",
                "identifying medical or personal narrative fragment combining symptoms, history, relation, age, location, or event timeline",
                "non-sensitive or utility-dominant text that should not be protected as private information",
            ]
            self.coarse_embeddings = self.model.encode(coarse_texts, normalize_embeddings=True, show_progress_bar=False)
        except Exception as exc:
            LOGGER.warning("Failed to initialize LearnedPHIProposalScorer with %s: %s", model_name, exc)
            self.model = None
            self.prototype_embeddings = None
            self.coarse_embeddings = None

    def score(
        self,
        query: str,
        base_phi_type: str,
        span_text: str,
        local_text: str,
        context_flags: dict[str, bool],
        evidence_source: str,
    ) -> dict[str, Any]:
        base_definition = phi_definition(base_phi_type)
        if base_definition.group == "direct":
            return {
                "protect": True,
                "phi_type": base_phi_type,
                "protect_prob": 1.0,
                "promotion_prob": 0.0,
                "promote": False,
            }

        if self.model is None or self.prototype_embeddings is None or self.coarse_embeddings is None:
            return self._fallback(base_phi_type, local_text, context_flags, evidence_source)

        text = (
            f"query: {query}\n"
            f"candidate: {span_text}\n"
            f"context: {local_text}\n"
            f"source: {evidence_source}\n"
            f"base_type: {base_phi_type}"
        )
        embedding = self.model.encode([text], normalize_embeddings=True, show_progress_bar=False)[0]
        fine_scores = self.prototype_embeddings @ embedding
        coarse_scores = self.coarse_embeddings @ embedding
        best_idx = int(fine_scores.argmax())
        best_name = self.prototype_names[best_idx]
        best_coarse_idx = int(coarse_scores.argmax())
        best_coarse = self.coarse_names[best_coarse_idx]
        narrative_score = float(fine_scores[self.prototype_names.index("IDENTIFYING_NARRATIVE")])
        support = 0.06 * sum(int(v) for v in context_flags.values())
        if evidence_source == "spacy":
            support += 0.04
        if evidence_source == "noun_chunk":
            support += 0.02

        protect_prob = 0.5 + 0.5 * max(float(coarse_scores[self.coarse_names.index("direct")]), float(coarse_scores[self.coarse_names.index("quasi")]), float(coarse_scores[self.coarse_names.index("narrative")]))
        protect_prob = max(0.0, min(1.0, protect_prob + support))
        promote = best_coarse == "narrative" or (narrative_score >= 0.18 and sum(int(v) for v in context_flags.values()) >= 2)
        promotion_prob = max(0.0, min(1.0, 0.5 + 0.5 * narrative_score + support))

        phi_type = best_name
        if promote and base_definition.group != "direct":
            phi_type = "IDENTIFYING_NARRATIVE"
        elif best_coarse == "ignore" and protect_prob < 0.58:
            phi_type = base_phi_type

        protect = protect_prob >= 0.58 or phi_definition(phi_type).group == "narrative"
        return {
            "protect": protect,
            "phi_type": phi_type,
            "protect_prob": protect_prob,
            "promotion_prob": promotion_prob,
            "promote": promote,
        }

    def _fallback(
        self,
        base_phi_type: str,
        local_text: str,
        context_flags: dict[str, bool],
        evidence_source: str,
    ) -> dict[str, Any]:
        narrative_hits = sum(int(v) for v in context_flags.values())
        promote = phi_definition(base_phi_type).group != "direct" and narrative_hits >= 2
        phi_type = "IDENTIFYING_NARRATIVE" if promote else base_phi_type
        protect = evidence_source == "spacy" or phi_definition(phi_type).group != "quasi" or narrative_hits >= 1
        return {
            "protect": protect,
            "phi_type": phi_type,
            "protect_prob": 0.7 if protect else 0.3,
            "promotion_prob": 0.8 if promote else 0.2,
            "promote": promote,
        }
