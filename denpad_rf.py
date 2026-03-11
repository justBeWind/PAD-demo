import logging
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F

try:
    import spacy
except ImportError:
    spacy = None


LOGGER = logging.getLogger(__name__)


MASK_PLACEHOLDER = "_"
GENERIC_LABELS = {"PERSON", "ORG", "GPE", "LOC", "DATE", "TIME", "MONEY", "PERCENT", "CARDINAL", "QUANTITY"}
GROUP_PUBLIC = "G_public_safe"
GROUP_HIDE = "G_hide_strict"
GROUP_PRESERVE = "G_preserve_soft"
GROUP_STRUCT = "G_structured"
GROUP_NUMERIC = "G_numeric"
PUBLIC_VIEW = "PUBLIC"
VIEW_TO_GROUP = {
    "VIEW_hide": GROUP_HIDE,
    "VIEW_preserve": GROUP_PRESERVE,
    "VIEW_struct": GROUP_STRUCT,
    "VIEW_num": GROUP_NUMERIC,
}
GROUP_ORDER = [GROUP_HIDE, GROUP_PRESERVE, GROUP_STRUCT, GROUP_NUMERIC]
DEFAULT_GROUP_BETAS = {
    GROUP_HIDE: 0.03,
    GROUP_PRESERVE: 0.12,
    GROUP_STRUCT: 0.02,
    GROUP_NUMERIC: 0.06,
}
DEFAULT_GROUP_WEIGHTS = {
    GROUP_HIDE: 0.9,
    GROUP_PRESERVE: 1.3,
    GROUP_STRUCT: 0.75,
    GROUP_NUMERIC: 0.95,
}
LABEL_RISK_PRIOR = {
    "EMAIL": 0.98,
    "PHONE": 0.98,
    "ID": 0.97,
    "PERSON": 0.83,
    "ORG": 0.72,
    "GPE": 0.74,
    "LOC": 0.74,
    "DATE": 0.68,
    "TIME": 0.55,
    "AGE": 0.85,
    "MONEY": 0.65,
    "PERCENT": 0.42,
    "QUANTITY": 0.46,
    "NUMERIC": 0.56,
    "CARDINAL": 0.40,
    "MISC": 0.58,
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


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def _tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def _safe_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


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
    group: str = GROUP_PUBLIC


@dataclass
class GroupSummary:
    name: str
    span_count: int = 0
    avg_risk: float = 0.0
    avg_utility: float = 0.0
    avg_copy_risk: float = 0.0
    avg_rarity: float = 0.0


@dataclass
class FusionStepRecord:
    step_index: int
    token_id: int
    token_text: str
    entropy: float
    group_lambdas: dict[str, float] = field(default_factory=dict)
    group_divergences: dict[str, float] = field(default_factory=dict)
    adaptive_factors: dict[str, float] = field(default_factory=dict)


class ContextPrivacyExtractor:
    def __init__(self, spacy_model: str = "en_core_web_sm", disable_age_date: bool = False) -> None:
        self.disable_age_date = disable_age_date
        self.nlp = None
        if spacy is None:
            LOGGER.warning("spaCy is unavailable. DenPAD-RF will rely on regex extraction only.")
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
            if any(not (match.end() <= item.start_char or match.start() >= item.end_char) for item in spans):
                continue
            spans.append(SensitiveSpan(match.group(0), "NUMERIC", match.start(), match.end(), doc_index, "regex"))
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
            if not _normalize_space(span.text):
                continue
            overlaps = False
            for start, end in occupied[span.doc_index]:
                if not (span.end_char <= start or span.start_char >= end):
                    overlaps = True
                    break
            if overlaps:
                continue
            merged.append(span)
            occupied[span.doc_index].append((span.start_char, span.end_char))
        return merged


class RiskUtilityScorer:
    def score(self, question: str, docs: list[str], spans: list[SensitiveSpan]) -> list[SensitiveSpan]:
        question_tokens = set(_tokenize(question))
        doc_token_counters = [Counter(_tokenize(doc)) for doc in docs]
        total_docs = max(len(docs), 1)

        for span in spans:
            span_tokens = set(_tokenize(span.text))
            local_window = docs[span.doc_index][max(0, span.start_char - 80) : min(len(docs[span.doc_index]), span.end_char + 80)]
            window_tokens = set(_tokenize(local_window))
            exact_overlap = 1.0 if _normalize_space(span.text).lower() in question.lower() else 0.0
            token_overlap = len(span_tokens & question_tokens) / max(len(span_tokens), 1)
            window_overlap = len(window_tokens & question_tokens) / max(len(question_tokens), 1)
            utility = 0.55 * exact_overlap + 0.25 * token_overlap + 0.20 * window_overlap

            token_counter = doc_token_counters[span.doc_index]
            normalized_tokens = _tokenize(span.text)
            rarity = 1.0
            if normalized_tokens:
                frequencies = [token_counter.get(token, 0) for token in normalized_tokens]
                rarity = 1.0 - min(1.0, float(sum(frequencies)) / max(sum(token_counter.values()), 1))
            copy_risk = max(token_overlap, SequenceMatcher(None, span.text.lower(), question.lower()).ratio())
            label_prior = LABEL_RISK_PRIOR.get(span.label, LABEL_RISK_PRIOR["MISC"])
            length_bonus = 0.12 if len(span.text) >= 12 else 0.0
            digit_bonus = 0.18 if re.search(r"\d", span.text) else 0.0
            doc_uniqueness = 1.0 / total_docs
            risk = label_prior + 0.20 * rarity + 0.10 * copy_risk + length_bonus + digit_bonus + 0.05 * doc_uniqueness

            span.utility_score = _clip(utility, 0.0, 1.0)
            span.rarity_score = _clip(rarity, 0.0, 1.0)
            span.copy_risk = _clip(copy_risk, 0.0, 1.0)
            span.risk_score = _clip(risk, 0.0, 1.0)
            span.group = self._assign_group(span)

        return spans

    def _assign_group(self, span: SensitiveSpan) -> str:
        if span.label in {"EMAIL", "PHONE", "ID"}:
            return GROUP_STRUCT
        if span.label in {"AGE", "DATE", "TIME", "MONEY", "PERCENT", "QUANTITY", "NUMERIC", "CARDINAL"}:
            return GROUP_NUMERIC
        if span.risk_score >= 0.78 and span.utility_score <= 0.25:
            return GROUP_HIDE
        if span.risk_score >= 0.65:
            return GROUP_PRESERVE
        return GROUP_PUBLIC


class ContextViewBuilder:
    def __init__(self, mask_placeholder: str = MASK_PLACEHOLDER) -> None:
        self.mask_placeholder = mask_placeholder

    def build(
        self,
        docs: list[str],
        spans: list[SensitiveSpan],
    ) -> tuple[list[str], dict[str, list[str]], dict[str, GroupSummary], list[dict[str, Any]]]:
        spans_by_doc: dict[int, list[SensitiveSpan]] = defaultdict(list)
        for span in spans:
            spans_by_doc[span.doc_index].append(span)

        for doc_index in spans_by_doc:
            spans_by_doc[doc_index].sort(key=lambda item: item.start_char, reverse=True)

        public_docs = [self._apply_mask(doc, spans_by_doc.get(idx, []), preserve_group=None) for idx, doc in enumerate(docs)]
        views = {
            PUBLIC_VIEW: list(public_docs),
            "VIEW_hide": [self._apply_mask(doc, spans_by_doc.get(idx, []), preserve_group=GROUP_HIDE) for idx, doc in enumerate(docs)],
            "VIEW_preserve": [self._apply_mask(doc, spans_by_doc.get(idx, []), preserve_group=GROUP_PRESERVE) for idx, doc in enumerate(docs)],
            "VIEW_struct": [self._apply_mask(doc, spans_by_doc.get(idx, []), preserve_group=GROUP_STRUCT) for idx, doc in enumerate(docs)],
            "VIEW_num": [self._apply_mask(doc, spans_by_doc.get(idx, []), preserve_group=GROUP_NUMERIC) for idx, doc in enumerate(docs)],
        }
        audit_records: list[dict[str, Any]] = []
        groups: dict[str, list[SensitiveSpan]] = defaultdict(list)
        for span in spans:
            groups[span.group].append(span)
            audit_records.append(
                {
                    "doc_index": span.doc_index,
                    "entity": span.text,
                    "label": span.label,
                    "group": span.group,
                    "risk_score": span.risk_score,
                    "utility_score": span.utility_score,
                    "copy_risk": span.copy_risk,
                    "rarity_score": span.rarity_score,
                    "evidence_source": span.evidence_source,
                }
            )
        summaries = {}
        for group_name in GROUP_ORDER:
            members = groups.get(group_name, [])
            summaries[group_name] = GroupSummary(
                name=group_name,
                span_count=len(members),
                avg_risk=_safe_mean([item.risk_score for item in members]),
                avg_utility=_safe_mean([item.utility_score for item in members]),
                avg_copy_risk=_safe_mean([item.copy_risk for item in members]),
                avg_rarity=_safe_mean([item.rarity_score for item in members]),
            )
        return public_docs, views, summaries, audit_records

    def _apply_mask(self, doc: str, spans: list[SensitiveSpan], preserve_group: Optional[str]) -> str:
        masked = doc
        for span in spans:
            if span.group == GROUP_PUBLIC:
                continue
            if preserve_group is not None and span.group == preserve_group:
                continue
            replacement = self._placeholder(span.label)
            masked = masked[: span.start_char] + replacement + masked[span.end_char :]
        return masked

    def _placeholder(self, label: str) -> str:
        return f"{self.mask_placeholder}{label}{self.mask_placeholder}"


class DenPADFusionAccountant:
    def __init__(self, alpha: float = 2.0, delta: float = 1e-5) -> None:
        self.alpha = alpha
        self.delta = delta
        self.history: dict[str, list[float]] = defaultdict(list)

    def add_step(self, group_name: str, divergence: float) -> None:
        self.history[group_name].append(max(0.0, float(divergence)))

    def compute_global_epsilon(self) -> float:
        if not self.history:
            return 0.0
        group_names = list(self.history.keys())
        step_count = max(len(self.history[name]) for name in group_names)
        total_rdp = 0.0
        n_groups = max(len(group_names), 1)
        for step in range(step_count):
            beta = 0.0
            for group_name in group_names:
                if step < len(self.history[group_name]):
                    beta = max(beta, self.history[group_name][step])
            total_rdp += self._eps_step(beta, n_groups)
        return total_rdp + math.log(1.0 / self.delta) / max(self.alpha - 1.0, 1e-9)

    def compute_per_group_epsilon(self) -> dict[str, float]:
        epsilons = {}
        n_groups = max(len(self.history), 1)
        for group_name, divergences in self.history.items():
            total_rdp = sum(self._eps_step(divergence, n_groups) for divergence in divergences)
            epsilons[group_name] = total_rdp + math.log(1.0 / self.delta) / max(self.alpha - 1.0, 1e-9)
        return epsilons

    def _eps_step(self, beta: float, n_groups: int) -> float:
        arg = (n_groups - 1.0) / n_groups + (1.0 / n_groups) * math.exp((self.alpha - 1.0) * 4.0 * beta)
        return (1.0 / max(self.alpha - 1.0, 1e-9)) * math.log(max(arg, 1e-12))


class RiskFusionDecoder:
    def __init__(
        self,
        model,
        tokenizer,
        alpha: float = 2.0,
        delta: float = 1e-5,
        group_betas: Optional[dict[str, float]] = None,
        group_weights: Optional[dict[str, float]] = None,
        public_weight: float = 0.15,
        max_input_length: int = 2048,
        verbose: bool = False,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.alpha = alpha
        self.delta = delta
        self.group_betas = {**DEFAULT_GROUP_BETAS, **(group_betas or {})}
        self.group_weights = {**DEFAULT_GROUP_WEIGHTS, **(group_weights or {})}
        self.public_weight = public_weight
        self.max_input_length = max_input_length
        self.verbose = verbose
        self.last_stats: dict[str, Any] = {}

    def generate(
        self,
        question: str,
        context_views: dict[str, list[str]],
        group_summaries: dict[str, Any],
        max_new_tokens: int = 256,
        temperature: float = 0.2,
        top_p: float = 0.9,
        do_sample: bool = True,
        repetition_penalty: float = 1.0,
    ) -> tuple[str, dict[str, Any]]:
        normalized_summaries = self._normalize_group_summaries(group_summaries)
        prompts = {
            name: self._build_prompt("\n\n".join(docs), question)
            for name, docs in context_views.items()
        }
        group_order = [PUBLIC_VIEW, *[name for name in prompts if name != PUBLIC_VIEW]]
        tokenized = self.tokenizer(
            [prompts[name] for name in group_order],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max(32, self.max_input_length - max_new_tokens - 16),
        )
        device = self.model.device
        input_ids = tokenized["input_ids"].to(device)
        attention_mask = tokenized["attention_mask"].to(device)

        accountant = DenPADFusionAccountant(alpha=self.alpha, delta=self.delta)
        step_records: list[FusionStepRecord] = []

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True, return_dict=True)
        model_kwargs = {
            "attention_mask": attention_mask,
            "past_key_values": outputs.past_key_values,
            "use_cache": True,
            "cache_position": torch.arange(input_ids.shape[1], device=device),
        }
        generated_tokens: list[int] = []

        for step_index in range(max_new_tokens):
            if step_index == 0:
                logits = self._last_step_logits(outputs.logits, attention_mask)
            else:
                logits = outputs.logits[:, -1, :]

            if repetition_penalty != 1.0 and generated_tokens:
                logits = self._apply_repetition_penalty(logits, generated_tokens, repetition_penalty)

            fused_probs, step_record = self._fuse_step(
                logits=logits,
                group_order=group_order,
                group_summaries=normalized_summaries,
                temperature=temperature,
                step_index=step_index,
            )
            next_token = self._sample(fused_probs, top_p=top_p, do_sample=do_sample)
            generated_tokens.append(next_token)
            step_record.token_id = next_token
            step_record.token_text = self.tokenizer.decode([next_token])
            step_records.append(step_record)
            for group_name, divergence in step_record.group_divergences.items():
                accountant.add_step(group_name, divergence)

            if next_token == self.tokenizer.eos_token_id:
                break

            next_tokens_batch = torch.full((len(group_order), 1), next_token, dtype=torch.long, device=device)
            attention_mask_next = torch.cat(
                [
                    model_kwargs["attention_mask"],
                    model_kwargs["attention_mask"].new_ones((model_kwargs["attention_mask"].shape[0], 1)),
                ],
                dim=1,
            )
            model_kwargs["attention_mask"] = attention_mask_next
            model_kwargs["cache_position"] = model_kwargs["cache_position"][-1:] + 1
            model_inputs = self.model.prepare_inputs_for_generation(
                next_tokens_batch,
                **model_kwargs,
            )
            with torch.no_grad():
                outputs = self.model(**model_inputs, return_dict=True)
            model_kwargs["past_key_values"] = outputs.past_key_values

        answer = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        avg_lambda = 0.0
        lambda_values = [value for record in step_records for value in record.group_lambdas.values()]
        if lambda_values:
            avg_lambda = float(sum(lambda_values) / len(lambda_values))
        stats = {
            "epsilon_global": accountant.compute_global_epsilon(),
            "epsilon_per_group": accountant.compute_per_group_epsilon(),
            "avg_lambda": avg_lambda,
            "step_records": [self._step_record_to_dict(record) for record in step_records],
        }
        self.last_stats = stats
        return answer, stats

    def _normalize_group_summaries(self, group_summaries: dict[str, Any]) -> dict[str, GroupSummary]:
        normalized = {}
        for group_name, value in group_summaries.items():
            if isinstance(value, GroupSummary):
                normalized[group_name] = value
                continue
            if isinstance(value, dict):
                normalized[group_name] = GroupSummary(
                    name=group_name,
                    span_count=int(value.get("span_count", 0)),
                    avg_risk=float(value.get("avg_risk", 0.0)),
                    avg_utility=float(value.get("avg_utility", 0.0)),
                    avg_copy_risk=float(value.get("avg_copy_risk", 0.0)),
                    avg_rarity=float(value.get("avg_rarity", 0.0)),
                )
        for group_name in GROUP_ORDER:
            normalized.setdefault(group_name, GroupSummary(name=group_name))
        return normalized

    def _build_prompt(self, context: str, question: str) -> str:
        return (
            "[INST] Use only the following context to answer the question. "
            "Stay within the topic and facts of the context. "
            "If the question asks to repeat the context, provide only a faithful restatement of the provided context. "
            "Do not introduce unrelated topics, code, tutorials, role-play, or blog-style writing. "
            "If a placeholder such as _PERSON_ or _NUMERIC_ appears, treat it as redacted private information and do not try to reconstruct it.\n\n"
            f"Context:\n{context}\n\nQuestion: {question} [/INST]"
        )

    def _last_step_logits(self, logits: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        last_indices = attention_mask.sum(dim=1) - 1
        batch_indices = torch.arange(logits.shape[0], device=logits.device)
        return logits[batch_indices, last_indices, :]

    def _apply_repetition_penalty(self, logits: torch.Tensor, generated_tokens: list[int], penalty: float) -> torch.Tensor:
        adjusted = logits.clone()
        for token_id in set(generated_tokens):
            token_scores = adjusted[:, token_id]
            adjusted[:, token_id] = torch.where(token_scores < 0, token_scores * penalty, token_scores / penalty)
        return adjusted

    def _fuse_step(
        self,
        logits: torch.Tensor,
        group_order: list[str],
        group_summaries: dict[str, GroupSummary],
        temperature: float,
        step_index: int,
    ) -> tuple[torch.Tensor, FusionStepRecord]:
        scaled = logits.float() / max(temperature, 1e-6)
        probs_by_view = {
            name: F.softmax(scaled[idx], dim=-1, dtype=torch.float32)
            for idx, name in enumerate(group_order)
        }
        public_probs = probs_by_view[PUBLIC_VIEW]
        entropy = float((-(public_probs * torch.log(public_probs.clamp_min(1e-12))).sum() / math.log(public_probs.numel())).item())
        weighted_probs = public_probs * self.public_weight
        total_weight = self.public_weight
        step_record = FusionStepRecord(step_index=step_index, token_id=-1, token_text="", entropy=entropy)

        for view_name in group_order[1:]:
            group_name = VIEW_TO_GROUP.get(view_name)
            if group_name is None:
                continue
            summary = group_summaries.get(group_name, GroupSummary(group_name))
            private_probs = probs_by_view[view_name]
            beta = self.group_betas.get(group_name, DEFAULT_GROUP_BETAS.get(group_name, 0.1))
            lambda_bound, divergence = find_lambda(private_probs, public_probs, self.alpha, beta)
            adaptive_factor = self._adaptive_factor(summary, entropy)
            lambda_value = lambda_bound * adaptive_factor
            mixed_probs = lambda_value * private_probs + (1.0 - lambda_value) * public_probs
            group_weight = self._group_weight(summary, group_name)
            weighted_probs = weighted_probs + mixed_probs * group_weight
            total_weight += group_weight
            step_record.group_lambdas[group_name] = float(lambda_value)
            step_record.group_divergences[group_name] = float(divergence)
            step_record.adaptive_factors[group_name] = float(adaptive_factor)

        fused_probs = weighted_probs / max(total_weight, 1e-9)
        fused_probs = fused_probs / fused_probs.sum()
        return fused_probs, step_record

    def _adaptive_factor(self, summary: GroupSummary, entropy: float) -> float:
        factor = 0.15 + 0.70 * summary.avg_utility + 0.20 * entropy - 0.40 * summary.avg_risk - 0.15 * summary.avg_copy_risk
        return _clip(factor, 0.05, 1.0)

    def _group_weight(self, summary: GroupSummary, group_name: str) -> float:
        base = self.group_weights.get(group_name, 1.0)
        adjusted = base + 0.60 * summary.avg_utility - 0.25 * summary.avg_risk
        return _clip(adjusted, 0.10, 2.5)

    def _sample(self, probs: torch.Tensor, top_p: float, do_sample: bool) -> int:
        if not do_sample:
            return int(torch.argmax(probs).item())
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        nucleus_mask = cumulative <= top_p
        nucleus_mask[0] = True
        filtered_probs = torch.where(nucleus_mask, sorted_probs, torch.zeros_like(sorted_probs))
        filtered_probs = filtered_probs / filtered_probs.sum().clamp_min(1e-12)
        sampled = torch.multinomial(filtered_probs, 1).item()
        return int(sorted_indices[sampled].item())

    def _step_record_to_dict(self, record: FusionStepRecord) -> dict[str, Any]:
        return {
            "step_index": record.step_index,
            "token_id": record.token_id,
            "token_text": record.token_text,
            "entropy": record.entropy,
            "group_lambdas": record.group_lambdas,
            "group_divergences": record.group_divergences,
            "adaptive_factors": record.adaptive_factors,
        }


class DenPADRFSanitizer:
    def __init__(
        self,
        spacy_model: str = "en_core_web_sm",
        mask_placeholder: str = MASK_PLACEHOLDER,
        disable_age_date: bool = False,
        collapse_groups: bool = False,
    ) -> None:
        self.extractor = ContextPrivacyExtractor(spacy_model=spacy_model, disable_age_date=disable_age_date)
        self.scorer = RiskUtilityScorer()
        self.builder = ContextViewBuilder(mask_placeholder=mask_placeholder)
        self.collapse_groups = collapse_groups

    def sanitize_retrieved_docs(
        self,
        docs: list[str],
        query: Optional[str] = None,
    ) -> tuple[list[str], dict[str, Any]]:
        query = query or ""
        spans: list[SensitiveSpan] = []
        for doc_index, doc in enumerate(docs):
            spans.extend(self.extractor.extract(doc, doc_index))
        spans = self.scorer.score(query, docs, spans)
        if self.collapse_groups:
            for span in spans:
                if span.group != GROUP_PUBLIC:
                    span.group = GROUP_PRESERVE
        public_docs, views, summaries, audit_records = self.builder.build(docs, spans)
        if self.collapse_groups:
            views = {name: docs for name, docs in views.items() if name in {PUBLIC_VIEW, "VIEW_preserve"}}
            summaries = {name: summary for name, summary in summaries.items() if name == GROUP_PRESERVE}
        metadata = {
            "num_entities": len(spans),
            "num_perturbed": sum(1 for span in spans if span.group != GROUP_PUBLIC),
            "retained_entities_by_label": dict(Counter(span.label for span in spans)),
            "selected_level_counts": {group: summary.span_count for group, summary in summaries.items()},
            "context_views": views,
            "view_summaries": {
                group: {
                    "span_count": summary.span_count,
                    "avg_risk": summary.avg_risk,
                    "avg_utility": summary.avg_utility,
                    "avg_copy_risk": summary.avg_copy_risk,
                    "avg_rarity": summary.avg_rarity,
                }
                for group, summary in summaries.items()
            },
            "audit_records": audit_records,
        }
        return public_docs, metadata


def compute_renyi_divergence_clipped_symmetric(p: torch.Tensor, q: torch.Tensor, alpha: float, eps: float = 1e-12) -> float:
    if alpha <= 1.0:
        raise ValueError("alpha must be > 1")
    p = torch.nan_to_num(p.float(), nan=eps, posinf=1.0, neginf=eps).clamp_min(eps)
    q = torch.nan_to_num(q.float(), nan=eps, posinf=1.0, neginf=eps).clamp_min(eps)
    p = p / p.sum().clamp_min(eps)
    q = q / q.sum().clamp_min(eps)
    term_pq = torch.sum(p.pow(alpha) * q.pow(1.0 - alpha), dim=-1).clamp_min(eps)
    term_qp = torch.sum(q.pow(alpha) * p.pow(1.0 - alpha), dim=-1).clamp_min(eps)
    div_pq = (1.0 / (alpha - 1.0)) * torch.log(term_pq)
    div_qp = (1.0 / (alpha - 1.0)) * torch.log(term_qp)
    divergence = torch.maximum(div_pq, div_qp)
    divergence = torch.nan_to_num(divergence, nan=0.0, posinf=1e6, neginf=0.0)
    return float(divergence.item())


def find_lambda(
    p_priv: torch.Tensor,
    p_pub: torch.Tensor,
    alpha: float,
    beta: float,
    max_iter: int = 24,
    tol: float = 1e-6,
) -> tuple[float, float]:
    if beta <= 0:
        return 0.0, 0.0
    div_at_one = compute_renyi_divergence_clipped_symmetric(p_priv, p_pub, alpha)
    if not math.isfinite(div_at_one):
        return 0.0, 0.0
    if div_at_one <= beta:
        return 1.0, div_at_one
    left, right = 0.0, 1.0
    final_div = 0.0
    for _ in range(max_iter):
        mid = 0.5 * (left + right)
        mixture = mid * p_priv + (1.0 - mid) * p_pub
        divergence = compute_renyi_divergence_clipped_symmetric(mixture, p_pub, alpha)
        final_div = divergence
        if divergence > beta:
            right = mid
        else:
            left = mid
        if (right - left) < tol:
            break
    final_lambda = left
    mixture = final_lambda * p_priv + (1.0 - final_lambda) * p_pub
    final_div = compute_renyi_divergence_clipped_symmetric(mixture, p_pub, alpha)
    return final_lambda, final_div
