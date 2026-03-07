import logging
import json
import math
import os
import random
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Optional

import numpy as np
from tqdm import tqdm
try:
    import gensim.downloader as api
except ImportError:
    api = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

try:
    import spacy
except ImportError:
    spacy = None

from candidate_llm_completion import CandidateLLMCompletion
from medical_typer import MedicalTyper
from typed_candidate_index import TypedCandidateIndex


LOGGER = logging.getLogger(__name__)


DEFAULT_ENTITY_LABELS = {
    "PERSON",
    "ORG",
    "GPE",
    "LOC",
    "DATE",
    "TIME",
    "MONEY",
    "PERCENT",
    "CARDINAL",
    "QUANTITY",
}

SENSITIVE_PRIORITY_LABELS = {
    "EMAIL",
    "PHONE",
    "AGE",
    "DISEASE",
    "DRUG",
    "ORG",
    "GPE",
    "LOC",
    "DATE",
}

SENSITIVE_PRIORITY_ORDER = [
    "DRUG",
    "DISEASE",
    "EMAIL",
    "PHONE",
    "AGE",
    "ORG",
    "GPE",
    "LOC",
    "DATE",
]

GENERIC_NON_SENSITIVE_TERMS = {
    "doc",
    "docs",
    "doctor",
    "google",
    "today",
    "yesterday",
    "tomorrow",
    "morning",
    "evening",
    "afternoon",
    "night",
    "week",
    "month",
    "year",
    "day",
    "few hours",
    "a week",
    "one month",
    "the past few months",
    "the last few days",
    "this morning",
    "doesn",
    "don",
    "didn",
    "isn",
    "aren",
    "won",
    "shouldn",
    "wouldn",
    "couldn",
}

GENERIC_DISEASE_TERMS = {
    "infection",
    "infections",
    "disease",
    "diseases",
    "cancer",
    "virus",
    "viral infection",
    "bacterial infection",
    "bacterial infections",
}

BAD_DISEASE_CANDIDATE_SUFFIXES = {
    "sufferer",
    "sufferers",
    "patient",
    "patients",
    "case",
    "cases",
}

ORG_HINT_TERMS = {
    "hospital",
    "clinic",
    "center",
    "centre",
    "lab",
    "laboratory",
    "university",
    "institute",
    "inc",
    "corp",
    "company",
    "ltd",
}

MONTH_TERMS = {
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
}

DEFAULT_FALLBACK_CANDIDATES = {
    "PERSON": ["Alex", "Jordan", "Taylor", "Morgan", "Casey"],
    "ORG": ["General Hospital", "City Clinic", "Health Center", "Regional Lab"],
    "GPE": ["Boston", "Chicago", "Seattle", "Denver"],
    "LOC": ["downtown clinic", "north campus", "main office"],
    "DISEASE": ["dermatitis", "migraine", "eczema", "sinusitis", "arthritis"],
    "DRUG": ["ibuprofen", "amoxicillin", "acetaminophen", "metformin", "aspirin"],
    "DATE": ["last week", "two months ago", "recently"],
    "AGE": ["29", "41", "52"],
}

DEFAULT_MEDICAL_DISEASE_TERMS = {
    "alopecia areata",
    "carpal tunnel syndrome",
    "interstitial lung disease",
    "gonorrhea",
    "aplastic anemia",
    "jaundice",
    "alopecia",
    "psoriasis",
    "eczema",
    "dermatitis",
    "asthma",
    "diabetes",
    "migraine",
    "cancer",
    "hypertension",
    "arthritis",
    "infection",
    "ild",
    "anxiety",
    "depression",
    "sinusitis",
}

DEFAULT_MEDICAL_DRUG_TERMS = {
    "ibuprofen",
    "amoxicillin",
    "amoxxicilin",
    "amoxycillin",
    "zoloft",
    "votrient",
    "eurax",
    "metformin",
    "paracetamol",
    "acetaminophen",
    "prednisone",
    "lyrica",
    "trileptal",
    "aspirin",
    "euthyrox",
    "levothyroxine",
}

MEDICAL_DISEASE_HINTS = {
    "h pylori",
    "hpylori",
    "interstitial lung",
    "ild",
    "gonorr",
    "alopecia",
    "psoriasis",
    "eczema",
    "dermatitis",
    "migraine",
    "cancer",
    "jaundice",
    "anemia",
    "syndrome",
}

MEDICAL_DRUG_HINTS = {
    "amoxi",
    "zoloft",
    "votrient",
    "eurax",
    "metformin",
    "ibuprofen",
    "aspirin",
    "prednisone",
    "trileptal",
    "lyrica",
    "acetaminophen",
    "paracetamol",
    "euthyrox",
    "thyrox",
    "levothyroxine",
}

MEDICAL_TEST_HINTS = {
    "scan",
    "biopsy",
    "blood",
    "serum",
    "creatinine",
    "prandial",
    "physical",
    "ultrasound",
    "mri",
    "ct",
    "xray",
}

EMAIL_PATTERN = re.compile(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b")
PHONE_PATTERN = re.compile(r"(?:\+?\d[\d\-\s()]{7,}\d)")
AGE_PATTERN = re.compile(r"\b(\d{1,3})\s*(?:years?\s*old|y/o)\b", re.IGNORECASE)


def get_default_resources_dir() -> str:
    return os.path.join(os.path.dirname(__file__), "resources")


@dataclass(frozen=True)
class ResourceTerm:
    term: str
    canonical_id: str = ""
    aliases: tuple[str, ...] = ()
    related: tuple[str, ...] = ()
    generalized: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()

    @property
    def normalized_term(self) -> str:
        return normalize_entity_text(self.term).lower()

    @property
    def normalized_aliases(self) -> tuple[str, ...]:
        return tuple(normalize_entity_text(alias).lower() for alias in self.aliases if normalize_entity_text(alias))

    @property
    def normalized_related(self) -> tuple[str, ...]:
        return tuple(normalize_entity_text(item).lower() for item in self.related if normalize_entity_text(item))

    @property
    def normalized_generalized(self) -> tuple[str, ...]:
        return tuple(normalize_entity_text(item).lower() for item in self.generalized if normalize_entity_text(item))

    @property
    def normalized_tags(self) -> tuple[str, ...]:
        return tuple(normalize_entity_text(tag).lower() for tag in self.tags if normalize_entity_text(tag))


class ResourceRegistry:
    def __init__(self, resources_dir: Optional[str] = None) -> None:
        self.resources_dir = resources_dir or get_default_resources_dir()
        self.loaded_resource_info: dict[str, dict[str, Any]] = {}
        self.resource_manifest = self._load_manifest()
        self.records = {
            "DISEASE": self._load_records(
                "DISEASE",
                ["medical_disease_index.json", "disease_terms.json"],
                DEFAULT_MEDICAL_DISEASE_TERMS,
            ),
            "DRUG": self._load_records(
                "DRUG",
                ["medical_drug_index.json", "drug_terms.json"],
                DEFAULT_MEDICAL_DRUG_TERMS,
            ),
            "PERSON": self._load_records("PERSON", "person_names.json", set(DEFAULT_FALLBACK_CANDIDATES["PERSON"])),
            "ORG": self._load_records("ORG", "org_terms.json", set(DEFAULT_FALLBACK_CANDIDATES["ORG"])),
            "GPE": self._load_records("GPE", "location_terms.json", set(DEFAULT_FALLBACK_CANDIDATES["GPE"])),
        }
        self.records["LOC"] = self.records["GPE"]
        self.disease_terms = self._flatten_terms("DISEASE")
        self.drug_terms = self._flatten_terms("DRUG")
        self.person_names = self._flatten_terms("PERSON")
        self.org_terms = self._flatten_terms("ORG")
        self.location_terms = self._flatten_terms("GPE")
        self.loaded_resource_info["LOC"] = self.loaded_resource_info.get("GPE", {})

    def _load_manifest(self) -> dict[str, Any]:
        manifest_path = os.path.join(self.resources_dir, "typed_resource_manifest.json")
        if not os.path.exists(manifest_path):
            return {}
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if isinstance(payload, dict):
                payload["_manifest_path"] = manifest_path
                return payload
        except Exception as exc:
            LOGGER.warning("Failed to load typed resource manifest %s: %s", manifest_path, exc)
        return {}

    def _load_records(self, category: str, filename: str | list[str], fallback: set[str]) -> list[ResourceTerm]:
        filenames = [filename] if isinstance(filename, str) else list(filename)
        last_error: Optional[Exception] = None
        for candidate_name in filenames:
            path = os.path.join(self.resources_dir, candidate_name)
            if not os.path.exists(path):
                continue
            try:
                with open(path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                if isinstance(payload, dict):
                    items = payload.get("items", [])
                    metadata = payload.get("metadata", {}) if isinstance(payload.get("metadata", {}), dict) else {}
                elif isinstance(payload, list):
                    items = payload
                    metadata = {}
                else:
                    raise ValueError(f"Unsupported resource payload type: {type(payload).__name__}")
                records = [self._parse_resource_item(item) for item in items]
                loaded = [record for record in records if record is not None]
                if loaded:
                    LOGGER.info("Loaded %d resource records from %s", len(loaded), path)
                    self.loaded_resource_info[category] = {
                        "source_file": path,
                        "source_type": "file",
                        "record_count": len(loaded),
                        "metadata": metadata,
                    }
                    return loaded
            except Exception as exc:
                last_error = exc
                LOGGER.warning("Failed to load resource %s: %s", path, exc)

        if last_error is not None:
            LOGGER.warning("Falling back to built-in defaults after resource load errors.")
        records = [self._parse_resource_item(item) for item in sorted(fallback)]
        fallback_records = [record for record in records if record is not None]
        self.loaded_resource_info[category] = {
            "source_file": None,
            "source_type": "fallback",
            "record_count": len(fallback_records),
            "metadata": {},
        }
        return fallback_records

    def _parse_resource_item(self, item: Any) -> Optional[ResourceTerm]:
        if isinstance(item, str):
            normalized = normalize_entity_text(item)
            if not normalized:
                return None
            return ResourceTerm(term=normalized, canonical_id=f"auto:{normalized.lower().replace(' ', '_')}")
        if isinstance(item, dict):
            term = normalize_entity_text(str(item.get("term", "")).strip())
            if not term:
                return None
            canonical_id = normalize_entity_text(str(item.get("canonical_id", "")).strip())
            if not canonical_id:
                canonical_id = f"auto:{term.lower().replace(' ', '_')}"
            aliases = tuple(
                normalize_entity_text(str(alias).strip())
                for alias in item.get("aliases", [])
                if normalize_entity_text(str(alias).strip())
            )
            related = tuple(
                normalize_entity_text(str(candidate).strip())
                for candidate in item.get("related", [])
                if normalize_entity_text(str(candidate).strip())
            )
            generalized = tuple(
                normalize_entity_text(str(candidate).strip())
                for candidate in item.get("generalized", [])
                if normalize_entity_text(str(candidate).strip())
            )
            tags = tuple(
                normalize_entity_text(str(tag).strip())
                for tag in item.get("tags", [])
                if normalize_entity_text(str(tag).strip())
            )
            return ResourceTerm(
                term=term,
                canonical_id=canonical_id,
                aliases=aliases,
                related=related,
                generalized=generalized,
                tags=tags,
            )
        return None

    def _flatten_terms(self, category: str) -> set[str]:
        flattened = set()
        for record in self.records.get(category, []):
            flattened.add(record.normalized_term)
            flattened.update(record.normalized_aliases)
            flattened.update(record.normalized_related)
            flattened.update(record.normalized_generalized)
        return flattened

    def get_terms(self, category: str) -> set[str]:
        if category == "DISEASE":
            return self.disease_terms
        if category == "DRUG":
            return self.drug_terms
        if category == "PERSON":
            return self.person_names
        if category == "ORG":
            return self.org_terms
        if category in {"GPE", "LOC"}:
            return self.location_terms
        return set()

    def get_candidates(self, category: str) -> list[str]:
        return sorted(self.get_terms(category))

    def find_record(self, category: str, text: str) -> Optional[ResourceTerm]:
        normalized = normalize_entity_text(text).lower()
        for record in self.records.get(category, []):
            if (
                normalized == record.normalized_term
                or normalized in record.normalized_aliases
                or normalized in record.normalized_related
                or normalized in record.normalized_generalized
            ):
                return record
        return None

    def get_candidates_for_query(self, category: str, text: str) -> list[str]:
        record = self.find_record(category, text)
        if record is None:
            return self.get_candidates(category)

        candidates = [record.term, *record.aliases, *record.related, *record.generalized]
        return [normalize_entity_text(candidate) for candidate in candidates if normalize_entity_text(candidate)]

    def candidate_level(self, category: str, original: str, candidate: str) -> str:
        normalized_original = normalize_entity_text(original).lower()
        normalized_candidate = normalize_entity_text(candidate).lower()
        if not normalized_candidate:
            return "unknown"
        if normalized_candidate == normalized_original:
            return "original"
        record = self.find_record(category, original)
        if record is None:
            return "global"
        if normalized_candidate == record.normalized_term:
            return "canonical"
        if normalized_candidate in record.normalized_aliases:
            return "alias"
        if normalized_candidate in record.normalized_generalized:
            return "generalized"
        if normalized_candidate in record.normalized_related:
            return "related"
        return "global"

    def is_generalized_candidate(self, category: str, original: str, candidate: str) -> bool:
        record = self.find_record(category, original)
        if record is None:
            return False
        normalized = normalize_entity_text(candidate).lower()
        return normalized in record.normalized_generalized

    def get_resource_summary(self) -> dict[str, Any]:
        summary = {
            "resources_dir": self.resources_dir,
            "manifest_loaded": bool(self.resource_manifest),
            "manifest_path": self.resource_manifest.get("_manifest_path"),
            "manifest_version": self.resource_manifest.get("manifest_version"),
            "schema_version": self.resource_manifest.get("schema_version"),
            "categories": {},
        }
        for category, info in self.loaded_resource_info.items():
            summary["categories"][category] = {
                "source_file": info.get("source_file"),
                "source_type": info.get("source_type"),
                "record_count": info.get("record_count", 0),
                "metadata": info.get("metadata", {}),
            }
        return summary


@dataclass
class ExtractedEntity:
    text: str
    label: str
    start_char: int
    end_char: int
    normalized_text: str
    category: str
    density: Optional[float] = None
    epsilon: Optional[float] = None
    doc_index: int = -1
    evidence_confidence: float = 0.0
    evidence_source: str = ""
    should_perturb: bool = True
    candidate_trace: dict[str, Any] = field(default_factory=dict)


@dataclass
class PerturbationRecord:
    original_text: str
    perturbed_text: str
    label: str
    category: str
    epsilon: float
    density: float
    mechanism: str
    start_char: int
    end_char: int
    candidates: list[str] = field(default_factory=list)
    candidate_levels: dict[str, str] = field(default_factory=dict)
    candidate_sources: dict[str, str] = field(default_factory=dict)
    candidate_scores: list[float] = field(default_factory=list)
    candidate_probabilities: list[float] = field(default_factory=list)
    selected_level: str = "unknown"
    candidate_trace: dict[str, Any] = field(default_factory=dict)


@dataclass
class SanitizationResult:
    original_text: str
    sanitized_text: str
    entities: list[ExtractedEntity]
    perturbations: list[PerturbationRecord]
    epsilon_doc: float
    metadata: dict[str, Any] = field(default_factory=dict)


def normalize_entity_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def is_ascii_text(text: str) -> bool:
    try:
        text.encode("ascii")
        return True
    except UnicodeEncodeError:
        return False


def looks_like_disease_text(text: str, registry: Optional[ResourceRegistry] = None) -> bool:
    lowered = text.lower()
    if registry is not None and lowered in registry.disease_terms:
        return True
    if lowered in DEFAULT_MEDICAL_DISEASE_TERMS:
        return True
    return any(hint in lowered for hint in MEDICAL_DISEASE_HINTS)


def looks_like_drug_text(text: str, registry: Optional[ResourceRegistry] = None) -> bool:
    lowered = text.lower()
    if registry is not None and lowered in registry.drug_terms:
        return True
    if lowered in DEFAULT_MEDICAL_DRUG_TERMS:
        return True
    return any(hint in lowered for hint in MEDICAL_DRUG_HINTS)


def looks_like_medical_test_text(text: str) -> bool:
    lowered = text.lower()
    return any(hint in lowered for hint in MEDICAL_TEST_HINTS)


def detect_entity_category(label: str, text: str) -> str:
    if label in {"EMAIL", "PHONE"}:
        return "structured"
    if label in {"DATE", "TIME", "MONEY", "PERCENT", "CARDINAL", "QUANTITY", "AGE"}:
        return "numeric"
    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", text.strip()):
        return "numeric"
    return "categorical"


def safe_softmax(logits: list[float]) -> list[float]:
    if not logits:
        return []
    arr = np.asarray(logits, dtype=np.float64)
    arr = arr - np.max(arr)
    exps = np.exp(arr)
    denom = np.sum(exps)
    if denom <= 0 or not np.isfinite(denom):
        return [1.0 / len(logits)] * len(logits)
    return (exps / denom).tolist()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def apply_perturbations_by_span(text: str, perturbations: list[PerturbationRecord]) -> str:
    result = text
    for record in sorted(perturbations, key=lambda item: item.start_char, reverse=True):
        result = result[: record.start_char] + record.perturbed_text + result[record.end_char :]
    return result


class EntityExtractor:
    def __init__(
        self,
        spacy_model: str = "en_core_web_trf",
        enabled_labels: Optional[set[str]] = None,
        resource_registry: Optional[ResourceRegistry] = None,
        medical_typer: Optional[MedicalTyper] = None,
    ) -> None:
        self.enabled_labels = enabled_labels or DEFAULT_ENTITY_LABELS
        self.resource_registry = resource_registry or ResourceRegistry()
        self.medical_typer = medical_typer or MedicalTyper(resource_registry=self.resource_registry)
        self.nlp = None
        if spacy is None:
            LOGGER.warning("spaCy is unavailable. DenPAD-L will fall back to regex and keyword extraction only.")
        else:
            try:
                self.nlp = spacy.load(spacy_model)
            except OSError:
                self.nlp = spacy.load("en_core_web_sm")

    def extract(self, text: str, doc_index: int = -1) -> list[ExtractedEntity]:
        entities = []
        entities.extend(self._extract_spacy_entities(text, doc_index=doc_index))
        entities.extend(self._extract_regex_entities(text, doc_index=doc_index))
        entities.extend(self._extract_medical_entities(text, doc_index=doc_index))
        return self._merge_entities(entities)

    def _extract_spacy_entities(self, text: str, doc_index: int = -1) -> list[ExtractedEntity]:
        if self.nlp is None:
            return []
        doc = self.nlp(text)
        results = []
        for ent in doc.ents:
            if ent.label_ not in self.enabled_labels:
                continue
            normalized_text = normalize_entity_text(ent.text)
            decision = self._canonicalize_label(ent.label_, normalized_text)
            label = decision.label
            entity = ExtractedEntity(
                text=ent.text,
                label=label,
                start_char=ent.start_char,
                end_char=ent.end_char,
                normalized_text=normalized_text,
                category=detect_entity_category(label, ent.text),
                doc_index=doc_index,
                evidence_confidence=decision.confidence,
                evidence_source=decision.source,
            )
            if self._should_keep(entity):
                results.append(entity)
        return results

    def _extract_regex_entities(self, text: str, doc_index: int = -1) -> list[ExtractedEntity]:
        results = []
        for match in EMAIL_PATTERN.finditer(text):
            results.append(
                ExtractedEntity(
                    text=match.group(0),
                    label="EMAIL",
                    start_char=match.start(),
                    end_char=match.end(),
                    normalized_text=normalize_entity_text(match.group(0)),
                    category="structured",
                    doc_index=doc_index,
                    evidence_confidence=0.99,
                    evidence_source="regex",
                )
            )
        for match in PHONE_PATTERN.finditer(text):
            candidate = match.group(0).strip()
            digit_count = len(re.sub(r"\D", "", candidate))
            if digit_count < 8:
                continue
            results.append(
                ExtractedEntity(
                    text=candidate,
                    label="PHONE",
                    start_char=match.start(),
                    end_char=match.end(),
                    normalized_text=normalize_entity_text(candidate),
                    category="structured",
                    doc_index=doc_index,
                    evidence_confidence=0.99,
                    evidence_source="regex",
                )
            )
        for match in AGE_PATTERN.finditer(text):
            results.append(
                ExtractedEntity(
                    text=match.group(1),
                    label="AGE",
                    start_char=match.start(1),
                    end_char=match.end(1),
                    normalized_text=normalize_entity_text(match.group(1)),
                    category="numeric",
                    doc_index=doc_index,
                    evidence_confidence=0.95,
                    evidence_source="regex",
                )
            )
        return results

    def _extract_medical_entities(self, text: str, doc_index: int = -1) -> list[ExtractedEntity]:
        results = []
        lowered = text.lower()
        for term in sorted(self.resource_registry.disease_terms, key=len, reverse=True):
            for match in re.finditer(r"\b%s\b" % re.escape(term), lowered):
                span_text = text[match.start() : match.end()]
                results.append(
                    ExtractedEntity(
                        text=span_text,
                        label="DISEASE",
                        start_char=match.start(),
                        end_char=match.end(),
                        normalized_text=normalize_entity_text(span_text),
                        category="categorical",
                        doc_index=doc_index,
                        evidence_confidence=0.99,
                        evidence_source="resource",
                    )
                )
        for term in sorted(self.resource_registry.drug_terms, key=len, reverse=True):
            for match in re.finditer(r"\b%s\b" % re.escape(term), lowered):
                span_text = text[match.start() : match.end()]
                results.append(
                    ExtractedEntity(
                        text=span_text,
                        label="DRUG",
                        start_char=match.start(),
                        end_char=match.end(),
                        normalized_text=normalize_entity_text(span_text),
                        category="categorical",
                        doc_index=doc_index,
                        evidence_confidence=0.99,
                        evidence_source="resource",
                    )
                )
        return results

    def _merge_entities(self, entities: list[ExtractedEntity]) -> list[ExtractedEntity]:
        deduped = []
        occupied = []
        priority = {label: index for index, label in enumerate(SENSITIVE_PRIORITY_ORDER)}
        for entity in sorted(
            entities,
            key=lambda item: (
                item.start_char,
                priority.get(item.label, 999),
                -(item.end_char - item.start_char),
            ),
        ):
            overlap = False
            for start, end in occupied:
                if not (entity.end_char <= start or entity.start_char >= end):
                    overlap = True
                    break
            if overlap or not self._should_keep(entity):
                continue
            occupied.append((entity.start_char, entity.end_char))
            deduped.append(entity)
        return deduped

    def _should_keep(self, entity: ExtractedEntity) -> bool:
        stripped = entity.normalized_text.strip()
        if not stripped:
            return False
        if len(stripped) <= 1:
            return False
        lowered = stripped.lower()
        if lowered in GENERIC_NON_SENSITIVE_TERMS:
            return False
        if entity.label == "DISEASE" and lowered in GENERIC_DISEASE_TERMS:
            return False
        if entity.label not in SENSITIVE_PRIORITY_LABELS:
            return False
        if entity.label in {"ORG", "GPE", "LOC", "PERSON"} and looks_like_medical_test_text(lowered):
            return False
        if entity.label in {"PERSON", "ORG", "GPE", "LOC"} and lowered.islower() and len(lowered) <= 4:
            return False
        if entity.label == "ORG" and not self._looks_like_org(stripped):
            return False
        if entity.label == "GPE" and not self._looks_like_location(stripped):
            return False
        if entity.label == "DATE" and not self._looks_like_sensitive_date(stripped):
            return False
        if entity.label == "TIME" and not self._looks_like_sensitive_time(stripped):
            return False
        if entity.label == "PERSON" and not self._looks_like_person(stripped):
            return False
        if entity.label in {"ORG", "GPE", "LOC"} and entity.evidence_confidence < 0.75:
            return False
        return True

    def _canonicalize_label(self, label: str, text: str):
        return self.medical_typer.classify(text, label)

    def _looks_like_org(self, text: str) -> bool:
        lowered = text.lower()
        if looks_like_disease_text(lowered, self.resource_registry) or looks_like_drug_text(lowered, self.resource_registry):
            return False
        if self.medical_typer.has_strong_evidence(text, "DISEASE") or self.medical_typer.has_strong_evidence(text, "DRUG"):
            return False
        if any(term in lowered for term in ORG_HINT_TERMS):
            return True
        if lowered in self.resource_registry.org_terms:
            return True
        parts = text.split()
        if len(parts) >= 2 and any(part[:1].isupper() for part in parts):
            return True
        return False

    def _looks_like_location(self, text: str) -> bool:
        lowered = text.lower()
        if (
            looks_like_disease_text(lowered, self.resource_registry)
            or looks_like_drug_text(lowered, self.resource_registry)
            or looks_like_medical_test_text(lowered)
            or self.medical_typer.has_strong_evidence(text, "DISEASE")
            or self.medical_typer.has_strong_evidence(text, "DRUG")
        ):
            return False
        if lowered in self.resource_registry.location_terms:
            return True
        if re.search(r"\d", text):
            return False
        parts = text.split()
        return bool(parts) and all(part[:1].isupper() for part in parts if part)

    def _looks_like_sensitive_date(self, text: str) -> bool:
        lowered = text.lower()
        if re.search(r"\b\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?\b", text):
            return True
        if re.search(r"\b(19|20)\d{2}\b", text):
            return True
        if any(month in lowered for month in MONTH_TERMS):
            return True
        return False

    def _looks_like_sensitive_time(self, text: str) -> bool:
        if re.search(r"\b\d{1,2}:\d{2}\b", text):
            return True
        if re.search(r"\b\d{1,2}\s*(am|pm)\b", text.lower()):
            return True
        return False

    def _looks_like_person(self, text: str) -> bool:
        lowered = text.lower()
        if (
            looks_like_disease_text(lowered, self.resource_registry)
            or looks_like_drug_text(lowered, self.resource_registry)
            or looks_like_medical_test_text(lowered)
            or self.medical_typer.has_strong_evidence(text, "DISEASE")
            or self.medical_typer.has_strong_evidence(text, "DRUG")
        ):
            return False
        if lowered in self.resource_registry.person_names:
            return True
        parts = text.split()
        if len(parts) >= 2 and all(part[:1].isupper() for part in parts if part):
            return True
        return False


class DensityScorer:
    def __init__(
        self,
        backend: str = "word2vec-google-news-300",
        k: int = 20,
        cache_path: Optional[str] = None,
    ) -> None:
        self.backend = backend
        self.k = k
        self.cache_path = cache_path
        self.model = None
        if api is None:
            LOGGER.warning("gensim is unavailable. DenPAD-L will use fallback densities and candidates.")
        else:
            try:
                self.model = api.load(backend)
            except Exception as exc:
                LOGGER.warning("Failed to load density backend %s: %s", backend, exc)
        self.sparsity_cache = {}
        self.min_sparsity = 0.0
        self.max_sparsity = 10.0

    def score_entities(self, entities: list[ExtractedEntity]) -> list[ExtractedEntity]:
        for entity in entities:
            entity.density = self.score_entity(entity.normalized_text, entity.label)
        return entities

    def score_entity(self, text: str, label: str) -> float:
        key = normalize_entity_text(text).lower()
        if key in self.sparsity_cache:
            return self.sparsity_cache[key]

        try:
            embedding = self._embed_text(key)
            sparsity = self._compute_sparsity(embedding, key)
            density = self._normalize_density(sparsity)
        except Exception:
            density = self._default_density_for_label(label)
        self.sparsity_cache[key] = density
        return density

    def _embed_text(self, text: str) -> np.ndarray:
        if self.model is None:
            raise KeyError(text)
        token_key = text.replace(" ", "_")
        if token_key in self.model:
            return np.asarray(self.model[token_key], dtype=np.float64)

        pieces = [piece for piece in re.split(r"\s+", text) if piece and piece in self.model]
        if not pieces:
            raise KeyError(text)
        vectors = [np.asarray(self.model[piece], dtype=np.float64) for piece in pieces]
        return np.mean(vectors, axis=0)

    def _compute_sparsity(self, embedding: np.ndarray, text: str) -> float:
        neighbors = self._neighbor_vectors(text)
        if not neighbors:
            raise KeyError(text)
        distances = [float(np.linalg.norm(embedding - neighbor)) for neighbor in neighbors]
        return float(np.mean(distances))

    def _neighbor_vectors(self, text: str) -> list[np.ndarray]:
        neighbors = []
        if self.model is None:
            return neighbors
        token_key = text.replace(" ", "_")
        try:
            similar = self.model.similar_by_word(token_key, topn=self.k)
        except Exception:
            pieces = [piece for piece in re.split(r"\s+", text) if piece in self.model]
            if not pieces:
                return neighbors
            similar = []
            for piece in pieces[:1]:
                similar.extend(self.model.similar_by_word(piece, topn=self.k))
        for candidate, _ in similar[: self.k]:
            if candidate in self.model:
                neighbors.append(np.asarray(self.model[candidate], dtype=np.float64))
        return neighbors

    def _normalize_density(self, sparsity: float) -> float:
        clipped = min(max(sparsity, self.min_sparsity), self.max_sparsity)
        density = 1.0 - (clipped - self.min_sparsity) / max(self.max_sparsity - self.min_sparsity, 1e-9)
        return float(min(max(density, 0.0), 1.0))

    def _default_density_for_label(self, label: str) -> float:
        if label in {"PERSON", "EMAIL", "PHONE"}:
            return 0.15
        if label in {"DISEASE", "DRUG"}:
            return 0.25
        if label in {"DATE", "AGE"}:
            return 0.45
        return 0.35


class BudgetAllocator:
    def __init__(
        self,
        epsilon_doc: float,
        lambda_smooth: float = 0.1,
        min_epsilon: float = 0.05,
    ) -> None:
        self.epsilon_doc = epsilon_doc
        self.lambda_smooth = lambda_smooth
        self.min_epsilon = min_epsilon

    def allocate(self, entities: list[ExtractedEntity]) -> list[ExtractedEntity]:
        if not entities:
            return entities

        if self.epsilon_doc <= 0:
            raise ValueError("epsilon_doc must be positive for DenPAD-L.")

        if self.min_epsilon * len(entities) >= self.epsilon_doc:
            base = self.epsilon_doc / len(entities)
            for entity in entities:
                entity.epsilon = base
            return entities

        raw_weights = self._raw_weights(entities)
        weight_sum = sum(raw_weights) or float(len(entities))
        remaining_budget = self.epsilon_doc - self.min_epsilon * len(entities)
        for entity, weight in zip(entities, raw_weights):
            entity.epsilon = self.min_epsilon + remaining_budget * (weight / weight_sum)
        return entities

    def allocate_query(self, entities: list[ExtractedEntity], epsilon_query: Optional[float] = None) -> list[ExtractedEntity]:
        original = self.epsilon_doc
        if epsilon_query is not None:
            self.epsilon_doc = epsilon_query
        try:
            return self.allocate(entities)
        finally:
            self.epsilon_doc = original

    def _raw_weights(self, entities: list[ExtractedEntity]) -> list[float]:
        return [(entity.density if entity.density is not None else 0.3) + self.lambda_smooth for entity in entities]


class SemanticReranker:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self.model = None
        self.embedding_cache: dict[str, np.ndarray] = {}
        if SentenceTransformer is None:
            LOGGER.warning("sentence-transformers is unavailable. DenPAD-L will fall back to embedding or lexical reranking.")
            return
        try:
            self.model = SentenceTransformer(model_name)
        except Exception as exc:
            LOGGER.warning("Failed to load semantic reranker %s: %s", model_name, exc)

    def similarity(self, left: str, right: str) -> float:
        if self.model is None:
            return SequenceMatcher(None, left.lower(), right.lower()).ratio()
        left_emb = self._encode(left)
        right_emb = self._encode(right)
        denom = np.linalg.norm(left_emb) * np.linalg.norm(right_emb)
        if denom <= 0:
            return 0.0
        return float(np.dot(left_emb, right_emb) / denom)

    def _encode(self, text: str) -> np.ndarray:
        normalized = normalize_entity_text(text).lower()
        if normalized in self.embedding_cache:
            return self.embedding_cache[normalized]
        embedding = self.model.encode(normalized, convert_to_numpy=True, normalize_embeddings=False)
        arr = np.asarray(embedding, dtype=np.float64)
        self.embedding_cache[normalized] = arr
        return arr


class CandidateGenerator:
    def __init__(
        self,
        density_scorer: DensityScorer,
        resource_registry: Optional[ResourceRegistry] = None,
        top_k: int = 20,
        semantic_reranker: Optional[SemanticReranker] = None,
        medical_typer: Optional[MedicalTyper] = None,
        llm_completion: Optional[CandidateLLMCompletion] = None,
    ) -> None:
        self.density_scorer = density_scorer
        self.resource_registry = resource_registry or ResourceRegistry()
        self.top_k = top_k
        self.semantic_reranker = semantic_reranker or SemanticReranker()
        self.medical_typer = medical_typer or MedicalTyper(resource_registry=self.resource_registry)
        self.llm_completion = llm_completion
        self.typed_index = TypedCandidateIndex(
            resource_registry=self.resource_registry,
            semantic_reranker=self.semantic_reranker,
            density_scorer=self.density_scorer,
            top_k=self.top_k,
            typer=self.medical_typer,
        )

    def generate(self, entity: ExtractedEntity) -> list[str]:
        if entity.category == "numeric":
            return self._generate_numeric_candidates(entity)
        if entity.category == "structured":
            return self._generate_structured_candidates(entity)
        return self._generate_categorical_candidates(entity)

    def _generate_categorical_candidates(self, entity: ExtractedEntity) -> list[str]:
        normalized = entity.normalized_text
        candidates = [normalized]
        source_map: dict[str, str] = {normalize_entity_text(normalized): "original"}
        level_map: dict[str, str] = {normalize_entity_text(normalized): "original"}
        resource_record = self.resource_registry.find_record(entity.label, normalized)
        if entity.label in {"DISEASE", "DRUG"} and self.llm_completion is not None:
            completion = self.llm_completion.complete(normalized, entity.label)
            for candidate in completion.generalized:
                normalized_candidate = normalize_entity_text(candidate)
                if normalized_candidate:
                    candidates.append(normalized_candidate)
                    source_map.setdefault(normalized_candidate, "llm_completion")
                    level_map[normalized_candidate] = self._classify_llm_candidate_level(
                        entity,
                        normalized_candidate,
                        preferred_level="generalized",
                    )
            for candidate in completion.safe_related:
                normalized_candidate = normalize_entity_text(candidate)
                if normalized_candidate:
                    candidates.append(normalized_candidate)
                    source_map.setdefault(normalized_candidate, "llm_completion")
                    level_map.setdefault(
                        normalized_candidate,
                        self._classify_llm_candidate_level(
                            entity,
                            normalized_candidate,
                            preferred_level="related",
                        ),
                    )
        resource_candidates = self._resource_pool_candidates(entity)
        candidates.extend(resource_candidates)
        trace = entity.candidate_trace or {}
        trace_sources = trace.get("candidate_sources", {})
        trace_levels = trace.get("candidate_layers", {})
        for candidate in resource_candidates:
            normalized_candidate = normalize_entity_text(candidate)
            if normalized_candidate:
                source_map.setdefault(normalized_candidate, trace_sources.get(normalized_candidate, "record"))
                level_map.setdefault(normalized_candidate, trace_levels.get(normalized_candidate, "global"))
        should_expand_distributional = (
            entity.label in {"DISEASE", "DRUG"}
            and len(resource_candidates) < max(4, self.top_k // 3)
            and (
                resource_record is None
                or (
                    not getattr(resource_record, "generalized", ())
                    and len(resource_candidates) < 3
                )
            )
        )
        if should_expand_distributional:
            distributional_candidates = self._distributional_candidates(entity)
            candidates.extend(distributional_candidates)
            for candidate in distributional_candidates:
                normalized_candidate = normalize_entity_text(candidate)
                if normalized_candidate:
                    source_map.setdefault(normalized_candidate, "neighbor")
                    level_map.setdefault(normalized_candidate, "neighbor")
        if not resource_candidates:
            medical_fallback = self._medical_fallback_candidates(entity)
            candidates.extend(medical_fallback)
            for candidate in medical_fallback:
                normalized_candidate = normalize_entity_text(candidate)
                if normalized_candidate:
                    source_map.setdefault(normalized_candidate, "fallback")
                    level_map.setdefault(normalized_candidate, "generalized" if entity.label in {"DISEASE", "DRUG"} else "global")
            default_fallback = DEFAULT_FALLBACK_CANDIDATES.get(entity.label, [])
            candidates.extend(default_fallback)
            for candidate in default_fallback:
                normalized_candidate = normalize_entity_text(candidate)
                if normalized_candidate:
                    source_map.setdefault(normalized_candidate, "global")
                    level_map.setdefault(normalized_candidate, "global")
        elif entity.label not in {"DISEASE", "DRUG"}:
            default_fallback = DEFAULT_FALLBACK_CANDIDATES.get(entity.label, [])
            candidates.extend(default_fallback)
            for candidate in default_fallback:
                normalized_candidate = normalize_entity_text(candidate)
                if normalized_candidate:
                    source_map.setdefault(normalized_candidate, "global")
                    level_map.setdefault(normalized_candidate, "global")
        filtered = [
            candidate
            for candidate in candidates
            if self._is_candidate_compatible(
                entity,
                candidate,
                candidate_source=source_map.get(normalize_entity_text(candidate)),
                candidate_level=level_map.get(normalize_entity_text(candidate)),
            )
        ]
        deduped = self._dedupe_candidates(filtered, normalized)
        ranked = self._rank_candidates(entity, deduped)
        if entity.label in {"DISEASE", "DRUG"}:
            ranked = self._apply_medical_guardrails(entity, ranked)
            ranked = self._ensure_generalized_floor(entity, ranked, minimum=2)
        self._attach_candidate_sources(
            entity,
            ranked,
            source_map,
            default_source="global",
            level_map=level_map,
            default_level="global",
        )
        return ranked

    def _generate_numeric_candidates(self, entity: ExtractedEntity) -> list[str]:
        text = entity.normalized_text
        try:
            value = float(text)
        except ValueError:
            self._attach_candidate_sources(
                entity,
                [text],
                {normalize_entity_text(text): "original"},
                default_source="numeric",
            )
            return [text]

        candidates = [text]
        source_map: dict[str, str] = {normalize_entity_text(text): "original"}
        if entity.label == "AGE":
            for delta in (-10, -5, -2, 2, 5, 10):
                candidate = min(max(int(round(value + delta)), 0), 120)
                candidates.append(str(candidate))
                source_map.setdefault(normalize_entity_text(str(candidate)), "numeric")
        else:
            for delta in (-3, -2, -1, 1, 2, 3):
                candidate = value + delta
                if text.isdigit():
                    formatted = str(int(round(candidate)))
                    candidates.append(formatted)
                    source_map.setdefault(normalize_entity_text(formatted), "numeric")
                else:
                    formatted = f"{candidate:.1f}"
                    candidates.append(formatted)
                    source_map.setdefault(normalize_entity_text(formatted), "numeric")
        deduped = self._dedupe_candidates(candidates, text)
        self._attach_candidate_sources(entity, deduped, source_map, default_source="numeric")
        return deduped

    def _generate_structured_candidates(self, entity: ExtractedEntity) -> list[str]:
        text = entity.normalized_text
        if entity.label == "EMAIL":
            local, _, domain = text.partition("@")
            domain = domain or "example.com"
            candidates = [text]
            source_map: dict[str, str] = {normalize_entity_text(text): "original"}
            for prefix in ("user", "contact", "member", "patient"):
                candidate = f"{prefix}{len(local)}@{domain}"
                candidates.append(candidate)
                source_map.setdefault(normalize_entity_text(candidate), "structured")
            deduped = self._dedupe_candidates(candidates, text)
            self._attach_candidate_sources(entity, deduped, source_map, default_source="structured")
            return deduped

        digits = re.sub(r"\D", "", text)
        candidates = [text]
        source_map = {normalize_entity_text(text): "original"}
        if digits:
            for suffix in ("1234", "5678", "2468", "1357"):
                new_digits = (digits[:-4] + suffix) if len(digits) >= 4 else suffix
                candidates.append(new_digits)
                source_map.setdefault(normalize_entity_text(new_digits), "structured")
        deduped = self._dedupe_candidates(candidates, text)
        self._attach_candidate_sources(entity, deduped, source_map, default_source="structured")
        return deduped

    def _attach_candidate_sources(
        self,
        entity: ExtractedEntity,
        candidates: list[str],
        source_map: dict[str, str],
        default_source: str,
        level_map: Optional[dict[str, str]] = None,
        default_level: str = "unknown",
    ) -> None:
        trace = dict(entity.candidate_trace or {})
        existing_sources = {
            normalize_entity_text(key): value
            for key, value in (trace.get("candidate_sources") or {}).items()
            if normalize_entity_text(key)
        }
        existing_levels = {
            normalize_entity_text(key): value
            for key, value in (trace.get("candidate_levels") or trace.get("candidate_layers") or {}).items()
            if normalize_entity_text(key)
        }
        merged_sources = dict(existing_sources)
        merged_levels = dict(existing_levels)
        for candidate in candidates:
            normalized_candidate = normalize_entity_text(candidate)
            if not normalized_candidate:
                continue
            if normalized_candidate == normalize_entity_text(entity.normalized_text):
                merged_sources[normalized_candidate] = "original"
                merged_levels[normalized_candidate] = "original"
            else:
                merged_sources[normalized_candidate] = source_map.get(normalized_candidate, default_source)
                if level_map is not None:
                    merged_levels[normalized_candidate] = level_map.get(normalized_candidate, default_level)
                else:
                    merged_levels.setdefault(normalized_candidate, default_level)
        trace["candidate_sources"] = merged_sources
        trace["candidate_levels"] = merged_levels
        trace["filtered_candidates"] = [normalize_entity_text(candidate) for candidate in candidates]
        entity.candidate_trace = trace

    def _dedupe_candidates(self, candidates: list[str], original: str) -> list[str]:
        seen = set()
        cleaned = []
        for candidate in candidates:
            candidate = normalize_entity_text(str(candidate))
            if not candidate or candidate in seen:
                continue
            seen.add(candidate)
            cleaned.append(candidate)
        if original not in seen:
            cleaned.insert(0, original)
        return cleaned[: self.top_k]

    def _is_candidate_compatible(
        self,
        entity: ExtractedEntity,
        candidate: str,
        candidate_source: Optional[str] = None,
        candidate_level: Optional[str] = None,
    ) -> bool:
        candidate = normalize_entity_text(candidate)
        if not candidate:
            return False
        if not is_ascii_text(candidate):
            return False
        if candidate.lower() in GENERIC_NON_SENSITIVE_TERMS:
            return False
        if re.search(r"[^A-Za-z0-9\s@._+\-]", candidate):
            return False

        original_tokens = entity.normalized_text.split()
        candidate_tokens = candidate.split()
        if entity.category == "categorical":
            if abs(len(candidate_tokens) - len(original_tokens)) > 1:
                return False
            if entity.label == "DISEASE":
                if any(token.lower() in BAD_DISEASE_CANDIDATE_SUFFIXES for token in candidate_tokens):
                    return False
                if candidate.lower().endswith(" infections") or candidate.lower().startswith("resistant "):
                    return False
            if entity.label in {"PERSON", "ORG", "GPE", "LOC"} and candidate.lower() == candidate and len(candidate_tokens) <= 1:
                return False
            if entity.label == "DISEASE" and not looks_like_disease_text(candidate, self.resource_registry):
                if not self._is_llm_generalized_candidate(entity, candidate, candidate_source, candidate_level):
                    return False
            if entity.label == "DRUG" and not looks_like_drug_text(candidate, self.resource_registry):
                if not self._is_llm_generalized_candidate(entity, candidate, candidate_source, candidate_level):
                    return False
            if entity.label in {"DISEASE", "DRUG", "PERSON", "ORG", "GPE", "LOC"}:
                if not self.medical_typer.candidate_matches_label(candidate, entity.label):
                    if not self._is_llm_generalized_candidate(entity, candidate, candidate_source, candidate_level):
                        return False
            if entity.label in {"DISEASE", "DRUG"}:
                if not self.medical_typer.candidate_group_matches(entity.normalized_text, candidate, entity.label):
                    if not self._is_llm_generalized_candidate(entity, candidate, candidate_source, candidate_level):
                        return False
        return True

    def _is_llm_generalized_candidate(
        self,
        entity: ExtractedEntity,
        candidate: str,
        candidate_source: Optional[str],
        candidate_level: Optional[str],
    ) -> bool:
        if candidate_source != "llm_completion":
            return False
        if candidate_level not in {"generalized", "related"}:
            return False
        return self._passes_llm_generalized_guard(entity, candidate)

    def _passes_llm_generalized_guard(self, entity: ExtractedEntity, candidate: str) -> bool:
        lowered = candidate.lower()
        token_count = len(lowered.split())
        if token_count == 0 or token_count > 4:
            return False
        if entity.label == "DISEASE":
            allowed_last_tokens = {
                "condition",
                "disorder",
                "disease",
                "infection",
                "syndrome",
                "issue",
            }
            disallowed_specific_tokens = {
                "eczema",
                "psoriasis",
                "arthritis",
                "discitis",
                "crohns",
                "crohn",
                "hepatitis",
                "gonorrhea",
                "alopecia",
                "graves",
                "hyperthyroid",
                "hyperthyroidism",
                "cancer",
                "cirrhosis",
                "anxiety",
                "zoster",
                "shingles",
            }
            tokens = [token for token in re.split(r"\s+", lowered) if token]
            if not tokens or tokens[-1] not in allowed_last_tokens:
                return False
            if any(token in disallowed_specific_tokens for token in tokens[:-1]):
                return False
            if token_count == 1:
                return False
            return True
        if entity.label == "DRUG":
            allowed_last_tokens = {
                "medication",
                "medicine",
                "drug",
                "therapy",
                "treatment",
            }
            disallowed_specific_tokens = {
                "ibuprofen",
                "prednisone",
                "euthyrox",
                "trileptal",
                "acetaminophen",
                "paracetamol",
                "levothyroxine",
                "thyroxine",
                "pill",
                "tablet",
                "hormone",
            }
            tokens = [token for token in re.split(r"\s+", lowered) if token]
            if not tokens or tokens[-1] not in allowed_last_tokens:
                return False
            if any(token in disallowed_specific_tokens for token in tokens[:-1]):
                return False
            if token_count == 1:
                return False
            return True
        return False

    def _classify_llm_candidate_level(
        self,
        entity: ExtractedEntity,
        candidate: str,
        preferred_level: str,
    ) -> str:
        if self._passes_llm_generalized_guard(entity, candidate):
            return "generalized"
        return preferred_level

    def _resource_pool_candidates(self, entity: ExtractedEntity) -> list[str]:
        candidates, trace = self.typed_index.merge_and_filter(entity, return_trace=True)
        entity.candidate_trace = {
            "category": trace.category,
            "original": trace.original,
            "merged_before_filter": trace.merged_before_filter,
            "filtered_candidates": trace.filtered_candidates,
            "candidate_layers": trace.candidate_layers,
            "candidate_sources": trace.candidate_sources,
            "layer_counts": trace.layer_counts,
            "record_hit": trace.record_hit,
        }
        return candidates

    def _distributional_candidates(self, entity: ExtractedEntity) -> list[str]:
        normalized = entity.normalized_text
        try:
            if self.density_scorer.model is None:
                raise KeyError(normalized)
            token_key = normalized.lower().replace(" ", "_")
            similar = self.density_scorer.model.similar_by_word(token_key, topn=self.top_k)
            return [
                candidate.replace("_", " ")
                for candidate, _ in similar
                if self._is_candidate_compatible(entity, candidate.replace("_", " "))
            ]
        except Exception:
            return []

    def _rank_candidates_by_similarity(self, original: str, candidates: list[str], top_k: int) -> list[str]:
        if not candidates:
            return []
        scored = []
        for candidate in candidates:
            score = self._candidate_similarity(original, candidate)
            scored.append((candidate, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return [candidate for candidate, _ in scored[:top_k]]

    def _rank_candidates(self, entity: ExtractedEntity, candidates: list[str]) -> list[str]:
        if not candidates:
            return []
        original = entity.normalized_text
        trace = entity.candidate_trace or {}
        candidate_sources = trace.get("candidate_sources", {}) or {}
        scored = []
        for candidate in candidates:
            semantic_score = self._candidate_similarity(original, candidate)
            lexical_score = SequenceMatcher(None, original.lower(), candidate.lower()).ratio()
            overlap_score = self._token_overlap(original, candidate)
            tag_bonus = self._tag_bonus(entity.label, original, candidate)
            score = 0.48 * semantic_score + 0.08 * lexical_score + 0.06 * overlap_score + 0.08 * tag_bonus
            level = self._candidate_level(entity, candidate)
            risk = self._entity_risk(entity)
            normalized_candidate = normalize_entity_text(candidate)
            source = candidate_sources.get(normalized_candidate, candidate_sources.get(candidate, "global"))
            score += self._candidate_level_prior(entity, candidate, level, risk)
            score += self._entity_specific_candidate_adjustment(entity, candidate)
            if source == "llm_completion" and level == "generalized" and entity.label in {"DISEASE", "DRUG"}:
                score += 0.22 if entity.label == "DISEASE" else 0.16
            if candidate != original and lexical_score >= 0.92:
                score -= 0.30 if entity.label in {"DISEASE", "DRUG"} else 0.12
            scored.append((candidate, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        ordered = [candidate for candidate, _ in scored]
        if original in ordered:
            ordered.remove(original)
        return [original, *ordered[: max(self.top_k - 1, 0)]]

    def _ensure_generalized_floor(self, entity: ExtractedEntity, candidates: list[str], minimum: int = 2) -> list[str]:
        if entity.label not in {"DISEASE", "DRUG"}:
            return candidates
        generalized = [candidate for candidate in candidates if self._candidate_level(entity, candidate) == "generalized"]
        if len(generalized) >= minimum:
            return self._retain_attack_candidates(entity, candidates)
        original = entity.normalized_text
        completed = list(candidates)
        completion = self.llm_completion.complete(original, entity.label) if self.llm_completion is not None else None
        if completion is not None:
            for candidate in completion.generalized:
                normalized_candidate = normalize_entity_text(candidate)
                if normalized_candidate and normalized_candidate not in completed and self._is_candidate_compatible(entity, normalized_candidate):
                    completed.append(normalized_candidate)
                    trace = dict(entity.candidate_trace or {})
                    candidate_sources = dict(trace.get("candidate_sources") or {})
                    candidate_levels = dict(trace.get("candidate_levels") or trace.get("candidate_layers") or {})
                    candidate_sources[normalized_candidate] = "llm_completion"
                    candidate_levels[normalized_candidate] = "generalized"
                    trace["candidate_sources"] = candidate_sources
                    trace["candidate_levels"] = candidate_levels
                    entity.candidate_trace = trace
        completed = self._dedupe_candidates(completed, original)
        generalized = [candidate for candidate in completed if self._candidate_level(entity, candidate) == "generalized"]
        if len(generalized) < minimum:
            return self._retain_attack_candidates(entity, completed)
        non_generalized = [candidate for candidate in completed if candidate != original and self._candidate_level(entity, candidate) != "generalized"]
        capped_tail = non_generalized[: max(self.top_k - (1 + len(generalized)), 0)]
        narrowed = self._dedupe_candidates([original, *generalized, *capped_tail], original)
        return self._retain_attack_candidates(entity, narrowed)

    def _retain_attack_candidates(self, entity: ExtractedEntity, candidates: list[str]) -> list[str]:
        if entity.label not in {"DISEASE", "DRUG"}:
            return candidates
        original = entity.normalized_text
        generalized = [candidate for candidate in candidates if self._candidate_level(entity, candidate) == "generalized"]
        if not generalized:
            return self._dedupe_candidates(candidates, original)
        ordered_generalized = list(dict.fromkeys(generalized))
        limit = max(self.top_k - 1, 1)
        return self._dedupe_candidates([original, *ordered_generalized[:limit]], original)

    def _candidate_similarity(self, original: str, candidate: str) -> float:
        if self.semantic_reranker.model is not None:
            return self.semantic_reranker.similarity(original, candidate)
        try:
            original_emb = self.density_scorer._embed_text(original.lower())
            candidate_emb = self.density_scorer._embed_text(candidate.lower())
        except Exception:
            return SequenceMatcher(None, original.lower(), candidate.lower()).ratio()
        denom = np.linalg.norm(original_emb) * np.linalg.norm(candidate_emb)
        if denom <= 0:
            return 0.0
        return float(np.dot(original_emb, candidate_emb) / denom)

    def _token_overlap(self, original: str, candidate: str) -> float:
        original_tokens = {token for token in re.split(r"\s+", original.lower()) if token}
        candidate_tokens = {token for token in re.split(r"\s+", candidate.lower()) if token}
        if not original_tokens or not candidate_tokens:
            return 0.0
        return len(original_tokens & candidate_tokens) / len(original_tokens | candidate_tokens)

    def _tag_bonus(self, label: str, original: str, candidate: str) -> float:
        category = "GPE" if label == "LOC" else label
        original_record = self.resource_registry.find_record(category, original)
        candidate_record = self.resource_registry.find_record(category, candidate)
        if original_record is None or candidate_record is None:
            return 0.0
        original_tags = set(original_record.normalized_tags)
        candidate_tags = set(candidate_record.normalized_tags)
        if not original_tags or not candidate_tags:
            return 0.0
        return 1.0 if original_tags.intersection(candidate_tags) else -0.2

    def _entity_specific_candidate_adjustment(self, entity: ExtractedEntity, candidate: str) -> float:
        if entity.label != "DISEASE":
            return 0.0
        original = entity.normalized_text.lower()
        candidate_l = candidate.lower()
        if "psoriasis" in original:
            if any(token in candidate_l for token in ("plaque", "scalp", "severe")):
                return -0.20
            if candidate_l in {"skin condition", "inflammatory skin disorder"}:
                return 0.10
        if "gonorr" in original:
            if candidate_l == "sexually transmitted infection":
                return 0.22
            if candidate_l == "bacterial infection":
                return 0.10
            if candidate_l != original and "gonorr" in candidate_l:
                return -0.22
        if "alopecia" in original:
            if candidate_l == "hair loss condition":
                return 0.16
            if candidate_l == "autoimmune hair disorder":
                return 0.08
            if candidate_l == "hair loss disorder":
                return 0.06
        if "anxiety" in original:
            if candidate_l == "mental health condition":
                return 0.16
            if candidate_l == "anxiety disorder":
                return 0.10
            if candidate_l == "panic disorder":
                return -0.40
        return 0.0

    def _entity_risk(self, entity: ExtractedEntity) -> float:
        density = entity.density if entity.density is not None else 0.35
        density_risk = max(0.0, min(1.0, 1.0 - density))
        type_weight = {
            "DISEASE": 1.00,
            "DRUG": 0.95,
            "PERSON": 0.85,
            "EMAIL": 0.90,
            "PHONE": 0.90,
            "AGE": 0.75,
            "ORG": 0.55,
            "GPE": 0.50,
            "LOC": 0.50,
            "DATE": 0.45,
        }.get(entity.label, 0.40)
        return max(0.0, min(1.0, (density_risk ** 1.35) * type_weight))

    def _candidate_level_prior(
        self,
        entity: ExtractedEntity,
        candidate: str,
        level: str,
        risk: float,
    ) -> float:
        # High-risk entities should strongly prefer generalized candidates and
        # only weakly preserve the original token as a fallback.
        if entity.label not in {"DISEASE", "DRUG"}:
            if level == "original":
                return -0.03 * risk
            if level == "generalized":
                return 0.04 * risk
            return 0.0

        if level == "original":
            return -0.24 - 0.52 * risk
        if level == "generalized":
            return 0.32 + 0.48 * risk
        if level == "related":
            return 0.02 + 0.08 * risk
        if level in {"alias", "canonical"}:
            return -0.06 + 0.01 * (1.0 - risk)
        if level == "global":
            return -0.12 + 0.02 * risk
        return 0.0

    def _apply_medical_guardrails(self, entity: ExtractedEntity, candidates: list[str]) -> list[str]:
        if not candidates:
            return [entity.normalized_text]
        original = entity.normalized_text
        original_record = self.resource_registry.find_record(entity.label, original)
        guarded = [original]
        min_similarity = 0.68 if entity.label == "DISEASE" else 0.72
        for candidate in candidates:
            if candidate == original:
                continue
            candidate_record = self.resource_registry.find_record(entity.label, candidate)
            is_generalized = self.resource_registry.is_generalized_candidate(entity.label, original, candidate)
            if not self.medical_typer.candidate_group_matches(original, candidate, entity.label):
                continue
            if original_record is not None:
                original_tags = set(original_record.normalized_tags)
                candidate_tags = set(candidate_record.normalized_tags) if candidate_record is not None else set()
                if original_tags and candidate_tags and not original_tags.intersection(candidate_tags):
                    continue
            semantic = self._candidate_similarity(original, candidate)
            lexical = SequenceMatcher(None, original.lower(), candidate.lower()).ratio()
            tag_bonus = self._tag_bonus(entity.label, original, candidate)
            overlap = self._token_overlap(original, candidate)
            local_min_similarity = 0.45 if is_generalized and entity.label == "DISEASE" else min_similarity
            if original_record is not None and candidate_record is None and semantic < local_min_similarity and overlap < 0.5:
                continue
            if semantic < local_min_similarity and lexical < 0.35 and tag_bonus <= 0:
                continue
            guarded.append(candidate)
        if len(guarded) == 1:
            return guarded
        return self._dedupe_candidates(guarded, original)

    def _medical_fallback_candidates(self, entity: ExtractedEntity) -> list[str]:
        lowered = entity.normalized_text.lower()
        if entity.label == "DISEASE":
            if "cancer" in lowered:
                return ["breast cancer", "lung cancer", "colon cancer", "bladder cancer"]
            if "jaundice" in lowered:
                return ["neonatal jaundice", "hepatitis", "liver disease"]
            if "lung" in lowered or "ild" in lowered:
                return ["pulmonary fibrosis", "chronic lung disease", "interstitial pneumonia"]
            if "h pylori" in lowered or "hpylori" in lowered:
                return ["gastritis", "peptic ulcer disease", "stomach infection"]
            if "alopecia" in lowered:
                return ["androgenetic alopecia", "hair loss disorder", "hair loss condition"]
            if "gonorr" in lowered:
                return ["sexually transmitted infection", "bacterial infection"]
            return []
        if entity.label == "DRUG":
            if "zoloft" in lowered:
                return ["sertraline", "fluoxetine", "escitalopram"]
            if "votrient" in lowered:
                return ["pazopanib", "sunitinib", "sorafenib"]
            if "amoxi" in lowered:
                return ["amoxicillin", "azithromycin", "doxycycline"]
            return []
        return []

    def should_force_generalized_mode(self, entity: ExtractedEntity, candidates: list[str]) -> bool:
        if entity.label not in {"DISEASE", "DRUG"}:
            return False
        registry = getattr(self, "resource_registry", None)
        if registry is None:
            return False
        category = "GPE" if entity.label == "LOC" else entity.label
        generalized = [
            candidate
            for candidate in candidates
            if self._candidate_level(entity, candidate) == "generalized"
        ]
        if not generalized:
            return False
        # For disease entities, once typed generalized candidates exist, always
        # enter generalized-dominant mode to prevent verbatim disease leakage.
        if entity.label == "DISEASE":
            return True
        risk = self._entity_risk(entity)
        if risk >= 0.78:
            return True
        lowered = entity.normalized_text.lower()
        trigger_patterns = (
            "gonorr",
            "alopecia",
            "thyroid cancer",
            "graves",
            "hyperthy",
            "trigeminal neuralgia",
            "deep vein thrombosis",
            " dvt",
            "euthyrox",
            "levothyroxine",
            "trileptal",
            "oxcarbazepine",
            "ibuprofen",
            "prednisone",
        )
        return any(pattern in lowered for pattern in trigger_patterns)

    def _candidate_level(self, entity: ExtractedEntity, candidate: str) -> str:
        trace = entity.candidate_trace or {}
        normalized_candidate = normalize_entity_text(candidate)
        levels = trace.get("candidate_levels") or trace.get("candidate_layers") or {}
        if normalized_candidate in levels:
            return levels[normalized_candidate]
        registry = getattr(self, "resource_registry", None)
        if registry is None:
            return "unknown"
        category = "GPE" if entity.label == "LOC" else entity.label
        return registry.candidate_level(category, entity.normalized_text, candidate)


class MechanismFactory:
    def __init__(
        self,
        candidate_generator: CandidateGenerator,
        density_scorer: DensityScorer,
        min_candidate_score: float = 0.25,
    ) -> None:
        self.candidate_generator = candidate_generator
        self.density_scorer = density_scorer
        self.min_candidate_score = min_candidate_score

    def perturb(self, entity: ExtractedEntity) -> PerturbationRecord:
        candidates = self.candidate_generator.generate(entity)
        if entity.category == "numeric":
            return self._perturb_numeric(entity, candidates)
        if entity.category == "structured":
            return self._perturb_structured(entity, candidates)
        return self._perturb_categorical(entity, candidates)

    def _perturb_categorical(self, entity: ExtractedEntity, candidates: list[str]) -> PerturbationRecord:
        scores = [self._utility_score(entity, candidate) for candidate in candidates]
        if entity.label in {"DISEASE", "DRUG"}:
            risk = self.candidate_generator._entity_risk(entity)
            viable = [
                candidate
                for candidate, score in zip(candidates, scores)
                if candidate == entity.normalized_text or score >= self.min_candidate_score
            ]
            if viable:
                candidates = viable
                scores = [self._utility_score(entity, candidate) for candidate in candidates]
            if getattr(self.candidate_generator, "resource_registry", None) is not None:
                generalized = [
                    candidate
                    for candidate in candidates
                    if self.candidate_generator._candidate_level(entity, candidate) == "generalized"
                ]
                if generalized and risk >= 0.55:
                    protected = []
                    if entity.label == "DISEASE":
                        for candidate in candidates:
                            level = self.candidate_generator._candidate_level(entity, candidate)
                            if candidate == entity.normalized_text or level == "generalized":
                                protected.append(candidate)
                    else:
                        for candidate in candidates:
                            level = self.candidate_generator._candidate_level(entity, candidate)
                            if candidate == entity.normalized_text:
                                protected.append(candidate)
                            elif level in {"generalized", "related"}:
                                protected.append(candidate)
                    if protected:
                        candidates = self.candidate_generator._dedupe_candidates(protected, entity.normalized_text)
                        scores = [self._utility_score(entity, candidate) for candidate in candidates]
                if self.candidate_generator.should_force_generalized_mode(entity, candidates):
                    generalized = [
                        candidate
                        for candidate in candidates
                        if self.candidate_generator._candidate_level(entity, candidate) == "generalized"
                    ]
                    if generalized:
                        protected = [*generalized]
                        if entity.normalized_text not in protected:
                            protected.append(entity.normalized_text)
                        candidates = self.candidate_generator._dedupe_candidates(protected, entity.normalized_text)
                        boosted_scores = []
                        for candidate in candidates:
                            score = self._utility_score(entity, candidate)
                            level = self.candidate_generator._candidate_level(entity, candidate)
                            if candidate == entity.normalized_text:
                                score -= 2.35 if entity.label == "DISEASE" else 1.60
                            elif level == "generalized":
                                score += 0.95 if entity.label == "DISEASE" else 0.55
                            boosted_scores.append(score)
                        scores = boosted_scores
        epsilon = entity.epsilon or 0.1
        probs = self._exponential_probabilities(scores, epsilon=epsilon)
        sample = random.choices(candidates, weights=probs, k=1)[0]
        candidate_levels = self._candidate_level_map(entity, candidates)
        candidate_sources = self._candidate_source_map(entity, candidates)
        return PerturbationRecord(
            original_text=entity.text,
            perturbed_text=sample,
            label=entity.label,
            category=entity.category,
            epsilon=epsilon,
            density=entity.density or 0.0,
            mechanism="categorical_exponential",
            start_char=entity.start_char,
            end_char=entity.end_char,
            candidates=candidates,
            candidate_levels=candidate_levels,
            candidate_sources=candidate_sources,
            candidate_scores=scores,
            candidate_probabilities=probs,
            selected_level=candidate_levels.get(sample, "unknown"),
            candidate_trace=entity.candidate_trace,
        )

    def _perturb_numeric(self, entity: ExtractedEntity, candidates: list[str]) -> PerturbationRecord:
        scores = [self._numeric_utility(entity.normalized_text, candidate) for candidate in candidates]
        epsilon = entity.epsilon or 0.1
        probs = self._exponential_probabilities(scores, epsilon=epsilon)
        sample = random.choices(candidates, weights=probs, k=1)[0]
        candidate_levels = self._candidate_level_map(entity, candidates)
        candidate_sources = self._candidate_source_map(entity, candidates)
        return PerturbationRecord(
            original_text=entity.text,
            perturbed_text=sample,
            label=entity.label,
            category=entity.category,
            epsilon=epsilon,
            density=entity.density or 0.0,
            mechanism="numeric_exponential",
            start_char=entity.start_char,
            end_char=entity.end_char,
            candidates=candidates,
            candidate_levels=candidate_levels,
            candidate_sources=candidate_sources,
            candidate_scores=scores,
            candidate_probabilities=probs,
            selected_level=candidate_levels.get(sample, "unknown"),
            candidate_trace=entity.candidate_trace,
        )

    def _perturb_structured(self, entity: ExtractedEntity, candidates: list[str]) -> PerturbationRecord:
        scores = [self._structured_utility(entity.normalized_text, candidate, entity.label) for candidate in candidates]
        epsilon = entity.epsilon or 0.1
        probs = self._exponential_probabilities(scores, epsilon=epsilon)
        sample = random.choices(candidates, weights=probs, k=1)[0]
        candidate_levels = self._candidate_level_map(entity, candidates)
        candidate_sources = self._candidate_source_map(entity, candidates)
        return PerturbationRecord(
            original_text=entity.text,
            perturbed_text=sample,
            label=entity.label,
            category=entity.category,
            epsilon=epsilon,
            density=entity.density or 0.0,
            mechanism="structured_exponential",
            start_char=entity.start_char,
            end_char=entity.end_char,
            candidates=candidates,
            candidate_levels=candidate_levels,
            candidate_sources=candidate_sources,
            candidate_scores=scores,
            candidate_probabilities=probs,
            selected_level=candidate_levels.get(sample, "unknown"),
            candidate_trace=entity.candidate_trace,
        )

    def _utility_score(self, entity: ExtractedEntity, candidate: str) -> float:
        original = entity.normalized_text
        label = entity.label
        category = entity.category
        semantic_sim = self._semantic_similarity(original, candidate)
        surface_sim = SequenceMatcher(None, original.lower(), candidate.lower()).ratio()
        type_bonus = 1.0 if label in DEFAULT_FALLBACK_CANDIDATES or category == "categorical" else 0.5
        overlap = self.candidate_generator._token_overlap(original, candidate)
        risk = self.candidate_generator._entity_risk(entity)
        level = self.candidate_generator._candidate_level(entity, candidate)
        score = 0.68 * semantic_sim + 0.06 * surface_sim + 0.06 * overlap + 0.08 * type_bonus
        score += self.candidate_generator._candidate_level_prior(entity, candidate, level, risk)
        if candidate == original:
            score -= 1.00 if label in {"DISEASE", "DRUG"} else 0.25
        elif surface_sim >= 0.94 or overlap >= 0.90:
            score -= 0.55 if label in {"DISEASE", "DRUG"} else 0.15
        if level == "generalized":
            score += 0.55 if label == "DISEASE" else 0.35
        elif level == "related":
            score += 0.08 if label in {"DISEASE", "DRUG"} else 0.02
        score += self.candidate_generator._entity_specific_candidate_adjustment(entity, candidate)
        return score

    def _numeric_utility(self, original: str, candidate: str) -> float:
        try:
            orig_value = float(original)
            cand_value = float(candidate)
        except ValueError:
            return 0.0
        distance = abs(orig_value - cand_value)
        return -distance

    def _structured_utility(self, original: str, candidate: str, label: str) -> float:
        score = SequenceMatcher(None, original.lower(), candidate.lower()).ratio()
        if label == "EMAIL":
            score += 0.2 if original.split("@")[-1] == candidate.split("@")[-1] else 0.0
        if label == "PHONE":
            score += 0.2 if len(re.sub(r"\D", "", original)) == len(re.sub(r"\D", "", candidate)) else 0.0
        return score

    def _semantic_similarity(self, original: str, candidate: str) -> float:
        reranker = getattr(self.candidate_generator, "semantic_reranker", None)
        if reranker is not None and getattr(reranker, "model", None) is not None:
            return reranker.similarity(original, candidate)
        try:
            original_emb = self.density_scorer._embed_text(original.lower())
            candidate_emb = self.density_scorer._embed_text(candidate.lower())
        except Exception:
            return 0.0
        denom = np.linalg.norm(original_emb) * np.linalg.norm(candidate_emb)
        if denom <= 0:
            return 0.0
        similarity = float(np.dot(original_emb, candidate_emb) / denom)
        return max(min(similarity, 1.0), -1.0)

    def _sample_exponential_mechanism(
        self,
        candidates: list[str],
        scores: list[float],
        epsilon: float,
        delta_u: float = 1.0,
    ) -> str:
        if not candidates:
            raise ValueError("Candidates must be non-empty.")
        if len(candidates) == 1:
            return candidates[0]
        probs = self._exponential_probabilities(scores, epsilon=epsilon, delta_u=delta_u)
        return random.choices(candidates, weights=probs, k=1)[0]

    def _exponential_probabilities(
        self,
        scores: list[float],
        epsilon: float,
        delta_u: float = 1.0,
    ) -> list[float]:
        logits = [epsilon * score / (2.0 * max(delta_u, 1e-9)) for score in scores]
        return safe_softmax(logits)

    def _candidate_level_map(self, entity: ExtractedEntity, candidates: list[str]) -> dict[str, str]:
        return {
            candidate: self.candidate_generator._candidate_level(entity, candidate)
            for candidate in candidates
        }

    def _candidate_source_map(self, entity: ExtractedEntity, candidates: list[str]) -> dict[str, str]:
        trace = entity.candidate_trace or {}
        sources = trace.get("candidate_sources", {}) if isinstance(trace, dict) else {}
        if entity.category == "numeric":
            default_source = "numeric"
        elif entity.category == "structured":
            default_source = "structured"
        else:
            default_source = "global"
        source_map = {}
        normalized_original = normalize_entity_text(entity.normalized_text)
        for candidate in candidates:
            normalized_candidate = normalize_entity_text(candidate)
            if normalized_candidate == normalized_original:
                source_map[candidate] = "original"
                continue
            source_map[candidate] = sources.get(normalized_candidate, sources.get(candidate, default_source))
        return source_map


class DenPADSanitizer:
    def __init__(
        self,
        epsilon_doc: float = 3.0,
        density_backend: str = "word2vec-google-news-300",
        density_k: int = 20,
        candidate_topk: int = 20,
        lambda_smooth: float = 0.1,
        min_epsilon: float = 0.05,
        resources_dir: Optional[str] = None,
        medical_ner_backend: Optional[str] = None,
        medical_typer_config: Optional[str] = None,
        enable_medical_ner: bool = True,
        disable_age_date: bool = False,
        min_candidate_score: float = 0.25,
        candidate_llm_model: Optional[str] = "Qwen/Qwen2.5-3B-Instruct",
        candidate_llm_topk: int = 5,
        seed: int = 42,
    ) -> None:
        seed_everything(seed)
        self.epsilon_doc = epsilon_doc
        self.resource_registry = ResourceRegistry(resources_dir=resources_dir)
        self.disable_age_date = disable_age_date
        self.candidate_llm_model = candidate_llm_model
        self.medical_typer = MedicalTyper(
            resource_registry=self.resource_registry,
            model_name=medical_ner_backend,
            enable_medical_ner=enable_medical_ner,
            config_path=medical_typer_config,
        )
        self.extractor = EntityExtractor(
            resource_registry=self.resource_registry,
            medical_typer=self.medical_typer,
        )
        self.density_scorer = DensityScorer(backend=density_backend, k=density_k)
        self.semantic_reranker = SemanticReranker()
        self.llm_completion = CandidateLLMCompletion(
            model_name=candidate_llm_model,
            top_k=candidate_llm_topk,
        )
        self.budget_allocator = BudgetAllocator(
            epsilon_doc=epsilon_doc,
            lambda_smooth=lambda_smooth,
            min_epsilon=min_epsilon,
        )
        self.candidate_generator = CandidateGenerator(
            density_scorer=self.density_scorer,
            resource_registry=self.resource_registry,
            top_k=candidate_topk,
            semantic_reranker=self.semantic_reranker,
            medical_typer=self.medical_typer,
            llm_completion=self.llm_completion,
        )
        self.mechanism_factory = MechanismFactory(
            candidate_generator=self.candidate_generator,
            density_scorer=self.density_scorer,
            min_candidate_score=min_candidate_score,
        )

    def sanitize_document(self, text: str, metadata: Optional[dict[str, Any]] = None) -> SanitizationResult:
        entities = self.extractor.extract(text)
        entities = self.density_scorer.score_entities(entities)
        entities = self._filter_query_entities(entities)
        entities = self.budget_allocator.allocate(entities)
        perturbations = [self.mechanism_factory.perturb(entity) for entity in entities]
        sanitized_text = apply_perturbations_by_span(text, perturbations)
        result_metadata = dict(metadata or {})
        result_metadata["num_entities"] = len(entities)
        result_metadata["num_perturbed"] = len(perturbations)
        result_metadata["epsilon_doc"] = self.epsilon_doc
        result_metadata["disable_age_date"] = self.disable_age_date
        result_metadata["candidate_llm_model"] = self.candidate_llm_model
        return SanitizationResult(
            original_text=text,
            sanitized_text=sanitized_text,
            entities=entities,
            perturbations=perturbations,
            epsilon_doc=self.epsilon_doc,
            metadata=result_metadata,
        )

    def sanitize_documents(
        self,
        documents: list[Any],
        show_progress: bool = True,
        progress_desc: str = "Applying DenPAD-L to corpus",
    ) -> tuple[list[Any], list[dict[str, Any]]]:
        sanitized_documents = []
        audit_records = []
        iterator = documents
        if show_progress:
            iterator = tqdm(documents, desc=progress_desc)
        for index, document in enumerate(iterator):
            result = self.sanitize_document(document.page_content, metadata=getattr(document, "metadata", None))
            metadata = dict(getattr(document, "metadata", {}) or {})
            metadata["original_page_content"] = document.page_content
            metadata["denpad_epsilon_doc"] = result.epsilon_doc
            metadata["denpad_num_entities"] = len(result.entities)
            metadata["denpad_num_perturbed"] = len(result.perturbations)
            sanitized_documents.append(document.__class__(page_content=result.sanitized_text, metadata=metadata))

            for record in result.perturbations:
                selected_source = record.candidate_sources.get(
                    record.perturbed_text,
                    "original" if record.perturbed_text == record.original_text else (
                        "numeric" if record.category == "numeric" else ("structured" if record.category == "structured" else "global")
                    ),
                )
                audit_records.append(
                    {
                        "doc_index": index,
                        "entity": record.original_text,
                        "replacement": record.perturbed_text,
                        "label": record.label,
                        "category": record.category,
                        "epsilon": record.epsilon,
                        "density": record.density,
                        "mechanism": record.mechanism,
                        "candidates": record.candidates,
                        "candidate_levels": record.candidate_levels,
                        "candidate_sources": record.candidate_sources,
                        "candidate_scores": record.candidate_scores,
                        "candidate_probabilities": record.candidate_probabilities,
                        "selected_level": record.selected_level,
                        "selected_source": selected_source,
                        "candidate_trace": record.candidate_trace,
                    }
                )
        return sanitized_documents, audit_records

    def sanitize_retrieved_docs(
        self,
        docs: list[str],
        query: Optional[str] = None,
    ) -> tuple[list[str], dict[str, Any]]:
        query_entities: list[ExtractedEntity] = []
        for index, text in enumerate(docs):
            query_entities.extend(self.extractor.extract(text, doc_index=index))

        query_entities = self.density_scorer.score_entities(query_entities)
        query_entities = self._filter_query_entities(query_entities)
        query_entities = self.budget_allocator.allocate_query(query_entities, epsilon_query=self.epsilon_doc)

        grouped_entities: dict[int, list[ExtractedEntity]] = {index: [] for index in range(len(docs))}
        for entity in query_entities:
            grouped_entities.setdefault(entity.doc_index, []).append(entity)

        sanitized_docs = list(docs)
        audit_records = []
        total_perturbed = 0
        for index, text in enumerate(docs):
            entities = grouped_entities.get(index, [])
            perturbations = [self.mechanism_factory.perturb(entity) for entity in entities]
            sanitized_docs[index] = apply_perturbations_by_span(text, perturbations)
            total_perturbed += len(perturbations)
            for record in perturbations:
                original_snippet = text[max(record.start_char - 40, 0) : min(record.end_char + 40, len(text))]
                sanitized_snippet = sanitized_docs[index][
                    max(record.start_char - 40, 0) : min(record.start_char + len(record.perturbed_text) + 40, len(sanitized_docs[index]))
                ]
                selected_source = record.candidate_sources.get(
                    record.perturbed_text,
                    "original" if record.perturbed_text == record.original_text else (
                        "numeric" if record.category == "numeric" else ("structured" if record.category == "structured" else "global")
                    ),
                )
                audit_records.append(
                    {
                        "doc_index": index,
                        "entity": record.original_text,
                        "replacement": record.perturbed_text,
                        "label": record.label,
                        "category": record.category,
                        "epsilon": record.epsilon,
                        "density": record.density,
                        "mechanism": record.mechanism,
                        "candidates": record.candidates,
                        "candidate_levels": record.candidate_levels,
                        "candidate_sources": record.candidate_sources,
                        "candidate_scores": record.candidate_scores,
                        "candidate_probabilities": record.candidate_probabilities,
                        "selected_level": record.selected_level,
                        "selected_source": selected_source,
                        "candidate_trace": record.candidate_trace,
                        "original_doc_snippet": original_snippet,
                        "sanitized_doc_snippet": sanitized_snippet,
                    }
                )
        level_counts: dict[str, int] = {}
        source_counts: dict[str, int] = {}
        for record in audit_records:
            level = record.get("selected_level", "unknown")
            level_counts[level] = level_counts.get(level, 0) + 1
            source = record.get("selected_source", "unknown")
            source_counts[source] = source_counts.get(source, 0) + 1
        return sanitized_docs, {
            "query": query,
            "num_docs": len(docs),
            "num_entities": len(query_entities),
            "num_perturbed": total_perturbed,
            "epsilon_doc": self.epsilon_doc,
            "epsilon_query": self.epsilon_doc,
            "disable_age_date": self.disable_age_date,
            "selected_level_counts": level_counts,
            "selected_source_counts": source_counts,
            "resource_summary": self.resource_registry.get_resource_summary(),
            "resource_manifest": self.resource_registry.resource_manifest,
            "audit_records": audit_records,
        }

    def _filter_query_entities(self, entities: list[ExtractedEntity]) -> list[ExtractedEntity]:
        filtered = []
        for entity in entities:
            if entity.density is None:
                entity.density = self.density_scorer.score_entity(entity.normalized_text, entity.label)
            if self.disable_age_date and entity.label in {"AGE", "DATE"}:
                entity.should_perturb = False
                continue
            if self.medical_typer.is_generic_sensitive(entity.label, entity.normalized_text.lower()):
                entity.should_perturb = False
                continue
            if not self.medical_typer.is_high_risk(
                entity.normalized_text,
                entity.label,
                density=entity.density,
                evidence_confidence=entity.evidence_confidence,
                evidence_source=entity.evidence_source,
            ):
                entity.should_perturb = False
                continue
            entity.should_perturb = True
            filtered.append(entity)
        return filtered
