import re
from dataclasses import dataclass
from typing import Iterable, Optional

try:
    import spacy
except ImportError:
    spacy = None


@dataclass(frozen=True)
class TypeDecision:
    label: str
    confidence: float
    source: str


class MedicalTyper:
    """
    Local type enhancer for retrieval-time DenPAD.

    This module prefers explicit public-resource evidence, falls back to an
    optional local biomedical model, and finally uses conservative heuristics.
    It is intentionally local-only and does not rely on external APIs.
    """

    def __init__(
        self,
        resource_registry=None,
        model_name: Optional[str] = None,
        enable_medical_ner: bool = True,
    ) -> None:
        self.resource_registry = resource_registry
        self.enable_medical_ner = enable_medical_ner
        self.model_name = model_name or "en_ner_bc5cdr_md"
        self.model = None
        if enable_medical_ner and spacy is not None:
            for candidate in (
                self.model_name,
                "en_ner_bc5cdr_md",
                "en_core_sci_md",
                "en_core_web_sm",
            ):
                try:
                    self.model = spacy.load(candidate)
                    break
                except Exception:
                    continue

    def classify(self, text: str, current_label: Optional[str] = None) -> TypeDecision:
        normalized = self._normalize(text)
        lowered = normalized.lower()
        if not normalized:
            return TypeDecision(current_label or "UNKNOWN", 0.0, "empty")

        if self._in_terms(lowered, "DRUG"):
            return TypeDecision("DRUG", 0.99, "resource")
        if self._in_terms(lowered, "DISEASE"):
            return TypeDecision("DISEASE", 0.99, "resource")
        if self._looks_like_email(normalized):
            return TypeDecision("EMAIL", 0.99, "regex")
        if self._looks_like_phone(normalized):
            return TypeDecision("PHONE", 0.99, "regex")
        if self._looks_like_age(normalized):
            return TypeDecision("AGE", 0.95, "regex")

        if self.model is not None:
            decision = self._classify_with_model(normalized)
            if decision is not None:
                return decision

        heuristic = self._classify_with_heuristics(normalized, current_label)
        if heuristic is not None:
            return heuristic
        return TypeDecision(current_label or "UNKNOWN", 0.4, "fallback")

    def is_high_risk(
        self,
        text: str,
        label: str,
        density: Optional[float] = None,
        evidence_confidence: Optional[float] = None,
    ) -> bool:
        lowered = self._normalize(text).lower()
        if self.is_generic_sensitive(label, lowered):
            return False
        if label in {"EMAIL", "PHONE"}:
            return True
        if label in {"DISEASE", "DRUG", "PERSON"}:
            return True
        if label == "AGE":
            return True
        if label == "DATE":
            return bool(re.search(r"\b(19|20)\d{2}\b", lowered) or re.search(r"\b\d{1,2}[/-]\d{1,2}\b", lowered))
        if label in {"ORG", "GPE", "LOC"}:
            confidence = evidence_confidence or 0.0
            if confidence >= 0.92:
                return True
            if density is not None and density <= 0.18 and confidence >= 0.8:
                return True
            return False
        return False

    def is_generic_sensitive(self, label: str, lowered_text: str) -> bool:
        if label == "DISEASE":
            generic = {
                "cancer",
                "infection",
                "infections",
                "disease",
                "diseases",
                "syndrome",
                "disorder",
            }
            return lowered_text in generic
        if label in {"ORG", "GPE", "LOC"}:
            generic = {
                "hospital",
                "clinic",
                "medical institute",
                "health center",
                "general hospital",
                "city clinic",
            }
            return lowered_text in generic
        return False

    def candidate_matches_label(self, candidate: str, target_label: str) -> bool:
        decision = self.classify(candidate, target_label)
        if target_label == "LOC":
            target_label = "GPE"
        return decision.label == target_label or (
            target_label == "GPE" and decision.label == "LOC"
        )

    def candidate_group_matches(self, original: str, candidate: str, target_label: str) -> bool:
        if target_label not in {"DISEASE", "DRUG"}:
            return self.candidate_matches_label(candidate, target_label)
        original_l = self._normalize(original).lower()
        candidate_l = self._normalize(candidate).lower()
        if original_l == candidate_l:
            return True
        # Hard guardrails for endocrine/psychiatric confusion that still slips
        # through semantic similarity and shared high-level tags.
        if any(token in original_l for token in ("graves", "hyperthy", "thyroid")) and any(
            token in candidate_l for token in ("addison", "adrenal")
        ):
            return False
        if any(token in original_l for token in ("addison", "adrenal")) and any(
            token in candidate_l for token in ("graves", "hyperthy", "thyroid")
        ):
            return False
        if "anxiety" in original_l and "panic disorder" == candidate_l:
            return False
        original_group = self._medical_group(original_l)
        candidate_group = self._medical_group(candidate_l)
        if original_group and candidate_group:
            return original_group == candidate_group
        return self.candidate_matches_label(candidate, target_label)

    def has_strong_evidence(self, text: str, label: str) -> bool:
        decision = self.classify(text, label)
        return decision.label == label and decision.confidence >= 0.8

    def _classify_with_model(self, text: str) -> Optional[TypeDecision]:
        try:
            doc = self.model(text)
        except Exception:
            return None
        labels = [ent.label_ for ent in getattr(doc, "ents", []) if ent.text.strip().lower() == text.strip().lower()]
        if not labels:
            return None
        label = labels[0]
        if label in {"DISEASE", "DISEASE_OR_SYNDROME"}:
            return TypeDecision("DISEASE", 0.9, "medical_model")
        if label in {"CHEMICAL", "DRUG"}:
            return TypeDecision("DRUG", 0.9, "medical_model")
        if label in {"PERSON", "ORG", "GPE", "LOC", "DATE", "TIME"}:
            return TypeDecision(label, 0.85, "model")
        return None

    def _classify_with_heuristics(self, text: str, current_label: Optional[str]) -> Optional[TypeDecision]:
        lowered = text.lower()
        if any(token in lowered for token in (" disease", " syndrome", "itis", "emia", "osis", "alopecia", "hyperthy", "addison", "graves")):
            return TypeDecision("DISEASE", 0.78, "heuristic")
        if any(token in lowered for token in ("mab", "azole", "cycline", "mycin", "oxetine", "metformin", "thyroxine")):
            return TypeDecision("DRUG", 0.78, "heuristic")
        if current_label == "PERSON" and self._looks_like_person(text):
            return TypeDecision("PERSON", 0.75, "heuristic")
        if current_label in {"ORG", "GPE", "LOC"} and self._looks_like_location_or_org(text, current_label):
            return TypeDecision(current_label, 0.72, "heuristic")
        return None

    def _in_terms(self, text: str, category: str) -> bool:
        if self.resource_registry is None:
            return False
        try:
            return text in self.resource_registry.get_terms(category)
        except Exception:
            return False

    def _normalize(self, text: str) -> str:
        return re.sub(r"\s+", " ", text.strip())

    def _looks_like_email(self, text: str) -> bool:
        return bool(re.fullmatch(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text))

    def _looks_like_phone(self, text: str) -> bool:
        digits = re.sub(r"\D", "", text)
        return len(digits) >= 8

    def _looks_like_age(self, text: str) -> bool:
        if not text.isdigit():
            return False
        value = int(text)
        return 0 <= value <= 120

    def _looks_like_person(self, text: str) -> bool:
        parts = [part for part in text.split() if part]
        return len(parts) >= 2 and all(part[:1].isupper() for part in parts)

    def _looks_like_location_or_org(self, text: str, label: str) -> bool:
        parts = [part for part in text.split() if part]
        if not parts:
            return False
        if label in {"GPE", "LOC"}:
            if any(token.lower() in {"disease", "syndrome", "cancer", "thyroid", "alopecia", "arthritis"} for token in parts):
                return False
            return len(parts) >= 1 and all(part[:1].isupper() for part in parts)
        if label == "ORG":
            return any(token.lower() in {"hospital", "clinic", "center", "institute", "lab"} for token in parts)
        return False

    def _medical_group(self, text: str) -> Optional[str]:
        if any(token in text for token in ("thyroid", "graves", "hyperthy")):
            return "thyroid_endocrine"
        if any(token in text for token in ("addison", "adrenal")):
            return "adrenal_endocrine"
        if any(token in text for token in ("alopecia", "psoriasis", "eczema", "dermatitis", "scalp")):
            return "dermatology"
        if any(token in text for token in ("lung", "pulmonary", "asthma", "bronchitis", "respiratory", "ild")):
            return "pulmonary"
        if any(token in text for token in ("h pylori", "hpylori", "gastr", "ulcer", "liver", "jaundice", "hepat")):
            return "gastrointestinal"
        if any(token in text for token in ("cancer", "oncology", "tumor", "carcinoma")):
            return "oncology"
        if any(token in text for token in ("anemia", "hemat")):
            return "hematology"
        if any(token in text for token in ("anxiety", "panic", "depress", "psychiatr", "mental health", "mood")):
            return "psychiatric"
        return None
