import json
import os
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


DEFAULT_TYPER_CONFIG = {
    "classification": {
        "min_score": 0.55,
        "current_label_prior": 0.12,
        "fallback_accept_confidence": 0.75,
    },
    "high_risk": {
        "org_gpe_confidence": 0.92,
        "org_gpe_density": 0.18,
        "org_gpe_min_confidence": 0.80,
        "org_min_confidence_strict": 0.88,
        "gpe_loc_min_confidence_strict": 0.95,
        "gpe_loc_density_strict": 0.12,
    },
    "weights": {
        "resource": 1.20,
        "model": 0.90,
        "heuristic": 0.78,
    },
    "heuristics": {
        "disease_contains": [
            " disease",
            " syndrome",
            "itis",
            "emia",
            "osis",
            "alopecia",
            "hyperthy",
            "addison",
            "graves",
        ],
        "drug_contains": [
            "mab",
            "azole",
            "cycline",
            "mycin",
            "oxetine",
            "metformin",
            "thyroxine",
        ],
    },
}


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
        config_path: Optional[str] = None,
    ) -> None:
        self.resource_registry = resource_registry
        self.enable_medical_ner = enable_medical_ner
        self.model_name = model_name or "en_ner_bc5cdr_md"
        self.config = self._load_config(config_path)
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

        if self._looks_like_email(normalized):
            return TypeDecision("EMAIL", 0.99, "regex")
        if self._looks_like_phone(normalized):
            return TypeDecision("PHONE", 0.99, "regex")
        if self._looks_like_age(normalized):
            return TypeDecision("AGE", 0.95, "regex")

        scores: dict[str, float] = {}
        evidence_sources: list[str] = []
        if self._in_terms(lowered, "DRUG"):
            self._add_score(scores, "DRUG", self._weight("resource"))
            evidence_sources.append("resource")
        if self._in_terms(lowered, "DISEASE"):
            self._add_score(scores, "DISEASE", self._weight("resource"))
            evidence_sources.append("resource")

        if self.model is not None:
            decision = self._classify_with_model(normalized)
            if decision is not None:
                self._add_score(scores, decision.label, self._weight("model"))
                evidence_sources.append("model")

        heuristic = self._classify_with_heuristics(normalized, current_label)
        if heuristic is not None:
            self._add_score(scores, heuristic.label, self._weight("heuristic"))
            evidence_sources.append("heuristic")

        if current_label and current_label != "UNKNOWN":
            self._add_score(scores, current_label, self._classification("current_label_prior"))
            evidence_sources.append("current_label")

        if not scores:
            return TypeDecision(current_label or "UNKNOWN", 0.4, "fallback")

        best_label, best_score = max(scores.items(), key=lambda item: item[1])
        min_score = self._classification("min_score")
        if best_score < min_score:
            return TypeDecision(current_label or "UNKNOWN", 0.4, "fallback")

        confidence = max(0.40, min(0.99, best_score / 1.5))
        source = "score_ensemble+" + "+".join(sorted(set(evidence_sources)))
        return TypeDecision(best_label, confidence, source)

    def is_high_risk(
        self,
        text: str,
        label: str,
        density: Optional[float] = None,
        evidence_confidence: Optional[float] = None,
        evidence_source: Optional[str] = None,
    ) -> bool:
        lowered = self._normalize(text).lower()
        if self.is_generic_sensitive(label, lowered):
            return False
        if label in {"EMAIL", "PHONE"}:
            return True
        if label in {"DISEASE", "DRUG"}:
            return True
        if label == "AGE":
            return True
        if label == "DATE":
            return bool(re.search(r"\b(19|20)\d{2}\b", lowered) or re.search(r"\b\d{1,2}[/-]\d{1,2}\b", lowered))
        if label in {"ORG", "GPE", "LOC"}:
            confidence = evidence_confidence or 0.0
            source = (evidence_source or "").lower()
            # Conservative guard: avoid perturbing location/org entities that are
            # mainly heuristic artifacts, which destabilized v5.
            if "heuristic" in source:
                return False
            # Restrict to resource-backed location/org evidence by default.
            if "resource" not in source and "regex" not in source:
                return False
            if label == "ORG":
                if confidence >= self._high_risk("org_min_confidence_strict"):
                    return True
                if (
                    density is not None
                    and density <= self._high_risk("org_gpe_density")
                    and confidence >= self._high_risk("org_gpe_min_confidence")
                ):
                    return True
                return False
            # GPE/LOC are even stricter to prevent over-perturbing public geography.
            if (
                confidence >= self._high_risk("gpe_loc_min_confidence_strict")
                and density is not None
                and density <= self._high_risk("gpe_loc_density_strict")
            ):
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
        predicted_match = decision.label == target_label or (
            target_label == "GPE" and decision.label == "LOC"
        )
        if not predicted_match:
            return False
        # Do not accept weak fallback matches as typed evidence.
        if decision.source == "fallback":
            return decision.confidence >= self._classification("fallback_accept_confidence")
        return True

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
        disease_tokens = self._heuristics("disease_contains")
        drug_tokens = self._heuristics("drug_contains")
        if any(token in lowered for token in disease_tokens):
            return TypeDecision("DISEASE", 0.78, "heuristic")
        if any(token in lowered for token in drug_tokens):
            return TypeDecision("DRUG", 0.78, "heuristic")
        if current_label == "PERSON" and self._looks_like_person(text):
            return TypeDecision("PERSON", 0.75, "heuristic")
        if current_label == "ORG" and self._looks_like_location_or_org(text, current_label):
            return TypeDecision("ORG", 0.72, "heuristic")
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

    def _load_config(self, config_path: Optional[str]) -> dict:
        config = json.loads(json.dumps(DEFAULT_TYPER_CONFIG))
        candidate_paths = []
        if config_path:
            candidate_paths.append(config_path)
        elif self.resource_registry is not None and hasattr(self.resource_registry, "resources_dir"):
            candidate_paths.append(os.path.join(self.resource_registry.resources_dir, "medical_typer_config.json"))

        for path in candidate_paths:
            if not path or not os.path.exists(path):
                continue
            try:
                with open(path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                if isinstance(payload, dict):
                    self._deep_merge(config, payload)
            except Exception:
                continue
        return config

    def _deep_merge(self, base: dict, update: dict) -> None:
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value

    def _add_score(self, scores: dict[str, float], label: str, weight: float) -> None:
        if not label or label == "UNKNOWN":
            return
        scores[label] = scores.get(label, 0.0) + float(weight)

    def _classification(self, key: str) -> float:
        return float(self.config.get("classification", {}).get(key, DEFAULT_TYPER_CONFIG["classification"][key]))

    def _high_risk(self, key: str) -> float:
        return float(self.config.get("high_risk", {}).get(key, DEFAULT_TYPER_CONFIG["high_risk"][key]))

    def _weight(self, key: str) -> float:
        return float(self.config.get("weights", {}).get(key, DEFAULT_TYPER_CONFIG["weights"][key]))

    def _heuristics(self, key: str) -> list[str]:
        values = self.config.get("heuristics", {}).get(key, DEFAULT_TYPER_CONFIG["heuristics"][key])
        return list(values) if isinstance(values, Iterable) and not isinstance(values, (str, bytes)) else list(DEFAULT_TYPER_CONFIG["heuristics"][key])
