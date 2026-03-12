from __future__ import annotations

from dataclasses import dataclass


RELATIONSHIP_TERMS = {
    "husband",
    "wife",
    "daughter",
    "son",
    "mother",
    "father",
    "sister",
    "brother",
    "family",
    "relative",
    "partner",
    "boyfriend",
    "girlfriend",
    "child",
    "children",
}

FIRST_PERSON_TERMS = {
    "i",
    "me",
    "my",
    "mine",
    "myself",
}

TEMPORAL_TERMS = {
    "today",
    "yesterday",
    "week",
    "weeks",
    "month",
    "months",
    "year",
    "years",
    "ago",
    "since",
    "recently",
    "previously",
    "before",
    "after",
    "during",
    "timeline",
}

CLINICAL_TERMS = {
    "pain",
    "fever",
    "infection",
    "disease",
    "symptom",
    "symptoms",
    "diagnosis",
    "diagnosed",
    "treatment",
    "tablet",
    "tablets",
    "capsule",
    "capsules",
    "dose",
    "dosage",
    "surgery",
    "history",
    "pregnant",
    "pregnancy",
    "doctor",
    "hospital",
    "clinic",
    "medicine",
    "medication",
    "blood",
}

MEASUREMENT_TERMS = {
    "mg",
    "ml",
    "kg",
    "cm",
    "mm",
    "dose",
    "dosage",
    "tablet",
    "tablets",
    "capsule",
    "capsules",
    "per day",
    "twice a day",
    "blood pressure",
    "blood sugar",
}


@dataclass(frozen=True)
class PHITypeDefinition:
    name: str
    group: str
    structured: bool
    narrative: bool
    base_risk: float
    base_utility: float
    left_window: int
    right_window: int
    prototype_text: str


PHI_TYPE_DEFINITIONS = {
    "EMAIL": PHITypeDefinition("EMAIL", "direct", True, False, 0.98, 0.05, 24, 24, "direct private identifier such as an email address or contact handle"),
    "PHONE": PHITypeDefinition("PHONE", "direct", True, False, 0.98, 0.05, 24, 24, "direct private identifier such as a phone number or contact number"),
    "ID": PHITypeDefinition("ID", "direct", True, False, 0.97, 0.05, 24, 24, "direct private identifier such as an account number, record id, or personal id"),
    "ADDRESS": PHITypeDefinition("ADDRESS", "direct", True, False, 0.95, 0.08, 32, 40, "exact address or highly specific contact location"),
    "DATE_TIME": PHITypeDefinition("DATE_TIME", "direct", True, False, 0.86, 0.12, 28, 32, "exact date or exact time that can identify a personal event"),
    "PERSON_NAME": PHITypeDefinition("PERSON_NAME", "quasi", False, False, 0.82, 0.22, 36, 48, "person name or named individual in a private narrative"),
    "LOCATION": PHITypeDefinition("LOCATION", "quasi", False, False, 0.76, 0.20, 40, 52, "location, city, region, or place linked to a private individual"),
    "ORG_AFFILIATION": PHITypeDefinition("ORG_AFFILIATION", "quasi", False, False, 0.72, 0.22, 36, 48, "organization, hospital, clinic, employer, or institutional affiliation"),
    "AGE": PHITypeDefinition("AGE", "quasi", False, False, 0.78, 0.10, 30, 36, "age or age expression associated with a specific person"),
    "TEMPORAL_MARKER": PHITypeDefinition("TEMPORAL_MARKER", "quasi", False, False, 0.70, 0.16, 28, 36, "time marker or event timeline linked to a personal case"),
    "RELATIONSHIP": PHITypeDefinition("RELATIONSHIP", "quasi", False, False, 0.74, 0.18, 28, 40, "family or relationship reference connected to a private individual"),
    "MEASUREMENT": PHITypeDefinition("MEASUREMENT", "quasi", False, False, 0.42, 0.42, 18, 24, "measurement, dosage, count, or clinical value that is less identifying on its own"),
    "IDENTIFYING_NARRATIVE": PHITypeDefinition("IDENTIFYING_NARRATIVE", "narrative", False, True, 0.90, 0.22, 56, 84, "local medical or personal narrative fragment combining history, symptoms, age, location, or relationship details"),
}


LABEL_TO_PHI_TYPE = {
    "EMAIL": "EMAIL",
    "PHONE": "PHONE",
    "ID": "ID",
    "DATE": "DATE_TIME",
    "TIME": "DATE_TIME",
    "PERSON": "PERSON_NAME",
    "ORG": "ORG_AFFILIATION",
    "GPE": "LOCATION",
    "LOC": "LOCATION",
    "AGE": "AGE",
    "NUMERIC": "MEASUREMENT",
    "CARDINAL": "MEASUREMENT",
    "QUANTITY": "MEASUREMENT",
    "MONEY": "MEASUREMENT",
    "PERCENT": "MEASUREMENT",
    "RELATION": "RELATIONSHIP",
    "MISC": "IDENTIFYING_NARRATIVE",
}


def phi_definition(phi_type: str) -> PHITypeDefinition:
    return PHI_TYPE_DEFINITIONS[phi_type]


def normalize_phi_type(label: str, local_context: str, text: str) -> str:
    lowered_context = local_context.lower()
    lowered_text = text.lower()
    phi_type = LABEL_TO_PHI_TYPE.get(label, "IDENTIFYING_NARRATIVE")

    if phi_type == "MEASUREMENT":
        if any(term in lowered_context for term in RELATIONSHIP_TERMS) or any(term in lowered_context for term in FIRST_PERSON_TERMS):
            if any(term in lowered_context for term in TEMPORAL_TERMS):
                return "IDENTIFYING_NARRATIVE"
        if any(term in lowered_context for term in MEASUREMENT_TERMS):
            return "MEASUREMENT"

    if phi_type in {"PERSON_NAME", "LOCATION", "ORG_AFFILIATION", "AGE", "TEMPORAL_MARKER"}:
        if any(term in lowered_context for term in CLINICAL_TERMS) and (
            any(term in lowered_context for term in FIRST_PERSON_TERMS)
            or any(term in lowered_context for term in RELATIONSHIP_TERMS)
            or any(term in lowered_context for term in TEMPORAL_TERMS)
        ):
            return "IDENTIFYING_NARRATIVE"

    if label == "MISC":
        if any(term in lowered_context for term in CLINICAL_TERMS | RELATIONSHIP_TERMS | TEMPORAL_TERMS):
            return "IDENTIFYING_NARRATIVE"

    if label == "RELATION" or lowered_text in RELATIONSHIP_TERMS:
        return "RELATIONSHIP"

    return phi_type
