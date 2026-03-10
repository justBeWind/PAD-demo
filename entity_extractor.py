import re
from dataclasses import dataclass
from typing import List

try:
    import spacy
except ImportError:
    spacy = None

@dataclass
class ExtractedEntity:
    text: str
    label: str
    start_char: int
    end_char: int
    category: str  # "numerical" or "categorical"

# Regex patterns for basic PII extraction
EMAIL_PATTERN = re.compile(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b")
PHONE_PATTERN = re.compile(r"(?:\+?\d[\d\-\s()]{7,}\d)")
AGE_PATTERN = re.compile(r"\b(\d{1,3})\s*(?:years?\s*old|y/o|yo|yr|yrs)\b", re.IGNORECASE)
DOSE_PATTERN = re.compile(r"\b(\d+(?:\.\d+)?)\s*(?:mg|g|mcg|ml|L|kg|lb|oz)\b", re.IGNORECASE)

class UniversalEntityExtractor:
    """
    Domain-agnostic entity extractor.
    Replaces rule-based/dictionary-based extraction with generalized NLP and Regex.
    Categorizes entities directly into 'numerical' (for PM) and 'categorical' (for Exponential).
    """
    def __init__(self, spacy_model: str = "en_core_web_sm"):
        self.nlp = None
        if spacy is not None:
            try:
                self.nlp = spacy.load(spacy_model)
            except OSError:
                import logging
                logging.getLogger(__name__).warning(f"spaCy model {spacy_model} not found.")

        # Entities that are typically bounded real values or categorical choices
        self.categorical_labels = {"PERSON", "ORG", "GPE", "LOC", "NORP", "FAC", "PRODUCT", "EVENT", "WORK_OF_ART", "LAW", "LANGUAGE", "DISEASE", "DRUG"}
        self.numeric_labels = {"DATE", "TIME", "PERCENT", "MONEY", "QUANTITY", "ORDINAL", "CARDINAL"}

    def extract(self, text: str) -> List[ExtractedEntity]:
        entities = []
        
        # 1. Regex Extractions (Highest Priority)
        entities.extend(self._extract_regex(text))
        
        # 2. Semantic NLP Extractions
        if self.nlp is not None:
            spacy_entities = self._extract_spacy(text)
            entities.extend(spacy_entities)
            
        # Deduplicate & resolve overlapping boundaries
        return self._resolve_overlaps(entities)

    def _extract_regex(self, text: str) -> List[ExtractedEntity]:
        results = []
        
        for match in EMAIL_PATTERN.finditer(text):
            results.append(ExtractedEntity(
                text=match.group(0), label="EMAIL", start_char=match.start(), end_char=match.end(), category="categorical"
            ))
            
        for match in PHONE_PATTERN.finditer(text):
            results.append(ExtractedEntity(
                text=match.group(0), label="PHONE", start_char=match.start(), end_char=match.end(), category="numerical"
            ))
            
        for match in AGE_PATTERN.finditer(text):
            results.append(ExtractedEntity(
                text=match.group(1), label="AGE", start_char=match.start(1), end_char=match.end(1), category="numerical"
            ))
            
        for match in DOSE_PATTERN.finditer(text):
            results.append(ExtractedEntity(
                text=match.group(1), label="DOSAGE", start_char=match.start(1), end_char=match.end(1), category="numerical"
            ))
            
        return results

    def _extract_spacy(self, text: str) -> List[ExtractedEntity]:
        doc = self.nlp(text)
        results = []
        for ent in doc.ents:
            category = "categorical"
            if ent.label_ in self.numeric_labels:
                category = "numerical"
                
            results.append(ExtractedEntity(
                text=ent.text,
                label=ent.label_,
                start_char=ent.start_char,
                end_char=ent.end_char,
                category=category
            ))
        return results

    def _resolve_overlaps(self, entities: List[ExtractedEntity]) -> List[ExtractedEntity]:
        # Sort by start_char, then by length (longest first)
        entities.sort(key=lambda e: (e.start_char, -(e.end_char - e.start_char)))
        resolved = []
        last_end = -1
        
        for ent in entities:
            # If current entity starts after or identically where the last one ended, accept it if no overlap
            # To be safe, any entity that starts before the last one ends is discarded
            if ent.start_char >= last_end:
                resolved.append(ent)
                last_end = ent.end_char
                
        return resolved
