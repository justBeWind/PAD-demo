import re
from typing import Callable, List
from dataclasses import dataclass

from entity_extractor import UniversalEntityExtractor, ExtractedEntity
from numeric_perturbator import SemanticBoundedNumericPerturbator
from categorical_perturbator import LLMCategoricalPerturbator
from auditor import auditor

class DenPADPipeline:
    """
    The main control plane for the Generalized DenPAD system.
    Coordinates entity extraction, budget allocation, and the application of 
    Numeric and Categorical DP Perturbations strictly adhering to Sequential Composition.
    """
    def __init__(
        self,
        llm_generate_fn: Callable[[str, float], str],
        similarity_fn: Callable[[str, str], float],
        spacy_model: str = "en_core_web_sm"
    ):
        """
        Receives generic callables for LLM generation and Embeddings similarity to stay decoupled 
        from any concrete backend (like HuggingFace, vLLM, or OpenAI).
        """
        self.extractor = UniversalEntityExtractor(spacy_model=spacy_model)
        self.numeric_perturbator = SemanticBoundedNumericPerturbator(llm_generate_fn=llm_generate_fn)
        self.categorical_perturbator = LLMCategoricalPerturbator(
            llm_generate_fn=llm_generate_fn, 
            similarity_fn=similarity_fn
        )

    def perturb_document(self, doc_id: str, text: str, total_epsilon: float) -> str:
        """
        Given a sensitive document text and a total epsilon budget:
        1. Identifies sensitive elements.
        2. Allocates epsilon across elements.
        3. Randomizes each element while maintaining context via Masked-Generation DP.
        4. Replaces elements and returns the secure document representation.
        """
        # 1. Extract entities
        entities: List[ExtractedEntity] = self.extractor.extract(text)
        
        if not entities:
            return text
            
        # 2. Privacy Budget Allocation (Sequential Composition)
        # Uniform allocation guarantees the entire document perturbation satisfies (total_epsilon)-DP.
        budget_per_entity = total_epsilon / len(entities)
        
        # We must iterate backwards to apply replacements safely without destroying character offsets
        # for elements that occur later in the string.
        entities.sort(key=lambda e: e.start_char, reverse=True)
        
        perturbed_text = text
        for ent in entities:
            # Isolate the original context, replacing just this entity with [MASK]
            # so the LLM generator is perfectly blind to the original sensitive data.
            prefix = text[:ent.start_char]
            suffix = text[ent.end_char:]
            masked_context = f"{prefix}[MASK]{suffix}"
            
            if ent.category == "numerical":
                try:
                    # Clean non-digit characters to parse numeric base
                    clean_str = ''.join(c for c in ent.text if c.isdigit() or c == '.')
                    if not clean_str: raise ValueError("No digits found")
                    original_val = float(clean_str)
                    
                    new_val = self.numeric_perturbator.perturb(
                        doc_id=doc_id,
                        original_value=original_val,
                        label=ent.label,
                        masked_context=masked_context,
                        epsilon=budget_per_entity
                    )
                    
                    # Try to preserve natural units like "years old" or "mg"
                    unit_match = re.search(r'[a-zA-Z%]+', ent.text)
                    unit = unit_match.group(0) if unit_match else ""
                    
                    # Formatting logic (drop decimal if it was naturally an integer)
                    if new_val.is_integer():
                        replacement = f"{int(new_val)} {unit}".strip()
                    else:
                        replacement = f"{new_val:.1f} {unit}".strip()
                        
                except Exception as e:
                    import logging
                    logging.getLogger(__name__).error(f"Numeric perturbation failed for '{ent.text}': {e}")
                    # Fallback to the Redaction or original on critical failure
                    replacement = "[REDACTED]"
            else:
                try:
                    replacement = self.categorical_perturbator.perturb(
                        doc_id=doc_id,
                        original_entity=ent.text,
                        label=ent.label,
                        masked_context=masked_context,
                        epsilon=budget_per_entity
                    )
                except Exception as e:
                    import logging
                    logging.getLogger(__name__).error(f"Categorical perturbation failed for '{ent.text}': {e}")
                    replacement = "[REDACTED]"
                
            # 4. Integrate into text
            perturbed_text = perturbed_text[:ent.start_char] + replacement + perturbed_text[ent.end_char:]
            
        return perturbed_text
