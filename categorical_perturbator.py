import math
import random
from typing import Callable, List
from auditor import auditor

class LLMCategoricalPerturbator:
    """
    Implements the Masked-Context Candidate Generation + Exponential Mechanism.
    1. Genereates robust candidate generalizations blindly based only on [MASK] context (epsilon-cost: 0).
    2. Measures semantic utility against the true entity.
    3. Selects a candidate strictly following the Exponential Mechanism formula.
    """
    def __init__(
        self, 
        llm_generate_fn: Callable[[str, float], str],
        similarity_fn: Callable[[str, str], float]
    ):
        """
        llm_generate_fn: function taking (prompt, temperature) -> string
        similarity_fn: function taking (str1, str2) -> float (normalized semantic utility, e.g. cosine sim from -1 to 1)
        """
        self.llm_generate = llm_generate_fn
        self.similarity_fn = similarity_fn

    def _generate_candidates(self, masked_context: str, label: str, k: int = 5) -> List[str]:
        """
        Blindly infers highly plausible generalizations or semantically valid substitutes.
        """
        prompt = (
            f"You are a helpful data anonymization assistant.\n"
            f"Given the following text where a '{label}' is hidden as [MASK], "
            f"generate {k} grammatically correct and logically sound alternatives or generalizations for the [MASK].\n"
            f"If it's about a medical condition, suggest 'a medical condition', 'a chronic illness', etc. rather than specific diseases.\n"
            f"If it's about a person, suggest 'a patient', 'the user', 'an individual'.\n"
            f"Output ONLY a Python list of strings, e.g. [\"option1\", \"option2\"].\n"
            f"Context: {masked_context}\n"
        )
        try:
            response = self.llm_generate(prompt, 0.0) # T=0 ensures independence from random noise, giving strict DP guarantee
            import ast
            import re
            list_str = re.search(r'\[.*?\]', response, re.DOTALL).group()
            candidates = ast.literal_eval(list_str)
            
            cleaned = [str(c).strip() for c in candidates if str(c).strip()]
            if not cleaned: raise ValueError("Empty candidate pool")
            return cleaned
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Fallback generation for {label} in context: {e}")
            return ["a related entity", "someone", "something", "an organization", "some location"]

    def _validate_candidates(self, candidates: List[str], label: str) -> List[str]:
        """
        (Optional) Secondary T=0 filter logic. For now, since the generating LLM is blind, 
        any item it generates is technically a valid Domain element.
        """
        validated = list(set(candidates))
        return validated

    def perturb(self, doc_id: str, original_entity: str, label: str, masked_context: str, epsilon: float) -> str:
        # 1. Zero-Knowledge Blind Generation
        candidates = self._generate_candidates(masked_context, label)
        candidates = self._validate_candidates(candidates, label)
        
        # 2. Score utility via semantic similarity mechanism
        scored_candidates = []
        for cand in candidates:
            u_score = self.similarity_fn(original_entity, cand)
            scored_candidates.append({
                "text": cand,
                "utility": u_score
            })
            
        # 3. Exponential Mechanism
        # Assuming Cosine Similarity, the range of maximum change Delta U between any two possible items is 2.0
        delta_u = 2.0
        
        sum_exp = 0.0
        for sc in scored_candidates:
            # Exponetial Mechanism Formula
            val = math.exp((epsilon * sc["utility"]) / (2.0 * delta_u))
            sc["exp_weight"] = val
            sum_exp += val
            
        # Sampling based on computed weights
        rand_val = random.uniform(0, sum_exp)
        cumulative = 0.0
        selected = scored_candidates[-1]["text"]
        
        for sc in scored_candidates:
            sc["probability"] = sc["exp_weight"] / sum_exp
            cumulative += sc["exp_weight"]
            if rand_val <= cumulative:
                selected = sc["text"]
                break
                
        # Clean formatting
        for sc in scored_candidates:
            sc.pop("exp_weight", None)
            
        # 4. Log everything completely transparently
        auditor.log_categorical_perturbation(
            doc_id=doc_id,
            original_entity=original_entity,
            entity_label=label,
            masked_context=masked_context,
            candidates=scored_candidates,
            selected_candidate=selected,
            epsilon=epsilon
        )
        
        return selected
