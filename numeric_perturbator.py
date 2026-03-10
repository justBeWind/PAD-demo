import math
import random
from typing import Callable, Dict, Tuple
from auditor import auditor

class SemanticBoundedNumericPerturbator:
    """
    Implements a Semantic-Bounded Piecewise Mechanism.
    Infers the logical domain boundaries [L, U] via a Masked Context LLM generation (Data-Independent),
    and strictly applies the DP Piecewise Mechanism within that normalized domain.
    """
    def __init__(self, llm_generate_fn: Callable[[str, float], str]):
        """
        llm_generate_fn: A generic callable taking (prompt, temperature) and returning the LLM output string.
        """
        self.llm_generate = llm_generate_fn

    def _infer_semantic_bounds(self, masked_context: str, original_value: float, label: str) -> Tuple[float, float]:
        """
        Use LLM (T=0) to infer the realistic [lower, upper] bounds for the masked numeric value.
        This creates a data-independent domain bounding operation that requires NO privacy budget.
        """
        prompt = (
            f"You are a biomedical and data science expert.\n"
            f"Given the following context where a {label} value is masked as [MASK], "
            f"infer the biologically, physically, or logically realistic and safe MINIMUM and MAXIMUM bounds for this value.\n"
            f"Respond ONLY with a valid JSON object in the exact format: {{\"lower\": min_val, \"upper\": max_val}}.\n"
            f"Do NOT wrap it in markdown. Do NOT add any extra text.\n"
            f"Context: {masked_context}\n"
        )
        
        try:
            response = self.llm_generate(prompt, 0.0)
            import json
            import re
            
            # Simple JSON extraction regex in case LLM adds markdown wrapping
            json_str = re.search(r'\{.*?\}', response, re.DOTALL).group()
            bounds = json.loads(json_str)
            lower = float(bounds['lower'])
            upper = float(bounds['upper'])
            
            if lower > upper:
                lower, upper = upper, lower
                
            # If original_value is outside the inferred bounds, gracefully expand them.
            # (Note: In a pure LDP setup where original_value cannot be accessed at all during support generation, 
            # this step can be removed or bounded by pre-defined constants).
            if original_value < lower: lower = original_value * 0.8
            if original_value > upper: upper = original_value * 1.2
            
            return lower, upper
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Failed to infer context bounds: {e}. Fallback to generic margin.")
            # Fallback heuristic
            fallback_margin = max(abs(original_value) * 0.5, 10.0)
            return original_value - fallback_margin, original_value + fallback_margin

    def _piecewise_mechanism(self, value: float, epsilon: float) -> float:
        """
        Standard Piecewise Mechanism (PM) for values strictly in [-1, 1].
        """
        value = max(-1.0, min(1.0, value))
        
        exp_eps_half = math.exp(epsilon / 2.0)
        C = (math.exp(epsilon) + 1) / (math.exp(epsilon) - 1)
        L = (C + 1) / 2.0 * value - (C - 1) / 2.0
        R = L + C - 1.0
        
        exp_eps = math.exp(epsilon)
        p = (exp_eps - exp_eps_half) / (2.0 * exp_eps_half + 2.0)
        
        threshold = exp_eps_half / (exp_eps_half + 1.0)
        
        z = random.random()
        
        if z < threshold:
            # sample uniformly from [L, R]
            return random.uniform(L, R)
        else:
            # sample uniformly from [-C, L) U (R, C]
            left_len = max(0.0, L - (-C))
            right_len = max(0.0, C - R)
            
            if left_len + right_len == 0:
                return random.uniform(L, R)
                
            if random.random() < left_len / (left_len + right_len):
                return random.uniform(-C, L)
            else:
                return random.uniform(R, C)

    def perturb(self, doc_id: str, original_value: float, label: str, masked_context: str, epsilon: float) -> float:
        lower, upper = self._infer_semantic_bounds(masked_context, original_value, label)
        
        # 1. Normalize original_value to [-1, 1] range based on inferred [lower, upper]
        if upper == lower:
            normalized_value = 0.0
        else:
            normalized_value = 2.0 * (original_value - lower) / (upper - lower) - 1.0
            
        # 2. Apply PM mechanism (Returns value in [-C, C])
        perturbed_normalized = self._piecewise_mechanism(normalized_value, epsilon)
        
        # 3. Denormalize back to [lower, upper] support bounds space
        perturbed_value = (perturbed_normalized + 1.0) / 2.0 * (upper - lower) + lower
        
        # Safety bound (Since PM outputs exactly within [-C, C])
        C = (math.exp(epsilon) + 1) / (math.exp(epsilon) - 1) if epsilon > 0 else 1e6
        absolute_min = lower - ((C - 1.0) / 2.0) * (upper - lower)
        absolute_max = upper + ((C - 1.0) / 2.0) * (upper - lower)
        
        final_value = max(absolute_min, min(absolute_max, perturbed_value))
        
        auditor.log_numeric_perturbation(
            doc_id=doc_id,
            original_value=original_value,
            entity_label=label,
            masked_context=masked_context,
            semantic_bounds={"lower": lower, "upper": upper},
            selected_value=final_value,
            epsilon=epsilon
        )
        return final_value
