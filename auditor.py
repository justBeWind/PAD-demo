import json
import logging
import os
import time
from typing import Any, Dict, List

class DPAuditor:
    """
    Global auditing infrastructure for the DP-RAG pipeline.
    Ensures every perturbation is cryptographically traceable for post-hoc ablation studies.
    """
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, f"dp_audit_{int(time.time())}.jsonl")
        
        # Configure standard logging to console as well
        self.logger = logging.getLogger("DPAuditor")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            ch = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)

    def log_categorical_perturbation(
        self,
        doc_id: str,
        original_entity: str,
        entity_label: str,
        masked_context: str,
        candidates: List[Dict[str, Any]],
        selected_candidate: str,
        epsilon: float
    ):
        """
        Logs a categorical perturbation via Exponential Mechanism.
        `candidates` must be a list of dicts with: {'text': str, 'utility': float, 'probability': float}
        """
        record = {
            "timestamp": time.time(),
            "type": "categorical_perturbation",
            "doc_id": doc_id,
            "mechanism": "Masked-Context-Exponential",
            "epsilon": epsilon,
            "original_entity": original_entity,
            "entity_label": entity_label,
            "masked_context": masked_context,
            "candidates": candidates,
            "selected_candidate": selected_candidate
        }
        self._write_record(record)
        self.logger.info(f"[Categorical DP] '{original_entity}' -> '{selected_candidate}' (eps={epsilon})")

    def log_numeric_perturbation(
        self,
        doc_id: str,
        original_value: float,
        entity_label: str,
        masked_context: str,
        semantic_bounds: Dict[str, float],
        selected_value: float,
        epsilon: float
    ):
        """
        Logs a numeric perturbation via Semantic-Bounded Piecewise Mechanism.
        `semantic_bounds` should be {'lower': float, 'upper': float}
        """
        record = {
            "timestamp": time.time(),
            "type": "numeric_perturbation",
            "doc_id": doc_id,
            "mechanism": "Semantic-Bounded-PM",
            "epsilon": epsilon,
            "original_value": original_value,
            "entity_label": entity_label,
            "masked_context": masked_context,
            "semantic_bounds": semantic_bounds,
            "selected_value": selected_value
        }
        self._write_record(record)
        self.logger.info(f"[Numeric DP] {original_value} -> {selected_value:.2f} within {semantic_bounds} (eps={epsilon})")

    def _write_record(self, record: Dict[str, Any]):
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

# Global singleton auditor for easy import
auditor = DPAuditor()
