from __future__ import annotations

import logging
from typing import Any

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

LOGGER = logging.getLogger(__name__)


class ContextCueEncoder:
    CUE_PROTOTYPES = {
        "self_reference": [
            "first person personal case narrative written by the patient or speaker",
            "self reference such as the speaker describing their own private situation",
        ],
        "relationship": [
            "family relation or spouse child parent reference linked to a private case",
            "relationship mention involving husband wife daughter son mother father or partner in a personal narrative",
        ],
        "temporal": [
            "timeline marker duration chronology or before after progression in a private case",
            "temporal expression describing weeks months years ago since recently or event order",
        ],
        "measurement": [
            "measurement dosage quantity count or clinical value such as dose tablet capsule mg ml blood pressure",
            "numeric medical measurement or dosage expression in a health related context",
        ],
        "clinical": [
            "medical complaint symptom diagnosis treatment history or personal health narrative",
            "clinical narrative involving symptoms treatment events disease history or medical advice",
        ],
    }

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model = None
        self.prototype_embeddings = None
        self.cue_names = list(self.CUE_PROTOTYPES.keys())
        if SentenceTransformer is None:
            LOGGER.warning("sentence-transformers unavailable; ContextCueEncoder falls back to zero scores.")
            return
        try:
            self.model = SentenceTransformer(model_name)
            prototype_texts = ["\n".join(self.CUE_PROTOTYPES[name]) for name in self.cue_names]
            self.prototype_embeddings = self.model.encode(
                prototype_texts,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        except Exception as exc:
            LOGGER.warning("Failed to initialize ContextCueEncoder with %s: %s", model_name, exc)
            self.model = None
            self.prototype_embeddings = None

    def encode_scores(self, query: str, candidate_text: str, local_context: str) -> dict[str, float]:
        if self.model is None or self.prototype_embeddings is None:
            return {name: 0.0 for name in self.cue_names}

        text = (
            f"query: {query}\n"
            f"candidate: {candidate_text}\n"
            f"context: {local_context}"
        )
        embedding = self.model.encode([text], normalize_embeddings=True, show_progress_bar=False)[0]
        scores = self.prototype_embeddings @ embedding
        return {
            name: max(0.0, min(1.0, 0.5 + 0.5 * float(scores[idx])))
            for idx, name in enumerate(self.cue_names)
        }

    def encode_feature_text(self, cue_scores: dict[str, float]) -> str:
        ordered = []
        for name in self.cue_names:
            ordered.append(f"{name}={cue_scores.get(name, 0.0):.2f}")
        return ", ".join(ordered)

