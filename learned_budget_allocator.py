from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from phi_taxonomy import phi_definition
from unit_constructor import ProtectionUnit

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


LOGGER = logging.getLogger(__name__)


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _safe_mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


@dataclass
class BudgetExample:
    query: str
    unit_text: str
    local_context: str
    phi_type: str
    risk: float
    utility: float


class BudgetHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LearnedBudgetAllocator:
    def __init__(
        self,
        epsilon: float = 0.2,
        min_sigma: float = 0.004,
        max_sigma: float = 0.04,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        self.epsilon = epsilon
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.embedding_model_name = embedding_model_name
        self.checkpoint_path = checkpoint_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = None
        self.head = None
        self.available = False

        if SentenceTransformer is None:
            LOGGER.warning("sentence-transformers unavailable; learned budget allocator disabled.")
            return

        try:
            self.encoder = SentenceTransformer(embedding_model_name)
            embedding_dim = self.encoder.get_sentence_embedding_dimension()
            self.head = BudgetHead(embedding_dim).to(self.device)
            if checkpoint_path and os.path.exists(checkpoint_path):
                self._load(checkpoint_path)
                self.available = True
            else:
                LOGGER.info("Learned budget allocator checkpoint not found at %s; falling back to teacher allocator.", checkpoint_path)
        except Exception as exc:
            LOGGER.warning("Failed to initialize learned budget allocator: %s", exc)
            self.encoder = None
            self.head = None
            self.available = False

    def _build_text(self, query: str, unit: ProtectionUnit) -> str:
        return (
            f"query: {query}\n"
            f"unit: {unit.local_text}\n"
            f"context: {unit.local_text}\n"
            f"phi_type: {unit.phi_type}"
        )

    def _encode(self, query: str, units: list[ProtectionUnit]) -> torch.Tensor:
        if self.encoder is None:
            raise RuntimeError("Encoder unavailable.")
        texts = [self._build_text(query, unit) for unit in units]
        embeddings = self.encoder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return torch.tensor(embeddings, dtype=torch.float32, device=self.device)

    def predict_scores(self, query: str, units: list[ProtectionUnit]) -> list[dict[str, float]]:
        if not self.available or self.head is None:
            raise RuntimeError("Learned budget allocator checkpoint unavailable.")
        with torch.no_grad():
            embeddings = self._encode(query, units)
            preds = self.head(embeddings).detach().cpu().numpy()
        scores: list[dict[str, float]] = []
        for risk, utility in preds:
            scores.append(
                {
                    "risk": _clip(float(risk), 0.0, 1.0),
                    "utility": _clip(float(utility), 0.0, 1.0),
                }
            )
        return scores

    def allocate(self, query: str, docs: list[str], spans, units: list[ProtectionUnit]) -> list[ProtectionUnit]:
        del docs, spans
        scores = self.predict_scores(query, units)
        for unit, score in zip(units, scores):
            risk = score["risk"]
            utility = score["utility"]
            privacy_pressure = _clip(risk * (1.0 - utility), 0.0, 1.0)
            unit.risk_score = risk
            unit.utility_score = utility
            unit.identifiability_score = risk
            unit.semantic_relevance = utility
            unit.retrieval_contribution = utility
            unit.copy_risk = _safe_mean([risk, privacy_pressure])
            unit.sigma = self._sigma(privacy_pressure)
            unit.clip_norm = self._clip_norm(risk, utility)
            unit.blend = self._blend(privacy_pressure)
            unit.midlayer_strength = self._midlayer_strength(unit.blend)

            for span in unit.spans:
                span.risk_score = max(span.risk_score, risk)
                span.utility_score = utility
                span.copy_risk = unit.copy_risk
                span.sigma = unit.sigma
                span.clip_norm = unit.clip_norm
        return units

    def should_perturb(self, unit: ProtectionUnit) -> bool:
        if phi_definition(unit.phi_type).group == "direct":
            return unit.risk_score >= 0.34
        protection_score = _safe_mean([unit.risk_score, unit.copy_risk, 1.0 - unit.utility_score])
        return protection_score >= 0.48

    def _sigma(self, privacy_pressure: float) -> float:
        epsilon_scale = _clip(1.0 / max(self.epsilon, 1e-3), 1.0, 5.0)
        sigma = self.min_sigma + (self.max_sigma - self.min_sigma) * privacy_pressure * (epsilon_scale / 5.0)
        return _clip(sigma, self.min_sigma, self.max_sigma)

    def _clip_norm(self, risk: float, utility: float) -> float:
        preservation = _safe_mean([risk, utility])
        return _clip(0.06 + 0.28 * preservation, 0.06, 0.34)

    def _blend(self, privacy_pressure: float) -> float:
        return _clip(0.08 + 0.36 * privacy_pressure, 0.08, 0.44)

    def _midlayer_strength(self, blend: float) -> float:
        return _clip(0.10 + 0.52 * blend, 0.10, 0.62)

    def _load(self, checkpoint_path: str) -> None:
        assert self.head is not None
        payload = torch.load(checkpoint_path, map_location=self.device)
        self.head.load_state_dict(payload["head_state_dict"])
        self.head.eval()


def load_budget_examples(path: str) -> list[BudgetExample]:
    examples: list[BudgetExample] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            examples.append(BudgetExample(**record))
    return examples


def train_budget_head(
    examples: list[BudgetExample],
    output_path: str,
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    epochs: int = 8,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    device: Optional[str] = None,
) -> dict[str, float]:
    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers is required to train the learned budget allocator.")
    if not examples:
        raise ValueError("No training examples provided.")

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    encoder = SentenceTransformer(embedding_model_name)
    texts = [
        (
            f"query: {example.query}\n"
            f"unit: {example.unit_text}\n"
            f"context: {example.local_context}\n"
            f"phi_type: {example.phi_type}"
        )
        for example in examples
    ]
    embeddings = encoder.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    x = torch.tensor(embeddings, dtype=torch.float32, device=device)
    y = torch.tensor([[example.risk, example.utility] for example in examples], dtype=torch.float32, device=device)

    head = BudgetHead(x.shape[1]).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        permutation = torch.randperm(x.size(0), device=device)
        epoch_loss = 0.0
        for start in range(0, x.size(0), batch_size):
            idx = permutation[start : start + batch_size]
            preds = head(x[idx])
            loss = criterion(preds, y[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item()) * idx.numel()
        epoch_loss /= float(x.size(0))
        LOGGER.info("Learned budget allocator epoch %d/%d loss=%.6f", epoch + 1, epochs, epoch_loss)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(
        {
            "embedding_model_name": embedding_model_name,
            "head_state_dict": head.state_dict(),
        },
        output_path,
    )
    with torch.no_grad():
        final_loss = float(criterion(head(x), y).item())
    return {"train_mse": final_loss, "num_examples": float(len(examples))}

