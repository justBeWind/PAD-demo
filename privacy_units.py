from dataclasses import dataclass, field


@dataclass
class SensitiveSpan:
    text: str
    label: str
    start_char: int
    end_char: int
    doc_index: int
    evidence_source: str
    phi_type: str = "MISC"
    risk_group: str = "quasi"
    is_structured: bool = False
    is_narrative: bool = False
    token_start: int = -1
    token_end: int = -1
    risk_score: float = 0.0
    utility_score: float = 0.0
    copy_risk: float = 0.0
    rarity_score: float = 0.0
    clip_norm: float = 0.0
    sigma: float = 0.0
    perturb_norm: float = 0.0


@dataclass
class ProtectionUnit:
    doc_index: int
    start_char: int
    end_char: int
    phi_type: str
    risk_group: str
    local_text: str
    token_start: int = -1
    token_end: int = -1
    spans: list[SensitiveSpan] = field(default_factory=list)
    risk_score: float = 0.0
    utility_score: float = 0.0
    copy_risk: float = 0.0
    rarity_score: float = 0.0
    clip_norm: float = 0.0
    sigma: float = 0.0
    perturb_norm: float = 0.0
    perturb_blend: float = 0.0
    midlayer_strength: float = 0.0
