import logging
import re
from dataclasses import dataclass
from typing import Optional

import torch

try:
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
except ImportError:
    AutoConfig = None
    AutoModelForCausalLM = None
    AutoModelForSeq2SeqLM = None
    AutoTokenizer = None


LOGGER = logging.getLogger(__name__)


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip())


def _normalize_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


@dataclass(frozen=True)
class GeneratedCandidateResult:
    generated: list[str]
    raw_output: str
    parsed_output: list[str]


@dataclass(frozen=True)
class CritiquedCandidateResult:
    approved: list[str]
    raw_output: str
    parsed_output: list[str]


class CandidateLLMCompletion:
    """Local deterministic generator/reranker for typed generalized candidates."""

    def __init__(
        self,
        model_name: Optional[str] = "Qwen/Qwen2.5-3B-Instruct",
        top_k: int = 5,
        max_new_tokens: int = 96,
        device: str = "auto",
    ) -> None:
        if isinstance(model_name, str) and model_name.strip().lower() in {"", "none", "disabled", "off"}:
            model_name = None
        self.model_name = model_name
        self.top_k = max(2, top_k)
        self.max_new_tokens = max_new_tokens
        self.device = device
        self.enabled = bool(model_name)
        self.tokenizer = None
        self.model = None
        self.model_family = "seq2seq"
        self.rerank_cache: dict[tuple[str, str, tuple[str, ...]], list[str]] = {}
        self.generate_cache: dict[tuple[str, str, tuple[str, ...]], GeneratedCandidateResult] = {}
        self.critique_cache: dict[tuple[str, str, tuple[str, ...], tuple[str, ...]], CritiquedCandidateResult] = {}
        if not self.enabled:
            return
        if AutoTokenizer is None or AutoConfig is None:
            LOGGER.warning("transformers is unavailable; local candidate reranker is disabled.")
            self.enabled = False
            return
        try:
            config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
            self.model_family = "seq2seq" if getattr(config, "is_encoder_decoder", False) else "causal"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            model_kwargs = {
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,
                "torch_dtype": self._resolve_dtype(),
            }
            if self.model_family == "seq2seq":
                if AutoModelForSeq2SeqLM is None:
                    raise ImportError("AutoModelForSeq2SeqLM is unavailable.")
                self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, **model_kwargs)
            else:
                if AutoModelForCausalLM is None:
                    raise ImportError("AutoModelForCausalLM is unavailable.")
                self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
                if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                    self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            resolved_device = self._resolve_device()
            self.model = self.model.to(resolved_device)
            self.model.eval()
            self.device = resolved_device
        except Exception as exc:
            LOGGER.warning("Failed to load local candidate reranker model %s: %s", model_name, exc)
            self.enabled = False
            self.tokenizer = None
            self.model = None

    def generate_candidates(
        self,
        entity: str,
        entity_type: str,
        family_hints: Optional[list[str]] = None,
    ) -> list[str]:
        return self.generate_candidates_debug(entity, entity_type, family_hints).generated

    def generate_candidates_debug(
        self,
        entity: str,
        entity_type: str,
        family_hints: Optional[list[str]] = None,
    ) -> GeneratedCandidateResult:
        entity = _normalize_text(entity)
        hints = [hint for hint in (family_hints or []) if hint]
        key = (entity_type.upper(), entity.lower(), tuple(hints))
        if key in self.generate_cache:
            return self.generate_cache[key]
        if not self.enabled or self.model is None or self.tokenizer is None:
            result = GeneratedCandidateResult(generated=[], raw_output="", parsed_output=[])
            self.generate_cache[key] = result
            return result
        prompt = self._build_generation_prompt(entity, entity_type.upper(), hints)
        raw_output = ""
        try:
            raw_output = self._run_prompt(prompt, max_length=448)
            parsed = self._parse_generated(raw_output)
        except Exception as exc:
            LOGGER.warning("Local candidate generation failed for %s (%s): %s", entity, entity_type, exc)
            parsed = []
        result = GeneratedCandidateResult(generated=list(parsed), raw_output=raw_output, parsed_output=list(parsed))
        self.generate_cache[key] = result
        return result

    def rerank_candidates(self, entity: str, entity_type: str, candidates: list[str]) -> list[str]:
        cleaned = self._clean_input_candidates(candidates)
        if len(cleaned) < 2:
            return cleaned
        key = (entity_type.upper(), _normalize_text(entity).lower(), tuple(cleaned))
        if key in self.rerank_cache:
            return list(self.rerank_cache[key])
        if not self.enabled or self.model is None or self.tokenizer is None:
            self.rerank_cache[key] = list(cleaned)
            return cleaned

        prompt = self._build_rerank_prompt(_normalize_text(entity), entity_type.upper(), cleaned)
        try:
            decoded = self._run_prompt(prompt, max_length=512)
            preferred = self._parse_preferred(decoded, cleaned)
        except Exception as exc:
            LOGGER.warning("Local candidate reranking failed for %s (%s): %s", entity, entity_type, exc)
            preferred = []

        if not preferred:
            ordered = cleaned
        else:
            remaining = [candidate for candidate in cleaned if candidate not in preferred]
            ordered = preferred + remaining
        self.rerank_cache[key] = list(ordered)
        return ordered

    def rerank(self, entity: str, entity_type: str, candidates: list[str]) -> list[str]:
        return self.rerank_candidates(entity, entity_type, candidates)

    def critique_candidates(
        self,
        entity: str,
        entity_type: str,
        candidates: list[str],
        family_hints: Optional[list[str]] = None,
    ) -> list[str]:
        return self.critique_candidates_debug(entity, entity_type, candidates, family_hints).approved

    def critique_candidates_debug(
        self,
        entity: str,
        entity_type: str,
        candidates: list[str],
        family_hints: Optional[list[str]] = None,
    ) -> CritiquedCandidateResult:
        cleaned = self._clean_input_candidates(candidates)
        if not cleaned:
            return CritiquedCandidateResult(approved=[], raw_output="", parsed_output=[])
        hints = [hint for hint in (family_hints or []) if hint]
        key = (entity_type.upper(), _normalize_text(entity).lower(), tuple(hints), tuple(cleaned))
        if key in self.critique_cache:
            return self.critique_cache[key]
        if not self.enabled or self.model is None or self.tokenizer is None:
            result = CritiquedCandidateResult(approved=list(cleaned), raw_output="", parsed_output=list(cleaned))
            self.critique_cache[key] = result
            return result
        prompt = self._build_critique_prompt(_normalize_text(entity), entity_type.upper(), cleaned, hints)
        raw_output = ""
        try:
            raw_output = self._run_prompt(prompt, max_length=640)
            approved = self._parse_approved(raw_output, cleaned)
        except Exception as exc:
            LOGGER.warning("Local candidate critique failed for %s (%s): %s", entity, entity_type, exc)
            approved = []
        if not approved:
            approved = list(cleaned)
        result = CritiquedCandidateResult(approved=list(approved), raw_output=raw_output, parsed_output=list(approved))
        self.critique_cache[key] = result
        return result

    def _resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _resolve_dtype(self):
        if not torch.cuda.is_available():
            return torch.float32
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16

    def _build_rerank_prompt(self, entity: str, entity_type: str, candidates: list[str]) -> str:
        bullets = "\n".join(f"{idx + 1}. {candidate}" for idx, candidate in enumerate(candidates))
        return (
            f"Entity type: {entity_type}\n"
            f"Original entity: {entity}\n"
            "Candidate generalized replacements:\n"
            f"{bullets}\n"
            "Task: choose up to 3 candidates from the list that are the safest and most natural broad replacements.\n"
            "Rules:\n"
            "- Choose only from the provided list.\n"
            "- Prefer broad medically plausible replacements.\n"
            "- Avoid another specific diagnosis or another specific drug name.\n"
            "- Avoid awkward phrases.\n"
            "Output exactly in one line:\n"
            "preferred: candidate text 1; candidate text 2; candidate text 3"
        )

    def _build_generation_prompt(self, entity: str, entity_type: str, family_hints: list[str]) -> str:
        hints_text = f"Known family hints: {', '.join(family_hints)}\n" if family_hints else ""
        if entity_type == "DISEASE":
            pattern = "... condition; ... disorder; ... disease; ... infection; ... syndrome; ... issue"
            examples = (
                "Examples:\n"
                "Entity: psoriasis -> generalized: skin condition; inflammatory skin disorder\n"
                "Entity: alopecia areata -> generalized: hair loss condition; autoimmune hair disorder\n"
                "Entity: gonorrhea -> generalized: sexually transmitted infection; bacterial infection"
            )
        elif entity_type == "DRUG":
            pattern = "... medication; ... treatment; ... therapy; ... medicine; ... drug"
            examples = (
                "Examples:\n"
                "Entity: ibuprofen -> generalized: pain medication; anti inflammatory medication\n"
                "Entity: euthyrox -> generalized: thyroid medication; hormone medication\n"
                "Entity: trileptal -> generalized: seizure medication; neurological medication"
            )
        else:
            pattern = "... condition"
            examples = "Examples:\nEntity: example -> generalized: medical condition"
        return (
            f"Entity type: {entity_type}\n"
            f"Entity: {entity}\n"
            f"{hints_text}"
            "Generate up to 3 broad, privacy-safe generalized replacements.\n"
            "Rules:\n"
            "- Do not repeat the original entity.\n"
            "- Do not output another specific diagnosis or another specific drug name.\n"
            f"- Prefer phrases shaped like: {pattern}\n"
            "- Keep phrases short and medically plausible.\n"
            f"{examples}\n"
            "Output exactly in one line:\n"
            "generalized: phrase 1; phrase 2; phrase 3"
        )

    def _build_critique_prompt(
        self,
        entity: str,
        entity_type: str,
        candidates: list[str],
        family_hints: list[str],
    ) -> str:
        hints_text = f"Family hints: {', '.join(family_hints)}\n" if family_hints else ""
        bullets = "\n".join(f"{idx + 1}. {candidate}" for idx, candidate in enumerate(candidates))
        if entity_type == "DISEASE":
            rubric = (
                "Good generalized disease replacements are broad phrases like "
                "'skin condition', 'endocrine disorder', 'infectious disease'.\n"
                "Bad replacements are another specific disease, a near-copy of the original, or an awkward phrase."
            )
        elif entity_type == "DRUG":
            rubric = (
                "Good generalized drug replacements are broad phrases like "
                "'pain medication', 'hormone medication', 'anti inflammatory treatment'.\n"
                "Bad replacements are another specific drug, a brand/generic synonym, or an awkward phrase."
            )
        else:
            rubric = "Approve only broad, safe generalized replacements."
        return (
            f"Entity type: {entity_type}\n"
            f"Original entity: {entity}\n"
            f"{hints_text}"
            f"{rubric}\n"
            "Review the candidate list and keep only candidates that are broad generalized replacements.\n"
            "Candidate list:\n"
            f"{bullets}\n"
            "Output exactly in one line:\n"
            "approved: candidate text 1; candidate text 2; candidate text 3"
        )

    def _encode_prompt(self, user_prompt: str) -> str:
        if self.model_family == "causal" and hasattr(self.tokenizer, "apply_chat_template"):
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You rank privacy-safe medical replacement candidates. "
                        "Use only the provided options. Output only the requested line."
                    ),
                },
                {"role": "user", "content": user_prompt},
            ]
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                return user_prompt + "\nassistant:\n"
        return user_prompt

    def _run_prompt(self, prompt: str, max_length: int) -> str:
        encoded_prompt = self._encode_prompt(prompt)
        inputs = self.tokenizer(encoded_prompt, return_tensors="pt", truncation=True, max_length=max_length)
        inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}
        prompt_length = int(inputs["input_ids"].shape[-1])
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                do_sample=False,
                num_beams=1,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        if self.model_family == "causal":
            generated_tokens = output[0][prompt_length:]
            return self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

    def _clean_input_candidates(self, candidates: list[str]) -> list[str]:
        cleaned = []
        seen = set()
        for candidate in candidates:
            normalized = _normalize_text(candidate)
            if not normalized:
                continue
            key = _normalize_key(normalized)
            if not key or key in seen:
                continue
            seen.add(key)
            cleaned.append(normalized)
        return cleaned[: self.top_k]

    def _parse_preferred(self, text: str, candidates: list[str]) -> list[str]:
        candidates_by_key = {_normalize_key(candidate): candidate for candidate in candidates}
        keyed_candidates = list(candidates_by_key.items())
        preferred: list[str] = []
        match = re.search(r"preferred\s*:\s*(.+)", text, flags=re.IGNORECASE)
        raw = match.group(1).strip() if match else text.strip()
        raw = raw.splitlines()[0].strip()
        parts = [part.strip(" -") for part in re.split(r"[;,|]", raw) if part.strip(" -")]
        for part in parts:
            if part.isdigit():
                idx = int(part) - 1
                if 0 <= idx < len(candidates):
                    candidate = candidates[idx]
                    if candidate not in preferred:
                        preferred.append(candidate)
                continue
            key = _normalize_key(part)
            if key in candidates_by_key:
                candidate = candidates_by_key[key]
                if candidate not in preferred:
                    preferred.append(candidate)
                continue
            for candidate_key, candidate in keyed_candidates:
                if key and (key in candidate_key or candidate_key in key):
                    if candidate not in preferred:
                        preferred.append(candidate)
                    break
        return preferred[: min(3, len(candidates))]

    def _parse_generated(self, text: str) -> list[str]:
        match = re.search(r"generalized\s*:\s*(.+)", text, flags=re.IGNORECASE)
        if match:
            raw = match.group(1).strip()
        else:
            # Fallback: many instruct models directly emit a bare semicolon/comma
            # separated list without the `generalized:` prefix.
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            raw = lines[0] if lines else text.strip()
        raw = raw.splitlines()[0].strip()
        parts = [part.strip(" -") for part in re.split(r"[;,|]", raw) if part.strip(" -")]
        if len(parts) <= 1:
            parts = [part.strip(" -") for part in re.split(r"\s+-\s+|\s*\d+\.\s*", raw) if part.strip(" -")]
        cleaned = []
        seen = set()
        for part in parts:
            normalized = _normalize_text(part)
            key = _normalize_key(normalized)
            if not key or key in seen:
                continue
            if len(normalized.split()) > 4:
                continue
            if re.search(r"[^A-Za-z0-9\s/_-]", normalized):
                continue
            seen.add(key)
            cleaned.append(normalized)
        return cleaned[:3]

    def _parse_approved(self, text: str, candidates: list[str]) -> list[str]:
        match = re.search(r"approved\s*:\s*(.+)", text, flags=re.IGNORECASE)
        raw = match.group(1).strip() if match else text.strip()
        raw = raw.splitlines()[0].strip()
        parts = [part.strip(" -") for part in re.split(r"[;,|]", raw) if part.strip(" -")]
        candidate_map = {_normalize_key(candidate): candidate for candidate in candidates}
        approved: list[str] = []
        for part in parts:
            key = _normalize_key(part)
            if key in candidate_map:
                candidate = candidate_map[key]
                if candidate not in approved:
                    approved.append(candidate)
                continue
            for candidate_key, candidate in candidate_map.items():
                if key and (key in candidate_key or candidate_key in key):
                    if candidate not in approved:
                        approved.append(candidate)
                    break
        return approved[: len(candidates)]
