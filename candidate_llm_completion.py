import json
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


@dataclass(frozen=True)
class CompletionResult:
    generalized: list[str]
    safe_related: list[str]


class CandidateLLMCompletion:
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
        self.cache: dict[tuple[str, str], CompletionResult] = {}
        if not self.enabled:
            return
        if AutoTokenizer is None or AutoConfig is None:
            LOGGER.warning("transformers is unavailable; local candidate completion is disabled.")
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
            LOGGER.warning("Failed to load local candidate completion model %s: %s", model_name, exc)
            self.enabled = False
            self.tokenizer = None
            self.model = None

    def complete(self, entity: str, entity_type: str) -> CompletionResult:
        normalized_entity = _normalize_text(entity)
        key = (entity_type.upper(), normalized_entity.lower())
        if key in self.cache:
            return self.cache[key]
        if not self.enabled or self.model is None or self.tokenizer is None:
            result = CompletionResult(generalized=[], safe_related=[])
            self.cache[key] = result
            return result

        prompt = self._build_prompt(normalized_entity, entity_type.upper())
        try:
            encoded_prompt = self._encode_prompt(prompt)
            inputs = self.tokenizer(encoded_prompt, return_tensors="pt", truncation=True, max_length=384)
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
                decoded = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            else:
                decoded = self.tokenizer.decode(output[0], skip_special_tokens=True)
            result = self._parse_output(decoded, normalized_entity)
        except Exception as exc:
            LOGGER.warning("Local candidate completion failed for %s (%s): %s", normalized_entity, entity_type, exc)
            result = CompletionResult(generalized=[], safe_related=[])
        self.cache[key] = result
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

    def _build_prompt(self, entity: str, entity_type: str) -> str:
        generalized_k = max(3, min(4, self.top_k))
        if entity_type == "DISEASE":
            guidance = (
                "Use only broad medical phrases that end with one of: condition, disorder, disease, infection, syndrome, issue. "
                "Do not output another specific diagnosis, subtype, anatomy-only phrase, or symptom-only phrase. "
                "Good examples: skin condition; endocrine disorder; bacterial infection."
            )
        elif entity_type == "DRUG":
            guidance = (
                "Use only broad treatment phrases that end with one of: medication, therapy, treatment, medicine, drug. "
                "Do not output another specific drug name, molecule name, dosage form, body part, or symptom phrase. "
                "Good examples: pain medication; hormone therapy; anti inflammatory treatment."
            )
        else:
            guidance = (
                "Use only broad safe phrases. "
                "Do not output another specific named entity."
            )
        return (
            f"Entity type: {entity_type}\n"
            f"Entity: {entity}\n"
            "Generate privacy-safe generalized replacement candidates.\n"
            f"Write exactly {generalized_k} short generalized phrases.\n"
            "Rules: short ASCII phrases, 1 to 4 words, no names, no places, no IDs, no exact copy.\n"
            "Each phrase must be broader and safer than the entity, not a synonym and not a specific subtype.\n"
            f"{guidance}\n"
            "Output exactly in this format:\n"
            "generalized: item1; item2; item3; item4"
        )

    def _encode_prompt(self, user_prompt: str) -> str:
        if self.model_family == "causal" and hasattr(self.tokenizer, "apply_chat_template"):
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You generate short privacy-safe medical replacement candidates. "
                        "Output only the requested lines and nothing else."
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

    def _parse_output(self, text: str, original: str) -> CompletionResult:
        generalized: list[str] = []
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                payload = json.loads(match.group(0))
                generalized = payload.get("generalized", []) or []
            except Exception:
                pass
        if not generalized:
            generalized = self._extract_field_values(text, "generalized")
        original_key = re.sub(r"[^a-z0-9]+", "", original.lower())
        generalized = self._clean_candidates(generalized, original_key)
        generalized = generalized[: max(3, min(4, self.top_k))]
        return CompletionResult(generalized=generalized, safe_related=[])

    def _extract_field_values(self, text: str, field_name: str) -> list[str]:
        patterns = [
            rf"{field_name}\s*:\s*(.+)",
            rf"{field_name.replace('_', '[_ ]')}\s*:\s*(.+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if not match:
                continue
            raw = match.group(1).strip()
            raw = raw.splitlines()[0].strip()
            return self._split_items(raw)

        # Fallback for bullet-like outputs such as:
        # generalized - item1 - item2
        match = re.search(rf"{field_name.replace('_', '[_ ]')}\s*[-:]\s*(.+)", text, flags=re.IGNORECASE)
        if match:
            raw = match.group(1).strip()
            raw = raw.splitlines()[0].strip()
            return self._split_items(raw)
        return []

    def _split_items(self, raw: str) -> list[str]:
        parts = re.split(r"[;,|]", raw)
        if len(parts) <= 1:
            parts = re.split(r"\s+-\s+", raw)
        return [part.strip(" -") for part in parts if part.strip(" -")]

    def _clean_candidates(self, candidates: list[str], original_key: str) -> list[str]:
        cleaned = []
        seen = set()
        for candidate in candidates:
            normalized = _normalize_text(candidate)
            if not normalized:
                continue
            if len(normalized.split()) > 4:
                continue
            if re.search(r"[^A-Za-z0-9\s/_-]", normalized):
                continue
            key = re.sub(r"[^a-z0-9]+", "", normalized.lower())
            if key in {"item1", "item2", "item3", "generalized", "saferelated", "related", "broader", "safer"}:
                continue
            if not key or key == original_key or original_key in key or key in original_key:
                continue
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(normalized)
        return cleaned
