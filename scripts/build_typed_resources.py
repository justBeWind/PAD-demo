import argparse
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from difflib import SequenceMatcher


BUILDER_VERSION = "2.1.0"
SCHEMA_VERSION = "typed_resource_v2"
RELATED_TOP_K = 3
RELATED_MIN_SCORE = 0.18


DISEASE_GENERALIZATION_MAP = {
    "alopecia areata": ["hair loss condition", "autoimmune hair disorder"],
    "alopecia": ["hair loss condition"],
    "androgenetic alopecia": ["hair loss condition", "hair loss disorder"],
    "gonorrhea": ["sexually transmitted infection", "bacterial infection"],
    "helicobacter pylori infection": ["stomach infection", "gastrointestinal infection"],
    "h pylori": ["stomach infection", "gastrointestinal infection"],
    "hpylori": ["stomach infection", "gastrointestinal infection"],
    "graves disease": ["thyroid disorder", "endocrine disorder"],
    "hyperthyroidism": ["thyroid disorder", "endocrine disorder"],
    "hyperthyroid": ["thyroid disorder", "endocrine disorder"],
    "addison disease": ["adrenal disorder", "endocrine disorder"],
    "the addison s": ["adrenal disorder", "endocrine disorder"],
    "thyroid cancer": ["thyroid disorder", "cancer condition"],
    "trigeminal neuralgia": ["neurological pain disorder", "nerve pain condition"],
    "deep vein thrombosis": ["vascular condition", "circulatory disorder"],
    "dvt": ["vascular condition", "circulatory disorder"],
    "psoriasis": ["skin condition", "inflammatory skin disorder"],
    "anxiety": ["mental health condition", "anxiety disorder"],
}


DRUG_GENERALIZATION_MAP = {
    "levothyroxine": ["thyroid medication", "hormone replacement therapy"],
    "euthyrox": ["thyroid medication", "hormone replacement therapy"],
    "oxcarbazepine": ["antiepileptic medication", "neurological medication"],
    "trileptal": ["antiepileptic medication", "neurological medication"],
    "pregabalin": ["nerve pain medication", "neurological medication"],
    "lyrica": ["nerve pain medication", "neurological medication"],
    "sertraline": ["antidepressant medication", "psychiatric medication"],
    "zoloft": ["antidepressant medication", "psychiatric medication"],
    "ibuprofen": ["pain medication", "anti inflammatory medication"],
    "aspirin": ["pain medication", "anti inflammatory medication"],
    "acetaminophen": ["pain medication"],
    "paracetamol": ["pain medication"],
    "prednisone": ["steroid medication", "anti inflammatory medication"],
    "prednisolone": ["steroid medication", "anti inflammatory medication"],
    "methylprednisolone": ["steroid medication", "anti inflammatory medication"],
}

FAMILY_TEMPLATES = {
    "DISEASE": {
        "dermatologic": ["skin condition", "dermatologic disorder"],
        "endocrine": ["endocrine disorder", "hormonal condition"],
        "neurological": ["neurological condition", "nerve disorder"],
        "respiratory": ["respiratory condition", "lung disorder"],
        "digestive": ["digestive disorder", "gastrointestinal condition"],
        "circulatory": ["circulatory disorder", "vascular condition"],
        "infectious": ["infectious disease", "bacterial infection"],
        "autoimmune": ["autoimmune disorder", "immune condition"],
        "mental_health": ["mental health condition", "psychiatric disorder"],
        "musculoskeletal": ["musculoskeletal disorder", "joint condition"],
        "urinary": ["urinary condition", "urinary disorder"],
        "hepatic": ["liver disorder", "hepatic condition"],
        "oncology": ["cancer condition", "medical condition"],
        "general_medical": ["medical condition", "health disorder"],
    },
    "DRUG": {
        "pain_relief": ["pain medication", "symptom relief medication"],
        "endocrine": ["hormone medication", "endocrine treatment"],
        "neurological": ["neurological medication", "nerve treatment"],
        "psychiatric": ["psychiatric medication", "mental health treatment"],
        "anti_inflammatory": ["anti inflammatory medication", "anti inflammatory treatment"],
        "anti_infective": ["anti infective medication", "infection treatment"],
        "cardiovascular": ["cardiovascular medication", "heart treatment"],
        "respiratory": ["respiratory medication", "breathing treatment"],
        "gastrointestinal": ["digestive medication", "gastrointestinal treatment"],
        "dermatologic": ["skin medication", "dermatologic treatment"],
        "general_therapy": ["medical treatment", "prescription medication"],
    },
}

DISEASE_FAMILY_HINTS = {
    "oncology": "oncology",
    "thyroid": "endocrine",
    "endocrine": "endocrine",
    "dermatology": "dermatologic",
    "hair": "dermatologic",
    "infectious": "infectious",
    "sexual_health": "infectious",
    "neurology": "neurological",
    "pain": "musculoskeletal",
    "vascular": "circulatory",
    "hematology": "circulatory",
    "gastrointestinal": "digestive",
    "hepatic": "hepatic",
    "psychiatric": "mental_health",
    "pulmonary": "respiratory",
    "autoimmune": "autoimmune",
}

DRUG_FAMILY_HINTS = {
    "ssri": "psychiatric",
    "psychiatric": "psychiatric",
    "oncology": "general_therapy",
    "tki": "general_therapy",
    "antibiotic": "anti_infective",
    "antifungal": "anti_infective",
    "dermatology": "dermatologic",
    "antidiabetic": "endocrine",
    "endocrine": "endocrine",
    "analgesic": "pain_relief",
    "pain": "pain_relief",
    "steroid": "anti_inflammatory",
    "anti-inflammatory": "anti_inflammatory",
    "neurology": "neurological",
    "antiepileptic": "neurological",
    "thyroid": "endocrine",
}


def normalize_text(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def slugify(text: str) -> str:
    text = normalize_text(text).lower()
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text or "unknown"


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_jsonl_records(path: str) -> list[dict | str]:
    items: list[dict | str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parsed = json.loads(line)
            if isinstance(parsed, dict) and "items" in parsed and isinstance(parsed["items"], list):
                for item in parsed["items"]:
                    if isinstance(item, (dict, str)):
                        items.append(item)
                continue
            if isinstance(parsed, (dict, str)):
                items.append(parsed)
    return items


def _parse_obo_synonym(line: str) -> str:
    match = re.search(r'"([^"]+)"', line)
    return normalize_text(match.group(1)) if match else ""


def _load_doid_obo_records(path: str) -> list[dict]:
    items: list[dict] = []
    current: dict | None = None
    in_term = False

    def flush_current() -> None:
        nonlocal current
        if not current:
            return
        if current.get("is_obsolete"):
            current = None
            return
        term = normalize_text(current.get("term", ""))
        if not term:
            current = None
            return
        aliases = []
        seen_aliases = set()
        for alias in current.get("aliases", []):
            norm = normalize_text(alias)
            lowered = norm.lower()
            if norm and lowered != term.lower() and lowered not in seen_aliases:
                seen_aliases.add(lowered)
                aliases.append(norm)
        tags = normalize_tags(current.get("tags", []))
        item = {
            "canonical_id": normalize_text(current.get("canonical_id", "")),
            "term": term,
            "aliases": aliases,
            "related": [],
            "generalized": [],
            "family": [],
            "tags": tags,
        }
        items.append(item)
        current = None

    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                if in_term:
                    flush_current()
                    in_term = False
                continue

            if line == "[Term]":
                if in_term:
                    flush_current()
                current = {"aliases": [], "tags": []}
                in_term = True
                continue

            if line.startswith("["):
                if in_term:
                    flush_current()
                    in_term = False
                continue

            if not in_term or current is None:
                continue

            if line.startswith("id:"):
                current["canonical_id"] = normalize_text(line.split(":", 1)[1])
            elif line.startswith("name:"):
                current["term"] = normalize_text(line.split(":", 1)[1])
            elif line.startswith("synonym:"):
                synonym = _parse_obo_synonym(line)
                if synonym:
                    current.setdefault("aliases", []).append(synonym)
            elif line.startswith("is_a:"):
                parent = normalize_text(line.split("!", 1)[-1] if "!" in line else line.split(":", 1)[1])
                if parent:
                    current.setdefault("tags", []).append(parent)
            elif line.startswith("subset:"):
                subset = normalize_text(line.split(":", 1)[1])
                if subset:
                    current.setdefault("tags", []).append(subset)
            elif line.startswith("namespace:"):
                namespace = normalize_text(line.split(":", 1)[1])
                if namespace:
                    current.setdefault("tags", []).append(namespace)
            elif line.startswith("is_obsolete:"):
                current["is_obsolete"] = line.endswith("true")

    if in_term:
        flush_current()

    return items


def infer_source_format(path: str, explicit_format: str | None = None) -> str:
    if explicit_format and explicit_format != "auto":
        return explicit_format
    lowered = path.lower()
    if lowered.endswith(".jsonl"):
        return "jsonl"
    if lowered.endswith(".json"):
        return "json"
    if lowered.endswith(".obo"):
        return "doid_obo"
    raise ValueError(f"Unable to infer source format from path: {path}")


def load_seed_records(path: str, source_format: str = "auto") -> list[dict | str]:
    resolved_format = infer_source_format(path, source_format)
    if resolved_format == "doid_obo":
        return _load_doid_obo_records(path)
    if path.lower().endswith(".jsonl"):
        return _load_jsonl_records(path)

    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, dict):
        items = payload.get("items", [])
    elif isinstance(payload, list):
        items = payload
    else:
        raise ValueError("Unsupported seed format: expected list or {'items': ...}")
    return [item for item in items if isinstance(item, (dict, str))]


def normalize_tags(tags) -> list[str]:
    cleaned = []
    for tag in tags or []:
        norm = normalize_text(tag).lower()
        if norm and norm not in cleaned:
            cleaned.append(norm)
    return cleaned


def normalize_record(item: dict | str, prefix: str) -> dict | None:
    if isinstance(item, str):
        term = normalize_text(item)
        if not term:
            return None
        return {
            "term": term,
            "aliases": [],
            "related": [],
            "generalized": [],
            "tags": [],
            "canonical_id": f"{prefix}:{slugify(term)}",
        }

    term = normalize_text(item.get("term", ""))
    if not term:
        return None
    aliases = []
    for alias in item.get("aliases", []):
        norm = normalize_text(alias)
        if norm and norm.lower() != term.lower() and norm not in aliases:
            aliases.append(norm)
    related = []
    for candidate in item.get("related", []):
        norm = normalize_text(candidate)
        if norm and norm.lower() != term.lower() and norm not in related:
            related.append(norm)
    generalized = []
    for candidate in item.get("generalized", []):
        norm = normalize_text(candidate)
        if norm and norm.lower() != term.lower() and norm not in generalized:
            generalized.append(norm)
    tags = normalize_tags(item.get("tags", []))
    family = normalize_tags(item.get("family", []))
    canonical_id = normalize_text(item.get("canonical_id", "")) or f"{prefix}:{slugify(term)}"
    return {
        "term": term,
        "aliases": aliases,
        "related": related,
        "generalized": generalized,
        "family": family,
        "tags": tags,
        "canonical_id": canonical_id,
    }


def _merge_unique(candidates: list[str], term_lower: str, limit: int = 3) -> list[str]:
    filtered = []
    for candidate in candidates:
        candidate = normalize_text(candidate)
        if not candidate or candidate.lower() == term_lower:
            continue
        if candidate not in filtered:
            filtered.append(candidate)
    return filtered[:limit]


def _lookup_seeded_generalizations(term_lower: str, aliases: list[str], mapping: dict[str, list[str]]) -> list[str]:
    values = []
    for key in [term_lower, *[normalize_text(alias).lower() for alias in aliases]]:
        if key in mapping:
            values.extend(mapping[key])
    return values


def _merge_families(families: list[str]) -> list[str]:
    out = []
    seen = set()
    for family in families:
        norm = normalize_text(family).lower()
        if norm and norm not in seen:
            seen.add(norm)
            out.append(norm)
    return out


def infer_families(prefix: str, term: str, aliases: list[str], tags: list[str], existing_family: list[str]) -> list[str]:
    term_lower = normalize_text(term).lower()
    families = list(existing_family or [])
    tag_map = DISEASE_FAMILY_HINTS if prefix == "disease" else DRUG_FAMILY_HINTS
    lexical_rules = {
        "disease": {
            "cancer": "oncology",
            "alopecia": "dermatologic",
            "psoriasis": "dermatologic",
            "dermatitis": "dermatologic",
            "gonorr": "infectious",
            "infection": "infectious",
            "hepat": "hepatic",
            "gastr": "digestive",
            "bronch": "respiratory",
            "lung": "respiratory",
            "heart": "circulatory",
            "vascular": "circulatory",
            "thyroid": "endocrine",
            "graves": "endocrine",
            "hyperthy": "endocrine",
            "anxiety": "mental_health",
            "depression": "mental_health",
            "neuralgia": "neurological",
            "dvt": "circulatory",
            "arthritis": "musculoskeletal",
        },
        "drug": {
            "ibuprofen": "pain_relief",
            "aspirin": "pain_relief",
            "acetaminophen": "pain_relief",
            "paracetamol": "pain_relief",
            "predni": "anti_inflammatory",
            "methylpred": "anti_inflammatory",
            "euthyrox": "endocrine",
            "thyrox": "endocrine",
            "levothyrox": "endocrine",
            "trileptal": "neurological",
            "oxcarbazepine": "neurological",
            "pregabalin": "neurological",
            "gabapentin": "neurological",
            "sertraline": "psychiatric",
            "fluoxetine": "psychiatric",
            "escitalopram": "psychiatric",
            "cillin": "anti_infective",
            "azole": "anti_infective",
        },
    }
    for tag in tags:
        family = tag_map.get(tag)
        if family:
            families.append(family)
    for needle, family in lexical_rules[prefix].items():
        if needle in term_lower or any(needle in normalize_text(alias).lower() for alias in aliases):
            families.append(family)
    if not families:
        families.append("general_medical" if prefix == "disease" else "general_therapy")
    return _merge_families(families)


def build_disease_generalized(term: str, aliases: list[str], tags: list[str], family: list[str]) -> list[str]:
    term_lower = normalize_text(term).lower()
    generalized = []
    for family_name in family:
        for candidate in FAMILY_TEMPLATES["DISEASE"].get(family_name, []):
            if candidate not in generalized:
                generalized.append(candidate)

    generalized.extend(_lookup_seeded_generalizations(term_lower, aliases, DISEASE_GENERALIZATION_MAP))

    if "cancer" in term_lower and "cancer condition" not in generalized:
        generalized.append("cancer condition")
    if "alopecia" in term_lower and "hair loss condition" not in generalized:
        generalized.append("hair loss condition")
    if "gonorr" in term_lower and "sexually transmitted infection" not in generalized:
        generalized.append("sexually transmitted infection")
    if "thyroid" in term_lower and "thyroid disorder" not in generalized:
        generalized.append("thyroid disorder")
    if "neuralgia" in term_lower and "neurological pain disorder" not in generalized:
        generalized.append("neurological pain disorder")
    if "anxiety" in term_lower and "mental health condition" not in generalized:
        generalized.append("mental health condition")
    if "arthritis" in term_lower and "joint condition" not in generalized:
        generalized.append("joint condition")
    if "fibrosis" in term_lower and "respiratory condition" not in generalized:
        generalized.append("respiratory condition")
    if "infection" in term_lower and "infectious condition" not in generalized:
        generalized.append("infectious condition")
    return _merge_unique(generalized, term_lower)


def build_drug_generalized(term: str, aliases: list[str], tags: list[str], family: list[str]) -> list[str]:
    term_lower = normalize_text(term).lower()
    generalized = []
    for family_name in family:
        for candidate in FAMILY_TEMPLATES["DRUG"].get(family_name, []):
            if candidate not in generalized:
                generalized.append(candidate)

    generalized.extend(_lookup_seeded_generalizations(term_lower, aliases, DRUG_GENERALIZATION_MAP))

    if "thyrox" in term_lower or "levothyrox" in term_lower:
        generalized.extend(["thyroid medication", "hormone replacement therapy"])
    if term_lower in {"sertraline", "fluoxetine", "escitalopram"}:
        generalized.extend(["antidepressant medication", "psychiatric medication"])
    if term_lower in {"oxcarbazepine", "carbamazepine", "pregabalin", "gabapentin"}:
        generalized.extend(["neurological medication", "antiepileptic medication"])
    if term_lower in {"ibuprofen", "acetaminophen", "paracetamol", "aspirin"}:
        generalized.extend(["pain medication"])
    if "predni" in term_lower or "methylpred" in term_lower:
        generalized.extend(["steroid medication", "anti inflammatory medication"])
    if "azole" in term_lower and "antifungal medication" not in generalized:
        generalized.append("antifungal medication")
    if "cillin" in term_lower and "antibiotic medication" not in generalized:
        generalized.append("antibiotic medication")
    return _merge_unique(generalized, term_lower)


def build_related(records: list[dict], top_k: int = RELATED_TOP_K, min_score: float = RELATED_MIN_SCORE) -> list[dict]:
    for index, record in enumerate(records):
        if record["related"]:
            continue
        scored = []
        record_tags = set(record["tags"])
        for other_index, other in enumerate(records):
            if index == other_index:
                continue
            other_tags = set(other["tags"])
            if not record_tags.intersection(other_tags):
                continue
            lexical = SequenceMatcher(None, record["term"].lower(), other["term"].lower()).ratio()
            overlap = len(record_tags.intersection(other_tags)) / max(len(record_tags.union(other_tags)), 1)
            score = lexical * 0.35 + overlap * 0.65
            scored.append((other["term"], score))
        scored.sort(key=lambda item: item[1], reverse=True)
        record["related"] = [term for term, score in scored[:top_k] if score >= min_score]
    return records


def _merge_unique_list(base: list[str], extension: list[str], avoid_term: str) -> list[str]:
    out = list(base)
    lowered_avoid = avoid_term.lower()
    seen = {item.lower() for item in out}
    for item in extension:
        norm = normalize_text(item)
        if not norm:
            continue
        lowered = norm.lower()
        if lowered == lowered_avoid or lowered in seen:
            continue
        out.append(norm)
        seen.add(lowered)
    return out


def dedupe_records(records: list[dict]) -> tuple[list[dict], dict]:
    by_term: dict[str, dict] = {}
    stats = {
        "normalized_records": 0,
        "duplicates_merged": 0,
    }
    for record in records:
        stats["normalized_records"] += 1
        key = record["term"].lower()
        if key not in by_term:
            by_term[key] = dict(record)
            continue
        stats["duplicates_merged"] += 1
        existing = by_term[key]
        existing["aliases"] = _merge_unique_list(existing.get("aliases", []), record.get("aliases", []), existing["term"])
        existing["related"] = _merge_unique_list(existing.get("related", []), record.get("related", []), existing["term"])
        existing["generalized"] = _merge_unique_list(existing.get("generalized", []), record.get("generalized", []), existing["term"])
        existing["family"] = _merge_unique_list(existing.get("family", []), record.get("family", []), existing["term"])
        existing_tags = list(existing.get("tags", []))
        for tag in record.get("tags", []):
            tag_norm = normalize_text(tag).lower()
            if tag_norm and tag_norm not in existing_tags:
                existing_tags.append(tag_norm)
        existing["tags"] = existing_tags
        if not existing.get("canonical_id"):
            existing["canonical_id"] = record.get("canonical_id", existing["canonical_id"])
    return sorted(by_term.values(), key=lambda item: item["term"].lower()), stats


def export_index(records: list[dict], output_path: str, metadata: dict) -> dict:
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "metadata": metadata,
        "items": records,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build typed public resource indices for DenPAD.")
    parser.add_argument("--disease_source", type=str, help="Seed JSON or JSONL-compatible disease source.")
    parser.add_argument(
        "--disease_source_format",
        type=str,
        default="auto",
        choices=["auto", "json", "jsonl", "doid_obo"],
        help="Disease source format. Use doid_obo for Disease Ontology .obo files.",
    )
    parser.add_argument("--disease_output", type=str, help="Output JSON path for the disease typed index.")
    parser.add_argument("--disease_source_name", type=str, default="unspecified", help="Human-readable source name for disease provenance.")
    parser.add_argument("--disease_source_url", type=str, default="", help="Source URL for disease provenance.")
    parser.add_argument("--disease_source_license", type=str, default="", help="Source license for disease provenance.")
    parser.add_argument("--disease_source_version", type=str, default="", help="Source version/date for disease provenance.")
    parser.add_argument("--drug_source", type=str, help="Seed JSON or JSONL-compatible drug source.")
    parser.add_argument(
        "--drug_source_format",
        type=str,
        default="auto",
        choices=["auto", "json", "jsonl"],
        help="Drug source format.",
    )
    parser.add_argument("--drug_output", type=str, help="Output JSON path for the drug typed index.")
    parser.add_argument("--drug_source_name", type=str, default="unspecified", help="Human-readable source name for drug provenance.")
    parser.add_argument("--drug_source_url", type=str, default="", help="Source URL for drug provenance.")
    parser.add_argument("--drug_source_license", type=str, default="", help="Source license for drug provenance.")
    parser.add_argument("--drug_source_version", type=str, default="", help="Source version/date for drug provenance.")
    parser.add_argument(
        "--manifest_output",
        type=str,
        default=None,
        help="Optional output path for typed resource manifest. Defaults to <output_dir>/typed_resource_manifest.json.",
    )
    parser.add_argument(
        "--family_templates_output",
        type=str,
        default=None,
        help="Optional output path for family template JSON. Defaults to <output_dir>/family_templates.json.",
    )
    return parser.parse_args()


def source_metadata_from_args(args: argparse.Namespace, prefix: str) -> dict:
    return {
        "name": normalize_text(getattr(args, f"{prefix}_source_name", "unspecified")),
        "url": normalize_text(getattr(args, f"{prefix}_source_url", "")),
        "license": normalize_text(getattr(args, f"{prefix}_source_license", "")),
        "version": normalize_text(getattr(args, f"{prefix}_source_version", "")),
    }


def build_index(
    source_path: str,
    output_path: str,
    prefix: str,
    source_format: str = "auto",
    source_metadata: dict | None = None,
) -> dict:
    resolved_format = infer_source_format(source_path, source_format)
    seed_items = load_seed_records(source_path, source_format=resolved_format)
    raw_records = []
    dropped_empty = 0
    for item in seed_items:
        record = normalize_record(item, prefix=prefix)
        if record is None:
            dropped_empty += 1
            continue
        raw_records.append(record)

    records, dedupe_stats = dedupe_records(raw_records)
    generalized_generated = 0
    family_generated = 0
    for record in records:
        if not record.get("family"):
            record["family"] = infer_families(prefix, record["term"], record["aliases"], record["tags"], record.get("family", []))
            if record["family"]:
                family_generated += 1
        if not record["generalized"]:
            if prefix == "disease":
                record["generalized"] = build_disease_generalized(
                    record["term"], record["aliases"], record["tags"], record["family"]
                )
            else:
                record["generalized"] = build_drug_generalized(
                    record["term"], record["aliases"], record["tags"], record["family"]
                )
            if record["generalized"]:
                generalized_generated += 1

    skip_related_builder = resolved_format == "doid_obo"
    if skip_related_builder:
        related_filled = 0
    else:
        related_missing_before = sum(1 for record in records if not record.get("related"))
        records = build_related(records, top_k=RELATED_TOP_K, min_score=RELATED_MIN_SCORE)
        related_filled = sum(1 for record in records if record.get("related")) - (len(records) - related_missing_before)
        related_filled = max(0, related_filled)

    source_sha = sha256_file(source_path)
    metadata = {
        "resource_type": prefix.upper(),
        "builder_version": BUILDER_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_path": source_path,
        "source_sha256": source_sha,
        "source_item_count": len(seed_items),
        "record_count": len(records),
        "source_metadata": source_metadata or {},
        "build_stats": {
            "raw_items": len(seed_items),
            "dropped_empty": dropped_empty,
            "normalized_records": dedupe_stats["normalized_records"],
            "duplicates_merged": dedupe_stats["duplicates_merged"],
            "family_generated": family_generated,
            "generalized_generated": generalized_generated,
            "related_filled": related_filled,
        },
        "normalization_policy": {
            "whitespace_normalization": True,
            "dedupe_case_insensitive": True,
            "ascii_slug_canonical_ids": True,
        },
        "related_builder": {
            "top_k": RELATED_TOP_K,
            "min_score": RELATED_MIN_SCORE,
            "lexical_weight": 0.35,
            "tag_overlap_weight": 0.65,
            "skipped": skip_related_builder,
        },
    }
    export_index(records, output_path, metadata)
    output_sha = sha256_file(output_path)
    return {
        "resource_type": prefix.upper(),
        "output_path": output_path,
        "output_sha256": output_sha,
        "record_count": len(records),
        "source_path": source_path,
        "source_sha256": source_sha,
        "source_item_count": len(seed_items),
        "source_format": resolved_format,
        "source_metadata": source_metadata or {},
        "build_stats": metadata["build_stats"],
    }


def write_family_templates(output_path: str) -> None:
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "builder_version": BUILDER_VERSION,
        "templates": FAMILY_TEMPLATES,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def write_manifest(manifest_path: str, entries: list[dict]) -> None:
    output_dir = os.path.dirname(manifest_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "manifest_version": "1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "builder": {
            "script": os.path.abspath(__file__),
            "builder_version": BUILDER_VERSION,
        },
        "resources": entries,
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def _default_manifest_path(args: argparse.Namespace) -> str:
    if args.manifest_output:
        return args.manifest_output
    out_paths = [p for p in (args.disease_output, args.drug_output) if p]
    base_dir = os.path.dirname(out_paths[0]) if out_paths else os.getcwd()
    return os.path.join(base_dir, "typed_resource_manifest.json")


def _default_family_templates_path(args: argparse.Namespace) -> str:
    if args.family_templates_output:
        return args.family_templates_output
    out_paths = [p for p in (args.disease_output, args.drug_output) if p]
    base_dir = os.path.dirname(out_paths[0]) if out_paths else os.getcwd()
    return os.path.join(base_dir, "family_templates.json")


def main() -> None:
    args = parse_args()
    if not any([args.disease_source, args.drug_source]):
        raise SystemExit("At least one of --disease_source or --drug_source must be provided.")

    if bool(args.disease_source) ^ bool(args.disease_output):
        raise SystemExit("Both --disease_source and --disease_output must be provided together.")
    if bool(args.drug_source) ^ bool(args.drug_output):
        raise SystemExit("Both --drug_source and --drug_output must be provided together.")

    entries: list[dict] = []
    if args.disease_source:
        result = build_index(
            args.disease_source,
            args.disease_output,
            prefix="disease",
            source_format=args.disease_source_format,
            source_metadata=source_metadata_from_args(args, "disease"),
        )
        entries.append(result)
        print(f"Wrote {result['record_count']} disease records to {args.disease_output}")
    if args.drug_source:
        result = build_index(
            args.drug_source,
            args.drug_output,
            prefix="drug",
            source_format=args.drug_source_format,
            source_metadata=source_metadata_from_args(args, "drug"),
        )
        entries.append(result)
        print(f"Wrote {result['record_count']} drug records to {args.drug_output}")

    manifest_path = _default_manifest_path(args)
    write_manifest(manifest_path, entries)
    print(f"Wrote typed resource manifest to {manifest_path}")
    family_templates_path = _default_family_templates_path(args)
    write_family_templates(family_templates_path)
    print(f"Wrote family templates to {family_templates_path}")


if __name__ == "__main__":
    main()
