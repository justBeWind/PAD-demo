import argparse
import json
import os
import re
from difflib import SequenceMatcher


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


def normalize_text(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def slugify(text: str) -> str:
    text = normalize_text(text).lower()
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text or "unknown"


def load_seed_records(path: str) -> list[dict]:
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
    canonical_id = normalize_text(item.get("canonical_id", "")) or f"{prefix}:{slugify(term)}"
    return {
        "term": term,
        "aliases": aliases,
        "related": related,
        "generalized": generalized,
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


def build_disease_generalized(term: str, aliases: list[str], tags: list[str]) -> list[str]:
    term_lower = normalize_text(term).lower()
    generalized = []

    tag_templates = {
        "oncology": "cancer condition",
        "thyroid": "thyroid disorder",
        "endocrine": "endocrine disorder",
        "dermatology": "skin condition",
        "hair": "hair loss condition",
        "infectious": "infectious condition",
        "sexual_health": "sexually transmitted infection",
        "neurology": "neurological condition",
        "pain": "pain disorder",
        "vascular": "vascular condition",
        "hematology": "blood disorder",
        "gastrointestinal": "gastrointestinal condition",
        "hepatic": "liver condition",
        "psychiatric": "mental health condition",
        "pulmonary": "respiratory condition",
    }

    for tag in tags:
        candidate = tag_templates.get(tag)
        if candidate and candidate not in generalized:
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


def build_drug_generalized(term: str, aliases: list[str], tags: list[str]) -> list[str]:
    term_lower = normalize_text(term).lower()
    generalized = []

    tag_templates = {
        "ssri": "antidepressant medication",
        "psychiatric": "psychiatric medication",
        "oncology": "oncology medication",
        "tki": "targeted cancer therapy",
        "antibiotic": "antibiotic medication",
        "antifungal": "antifungal medication",
        "dermatology": "skin treatment",
        "antidiabetic": "diabetes medication",
        "endocrine": "hormone or endocrine medication",
        "analgesic": "pain medication",
        "pain": "pain medication",
        "steroid": "steroid medication",
        "anti-inflammatory": "anti inflammatory medication",
        "neurology": "neurological medication",
        "antiepileptic": "antiepileptic medication",
        "thyroid": "thyroid medication",
    }

    for tag in tags:
        candidate = tag_templates.get(tag)
        if candidate and candidate not in generalized:
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


def build_related(records: list[dict], top_k: int = 3) -> list[dict]:
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
        record["related"] = [term for term, score in scored[:top_k] if score >= 0.18]
    return records


def export_index(records: list[dict], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    payload = {"items": records}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build typed public resource indices for DenPAD.")
    parser.add_argument("--disease_source", type=str, help="Seed JSON or JSONL-compatible disease source.")
    parser.add_argument("--disease_output", type=str, help="Output JSON path for the disease typed index.")
    parser.add_argument("--drug_source", type=str, help="Seed JSON or JSONL-compatible drug source.")
    parser.add_argument("--drug_output", type=str, help="Output JSON path for the drug typed index.")
    return parser.parse_args()


def build_index(source_path: str, output_path: str, prefix: str) -> int:
    seed_items = load_seed_records(source_path)
    records = []
    for item in seed_items:
        record = normalize_record(item, prefix=prefix)
        if record is None:
            continue
        if not record["generalized"]:
            if prefix == "disease":
                record["generalized"] = build_disease_generalized(
                    record["term"], record["aliases"], record["tags"]
                )
            else:
                record["generalized"] = build_drug_generalized(
                    record["term"], record["aliases"], record["tags"]
                )
        records.append(record)
    records = build_related(records)
    export_index(records, output_path)
    return len(records)


def main() -> None:
    args = parse_args()
    if not any([args.disease_source, args.drug_source]):
        raise SystemExit("At least one of --disease_source or --drug_source must be provided.")

    if bool(args.disease_source) ^ bool(args.disease_output):
        raise SystemExit("Both --disease_source and --disease_output must be provided together.")
    if bool(args.drug_source) ^ bool(args.drug_output):
        raise SystemExit("Both --drug_source and --drug_output must be provided together.")

    if args.disease_source:
        count = build_index(args.disease_source, args.disease_output, prefix="disease")
        print(f"Wrote {count} disease resource records to {args.disease_output}")
    if args.drug_source:
        count = build_index(args.drug_source, args.drug_output, prefix="drug")
        print(f"Wrote {count} drug resource records to {args.drug_output}")


if __name__ == "__main__":
    main()
