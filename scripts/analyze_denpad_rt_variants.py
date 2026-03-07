#!/usr/bin/env python3
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def load_json(path: Path):
    with path.open() as f:
        return json.load(f)


def load_jsonl(path: Path):
    with path.open() as f:
        return [json.loads(line) for line in f]


def group_audit_by_prompt(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row.get("prompt_index")].append(row)
    return grouped


def normalize_text(text):
    return " ".join(str(text).split()).strip()


def summarize_audit(rows):
    selected_source = Counter()
    selected_level = Counter()
    labels = Counter()
    same_pick = 0
    llm_selected = []
    for row in rows:
        selected_source[row.get("selected_source", "unknown")] += 1
        selected_level[row.get("selected_level", "unknown")] += 1
        labels[row.get("label", "UNKNOWN")] += 1
        if normalize_text(row.get("replacement", "")).lower() == normalize_text(row.get("entity", "")).lower():
            same_pick += 1
        if row.get("selected_source") == "llm_completion":
            llm_selected.append(
                {
                    "prompt_index": row.get("prompt_index"),
                    "entity": row.get("entity"),
                    "label": row.get("label"),
                    "replacement": row.get("replacement"),
                    "selected_level": row.get("selected_level"),
                }
            )
    return {
        "rows": len(rows),
        "same_pick": same_pick,
        "same_pick_rate": same_pick / len(rows) if rows else 0.0,
        "selected_source": dict(selected_source),
        "selected_level": dict(selected_level),
        "labels": dict(labels),
        "llm_selected": llm_selected,
    }


def compare_outputs(left_json, right_json, left_audit_rows, right_audit_rows):
    left_grouped = group_audit_by_prompt(left_audit_rows)
    right_grouped = group_audit_by_prompt(right_audit_rows)
    changed = []
    for idx, (left_row, right_row) in enumerate(zip(left_json, right_json)):
        docs_changed = left_row.get("retrieved_docs") != right_row.get("retrieved_docs")
        ans_changed = left_row.get("answer") != right_row.get("answer")
        if not docs_changed and not ans_changed:
            continue
        changed.append(
            {
                "prompt_index": idx,
                "question": left_row.get("question") or right_row.get("question"),
                "docs_changed": docs_changed,
                "answers_changed": ans_changed,
                "left_llm_selected": [
                    item for item in left_grouped.get(idx, []) if item.get("selected_source") == "llm_completion"
                ],
                "right_llm_selected": [
                    item for item in right_grouped.get(idx, []) if item.get("selected_source") == "llm_completion"
                ],
                "left_doc0": (left_row.get("retrieved_docs") or [None])[0],
                "right_doc0": (right_row.get("retrieved_docs") or [None])[0],
                "left_answer": left_row.get("answer"),
                "right_answer": right_row.get("answer"),
            }
        )
    return changed


def main():
    parser = argparse.ArgumentParser(description="Analyze DenPAD RT variant outputs and audits.")
    parser.add_argument("--left-json", required=True)
    parser.add_argument("--left-audit", required=True)
    parser.add_argument("--right-json", required=True)
    parser.add_argument("--right-audit", required=True)
    parser.add_argument("--max-changed", type=int, default=10)
    args = parser.parse_args()

    left_json = load_json(Path(args.left_json))
    right_json = load_json(Path(args.right_json))
    left_audit = load_jsonl(Path(args.left_audit))
    right_audit = load_jsonl(Path(args.right_audit))

    report = {
        "left_summary": summarize_audit(left_audit),
        "right_summary": summarize_audit(right_audit),
        "changed_samples": compare_outputs(left_json, right_json, left_audit, right_audit)[: args.max_changed],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
