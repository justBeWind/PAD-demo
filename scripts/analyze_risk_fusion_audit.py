import argparse
import json
from collections import Counter, defaultdict
from statistics import mean


def load_results(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, list):
        raise ValueError("Expected a JSON list of result entries.")
    return payload


def load_audit_jsonl(path: str) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def bucketize(value: float) -> str:
    if value < 0.2:
        return "[0.0,0.2)"
    if value < 0.4:
        return "[0.2,0.4)"
    if value < 0.6:
        return "[0.4,0.6)"
    if value < 0.8:
        return "[0.6,0.8)"
    return "[0.8,1.0]"


def summarize_results(results: list[dict]) -> dict:
    epsilon_values = []
    lambda_values = []
    generation_times = []
    span_counts = []
    view_summary_totals: dict[str, Counter] = defaultdict(Counter)

    for entry in results:
        epsilon = entry.get("epsilon_dp")
        if epsilon is not None:
            epsilon_values.append(float(epsilon))
        gamma = entry.get("gamma_dp")
        if gamma is not None:
            lambda_values.append(float(gamma))
        if entry.get("generation_time_sec") is not None:
            generation_times.append(float(entry["generation_time_sec"]))
        span_counts.append(int(entry.get("denpad_num_entities", 0)))
        for group_name, stats in (entry.get("denpad_view_summaries") or {}).items():
            view_summary_totals[group_name]["span_count"] += int(stats.get("span_count", 0))
            view_summary_totals[group_name]["samples"] += 1
            view_summary_totals[group_name]["risk_sum"] += float(stats.get("avg_risk", 0.0))
            view_summary_totals[group_name]["utility_sum"] += float(stats.get("avg_utility", 0.0))

    group_summaries = {}
    for group_name, stats in view_summary_totals.items():
        samples = max(stats["samples"], 1)
        group_summaries[group_name] = {
            "avg_span_count_per_sample": stats["span_count"] / samples,
            "avg_risk": stats["risk_sum"] / samples,
            "avg_utility": stats["utility_sum"] / samples,
        }

    return {
        "num_samples": len(results),
        "epsilon_mean": mean(epsilon_values) if epsilon_values else None,
        "epsilon_max": max(epsilon_values) if epsilon_values else None,
        "avg_lambda": mean(lambda_values) if lambda_values else None,
        "avg_generation_time_sec": mean(generation_times) if generation_times else None,
        "avg_detected_spans": mean(span_counts) if span_counts else 0.0,
        "view_summary": group_summaries,
    }


def summarize_audit(records: list[dict]) -> dict:
    by_label = Counter()
    by_group = Counter()
    risk_bins = Counter()
    utility_bins = Counter()
    top_entities = Counter()

    for record in records:
        label = record.get("label", "UNKNOWN")
        group = record.get("group", "UNKNOWN")
        risk = float(record.get("risk_score", 0.0))
        utility = float(record.get("utility_score", 0.0))
        entity = str(record.get("entity", "")).strip().lower()

        by_label[label] += 1
        by_group[group] += 1
        risk_bins[bucketize(risk)] += 1
        utility_bins[bucketize(utility)] += 1
        if entity:
            top_entities[entity] += 1

    return {
        "num_audit_records": len(records),
        "count_by_label": dict(by_label),
        "count_by_group": dict(by_group),
        "risk_bins": dict(risk_bins),
        "utility_bins": dict(utility_bins),
        "top_entities": top_entities.most_common(20),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate DenPAD-RF / ContextPAD audit outputs.")
    parser.add_argument("--results_file", type=str, required=True, help="Path to result JSON produced by generate.py")
    parser.add_argument("--audit_jsonl", type=str, default=None, help="Optional audit JSONL file emitted during generation")
    parser.add_argument("--output_file", type=str, default=None, help="Optional path to write the aggregated JSON report")
    args = parser.parse_args()

    report = {"results_summary": summarize_results(load_results(args.results_file))}
    if args.audit_jsonl:
        report["audit_summary"] = summarize_audit(load_audit_jsonl(args.audit_jsonl))

    payload = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output_file:
        with open(args.output_file, "w", encoding="utf-8") as f:
            f.write(payload)
    print(payload)


if __name__ == "__main__":
    main()
