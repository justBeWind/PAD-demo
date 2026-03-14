#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Iterable

from budget_allocator import AdaptiveBudgetAllocator
from denpad_latent import DenPADLatentSanitizer
from generate import load_test_data, load_or_build_retrieval_db
from learned_budget_allocator import BudgetExample, train_budget_head
from retriever import RetrievalDatabaseBuilder


LOGGER = logging.getLogger(__name__)


def _prompt_strings(records) -> list[str]:
    prompts: list[str] = []
    for item in records:
        if isinstance(item, dict):
            prompts.append(item.get("question") or item.get("input") or item.get("prompt") or "")
        else:
            prompts.append(str(item))
    return [prompt for prompt in prompts if prompt]


def _iter_examples(
    prompts: Iterable[str],
    db,
    sanitizer: DenPADLatentSanitizer,
    teacher: AdaptiveBudgetAllocator,
    retrieval_k: int,
) -> Iterable[BudgetExample]:
    for query in prompts:
        docs = [doc.page_content for doc in db.similarity_search(query, k=retrieval_k)]
        spans = sanitizer.unit_constructor.propose_spans(docs)
        units = sanitizer.unit_constructor.build_units(query=query, docs=docs, spans=spans)
        units = teacher.allocate(query=query, docs=docs, spans=spans, units=units)
        for unit in units:
            yield BudgetExample(
                query=query,
                unit_text=unit.local_text,
                local_context=unit.local_text,
                phi_type=unit.phi_type,
                risk=unit.risk_score,
                utility=unit.utility_score,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a distilled learned budget allocator from the current teacher allocator.")
    parser.add_argument("--datasets", nargs="+", default=["healthcaremagic", "icliniq"])
    parser.add_argument("--prompt_limit", type=int, default=100)
    parser.add_argument("--retrieval_k", type=int, default=6)
    parser.add_argument("--retriever_model", type=str, default="BAAI/bge-large-en-v1.5")
    parser.add_argument("--embedding_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--examples_output", type=str, default="/root/autodl-tmp/PAD-demo/checkpoints/learned_budget_examples.jsonl")
    parser.add_argument("--checkpoint_output", type=str, default="/root/autodl-tmp/PAD-demo/checkpoints/learned_budget_allocator.pt")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    builder = RetrievalDatabaseBuilder(device=args.device)
    sanitizer = DenPADLatentSanitizer(epsilon=args.epsilon)
    teacher = AdaptiveBudgetAllocator(epsilon=args.epsilon, embedding_model_name=args.embedding_model)

    os.makedirs(os.path.dirname(args.examples_output), exist_ok=True)
    total = 0
    with open(args.examples_output, "w", encoding="utf-8") as handle:
        for dataset in args.datasets:
            prompt_file = (
                f"data/{dataset}_prompt.json"
                if dataset not in {"healthcaremagic", "icliniq"}
                else f"data/{dataset}_prompt.json"
            )
            records = load_test_data(prompt_file)[: args.prompt_limit]
            prompts = _prompt_strings(records)

            class ArgsLike:
                pass

            args_like = ArgsLike()
            args_like.dataset = dataset
            args_like.method = "denpad"
            args_like.retriever_model = args.retriever_model
            args_like.debug_corpus_limit = None
            args_like.epsilon = args.epsilon
            args_like.force_rebuild_retrieval_db = False

            db = load_or_build_retrieval_db(args_like, builder)
            for example in _iter_examples(prompts, db, sanitizer, teacher, args.retrieval_k):
                handle.write(json.dumps(example.__dict__) + "\n")
                total += 1

    LOGGER.info("Collected %d budget examples at %s", total, args.examples_output)
    examples = []
    with open(args.examples_output, "r", encoding="utf-8") as handle:
        for line in handle:
            examples.append(BudgetExample(**json.loads(line)))
    stats = train_budget_head(
        examples=examples,
        output_path=args.checkpoint_output,
        embedding_model_name=args.embedding_model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device="cuda" if args.device.startswith("cuda") else "cpu",
    )
    LOGGER.info("Saved learned budget allocator checkpoint to %s", args.checkpoint_output)
    LOGGER.info("Training stats: %s", stats)


if __name__ == "__main__":
    main()
