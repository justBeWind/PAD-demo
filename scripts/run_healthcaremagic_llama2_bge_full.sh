#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODEL_NAME="meta-llama/Llama-2-7b-chat-hf"
RETRIEVER_MODEL="BAAI/bge-large-en-v1.5"
DATASET="healthcaremagic"
DEVICE="${DEVICE:-cuda:0}"
TORCH_DTYPE="${TORCH_DTYPE:-float16}"
MAX_TOKENS="${MAX_TOKENS:-256}"
TEMPERATURE="${TEMPERATURE:-0.2}"
TOP_P="${TOP_P:-0.9}"
RETRIEVAL_K="${RETRIEVAL_K:-6}"
RERANK_TOP_N="${RERANK_TOP_N:-3}"

OUT_DIR="result/${DATASET}/llama2"
mkdir -p "$OUT_DIR"

run_method() {
  local method="$1"
  local output_json="$OUT_DIR/${method}_full.json"
  local output_txt="$OUT_DIR/${method}_full.txt"

  echo "[RUN] ${method} -> ${output_json}"

  if [[ "$method" == "denpad" || "$method" == "contextpad" ]]; then
    python generate.py \
      --method "$method" \
      --dataset "$DATASET" \
      --model_name "$MODEL_NAME" \
      --retriever_model "$RETRIEVER_MODEL" \
      --device "$DEVICE" \
      --torch_dtype "$TORCH_DTYPE" \
      --temperature "$TEMPERATURE" \
      --top_p "$TOP_P" \
      --max_tokens "$MAX_TOKENS" \
      --retrieval_k "$RETRIEVAL_K" \
      --rerank_top_n "$RERANK_TOP_N" \
      --output_file "$output_json" \
      --denpad_audit_file "${output_json%.json}.audit.jsonl"
  else
    python generate.py \
      --method "$method" \
      --dataset "$DATASET" \
      --model_name "$MODEL_NAME" \
      --retriever_model "$RETRIEVER_MODEL" \
      --device "$DEVICE" \
      --torch_dtype "$TORCH_DTYPE" \
      --temperature "$TEMPERATURE" \
      --top_p "$TOP_P" \
      --max_tokens "$MAX_TOKENS" \
      --retrieval_k "$RETRIEVAL_K" \
      --rerank_top_n "$RERANK_TOP_N" \
      --output_file "$output_json"
  fi

  python evaluate.py --input_file "$output_json" > "$output_txt"
}

run_method baseline
run_method pad
run_method contextpad
run_method denpad
