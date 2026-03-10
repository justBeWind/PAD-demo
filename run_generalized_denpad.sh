#!/bin/bash
# generalized_denpad_runner.sh
# Server Execution Script for the Generalized DenPAD Architecture

set -e

MODEL_PATH=${1:-"meta-llama/Llama-2-7b-chat-hf"}
EPSILON=${2:-5.0}

echo "=============================================="
echo "Starting Generalized DenPAD Execution"
echo "Model: $MODEL_PATH"
echo "Epsilon Budget: $EPSILON"
echo "=============================================="

# Generate Safe Contexts and Answers
python generate_generalized_denpad.py --model "$MODEL_PATH" --epsilon $EPSILON

echo ""
echo "=============================================="
echo "Running Extractive Leakage Evaluation"
echo "=============================================="
# Assuming evaluate.py stays standard
python evaluate.py --input_file "generalized_denpad_results.json" --rouge_threshold 0.5 

echo "Done. Audit logs are located in ./logs/"
