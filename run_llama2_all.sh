#!/bin/bash

# ==============================================================================
# Llama-2-7b-chat 全量实验脚本 (3 Datasets x 4 Methods)
# ==============================================================================

# 1. 核心配置
MODEL="meta-llama/Llama-2-7b-chat-hf"
RETRIEVER="BAAI/bge-large-en-v1.5"
DENSITY_FILE="data/llama_2_7b_density.json"

# 数据集列表 (注意: enron_mail 的名字要和代码里的定义一致)
DATASETS=("healthcaremagic" "icliniq" "enron_mail")

echo "################################################################"
echo "Starting All-in-One Experiment on Llama-2"
echo "Datasets: ${DATASETS[*]}"
echo "################################################################"

# 2. 准备阶段: 计算密度图 (Global for Llama-2)
# 因为 calculate_density.py 是基于静态权重的，所以所有数据集共用这一份 map
if [ ! -f "$DENSITY_FILE" ]; then
    echo "[Init] Calculating Density Map for Llama-2 (One-time setup)..."
    python calculate_density.py \
        --model_name $MODEL \
        --output $DENSITY_FILE \
        --k 20
else
    echo "[Init] Density Map found: $DENSITY_FILE (Skipping calculation)"
fi

# 3. 循环跑实验
for DATASET in "${DATASETS[@]}"; do
    echo ""
    echo "================================================================"
    echo "Processing Dataset: $DATASET"
    echo "================================================================"
    
    # 创建输出目录
    OUTPUT_DIR="result/${DATASET}/llama2"
    mkdir -p $OUTPUT_DIR
    
    # --- Experiment A: Baseline ---
    echo "[${DATASET}] Running Baseline..."
    python generate.py \
        --method baseline \
        --dataset $DATASET \
        --model_name $MODEL \
        --retriever_model $RETRIEVER \
        --temperature 0.2 \
        --max_tokens 256 \
        --output_file "${OUTPUT_DIR}/baseline_attack.json"

    # --- Experiment B: PAD (Standard: eps=0.2, amp=3.0) ---
    echo "[${DATASET}] Running PAD..."
    python generate.py \
        --method pad \
        --dataset $DATASET \
        --model_name $MODEL \
        --retriever_model $RETRIEVER \
        --temperature 0.2 \
        --max_tokens 256 \
        --epsilon 0.2 \
        --noise_amplification 3.0 \
        --min_sensitivity 0.4 \
        --output_file "${OUTPUT_DIR}/pad_attack.json"

    # --- Experiment C: LPRAG (Input DP: eps=3.0) ---
    echo "[${DATASET}] Running LPRAG..."
    python generate.py \
        --method lprag \
        --dataset $DATASET \
        --model_name $MODEL \
        --retriever_model $RETRIEVER \
        --temperature 0.2 \
        --max_tokens 256 \
        --lprag_epsilon 3.0 \
        --output_file "${OUTPUT_DIR}/lprag_attack.json"

    # --- Experiment D: DenPAD (Ours: eps=0.2, amp=3.0, Density-Aware) ---
    echo "[${DATASET}] Running DenPAD (SOTA)..."
    python generate.py \
        --method denpad \
        --dataset $DATASET \
        --model_name $MODEL \
        --retriever_model $RETRIEVER \
        --temperature 0.2 \
        --max_tokens 256 \
        --epsilon 0.2 \
        --noise_amplification 3.0 \
        --min_sensitivity 0.4 \
        --density_map $DENSITY_FILE \
        --output_file "${OUTPUT_DIR}/den_pad_attack.json" \
        --ablation_mode full

    echo ">>> ${DATASET} Finished!"
done

echo ""
echo "################################################################"
echo "All Experiments Completed Successfully!"
echo "################################################################"
