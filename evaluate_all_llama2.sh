#!/bin/bash

# ==============================================================================
# Llama-2 全量实验评估脚本 (3 Datasets x 4 Methods)
# 功能：运行 evaluate.py 并自动提取核心指标生成汇总报表
# ==============================================================================

# 定义配置
DATASETS=("healthcaremagic" "icliniq" "enron_mail")
METHODS=("baseline_attack" "pad_attack" "lprag_attack" "den_pad_attack")
BASE_DIR="result"

# 创建一个临时文件存汇总数据
SUMMARY_FILE="evaluation_summary_llama2.csv"
echo "Dataset,Method,Repeat_Prompts,Repeat_Contexts,PPL,Rouge_Prompts" > $SUMMARY_FILE

echo "################################################################"
echo "Starting Batch Evaluation for Llama-2 Results"
echo "################################################################"

# 打印表头 (为了终端好看)
printf "%-15s %-20s %-15s %-15s %-10s\n" "DATASET" "METHOD" "RPT_PROMPTS" "RPT_CONTEXTS" "PPL"
echo "--------------------------------------------------------------------------------"

for DATASET in "${DATASETS[@]}"; do
    for METHOD in "${METHODS[@]}"; do
        # 构造文件路径
        INPUT_JSON="${BASE_DIR}/${DATASET}/llama2/${METHOD}.json"
        OUTPUT_EVAL="${BASE_DIR}/${DATASET}/llama2/${METHOD}_eval.txt"
        
        # 检查文件是否存在
        if [ -f "$INPUT_JSON" ]; then
            # 1. 运行评估脚本 (并将详细输出写入 txt)
            python evaluate.py --input_file "$INPUT_JSON" > "$OUTPUT_EVAL"
            
            # 2. 从 txt 结果中提取核心指标 (使用 grep 和 awk)
            # 提取 Repeat Prompts (例如: "Repeat Prompts: 129")
            RPT_PROMPTS=$(grep "Repeat Prompts:" "$OUTPUT_EVAL" | awk -F':' '{print $2}' | tr -d ' ')
            
            # 提取 Repeat Contexts
            RPT_CONTEXTS=$(grep "Repeat Contexts:" "$OUTPUT_EVAL" | awk -F':' '{print $2}' | tr -d ' ')
            
            # 提取 Perplexity (例如: "Avg Perplexity: 6.60" 或 "Average Perplexity: 6.60")
            PPL=$(grep -i "Perplexity:" "$OUTPUT_EVAL" | awk -F':' '{print $2}' | awk '{print $1}' | tr -d ' ')
            
            # 提取 Rouge Prompts (可选)
            ROUGE_PROMPTS=$(grep "Rouge Prompts:" "$OUTPUT_EVAL" | awk -F':' '{print $2}' | tr -d ' ')
            
            # 3. 打印到终端表格
            # 如果没提取到值（可能是评估报错），显示 N/A
            if [ -z "$RPT_PROMPTS" ]; then RPT_PROMPTS="N/A"; fi
            if [ -z "$RPT_CONTEXTS" ]; then RPT_CONTEXTS="N/A"; fi
            if [ -z "$PPL" ]; then PPL="N/A"; fi
            
            printf "%-15s %-20s %-15s %-15s %-10s\n" "$DATASET" "$METHOD" "$RPT_PROMPTS" "$RPT_CONTEXTS" "$PPL"
            
            # 4. 写入 CSV 方便复制
            echo "${DATASET},${METHOD},${RPT_PROMPTS},${RPT_CONTEXTS},${PPL},${ROUGE_PROMPTS}" >> $SUMMARY_FILE
        else
            # 文件不存在 (可能还没跑完)
            printf "%-15s %-20s %-15s\n" "$DATASET" "$METHOD" "[File Not Found]"
        fi
    done
    echo "--------------------------------------------------------------------------------"
done

echo ""
echo "================================================================"
echo "Evaluation Complete!"
echo "Summary saved to: $SUMMARY_FILE"
echo "Detailed reports are in result/{dataset}/llama2/*_eval.txt"
echo "================================================================"