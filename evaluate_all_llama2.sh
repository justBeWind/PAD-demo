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

# [修复点 1]：表头必须包含所有列，并且顺序要和下面写入的顺序完全一致
echo "Dataset,Method,Repeat_Prompts,Repeat_Contexts,Rouge_Prompts,Rouge_Contexts,PPL" > $SUMMARY_FILE

echo "################################################################"
echo "Starting Batch Evaluation for Llama-2 Results"
echo "################################################################"

# [修复点 2]：优化终端打印格式，增加 Rouge 列的显示
# 调整列宽以适应更多数据
printf "%-15s %-18s %-10s %-10s %-10s %-10s %-8s\n" "DATASET" "METHOD" "RPT_PR" "RPT_CX" "RGE_PR" "RGE_CX" "PPL"
echo "------------------------------------------------------------------------------------------------"

for DATASET in "${DATASETS[@]}"; do
    for METHOD in "${METHODS[@]}"; do
        # 构造文件路径
        INPUT_JSON="${BASE_DIR}/${DATASET}/llama2/${METHOD}.json"
        OUTPUT_EVAL="${BASE_DIR}/${DATASET}/llama2/${METHOD}_eval.txt"
        
        # 检查文件是否存在
        if [ -f "$INPUT_JSON" ]; then
            # 1. 运行评估脚本 (并将详细输出写入 txt)
            # 注意：如果不需要每次都重新跑 evaluate.py，可以注释掉下面这行，直接读取已有的 _eval.txt
            python evaluate.py --input_file "$INPUT_JSON" > "$OUTPUT_EVAL"
            
            # 2. 从 txt 结果中提取核心指标
            
            # 提取 Repeat Prompts
            RPT_PROMPTS=$(grep "Repeat Prompts:" "$OUTPUT_EVAL" | awk -F':' '{print $2}' | tr -d '[:space:]')
            
            # 提取 Repeat Contexts
            RPT_CONTEXTS=$(grep "Repeat Contexts:" "$OUTPUT_EVAL" | awk -F':' '{print $2}' | tr -d '[:space:]')
            
            # 提取 Rouge Prompts
            ROUGE_PROMPTS=$(grep "Rouge Prompts:" "$OUTPUT_EVAL" | awk -F':' '{print $2}' | tr -d '[:space:]')

            # 提取 Rouge Contexts (你新增的部分)
            ROUGE_CONTEXTS=$(grep "Rouge Contexts:" "$OUTPUT_EVAL" | awk -F':' '{print $2}' | tr -d '[:space:]')

            # 提取 Perplexity
            PPL=$(grep -i "Perplexity:" "$OUTPUT_EVAL" | awk -F':' '{print $2}' | awk '{print $1}' | tr -d '[:space:]')
            
            # 3. 处理空值 (防止提取失败导致格式错乱)
            if [ -z "$RPT_PROMPTS" ]; then RPT_PROMPTS="N/A"; fi
            if [ -z "$RPT_CONTEXTS" ]; then RPT_CONTEXTS="N/A"; fi
            if [ -z "$ROUGE_PROMPTS" ]; then ROUGE_PROMPTS="N/A"; fi
            if [ -z "$ROUGE_CONTEXTS" ]; then ROUGE_CONTEXTS="N/A"; fi
            if [ -z "$PPL" ]; then PPL="N/A"; fi
            
            # 4. 打印到终端表格 (格式化输出)
            printf "%-15s %-18s %-10s %-10s %-10s %-10s %-8s\n" "$DATASET" "$METHOD" "$RPT_PROMPTS" "$RPT_CONTEXTS" "$ROUGE_PROMPTS" "$ROUGE_CONTEXTS" "$PPL"
            
            # 5. [修复点 3]：写入 CSV，顺序必须与表头一致
            # 顺序: Dataset, Method, Rpt_P, Rpt_C, Rouge_P, Rouge_C, PPL
            echo "${DATASET},${METHOD},${RPT_PROMPTS},${RPT_CONTEXTS},${ROUGE_PROMPTS},${ROUGE_CONTEXTS},${PPL}" >> $SUMMARY_FILE
        else
            # 文件不存在
            printf "%-15s %-18s %-10s\n" "$DATASET" "$METHOD" "[File Not Found]"
        fi
    done
    echo "------------------------------------------------------------------------------------------------"
done

echo ""
echo "================================================================"
echo "Evaluation Complete!"
echo "Summary saved to: $SUMMARY_FILE"
echo "================================================================"