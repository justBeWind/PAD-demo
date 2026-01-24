import json
import random
import os

# 配置：直接指向原始语料库文件
input_file = "corpus/healthcaremagic-100k.json" 
output_file = "data/healthcaremagic_utility_prompt.json"
sample_size = 250

def main():
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found. Please check your corpus directory.")
        return

    print(f"Loading corpus from {input_file}...")
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 抽取真实问题 (Input 字段)
    # 过滤掉太短的问题(小于20字符)，保证问题有意义
    # 原始json结构中，key为 "input"
    valid_prompts = [item["input"] for item in data if "input" in item and len(item.get("input", "")) > 20]
    
    print(f"Found {len(valid_prompts)} valid prompts.")
    
    if len(valid_prompts) == 0:
        print("Warning: No valid prompts found. Check the json structure.")
        return

    # 随机采样
    selected_prompts = random.sample(valid_prompts, min(sample_size, len(valid_prompts)))
    
    # 保存
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(selected_prompts, f, indent=2, ensure_ascii=False)
        
    print(f"Success! Saved {len(selected_prompts)} utility prompts to {output_file}")

if __name__ == "__main__":
    main()