# Privacy-Aware Decoding (PAD)

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-blue.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)]()
[![arXiv](https://img.shields.io/badge/arXiv-2508.03098-b31b1b.svg)](https://arxiv.org/abs/2508.03098)

## 📖 Overview

This repository contains the official implementation and datasets for our paper:

**[Privacy-Aware Decoding: Mitigating Privacy Leakage of Large Language Models in Retrieval-Augmented Generation](https://arxiv.org/abs/2508.03098)**

## Project Structure

```
PAD/
├── 📁 data/                 # Attack prompts
├── 📁 result/               # Output results
├── 📁 processed/            # Processed data files
├── 📁 corpus/               # Corpus files for retrieval
├── 📁 RetrievalBase/        
├── 🐍 generate.py           # Main generation script
├── 🐍 llm.py                # LLM engine with PAD
├── 🐍 retriever.py          # Retrieval system
├── 🐍 evaluate.py           # Evaluation script
├── 🐍 utils.py              
├── 📄 environment.yml       
└── 📄 .gitignore           
```

## ⚙️ Installation & Setup

### Prerequisites
- **Python**: 3.9 or higher
- **Conda**: For environment management

### Quick Start

1. **Create and activate conda environment**:
   ```bash
   conda env create -n pad --file environment.yml
   conda activate pad
   ```

2. **Download required datasets**:
   
   **Medical Datasets**:
   - [HealthCareMagic](https://drive.google.com/file/d/1lyfqIwlLSClhgrCutWuEe_IACNq6XNUt/view) - Place in `corpus/`
   - [iCliniq](https://drive.google.com/file/d/1ZKbqgYqWc7DJHs3N9TQYQVPdDQmZaClA/view) - Place in `corpus/`
   
   **Email Dataset**:
   - [Enron Mail](https://www.cs.cmu.edu/~enron/) - Download and extract to `corpus/`

## 🚀 Usage

### Running Extraction Attacks (Baseline)

```bash
python generate.py \
    --dataset healthcaremagic \
    --model_name EleutherAI/pythia-6.9b \
    --retriever_model BAAI/bge-large-en-v1.5 \
    --temperature 0.2 \
    --max_tokens 256 \
    --output_file result/healthcaremagic/pythia/baseline.json
```

### Running PAD (Privacy-Aware Decoding)

```bash
python generate.py \
    --dataset healthcaremagic \
    --model_name EleutherAI/pythia-6.9b \
    --retriever_model BAAI/bge-large-en-v1.5 \
    --temperature 0.2 \
    --add_noise \
    --epsilon 0.2 \
    --noise_amplification 3.0 \
    --min_sensitivity 0.4 \
    --max_tokens 256 \
    --output_file result/healthcaremagic/pythia/pad.json
```

### 📊 Evaluation

**Evaluate baseline extraction attack**:
```bash
python evaluate.py \
    --input_file result/healthcaremagic/pythia/baseline.json \
    > result/healthcaremagic/pythia/baseline.txt
```

**Evaluate PAD results**:
```bash
python evaluate.py \
    --input_file result/healthcaremagic/pythia/pad.json \
    > result/healthcaremagic/pythia/pad.txt
```

## 📚 Citation

If you find this work useful, please cite our paper:

```bibtex
@article{wang2025privacy,
  title={Privacy-Aware Decoding: Mitigating Privacy Leakage of Large Language Models in Retrieval-Augmented Generation},
  author={Wang, Haoran and Xu, Xiongxiao and Huang, Baixiang and Shu, Kai},
  journal={arXiv preprint arXiv:2508.03098},
  year={2025}
}
```

## License

This project is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International License](https://creativecommons.org/licenses/by-nc/4.0/legalcode.txt).
