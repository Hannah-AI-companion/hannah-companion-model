# Hannah - Model Training & Data Pipeline

This repository is dedicated to the creation and refinement of the Hannah language model (OLMo 360M). It contains the full pipeline from raw data ingestion to Direct Preference Optimization (DPO).

## 🏗 Repository Structure

```
hannah-companion-model/
├── data/                   # (Ignored) Training datasets (Raw, Processed, Bins)
├── scripts/
│   ├── data_pipeline/      # Ingestion, cleaning (MinHash), and tokenization
│   ├── processing/         # SFT and DPO corpus construction
│   └── tests/              # Model and tokenizer validation scripts
├── src/
│   ├── tokenizer/          # BPE Tokenizer training logic
│   ├── model/              # OLMo architecture definition
│   └── training/           # Pretraining, SFT, and DPO training loops
├── configs/                # Model and training hyperparameters
└── requirements.txt        # Training dependencies (torch, olmo-core, etc.)
```

## 🚀 Execution Steps (Step-by-Step)

### 1. Preparation
Install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Data Ingestion & Cleaning
Download raw data (optional) and run the cleaning pipeline:
```bash
# Optional download
python scripts/data_pipeline/download/hf_datasets.py
python scripts/data_pipeline/download/gutenberg.py

# Merge, clean, and deduplicate
cd scripts/data_pipeline && python build_corpus.py
```

### 3. Tokenization
Convert the cleaned corpus into binary files for training:
```bash
python scripts/data_pipeline/prepare_corpus.py
```

### 4. Training (Pipeline Phases)
Execute the training scripts in order. **Caution:** Phase 1 requires significant GPU time.

- **Phase 1: Pretraining**
  ```bash
  python src/training/train_hannah.py
  ```
- **Phase 2: SFT (Supervised Fine-Tuning)**
  ```bash
  python src/training/train_sft_hannah.py
  ```
- **Phase 3: DPO (Preference Optimization)**
  ```bash
  python src/training/train_dpo_hannah.py
  ```

### 5. Validation
Test the resulting checkpoints:
```bash
python scripts/tests/test_hannah.py
python scripts/tests/test_sft_hannah.py
```

## 📦 Requirements

- NVIDIA GPU with 16GB+ VRAM recommended.
- Python 3.10+
- `pip install -r requirements.txt`
