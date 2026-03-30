# Finance Phi-3 QLoRA — Fine-tuning LLM for Financial Domain

> Fine-tuning of **Phi-3-mini-4k-instruct** using **QLoRA** on financial Q&A data, achieving a **+129% improvement in ROUGE-L** over the base model.

---

## Overview

This project demonstrates domain adaptation of a small LLM (3.8B parameters) for the financial domain using parameter-efficient fine-tuning (PEFT) with QLoRA. Only **0.23% of parameters** were trained, yet the model shows significant improvement in answering financial questions accurately and concisely.

| Metric | Base Model | Fine-tuned | Delta |
|---|---|---|---|
| ROUGE-L (avg) | 0.218 | 0.499 | **+0.281 (+129%)** |
| Val Loss (epoch 1) | — | 0.8504 | — |
| Val Loss (epoch 2) | — | 0.8108 | ↓ |
| Val Loss (epoch 3) | — | 0.8068 | ↓ |
| Trainable params | — | 8.9M / 3.8B | 0.23% |

---

## Results — Base vs Fine-tuned

**Q: What is EBITDA and how is it calculated?**
- **Base (ROUGE-L 0.348):** Long, verbose explanation with minor hallucinations.
- **Fine-tuned (ROUGE-L 0.880):** "EBITDA stands for Earnings Before Interest, Taxes, Depreciation, and Amortization. It is calculated by adding back interest, taxes, depreciation, and amortization to a company's earnings."

**Q: What is the difference between gross profit and net profit?**
- **Base (ROUGE-L 0.234):** Overly detailed, loses focus.
- **Fine-tuned (ROUGE-L 0.571):** "Gross profit is the difference between revenue and the cost of goods sold, while net profit is the difference between revenue and all expenses, including taxes and interest."

---

## Stack

| Component | Tool |
|---|---|
| Base model | `microsoft/Phi-3-mini-4k-instruct` |
| Fine-tuning method | QLoRA (4-bit NF4 quantization) |
| PEFT library | `peft` |
| Training framework | `trl` — SFTTrainer |
| Dataset | FinQA (1,000 train / 200 eval) |
| Evaluation | ROUGE-L (`evaluate`) |
| Training hardware | Google Colab T4 (16GB VRAM) |
| Training time | ~2 hours |

---

## Project Structure

```
finance-phi3-qlora/
├── config/
│   ├── qlora_config.yaml       # LoRA hyperparameters
│   └── training_args.yaml      # Training configuration
├── data/
│   ├── ram_finqa.jsonl         # FinQA dataset
│   └── synthetic_qa.jsonl      # Synthetic financial Q&A
├── demo/
│   ├── app.py                  # Gradio demo app
│   └── requirements.txt
├── notebook/
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_baseline_eval.ipynb
│   ├── 03_finetuned_eval.ipynb
│   ├── 04_comparison_results.ipynb
│   └── finance_llm_finetune_colab.ipynb  # Main training notebook
├── outputs/
│   ├── checkpoints/
│   └── final_model/            # LoRA adapter weights
├── src/
│   ├── dataset.py
│   ├── evaluate.py
│   ├── model.py
│   ├── train.py
│   └── utils.py
└── requirements.txt
```

---

## QLoRA Configuration

```python
# 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# LoRA adapters
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
```

---

## Training Configuration

```yaml
num_train_epochs: 3
per_device_train_batch_size: 2
gradient_accumulation_steps: 8   # effective batch size = 16
learning_rate: 2e-4
lr_scheduler_type: cosine
warmup_ratio: 0.05
max_seq_length: 512
optim: paged_adamw_8bit
```

---

## How to Run

### 1. Clone and install
```bash
git clone https://github.com/matheusIACreator/finance-phi3-qlora
cd finance-phi3-qlora
pip install -r requirements.txt
```

### 2. Train (Google Colab recommended)
Open `notebook/finance_llm_finetune_colab.ipynb` in Google Colab with a T4 GPU runtime and run all cells.

### 3. Run the demo
```bash
cd demo
pip install -r requirements.txt
python app.py
```

### 4. Load the fine-tuned model
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    load_in_4bit=True,
    device_map="auto",
)
model = PeftModel.from_pretrained(base_model, "./outputs/final_model")
tokenizer = AutoTokenizer.from_pretrained("./outputs/final_model")
```

---

## Key Takeaways

- **QLoRA enables domain adaptation on consumer hardware** — the full Phi-3-mini (3.8B) was fine-tuned on a free T4 GPU by training only 8.9M parameters.
- **Conciseness improves with fine-tuning** — the base model tends to be verbose and unfocused; the fine-tuned model gives direct, accurate financial definitions.
- **ROUGE-L improvement of +129%** confirms the adapter successfully adapted the model's output distribution to match financial Q&A style.

---

## Author

**Matheus Masago** — Junior AI/ML Engineer  
[LinkedIn](https://linkedin.com/in/matheus-masago) · [Hugging Face](https://huggingface.co/Shinigami4242557) · [GitHub](https://github.com/matheusIACreator)