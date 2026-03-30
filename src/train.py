"""
src/train.py
Fine-tuning do Phi-3-mini-4k-instruct com QLoRA no FinQA.
Integração com Weights & Biases para rastreamento.

Uso:
    python src/train.py

Variáveis de ambiente necessárias (.env):
    HF_TOKEN       — acesso ao modelo no HF Hub
    WANDB_API_KEY  — rastreamento no W&B (opcional mas recomendado)
"""

import os
from pathlib import Path

import torch
from dotenv import load_dotenv
from transformers import TrainingArguments
from trl import SFTTrainer, SFTConfig

from dataset import load_datasets
from model import load_model, load_tokenizer

load_dotenv()

# ── Caminhos ──────────────────────────────────────────────────────────────────
ROOT_DIR    = Path(__file__).parent.parent
OUTPUT_DIR  = ROOT_DIR / "outputs" / "checkpoints"
FINAL_DIR   = ROOT_DIR / "outputs" / "final_model"
DATA_DIR    = ROOT_DIR / "data"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FINAL_DIR.mkdir(parents=True, exist_ok=True)

# ── W&B ───────────────────────────────────────────────────────────────────────
WANDB_KEY = os.getenv("WANDB_API_KEY")
if WANDB_KEY:
    import wandb
    wandb.login(key=WANDB_KEY)
    os.environ["WANDB_PROJECT"] = "finance-llm-finetune"
    REPORT_TO = "wandb"
    RUN_NAME  = "phi3-mini-qlora-finqa-v1"
else:
    REPORT_TO = "none"
    RUN_NAME  = None
    print("⚠ WANDB_API_KEY não encontrado — treino sem rastreamento W&B.")


# ── Hiperparâmetros ───────────────────────────────────────────────────────────
# Ajuste conforme sua GPU:
#   RTX 3080 (10GB)  → batch=4, grad_accum=4  (effective batch = 16)
#   RTX 3090 (24GB)  → batch=8, grad_accum=2  (effective batch = 16)
#   Colab T4  (16GB) → batch=2, grad_accum=8  (effective batch = 16)
#   Colab A100(40GB) → batch=8, grad_accum=2

PER_DEVICE_BATCH  = 4
GRAD_ACCUM        = 4
NUM_EPOCHS        = 3
LEARNING_RATE     = 2e-4
MAX_SEQ_LENGTH    = 512     # Phi-3 suporta 4k, mas 512 economiza VRAM no FinQA
WARMUP_RATIO      = 0.05
LR_SCHEDULER      = "cosine"


def main():
    print("=" * 60)
    print("  Fine-tuning: Phi-3-mini-4k-instruct + QLoRA + FinQA")
    print("=" * 60)

    # 1. Datasets
    print("\n[1/4] Carregando datasets...")
    train_ds, eval_ds = load_datasets(str(DATA_DIR))

    # 2. Modelo + Tokenizer
    print("\n[2/4] Carregando modelo e tokenizer...")
    tokenizer = load_tokenizer()
    model     = load_model(lora=True)

    # 3. Training arguments
    print("\n[3/4] Configurando treino...")
    training_args = SFTConfig(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=NUM_EPOCHS,

        # Batch
        per_device_train_batch_size=PER_DEVICE_BATCH,
        per_device_eval_batch_size=PER_DEVICE_BATCH,
        gradient_accumulation_steps=GRAD_ACCUM,

        # Otimizador
        learning_rate=LEARNING_RATE,
        lr_scheduler_type=LR_SCHEDULER,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=0.01,
        optim="paged_adamw_8bit",       # paged optimizer — economiza VRAM

        # Precisão
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),

        # Sequência
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_text_field="text",
        packing=False,                  # False = mais simples; True = mais eficiente

        # Avaliação + checkpoints
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,             # mantém só os 2 melhores checkpoints
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",

        # Logging
        logging_steps=20,
        report_to=REPORT_TO,
        run_name=RUN_NAME,

        # Misc
        dataloader_num_workers=2,
        gradient_checkpointing=True,    # reduz VRAM à custo de ~20% de velocidade
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    # 4. Trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=training_args,
    )

    print("\n[4/4] Iniciando treino...\n")
    trainer.train()

    # 5. Salvar modelo final (apenas os adaptadores LoRA)
    print(f"\nSalvando modelo final em {FINAL_DIR}...")
    trainer.model.save_pretrained(str(FINAL_DIR))
    tokenizer.save_pretrained(str(FINAL_DIR))

    print("\n✓ Treino concluído!")
    print(f"  Checkpoints : {OUTPUT_DIR}")
    print(f"  Modelo final: {FINAL_DIR}")
    print("\nPróximo passo: python src/evaluate.py")


if __name__ == "__main__":
    main()