"""
src/model.py
Carrega o Phi-3-mini-4k-instruct quantizado em 4-bit (QLoRA)
e aplica os adaptadores LoRA via PEFT.
"""

import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from dotenv import load_dotenv

load_dotenv()

MODEL_ID  = "microsoft/Phi-3-mini-4k-instruct"
HF_TOKEN  = os.getenv("HF_TOKEN")


# ── Quantização ──────────────────────────────────────────────────────────────

def get_bnb_config() -> BitsAndBytesConfig:
    """4-bit NF4 com double quantization — padrão QLoRA."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,  # bfloat16 > float16 no Phi-3
        bnb_4bit_use_double_quant=True,
    )


# ── LoRA ─────────────────────────────────────────────────────────────────────

def get_lora_config() -> LoraConfig:
    """
    Configuração LoRA para o Phi-3-mini.
    target_modules: camadas de atenção e projeção do Phi-3.
    r=16, alpha=32 é um ponto de partida sólido para domínio específico.
    """
    return LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )


# ── Tokenizer ────────────────────────────────────────────────────────────────

def load_tokenizer() -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        token=HF_TOKEN,
        trust_remote_code=True,
    )
    # Phi-3 não tem pad_token por padrão — usa eos como pad
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"   # evita warnings do SFTTrainer
    return tokenizer


# ── Model ────────────────────────────────────────────────────────────────────

def load_model(lora: bool = True):
    """
    Carrega o Phi-3-mini em 4-bit e, opcionalmente, aplica LoRA.

    Args:
        lora: Se True, retorna modelo pronto para fine-tuning.
              Se False, retorna modelo base (para avaliação de baseline).
    """
    print(f"Carregando {MODEL_ID}...")
    print(f"  Modo: {'QLoRA fine-tuning' if lora else 'baseline (sem LoRA)'}")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=get_bnb_config(),
        device_map="auto",
        trust_remote_code=True,
        token=HF_TOKEN,
        attn_implementation="eager",  # Phi-3 recomenda eager em vez de flash_attn
    )

    if lora:
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, get_lora_config())
        model.print_trainable_parameters()

    return model


# ── Diagnóstico rápido ───────────────────────────────────────────────────────

if __name__ == "__main__":
    tokenizer = load_tokenizer()
    model = load_model(lora=True)

    sample = "<|system|>\nVocê é um especialista financeiro.<|end|>\n<|user|>\nO que é EBITDA?<|end|>\n<|assistant|>\n"
    inputs = tokenizer(sample, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=80, do_sample=False)

    print("\nResposta do modelo (antes do fine-tuning):")
    print(tokenizer.decode(out[0], skip_special_tokens=True))