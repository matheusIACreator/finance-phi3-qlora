import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# ── Config ────────────────────────────────────────────────────────────────────
BASE_MODEL = "microsoft/Phi-3-mini-4k-instruct"
ADAPTER_PATH = "../outputs/final_model"  # relative to demo/
MAX_NEW_TOKENS = 300
TEMPERATURE = 0.7
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EXAMPLE_QUESTIONS = [
    "What is EBITDA and how is it calculated?",
    "What is the difference between gross profit and net profit?",
    "What does a current ratio below 1 indicate?",
    "How do you analyze the free cash flow of a company?",
    "What is the difference between accounts payable and accounts receivable?",
]

# ── Load model ────────────────────────────────────────────────────────────────
def load_model():
    print(f"Loading base model: {BASE_MODEL}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"Applying LoRA adapters from: {ADAPTER_PATH}")
    model = PeftModel.from_pretrained(base, ADAPTER_PATH)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH, trust_remote_code=True)

    print("✓ Model ready.")
    return model, tokenizer


# ── Inference ─────────────────────────────────────────────────────────────────
def generate_answer(question: str, model, tokenizer) -> str:
    prompt = (
        "<|system|>\n"
        "You are a concise financial expert assistant. "
        "Answer questions about corporate finance, accounting, and financial analysis accurately and directly.\n"
        "<|end|>\n"
        "<|user|>\n"
        f"{question.strip()}\n"
        "<|end|>\n"
        "<|assistant|>\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode only the generated tokens (skip the prompt)
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return answer


# ── Gradio UI ─────────────────────────────────────────────────────────────────
def build_ui(model, tokenizer):

    def respond(question: str) -> str:
        if not question.strip():
            return "Please enter a question."
        return generate_answer(question, model, tokenizer)

    with gr.Blocks(
        title="Finance Phi-3 QLoRA",
        theme=gr.themes.Base(
            primary_hue="slate",
            neutral_hue="slate",
            font=gr.themes.GoogleFont("IBM Plex Mono"),
        ),
        css="""
            body { background: #0d1117; }
            .gradio-container { max-width: 780px !important; margin: 0 auto; }
            #header { text-align: center; padding: 2rem 0 1rem; }
            #header h1 { font-size: 1.6rem; color: #e6edf3; letter-spacing: -0.02em; }
            #header p  { color: #8b949e; font-size: 0.85rem; margin-top: 0.4rem; }
            #badge { display: inline-block; background: #1f2d3d; color: #58a6ff;
                     padding: 2px 10px; border-radius: 20px; font-size: 0.75rem;
                     margin-top: 0.5rem; border: 1px solid #30363d; }
            .answer-box textarea { font-family: 'IBM Plex Mono', monospace !important;
                                   font-size: 0.9rem !important; line-height: 1.6 !important; }
        """,
    ) as demo:

        gr.HTML("""
            <div id="header">
                <h1>Finance Phi-3 QLoRA</h1>
                <p>Phi-3-mini-4k-instruct fine-tuned on financial Q&amp;A via QLoRA</p>
                <span id="badge">0.23% parameters trained · ROUGE-L +129%</span>
            </div>
        """)

        with gr.Row():
            question_input = gr.Textbox(
                label="Question",
                placeholder="Ask a financial question...",
                lines=2,
                elem_id="question-box",
            )

        submit_btn = gr.Button("Ask", variant="primary")

        answer_output = gr.Textbox(
            label="Answer",
            lines=6,
            interactive=False,
            elem_classes=["answer-box"],
        )

        gr.Examples(
            examples=EXAMPLE_QUESTIONS,
            inputs=question_input,
            label="Example questions",
        )

        gr.Markdown(
            """
            ---
            **Model:** `microsoft/Phi-3-mini-4k-instruct` + LoRA adapters (QLoRA, 4-bit NF4)  
            **Dataset:** FinQA · **Training:** Google Colab T4 · 3 epochs  
            **Stack:** `transformers` · `peft` · `trl` · `bitsandbytes`
            """,
            elem_id="footer",
        )

        submit_btn.click(fn=respond, inputs=question_input, outputs=answer_output)
        question_input.submit(fn=respond, inputs=question_input, outputs=answer_output)

    return demo


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model, tokenizer = load_model()
    demo = build_ui(model, tokenizer)
    demo.launch(share=False)