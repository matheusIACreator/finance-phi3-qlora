"""
Dataset.py
Carrega train.jsonl e eval.jsonl e os prepara para SFTTrainer
Template: Phi-3-mini  <|user|>...<|end|>\n<|assistant|>...<|end|>
"""

from pathlib import Path
import json
from datasets import Dataset

SYSTEM_PROMPT = """
    Você é um assistente especialista em finanças corporativas
    Responda perguntas sobre balanços patrimoniais, demonstrações de resultados,
    fluxo de caixa e termos contábeis de forma clara e precisa.
    NÃO responda em markdown, com ** ou ##
"""

def phi_3_template(question:str, context:str,answer:str):
    """
    Template Oficial do Phi-3-mini
    O <|system|> é suportado na versão instruct — incluímos para dar contexto ao modelo.
    """

    if context:
        user_content = f"Contexto financeiro: \n{context}\n\nPergunta: {question}"
    else:
        user_content=f"Pergunta: {question}"

    return(
        f"<|system|>\n{SYSTEM_PROMPT}<|end|>\n"
        f"<|user|>\n{user_content}<|end|>\n"
        f"<|assistant|>\n{answer}<|end|>"
    )

def load_datasets(data_dir:str = "data")-> tuple[Dataset, Dataset]:
    """
    Retorna (train_dataset, eval_dataset) prontos para o SFTTrainer
    Cada exemplo já vem com o campo 'text' no template Phi-3
    """

    data_path=Path(data_dir)
    train_records=_load_jsonl(data_path/"train.jsonl")
    eval_recods= _load_jsonl(data_path/"eval.jsonl")

    def reformat(records:list[dict])->list[dict]:
        out=[]
        for r in records:
            # Os registros já têm 'text' do prepare_dataset.py,
            # mas aqui reforçamos o template Phi-3 correto.

            question=r.get("question","")
            answer=r.get("answer","")
            # context fica vazio se não veio separado — usamos o text original
            # como fallback para não perder dados já formatados

            if question and answer:
                text = _phi3_template(question, r.get("context",""), answer)

            else:
                text=r.get("text","")

            if text :
                out.append({"text":text})
        return out

    train_data = reformat(train_records)
    eval_data = reformat(eval_recods)
    print(f"Dataset carregado — treino: {len(train_data)}  eval: {len(eval_data)}")


    return Dataset.from_list(train_data), Dataset.from_list(eval_data)

def preview(dataset:Dataset, n:int=2)->None:
    """Imprime n exemplos formatados para inspeção rápida."""
    for i, ex in enumerate(dataset.select(range(n))):
        print(f"\n{'='*60}\nExemplo {i+1}\n{'='*60}")
        print(ex["text"][:800])
        if len(ex["text"]) > 800:
            print("...")

if __name__ == "__main__":
    train_ds, eval_ds = load_datasets()
    preview(train_ds)