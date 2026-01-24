import os
import yaml
from llama_cpp import Llama
from dotenv import load_dotenv

load_dotenv()

MODEL_PATH = os.getenv("MODEL_PATH")

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=4096,
    n_threads=12,
    n_gpu_layers=35,
    verbose=False,
)

def answer_prompt(question: str, context: str) -> str:
    return f"""Answer the question below using the provided context.
Keep the answer concise (1-2 sentences).

Question:
{question}

Context:
{context}
"""


def generate_answer(question: str, context: str) -> str:
    prompt = answer_prompt(question, context)
    output = llm(
        prompt,
        max_tokens=160,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        min_p=0.0,
    )
    if isinstance(output, dict):
        text = output.get("choices", [{}])[0].get("text", "").strip()
    else:
        text = str(output).strip()
    if not text:
        retry = llm(
            prompt + " Provide a concise answer.",
            max_tokens=160,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            min_p=0.0,
        )
        if isinstance(retry, dict):
            text = retry.get("choices", [{}])[0].get("text", "").strip()
        else:
            text = str(retry).strip()
    return text


if __name__ == "__main__":
    target = "data/processed/How Stock Markets Work/algorithmic_trading,_high-frequency_trading_(HFT),_and_market_microstructure/chunks.yaml"
    with open(target, "r", encoding="utf-8") as f:
        raw = f.read().strip()
    data = yaml.safe_load(raw) or {}
    context = raw
    questions = data.get("questions_to_think", []) or []

    print("Questions to think (with answers):")
    for q in questions:
        ans = generate_answer(q, context)
        print(f"- Q: {q}")
        print(f"  A: {ans}")
