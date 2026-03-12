import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

DEFAULT_INSTRUCTION = (
    "Give a helpful hint for the question without giving the answer. "
    "Return exactly 1-2 short sentences."
)
STRICT_HINT_INSTRUCTION = (
    "Give a conceptual clue for the question without giving the answer. "
    "Return exactly 1-2 short sentences. "
    "Do not define the term directly, do not repeat the question wording, "
    "and do not use the exact target term when it is being defined."
)


def build_prompt(row: dict, instruction_override: str | None = None) -> str:
    instruction = instruction_override or row.get("instruction", "")
    return (
        "### Instruction\n"
        + instruction
        + "\n\n### Input\n"
        + row.get("input", "")
        + "\n\n### Response\n"
    )

def clean_generated(text: str) -> str:
    # Cut off if the model starts emitting the next template block.
    for marker in ("\n### Input", "\n### Instruction"):
        if marker in text:
            text = text.split(marker, 1)[0]
    return text.strip()


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _normalize_options(raw_options) -> list[str]:
    if raw_options is None:
        return []

    # Handle mixed payloads like ["A", "text", "B", "text"].
    if isinstance(raw_options, list):
        out: list[str] = []
        i = 0
        while i < len(raw_options):
            item = raw_options[i]
            if isinstance(item, dict):
                for key, val in item.items():
                    out.append(f"{key}. {str(val).strip()}")
                i += 1
                continue

            text = str(item).strip()
            if len(text) == 1 and text.isalpha() and i + 1 < len(raw_options):
                nxt = str(raw_options[i + 1]).strip()
                out.append(f"{text}. {nxt}")
                i += 2
                continue

            out.append(text)
            i += 1
        return [x for x in out if x]

    return [str(raw_options).strip()]


def load_supabase_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    rows: list[dict] = []
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON array in {path}")

    for item in payload:
        cj = item.get("content_json", {}) if isinstance(item, dict) else {}
        if not isinstance(cj, dict):
            continue

        q_type = str(cj.get("type", "")).strip()
        question = str(cj.get("question", "")).strip()
        scenario = str(cj.get("scenario", "")).strip()
        options = _normalize_options(cj.get("options"))

        lines = []
        if q_type:
            lines.append(f"Type: {q_type}")
        if question:
            lines.append(f"Question: {question}")
        if scenario:
            lines.append(f"Scenario: {scenario}")
        if options:
            lines.append("Options: " + " | ".join(options))

        # drag_drop rows don't always include a flat "options" list.
        if q_type == "drag_drop" and not options:
            dd = cj.get("drag_drop", {})
            if isinstance(dd, dict):
                pairs = dd.get("pairs", [])
                if isinstance(pairs, list) and pairs:
                    items = [str(p.get("left", "")).strip() for p in pairs if isinstance(p, dict)]
                    items = [x for x in items if x]
                    if items:
                        lines.append("Items: " + ", ".join(items))

        if not lines:
            continue

        rows.append(
            {
                "instruction": DEFAULT_INSTRUCTION,
                "input": "\n".join(lines),
                "output": "",
                "answer": cj.get("answer", ""),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", required=True, help="Path to base HF model folder")
    parser.add_argument("--adapter", required=True, help="Path to LoRA adapter folder")
    parser.add_argument("--data", default="finetuning/hints_train.jsonl")
    parser.add_argument("--supabase_questions", default="", help="Path to Supabase JSON export with content_json rows.")
    parser.add_argument("--num_samples", type=int, default=0, help="Number of rows to run. 0 means all rows in order.")
    parser.add_argument("--max_new_tokens", type=int, default=80)
    parser.add_argument("--do_sample", action="store_true", help="Enable sampling for more varied outputs.")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument(
        "--strict_hint_mode",
        action="store_true",
        help="Use a stronger anti-spoiler, anti-echo instruction at inference.",
    )
    args = parser.parse_args()

    source_path = Path(args.supabase_questions) if args.supabase_questions else Path(args.data)
    if args.supabase_questions:
        rows = load_supabase_rows(source_path)
    else:
        rows = load_rows(source_path)

    if not rows:
        raise ValueError(f"No rows found in {source_path}")

    if args.num_samples > 0:
        rows = rows[: args.num_samples]
    sample_size = len(rows)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not detected. Run this script on your RTX machine with the venv activated.")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    # Avoid generation warnings when running greedy decoding.
    model.generation_config.do_sample = bool(args.do_sample)
    if not args.do_sample:
        model.generation_config.temperature = None
        model.generation_config.top_p = None

    print(f"Loaded {sample_size} samples from {source_path}")
    print("=" * 80)

    for i, row in enumerate(rows, start=1):
        prompt_instruction = STRICT_HINT_INSTRUCTION if args.strict_hint_mode else None
        prompt = build_prompt(row, instruction_override=prompt_instruction)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        gen_kwargs = dict(
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        if args.do_sample:
            gen_kwargs.update(
                temperature=args.temperature,
                top_p=args.top_p,
            )
        with torch.no_grad():
            output_ids = model.generate(**inputs, **gen_kwargs)

        generated = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)
        generated = clean_generated(generated)
        gold = (row.get("output") or "").strip()

        print(f"[Sample {i}]")
        print("Input:")
        print(row.get("input", ""))
        print("- Generated hint:")
        print(generated)
        if gold:
            print("- Reference hint:")
            print(gold)
        if row.get("answer", "") != "":
            print("- Answer key:")
            print(row.get("answer", ""))
        print("-" * 80)


if __name__ == "__main__":
    main()
