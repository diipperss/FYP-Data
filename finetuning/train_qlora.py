import argparse
import glob
import inspect
import os
import re

import torch
import unsloth  # Must be imported before trl/transformers for full Unsloth patching.
from datasets import Dataset, concatenate_datasets, load_dataset
from transformers import TrainingArguments, EarlyStoppingCallback
from trl import SFTTrainer
from unsloth import FastLanguageModel

try:
    from trl import DataCollatorForCompletionOnlyLM
except Exception:
    DataCollatorForCompletionOnlyLM = None


WORD_RE = re.compile(r"[A-Za-z0-9_']+")

ANALOGY_OPENERS = (
    "think of", "visualize", "imagine", "look for",
    "picture", "consider", "remember", "instead of",
)

SYSTEM_PROMPT = (
    "You are a financial educator who explains trading concepts using vivid, "
    "real-world analogies and metaphors. Never use finance jargon in your hint. "
    "Use physical, sensory, or everyday-life imagery. "
    "Start with 'Think of', 'Visualize', 'Imagine', 'Look for', or a similar opener."
)

# Set once in train() so prompt formatting + loss masking stay consistent.
_USE_CHAT_TEMPLATE = False
_RESPONSE_TEMPLATE = "### Response\n"


def detect_prompt_mode(tokenizer) -> bool:
    """
    Probe actual chat-template usability. Some tokenizers expose the attribute
    but still fail at runtime when apply_chat_template is called.
    """
    try:
        tokenizer.apply_chat_template(
            [{"role": "user", "content": "test"}],
            tokenize=False,
            add_generation_prompt=False,
        )
        return True
    except Exception:
        return False


def _token_set(text: str) -> set:
    return {w.lower() for w in WORD_RE.findall((text or "").lower()) if len(w) > 2}


def _overlap_ratio(inp: str, out: str) -> float:
    in_tok = _token_set(inp)
    out_tok = _token_set(out)
    if not out_tok:
        return 1.0
    return len(in_tok & out_tok) / len(out_tok)


def _has_analogy_style(output: str) -> bool:
    return output.lower().lstrip().startswith(ANALOGY_OPENERS)


def _output_length_ok(output: str, min_words: int = 10, max_words: int = 60) -> bool:
    words = WORD_RE.findall(output or "")
    return min_words <= len(words) <= max_words


def format_prompt(example: dict, tokenizer) -> str:
    """
    Format training rows using the model's native chat template.
    This improves instruction following for Llama-3.x models.
    """
    user_content = (example.get("input", "") or "").strip()
    assistant_content = (example.get("output", "") or "").strip()

    # Keep the original per-row instruction while enforcing global style behavior.
    row_instruction = (example.get("instruction", "") or "").strip()
    system_content = SYSTEM_PROMPT
    if row_instruction:
        system_content = f"{SYSTEM_PROMPT}\n\nTask constraint: {row_instruction}"

    # Use probed mode, decided once in train().
    if _USE_CHAT_TEMPLATE:
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

    eos_token = tokenizer.eos_token or ""
    return (
        "### Instruction\n" + system_content + "\n\n"
        + "### Input\n" + user_content + "\n\n"
        + _RESPONSE_TEMPLATE + assistant_content + eos_token
    )


def load_all_jsonl(data_dir: str) -> Dataset:
    files = sorted(glob.glob(os.path.join(data_dir, "**/*.jsonl"), recursive=True))
    files += sorted(glob.glob(os.path.join(data_dir, "*.jsonl")))
    files = list(dict.fromkeys(files))

    if not files:
        raise FileNotFoundError(f"No .jsonl files found under: {data_dir}")

    print(f"[data] Found {len(files)} JSONL file(s).")
    datasets = []
    for f in files:
        try:
            ds = load_dataset("json", data_files=f, split="train")
            datasets.append(ds)
        except Exception as e:
            print(f"[warn] Skipping {f}: {e}")

    if not datasets:
        raise RuntimeError("All JSONL files failed to load.")

    merged = concatenate_datasets(datasets)
    print(f"[data] Total rows before filtering: {len(merged)}")
    return merged


def load_single_jsonl(data_path: str) -> Dataset:
    return load_dataset("json", data_files=data_path, split="train")


def apply_filters(dataset: Dataset, max_overlap: float) -> Dataset:
    n_start = len(dataset)

    dataset = dataset.filter(
        lambda x: (x.get("output") or "").strip().upper() not in ("TODO", "")
    )
    print(f"[filter] After TODO removal: {len(dataset)} / {n_start}")

    dataset = dataset.filter(lambda x: _output_length_ok(x.get("output", "")))
    print(f"[filter] After length filter: {len(dataset)}")

    dataset = dataset.filter(
        lambda x: _overlap_ratio(x.get("input", ""), x.get("output", "")) <= max_overlap
    )
    print(f"[filter] After overlap filter: {len(dataset)}")

    analogy_count = sum(1 for x in dataset if _has_analogy_style(x.get("output", "")))
    print(
        f"[filter] Analogy-style outputs: {analogy_count} / {len(dataset)} "
        f"({100 * analogy_count / max(len(dataset), 1):.1f}%)"
    )

    return dataset


def build_model(model_dir: str, max_seq_len: int, lora_r: int):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_dir,
        max_seq_length=max_seq_len,
        load_in_4bit=True,
        dtype=None,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        lora_alpha=lora_r,
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    return model, tokenizer


def train(args):
    global _USE_CHAT_TEMPLATE, _RESPONSE_TEMPLATE

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not detected. Training requires a CUDA-capable GPU.")

    if os.path.isdir(args.data):
        dataset = load_all_jsonl(args.data)
    else:
        dataset = load_single_jsonl(args.data)

    dataset = apply_filters(dataset, args.max_input_output_overlap)

    if len(dataset) == 0:
        raise RuntimeError("Dataset is empty after filtering. Check your JSONL files.")

    model, tokenizer = build_model(args.model_dir, args.max_seq_len, args.lora_r)
    _USE_CHAT_TEMPLATE = detect_prompt_mode(tokenizer)
    if _USE_CHAT_TEMPLATE:
        _RESPONSE_TEMPLATE = "<|start_header_id|>assistant<|end_header_id|>\n\n"
        print("[format] Chat template detected - using Llama chat format.")
    else:
        _RESPONSE_TEMPLATE = "### Response\n"
        print("[format] No usable chat template - using plain prompt format.")

    dataset = dataset.map(
        lambda x: {"text": format_prompt(x, tokenizer)},
        remove_columns=dataset.column_names,
    )

    if args.eval_split > 0:
        split = dataset.train_test_split(test_size=args.eval_split, seed=42)
        train_dataset = split["train"]
        eval_dataset = split["test"]
        print(f"[split] Train: {len(train_dataset)} | Eval: {len(eval_dataset)}")
    else:
        train_dataset = dataset
        eval_dataset = None

    use_bf16 = torch.cuda.is_bf16_supported()
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        fp16=not use_bf16,
        bf16=use_bf16,
        weight_decay=0.01,
        logging_steps=10,
        save_strategy="epoch" if eval_dataset is not None else "steps",
        save_steps=200,
        save_total_limit=3,
        eval_strategy="epoch" if eval_dataset is not None else "no",
        load_best_model_at_end=eval_dataset is not None,
        metric_for_best_model="eval_loss" if eval_dataset is not None else None,
        greater_is_better=False,
        neftune_noise_alpha=5.0,
        report_to="none",
        dataloader_num_workers=0,
        group_by_length=True,
        seed=42,
    )

    callbacks = []
    if eval_dataset is not None and args.early_stopping_patience > 0:
        callbacks.append(
            EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)
        )

    trainer_kwargs = {
        "model": model,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "max_seq_length": args.max_seq_len,
        "args": training_args,
        "callbacks": callbacks if callbacks else None,
    }

    # Mask loss on prompt tokens so optimization focuses on assistant completion.
    if DataCollatorForCompletionOnlyLM is not None:
        response_template_ids = tokenizer.encode(
            _RESPONSE_TEMPLATE,
            add_special_tokens=False,
        )
        trainer_kwargs["data_collator"] = DataCollatorForCompletionOnlyLM(
            response_template=response_template_ids,
            tokenizer=tokenizer,
        )
        print(f"[collator] Completion-only loss active with template: {repr(_RESPONSE_TEMPLATE)}")
    else:
        print("[warn] DataCollatorForCompletionOnlyLM not available in this TRL version; using default collator.")

    # Handle TRL API differences across versions.
    sft_params = set(inspect.signature(SFTTrainer.__init__).parameters.keys())
    if "processing_class" in sft_params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in sft_params:
        trainer_kwargs["tokenizer"] = tokenizer

    if "dataset_text_field" in sft_params:
        trainer_kwargs["dataset_text_field"] = "text"
    else:
        trainer_kwargs["formatting_func"] = lambda x: x["text"]

    trainer = SFTTrainer(**trainer_kwargs)

    print(f"\n[train] Starting training on {len(train_dataset)} examples...")
    print(
        f"[train] LoRA r={args.lora_r}, epochs={args.epochs}, lr={args.lr}, "
        f"effective_batch={args.batch_size * args.grad_accum}\n"
    )

    trainer.train()

    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\n[done] Model saved to: {args.output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Optimized QLoRA fine-tuning for analogy-style hint generation"
    )

    parser.add_argument("--model_dir", required=True)
    parser.add_argument(
        "--data", required=True,
        help="Single .jsonl file OR a directory containing multiple .jsonl files"
    )
    parser.add_argument("--output_dir", default="finetuning/output_lora_v3")

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--lora_r", type=int, default=32)

    parser.add_argument("--max_input_output_overlap", type=float, default=0.50)

    parser.add_argument("--eval_split", type=float, default=0.05)
    parser.add_argument("--early_stopping_patience", type=int, default=3)

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
