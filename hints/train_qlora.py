import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

model_id = "google/gemma-2-12b-it"   # example base model
dataset_path = "example.jsonl"

tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

dataset = load_dataset("json", data_files=dataset_path)["train"]

def format_fn(ex):
    text = ex["prompt"] + ex["completion"]
    return {"text": text}

dataset = dataset.map(format_fn)

def tokenize(ex):
    return tokenizer(ex["text"], truncation=True, max_length=2048)
dataset = dataset.map(tokenize, remove_columns=dataset.column_names)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    load_in_4bit=True,
    device_map="auto",
)

model = prepare_model_for_kbit_training(model)
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, peft_config)

args = TrainingArguments(
    output_dir="qlora_out",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    num_train_epochs=2,
    fp16=True,
    logging_steps=10,
    save_steps=200,
)

trainer = Trainer(model=model, args=args, train_dataset=dataset)
trainer.train()
model.save_pretrained("qlora_adapter")
tokenizer.save_pretrained("qlora_adapter")
