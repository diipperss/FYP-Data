from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

model_id = "google/gemma-2-12b-it"
adapter_path = "qlora_adapter"

tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
base = AutoModelForCausalLM.from_pretrained(
    model_id,
    load_in_4bit=True,
    device_map="auto",
)

model = PeftModel.from_pretrained(base, adapter_path)

prompt = "Question: A buyer offers $50 and a seller asks $55. Which is the bid?\nHint:"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=40)
print(tokenizer.decode(out[0], skip_special_tokens=True))
