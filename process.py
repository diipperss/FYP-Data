import os
import re
from llama_cpp import Llama
from dotenv import load_dotenv

load_dotenv()

RAW_DIR = os.getenv("RAW_DIR")
PROCESSED_DIR = os.getenv("PROCESSED_DIR")
MODEL_PATH = os.getenv("MODEL_PATH")

# ===============================
# CONFIG
# ===============================

CTX_SIZE = 4096
MAX_FINAL_TOKENS = 2000
MAX_CHUNK_CHARS = 6000 

# ===============================
# INITIALIZE MODEL
# ===============================
# Note: Increased n_ctx to match config
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=CTX_SIZE,
    n_threads=12,
    n_gpu_layers=35,
    verbose=False
)

# ===============================
# PROMPT (UPDATED)
# ===============================
def final_summary_prompt(raw_text, topic, subtopic, source):
    # We use a strict ChatML/Llama-3 format to separate instructions from data
    # We also PRE-FILL the start of the response with "title:" to force compliance
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a strict data extraction engine. You convert educational text into YAML.
- DO NOT speak to the user.
- DO NOT think out loud.
- DO NOT output markdown code fences (```).
- Output valid YAML immediately.

YAML Structure required:
title: <string>
summary: <paragraph>
key_points:
  - <string>
examples:
  - <string>
definitions:
  - term: <string>
    definition: <string>
common_mistakes:
  - <string>
questions_to_think:
  - <string>
source: <string>
<|eot_id|><|start_header_id|>user<|end_header_id|>
Text to process:
{raw_text}

Topic: {topic}
Subtopic: {subtopic}
Source Hint: {source}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
title:"""

# ===============================
# HELPERS
# ===============================
def read_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()

def save_yaml(text, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text.strip())

def safe_truncate(text, max_chars):
    return text[:max_chars]

def extract_source(raw_text):
    # specific logic to find source lines
    for line in raw_text.splitlines():
        if line.strip().lower().startswith("source:"):
            return line.split(":", 1)[1].strip()
    return None

def clean_text(raw_text):
    source = extract_source(raw_text)
    lines = []
    # Remove metadata lines from the content we feed the LLM
    for line in raw_text.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("text:"): continue
        if stripped.lower().startswith("source:"): continue
        lines.append(line.rstrip())

    # Basic paragraph cleanup
    paragraphs = []
    buffer = []
    for line in lines:
        if not line.strip():
            if buffer:
                paragraphs.append(" ".join(buffer).strip())
                buffer = []
            continue
        buffer.append(line)
    if buffer:
        paragraphs.append(" ".join(buffer).strip())

    cleaned = "\n\n".join(paragraphs).strip()
    return cleaned

def generate(prompt, max_tokens):
    # We add stop tokens so it stops generating once the YAML is done
    # We allow the model to generate up to max_tokens
    output = llm(
        prompt, 
        max_tokens=max_tokens,
        stop=["<|eot_id|>", "<|end_of_text|>", "user:", "User:"] 
    )
    
    generated_text = output["choices"][0]["text"]
    
    # CRITICAL: We manually add "title:" back because we put it in the prompt (pre-fill)
    # The model only generates what comes AFTER "title:"
    full_response = "title:" + generated_text
    return full_response.strip()

# ===============================
# PIPELINE
# ===============================
if __name__ == "__main__":
    print(f"Starting pipeline reading from: {RAW_DIR}")
    
    for main_topic in os.listdir(RAW_DIR):
        main_path = os.path.join(RAW_DIR, main_topic)
        if not os.path.isdir(main_path):
            continue

        for subtopic in os.listdir(main_path):
            sub_path = os.path.join(main_path, subtopic)
            if not os.path.isdir(sub_path):
                continue

            print(f"\nProcessing: {main_topic} -> {subtopic}")

            for file in sorted(os.listdir(sub_path)):
                if not file.endswith(".txt"):
                    continue

                raw_path = os.path.join(sub_path, file)
                raw_text = read_file(raw_path)
                
                if not raw_text:
                    continue

                cleaned_text = clean_text(raw_text)
                if not cleaned_text:
                    continue
                
                # Truncate to avoid context limit errors
                cleaned_text = safe_truncate(cleaned_text, MAX_CHUNK_CHARS)
                source_hint = extract_source(raw_text) or "unknown"

                print(f"  Generating summary for: {file}...")
                
                prompt = final_summary_prompt(cleaned_text, main_topic, subtopic, source_hint)
                
                try:
                    final_yaml = generate(prompt, MAX_FINAL_TOKENS)
                    
                    # Sanity check: verify it looks like YAML
                    if "summary:" not in final_yaml:
                        print(f"  [WARN] Output for {file} might be malformed.")
                    
                    out_path = os.path.join(PROCESSED_DIR, main_topic, subtopic, f"{os.path.splitext(file)[0]}.yaml")
                    save_yaml(final_yaml, out_path)
                    print(f"  -> Saved to {out_path}")
                    
                except Exception as e:
                    print(f"  [ERROR] Failed processing {file}: {e}")

    print("\nCompleted all processing.")