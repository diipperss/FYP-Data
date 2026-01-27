import os
import yaml
import re
from llama_cpp import Llama
from dotenv import load_dotenv

load_dotenv()

MODEL_PATH = os.getenv("MODEL_PATH")

CTX_SIZE = 4096
MAX_TOKENS = 800

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=CTX_SIZE,
    n_threads=2,
    n_gpu_layers=35,
    verbose=False,
)


def build_context(data: dict) -> str:
    """Build context string from chunks data (YAML dict)."""
    title = data.get("title", "")
    summary = data.get("summary", "")
    key_points = data.get("key_points", []) or []
    examples = data.get("examples", []) or []
    definitions = data.get("definitions", []) or []

    parts = []
    if title:
        parts.append(f"Title: {title}")
    if summary:
        parts.append(f"Summary: {summary}")
    if key_points:
        parts.append("Key points:")
        parts.extend([f"- {p}" for p in key_points])
    if examples:
        parts.append("Examples:")
        parts.extend([f"- {e}" for e in examples])
    if definitions:
        parts.append("Definitions:")
        if isinstance(definitions, list):
            for d in definitions:
                if isinstance(d, dict):
                    parts.append(f"- {d.get('term', '')}: {d.get('definition', '')}")
    return "\n".join(parts).strip()


def question_prompt(context: str, level: str) -> str:
    """Generate prompt for question generation at specific level."""
    return f"""You are creating Duolingo-style practice questions for a {level} learner.
Generate exactly ONE question for EACH type below, with answers.

Types required (one each):
1) fill_blank_mcq
2) drag_drop
3) case_study_mcq
4) true_false

RULES:
- Output valid YAML only (no extra text).
- Each question must be a single sentence.
- Keep language appropriate for {level} learners.
- For fill_blank_mcq, include "___" in the question and 4 options (A, B, C, D).
- For drag_drop, include 3 pairs to match.
- For case_study_mcq, include a short scenario AND a question, with 4 options (A, B, C, D).
- For true_false, answer must be "true" or "false".

YAML format:
questions:
  - type: fill_blank_mcq
    question: <string with ___>
    options: [A, B, C, D]
    answer: <A/B/C/D>
  - type: drag_drop
    question: <string>
    drag_drop:
      prompt: <string>
      pairs:
        - left: <string>
          right: <string>
        - left: <string>
          right: <string>
        - left: <string>
          right: <string>
    answer: <string>
  - type: case_study_mcq
    scenario: <string>
    question: <string>
    options: [A, B, C, D]
    answer: <A/B/C/D>
  - type: true_false
    question: <string>
    answer: <true/false>

Content:
{context}
"""


def generate_questions(context: str, level: str) -> str:
    """Generate questions for a specific level."""
    prompt = question_prompt(context, level)
    output = llm(
        prompt,
        max_tokens=MAX_TOKENS,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        min_p=0.0,
    )
    if isinstance(output, dict):
        return output.get("choices", [{}])[0].get("text", "").strip()
    return str(output).strip()


def output_path_for(input_path: str) -> str:
    """Generate output path for questions_by_level file."""
    base, ext = os.path.splitext(input_path)
    return f"{base.replace('chunks_by_level', 'questions_by_level')}{ext}"


def sanitize_yaml_text(text: str) -> str:
    """Quote scalar values that contain ':' to avoid YAML parse errors."""
    # Strip code fences if present
    text = re.sub(r"^\s*```(?:yaml)?\s*$", "", text, flags=re.M).strip()
    lines = text.splitlines()
    fixed = []
    for line in lines:
        if not line.strip():
            fixed.append(line)
            continue
        m = re.match(r"^(\s*[^:#\n]+:\s*)(.+)$", line)
        if not m:
            fixed.append(line)
            continue
        key, value = m.groups()
        v = value.strip()
        if v.startswith('"') or v.startswith("'"):
            fixed.append(line)
            continue
        if ":" in v:
            v_escaped = v.replace("\\", "\\\\").replace('"', '\\"')
            fixed.append(f"{key}\"{v_escaped}\"")
        else:
            fixed.append(line)
    return "\n".join(fixed)


def extract_level_content(yaml_string: str, level: str) -> dict:
    """Parse YAML string and extract level-specific content."""
    try:
        # If it's a YAML block string, parse it
        data = yaml.safe_load(sanitize_yaml_text(yaml_string))
        return data if isinstance(data, dict) else {}
    except Exception as e:
        print(f"Error parsing YAML for {level}: {e}")
        return {}


def process_file(chunks_by_level_path: str) -> dict:
    """Process chunks_by_level.yaml and generate questions for each level."""
    if not os.path.isfile(chunks_by_level_path):
        raise RuntimeError(f"File not found: {chunks_by_level_path}")

    # Read the chunks_by_level.yaml file
    with open(chunks_by_level_path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    # Parse the file
    data = yaml.safe_load(sanitize_yaml_text(content))
    
    if not isinstance(data, dict):
        raise RuntimeError("Invalid chunks_by_level.yaml format - not a dict")
    
    # Check if levels are top-level keys (new format) or nested under by_level (old format)
    if "by_level" in data:
        by_level = data["by_level"]
    elif any(level in data for level in ["beginner", "intermediate", "advanced"]):
        by_level = data
    else:
        raise RuntimeError("Invalid chunks_by_level.yaml format - no level keys found")

    results = {}

    # Process each level
    for level in ["beginner", "intermediate", "advanced"]:
        if level not in by_level:
            print(f"  Skipping {level} (not found in chunks_by_level)")
            continue

        print(f"  Processing {level}...", end=" ", flush=True)
        level_content = by_level[level]

        # Parse the YAML content for this level
        if isinstance(level_content, str):
            # Strip code fences if present
            level_content = sanitize_yaml_text(level_content)
            level_data = yaml.safe_load(level_content)
        else:
            level_data = level_content

        if not isinstance(level_data, dict):
            print("✗ Error: Could not parse level content")
            continue

        # Build context and generate questions
        context = build_context(level_data)
        questions_yaml = generate_questions(context, level)
        results[level] = questions_yaml
        print("✓")

    return results


def save_by_level_file(results: dict, output_path: str):
    """Save questions in by_level format."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("by_level:\n")
        for level, yaml_text in results.items():
            f.write(f"  {level}: |-\n")
            for line in (yaml_text or "").splitlines():
                f.write(f"    {line}\n")


def main():
    if not MODEL_PATH:
        raise RuntimeError("MODEL_PATH is not set in .env")

    PROCESSED_DIR = "data/processed"

    import argparse
    parser = argparse.ArgumentParser(description="Generate questions from chunks_by_level.yaml files.")
    parser.add_argument("--file", help="Process a single chunks_by_level.yaml file")
    parser.add_argument("--all", action="store_true", help="Process all chunks_by_level.yaml files in data/processed/")
    args = parser.parse_args()

    # Process all files if --all flag or no specific file provided
    if args.all or not args.file:
        file_count = 0
        for root, _, files in os.walk(PROCESSED_DIR):
            for name in files:
                if name != "chunks_by_level.yaml":
                    continue
                path = os.path.join(root, name)
                out_path = output_path_for(path)
                
                # Skip if already processed
                if os.path.exists(out_path):
                    print(f"Skipping (already exists): {path}")
                    continue
                
                file_count += 1
                print(f"\nProcessing: {path}")
                
                try:
                    results = process_file(path)
                    save_by_level_file(results, out_path)
                    print(f"✓ Saved to: {out_path}")
                except Exception as e:
                    print(f"✗ Error processing {path}: {e}")
        
        print(f"\n\nCompleted processing {file_count} files.")
        return

    # Use provided file only
    target_file = args.file

    if not os.path.isfile(target_file):
        print(f"✗ File not found: {target_file}")
        return

    out_path = output_path_for(target_file)
    print(f"Processing: {target_file}")

    try:
        results = process_file(target_file)
        save_by_level_file(results, out_path)
        print(f"✓ Saved to: {out_path}")
    except Exception as e:
        print(f"✗ Error: {e}")


if __name__ == "__main__":
    main()
