import os
import yaml
from llama_cpp import Llama
from dotenv import load_dotenv

load_dotenv()

MODEL_PATH = os.getenv("MODEL_PATH")
PROCESSED_DIR = os.getenv("PROCESSED_DIR", "data/processed")

# Default file to process
DEFAULT_FILE = "data/processed/How Stock Markets Work/algorithmic_trading,_high-frequency_trading_(HFT),_and_market_microstructure/chunks.yaml"

CTX_SIZE = 4096
MAX_TOKENS = 1500

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=CTX_SIZE,
    n_threads=2,
    n_gpu_layers=150,
    verbose=False,
)


def build_context(data: dict) -> str:
    """Build context string from chunks data for LLM."""
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
        for d in definitions:
            if isinstance(d, dict):
                parts.append(f"- {d.get('term', '')}: {d.get('definition', '')}")
    return "\n".join(parts).strip()


def content_prompt(context: str, level: str) -> str:
    """Generate prompt for content classification at specific level."""
    
    if level == "beginner":
        goal = """BEGINNER ONLY - NO OVERLAPS WITH OTHER LEVELS:
Select UNIQUE key points (only the most basic ones, skip all intermediate/advanced ones).
Select UNIQUE examples (only simplest ones, skip complex examples).
Select UNIQUE definitions (only: Market Order, Limit Order, Market Microstructure, Liquidity, Limit Order Book).
EXCLUDE: "Walking the Book", HFT, Spread, Slippage, and any advanced concepts."""
    elif level == "intermediate":
        goal = """INTERMEDIATE ONLY - NO OVERLAPS WITH OTHER LEVELS:
Select DIFFERENT key points (medium concepts, NOT the basic beginner ones and NOT the advanced ones).
Select DIFFERENT examples (moderate difficulty, NOT the simplest and NOT the most complex).
Select DIFFERENT definitions (intermediate terms like: Walking the Book, Algorithmic Trading, maybe Spread).
Do NOT include beginner-only or advanced-only content."""
    else:  # advanced
        goal = """ADVANCED ONLY - NO OVERLAPS WITH OTHER LEVELS:
Select UNIQUE key points (only the most complex/technical ones not covered in beginner/intermediate).
Select UNIQUE examples (only detailed/complex scenarios).
Select UNIQUE definitions (only advanced terms: detailed HFT concepts, complex market mechanics).
Do NOT include beginner or intermediate content."""
    
    return f"""You are PARTITIONING content into NON-OVERLAPPING difficulty levels.
Each level gets COMPLETELY DIFFERENT items - no sharing between levels!

{goal}

CRITICAL RULES:
1. NO OVERLAPS - each item appears in ONLY ONE level
2. Copy text WORD FOR WORD (no changes)
3. Distribute all content across the 3 levels (beginner gets some, intermediate gets others, advanced gets rest)
4. Keep title, summary, source EXACTLY the same for all
5. Each level should have different number of items based on difficulty

Partition the content:
{context}

Output YAML with UNIQUE items for {level} only (no sharing with other levels):
"""


def generate_level_content(context: str, level: str) -> str:
    """Generate YAML content for a specific level."""
    prompt = content_prompt(context, level)
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
    """Generate output path for by_level file."""
    base, ext = os.path.splitext(input_path)
    return f"{base}_by_level{ext}"


def process_one_file(path: str, level: str) -> str:
    """Process one chunks file for a specific level."""
    if not os.path.isfile(path):
        raise RuntimeError(f"File not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    data = yaml.safe_load(content) or {}
    context = build_context(data)

    content_yaml = generate_level_content(context, level)
    return content_yaml


def process_all_levels(target_path: str) -> dict:
    """Process a file for all three levels."""
    results = {}
    for level in ["beginner", "intermediate", "advanced"]:
        print(f"  Processing {level}...", end=" ", flush=True)
        results[level] = process_one_file(target_path, level)
        print("✓")
    return results


def save_by_level_file(results: dict, output_path: str):
    """Save results in by_level format."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("by_level:\n")
        for level, yaml_text in results.items():
            f.write(f"  {level}: |-\n")
            for line in (yaml_text or "").splitlines():
                f.write(f"    {line}\n")


def main():
    import argparse
    if not MODEL_PATH:
        raise RuntimeError("MODEL_PATH is not set in .env")

    parser = argparse.ArgumentParser(description="Generate beginner/intermediate/advanced content for chunks.yaml files.")
    parser.add_argument("--file", help="Process a single chunks.yaml file")
    parser.add_argument("--all", action="store_true", help="Process all chunks.yaml files in data/processed/")
    args = parser.parse_args()

    # Use default file if no args provided
    target_file = args.file or DEFAULT_FILE
    
    # Process single file if specified or default
    if not args.all:
        if not os.path.isfile(target_file):
            print(f"✗ File not found: {target_file}")
            return
        
        out_path = output_path_for(target_file)
        print(f"Processing: {target_file}")
        
        try:
            results = process_all_levels(target_file)
            save_by_level_file(results, out_path)
            print(f"✓ Saved to: {out_path}")
        except Exception as e:
            print(f"✗ Error: {e}")
        return
    
    # Process all files if --all flag
    file_count = 0
    for root, _, files in os.walk(PROCESSED_DIR):
        for name in files:
            if name != "chunks.yaml":
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
                results = process_all_levels(path)
                save_by_level_file(results, out_path)
                print(f"✓ Saved to: {out_path}")
            except Exception as e:
                print(f"✗ Error processing {path}: {e}")
    
    print(f"\n\nCompleted processing {file_count} files.")


if __name__ == "__main__":
    main()
 