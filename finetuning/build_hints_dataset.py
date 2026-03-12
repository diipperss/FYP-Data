import json
import re
from pathlib import Path
import argparse
import random
import yaml

CONTENT_ROOT = Path(__file__).resolve().parents[1] / "data" / "processed"

INSTRUCTION = (
    "Give a helpful hint for the question without giving the answer. "
    "Return exactly 1-2 short sentences. No meta-advice, no mention of question type, and no repetition."
)


def _parse_level_data(level_data):
    if isinstance(level_data, dict):
        return level_data
    if isinstance(level_data, str):
        inner = level_data
        if "```yaml" in inner:
            inner = inner.split("```yaml", 1)[1]
            inner = inner.split("```", 1)[0]
        try:
            return yaml.safe_load(inner) or {}
        except yaml.YAMLError:
            return {}
    return {}


def _format_options(options, qtype):
    if qtype == "true_false":
        return "True / False"
    if not options:
        return None
    # Options can be list[str] or list[dict]
    formatted = []
    for item in options:
        if isinstance(item, dict) and len(item) == 1:
            k, v = next(iter(item.items()))
            formatted.append(f"{k}. {v}")
        else:
            formatted.append(str(item))
    return "; ".join(formatted)


def _build_input(q):
    qtype = q.get("type")
    lines = [f"Type: {qtype}"]
    if q.get("question"):
        lines.append(f"Question: {q['question']}")
    if q.get("scenario"):
        lines.append(f"Scenario: {q['scenario']}")

    options = _format_options(q.get("options"), qtype)
    if options:
        lines.append(f"Options: {options}")

    # For drag_drop, avoid including right-side answers.
    if qtype == "drag_drop":
        dd = q.get("drag_drop") or {}
        left_items = [p.get("left") for p in dd.get("pairs", []) if p.get("left")]
        if left_items:
            lines.append("Items: " + ", ".join(left_items))
        if dd.get("prompt"):
            lines.append(f"Prompt: {dd['prompt']}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(Path(__file__).with_name("hints_train.jsonl")))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = []
    for topic_dir in CONTENT_ROOT.iterdir():
        if not topic_dir.is_dir():
            continue
        for subtopic_dir in topic_dir.iterdir():
            if not subtopic_dir.is_dir():
                continue
            qfile = subtopic_dir / "questions_by_level.yaml"
            if not qfile.exists():
                continue
            data = yaml.safe_load(qfile.read_text()) or {}
            levels = data.get("by_level") or {}
            for level_name, level_data in levels.items():
                level_dict = _parse_level_data(level_data)
                questions = level_dict.get("questions", [])
                for q in questions:
                    inp = _build_input(q)
                    rows.append({
                        "instruction": INSTRUCTION,
                        "input": inp,
                        "output": "TODO",
                        "meta": {
                            "topic": topic_dir.name,
                            "subtopic": subtopic_dir.name,
                            "level": level_name,
                            "type": q.get("type"),
                        },
                    })

    if args.limit is not None and args.limit < len(rows):
        random.seed(args.seed)
        rows = random.sample(rows, args.limit)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
