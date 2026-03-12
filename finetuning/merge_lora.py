import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import shutil

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def _pick_dtype(device: str, dtype_arg: str) -> torch.dtype:
    if dtype_arg == "float16":
        return torch.float16
    if dtype_arg == "bfloat16":
        return torch.bfloat16
    if dtype_arg == "float32":
        return torch.float32

    if device == "cpu":
        return torch.float32

    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def _resolve_device(device_arg: str) -> str:
    if device_arg in ("cpu", "cuda"):
        return device_arg
    return "cuda" if torch.cuda.is_available() else "cpu"


def _ensure_output_path(out_dir: Path, force: bool) -> None:
    if not out_dir.exists():
        return
    if not out_dir.is_dir():
        raise ValueError(f"Output path exists and is not a directory: {out_dir}")
    has_files = any(out_dir.iterdir())
    if has_files and not force:
        raise ValueError(
            f"Output directory is not empty: {out_dir}\n"
            "Use --force to allow writing into an existing non-empty folder."
        )


def _disk_free_gb(path: Path) -> float:
    usage = shutil.disk_usage(path)
    return usage.free / (1024 ** 3)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge a LoRA adapter into a base HF model and save a full merged model."
    )
    parser.add_argument("--base_model", required=True, help="Path to base HF model")
    parser.add_argument("--adapter", required=True, help="Path to LoRA adapter directory")
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory to write merged model and tokenizer",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device selection. Use cpu for the lowest crash risk in WSL.",
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto",
        help="Torch dtype to load/save model.",
    )
    parser.add_argument(
        "--max_shard_size",
        default="2GB",
        help="Max shard size for saved safetensors files.",
    )
    parser.add_argument(
        "--offload_dir",
        default="finetuning/.merge_offload",
        help="Folder for CPU/disk offload when using CUDA device_map=auto.",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Enable only if the model requires custom code.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow writing to a non-empty output directory.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only run safety checks and print planned config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    base_model = Path(args.base_model).expanduser().resolve()
    adapter = Path(args.adapter).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    offload_dir = Path(args.offload_dir).expanduser().resolve()

    if not base_model.exists():
        raise FileNotFoundError(f"Base model path not found: {base_model}")
    if not adapter.exists():
        raise FileNotFoundError(f"Adapter path not found: {adapter}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    _ensure_output_path(output_dir, args.force)

    device = _resolve_device(args.device)
    dtype = _pick_dtype(device, args.dtype)
    free_gb = _disk_free_gb(output_dir.parent)

    print("[preflight] Base model:", base_model)
    print("[preflight] Adapter:", adapter)
    print("[preflight] Output dir:", output_dir)
    print("[preflight] Offload dir:", offload_dir)
    print("[preflight] Device:", device)
    print("[preflight] Dtype:", dtype)
    print(f"[preflight] Free disk: {free_gb:.2f} GB")

    if args.dry_run:
        print("[dry-run] No model loading or writing performed.")
        return

    model_kwargs = {
        "dtype": dtype,
        "trust_remote_code": args.trust_remote_code,
        "low_cpu_mem_usage": True,
    }
    if device == "cpu":
        model_kwargs["device_map"] = {"": "cpu"}
    else:
        model_kwargs["device_map"] = "auto"
        offload_dir.mkdir(parents=True, exist_ok=True)
        model_kwargs["offload_folder"] = str(offload_dir)
        model_kwargs["offload_buffers"] = True

    tokenizer = AutoTokenizer.from_pretrained(str(base_model), use_fast=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    print("[load] Loading base model...")
    base = AutoModelForCausalLM.from_pretrained(str(base_model), **model_kwargs)
    print("[load] Loading adapter...")
    peft_kwargs = {}
    if device != "cpu":
        peft_kwargs["offload_folder"] = str(offload_dir)
    peft_model = PeftModel.from_pretrained(base, str(adapter), **peft_kwargs)

    print("[merge] Merging adapter into base weights...")
    merged = peft_model.merge_and_unload()

    output_dir.mkdir(parents=True, exist_ok=True)
    print("[save] Saving merged model...")
    merged.save_pretrained(
        str(output_dir),
        safe_serialization=True,
        max_shard_size=args.max_shard_size,
    )
    tokenizer.save_pretrained(str(output_dir))

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_model": str(base_model),
        "adapter": str(adapter),
        "output_dir": str(output_dir),
        "device": device,
        "dtype": str(dtype),
        "max_shard_size": args.max_shard_size,
    }
    metadata_path = output_dir / "merge_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[done] Merged model written to: {output_dir}")
    print(f"[done] Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
