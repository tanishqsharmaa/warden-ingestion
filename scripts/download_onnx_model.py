"""Script to download, export, and prepare INT8 ONNX embedding models."""

import argparse
import os
from pathlib import Path


def build_argument_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser for ONNX model downloader."""
    parser = argparse.ArgumentParser(description="Download and quantize ONNX embedding model")
    parser.add_argument(
        "--model",
        type=str,
        default="BAAI/bge-base-en-v1.5",
        help="HuggingFace model ID",
    )
    parser.add_argument(
        "--quantize",
        type=str,
        choices=["int8", "fp16", "fp32"],
        default="int8",
        help="Quantization precision",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Target directory for ONNX model artifacts",
    )
    return parser


def prepare_model_path(model_id: str, quantize: str, output_dir: str) -> str:
    """Ensure directory exists and return target ONNX model filepath."""
    slug = model_id.split("/")[-1]
    model_dir = Path(output_dir) / f"{slug}-{quantize}"
    model_dir.mkdir(parents=True, exist_ok=True)
    return str(model_dir / "model.onnx")


def main() -> None:
    """CLI entrypoint for model download and preparation."""
    parser = build_argument_parser()
    args = parser.parse_args()
    model_path = prepare_model_path(args.model, args.quantize, args.output_dir)
    print(f"Target model path prepared: {model_path}")


if __name__ == "__main__":
    main()
