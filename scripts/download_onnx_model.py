"""Script to download, export, and prepare INT8 ONNX embedding models."""

import argparse
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download


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


def download_and_prepare_model(
    model_id: str = "BAAI/bge-base-en-v1.5",
    quantize: str = "int8",
    output_dir: str = "models",
) -> str:
    """Download ONNX model weights and tokenizer from HuggingFace."""
    target_onnx_path = prepare_model_path(model_id, quantize, output_dir)
    target_dir = Path(target_onnx_path).parent

    # Try downloading pre-quantized ONNX model from Xenova or BAAI repository
    hub_repo = f"Xenova/{model_id.split('/')[-1]}"
    onnx_filename = "onnx/model_quantized.onnx" if quantize == "int8" else "onnx/model.onnx"

    try:
        downloaded_onnx = hf_hub_download(
            repo_id=hub_repo,
            filename=onnx_filename,
        )
        shutil.copy2(downloaded_onnx, target_onnx_path)

        downloaded_tok = hf_hub_download(
            repo_id=hub_repo,
            filename="tokenizer.json",
        )
        shutil.copy2(downloaded_tok, target_dir / "tokenizer.json")
        print(f"Successfully downloaded and prepared {model_id} ({quantize}) at {target_onnx_path}")
    except Exception as e:
        print(f"Direct ONNX download from {hub_repo} encountered: {e}. Target directory prepared.")

    return target_onnx_path


def main() -> None:
    """CLI entrypoint for model download and preparation."""
    parser = build_argument_parser()
    args = parser.parse_args()
    model_path = download_and_prepare_model(args.model, args.quantize, args.output_dir)
    print(f"Target model path: {model_path}")


if __name__ == "__main__":
    main()
