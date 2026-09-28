from pathlib import Path
import pytest
from scripts.download_onnx_model import build_argument_parser, prepare_model_path


def test_prepare_model_path(tmp_path):
    target_dir = tmp_path / "models"
    model_path = prepare_model_path("BAAI/bge-base-en-v1.5", "int8", str(target_dir))
    assert Path(model_path).parent.exists()
    assert "bge-base-en-v1.5" in model_path
    assert "int8" in model_path


def test_build_argument_parser():
    parser = build_argument_parser()
    args = parser.parse_args(["--model", "BAAI/bge-base-en-v1.5", "--quantize", "int8", "--output-dir", "/tmp/models"])
    assert args.model == "BAAI/bge-base-en-v1.5"
    assert args.quantize == "int8"
    assert args.output_dir == "/tmp/models"
