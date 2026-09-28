from pathlib import Path


def test_dockerfile_ingestion_directives():
    dockerfile_path = Path("docker/Dockerfile.ingestion")
    assert dockerfile_path.exists(), "docker/Dockerfile.ingestion does not exist"

    content = dockerfile_path.read_text(encoding="utf-8")
    assert "FROM python:3.12" in content
    assert "10001" in content or "appuser" in content
    assert "HEALTHCHECK" in content
    assert "EXPOSE 8000" in content
    assert "warden_ingestion.main:app" in content
