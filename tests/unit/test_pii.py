import json
from pathlib import Path
import pytest
from warden_ingestion.pii import PresidioScrubberPool


@pytest.mark.pii
def test_presidio_scrubbing_100_percent_recall():
    fixture_path = Path("tests/fixtures/synthetic_pii_corpus.json")
    with open(fixture_path, "r", encoding="utf-8") as f:
        cases = json.load(f)

    pool = PresidioScrubberPool(workers=2, threshold=0.75)
    try:
        for case in cases:
            res = pool.scrub_text(case["raw"])
            for token in case["expected_tokens"]:
                assert token in res.text, f"Token {token} not found in scrubbed output: {res.text}"
            for entity in case["expected_entities"]:
                assert res.redaction_counts.get(entity, 0) > 0, f"Entity {entity} count was 0 in {case['id']}"
            if not case["expected_tokens"]:
                assert len(res.redaction_counts) == 0
                assert res.character_count_delta == 0
    finally:
        pool.shutdown()


@pytest.mark.asyncio
@pytest.mark.pii
async def test_presidio_scrub_batch_multiprocess():
    texts = [
        "Please email support@example.com for help.",
        "Contact Jane Doe at 555-432-1098 regarding benefits.",
        "Standard handbook text with no private data.",
    ]
    pool = PresidioScrubberPool(workers=2, threshold=0.75)
    try:
        results = await pool.scrub_batch(texts)
        assert len(results) == 3
        assert "<ANONYMIZED_EMAIL>" in results[0].text
        assert "<ANONYMIZED_PERSON>" in results[1].text
        assert "<ANONYMIZED_PHONE>" in results[1].text
        assert results[2].character_count_delta == 0
    finally:
        pool.shutdown()
