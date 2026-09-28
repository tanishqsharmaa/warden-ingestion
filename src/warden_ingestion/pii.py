"""Multiprocess Microsoft Presidio PII scrubbing pool and anonymization primitives."""

import asyncio
from concurrent.futures import ProcessPoolExecutor
from typing import Optional

from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

from warden_ingestion.models import ScrubbedResult

# Global worker cache for engines to avoid re-initialization overhead per chunk
_WORKER_ANALYZER: Optional[AnalyzerEngine] = None
_WORKER_ANONYMIZER: Optional[AnonymizerEngine] = None


def _init_worker(model_name: str) -> None:
    """Initialize Presidio engines inside worker process."""
    global _WORKER_ANALYZER, _WORKER_ANONYMIZER
    try:
        provider = NlpEngineProvider(
            nlp_configuration={
                "nlp_engine_name": "spacy",
                "models": [{"lang_code": "en", "model_name": model_name}],
            }
        )
        nlp_engine = provider.create_engine()
        analyzer = AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=["en"])
    except Exception:
        # Fallback to default spacy model
        analyzer = AnalyzerEngine()

    # Register high-confidence pattern recognizers for robust phone & SSN extraction
    phone_pattern = Pattern(
        "us_phone_strict",
        r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
        0.85,
    )
    ssn_pattern = Pattern(
        "us_ssn_strict",
        r"\b\d{3}-\d{2}-\d{4}\b",
        0.85,
    )

    analyzer.registry.add_recognizer(
        PatternRecognizer(
            supported_entity="PHONE_NUMBER",
            patterns=[phone_pattern],
            name="custom_phone_recognizer",
        )
    )
    analyzer.registry.add_recognizer(
        PatternRecognizer(
            supported_entity="US_SSN",
            patterns=[ssn_pattern],
            name="custom_ssn_recognizer",
        )
    )

    _WORKER_ANALYZER = analyzer
    _WORKER_ANONYMIZER = AnonymizerEngine()


def _worker_scrub(text: str, threshold: float, model_name: str) -> ScrubbedResult:
    """Worker function executed inside ProcessPoolExecutor."""
    global _WORKER_ANALYZER, _WORKER_ANONYMIZER
    if _WORKER_ANALYZER is None or _WORKER_ANONYMIZER is None:
        _init_worker(model_name)

    assert _WORKER_ANALYZER is not None
    assert _WORKER_ANONYMIZER is not None

    target_entities = ["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "US_SSN"]
    analyzer_results = _WORKER_ANALYZER.analyze(
        text=text,
        entities=target_entities,
        language="en",
        score_threshold=threshold,
    )

    if not analyzer_results:
        return ScrubbedResult(
            text=text,
            redaction_counts={},
            character_count_delta=0,
        )

    # Count redaction hits by entity
    redaction_counts: dict[str, int] = {}
    for r in analyzer_results:
        redaction_counts[r.entity_type] = redaction_counts.get(r.entity_type, 0) + 1

    operators = {
        "PERSON": OperatorConfig("replace", {"new_value": "<ANONYMIZED_PERSON>"}),
        "EMAIL_ADDRESS": OperatorConfig("replace", {"new_value": "<ANONYMIZED_EMAIL>"}),
        "PHONE_NUMBER": OperatorConfig("replace", {"new_value": "<ANONYMIZED_PHONE>"}),
        "US_SSN": OperatorConfig("replace", {"new_value": "<ANONYMIZED_SSN>"}),
    }

    anonymized_result = _WORKER_ANONYMIZER.anonymize(
        text=text,
        analyzer_results=analyzer_results,
        operators=operators,
    )

    scrubbed_text = anonymized_result.text
    delta = len(scrubbed_text) - len(text)

    return ScrubbedResult(
        text=scrubbed_text,
        redaction_counts=redaction_counts,
        character_count_delta=delta,
    )


class PresidioScrubberPool:
    """ProcessPoolExecutor-backed pool for concurrent Presidio PII scrubbing."""

    def __init__(
        self,
        workers: int = 4,
        threshold: float = 0.75,
        model_name: str = "en_core_web_sm",
    ) -> None:
        self.workers = max(1, workers)
        self.threshold = threshold
        self.model_name = model_name
        self.executor = ProcessPoolExecutor(
            max_workers=self.workers,
            initializer=_init_worker,
            initargs=(self.model_name,),
        )

    def scrub_text(self, text: str) -> ScrubbedResult:
        """Synchronously scrub a single text string."""
        future = self.executor.submit(_worker_scrub, text, self.threshold, self.model_name)
        return future.result()

    async def scrub_batch(self, texts: list[str]) -> list[ScrubbedResult]:
        """Asynchronously scrub a batch of text strings across the worker pool."""
        loop = asyncio.get_running_loop()
        futures = [
            loop.run_in_executor(self.executor, _worker_scrub, text, self.threshold, self.model_name)
            for text in texts
        ]
        return await asyncio.gather(*futures)

    def shutdown(self) -> None:
        """Terminate the worker pool and release resources."""
        self.executor.shutdown(wait=True)
