import uuid

import numpy as np

from warden_ingestion.embedder import OnnxEmbedder, generate_point_id
from warden_ingestion.models import ChunkPayload


def test_deterministic_uuidv5_generation():
    doc_id = "DOC-HR-LEAVE-2026"
    chunk_index = 2
    pid = generate_point_id(doc_id, chunk_index)
    expected = str(uuid.uuid5(uuid.NAMESPACE_OID, f"{doc_id}_{chunk_index}"))
    assert pid == expected


def test_embedder_vector_dimensions_and_unit_norm():
    class MockSession:
        def __init__(self):
            # ONNX model input names
            class InputMeta:
                def __init__(self, name):
                    self.name = name
            self._inputs = [InputMeta("input_ids"), InputMeta("attention_mask")]

        def get_inputs(self):
            return self._inputs

        def run(self, output_names, input_feed):
            batch_size = len(input_feed["input_ids"])
            seq_len = len(input_feed["input_ids"][0])
            # Return token embeddings (batch, seq_len, 768)
            raw = np.random.randn(batch_size, seq_len, 768).astype(np.float32)
            return [raw]

    class MockTokenizer:
        def encode_batch(self, texts):
            class Encoding:
                def __init__(self):
                    self.ids = [101, 2054, 102]
                    self.attention_mask = [1, 1, 1]
            return [Encoding() for _ in texts]

    embedder = OnnxEmbedder(session=MockSession(), tokenizer=MockTokenizer(), batch_size=2)
    chunks = [
        ChunkPayload("DOC-001", 0, "PTO is 18 days.", ["Employee"], "file:///doc.md", 10, True),
        ChunkPayload(
            "DOC-001", 1, "Carryover is 5 days.", ["Employee"], "file:///doc.md", 12, True
        ),
        ChunkPayload("DOC-002", 0, "Health insurance.", ["Employee"], "file:///doc2.md", 8, True),
    ]
    points = embedder.embed_chunks(chunks)
    assert len(points) == 3
    for p in points:
        vec = np.array(p.dense_vector)
        assert vec.shape == (768,)
        norm = np.linalg.norm(vec)
        assert np.isclose(norm, 1.0, atol=1e-4)
        assert p.point_id == generate_point_id(p.doc_id, p.chunk_index)
