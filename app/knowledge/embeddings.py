"""Embedding model behind an interface: real local embeddings by default
(FastEmbed, ONNX, no API key), a deterministic fake for offline tests —
mirrors the `LLM` interface's Bedrock/FakeLLM split from the project spec.
"""

from typing import Any

from llama_index.core.embeddings import BaseEmbedding
from pydantic import PrivateAttr

from app.config import settings

_FAKE_EMBEDDING_DIM = 384


class FakeEmbedding(BaseEmbedding):
    """Deterministic, hash-based embedding — no model download, no network.

    Not semantically meaningful (equal-length hashes aren't similarity-aware);
    it exists only to make retrieval *plumbing* (filtering, top_k, citations)
    testable offline, the same way FakeLLM keeps the recommendation chain
    testable without a real model call.
    """

    def _hash_to_vector(self, text: str) -> list[float]:
        import hashlib

        digest = hashlib.sha256(text.encode("utf-8")).digest()
        repeated = (digest * ((_FAKE_EMBEDDING_DIM // len(digest)) + 1))[:_FAKE_EMBEDDING_DIM]
        return [b / 255.0 for b in repeated]

    def _get_query_embedding(self, query: str) -> list[float]:
        return self._hash_to_vector(query)

    def _get_text_embedding(self, text: str) -> list[float]:
        return self._hash_to_vector(text)

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._hash_to_vector(query)

    async def _aget_text_embedding(self, text: str) -> list[float]:
        return self._hash_to_vector(text)


class FastEmbedEmbedding(BaseEmbedding):
    """Local ONNX embedding via `fastembed` — real semantic search, no API key,
    one-time model download, no torch dependency."""

    _model: Any = PrivateAttr()

    def __init__(self, model_name: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name=model_name)

    def _get_query_embedding(self, query: str) -> list[float]:
        return next(self._model.query_embed(query)).tolist()

    def _get_text_embedding(self, text: str) -> list[float]:
        return next(self._model.embed([text])).tolist()

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        return [vector.tolist() for vector in self._model.embed(texts)]

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> list[float]:
        return self._get_text_embedding(text)


def get_embedding_model() -> BaseEmbedding:
    if settings.embedding_backend == "fake":
        return FakeEmbedding()
    return FastEmbedEmbedding(model_name=settings.fastembed_model_name)
