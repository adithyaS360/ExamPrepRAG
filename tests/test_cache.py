"""Regression test for the semantic cache.

The cache-write call in RagService.query() used to be entirely commented
out, so `cache.hit` was always False no matter how many times the same
question was asked — even though the README documents (and the resume
claims) a working semantic cache that skips retrieval/LLM on a repeat
question. These tests exercise the fix directly against RagService rather
than through the ingestion pipeline, so they don't depend on any real
corpus being present.
"""
from __future__ import annotations

import numpy as np
import pytest

from app.config import Settings
from app.service import CacheEntry, RagService


@pytest.fixture()
def service(tmp_path, monkeypatch):
    # Point the store at an empty temp dir so _ensure_index has nothing to
    # do, and avoid touching the real data/ directory.
    monkeypatch.chdir(tmp_path)
    settings = Settings(
        index_dir=tmp_path / "index",
        embedding_model="unused",
        embedding_backend="hash",
        groq_api_key=None,
        groq_model="unused",
        top_k=5,
        cache_similarity_threshold=0.97,
        llm_timeout_seconds=1.0,
    )
    return RagService(settings)


def _fake_response(**overrides):
    base = {
        "question": "CNS module 1 topics",
        "subject": "CNS",
        "module": 1,
        "answer": "some grounded answer",
        "mode": "llm",
        "fallback_reason": None,
        "citations": [],
        "retrieved_chunks": [],
        "cache": {"hit": False},
        "latency_ms": {"embedding": 1.0, "retrieval": 1.0, "llm": 1.0, "total": 3.0},
    }
    base.update(overrides)
    return base


def test_identical_vector_is_a_cache_hit(service):
    vector = np.ones(8, dtype="float32")
    vector /= np.linalg.norm(vector)

    service.cache.append(
        CacheEntry(vector=vector, subject="CNS", module=1, response=_fake_response())
    )

    hit, score = service._find_cache_hit(vector, "CNS", 1)

    assert hit is not None
    assert score == pytest.approx(1.0)


def test_different_subject_is_not_a_cache_hit(service):
    vector = np.ones(8, dtype="float32")
    vector /= np.linalg.norm(vector)

    service.cache.append(
        CacheEntry(vector=vector, subject="CNS", module=1, response=_fake_response())
    )

    hit, _ = service._find_cache_hit(vector, "BDA", 1)

    assert hit is None


def test_dissimilar_question_same_subject_is_not_a_cache_hit(service):
    cached_vector = np.array([1.0, 0.0, 0.0, 0.0], dtype="float32")
    query_vector = np.array([0.0, 1.0, 0.0, 0.0], dtype="float32")

    service.cache.append(
        CacheEntry(vector=cached_vector, subject="CNS", module=1, response=_fake_response())
    )

    hit, score = service._find_cache_hit(query_vector, "CNS", 1)

    assert hit is None
    assert score == pytest.approx(0.0)


def test_query_populates_cache_and_second_call_is_a_hit(service, monkeypatch):
    # Avoid depending on a real FAISS index / corpus: stub search to return
    # no matches, and force the LLM path to retrieval_fallback (no API key
    # configured in the test Settings), which is enough to exercise the
    # cache read/write path around retrieval+LLM.
    monkeypatch.setattr(service.store, "search", lambda *a, **k: [])

    first = service.query("CNS module 1 topics")
    assert first["cache"]["hit"] is False
    assert len(service.cache) == 1

    second = service.query("CNS module 1 topics")
    assert second["cache"]["hit"] is True
    assert second["latency_ms"]["retrieval"] == 0.0
    assert second["latency_ms"]["llm"] == 0.0


def test_force_fallback_is_never_cached_or_served_from_cache(service, monkeypatch):
    monkeypatch.setattr(service.store, "search", lambda *a, **k: [])

    service.query("CNS module 1 topics", force_fallback=True)
    assert len(service.cache) == 0

    normal = service.query("CNS module 1 topics")
    assert normal["cache"]["hit"] is False
