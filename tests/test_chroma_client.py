"""Regression tests for the ChromaDB singleton client (issue #326).

Covers the fast-fail preflight (so an unreachable ChromaDB doesn't block
startup for the full OS connection timeout) and the rule that a failed
connection must not poison the cached singleton.
"""
import socket
import time

import pytest

import src.chroma_client as cc


class _FakeEmbeddingModel:
    url = "http://embeddings.test/v1"

    def __init__(self, model: str, dim: int):
        self.model = model
        self._dim = dim

    def get_sentence_embedding_dimension(self):
        return self._dim


class _FakeCollection:
    def __init__(self, name: str, metadata=None, embeddings=None):
        self.name = name
        self.metadata = dict(metadata or {})
        self.embeddings = embeddings or []

    def count(self):
        return len(self.embeddings)

    def get(self, **kwargs):
        limit = kwargs.get("limit")
        embeddings = self.embeddings[:limit] if limit else self.embeddings
        return {"ids": [f"id-{i}" for i in range(len(embeddings))], "embeddings": embeddings}

    def modify(self, metadata=None):
        self.metadata = dict(metadata or {})


class _FakeClient:
    def __init__(self, collection=None, overwrite_existing_metadata=False):
        self.collections = {}
        self.deleted = []
        self.overwrite_existing_metadata = overwrite_existing_metadata
        if collection is not None:
            self.collections[collection.name] = collection

    def get_collection(self, name):
        if name not in self.collections:
            raise KeyError(name)
        return self.collections[name]

    def get_or_create_collection(self, name, metadata=None):
        if name not in self.collections:
            self.collections[name] = _FakeCollection(name, metadata=metadata)
        elif self.overwrite_existing_metadata:
            self.collections[name].metadata = dict(metadata or {})
        return self.collections[name]

    def delete_collection(self, name):
        self.deleted.append(name)
        self.collections.pop(name, None)


def _free_port() -> int:
    """Bind to port 0, grab the assigned port, release it — nothing listens."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_port_open_false_for_closed_port_and_is_fast():
    port = _free_port()
    t0 = time.monotonic()
    assert cc._port_open("127.0.0.1", port, timeout=1.0) is False
    # The whole point: we fail fast, nowhere near the 30-60s OS timeout.
    assert time.monotonic() - t0 < 5.0


def test_port_open_true_for_listening_socket():
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    host, port = srv.getsockname()
    try:
        assert cc._port_open(host, port, timeout=1.0) is True
    finally:
        srv.close()


def test_get_chroma_client_does_not_cache_when_unreachable(monkeypatch):
    pytest.importorskip("chromadb")
    cc.reset_client()
    monkeypatch.setenv("CHROMADB_HOST", "127.0.0.1")
    monkeypatch.setenv("CHROMADB_PORT", str(_free_port()))
    with pytest.raises(RuntimeError):
        cc.get_chroma_client()
    # A failed connection must leave the singleton unset so a later call
    # (once ChromaDB is up) can succeed.
    assert cc._client is None


def test_embedding_collection_stamps_new_collection_metadata():
    model = _FakeEmbeddingModel("embed-small", 384)
    client = _FakeClient()

    collection = cc.ensure_embedding_collection(
        client,
        "odysseus_test",
        model,
        metadata={"hnsw:space": "cosine"},
    )

    assert collection.metadata["hnsw:space"] == "cosine"
    assert collection.metadata[cc.EMBEDDING_DIMENSION_KEY] == 384
    assert collection.metadata[cc.EMBEDDING_MODEL_KEY] == "embed-small"
    assert collection.metadata[cc.EMBEDDING_SIGNATURE_KEY] == cc.embedding_signature(model)


def test_embedding_collection_resets_when_stored_dimension_differs():
    old = _FakeCollection(
        "odysseus_test",
        metadata={cc.EMBEDDING_DIMENSION_KEY: 384},
        embeddings=[[0.0] * 384],
    )
    client = _FakeClient(old)

    collection = cc.ensure_embedding_collection(
        client,
        "odysseus_test",
        _FakeEmbeddingModel("embed-large", 768),
    )

    assert client.deleted == ["odysseus_test"]
    assert collection.count() == 0
    assert collection.metadata[cc.EMBEDDING_DIMENSION_KEY] == 768


def test_embedding_collection_resets_on_same_dimension_signature_change():
    old_model = _FakeEmbeddingModel("embed-a", 384)
    old_metadata = cc.embedding_metadata(old_model)
    old = _FakeCollection("odysseus_test", metadata=old_metadata, embeddings=[[0.0] * 384])
    client = _FakeClient(old)

    collection = cc.ensure_embedding_collection(
        client,
        "odysseus_test",
        _FakeEmbeddingModel("embed-b", 384),
    )

    assert client.deleted == ["odysseus_test"]
    assert collection.count() == 0
    assert collection.metadata[cc.EMBEDDING_MODEL_KEY] == "embed-b"


def test_embedding_collection_infers_legacy_dimension_before_reuse():
    old = _FakeCollection("odysseus_test", metadata={}, embeddings=[[0.0] * 384])
    client = _FakeClient(old)

    collection = cc.ensure_embedding_collection(
        client,
        "odysseus_test",
        _FakeEmbeddingModel("embed-large", 768),
    )

    assert client.deleted == ["odysseus_test"]
    assert collection.metadata[cc.EMBEDDING_DIMENSION_KEY] == 768


def test_embedding_collection_reads_existing_metadata_before_get_or_create():
    old_model = _FakeEmbeddingModel("embed-a", 384)
    old = _FakeCollection(
        "odysseus_test",
        metadata=cc.embedding_metadata(old_model),
        embeddings=[[0.0] * 384],
    )
    client = _FakeClient(old, overwrite_existing_metadata=True)

    collection = cc.ensure_embedding_collection(
        client,
        "odysseus_test",
        _FakeEmbeddingModel("embed-b", 384),
    )

    assert client.deleted == ["odysseus_test"]
    assert collection.metadata[cc.EMBEDDING_MODEL_KEY] == "embed-b"
