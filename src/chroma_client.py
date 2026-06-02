"""
chroma_client.py

Singleton ChromaDB HTTP client.
Connects to a ChromaDB instance running as a standalone service.
"""

import os
import socket
import logging
import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

logger = logging.getLogger(__name__)

_client = None

# A short connect probe so an unreachable ChromaDB fails fast instead of
# blocking on the OS connection timeout (~30-60s, WinError 10060 on Windows),
# which otherwise stalls app startup. Tunable via CHROMADB_CONNECT_TIMEOUT.
_CONNECT_TIMEOUT = float(os.getenv("CHROMADB_CONNECT_TIMEOUT", "2.0"))

EMBEDDING_DIMENSION_KEY = "odysseus:embedding_dimension"
EMBEDDING_MODEL_KEY = "odysseus:embedding_model"
EMBEDDING_URL_KEY = "odysseus:embedding_url"
EMBEDDING_BACKEND_KEY = "odysseus:embedding_backend"
EMBEDDING_SIGNATURE_KEY = "odysseus:embedding_signature"


def _int_or_none(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def embedding_dimension(embedding_model: Any) -> int | None:
    """Return the active embedding dimension, probing if needed."""
    if embedding_model is None:
        return None

    dim = _int_or_none(getattr(embedding_model, "_dim", None))
    if dim:
        return dim

    getter = getattr(embedding_model, "get_sentence_embedding_dimension", None)
    if callable(getter):
        return _int_or_none(getter())

    return None


def embedding_signature(embedding_model: Any, dimension: int | None = None) -> str:
    """Stable fingerprint for the active embedding backend."""
    dim = embedding_dimension(embedding_model) if dimension is None else dimension
    parts = [
        embedding_model.__class__.__name__ if embedding_model is not None else "",
        str(getattr(embedding_model, "url", "") or ""),
        str(getattr(embedding_model, "model", "") or ""),
        str(dim or ""),
    ]
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()


def embedding_metadata(embedding_model: Any, base_metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Metadata stamped on Chroma collections created by Odysseus."""
    metadata = dict(base_metadata or {})
    dim = embedding_dimension(embedding_model)
    if dim:
        metadata[EMBEDDING_DIMENSION_KEY] = dim
    if embedding_model is not None:
        metadata[EMBEDDING_BACKEND_KEY] = embedding_model.__class__.__name__
    model = getattr(embedding_model, "model", None)
    if model:
        metadata[EMBEDDING_MODEL_KEY] = str(model)
    url = getattr(embedding_model, "url", None)
    if url:
        metadata[EMBEDDING_URL_KEY] = str(url)
    metadata[EMBEDDING_SIGNATURE_KEY] = embedding_signature(embedding_model, dim)
    return metadata


def _collection_metadata(collection: Any) -> dict[str, Any]:
    metadata = getattr(collection, "metadata", None)
    return dict(metadata) if isinstance(metadata, Mapping) else {}


def _metadata_dimension(metadata: Mapping[str, Any]) -> int | None:
    return _int_or_none(metadata.get(EMBEDDING_DIMENSION_KEY))


def _metadata_signature(metadata: Mapping[str, Any]) -> str | None:
    sig = metadata.get(EMBEDDING_SIGNATURE_KEY)
    return str(sig) if sig else None


def _sequence_len(value: Any) -> int | None:
    shape = getattr(value, "shape", None)
    if shape is not None and len(shape) >= 1:
        try:
            return int(shape[-1])
        except (TypeError, ValueError):
            pass
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return len(value)
    try:
        return len(value)
    except TypeError:
        return None


def _first_embedding_dimension(collection: Any) -> int | None:
    """Infer legacy collection dimensionality from the first stored vector."""
    try:
        if collection.count() <= 0:
            return None
    except Exception:
        return None

    try:
        data = collection.get(limit=1, include=["embeddings"])
    except TypeError:
        data = collection.get(include=["embeddings"])
    except Exception:
        return None

    embeddings = (data or {}).get("embeddings")
    if embeddings is None:
        return None

    shape = getattr(embeddings, "shape", None)
    if shape is not None and len(shape) == 2:
        return _int_or_none(shape[1])

    try:
        if len(embeddings) == 0:
            return None
        first = embeddings[0]
    except Exception:
        return None

    return _sequence_len(first)


def _stamp_collection_metadata(collection: Any, desired_metadata: Mapping[str, Any]) -> None:
    current = _collection_metadata(collection)
    merged = {**current, **dict(desired_metadata)}
    if merged == current:
        return
    modifier = getattr(collection, "modify", None)
    if callable(modifier):
        try:
            modifier(metadata=merged)
            return
        except Exception as exc:
            logger.debug("Could not stamp Chroma collection metadata: %s", exc)
    try:
        collection.metadata = merged
    except Exception:
        pass


def _get_or_create_collection(client: Any, name: str, metadata: Mapping[str, Any]) -> Any:
    getter = getattr(client, "get_collection", None)
    if callable(getter):
        try:
            return getter(name=name)
        except Exception:
            return client.get_or_create_collection(name=name, metadata=metadata)
    return client.get_or_create_collection(name=name, metadata=metadata)


def ensure_embedding_collection(
    client: Any,
    name: str,
    embedding_model: Any,
    metadata: Mapping[str, Any] | None = None,
) -> Any:
    """Return a Chroma collection compatible with ``embedding_model``.

    ChromaDB fixes collection dimensionality on first insert. If Odysseus later
    switches embedding backends, reusing that collection can fail at add/query
    time, or silently compare vectors from incompatible models when dimensions
    happen to match. Existing legacy collections without this metadata are
    checked by reading one stored embedding.
    """
    desired_metadata = embedding_metadata(embedding_model, metadata)
    desired_dim = _metadata_dimension(desired_metadata)
    desired_sig = _metadata_signature(desired_metadata)

    collection = _get_or_create_collection(client, name, desired_metadata)
    current_metadata = _collection_metadata(collection)
    current_sig = _metadata_signature(current_metadata)
    current_dim = _metadata_dimension(current_metadata)

    needs_reset = False
    reason = ""

    if current_sig and desired_sig and current_sig != desired_sig:
        needs_reset = True
        reason = "embedding signature changed"
    else:
        if current_dim is None:
            current_dim = _first_embedding_dimension(collection)
        if desired_dim and current_dim and desired_dim != current_dim:
            needs_reset = True
            reason = f"embedding dimension changed ({current_dim} -> {desired_dim})"

    if needs_reset:
        logger.warning("Resetting Chroma collection %s: %s", name, reason)
        try:
            client.delete_collection(name)
        except Exception as exc:
            logger.debug("delete_collection(%s) before reset failed: %s", name, exc)
        return client.get_or_create_collection(name=name, metadata=desired_metadata)

    _stamp_collection_metadata(collection, desired_metadata)
    return collection


def _port_open(host: str, port: int, timeout: float = None) -> bool:
    """Return True if a TCP connection to host:port succeeds within timeout."""
    try:
        with socket.create_connection((host, port), timeout=timeout or _CONNECT_TIMEOUT):
            return True
    except OSError:
        return False


def get_chroma_client():
    """Get or create the singleton ChromaDB HTTP client.

    Raises RuntimeError with a clear install hint if the `chromadb` package
    is not installed — it's an optional dependency (RAG + memory vectors).
    """
    global _client
    if _client is not None:
        return _client

    try:
        import chromadb
    except ImportError as e:
        raise RuntimeError(
            "ChromaDB integration is not installed. Install the optional "
            "dependency with: pip install chromadb-client"
        ) from e

    host = os.getenv("CHROMADB_HOST", "localhost")
    port = int(os.getenv("CHROMADB_PORT", "8100"))

    if not _port_open(host, port):
        raise RuntimeError(
            f"ChromaDB is not reachable at {host}:{port}. Start the ChromaDB "
            f"service (e.g. `docker compose up chromadb`) or set CHROMADB_HOST / "
            f"CHROMADB_PORT to point at a running instance."
        )

    client = chromadb.HttpClient(host=host, port=port)

    # Health check before caching — if the port is open but the service isn't
    # healthy yet (e.g. still starting), don't poison the singleton with a dead
    # client; leave _client unset so the next call retries.
    client.heartbeat()
    _client = client
    logger.info(f"ChromaDB connected: {host}:{port}")
    return _client


def reset_client():
    """Reset the singleton (e.g. after config change)."""
    global _client
    _client = None
