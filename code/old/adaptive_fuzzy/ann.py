"""Approximate nearest-neighbour retrieval utilities for candidate generation."""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:  # pragma: no cover - optional dependency
    import faiss  # type: ignore
except ImportError:  # pragma: no cover
    faiss = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from sentence_transformers import SentenceTransformer  # type: ignore
except ImportError:  # pragma: no cover
    SentenceTransformer = None  # type: ignore

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _require_sentence_transformers() -> None:
    if SentenceTransformer is None:  # pragma: no cover - defensive
        raise RuntimeError(
            "sentence-transformers package is required for ANN embeddings. "
            "Install with `pip install sentence-transformers`."
        )


def _require_faiss() -> None:
    if faiss is None:  # pragma: no cover - defensive
        raise RuntimeError(
            "faiss library is required for ANN retrieval. "
            "Install with `pip install faiss-cpu` or `faiss-gpu`."
        )


def _names_hash(names: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for name in names:
        digest.update(name.encode("utf-8", errors="ignore"))
        digest.update(b"\0")
    return digest.hexdigest()


def _meta_path(embedding_path: Path) -> Path:
    return embedding_path.with_suffix(embedding_path.suffix + ".meta.json")


def generate_ann_embeddings(
    names: Sequence[str],
    embedding_path: Path,
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 512,
    device: Optional[str] = None,
    normalize: bool = True,
) -> Dict[str, object]:
    """Encode names into sentence embeddings and persist as a memmap."""

    _require_sentence_transformers()
    embedding_path = Path(embedding_path)
    embedding_path.parent.mkdir(parents=True, exist_ok=True)
    total = len(names)
    if total == 0:
        raise ValueError("No names provided to generate embeddings.")

    model = SentenceTransformer(model_name, device=device)  # type: ignore[call-arg]
    dim = int(model.get_sentence_embedding_dimension())  # type: ignore[attr-defined]

    mmap = np.memmap(embedding_path, dtype="float32", mode="w+", shape=(total, dim))
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch = names[start:end]
        if not batch:
            continue
        embeddings = model.encode(  # type: ignore[arg-type]
            batch,
            batch_size=min(batch_size, len(batch)),
            convert_to_numpy=True,
            normalize_embeddings=normalize,
        )
        mmap[start:end] = np.asarray(embeddings, dtype=np.float32)
    mmap.flush()

    metadata = {
        "rows": total,
        "dim": dim,
        "model": model_name,
        "normalize": normalize,
        "batch_size": batch_size,
        "names_hash": _names_hash(names),
        "created_at": _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    _meta_path(embedding_path).write_text(json.dumps(metadata, indent=2))
    return metadata


def _load_embedding_meta(embedding_path: Path) -> Dict[str, object]:
    meta_file = _meta_path(embedding_path)
    if not meta_file.exists():
        raise FileNotFoundError(
            f"Embedding metadata file {meta_file} not found. Regenerate embeddings or restore metadata."
        )
    with meta_file.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_faiss_index(
    embedding_path: Path,
    index_path: Path,
    nlist: int = 4096,
    m: int = 48,
    nbits: int = 8,
    training_samples: int = 100_000,
    random_seed: int = 42,
) -> None:
    """Build an IVF-PQ index over the embedding store."""

    _require_faiss()
    embedding_path = Path(embedding_path)
    index_path = Path(index_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)

    meta = _load_embedding_meta(embedding_path)
    rows = int(meta["rows"])
    dim = int(meta["dim"])
    if rows == 0:
        raise ValueError("Embedding store is empty; cannot build ANN index.")

    vectors = np.memmap(embedding_path, dtype="float32", mode="r", shape=(rows, dim))
    if nlist > rows:
        nlist = max(1, rows // 2)
    if nlist <= 0:
        raise ValueError("nlist must be positive after adjustment.")
    if training_samples > rows:
        training_samples = rows

    rng = np.random.default_rng(random_seed)
    sample_indices = rng.choice(rows, size=training_samples, replace=False)
    sample = np.asarray(vectors[sample_indices], dtype=np.float32)

    quantizer = faiss.IndexFlatIP(dim)  # type: ignore[attr-defined]
    index = faiss.IndexIVFPQ(quantizer, dim, nlist, m, nbits)  # type: ignore[attr-defined]
    if hasattr(index, "rng"):  # pragma: no cover - compatibility hook
        index.rng.seed(random_seed)
    index.train(sample)
    index.add(np.asarray(vectors, dtype=np.float32))
    faiss.write_index(index, str(index_path))  # type: ignore[attr-defined]


@dataclass
class AnnRetriever:
    """Query wrapper around a FAISS IVF-PQ index for candidate generation."""

    names: Sequence[str]
    embedding_path: Path
    index_path: Path
    top_k: int = 50
    nprobe: int = 32
    batch_size: int = 100_000

    def __post_init__(self) -> None:
        self.embedding_path = Path(self.embedding_path)
        self.index_path = Path(self.index_path)
        self._meta = _load_embedding_meta(self.embedding_path)
        expected_hash = self._meta.get("names_hash")
        actual_hash = _names_hash(self.names)
        if expected_hash and expected_hash != actual_hash:
            raise ValueError(
                "ANN embedding store was generated for a different ordering of names. "
                "Rebuild embeddings to match the current dataset."
            )
        rows = int(self._meta["rows"])
        dim = int(self._meta["dim"])
        self._vectors = np.memmap(self.embedding_path, dtype="float32", mode="r", shape=(rows, dim))
        self._index = self._load_index()
        self._lookup: Dict[str, int] = {name: idx for idx, name in enumerate(self.names)}

    def _load_index(self):
        _require_faiss()
        if not self.index_path.exists():
            raise FileNotFoundError(
                f"ANN index file {self.index_path} not found. Build the index before enabling ANN retrieval."
            )
        index = faiss.read_index(str(self.index_path))  # type: ignore[attr-defined]
        try:
            index.nprobe = int(self.nprobe)  # type: ignore[attr-defined]
        except AttributeError:  # pragma: no cover - defensive
            warnings.warn("Unable to set nprobe on FAISS index; retrieval accuracy may be affected.")
        return index

    def generate_candidates(
        self,
        sources: Sequence[str],
        limit: Optional[int] = None,
        allowed_targets: Optional[Sequence[str]] = None,
        stream: bool = False
    ):
        
        """Return ANN-derived candidate pairs drawn from the supplied subset."""

        # map names -> indices once
        name_to_idx = {n: i for i, n in enumerate(self.names)}
        src_idx = [name_to_idx[n] for n in sources if n in name_to_idx]
        if not src_idx:
            return [] if not stream else iter(())

        # allowed targets (default: whole corpus)
        allowed = None if allowed_targets is None else set(allowed_targets)

        def _pairs_for_batch(batch_idx):
            vectors = np.asarray(self._vectors[batch_idx], dtype=np.float32)
            D, I = self._index.search(vectors, self.top_k + 1)  # faiss
            for row_i, src in enumerate(batch_idx):
                a = self.names[src]
                for neigh, score in zip(I[row_i], D[row_i]):
                    j = int(neigh)
                    if j < 0 or j >= len(self.names) or j == src:
                        continue
                    b = self.names[j]
                    if allowed is not None and b not in allowed:
                        continue
                    yield a, b, float(score)

        if stream:
            # yield on the fly to avoid giant dicts/sorts
            for s in range(0, len(src_idx), self.batch_size):
                yield from _pairs_for_batch(src_idx[s:s + self.batch_size])
        else:
            # collect with de-dup and optional global trim
            pair_scores = {}
            produced = 0
            for s in range(0, len(src_idx), self.batch_size):
                for a, b, score in _pairs_for_batch(src_idx[s:s + self.batch_size]):
                    k = tuple(sorted((a, b)))
                    # keep best score
                    if k not in pair_scores or score > pair_scores[k]:
                        pair_scores[k] = score
                    produced += 1
                    if limit and produced >= limit:
                        break
            items = pair_scores.items()
            if limit and limit < len(items):
                items = sorted(items, key=lambda kv: kv[1], reverse=True)[:limit]
            return [(a, b, s) for (a, b), s in items]


__all__ = [
    "generate_ann_embeddings",
    "build_faiss_index",
    "AnnRetriever",
]
