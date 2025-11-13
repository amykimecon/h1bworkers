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
        subset: Sequence[str],
        limit: Optional[int] = None,
    ) -> List[Tuple[str, str, float]]:
        """Return ANN-derived candidate pairs drawn from the supplied subset."""

        if not subset:
            return []
        indices: List[int] = []
        missing: List[str] = []
        for name in subset:
            idx = self._lookup.get(name)
            if idx is None:
                missing.append(name)
                continue
            indices.append(idx)
        if missing:
            warnings.warn(
                f"{len(missing)} names missing from ANN embeddings; rebuild embeddings to include them.",
                RuntimeWarning,
            )
        if not indices:
            return []

        vectors = np.asarray(self._vectors[indices], dtype=np.float32)
        top_k = max(1, self.top_k)
        distances, neighbor_ids = self._index.search(vectors, top_k + 1)  # type: ignore[attr-defined]

        subset_set = set(subset)
        pair_scores: Dict[Tuple[str, str], float] = {}
        for row_idx, source_idx in enumerate(indices):
            source_name = self.names[source_idx]
            neighbor_list = neighbor_ids[row_idx]
            neighbor_scores = distances[row_idx]
            for neigh_idx, score in zip(neighbor_list, neighbor_scores):
                candidate_idx = int(neigh_idx)
                if candidate_idx < 0 or candidate_idx >= len(self.names):
                    continue
                if candidate_idx == source_idx:
                    continue
                target_name = self.names[candidate_idx]
                if target_name not in subset_set:
                    continue
                ordered = tuple(sorted((source_name, target_name)))
                numeric_score = float(score)
                if ordered not in pair_scores or numeric_score > pair_scores[ordered]:
                    pair_scores[ordered] = numeric_score

        sorted_pairs = sorted(pair_scores.items(), key=lambda item: item[1], reverse=True)
        if limit is not None and limit > 0:
            sorted_pairs = sorted_pairs[:limit]
        return [(a, b, score) for (a, b), score in sorted_pairs]


__all__ = [
    "generate_ann_embeddings",
    "build_faiss_index",
    "AnnRetriever",
]
