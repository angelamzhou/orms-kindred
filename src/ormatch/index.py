"""Dense paper index: embeddings.npy (float16) + paper_ids.json, cosine search via numpy."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class PaperIndex:
    embeddings: np.ndarray  # (n, d) float32, L2-normalised
    paper_ids: List[str]
    meta: Dict = field(default_factory=dict)
    _pos: Dict[str, int] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        self.embeddings = np.asarray(self.embeddings, dtype=np.float32)
        if self.embeddings.shape[0] != len(self.paper_ids):
            raise ValueError("embeddings rows must match paper_ids length")
        self._pos = {pid: i for i, pid in enumerate(self.paper_ids)}

    def __len__(self) -> int:
        return len(self.paper_ids)

    # ------------------------------------------------------------- persistence
    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        np.save(os.path.join(directory, "embeddings.npy"), self.embeddings.astype(np.float16))
        with open(os.path.join(directory, "paper_ids.json"), "w") as f:
            json.dump(self.paper_ids, f)
        with open(os.path.join(directory, "meta.json"), "w") as f:
            json.dump(self.meta, f, indent=1)

    @classmethod
    def load(cls, directory: str) -> "PaperIndex":
        emb = np.load(os.path.join(directory, "embeddings.npy")).astype(np.float32)
        with open(os.path.join(directory, "paper_ids.json")) as f:
            ids = json.load(f)
        meta_p = os.path.join(directory, "meta.json")
        meta = json.load(open(meta_p)) if os.path.exists(meta_p) else {}
        # re-normalise: float16 round-trip perturbs norms slightly
        n = np.linalg.norm(emb, axis=1, keepdims=True)
        n[n == 0] = 1.0
        return cls(emb / n, ids, meta)

    # ------------------------------------------------------------- search
    def position(self, paper_id: str) -> Optional[int]:
        return self._pos.get(paper_id)

    def similarities(self, query: np.ndarray) -> np.ndarray:
        """Cosine similarity of one or more query vectors to all papers. (q, n) or (n,)."""
        q = np.asarray(query, dtype=np.float32)
        single = q.ndim == 1
        if single:
            q = q[None, :]
        n = np.linalg.norm(q, axis=1, keepdims=True)
        n[n == 0] = 1.0
        sims = (q / n) @ self.embeddings.T
        return sims[0] if single else sims

    def top_k(
        self, query: np.ndarray, k: int = 10, exclude: Optional[Sequence[str]] = None
    ) -> List[Tuple[str, float]]:
        """Top-k (paper_id, cosine) for a single query vector."""
        sims = self.similarities(query)
        if exclude:
            for pid in exclude:
                i = self._pos.get(pid)
                if i is not None:
                    sims[i] = -np.inf
        k = min(k, len(sims))
        if k <= 0:
            return []
        idx = np.argpartition(-sims, k - 1)[:k]
        idx = idx[np.argsort(-sims[idx])]
        return [(self.paper_ids[i], float(sims[i])) for i in idx if np.isfinite(sims[i])]

    def without(self, paper_ids: Sequence[str]) -> "PaperIndex":
        """Return a copy with the given papers removed (used by leave-one-out eval)."""
        drop = {self._pos[p] for p in paper_ids if p in self._pos}
        keep = [i for i in range(len(self)) if i not in drop]
        return PaperIndex(self.embeddings[keep], [self.paper_ids[i] for i in keep], dict(self.meta))


def build_index(embeddings: np.ndarray, paper_ids: Sequence[str], meta: Optional[Dict] = None) -> PaperIndex:
    return PaperIndex(np.asarray(embeddings, dtype=np.float32), list(paper_ids), meta or {})
