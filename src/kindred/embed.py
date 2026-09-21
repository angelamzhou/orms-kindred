"""Paper embedding backends for Kindred.

Backends
--------
- ``specter2``: allenai/specter2_base + proximity adapter ``allenai/specter2``
  (via the ``adapters`` library). CLS pooling.
- ``scincl``: malteos/scincl via plain ``transformers``. CLS pooling.
- ``tfidf``: scikit-learn TF-IDF fallback (no torch needed). Fitted on the
  first ``encode`` call (or via ``fit``), so query papers embedded later use
  the same vocabulary as the index.

All backends take ``title + SEP + abstract`` as input and return L2-normalised
float32 arrays of shape (n, dim). CPU by default.
"""
from __future__ import annotations

import logging
from typing import Iterable, List, Optional, Sequence

import numpy as np

log = logging.getLogger(__name__)

BACKENDS = ("specter2", "scincl", "tfidf")


def _l2norm(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n


def _from_pretrained(cls, name: str, **kw):
    """Load from the local Hugging Face cache without touching the network; download only
    if the model is not cached yet. Keeps `kindred suggest` / `verify-offline` fully offline
    after the first download."""
    try:
        return cls.from_pretrained(name, local_files_only=True, **kw)
    except Exception:  # noqa: BLE001 - not cached (or partial cache): fetch it
        return cls.from_pretrained(name, **kw)


class Embedder:
    """Encode (title, abstract) pairs into dense vectors.

    Parameters
    ----------
    backend : one of ``specter2``, ``scincl``, ``tfidf``, or ``auto`` (tries
        specter2 -> scincl -> tfidf, using whatever imports/loads succeed).
    device : torch device string, default ``cpu``.
    batch_size : encoding batch size.
    max_length : tokenizer truncation length (512 for both transformer models).
    """

    def __init__(
        self,
        backend: str = "auto",
        device: str = "cpu",
        batch_size: int = 16,
        max_length: int = 512,
        tfidf_dim: int = 512,
    ):
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.tfidf_dim = tfidf_dim
        self.backend: Optional[str] = None
        self._model = None
        self._tokenizer = None
        self._vectorizer = None
        self._svd = None

        order = list(BACKENDS) if backend == "auto" else [backend]
        errors = {}
        for b in order:
            try:
                getattr(self, f"_load_{b}")()
                self.backend = b
                log.info("Embedder backend: %s", b)
                break
            except Exception as e:  # noqa: BLE001 - deliberate fallback
                errors[b] = repr(e)
                log.warning("backend %s failed: %s", b, e)
        if self.backend is None:
            raise RuntimeError(f"No embedding backend available: {errors}")

    # ------------------------------------------------------------------ load
    def _load_specter2(self) -> None:
        import torch  # noqa: F401
        from adapters import AutoAdapterModel
        from transformers import AutoTokenizer

        self._tokenizer = _from_pretrained(AutoTokenizer, "allenai/specter2_base")
        try:
            model = _from_pretrained(AutoAdapterModel, "allenai/specter2_base")
            # adapters' load_adapter(source="hf") always contacts the hub, so resolve the cached
            # snapshot ourselves (no network) and hand it a local path; download only if absent.
            try:
                from huggingface_hub import snapshot_download
                adapter_path = snapshot_download("allenai/specter2", local_files_only=True)
            except Exception:  # noqa: BLE001 - not cached yet
                adapter_path = "allenai/specter2"
            model.load_adapter(adapter_path, source="hf" if adapter_path == "allenai/specter2" else None,
                               load_as="proximity", set_active=True)
        except ValueError as e:
            # allenai/specter2_base and allenai/specter2 are only published as pickled
            # .bin files; transformers>=4.52 refuses to torch.load those on torch<2.6
            # (CVE-2025-32434). scincl ships safetensors and is unaffected.
            if "torch.load" in str(e) or "CVE-2025-32434" in str(e):
                raise RuntimeError(
                    "specter2 weights are pickled .bin files that transformers will not load on "
                    "torch<2.6; run `pip install 'torch>=2.6'` or use backend 'scincl'"
                ) from e
            raise
        model.eval().to(self.device)
        self._model = model

    def _load_scincl(self) -> None:
        import torch  # noqa: F401
        from transformers import AutoModel, AutoTokenizer

        self._tokenizer = _from_pretrained(AutoTokenizer, "malteos/scincl")
        model = _from_pretrained(AutoModel, "malteos/scincl")
        model.eval().to(self.device)
        self._model = model

    def _load_tfidf(self) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: F401

        self._vectorizer = None  # fitted lazily
        self._sep = " [SEP] "

    # ------------------------------------------------------------------ text
    @property
    def sep_token(self) -> str:
        if self._tokenizer is not None:
            return self._tokenizer.sep_token
        return self._sep

    def _texts(self, titles: Sequence[str], abstracts: Sequence[Optional[str]]) -> List[str]:
        sep = self.sep_token
        out = []
        for t, a in zip(titles, abstracts):
            # pandas stores missing text as None or NaN (a float); treat both as empty
            t = t.strip() if isinstance(t, str) else ""
            a = a.strip() if isinstance(a, str) else ""
            out.append(t + sep + a if a else t)
        return out

    # ------------------------------------------------------------------ encode
    def fit(self, titles: Sequence[str], abstracts: Sequence[Optional[str]]) -> "Embedder":
        """Fit the TF-IDF vocabulary (no-op for neural backends)."""
        if self.backend != "tfidf":
            return self
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        texts = self._texts(titles, abstracts)
        self._vectorizer = TfidfVectorizer(
            stop_words="english", ngram_range=(1, 2), min_df=1, max_features=200_000,
            sublinear_tf=True,
        )
        X = self._vectorizer.fit_transform(texts)
        k = min(self.tfidf_dim, X.shape[1] - 1, X.shape[0] - 1)
        self._svd = TruncatedSVD(n_components=max(2, k), random_state=0).fit(X) if k >= 2 else None
        return self

    def encode(
        self,
        titles: Sequence[str],
        abstracts: Sequence[Optional[str]],
        show_progress: bool = False,
    ) -> np.ndarray:
        titles = list(titles)
        abstracts = list(abstracts)
        if len(titles) != len(abstracts):
            raise ValueError("titles and abstracts must have equal length")
        if len(titles) == 0:
            return np.zeros((0, self.dim or 0), dtype=np.float32)
        if self.backend == "tfidf":
            return self._encode_tfidf(titles, abstracts)
        return self._encode_torch(titles, abstracts, show_progress)

    def _encode_tfidf(self, titles, abstracts) -> np.ndarray:
        if self._vectorizer is None:
            self.fit(titles, abstracts)
        X = self._vectorizer.transform(self._texts(titles, abstracts))
        if self._svd is not None:
            X = self._svd.transform(X)
        else:
            X = X.toarray()
        return _l2norm(X)

    def _encode_torch(self, titles, abstracts, show_progress) -> np.ndarray:
        import torch

        texts = self._texts(titles, abstracts)
        chunks: List[np.ndarray] = []
        rng = range(0, len(texts), self.batch_size)
        if show_progress:
            try:
                from tqdm import tqdm
                rng = tqdm(rng, desc=f"embed[{self.backend}]")
            except ImportError:
                pass
        with torch.no_grad():
            for i in rng:
                batch = texts[i : i + self.batch_size]
                enc = self._tokenizer(
                    batch, padding=True, truncation=True, max_length=self.max_length,
                    return_tensors="pt", return_token_type_ids=False,
                ).to(self.device)
                out = self._model(**enc)
                cls = out.last_hidden_state[:, 0, :]
                chunks.append(cls.float().cpu().numpy())
        return _l2norm(np.concatenate(chunks, axis=0))

    @property
    def dim(self) -> Optional[int]:
        if self.backend == "tfidf":
            if self._svd is not None:
                return self._svd.n_components
            return None if self._vectorizer is None else len(self._vectorizer.vocabulary_)
        if self._model is not None:
            return int(self._model.config.hidden_size)
        return None

    # ------------------------------------------------------------------ persist (tfidf only)
    def save(self, path: str) -> None:
        """Persist the fitted TF-IDF state so queries can be embedded later."""
        if self.backend == "tfidf":
            import pickle
            with open(path, "wb") as f:
                pickle.dump({"vectorizer": self._vectorizer, "svd": self._svd}, f)

    def load(self, path: str) -> "Embedder":
        if self.backend == "tfidf":
            import pickle
            with open(path, "rb") as f:
                st = pickle.load(f)
            self._vectorizer, self._svd = st["vectorizer"], st["svd"]
        return self
