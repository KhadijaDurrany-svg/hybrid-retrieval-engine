"""Dense text embedders.

The default embedder is a fully offline Latent Semantic model: TF-IDF over the
corpus vocabulary followed by a truncated SVD projection. It is deterministic,
needs no network and no model download, and gives the dense index a genuine
term-correlation signal that complements exact-match BM25.

An optional transformer backend (`OnnxEmbedder`) is provided behind the same
interface for users who want neural embeddings; it is only imported when a model
file is actually supplied, so the package stays dependency-light by default.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from .analysis import Analyzer


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return mat / norms


class Embedder(Protocol):
    dim: int

    def encode(self, texts: list[str]) -> np.ndarray:
        """Return an (n, dim) float32 array of L2-normalized row vectors."""
        ...


class HashingLSAEmbedder:
    """Deterministic offline embedder: TF-IDF -> truncated SVD (LSA)."""

    def __init__(self, dim: int = 128, analyzer: Analyzer | None = None, seed: int = 13) -> None:
        self.requested_dim = dim
        self.seed = seed
        self._analyzer = analyzer or Analyzer()
        self._vectorizer = TfidfVectorizer(
            analyzer=self._analyzer.analyze,
            sublinear_tf=True,
            norm="l2",
        )
        self._svd: TruncatedSVD | None = None
        self.dim = dim
        self._fitted = False

    def fit(self, texts: list[str]) -> HashingLSAEmbedder:
        if not texts:
            raise ValueError("cannot fit an embedder on an empty corpus")
        tfidf = self._vectorizer.fit_transform(texts)
        n_samples, n_features = tfidf.shape
        # SVD components must stay below both matrix dimensions.
        dim = max(1, min(self.requested_dim, n_samples - 1, n_features - 1))
        self._svd = TruncatedSVD(n_components=dim, random_state=self.seed)
        self._svd.fit(tfidf)
        self.dim = dim
        self._fitted = True
        return self

    def encode(self, texts: list[str]) -> np.ndarray:
        if not self._fitted or self._svd is None:
            raise RuntimeError("embedder must be fit() before encode()")
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        tfidf = self._vectorizer.transform(texts)
        dense = self._svd.transform(tfidf).astype(np.float32)
        return _l2_normalize(dense)


class OnnxEmbedder:
    """Optional sentence-transformer backend loaded from a local ONNX model.

    Kept behind a lazy import so the core package has no heavy dependency. Only
    construct this when `model_dir` points at a real exported model; otherwise
    use `HashingLSAEmbedder`.
    """

    def __init__(self, model_dir: str, dim: int = 384) -> None:  # pragma: no cover
        try:
            import onnxruntime as ort  # noqa: F401
            from tokenizers import Tokenizer
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "OnnxEmbedder needs onnxruntime and tokenizers installed, and a "
                "locally exported model. Install the extras or use HashingLSAEmbedder."
            ) from exc
        self.model_dir = model_dir
        self.dim = dim
        self._tok = Tokenizer.from_file(f"{model_dir}/tokenizer.json")
        self._sess = ort.InferenceSession(f"{model_dir}/model.onnx")

    def encode(self, texts: list[str]) -> np.ndarray:  # pragma: no cover
        import numpy as _np

        encs = [self._tok.encode(t) for t in texts]
        max_len = max((len(e.ids) for e in encs), default=1)
        ids = _np.zeros((len(encs), max_len), dtype=_np.int64)
        mask = _np.zeros((len(encs), max_len), dtype=_np.int64)
        for i, e in enumerate(encs):
            ids[i, : len(e.ids)] = e.ids
            mask[i, : len(e.ids)] = 1
        out = self._sess.run(None, {"input_ids": ids, "attention_mask": mask})[0]
        masked = out * mask[:, :, None]
        pooled = masked.sum(axis=1) / _np.clip(mask.sum(axis=1)[:, None], 1, None)
        return _l2_normalize(pooled.astype(_np.float32))
