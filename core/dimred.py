# core/dimred.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import issparse

# Dependencias scikit/umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

try:
    import umap  # pip install umap-learn
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False


def _to_dense_float(X) -> np.ndarray:
    """Convierte DataFrame/sparse/array a float64 2D sin dtype=object."""
    if isinstance(X, pd.DataFrame):
        X = X.to_numpy()
    elif issparse(X):
        X = X.toarray()
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    return X.astype(float, copy=False)


@dataclass
class DimredResult:
    """Resultado de reducción de dimensión con metadatos útiles.

    - .coords: np.ndarray (n, d)
    - .model: objeto del modelo (PCA/TSNE/UMAP)
    - .extra: dict con metadatos (p.ej. explained_variance_ratio_)
    """
    coords: np.ndarray
    model: Any = None
    extra: Optional[dict] = None

    # ---- Interfaz "array-like" para que funcione como np.ndarray ----
    def __array__(self, dtype=None):
        arr = np.asarray(self.coords)
        return arr.astype(dtype) if dtype is not None else arr

    def __iter__(self):
        return iter(self.coords)

    def __getitem__(self, item):
        return self.coords[item]

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.coords.shape

    @property
    def ndim(self) -> int:
        return self.coords.ndim

    def astype(self, dtype):
        return self.coords.astype(dtype)

    def to_numpy(self) -> np.ndarray:
        return np.asarray(self.coords)


def run_pca(X, n_components: int = 2, random_state: int = 42) -> DimredResult:
    """PCA → DimredResult (array-like)."""
    X = _to_dense_float(X)
    n_components = min(n_components, X.shape[1]) if X.shape[1] else 2
    pca = PCA(n_components=n_components, random_state=random_state)
    coords = pca.fit_transform(X)
    return DimredResult(
        coords=np.asarray(coords, dtype=float),
        model=pca,
        extra={"explained_variance_ratio_": getattr(pca, "explained_variance_ratio_", None)}
    )


def run_tsne(X, n_components: int = 2, perplexity: int = 30,
             learning_rate: int = 200, n_iter: int = 1000,
             random_state: int = 42) -> DimredResult:
    """t-SNE → DimredResult (array-like)."""
    X = _to_dense_float(X)
    n_components = max(1, n_components)
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        learning_rate=learning_rate,
        n_iter=n_iter,
        random_state=random_state,
        init="random"
    )
    coords = tsne.fit_transform(X)
    return DimredResult(coords=np.asarray(coords, dtype=float), model=tsne, extra=None)


def run_umap(X, n_components: int = 2, n_neighbors: int = 15,
             min_dist: float = 0.1, random_state: int = 42) -> DimredResult:
    """UMAP → DimredResult (array-like). Requiere umap-learn."""
    if not HAS_UMAP:
        raise ImportError("UMAP no disponible. Instala 'umap-learn'.")

    X = _to_dense_float(X)
    n_components = max(1, n_components)
    um = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state
    )
    coords = um.fit_transform(X)
    return DimredResult(coords=np.asarray(coords, dtype=float), model=um, extra=None)
