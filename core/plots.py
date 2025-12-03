# core/plots.py
import numpy as np
import pandas as pd
from matplotlib.axes import Axes

# SciPy puede no estar siempre; damos un fallback ligero
try:
    from scipy.sparse import issparse  # type: ignore
except Exception:  # pragma: no cover
    def issparse(x):
        return hasattr(x, "tocsr") or hasattr(x, "toarray")

def _to_dense_float(X):
    """Convierte DataFrame / csr_matrix a ndarray float para graficar."""
    if isinstance(X, pd.DataFrame):
        X = X.to_numpy()
    elif issparse(X):
        X = X.toarray()
    X = np.asarray(X)
    return X.astype(float, copy=False)

def plot_confusion(ax: Axes, cm: np.ndarray, classes):
    """Matriz de confusión con anotaciones."""
    import itertools
    cm = np.asarray(cm)
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        ylabel="True label",
        xlabel="Predicted label",
    )
    ax.set_title("Matriz de Confusión")
    thresh = cm.max() / 2.0 if cm.size else 0.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(
            j, i, format(cm[i, j], "d"),
            ha="center", va="center",
            color="white" if cm[i, j] > thresh else "black",
            fontsize=9,
        )

def plot_regression_scatter(ax: Axes, y_true, y_pred):
    """Dispersión y_true vs y_pred con línea y=x."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    ax.scatter(y_true, y_pred, s=12, alpha=0.8)
    if y_true.size:
        mn = float(np.min([y_true.min(), y_pred.min()]))
        mx = float(np.max([y_true.max(), y_pred.max()]))
        ax.plot([mn, mx], [mn, mx], "k--", lw=1)
    ax.set_xlabel("y verdadera")
    ax.set_ylabel("y predicha")
    ax.set_title("Predicción vs Real")

def plot_clusters_pca(ax: Axes, X2d, labels):
    """Scatter 2D por etiquetas ya proyectado (o pseudo-PCA)."""
    X2d = _to_dense_float(X2d)  # <- evita crash con csr_matrix
    labels = np.asarray(labels).ravel()
    cats = pd.Categorical(labels)
    for lab in cats.categories:
        mask = (labels == lab)
        ax.scatter(X2d[mask, 0], X2d[mask, 1], s=12, alpha=0.85, label=str(lab))
    ax.set_xlabel("dim1")
    ax.set_ylabel("dim2")
    ax.legend(loc="best", fontsize=8)
