from __future__ import annotations
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.inspection import permutation_importance

# SHAP es opcional (lo tienes en requirements.txt)
try:
    import shap  # type: ignore
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False


def _dense_float(X):
    """Convierte matriz a np.ndarray float, densificando si es esparsa."""
    try:
        from scipy.sparse import issparse
        if issparse(X):
            X = X.toarray()
    except Exception:
        if hasattr(X, "toarray"):
            X = X.toarray()
    return np.asarray(X, dtype=float)


def get_transformed_feature_names(
    preprocessor: ColumnTransformer,
    X_sample: Optional[pd.DataFrame] = None
) -> List[str]:
    """
    Devuelve los nombres de las columnas DESPUÉS del preprocesamiento (OHE, escalado, passthrough).
    - Si el ColumnTransformer está 'fitted' (tiene transformers_), usa get_feature_names_out().
    - Si NO está 'fitted' y se provee X_sample, clona y fitea con una muestra para derivar los nombres.
    - Si falla, hace fallback a columnas originales de X_sample o nombres genéricos.
    """
    # 1) Si ya está ajustado, intentamos directamente
    if hasattr(preprocessor, "transformers_"):
        try:
            return list(preprocessor.get_feature_names_out())
        except Exception:
            # Construcción manual por ramas, si el método genérico fallara
            names: List[str] = []
            for name, transformer, cols in preprocessor.transformers_:
                if name == 'remainder' and transformer == 'drop':
                    continue
                # Determina el "final" de la rama (pipeline o estimator directo)
                final = transformer.steps[-1][1] if hasattr(transformer, 'steps') else transformer
                # Intentar get_feature_names_out del final
                got = False
                if hasattr(final, 'get_feature_names_out'):
                    try:
                        out = final.get_feature_names_out(cols)
                        names.extend(list(out))
                        got = True
                    except TypeError:
                        try:
                            out = final.get_feature_names_out()
                            names.extend(list(out))
                            got = True
                        except Exception:
                            pass
                if not got:
                    # Passthrough (o sin método de nombres): devolver cols
                    if cols == 'remainder':
                        # Resto real ya resuelto por ColumnTransformer fitted
                        # No conocemos exactamente cuáles fueron, así que omitimos aquí
                        # (el caso normal lo cubre get_feature_names_out arriba)
                        continue
                    names.extend(list(cols if isinstance(cols, (list, tuple, np.ndarray)) else [cols]))
            if names:
                return names
            # si no logramos construir, cae a fallbacks más abajo

    # 2) Si no está ajustado y tenemos muestra, clonar y ajustar con pocas filas
    if X_sample is not None:
        try:
            pre_clone = clone(preprocessor)
            # Conservar tamaño pequeño para velocidad
            if hasattr(X_sample, "iloc"):
                X_fit = X_sample.iloc[:200].copy()
            else:
                X_fit = X_sample[:200]
            pre_clone.fit(X_fit)
            try:
                return list(pre_clone.get_feature_names_out())
            except Exception:
                # Intento manual post-ajuste del clone
                names: List[str] = []
                for name, transformer, cols in pre_clone.transformers_:
                    if name == 'remainder' and transformer == 'drop':
                        continue
                    final = transformer.steps[-1][1] if hasattr(transformer, 'steps') else transformer
                    got = False
                    if hasattr(final, 'get_feature_names_out'):
                        try:
                            out = final.get_feature_names_out(cols)
                            names.extend(list(out))
                            got = True
                        except TypeError:
                            try:
                                out = final.get_feature_names_out()
                                names.extend(list(out))
                                got = True
                            except Exception:
                                pass
                    if not got:
                        if cols == 'remainder':
                            # En este camino no tenemos tracking exacto del "resto"
                            continue
                        names.extend(list(cols if isinstance(cols, (list, tuple, np.ndarray)) else [cols]))
                if names:
                    return names
        except Exception:
            pass

    # 3) Fallbacks
    # a) Si tenemos DataFrame, usar sus columnas originales
    if X_sample is not None and hasattr(X_sample, "columns"):
        return list(map(str, X_sample.columns))
    # b) Si sabemos el ancho, generar genéricos
    if X_sample is not None and hasattr(X_sample, "shape"):
        n = int(X_sample.shape[1])
        return [f"x{i}" for i in range(n)]
    # c) Vacío
    return []


def model_feature_importance(pipe_or_tuple, X: pd.DataFrame) -> Optional[List[Tuple[str, float]]]:
    """
    Extrae pares (feature_name, importance) si el modelo lo soporta.
    Soporta:
      - Árboles (RandomForest*, GradientBoosting*) via feature_importances_
      - Lineales: LogisticRegression, LinearSVM via |coef_|
    Retorna None si el modelo no soporta importancia (kNN, SVM RBF, SVR, etc.)
    """
    if isinstance(pipe_or_tuple, tuple):
        pre, model = pipe_or_tuple
    else:
        pre = pipe_or_tuple.named_steps['pre']
        model = pipe_or_tuple.named_steps['model']

    feat_names = get_transformed_feature_names(pre, X_sample=X)

    # Árboles
    if hasattr(model, 'feature_importances_'):
        imps = np.asarray(model.feature_importances_, dtype=float)
        L = min(len(feat_names), len(imps))
        pairs = list(zip(feat_names[:L], imps[:L]))
        pairs.sort(key=lambda t: t[1], reverse=True)
        return pairs

    # Lineales
    if hasattr(model, 'coef_'):
        coef = np.asarray(model.coef_, dtype=float)
        if coef.ndim == 2:
            imps = np.linalg.norm(coef, axis=0)  # multiclase
        else:
            imps = np.abs(coef)
        L = min(len(feat_names), len(imps))
        pairs = list(zip(feat_names[:L], imps[:L]))
        pairs.sort(key=lambda t: t[1], reverse=True)
        return pairs

    return None


def permutation_feature_importance(
    model_or_pipe,
    X: pd.DataFrame,
    y: pd.Series,
    scoring: Optional[str] = None,
    n_repeats: int = 10,
    random_state: int = 42
) -> List[Tuple[str, float, float]]:
    """
    Importancias por permutación → [(feature, mean_importance, std_importance)]
    Funciona para cualquier modelo sklearn ajustado.
    """
    if isinstance(model_or_pipe, tuple):
        pre, model = model_or_pipe
    else:
        pre = model_or_pipe.named_steps['pre']
        model = model_or_pipe.named_steps['model']

    feat_names = get_transformed_feature_names(pre, X_sample=X)
    X_trans = pre.transform(X)
    X_trans = _dense_float(X_trans)

    result = permutation_importance(
        model, X_trans, y,
        scoring=scoring, n_repeats=n_repeats,
        random_state=random_state, n_jobs=-1
    )
    means = result.importances_mean
    stds = result.importances_std
    L = min(len(feat_names), len(means))
    return [(feat_names[i], float(means[i]), float(stds[i])) for i in range(L)]


def shap_global_importance(
    model_or_pipe,
    X: pd.DataFrame,
    max_samples: int = 500,
    random_state: int = 42
) -> Optional[List[Tuple[str, float]]]:
    """
    Calcula importancias globales con SHAP (contribución media absoluta).
    Devuelve lista [(feature, importance)].
    """
    if not HAS_SHAP:
        return None

    if isinstance(model_or_pipe, tuple):
        pre, model = model_or_pipe
    else:
        pre = model_or_pipe.named_steps['pre']
        model = model_or_pipe.named_steps['model']

    feat_names = get_transformed_feature_names(pre, X_sample=X)

    # Muestreo para velocidad
    X_sample = X.sample(max_samples, random_state=random_state) if len(X) > max_samples else X
    X_trans = pre.transform(X_sample)
    X_trans = _dense_float(X_trans)

    try:
        if hasattr(model, 'estimators_') or hasattr(model, 'feature_importances_'):
            # Modelos de árboles → TreeExplainer directo
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(X_trans)
            if isinstance(shap_vals, list):
                vals = np.mean([np.abs(v) for v in shap_vals], axis=0)
            else:
                vals = np.abs(shap_vals)
        else:
            # Generico: usar Kernel/Model-agnostic vía shap.Explainer(model.predict, ...)
            # Nota: el estimator final espera ya X_trans (post-preprocessing)
            explainer = shap.Explainer(lambda Z: model.predict(Z), X_trans)
            sv = explainer(X_trans)
            vals = np.abs(getattr(sv, "values", np.asarray(sv)))  # compatibilidad
        contrib = np.mean(vals, axis=0)
        L = min(len(feat_names), len(contrib))
        pairs = list(zip(feat_names[:L], contrib[:L]))
        pairs.sort(key=lambda t: t[1], reverse=True)
        return pairs
    except Exception:
        return None

# ===== Backward-compat aliases (para UI antigua) =====

def model_importances(model_or_pipe, X):
    # Antes la UI llamaba model_importances(...); mapea al nombre nuevo
    return model_feature_importance(model_or_pipe, X)

def permutation_importances(model_or_pipe, X, y, scoring=None, n_repeats=10, random_state=42):
    return permutation_feature_importance(
        model_or_pipe, X, y,
        scoring=scoring, n_repeats=n_repeats, random_state=random_state
    )

def compute_shap_global(model_or_pipe, X, max_samples=500, random_state=42):
    return shap_global_importance(model_or_pipe, X, max_samples=max_samples, random_state=random_state)

# Algunos helpers que la UI vieja podía importar
def to_dataframe(X, feature_names=None):
    import pandas as pd
    import numpy as np
    if isinstance(X, pd.DataFrame):
        return X
    if isinstance(X, np.ndarray):
        cols = feature_names if feature_names else [f"x{i}" for i in range(X.shape[1])]
        return pd.DataFrame(X, columns=cols)
    # Fallback
    try:
        return pd.DataFrame(X)
    except Exception:
        return pd.DataFrame()

def plot_importances_bar(df, ax=None, title="Feature importance"):
    import matplotlib.pyplot as plt
    if df is None or len(df) == 0:
        return None
    if ax is None:
        ax = plt.gca()
    # Acepta dataframes con columnas ["feature","importance"] o ["feature","mean","std"]
    cols = df.columns.tolist()
    if "importance" not in cols and "mean" in cols:
        df = df.rename(columns={"mean": "importance"})
    df = df.sort_values("importance", ascending=False)
    ax.barh(df["feature"], df["importance"])
    ax.set_title(title)
    ax.invert_yaxis()
    return ax

def plot_shap_summary(*args, **kwargs):
    # Si tu UI lo llama, no truena. Implementa luego si lo necesitas.
    return None
