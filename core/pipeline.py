from __future__ import annotations
from typing import List
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(X: pd.DataFrame, scale_numeric: bool = True) -> ColumnTransformer:
    num_cols: List[str] = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols: List[str] = [c for c in X.columns if c not in num_cols]

    if scale_numeric:
        num_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
    else:
        num_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
        ])

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("oh", OneHotEncoder(handle_unknown="ignore")),
    ])

    pre = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ])
    return pre