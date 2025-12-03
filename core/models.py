from __future__ import annotations
from typing import Optional
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.svm import SVC, SVR
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

def make_model(task_type: str, name: str, *,
               C: float = 1.0, gamma: float = 0.1,
               n_estimators: int = 200, max_depth: Optional[int] = None,
               k: int = 5, kmeans_k: int = 3, eps: float = 0.5, min_samples: int = 5,
               class_weight: Optional[str] = None):
    """
    class_weight: usa 'balanced' para modelos que lo soportan (clasificación).
    """
    if task_type == "classification":
        if name == "LogisticRegression":
            return LogisticRegression(max_iter=200, class_weight=class_weight)
        if name == "LinearSVM":
            return SVC(kernel="linear", probability=True, C=C, class_weight=class_weight)
        if name == "SVM(RBF)":
            return SVC(kernel="rbf", probability=True, C=C, gamma=gamma, class_weight=class_weight)
        if name == "RandomForestClassifier":
            return RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, n_jobs=-1, class_weight=class_weight)
        if name == "GradientBoostingClassifier":
            return GradientBoostingClassifier(n_estimators=n_estimators)
        if name == "kNN(Classifier)":
            return KNeighborsClassifier(n_neighbors=k)
        if name == "GaussianNB":
            return GaussianNB()
        if name == "DecisionTreeClassifier":
            return DecisionTreeClassifier(max_depth=max_depth, class_weight=class_weight, random_state=42)

    elif task_type == "regression":
        if name == "LinearRegression":
            return LinearRegression()
        if name == "Ridge":
            return Ridge(alpha=1.0)
        if name == "Lasso":
            return Lasso(alpha=0.01, max_iter=5000)
        if name == "RandomForestRegressor":
            return RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, n_jobs=-1, random_state=42)
        if name == "GradientBoostingRegressor":
            return GradientBoostingRegressor(n_estimators=n_estimators)
        if name == "SVR(RBF)":
            return SVR(C=C, gamma=gamma)
        if name == "kNN(Regressor)":
            return KNeighborsRegressor(n_neighbors=k)
        if name == "DecisionTreeRegressor":
            return DecisionTreeRegressor(max_depth=max_depth, random_state=42)

    elif task_type == "clustering":
        if name == "KMeans":
            return KMeans(n_clusters=kmeans_k, n_init=10, random_state=42)
        if name == "DBSCAN":
            return DBSCAN(eps=eps, min_samples=min_samples)
        if name == "AgglomerativeClustering":
            # Reutilizamos kmeans_k como n_clusters para no complicar el UI
            return AgglomerativeClustering(n_clusters=kmeans_k, linkage="ward")

    raise ValueError(f"Modelo no soportado: task={task_type}, name={name}")
