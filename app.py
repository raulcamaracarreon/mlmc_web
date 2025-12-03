import os
from io import BytesIO
from datetime import datetime

import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, session, send_file, flash
import pandas.api.types as ptypes
import joblib

import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from scipy.sparse import issparse

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# Núcleo reutilizado
from core.pipeline import build_preprocessor
from core.models import make_model
from core.evaluate import eval_classification, eval_regression, eval_clustering
from core.plots import plot_confusion, plot_regression_scatter, plot_clusters_pca
from core.dimred import run_pca, run_tsne, run_umap
from core.report import fig_to_base64_png, render_html_report
from core import interpret as core_interp

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "change-me")
os.makedirs("uploads", exist_ok=True)
os.makedirs("outputs", exist_ok=True)
os.makedirs("models_store", exist_ok=True)

def fig_to_b64(fig: Figure) -> str:
    return fig_to_base64_png(fig)

def suggest_task_from_target(y: pd.Series) -> str:
    if not ptypes.is_numeric_dtype(y):
        return "classification"
    unique = y.nunique(dropna=True)
    n = len(y)
    if unique <= min(20, max(2, int(0.05 * n))):
        return "classification"
    return "regression"

def to_dense_float(X) -> np.ndarray:
    if issparse(X):
        X = X.toarray()
    elif isinstance(X, pd.DataFrame):
        X = X.to_numpy()
    else:
        X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    return X.astype(float, copy=False)

def to_2d_coords(X: np.ndarray) -> np.ndarray:
    n, m = X.shape
    if m >= 2:
        try:
            return PCA(n_components=2, random_state=42).fit_transform(X)
        except Exception:
            return X[:, :2]
    elif m == 1:
        return np.c_[X[:, 0], np.zeros(n)]
    else:
        return np.zeros((n, 2))

def get_feature_names_out(preproc, original_cols):
    try:
        return preproc.get_feature_names_out(original_cols)
    except Exception:
        try:
            return preproc.get_feature_names_out()
        except Exception:
            return np.array(original_cols)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        session.clear()
        has_labels = (request.form.get("labels") == "yes")
        session["has_labels"] = has_labels
        if has_labels:
            session["task_type"] = request.form.get("task_type_superv", "classification")
        else:
            session["task_type"] = request.form.get("task_type_unsuperv", "clustering")
        return redirect(url_for("data"))
    return render_template("index.html")

@app.route("/data", methods=["GET", "POST"])
def data():
    cols = []
    if request.method == "POST":
        if "csv" in request.files and request.files["csv"].filename:
            f = request.files["csv"]
            path = os.path.join("uploads", f.filename)
            f.save(path)
            session["csv_path"] = path

        if session.get("csv_path"):
            cols = pd.read_csv(session["csv_path"], nrows=1).columns.tolist()
            if request.form.get("submitted") == "1":
                feats = request.form.getlist("features")
                if feats:
                    session["features"] = feats
                if session.get("has_labels"):
                    session["target"] = request.form.get("target")
                    if session["target"]:
                        y_col = session["target"]
                        y_series = pd.read_csv(session["csv_path"], usecols=[y_col])[y_col]
                        auto_task = suggest_task_from_target(y_series)
                        original = session.get("task_type")
                        if original in {"classification", "regression"} and auto_task != original:
                            session["task_type"] = auto_task
                            human = "categórico" if auto_task == "classification" else "numérico continuo"
                            flash(f"Detectamos que el objetivo es {human}. Cambiamos la tarea a {auto_task.capitalize()} para evitar errores.")
                return redirect(url_for("algo" if session.get("task_type") != "dimred" else "dimred"))
    else:
        if session.get("csv_path"):
            cols = pd.read_csv(session["csv_path"], nrows=1).columns.tolist()

    return render_template("data.html", cols=cols, has_labels=session.get("has_labels", False))

@app.route("/algo", methods=["GET", "POST"])
def algo():
    task = session.get("task_type", "classification")

    if request.method == "POST":
        cfg = dict(
            test_size=float(request.form.get("test_size", 0.2)),
            scale=bool(request.form.get("scale")),
        )

        if task == "classification":
            allowed = {
                "LogisticRegression", "LinearSVM", "SVM(RBF)",
                "RandomForestClassifier", "GradientBoostingClassifier", "kNN(Classifier)"
            }
            algo_name = request.form.get("algo", "RandomForestClassifier")
            cfg.update(dict(
                algo=algo_name if algo_name in allowed else "RandomForestClassifier",
                n_estimators=int(request.form.get("n_estimators", 200)),
                max_depth=int(request.form.get("max_depth", 0)) or None,
                k=int(request.form.get("k", 5)),
                C=float(request.form.get("C", 1.0)),
                gamma=float(request.form.get("gamma", 0.1)),
                imbalance=request.form.get("imbalance", "None")
                          if request.form.get("imbalance", "None") in {"None", "class_weight", "SMOTE"}
                          else "None",
                smote_k=int(request.form.get("smote_k", 5)),
            ))

        elif task == "regression":
            allowed = {
                "LinearRegression", "Ridge", "Lasso",
                "RandomForestRegressor", "GradientBoostingRegressor",
                "SVR(RBF)", "kNN(Regressor)"
            }
            algo_name = request.form.get("algo", "LinearRegression")
            cfg.update(dict(
                algo=algo_name if algo_name in allowed else "LinearRegression",
                n_estimators=int(request.form.get("n_estimators", 200)),
                max_depth=int(request.form.get("max_depth", 0)) or None,
                k=int(request.form.get("k", 5)),
                C=float(request.form.get("C", 1.0)),
                gamma=float(request.form.get("gamma", 0.1)),
                imbalance="None",
                smote_k=5,
            ))

        elif task == "clustering":
            allowed = {"KMeans", "DBSCAN"}
            algo_name = request.form.get("algo", "KMeans")
            cfg.update(dict(
                algo=algo_name if algo_name in allowed else "KMeans",
                kmeans_k=int(request.form.get("kmeans_k", 3)),
                eps=float(request.form.get("eps", 0.5)),
                min_samples=int(request.form.get("min_samples", 5)),
                imbalance="None",
                smote_k=5,
            ))
        else:
            return redirect(url_for("dimred"))

        session["cfg"] = cfg
        return redirect(url_for("run"))

    return render_template("algo.html", task_type=task)

@app.route("/run")
def run():
    # Evita abrir /run sin haber configurado algo
    if "cfg" not in session:
        flash("Configura el algoritmo primero.")
        return redirect(url_for("algo"))

    df = pd.read_csv(session["csv_path"])
    X = df[session["features"]].copy()
    pre = build_preprocessor(X, scale_numeric=session["cfg"].get("scale", False))
    task = session.get("task_type", "classification")

    fig = Figure(figsize=(6, 3.2)); ax = fig.add_subplot(111)
    metrics = {}; extra_text = None; task_label = "ML"

    model_path = None
    feat_names_out = None

    if session.get("has_labels"):
        y = df[session["target"]]

        if task == "regression" and not ptypes.is_numeric_dtype(y):
            session["task_type"] = "classification"
            flash("El objetivo no es numérico. Cambiamos automáticamente a Clasificación.")
            return redirect(url_for("algo"))

        if task == "classification" and ptypes.is_numeric_dtype(y):
            unique = y.nunique(dropna=True); n = len(y)
            if unique > min(20, max(2, int(0.05 * n))):
                flash("Aviso: el objetivo parece continuo con muchas modalidades; quizá 'Regresión' sea más apropiado.")

        strat = y if task == "classification" else None
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=session["cfg"].get("test_size", 0.2),
            random_state=42, stratify=strat
        )

        imbalance = session["cfg"].get("imbalance", "None")
        cw = "balanced" if (task == "classification" and imbalance == "class_weight") else None

        base = make_model(
            task, session["cfg"]["algo"],
            C=session["cfg"].get("C", 1.0), gamma=session["cfg"].get("gamma", 0.1),
            n_estimators=session["cfg"].get("n_estimators", 200),
            max_depth=session["cfg"].get("max_depth"),
            k=session["cfg"].get("k", 5), class_weight=cw
        )

        if task == "classification" and imbalance == "SMOTE":
            pipe = ImbPipeline([
                ("pre", pre),
                ("smote", SMOTE(k_neighbors=session["cfg"].get("smote_k", 5), random_state=42)),
                ("model", base)
            ])
        else:
            pipe = Pipeline([("pre", pre), ("model", base)])

        pipe.fit(Xtr, ytr)
        preds = pipe.predict(Xte)

        try:
            feat_names_out = get_feature_names_out(pipe.named_steps["pre"], session["features"])
        except Exception:
            feat_names_out = np.array(session["features"])

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join("models_store", f"model_{task}_{ts}.joblib")
        joblib.dump(pipe, model_path)
        session["last_model_path"] = model_path
        session["last_model_task"] = task
        session["last_feature_names_out"] = feat_names_out.tolist()

        if task == "classification":
            from sklearn.metrics import confusion_matrix
            m = eval_classification(yte, preds); task_label = "Clasificación"
            cm = confusion_matrix(yte, preds); plot_confusion(ax, cm, np.unique(yte))
            metrics = {"accuracy": m["accuracy"], "precision_w": m["precision_w"], "recall_w": m["recall_w"], "f1_w": m["f1_w"]}
            extra_text = m["report"]
        else:
            m = eval_regression(yte, preds); task_label = "Regresión"
            plot_regression_scatter(ax, yte, preds)
            metrics = {"MAE": m["mae"], "MSE": m["mse"], "RMSE": m["rmse"], "R2": m["r2"]}

    else:
        # Clustering
        model = make_model("clustering", session["cfg"]["algo"],
                           kmeans_k=session["cfg"].get("kmeans_k", 3),
                           eps=session["cfg"].get("eps", 0.5),
                           min_samples=session["cfg"].get("min_samples", 5))
        X_t = pre.fit_transform(X); X_dense = to_dense_float(X_t)
        model.fit(X_dense); labels = getattr(model, "labels_", None)
        if labels is not None:
            metrics = eval_clustering(X_dense, labels)
            coords = to_2d_coords(X_dense); plot_clusters_pca(ax, coords, labels)
        task_label = "Clustering"
        # Guardar pipeline simple
        pipe = Pipeline([("pre", pre), ("model", model)])
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join("models_store", f"model_{task_label}_{ts}.joblib")
        joblib.dump(pipe, model_path)
        session["last_model_path"] = model_path
        session["last_model_task"] = "clustering"
        session["last_feature_names_out"] = get_feature_names_out(pre, session["features"]).tolist()

    b64 = fig_to_b64(fig)

    html_report = render_html_report(
        title="Reporte de Experimento ML", task=task_label,
        dataset_name=os.path.basename(session["csv_path"]),
        features=session.get("features"), target=session.get("target"),
        metrics=metrics, extra_text=extra_text, chart_data_url=b64
    )
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join("outputs", f"reporte_ml_{ts}.html")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html_report)
    session["last_report_path"] = report_path

    return render_template("run.html", metrics=metrics, chart_b64=b64)

@app.route("/download_report")
def download_report():
    path = session.get("last_report_path")
    if path and os.path.exists(path):
        return send_file(path, mimetype="text/html", as_attachment=True,
                         download_name=os.path.basename(path))
    bio = BytesIO("<h1>No hay reporte</h1>".encode("utf-8"))
    return send_file(bio, mimetype="text/html", as_attachment=True, download_name="reporte_ml.html")

@app.route("/dimred", methods=["GET", "POST"])
def dimred():
    if not session.get("csv_path") or not session.get("features"):
        return render_template("dimred.html", cols=None)

    df = pd.read_csv(session["csv_path"])
    cols = df.columns.tolist()

    chart_b64 = None
    form_algo = "PCA"; form_scale = False
    form_perplexity = 30; form_lr = 200; form_iter = 1000
    form_neighbors = 15; form_min_dist = 0.1
    form_color_by = ""

    if request.method == "POST":
        form_algo = request.form.get("algo", "PCA")
        form_scale = bool(request.form.get("scale"))
        form_perplexity = int(request.form.get("perplexity", 30))
        form_lr = int(request.form.get("learning_rate", 200))
        form_iter = int(request.form.get("n_iter", 1000))
        form_neighbors = int(request.form.get("n_neighbors", 15))
        try:
            form_min_dist = float(request.form.get("min_dist", 0.1))
        except Exception:
            form_min_dist = 0.1
        form_color_by = request.form.get("color_by", "")

        X = df[session["features"]].copy()
        pre = build_preprocessor(X, scale_numeric=form_scale)
        X_t = pre.fit_transform(X); X_dense = to_dense_float(X_t)

        try:
            if form_algo == "PCA":
                coords = np.asarray(run_pca(X_dense, n_components=2))
            elif form_algo == "TSNE":
                # ---- Fallback robusto (con y sin n_iter) ----
                try:
                    result = run_tsne(
                        X_dense, n_components=2,
                        perplexity=form_perplexity,
                        learning_rate=form_lr,
                        n_iter=form_iter,
                        random_state=42
                    )
                    coords = np.asarray(result)
                except TypeError:
                    try:
                        result = run_tsne(
                            X_dense, n_components=2,
                            perplexity=form_perplexity,
                            learning_rate=form_lr,
                            random_state=42
                        )
                        coords = np.asarray(result)
                    except Exception:
                        from sklearn.manifold import TSNE
                        try:
                            coords = TSNE(
                                n_components=2,
                                perplexity=form_perplexity,
                                learning_rate=form_lr,
                                n_iter=form_iter,
                                random_state=42,
                                init="random"
                            ).fit_transform(X_dense)
                        except TypeError:
                            coords = TSNE(
                                n_components=2,
                                perplexity=form_perplexity,
                                learning_rate=form_lr,
                                random_state=42,
                                init="random"
                            ).fit_transform(X_dense)
            elif form_algo == "UMAP":
                coords = np.asarray(run_umap(X_dense, n_components=2,
                                             n_neighbors=form_neighbors,
                                             min_dist=form_min_dist, random_state=42))
            else:
                coords = to_2d_coords(X_dense)
        except Exception as e:
            flash(f"Error en {form_algo}: {e}")
            coords = to_2d_coords(X_dense)

        fig = Figure(figsize=(6, 3.2)); ax = fig.add_subplot(111)
        ax.set_xlabel("dim1"); ax.set_ylabel("dim2"); ax.set_title(form_algo)
        if form_color_by and form_color_by in df.columns:
            series = df[form_color_by]; cats = pd.Categorical(series)
            for lab in cats.categories:
                mask = (series == lab).to_numpy()
                ax.scatter(coords[mask, 0], coords[mask, 1], s=12, alpha=0.85, label=str(lab))
            ax.legend(title=form_color_by, loc="best", fontsize=8)
        else:
            ax.scatter(coords[:, 0], coords[:, 1], s=12, alpha=0.85)
        chart_b64 = fig_to_b64(fig)

        out = pd.DataFrame({"dim1": coords[:, 0], "dim2": coords[:, 1]})
        if form_color_by and form_color_by in df.columns:
            out[form_color_by] = df[form_color_by].values
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join("outputs", f"dimred_{form_algo}_{ts}.csv")
        out.to_csv(csv_path, index=False, encoding="utf-8")
        session["last_dimred_csv"] = csv_path

    return render_template("dimred.html", cols=cols, chart_b64=chart_b64,
                           form_algo=form_algo, form_scale=form_scale,
                           form_perplexity=form_perplexity, form_lr=form_lr, form_iter=form_iter,
                           form_neighbors=form_neighbors, form_min_dist=form_min_dist,
                           form_color_by=form_color_by)

@app.route("/download_dimred")
def download_dimred():
    path = session.get("last_dimred_csv")
    if path and os.path.exists(path):
        return send_file(path, mimetype="text/csv", as_attachment=True,
                         download_name=os.path.basename(path))
    bio = BytesIO("dim1,dim2\n".encode("utf-8"))
    return send_file(bio, mimetype="text/csv", as_attachment=True, download_name="dimred.csv")

@app.route("/interpret", methods=["GET", "POST"])
def interpret():
    from matplotlib.figure import Figure
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.inspection import permutation_importance

    model_path = session.get("last_model_path")
    task = session.get("last_model_task")
    if not model_path or not os.path.exists(model_path):
        return render_template("interpret.html", last_model_ready=False)

    obj = joblib.load(model_path)

    # Aceptar Pipeline o estimador suelto
    pipe = obj if hasattr(obj, "named_steps") else None
    if pipe is not None:
        pre = pipe.named_steps.get("pre", None)
        est = pipe.named_steps.get("model", pipe.steps[-1][1])
    else:
        pre = None
        est = obj  # estimador sin pipeline

    if task == "clustering":
        flash("Interpretabilidad basada en importancias no está disponible para Clustering.")
        return render_template("interpret.html", last_model_ready=True,
                               form_method="model", form_topk=20,
                               form_nrepeats=10, form_scoring="",
                               chart_b64=None, table=None, csv_ready=False)

    features = session.get("features", [])
    df = pd.read_csv(session["csv_path"])

    feat_names_out = session.get("last_feature_names_out")
    if not feat_names_out:
        try:
            feat_names_out = get_feature_names_out(pre, features).tolist()
        except Exception:
            feat_names_out = features
    feat_names_out = list(feat_names_out)

    form_method = "model"; form_topk = 20; form_nrepeats = 10; form_scoring = ""
    chart_b64 = None; table = []; csv_ready = False

    if request.method == "POST":
        form_method = request.form.get("method", "model")
        form_topk = int(request.form.get("topk", 20))
        form_nrepeats = int(request.form.get("n_repeats", 10))
        form_scoring = (request.form.get("scoring", "") or None)

        def plot_bars(df_imp, title="Importancias"):
            fig = Figure(figsize=(6, 3.2)); ax = fig.add_subplot(111)
            d = df_imp.sort_values("importance", ascending=False).head(form_topk)
            ax.barh(list(d["feature"])[::-1], list(d["importance"])[::-1])
            ax.set_title(title); ax.set_xlabel("Importance"); ax.set_ylabel("")
            from core.report import fig_to_base64_png
            return fig_to_base64_png(fig)

        try:
            if form_method == "model":
                # Sin depender del core: feature_importances_ / coef_
                df_imp = None
                if hasattr(est, "feature_importances_"):
                    vals = np.asarray(est.feature_importances_).ravel()
                    df_imp = pd.DataFrame({"feature": feat_names_out[:len(vals)],
                                           "importance": vals[:len(feat_names_out)]})
                elif hasattr(est, "coef_"):
                    co = np.asarray(est.coef_)
                    vals = np.mean(np.abs(co), axis=0) if co.ndim > 1 else np.abs(co)
                    vals = np.asarray(vals).ravel()
                    df_imp = pd.DataFrame({"feature": feat_names_out[:len(vals)],
                                           "importance": vals[:len(feat_names_out)]})
                else:
                    # Último recurso: intenta core.model_feature_importance si existe
                    try:
                        df_imp = core_interp.model_feature_importance((pre, est), X=df[features])
                        if df_imp is not None:
                            df_imp = pd.DataFrame(df_imp, columns=["feature", "importance"])
                    except Exception:
                        df_imp = None
                if df_imp is None:
                    raise ValueError("El estimador no expone importancias (feature_importances_ / coef_).")

                chart_b64 = plot_bars(df_imp, "Model feature importance")

            elif form_method == "permutation":
                if pipe is None:
                    flash("Permutation importance necesita el Pipeline completo. "
                          "Vuelve a entrenar en 'Algoritmo → Ejecutar' para regenerarlo.")
                    return render_template("interpret.html", last_model_ready=True,
                                           form_method=form_method, form_topk=form_topk,
                                           form_nrepeats=form_nrepeats, form_scoring=form_scoring,
                                           chart_b64=None, table=None, csv_ready=False)
                if not session.get("has_labels"):
                    flash("Permutation importance requiere objetivo (supervisado).")
                    return render_template("interpret.html", last_model_ready=True,
                                           form_method=form_method, form_topk=form_topk,
                                           form_nrepeats=form_nrepeats, form_scoring=form_scoring,
                                           chart_b64=None, table=None, csv_ready=False)
                y = df[session["target"]]; X = df[features].copy()
                Xtr, Xte, ytr, yte = train_test_split(
                    X, y, test_size=0.2, random_state=42,
                    stratify=y if task == "classification" else None
                )
                scoring = form_scoring or ("accuracy" if task == "classification" else "r2")
                perm = permutation_importance(pipe, Xte, yte, scoring=scoring,
                                              n_repeats=form_nrepeats, random_state=42)
                df_imp = pd.DataFrame({
                    "feature": feat_names_out[:len(perm.importances_mean)],
                    "importance": perm.importances_mean[:len(feat_names_out)],
                    "std": perm.importances_std[:len(feat_names_out)]
                })
                chart_b64 = plot_bars(df_imp, "Permutation importance")

            else:  # SHAP (global)
                if pipe is None:
                    flash("SHAP requiere el Pipeline completo (preprocesamiento + modelo). "
                          "Vuelve a entrenar para guardarlo y luego calcula SHAP.")
                    return render_template("interpret.html", last_model_ready=True,
                                           form_method=form_method, form_topk=form_topk,
                                           form_nrepeats=form_nrepeats, form_scoring=form_scoring,
                                           chart_b64=None, table=None, csv_ready=False)
                # Usa la función del core con el Pipeline (como espera)
                pairs = core_interp.shap_global_importance(pipe, df[features], max_samples=2000)
                if not pairs:
                    raise ValueError("SHAP no disponible o falló el cálculo.")
                df_imp = pd.DataFrame(pairs, columns=["feature", "importance"])
                chart_b64 = plot_bars(df_imp, "SHAP (global)")

            # Guardar CSV + tabla
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_path = os.path.join("outputs", f"importances_{form_method}_{ts}.csv")
            df_imp.to_csv(csv_path, index=False, encoding="utf-8")
            session["last_imp_csv"] = csv_path
            csv_ready = True

            df_show = df_imp.sort_values("importance", ascending=False).head(form_topk).fillna("")
            table = [dict(feature=r["feature"], importance=r["importance"],
                          std=r.get("std", ""), type=r.get("type", ""))
                     for _, r in df_show.iterrows()]

        except ImportError as e:
            flash(f"Dependencia faltante: {e}")
        except Exception as e:
            flash(f"Error en interpretabilidad: {e}")

    return render_template("interpret.html",
                           last_model_ready=True,
                           form_method=form_method, form_topk=form_topk,
                           form_nrepeats=form_nrepeats, form_scoring=form_scoring,
                           chart_b64=chart_b64, table=table, csv_ready=csv_ready)


    model_path = session.get("last_model_path")
    task = session.get("last_model_task")  # "classification" | "regression" | "clustering"
    if not model_path or not os.path.exists(model_path):
        return render_template("interpret.html", last_model_ready=False)

    obj = joblib.load(model_path)

    # Aceptar tanto Pipeline como estimador suelto
    pipe = obj if hasattr(obj, "named_steps") else None
    if pipe is not None:
        pre = pipe.named_steps.get("pre", None)
        est = pipe.named_steps.get("model", pipe.steps[-1][1])
    else:
        pre = None
        est = obj  # estimador final

    # Si el último entrenamiento fue de clustering, avisamos que no aplica
    if task == "clustering":
        flash("Interpretabilidad basada en importancias no está disponible para Clustering. "
              "Entrena un modelo supervisado (Clasificación/Regresión) y vuelve aquí.")
        return render_template("interpret.html", last_model_ready=True,
                               form_method="model", form_topk=20,
                               form_nrepeats=10, form_scoring="",
                               chart_b64=None, table=None, csv_ready=False)

    features = session.get("features", [])
    df = pd.read_csv(session["csv_path"])

    # Nombres de features transformadas
    feat_names_out = session.get("last_feature_names_out")
    if not feat_names_out:
        try:
            feat_names_out = get_feature_names_out(pre, features).tolist()
        except Exception:
            feat_names_out = features
    feat_names_out = list(feat_names_out)

    form_method = "model"
    form_topk = 20
    form_nrepeats = 10
    form_scoring = ""
    chart_b64 = None
    table = []
    csv_ready = False

    if request.method == "POST":
        form_method = request.form.get("method", "model")
        form_topk = int(request.form.get("topk", 20))
        form_nrepeats = int(request.form.get("n_repeats", 10))
        form_scoring = request.form.get("scoring", "") or None

        def plot_bars(df_imp, title="Importancias"):
            fig = Figure(figsize=(6, 3.2)); ax = fig.add_subplot(111)
            d = df_imp.sort_values("importance", ascending=False).head(form_topk)
            ax.barh(list(d["feature"])[::-1], list(d["importance"])[::-1])
            ax.set_title(title); ax.set_xlabel("Importance"); ax.set_ylabel("")
            return fig_to_b64(fig)

        try:
            # --- MODEL FEATURE IMPORTANCE ---
            if form_method == "model":
                df_imp = None

                # 1) Intenta usar tu core con diversas firmas
                try:
                    df_imp = core_interp.model_feature_importance(est, feature_names=feat_names_out)
                except TypeError:
                    try:
                        df_imp = core_interp.model_feature_importance(est, feat_names_out)
                    except TypeError:
                        try:
                            df_imp = core_interp.model_feature_importance(est)
                            # Si devuelve sin nombres, los añadimos si encaja el largo
                            if hasattr(df_imp, "columns") and "feature" in df_imp.columns:
                                pass
                            elif hasattr(df_imp, "__len__") and len(df_imp) == len(feat_names_out):
                                df_imp = pd.DataFrame({
                                    "feature": feat_names_out,
                                    "importance": np.asarray(df_imp).ravel()
                                })
                        except Exception:
                            df_imp = None

                # 2) Fallback manual si el core no pudo
                if df_imp is None:
                    if hasattr(est, "feature_importances_"):  # Árboles / GBM / RF
                        vals = np.asarray(est.feature_importances_).ravel()
                        df_imp = pd.DataFrame({"feature": feat_names_out[:len(vals)],
                                               "importance": vals[:len(feat_names_out)]})
                    elif hasattr(est, "coef_"):  # Lineales / SVM lineal / LogReg
                        co = np.asarray(est.coef_)
                        vals = np.mean(np.abs(co), axis=0) if co.ndim > 1 else np.abs(co)
                        vals = np.asarray(vals).ravel()
                        df_imp = pd.DataFrame({"feature": feat_names_out[:len(vals)],
                                               "importance": vals[:len(feat_names_out)]})
                    else:
                        raise ValueError("El estimador no expone importancias (feature_importances_ / coef_).")

                chart_b64 = plot_bars(df_imp, "Model feature importance")

            # --- PERMUTATION IMPORTANCE ---
            elif form_method == "permutation":
                if not session.get("has_labels"):
                    flash("Permutation importance requiere objetivo (supervisado).")
                    return render_template("interpret.html", last_model_ready=True,
                                           form_method=form_method, form_topk=form_topk,
                                           form_nrepeats=form_nrepeats, form_scoring=form_scoring,
                                           chart_b64=None, table=None, csv_ready=False)
                if pipe is None:
                    flash("El último modelo no se guardó como Pipeline, por lo que no podemos aplicar "
                          "Permutation Importance de forma segura. Vuelve a entrenar para regenerar el pipeline.")
                    return render_template("interpret.html", last_model_ready=True,
                                           form_method=form_method, form_topk=form_topk,
                                           form_nrepeats=form_nrepeats, form_scoring=form_scoring,
                                           chart_b64=None, table=None, csv_ready=False)

                y = df[session["target"]]
                X = df[features].copy()
                Xtr, Xte, ytr, yte = train_test_split(
                    X, y, test_size=0.2, random_state=42,
                    stratify=y if task == "classification" else None
                )
                df_imp = core_interp.permutation_feature_importance_df(
                    pipe, Xte, yte, scoring=form_scoring, n_repeats=form_nrepeats, random_state=42
                )
                chart_b64 = plot_bars(df_imp, "Permutation importance")

            # --- SHAP ---
            else:
                X = df[features].copy()
                Xp = pre.transform(X) if pre is not None else X
                Xp = to_dense_float(Xp)
                Xp_df = pd.DataFrame(Xp, columns=feat_names_out)
                df_imp = core_interp.shap_global_importance(est, Xp_df, max_samples=2000)
                chart_b64 = plot_bars(df_imp, "SHAP (global)")

            # Guardar CSV y tabla
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_path = os.path.join("outputs", f"importances_{form_method}_{ts}.csv")
            df_imp.to_csv(csv_path, index=False, encoding="utf-8")
            session["last_imp_csv"] = csv_path
            csv_ready = True

            df_show = df_imp.head(form_topk).fillna("")
            table = [dict(feature=r["feature"], importance=r["importance"],
                          std=r.get("std", ""), type=r.get("type", ""))
                     for _, r in df_show.iterrows()]

        except ImportError as e:
            flash(f"Dependencia faltante: {e}")
        except Exception as e:
            flash(f"Error en interpretabilidad: {e}")

    return render_template("interpret.html",
                           last_model_ready=True,
                           form_method=form_method, form_topk=form_topk,
                           form_nrepeats=form_nrepeats, form_scoring=form_scoring,
                           chart_b64=chart_b64, table=table, csv_ready=csv_ready)


@app.route("/download_importances")
def download_importances():
    path = session.get("last_imp_csv")
    if path and os.path.exists(path):
        return send_file(path, mimetype="text/csv", as_attachment=True,
                         download_name=os.path.basename(path))
    bio = BytesIO("feature,importance,std,type\n".encode("utf-8"))
    return send_file(bio, mimetype="text/csv", as_attachment=True, download_name="importances.csv")

if __name__ == "__main__":
    app.run(debug=True)
