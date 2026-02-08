#!/usr/bin/env python3
"""
EchoData — Synthetic Data Studio
Flask backend: file upload, synthesis, ML comparison.
"""

import os
import sys
import io
import json
import uuid
import webbrowser
import threading
import pandas as pd
import numpy as np
from datetime import datetime

from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename

from shattered_synth import ShatteredSynth

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    mean_absolute_error, r2_score,
    accuracy_score, f1_score,
)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200 MB
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# In-memory session store (single-user desktop app)
session_data = {}


def clean_dataframe(df):
    """
    Clean a DataFrame for synthesis:
    - Drop fully-empty and Unnamed columns
    - Convert currency-formatted strings to numeric
    - Strip whitespace from string columns
    """
    df = df.copy()

    # Drop fully-empty columns
    df = df.dropna(axis=1, how="all")

    # Drop Unnamed columns (artifacts from messy CSVs)
    unnamed = [c for c in df.columns if str(c).startswith("Unnamed")]
    if unnamed:
        df = df.drop(columns=unnamed)

    # Clean each column
    for col in df.columns:
        if df[col].dtype == object:
            # Strip whitespace
            df[col] = df[col].astype(str).str.strip()
            # Replace literal 'nan' strings with actual NaN
            df[col] = df[col].replace({"nan": pd.NA, "NaN": pd.NA, "": pd.NA})

            # Check if this looks like currency/numeric data
            non_null = df[col].dropna().astype(str)
            if len(non_null) == 0:
                continue
            cleaned = non_null.str.replace(r'[\$,£€¥₹\s\xa0]', '', regex=True)
            numeric_parse = pd.to_numeric(cleaned, errors="coerce")

            if numeric_parse.notna().sum() >= len(non_null) * 0.7:
                df[col] = df[col].astype(str).str.replace(r'[\$,£€¥₹\s\xa0]', '', regex=True)
                df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# ── Routes ────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    """Upload CSV, return column info."""
    f = request.files.get("file")
    if not f:
        return jsonify(error="No file provided"), 400

    filename = secure_filename(f.filename)
    if not filename.lower().endswith(".csv"):
        return jsonify(error="Only CSV files are supported"), 400

    path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}_{filename}")
    f.save(path)

    try:
        df = pd.read_csv(path)
    except Exception as e:
        return jsonify(error=f"Could not parse CSV: {e}"), 400

    # Clean up the dataframe
    df = clean_dataframe(df)

    # Re-save the cleaned version
    df.to_csv(path, index=False)

    # Detect column types
    col_info = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            nunique = df[col].nunique()
            if nunique <= 10 and nunique / max(len(df), 1) < 0.05:
                ctype = "categorical"
            else:
                ctype = "numeric"
        else:
            ctype = "categorical"
        col_info.append({
            "name": col,
            "type": ctype,
            "nunique": int(df[col].nunique()),
            "nulls": int(df[col].isna().sum()),
            "sample": str(df[col].dropna().iloc[0]) if len(df[col].dropna()) > 0 else "—",
        })

    sid = uuid.uuid4().hex[:12]
    session_data[sid] = {
        "path": path,
        "filename": filename,
        "df": df,
        "synthetic_df": None,
        "synth_path": None,
    }

    return jsonify(
        session=sid,
        filename=filename,
        rows=len(df),
        cols=len(df.columns),
        columns=col_info,
        numeric_count=sum(1 for c in col_info if c["type"] == "numeric"),
        categorical_count=sum(1 for c in col_info if c["type"] == "categorical"),
    )


@app.route("/synthesize", methods=["POST"])
def synthesize():
    """Run synthesis engine."""
    body = request.json
    sid = body.get("session")
    epsilon = float(body.get("epsilon", 1.0))
    num_rows = int(body.get("num_rows", 500))

    sd = session_data.get(sid)
    if not sd:
        return jsonify(error="Session not found"), 404

    df = clean_dataframe(sd["df"].copy())
    sd["df"] = df  # store cleaned version back
    logs = []

    try:
        # Capture prints
        old_stdout = sys.stdout
        sys.stdout = buf = io.StringIO()

        synth = ShatteredSynth(epsilon=epsilon, seed=42)
        synth.shatter(df)
        synthetic_df = synth.generate(n=num_rows)

        sys.stdout = old_stdout
        logs = [l for l in buf.getvalue().split("\n") if l.strip()]

        # Save
        base = os.path.splitext(sd["filename"])[0]
        synth_path = os.path.join(UPLOAD_DIR, f"synthetic_{base}_{sid}.csv")
        synthetic_df.to_csv(synth_path, index=False)

        sd["synthetic_df"] = synthetic_df
        sd["synth_path"] = synth_path

        # Privacy report
        report = synth.get_privacy_report()

        # Quick stats comparison
        stats = []
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]) and col in synthetic_df.columns:
                orig_col = pd.to_numeric(df[col], errors="coerce")
                syn_col = pd.to_numeric(synthetic_df[col], errors="coerce")
                om, sm = orig_col.mean(), syn_col.mean()
                os_, ss = orig_col.std(), syn_col.std()
                delta = abs(sm - om) / (abs(om) + 1e-10) * 100
                stats.append({
                    "column": col,
                    "orig_mean": round(om, 3),
                    "synth_mean": round(sm, 3),
                    "orig_std": round(os_, 3),
                    "synth_std": round(ss, 3),
                    "delta_pct": round(delta, 1),
                })

        # Sanitize NaN/Inf values for JSON
        import math
        def sanitize(obj):
            if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
                return None
            if isinstance(obj, dict):
                return {k: sanitize(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [sanitize(v) for v in obj]
            return obj

        return jsonify(
            ok=True,
            logs=logs,
            report=sanitize(report),
            stats=sanitize(stats),
            synth_rows=len(synthetic_df),
        )

    except Exception as e:
        sys.stdout = sys.__stdout__
        return jsonify(error=str(e)), 500


@app.route("/compare", methods=["POST"])
def compare():
    """Train ML models on both datasets, compare."""
    body = request.json
    sid = body.get("session")
    target_col = body.get("target")

    sd = session_data.get(sid)
    if not sd or sd["synthetic_df"] is None:
        return jsonify(error="No synthetic data. Run synthesis first."), 400

    orig_df = sd["df"]
    synth_df = sd["synthetic_df"]

    if target_col not in orig_df.columns:
        return jsonify(error=f"Column '{target_col}' not found"), 400

    try:
        r1, logs1, artifacts1 = _train_generic(orig_df, target_col, "Original")
        r2, logs2, artifacts2 = _train_generic(synth_df, target_col, "Synthetic")

        # Store model artifacts for later prediction
        sd["model_original"] = artifacts1
        sd["model_synthetic"] = artifacts2
        sd["target_col"] = target_col

        # Generate random sample cases and predict with both models
        sample_cases = _generate_sample_cases(artifacts1, artifacts2, n=5)

        # Verdict
        task_type = r1["task_type"]
        if task_type == "regression":
            v1, v2 = r1["metrics"]["MAE"], r2["metrics"]["MAE"]
            diff = abs(v1 - v2) / (v1 + 1e-10) * 100
            if diff < 10:
                verdict = {"text": f"Excellent — MAE within {diff:.1f}% of each other", "level": "success"}
            elif diff < 25:
                verdict = {"text": f"Good — {diff:.1f}% MAE gap. Synthetic data is usable.", "level": "warning"}
            else:
                verdict = {"text": f"Notable gap — {diff:.1f}% MAE difference. Try adjusting ε.", "level": "error"}
        else:
            v1, v2 = r1["metrics"]["Accuracy"], r2["metrics"]["Accuracy"]
            diff = abs(v1 - v2) * 100
            if diff < 5:
                verdict = {"text": f"Excellent — Accuracy within {diff:.1f} points", "level": "success"}
            elif diff < 15:
                verdict = {"text": f"Good — {diff:.1f}pp accuracy gap.", "level": "warning"}
            else:
                verdict = {"text": f"Notable gap — {diff:.1f}pp accuracy difference.", "level": "error"}

        import math
        def sanitize(obj):
            if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
                return None
            if isinstance(obj, dict):
                return {k: sanitize(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [sanitize(v) for v in obj]
            return obj

        return jsonify(
            ok=True,
            task_type=task_type,
            original=sanitize(r1),
            synthetic=sanitize(r2),
            verdict=verdict,
            logs=logs1 + [""] + logs2,
            sample_cases=sanitize(sample_cases),
            feature_meta=sanitize(r1.get("feature_meta", [])),
        )
    except Exception as e:
        return jsonify(error=str(e)), 500


@app.route("/download/<sid>")
def download(sid):
    sd = session_data.get(sid)
    if not sd or not sd["synth_path"]:
        return "Not found", 404
    return send_file(sd["synth_path"], as_attachment=True)


# ── ML Helper ─────────────────────────────────────────────

def _train_generic(df, target_col, label="Model"):
    logs = [f"═══ Training on {label} data ({len(df)} rows) ═══"]
    df = df.copy().dropna(subset=[target_col])

    # Safety: clean the target column if it looks like disguised numeric data
    if not pd.api.types.is_numeric_dtype(df[target_col]):
        cleaned = df[target_col].astype(str).str.strip().str.replace(r'[\$,£€¥₹\s]', '', regex=True)
        as_numeric = pd.to_numeric(cleaned, errors="coerce")
        if as_numeric.notna().sum() >= len(df) * 0.7:
            df[target_col] = as_numeric
            logs.append(f"  Cleaned {target_col} → numeric (stripped currency symbols)")

    # Detect task
    target_series = df[target_col]
    if pd.api.types.is_numeric_dtype(target_series):
        nunique = target_series.nunique()
        task = "classification" if (nunique <= 15 and nunique / max(len(df), 1) < 0.05) else "regression"
    else:
        task = "classification"

    logs.append(f"  Task: {task}  |  Target: {target_col}")

    feature_cols = [c for c in df.columns if c != target_col]
    encoders = {}
    enc_features = []
    # Track metadata for each feature so we can build prediction inputs later
    feature_meta = []  # [{name, encoded_name, type, categories?, min?, max?, median?}]

    for col in feature_cols:
        # Try to convert disguised numeric columns (currency, etc.)
        if not pd.api.types.is_numeric_dtype(df[col]):
            cleaned = df[col].astype(str).str.strip().str.replace(r'[\$,£€¥₹\s]', '', regex=True)
            as_numeric = pd.to_numeric(cleaned, errors="coerce")
            if as_numeric.notna().sum() >= len(df) * 0.7:
                df[col] = as_numeric

        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors="coerce")
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            enc_features.append(col)
            feature_meta.append({
                "name": col,
                "encoded_name": col,
                "type": "numeric",
                "min": round(float(df[col].min()), 2),
                "max": round(float(df[col].max()), 2),
                "median": round(float(median_val), 2),
            })
        else:
            le = LabelEncoder()
            df[col] = df[col].astype(str).fillna("_missing_")
            df[f"{col}_enc"] = le.fit_transform(df[col])
            encoders[col] = le
            enc_features.append(f"{col}_enc")
            cats = list(le.classes_)
            # Remove internal tokens
            cats = [c for c in cats if c != "_missing_"]
            feature_meta.append({
                "name": col,
                "encoded_name": f"{col}_enc",
                "type": "categorical",
                "categories": cats[:50],  # cap for UI
            })

    X = df[enc_features]
    if task == "regression":
        y = pd.to_numeric(df[target_col], errors="coerce").fillna(0)
    else:
        le_t = LabelEncoder()
        y = le_t.fit_transform(df[target_col].astype(str))
        encoders["__target__"] = le_t

    if len(X) < 10:
        return {"task_type": task, "metrics": {}, "features": [], "feature_meta": feature_meta}, logs

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    if task == "regression":
        model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    else:
        model = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {}
    if task == "regression":
        metrics["MAE"] = round(float(mean_absolute_error(y_test, y_pred)), 4)
        metrics["R²"] = round(float(r2_score(y_test, y_pred)), 4)
        logs.append(f"  MAE: {metrics['MAE']}  |  R²: {metrics['R²']}")
    else:
        metrics["Accuracy"] = round(float(accuracy_score(y_test, y_pred)), 4)
        metrics["F1"] = round(float(f1_score(y_test, y_pred, average="weighted", zero_division=0)), 4)
        logs.append(f"  Accuracy: {metrics['Accuracy']}  |  F1: {metrics['F1']}")

    importances = sorted(
        zip(enc_features, model.feature_importances_),
        key=lambda x: x[1], reverse=True
    )
    top_features = [{"name": n, "importance": round(float(v), 4)} for n, v in importances[:8]]
    logs.append(f"  Top feature: {top_features[0]['name']} ({top_features[0]['importance']:.2%})" if top_features else "")

    # Sample predictions from test set
    sample_preds = []
    num_samples = min(5, len(X_test))
    sample_X = X_test.iloc[:num_samples]
    sample_y_true = np.array(y_test)[:num_samples] if not isinstance(y_test, np.ndarray) else y_test[:num_samples]
    sample_y_pred = model.predict(sample_X)
    for i in range(num_samples):
        sample_preds.append({
            "actual": round(float(sample_y_true[i]), 2) if task == "regression" else str(sample_y_true[i]),
            "predicted": round(float(sample_y_pred[i]), 2) if task == "regression" else str(sample_y_pred[i]),
        })

    result = {
        "task_type": task,
        "metrics": metrics,
        "features": top_features,
        "feature_meta": feature_meta,
        "sample_preds": sample_preds,
    }

    # Store model artifacts for later prediction
    model_artifacts = {
        "model": model,
        "encoders": encoders,
        "enc_features": enc_features,
        "feature_meta": feature_meta,
        "task_type": task,
    }

    return result, logs, model_artifacts


# ── Prediction Helpers ────────────────────────────────────

def _predict_single(artifacts, input_values):
    """Make a prediction using stored model artifacts."""
    model = artifacts["model"]
    encoders = artifacts["encoders"]
    enc_features = artifacts["enc_features"]
    feature_meta = artifacts["feature_meta"]

    row = {}
    for fm in feature_meta:
        name = fm["name"]
        enc_name = fm["encoded_name"]
        val = input_values.get(name)

        if fm["type"] == "numeric":
            try:
                row[enc_name] = float(val) if val is not None else fm.get("median", 0)
            except (ValueError, TypeError):
                row[enc_name] = fm.get("median", 0)
        else:
            le = encoders.get(name)
            if le is None:
                row[enc_name] = 0
            else:
                str_val = str(val) if val is not None else "_missing_"
                if str_val in le.classes_:
                    row[enc_name] = le.transform([str_val])[0]
                else:
                    row[enc_name] = 0

    X = pd.DataFrame([row])[enc_features]
    pred = model.predict(X)[0]
    return round(float(pred), 2)


def _generate_sample_cases(artifacts1, artifacts2, n=5):
    """Generate n random sample cases and predict with both models."""
    meta = artifacts1["feature_meta"]
    cases = []

    for _ in range(n):
        input_vals = {}
        for fm in meta:
            if fm["type"] == "numeric":
                lo, hi = fm.get("min", 0), fm.get("max", 100)
                val = round(np.random.uniform(lo, hi), 2)
                if lo == int(lo) and hi == int(hi) and (hi - lo) < 500:
                    val = int(round(val))
                input_vals[fm["name"]] = val
            else:
                cats = fm.get("categories", ["Unknown"])
                val = np.random.choice(cats) if cats else "Unknown"
                input_vals[fm["name"]] = val

        pred1 = _predict_single(artifacts1, input_vals)
        pred2 = _predict_single(artifacts2, input_vals)

        cases.append({
            "inputs": input_vals,
            "pred_original": pred1,
            "pred_synthetic": pred2,
            "diff": round(abs(pred1 - pred2), 2),
        })

    return cases


@app.route("/predict", methods=["POST"])
def predict():
    """Predict with both models given user-provided feature values."""
    body = request.json
    sid = body.get("session")
    input_values = body.get("inputs", {})

    sd = session_data.get(sid)
    if not sd:
        return jsonify(error="Session not found"), 404

    a1 = sd.get("model_original")
    a2 = sd.get("model_synthetic")
    if not a1 or not a2:
        return jsonify(error="Models not trained yet."), 400

    try:
        pred1 = _predict_single(a1, input_values)
        pred2 = _predict_single(a2, input_values)

        import math
        def sanitize(obj):
            if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
                return None
            return obj

        return jsonify(
            ok=True,
            pred_original=sanitize(pred1),
            pred_synthetic=sanitize(pred2),
            diff=sanitize(round(abs(pred1 - pred2), 2)),
        )
    except Exception as e:
        return jsonify(error=str(e)), 500


@app.route("/random_cases", methods=["POST"])
def random_cases():
    """Generate new random sample cases."""
    body = request.json
    sid = body.get("session")
    n = int(body.get("n", 5))

    sd = session_data.get(sid)
    if not sd:
        return jsonify(error="Session not found"), 404

    a1 = sd.get("model_original")
    a2 = sd.get("model_synthetic")
    if not a1 or not a2:
        return jsonify(error="Models not trained yet."), 400

    try:
        cases = _generate_sample_cases(a1, a2, n=n)

        import math
        def sanitize(obj):
            if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
                return None
            if isinstance(obj, dict):
                return {k: sanitize(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [sanitize(v) for v in obj]
            return obj

        return jsonify(ok=True, cases=sanitize(cases))
    except Exception as e:
        return jsonify(error=str(e)), 500


# ── Entry point ───────────────────────────────────────────

if __name__ == "__main__":
    port = 5111
    print(f"\n  ◈ EchoData — Synthetic Data Studio")
    print(f"  Open http://localhost:{port} in your browser\n")

    # Auto-open browser after short delay
    def open_browser():
        import time
        time.sleep(1.2)
        webbrowser.open(f"http://localhost:{port}")

    threading.Thread(target=open_browser, daemon=True).start()

    app.run(host="127.0.0.1", port=port, debug=False)
