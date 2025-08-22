import os
from typing import List, Dict, Any

import joblib
import pandas as pd
import numpy as np
import shap
from sklearn.preprocessing import RobustScaler
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import Transaction, Result, TestSummary

def _topk_from_exp(exp, feature_names: List[str], k: int = 8) -> Dict[str, Any]:
    vals = exp.values[0][0] if np.ndim(exp.values[0]) > 1 else exp.values[0]
    base_values = exp.base_values[0]
    base = float(base_values[0] if np.ndim(base_values) else base_values)
    features = np.array(feature_names)
    idx = np.argsort(np.abs(vals))[::-1][:k]
    return {
        "base_value": base,
        "features": [
            {"name": features[i], "shap": float(vals[i])}
            for i in idx
        ],
    }

def load_models():
    brf = joblib.load(os.path.join(os.getcwd(), "models", "balanced_random_forest_model.joblib"))
    iso = joblib.load(os.path.join(os.getcwd(), "models", "isolated_forest_model.joblib"))
    lr = joblib.load(os.path.join(os.getcwd(), "models", "logistic_regression_model.joblib"))
    xgb = joblib.load(os.path.join(os.getcwd(), "models", "xgboost_model.joblib"))
    return brf, iso, lr, xgb

def load_test_data():
    df = pd.read_csv(os.path.join(os.getcwd(), "data", "test_data.csv"))
    return df

@asynccontextmanager
async def lifespan(app: FastAPI):
    brf, iso, lr, xgb = load_models()
    app.state.models = {"brf": brf, "iso": iso, "lr": lr, "xgb": xgb}
    app.state.test_data = load_test_data()
    app.state.scaler = joblib.load(os.path.join(os.getcwd(), "models", "scaler.joblib"))
    app.state.background = {
        "no_smote": pd.read_csv(os.path.join(os.getcwd(), "data", "no_smote_background.csv")),
        "smote": pd.read_csv(os.path.join(os.getcwd(), "data", "smote_background.csv"))
    }
    with open(os.path.join(os.getcwd(), "data", "feature_order.txt"), "r") as f:
        app.state.feature_order = [line.strip() for line in f.readlines()]
    app.state.explainers = {
        "brf": shap.Explainer(app.state.models["brf"], app.state.background["smote"]),
        "iso": shap.Explainer(app.state.models["iso"], app.state.background["no_smote"]),
        "lr": shap.Explainer(app.state.models["lr"], app.state.background["smote"]),
        "xgb": shap.Explainer(app.state.models["xgb"], app.state.background["no_smote"])
    }
    yield
    app.state.models = None
    app.state.test_data = None
    app.state.scaler = None
    app.state.background = None
    app.state.feature_order = None
    app.state.explainers = None

app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"Detail": "Demo API is running"}

@app.get("/home")
def serve_home():
    with open("static/main.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.get("/random-transaction", response_model=Transaction)
def get_random_transaction(type: str):
    if type not in ["legit", "fraud"]:
        return {"Error": "Invalid transaction type"}
    fraud = 1 if type == "fraud" else 0
    assert isinstance(app.state.test_data, pd.DataFrame), "Test data not loaded"
    filtered_data = app.state.test_data[app.state.test_data["fraud"] == fraud]
    data = filtered_data.sample().to_dict(orient="records")[0]
    transaction = Transaction(**data)
    return transaction

@app.post("/ingest", response_model=Result)
def ingest_transaction(transaction: Transaction):
    brf = app.state.models["brf"]
    iso = app.state.models["iso"]
    lr = app.state.models["lr"]
    xgb = app.state.models["xgb"]
    scaler = app.state.scaler
    feature_order = app.state.feature_order[:-1]
    explainers = app.state.explainers

    data = pd.DataFrame([transaction.model_dump(exclude={"fraud"})])

    if feature_order is not None:
        missing = set(feature_order) - set(data.columns)
        if missing:
            raise ValueError(f"Missing features in test set: {sorted(missing)}")
        data = data.reindex(columns=feature_order)

    data = pd.DataFrame(scaler.transform(data), columns=feature_order)

    brf_pred = int(brf.predict(data)[0])
    iso_pred = int(iso.predict(data)[0] == -1)
    lr_pred = int(lr.predict(data)[0])
    xgb_pred = int(xgb.predict(data)[0])

    brf_exp = explainers["brf"](data)
    iso_exp = explainers["iso"](data)
    lr_exp = explainers["lr"](data)
    xgb_exp = explainers["xgb"](data)

    shaps = {
        "brf": _topk_from_exp(brf_exp, feature_order),
        "iso": _topk_from_exp(iso_exp, feature_order),
        "lr": _topk_from_exp(lr_exp, feature_order),
        "xgb": _topk_from_exp(xgb_exp, feature_order)
    }

    pos = {"brf": [], "iso": [], "lr": [], "xgb": []}
    neg = {"brf": [], "iso": [], "lr": [], "xgb": []}

    for key, value in shaps.items():
        for feat in value["features"]:
            if feat["shap"] > 0:
                pos[key].append(feat["name"])
            else:
                neg[key].append(feat["name"])
            pos[key].sort(key=lambda name: next(feat["shap"] for feat in value["features"] if feat["name"] == name), reverse=True)
            neg[key].sort(key=lambda name: abs(next(feat["shap"] for feat in value["features"] if feat["name"] == name)), reverse=True)

    shap_brf = "Indicates legit: " + "\n\t".join(neg["brf"]) + "\n\nIndicates fraud: " + "\n\t".join(pos["brf"])
    shap_iso = "Indicates legit: " + "\n\t".join(pos["iso"]) + "\n\nIndicates fraud: " + "\n\t".join(neg["iso"])
    shap_lr = "Indicates legit: " + "\n\t".join(neg["lr"]) + "\n\nIndicates fraud: " + "\n\t".join(pos["lr"])
    shap_xgb = "Indicates legit: " + "\n\t".join(neg["xgb"]) + "\n\nIndicates fraud: " + "\n\t".join(pos["xgb"])

    return Result(
        brf=bool(brf_pred),
        iso=bool(iso_pred),
        lr=bool(lr_pred),
        xgb=bool(xgb_pred),
        true=transaction.fraud,
        shap_brf=shap_brf,
        shap_iso=shap_iso,
        shap_lr=shap_lr,
        shap_xgb=shap_xgb
    )

@app.post("/test", response_model=List[TestSummary])
def test_models():
    assert isinstance(app.state.test_data, pd.DataFrame), "Test data not loaded"
    df = app.state.test_data
    brf = app.state.models["brf"]
    iso = app.state.models["iso"]
    lr = app.state.models["lr"]
    xgb = app.state.models["xgb"]
    scaler = app.state.scaler
    feature_order = app.state.feature_order[:-1]

    X = df.drop(columns=["fraud"])
    y = df["fraud"].to_numpy()

    if feature_order is not None:
        missing = set(feature_order) - set(X.columns)
        if missing:
            raise ValueError(f"Missing features in test set: {sorted(missing)}")
        X = X.reindex(columns=feature_order)

    X = scaler.transform(X)

    brf_pred = brf.predict(X).astype(int)
    iso_raw = iso.predict(X)
    iso_pred = (iso_raw == -1).astype(int)
    lr_pred = lr.predict(X).astype(int)
    xgb_pred = xgb.predict(X).astype(int)

    summaries = []
    for model_name, preds in [("brf", brf_pred), ("iso", iso_pred), ("lr", lr_pred), ("xgb", xgb_pred)]:
        accuracy = float((preds == y).mean())
        false_positives = int(((preds == 1) & (y == 0)).sum())
        missed_frauds = int(((preds == 0) & (y == 1)).sum())
        miss_percentage = float(missed_frauds) / y.sum() if y.sum() > 0 else 0.0
        summaries.append(TestSummary(
            model=model_name,
            accuracy=accuracy,
            false_positives=false_positives,
            missed_frauds=missed_frauds,
            miss_percentage=miss_percentage
        ))
    return summaries