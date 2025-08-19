import os
from typing import List

import joblib
import pandas as pd
from sklearn.preprocessing import RobustScaler
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import Transaction, Result, TestSummary

def load_models():
    brf = joblib.load(os.path.join(os.getcwd(), "models", "balanced_random_forest_model.joblib"))
    iso = joblib.load(os.path.join(os.getcwd(), "models", "isolated_forest_model.joblib"))
    lr = joblib.load(os.path.join(os.getcwd(), "models", "logistic_regression_model.joblib"))
    return brf, iso, lr

def load_test_data():
    df = pd.read_csv(os.path.join(os.getcwd(), "data", "test_data.csv"))
    return df

@asynccontextmanager
async def lifespan(app: FastAPI):
    brf, iso, lr = load_models()
    app.state.models = {"brf": brf, "iso": iso, "lr": lr}
    app.state.test_data = load_test_data()
    app.state.scaler = joblib.load(os.path.join(os.getcwd(), "models", "scaler.joblib"))
    with open(os.path.join(os.getcwd(), "data", "feature_order.txt"), "r") as f:
        app.state.feature_order = [line.strip() for line in f.readlines()]
    yield
    app.state.models = None
    app.state.test_data = None
    app.state.scaler = None
    app.state.feature_order = None

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
    scaler = app.state.scaler
    feature_order = app.state.feature_order[:-1]

    data = pd.DataFrame([transaction.model_dump(exclude={"fraud"})])

    if feature_order is not None:
        missing = set(feature_order) - set(data.columns)
        if missing:
            raise ValueError(f"Missing features in test set: {sorted(missing)}")
        data = data.reindex(columns=feature_order)

    data = scaler.transform(data)

    brf_pred = int(brf.predict(data)[0])
    iso_pred = int(iso.predict(data)[0] == -1)
    lr_pred = int(lr.predict(data)[0])

    return Result(
        brf=bool(brf_pred),
        iso=bool(iso_pred),
        lr=bool(lr_pred),
        true=transaction.fraud
    )

@app.post("/test", response_model=List[TestSummary])
def test_models():
    assert isinstance(app.state.test_data, pd.DataFrame), "Test data not loaded"
    df = app.state.test_data
    brf = app.state.models["brf"]
    iso = app.state.models["iso"]
    lr = app.state.models["lr"]
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

    summaries = []
    for model_name, preds in [("brf", brf_pred), ("iso", iso_pred), ("lr", lr_pred)]:
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