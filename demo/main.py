import os
from typing import List

import joblib
import pandas as pd
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
    yield
    app.state.models = None
    app.state.test_data = None

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

    data = pd.DataFrame([transaction.model_dump(exclude={"fraud"})])

    brf_pred = brf.predict(data.drop(columns=["repeat_retailer", "ratio_to_median_purchase_price"]))
    iso_pred = iso.predict(data)
    lr_pred = lr.predict(data)

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

    X = df.drop(columns=["fraud"])
    y = df["fraud"]

    # For brf, drop columns as in ingest_transaction
    X_brf = X.drop(columns=["repeat_retailer", "ratio_to_median_purchase_price"])

    brf_pred = brf.predict(X_brf)
    iso_pred = iso.predict(X)
    lr_pred = lr.predict(X)

    summaries = []
    for model_name, preds in zip(["brf", "iso", "lr"], [brf_pred, iso_pred, lr_pred]):
        accuracy = (preds == y).mean()
        false_positives = ((preds == 1) & (y == 0)).sum()
        missed_frauds = ((preds == 0) & (y == 1)).sum()
        summaries.append(TestSummary(
            model=model_name,
            accuracy=accuracy,
            false_positives=false_positives,
            missed_frauds=missed_frauds
        ))
    return summaries