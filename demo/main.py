from fastapi import FastAPI

from app.schemas import Transaction

app = FastAPI()

@app.get("/")
def read_root():
    return {"Detail": "Demo API is running"}

@app.post("/ingest")
def ingest_transaction(transaction: Transaction):
    return {"Detail": "Transaction ingested", "Transaction": transaction}