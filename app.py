# app.py
from fastapi import FastAPI

app = FastAPI(title="VeriFace - Skeleton")

@app.get("/health")
def health():
    return {"status": "ok", "version": "0.1"}
