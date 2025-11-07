# app.py
from fastapi import FastAPI, File, UploadFile, HTTPException
import shutil, os
from tempfile import NamedTemporaryFile
from VeriFace.inference import ensemble_predict

app = FastAPI(title="VeriFace Inference API")

@app.get("/health")
def health():
    return {"status": "ok", "version": "0.1"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")
    suffix = os.path.splitext(file.filename)[1] or ".mp4"
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        try:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name
        finally:
            file.file.close()
    try:
        result = ensemble_predict(tmp_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
    return result
