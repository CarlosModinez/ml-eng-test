import tempfile
import os

import cv2
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
import io

from core.pipeline import WallDetectionPipeline

app = FastAPI(title="Wall Detection API")
pipeline = WallDetectionPipeline()


@app.post("/detect-walls")
async def detect_walls(file: UploadFile):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = pipeline.run(tmp_path)
        _, buf = cv2.imencode(".png", result)
        return StreamingResponse(
            io.BytesIO(buf.tobytes()),
            media_type="image/png",
            headers={"Content-Disposition": f"inline; filename=walls_{file.filename}.png"},
        )
    finally:
        os.unlink(tmp_path)


@app.get("/health")
async def health():
    return {"status": "ok"}