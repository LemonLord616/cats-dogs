from pathlib import Path

import httpx
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from config import settings

app = FastAPI(title="Cat vs Dog App")

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/classify")
async def classify(image: UploadFile = File(...)):
    data = await image.read()
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"{settings.api_url}/predict",
                files={"image": (image.filename, data, image.content_type)},
            )
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Model API unavailable: {exc}")


@app.post("/reload-model")
async def reload_model():
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(f"{settings.api_url}/reload-model")
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Model API unavailable: {exc}")


@app.get("/health")
async def health():
    return {"status": "ok"}