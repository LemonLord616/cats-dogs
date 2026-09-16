import io
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

from cats_dogs.model import CNNClassifier
from config import settings as app_settings

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
app = FastAPI(title="Cat vs Dog Classifier API")


def load_model() -> nn.Module:
    model = CNNClassifier().to(device)
    model_path = Path("/srv/models") / app_settings.model_name
    state = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


model = load_model()


def reload_model() -> None:
    global model
    model = load_model()


def preprocess(image: Image.Image) -> torch.Tensor:
    size = app_settings.model_size
    image = image.convert("RGB").resize((size, size), Image.BILINEAR)
    x = np.asarray(image, dtype="float32") / 255.0
    x = torch.from_numpy(x)
    x = x.permute(2, 0, 1).unsqueeze(0).to(device)
    return x


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/reload-model")
async def reload_model_endpoint():
    try:
        reload_model()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to reload model: {exc}")
    return {"status": "ok", "model": app_settings.model_name}


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    raw = await image.read()
    pil = Image.open(io.BytesIO(raw))
    with torch.no_grad():
        logit = model(preprocess(pil)).squeeze(1).item()
    prob = torch.sigmoid(torch.tensor(logit)).item()
    return {"label": int(prob >= 0.5), "prob": round(prob, 4)}
