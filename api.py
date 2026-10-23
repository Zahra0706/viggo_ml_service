
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from inference import load_model, CLASSES
from model_architecture import RGBDriverFusionNet
from preprocessing import load_streams

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "rgb_best_model.pth"
SAMPLES_DIR = (BASE_DIR / "samples").resolve()

model = None
model_lock = Lock()
device = torch.device("cpu")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model

    print("Chargement du modele RGB...")

    model = load_model(
        RGBDriverFusionNet,
        str(MODEL_PATH),
        device
    )

    print("Modele RGB charge avec succes !")

    yield

    model = None


app = FastAPI(
    title="VIGGO ML API",
    version="1.1",
    lifespan=lifespan
)


class PredictionRequest(BaseModel):
    frames_root: str


@app.get("/health")
def health():
    return {
        "status": "ready" if model is not None else "not_ready",
        "model": "RGBDriverFusionNet",
        "loaded": model is not None
    }


@app.post("/predict")
def predict_driver(request: PredictionRequest):

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Modele non charge"
        )

    folder = (BASE_DIR / request.frames_root).resolve()

    # Autoriser uniquement les dossiers de test.
    if not folder.is_relative_to(SAMPLES_DIR):
        raise HTTPException(
            status_code=400,
            detail="Chemin non autorise"
        )

    if not folder.is_dir():
        raise HTTPException(
            status_code=400,
            detail="Dossier introuvable"
        )

    try:
        streams = {
            key: value.to(device)
            for key, value in load_streams(
                folder, "rgb"
            ).items()
        }

        # Eviter des executions simultanees du modele.
        with model_lock:
            with torch.inference_mode():
                outputs = model(streams)

                probabilities = {
                    task: torch.softmax(logits, dim=1)
                    for task, logits in outputs.items()
                }

        predictions = {}

        for task, probs in probabilities.items():
            class_index = int(
                probs.argmax(dim=1).item()
            )

            predictions[task] = {
                "class": CLASSES[task][class_index],
                "class_index": class_index,
                "scores": {
                    label: round(
                        float(probs[0, i].item()), 6
                    )
                    for i, label in enumerate(CLASSES[task])
                }
            }

        return {
            "mode": "rgb_only",
            "predictions": predictions
        }

    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
