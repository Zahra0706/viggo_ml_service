import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock

import cv2
import torch
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel

from inference import load_model, CLASSES
from model_architecture import RGBDriverFusionNet
from preprocessing import load_streams
from swin_service import SwinInferenceService

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "rgb_best_model.pth"
SAMPLES_DIR = (BASE_DIR / "samples").resolve()

# Constantes pour la validation des vidéos capturées
MAX_VIDEO_SIZE_BYTES = 25 * 1024 * 1024  # 25 Mo max pour le prototype
MAX_VIDEO_DURATION_SECONDS = 30.0        # 30 secondes max (prototype VIGGO = 8 secondes)
MAX_IMAGE_SIZE_BYTES = 5 * 1024 * 1024   # 5 Mo max pour une frame JPEG/PNG

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

    # Préchargement lazy / démarrage Swin ONNX
    try:
        SwinInferenceService.get_instance()
    except Exception as e:
        print(f"Avertissement lors du prechargement Swin: {e}")

    yield

    model = None


app = FastAPI(
    title="VIGGO ML API",
    version="1.2",
    lifespan=lifespan
)


class PredictionRequest(BaseModel):
    frames_root: str


@app.get("/health")
def health():
    swin_instance = SwinInferenceService._instance
    return {
        "status": "ready" if model is not None else "not_ready",
        "model": "RGBDriverFusionNet",
        "loaded": model is not None,
        "swin_model": "driver_swin.onnx",
        "swin_loaded": swin_instance is not None
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


@app.post("/video/inspect")
async def inspect_video(file: UploadFile = File(...)):
    """
    Vérifie techniquement et extrait les métadonnées d'une vidéo capturée.
    - Vérification de signature conteneur (magic bytes MP4).
    - Validation de la décodabilité réelle via OpenCV (sonde de décodage sans chargement intégral).
    - Extraction de durée, résolution, FPS et nombre de frames.
    - Nettoyage systématique des fichiers temporaires.
    """
    filename = Path(file.filename or "video.mp4").name

    # Répertoire temporaire dédié
    temp_dir = Path(tempfile.gettempdir()) / "viggo_video_inspect"
    temp_dir.mkdir(parents=True, exist_ok=True)

    temp_file = tempfile.NamedTemporaryFile(dir=temp_dir, suffix=".mp4", delete=False)
    temp_path = Path(temp_file.name)
    total_bytes = 0

    try:
        try:
            chunk_size = 64 * 1024  # 64 Ko
            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > MAX_VIDEO_SIZE_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail="La taille de la vidéo dépasse la limite autorisée de 25 Mo."
                    )
                temp_file.write(chunk)
            temp_file.flush()
        finally:
            temp_file.close()

        if total_bytes == 0:
            raise HTTPException(
                status_code=400,
                detail="Le fichier vidéo est vide (0 octet)."
            )

        # 1. Vérification de la signature du conteneur MP4 (ftyp box dans les 16 premiers octets)
        with open(temp_path, "rb") as f:
            header = f.read(32)

        if len(header) < 8 or b"ftyp" not in header[:16]:
            raise HTTPException(
                status_code=400,
                detail="Signature de conteneur MP4 invalide ou absente."
            )

        # 2. Ouverture et inspection OpenCV (OpenCV est déjà utilisé par preprocessing.py et
        # évite toute dépendance binaire externe comme ffprobe sur l'environnement Windows)
        cap = cv2.VideoCapture(str(temp_path))
        if not cap.isOpened():
            raise HTTPException(
                status_code=400,
                detail="Impossible d'ouvrir le conteneur vidéo avec le décodeur OpenCV."
            )

        try:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if width <= 0 or height <= 0:
                raise HTTPException(
                    status_code=400,
                    detail="Dimensions vidéo invalides détectées."
                )

            # Test de décodabilité réelle sur une frame (évite le décodage intégral inutile)
            ret, first_frame = cap.read()
            if not ret or first_frame is None:
                raise HTTPException(
                    status_code=400,
                    detail="Flux vidéo corrompu : impossible de décoder les frames."
                )

            # Calcul de durée
            duration_seconds = round(frame_count / fps, 2) if (fps > 0 and frame_count > 0) else 0.0

            if duration_seconds > MAX_VIDEO_DURATION_SECONDS:
                raise HTTPException(
                    status_code=400,
                    detail=f"Durée vidéo excessive ({duration_seconds}s). Le maximum autorisé est de {MAX_VIDEO_DURATION_SECONDS}s."
                )

            return {
                "success": True,
                "data": {
                    "filename": filename,
                    "duration_seconds": duration_seconds,
                    "fps": round(fps, 2) if fps > 0 else 0.0,
                    "width": width,
                    "height": height,
                    "frame_count": frame_count,
                    "size_bytes": total_bytes,
                    "status": "video_valid"
                }
            }
        finally:
            cap.release()

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Erreur lors de l'inspection de la vidéo : {str(e)}"
        )
    finally:
        # 3. Nettoyage systématique du fichier temporaire sur le serveur Python
        try:
            if temp_path.exists():
                temp_path.unlink()
        except Exception:
            pass


@app.post("/video/predict-swin")
async def predict_video_swin(
    file: UploadFile = File(...),
    num_frames: int = 5
):
    """
    Exécute l'inférence Swin Transformer ONNX sur une vidéo Android capturée.
    - Échantillonnage uniforme de num_frames (défaut 5).
    - Validation signature conteneur MP4 et taille maximale (25 Mo).
    - Classification image par image sans modèle temporel.
    - Agrégation des prédictions, métriques de confiance et avertissements.
    - Nettoyage systématique du fichier temporaire.
    """
    if num_frames < 1 or num_frames > 16:
        raise HTTPException(
            status_code=400,
            detail="Le paramètre num_frames doit être compris entre 1 et 16."
        )

    filename = Path(file.filename or "video.mp4").name
    temp_dir = Path(tempfile.gettempdir()) / "viggo_video_swin"
    temp_dir.mkdir(parents=True, exist_ok=True)

    temp_file = tempfile.NamedTemporaryFile(dir=temp_dir, suffix=".mp4", delete=False)
    temp_path = Path(temp_file.name)
    total_bytes = 0

    try:
        try:
            chunk_size = 64 * 1024
            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > MAX_VIDEO_SIZE_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail="La taille de la vidéo dépasse la limite autorisée de 25 Mo."
                    )
                temp_file.write(chunk)
            temp_file.flush()
        finally:
            temp_file.close()

        if total_bytes == 0:
            raise HTTPException(
                status_code=400,
                detail="Le fichier vidéo est vide (0 octet)."
            )

        # 1. Vérification de la signature du conteneur MP4 (ftyp box dans les 16 premiers octets)
        with open(temp_path, "rb") as f:
            header = f.read(32)

        if len(header) < 8 or b"ftyp" not in header[:16]:
            raise HTTPException(
                status_code=400,
                detail="Signature de conteneur MP4 invalide ou absente."
            )

        # 2. Inférence Swin via SwinInferenceService
        swin_service = SwinInferenceService.get_instance()
        result = swin_service.predict_video(temp_path, num_frames=num_frames)

        return {
            "success": True,
            "data": result
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Erreur lors de l'inférence Swin : {str(e)}"
        )
    finally:
        # Nettoyage systématique du fichier temporaire
        try:
            if temp_path.exists():
                temp_path.unlink()
        except Exception:
            pass


@app.post("/image/predict-swin")
async def predict_image_swin(
    file: UploadFile = File(...)
):
    """
    Exécute l'inférence Swin Transformer ONNX sur une frame unique capturée en temps réel pendant un trajet.
    - Accepte JPEG, PNG, WEBP.
    - Validation stricte de taille (5 Mo max) et des magic bytes réels (aucun stockage disque ni MongoDB).
    - Retourne la classe prédite, les 6 scores, la détection visage et les métriques de latence.
    """
    image_bytes = await file.read(MAX_IMAGE_SIZE_BYTES + 1)
    if len(image_bytes) == 0:
        raise HTTPException(
            status_code=400,
            detail="Le fichier image est vide (0 octet)."
        )

    if len(image_bytes) > MAX_IMAGE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail="La taille de l'image dépasse la limite maximale autorisée de 5 Mo."
        )

    # Validation des magic bytes réels (JPEG: \xFF\xD8\xFF, PNG: \x89PNG\r\n\x1a\n, WEBP: RIFF...WEBP)
    is_jpeg = image_bytes.startswith(b"\xff\xd8\xff")
    is_png = image_bytes.startswith(b"\x89PNG\r\n\x1a\n")
    is_webp = len(image_bytes) >= 12 and image_bytes[:4] == b"RIFF" and image_bytes[8:12] == b"WEBP"

    if not (is_jpeg or is_png or is_webp):
        raise HTTPException(
            status_code=400,
            detail="Format d'image non supporté ou signature invalide. Seuls JPEG, PNG et WEBP sont acceptés."
        )

    try:
        swin_service = SwinInferenceService.get_instance()
        result = swin_service.predict_image(image_bytes)
        return {
            "success": True,
            "data": result
        }
    except ValueError as ve:
        raise HTTPException(
            status_code=400,
            detail=str(ve)
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur interne lors de l'analyse de l'image : {str(e)}"
        )


