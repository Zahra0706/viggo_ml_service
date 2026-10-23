"""Prétraitement reproduisant la cellule 10 du notebook : BGR, resize 224, 8 segments."""
from pathlib import Path
import cv2
import numpy as np
import torch

N_SEGMENTS = 8
MEAN = {"rgb": (0.485, 0.456, 0.406), "ir": (0.485, 0.456, 0.406), "depth": (0.5, 0.5, 0.5)}
STD = {"rgb": (0.229, 0.224, 0.225), "ir": (0.229, 0.224, 0.225), "depth": (0.5, 0.5, 0.5)}

def load_stream(folder, modality):
    """Accepte un dossier de frames JPEG ordonnées; erreur si données incomplètes."""
    folder = Path(folder)
    files = sorted([*folder.glob("frame_*.jpg")])
    if len(files) < N_SEGMENTS:
        raise ValueError(f"{folder}: au moins {N_SEGMENTS} frames sont nécessaires (trouvées: {len(files)}).")
    indices = [min(int((i + .5) * len(files) / N_SEGMENTS), len(files)-1) for i in range(N_SEGMENTS)]
    frames = []
    for idx in indices:
        im = cv2.imread(str(files[idx]), cv2.IMREAD_COLOR)
        if im is None:
            raise ValueError(f"Image illisible: {files[idx]}")
        im = cv2.resize(im, (224, 224), interpolation=cv2.INTER_LINEAR)
        frames.append(im)
    arr = np.stack(frames).transpose(0, 3, 1, 2).copy()
    data = torch.from_numpy(arr).float().div_(255)
    mean = torch.tensor(MEAN[modality]).view(1,3,1,1)
    std = torch.tensor(STD[modality]).view(1,3,1,1)
    return ((data - mean) / std).unsqueeze(0)

def load_streams(root, mode):
    root = Path(root)
    keys = [f"{mod}_{view}" for mod in (["rgb"] if mode=="rgb" else ["ir", "depth"]) for view in ("body", "face", "hands")]
    return {key: load_stream(root / f"frames_{key}", key.split('_')[0]) for key in keys}
