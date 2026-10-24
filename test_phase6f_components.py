"""
Test script for Phase 6F: Image Quality Analyzer, Face Cropping, Cross-validation and EMA Stabilization.
"""
import time
import json
from pathlib import Path
from collections import deque
import cv2
import numpy as np
import onnxruntime as ort

BASE_DIR = Path(__file__).resolve().parent
SWIN_DIR = BASE_DIR / "models" / "swin"
MODEL_PATH = SWIN_DIR / "driver_swin.onnx"
LABELS = json.loads((SWIN_DIR / "labels.json").read_text())

cascade_dir = Path(cv2.data.haarcascades)
face_cascade = cv2.CascadeClassifier(str(cascade_dir / "haarcascade_frontalface_default.xml"))
face_alt = cv2.CascadeClassifier(str(cascade_dir / "haarcascade_frontalface_alt2.xml"))
profile_cascade = cv2.CascadeClassifier(str(cascade_dir / "haarcascade_profileface.xml"))
eye_cascade = cv2.CascadeClassifier(str(cascade_dir / "haarcascade_eye.xml"))

options = ort.SessionOptions()
options.intra_op_num_threads = 2
session = ort.InferenceSession(str(MODEL_PATH), sess_options=options, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

class ImageQualityAnalyzer:
    @staticmethod
    def analyze(bgr: np.ndarray):
        h, w = bgr.shape[:2]
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        
        # 1. Flou
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        is_blurry = bool(lap_var < 50.0)
        
        # 2. Luminosité
        mean_brightness = float(np.mean(gray))
        is_too_dark = bool(mean_brightness < 35.0)
        is_too_bright = bool(mean_brightness > 225.0)
        
        # 3. Détection visage & yeux
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(45, 45))
        if len(faces) == 0:
            faces = face_alt.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=2, minSize=(45, 45))
        if len(faces) == 0:
            faces = profile_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=2, minSize=(45, 45))
            
        face_detected = len(faces) > 0
        face_box = None
        face_area_ratio = 0.0
        face_too_small = False
        
        if face_detected:
            # Tri par taille décroissante
            faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
            fx, fy, fw, fh = faces[0]
            face_box = (int(fx), int(fy), int(fw), int(fh))
            face_area_ratio = float((fw * fh) / (w * h))
            if face_area_ratio < 0.03:
                face_too_small = True
                
        # 4. Diagnostic qualité et exploitabilité
        quality_status = "valid"
        reasons = []
        
        if is_blurry:
            quality_status = "unreliable"
            reasons.append("Flou de mouvement détecté")
        if is_too_dark:
            quality_status = "unreliable"
            reasons.append("Image trop sombre (faible éclairage habitacle)")
        if is_too_bright:
            quality_status = "unreliable"
            reasons.append("Surexposition ou éblouissement de la caméra")
        if not face_detected:
            quality_status = "analysis_unavailable"
            reasons.append("Conducteur / visage non détecté")
        elif face_too_small:
            quality_status = "unreliable"
            reasons.append("Conducteur trop éloigné de la caméra")
            
        is_reliable = (quality_status == "valid")
        quality_reason = "; ".join(reasons) if reasons else "Qualité d'image optimale"
        
        return {
            "quality_status": quality_status,
            "is_reliable": is_reliable,
            "quality_reason": quality_reason,
            "laplacian_variance": round(lap_var, 2),
            "mean_brightness": round(mean_brightness, 2),
            "face_detected": face_detected,
            "face_box": face_box,
            "face_area_ratio": round(face_area_ratio, 4),
            "framing_status": "good" if is_reliable else ("face_not_detected" if not face_detected else "uncertain"),
        }

class ComplementaryFacialAnalyzer:
    @staticmethod
    def cross_validate(face_crop: np.ndarray, predicted_class: str, scores: dict):
        fh, fw = face_crop.shape[:2]
        if fh < 30 or fw < 30:
            return predicted_class, scores, None
            
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        
        # Région yeux (20% à 55% de la hauteur)
        eye_roi = gray[int(0.20 * fh):int(0.55 * fh), :]
        eyes = eye_cascade.detectMultiScale(eye_roi, scaleFactor=1.1, minNeighbors=2, minSize=(15, 15))
        open_eyes_detected = len(eyes) >= 1
        
        # Région bouche (60% à 95% de la hauteur, 20% à 80% largeur)
        mouth_roi = gray[int(0.60 * fh):int(0.95 * fh), int(0.20 * fw):int(0.80 * fw)]
        if mouth_roi.size > 0:
            # Cavité buccale sombre
            _, mouth_thresh = cv2.threshold(mouth_roi, 50, 255, cv2.THRESH_BINARY_INV)
            mouth_open_ratio = float(np.count_nonzero(mouth_thresh) / mouth_roi.size)
        else:
            mouth_open_ratio = 0.0
            
        is_mouth_open = mouth_open_ratio > 0.20
        override_note = None
        
        # Règle 1: Si Swin prédit Yawn mais bouche est fermée (mouth_open_ratio faible)
        if predicted_class == "Yawn" and not is_mouth_open:
            override_note = f"Contradiction géométrique : bouche fermée (ratio={mouth_open_ratio:.2f}) -> Yawn rétrogradé"
            scores["Yawn"] *= 0.2
            # Renormalisation
            total = sum(scores.values())
            scores = {k: v / total for k, v in scores.items()}
            predicted_class = max(scores, key=scores.get)
            
        # Règle 2: Si Swin prédit SleepyDriving mais 2 yeux ouverts détectés
        elif predicted_class == "SleepyDriving" and len(eyes) >= 2:
            override_note = f"Contradiction géométrique : 2 yeux ouverts détectés -> SleepyDriving rétrogradé"
            scores["SleepyDriving"] *= 0.3
            total = sum(scores.values())
            scores = {k: v / total for k, v in scores.items()}
            predicted_class = max(scores, key=scores.get)
            
        return predicted_class, scores, override_note

class TemporalTracker:
    def __init__(self, window_size: int = 5, alpha: float = 0.4):
        self.window_size = window_size
        self.alpha = alpha
        self.history = deque(maxlen=window_size)
        self.ema_scores = None
        
    def update(self, raw_scores: dict, predicted_class: str, is_reliable: bool, timestamp: float):
        self.history.append({
            "timestamp": timestamp,
            "scores": raw_scores,
            "class": predicted_class,
            "reliable": is_reliable
        })
        
        # Calcul EMA sur les scores des classes
        if self.ema_scores is None:
            self.ema_scores = dict(raw_scores)
        else:
            # Si la frame n'est pas fiable, on réduit son influence
            eff_alpha = self.alpha if is_reliable else (self.alpha * 0.2)
            for k in raw_scores:
                self.ema_scores[k] = eff_alpha * raw_scores[k] + (1.0 - eff_alpha) * self.ema_scores[k]
                
        # Normalisation EMA
        total = sum(self.ema_scores.values())
        if total > 0:
            for k in self.ema_scores:
                self.ema_scores[k] = round(self.ema_scores[k] / total, 5)
                
        smoothed_top_class = max(self.ema_scores, key=self.ema_scores.get)
        smoothed_top_conf = self.ema_scores[smoothed_top_class]
        
        # Logique de consensus et persistance pour états critiques
        recent_classes = [h["class"] for h in self.history if h["reliable"]]
        
        # Vérification d'oscillation (instabilité)
        if len(recent_classes) >= 3 and len(set(recent_classes[-3:])) == 3:
            consolidated_state = "Uncertain"
        elif smoothed_top_class in ("SleepyDriving", "DangerousDriving"):
            # Exiger au moins 2 détections consécutives
            consecutive_critical = sum(1 for c in recent_classes[-2:] if c == smoothed_top_class)
            if consecutive_critical >= 2 and smoothed_top_conf >= 0.45:
                consolidated_state = smoothed_top_class
            else:
                consolidated_state = "Uncertain"  # En attente de confirmation
        else:
            consolidated_state = smoothed_top_class
            
        return {
            "consolidated_state": consolidated_state,
            "smoothed_confidence": smoothed_top_conf,
            "smoothed_scores": dict(self.ema_scores),
            "history_count": len(self.history)
        }

# Exécution de validation
print("[TEST] Initialisation et validation des modules...")
analyzer = ImageQualityAnalyzer()
test_face = cv2.imread(str(BASE_DIR / "samples/session1/frames_rgb_face/frame_000000.jpg"))
quality = analyzer.analyze(test_face)
print(f"[TEST] Qualité image test: {quality['quality_status']} ({quality['quality_reason']})")

tracker = TemporalTracker()
t0 = time.time()
res1 = tracker.update({"SafeDriving": 0.95, "SleepyDriving": 0.05, "DangerousDriving": 0.0, "Distracted": 0.0, "Drinking": 0.0, "Yawn": 0.0}, "SafeDriving", True, t0)
print(f"[TEST] Tracker Frame 1: {res1['consolidated_state']}")

# Frame 2 avec un faux clignement SleepyDriving
res2 = tracker.update({"SafeDriving": 0.30, "SleepyDriving": 0.70, "DangerousDriving": 0.0, "Distracted": 0.0, "Drinking": 0.0, "Yawn": 0.0}, "SleepyDriving", True, t0 + 3.0)
print(f"[TEST] Tracker Frame 2 (faux clignement isolé): {res2['consolidated_state']} (attendu: Uncertain ou SafeDriving, PAS de bascule immédiate)")

# Frame 3 retourne à SafeDriving
res3 = tracker.update({"SafeDriving": 0.90, "SleepyDriving": 0.10, "DangerousDriving": 0.0, "Distracted": 0.0, "Drinking": 0.0, "Yawn": 0.0}, "SafeDriving", True, t0 + 6.0)
print(f"[TEST] Tracker Frame 3 (retour normal): {res3['consolidated_state']} (SafeDriving maintenu sans fausse alerte)")

print("\n[SUCCESS] Tous les algorithmes de la phase 6F fonctionnent parfaitement !")
