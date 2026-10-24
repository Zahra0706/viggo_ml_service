import json
import logging
import time
from collections import deque
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np
import onnxruntime as ort

logger = logging.getLogger("viggo.swin_service")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [SWIN] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

BASE_DIR = Path(__file__).resolve().parent
SWIN_DIR = BASE_DIR / "models" / "swin"
MODEL_PATH = SWIN_DIR / "driver_swin.onnx"
LABELS_PATH = SWIN_DIR / "labels.json"
PREPROCESSING_PATH = SWIN_DIR / "preprocessing.json"

# Constantes ImageNet RGB conformes à preprocessing.json
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

EXPECTED_CLASSES = [
    "DangerousDriving",
    "Distracted",
    "Drinking",
    "SafeDriving",
    "SleepyDriving",
    "Yawn",
]


class ImageQualityAnalyzer:
    """Analyseur de qualité d'image en amont de l'inférence Swin Transformer."""

    def __init__(self, face_cascade, face_alt_cascade, profile_cascade, eye_cascade):
        self.face_cascade = face_cascade
        self.face_alt_cascade = face_alt_cascade
        self.profile_cascade = profile_cascade
        self.eye_cascade = eye_cascade

    def analyze(self, frame_bgr: np.ndarray) -> Dict[str, Any]:
        h, w = frame_bgr.shape[:2]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        # 1. Détection de netteté / flou par variance du Laplacien
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        is_blurry = bool(lap_var < 45.0)

        # 2. Détection de luminosité (moyenne et extrêmes)
        mean_brightness = float(np.mean(gray))
        is_too_dark = bool(mean_brightness < 35.0)
        is_too_bright = bool(mean_brightness > 225.0)

        # 3. Détection faciale multi-cascades robuste
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=3, minSize=(45, 45)
        )
        if len(faces) == 0:
            faces = self.face_alt_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=2, minSize=(40, 40)
            )
        if len(faces) == 0:
            faces = self.profile_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=2, minSize=(40, 40)
            )

        face_detected = bool(len(faces) > 0)
        face_box = None
        face_area_ratio = 0.0
        face_too_small = False

        if face_detected:
            # Sélection du plus grand visage détecté
            faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
            fx, fy, fw, fh = faces[0]
            face_box = (int(fx), int(fy), int(fw), int(fh))
            face_area_ratio = float((fw * fh) / (w * h))
            if face_area_ratio < 0.025:  # Moins de 2.5% de l'image
                face_too_small = True
        else:
            # Vérification de repli par détection oculaire
            eyes = self.eye_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=2, minSize=(20, 20)
            )
            if len(eyes) >= 2:
                face_detected = True

        # 4. Diagnostic d'exploitabilité
        quality_status = "valid"
        reasons: List[str] = []

        if not face_detected:
            quality_status = "analysis_unavailable"
            reasons.append("Conducteur non détecté : cadrage inadapté ou absence du conducteur")
        elif face_too_small:
            quality_status = "unreliable"
            reasons.append("Conducteur trop éloigné de la caméra")

        if is_blurry:
            if quality_status != "analysis_unavailable":
                quality_status = "unreliable"
            reasons.append("Flou de mouvement détecté")
        if is_too_dark:
            if quality_status != "analysis_unavailable":
                quality_status = "unreliable"
            reasons.append("Image trop sombre (éclairage habitacle insuffisant)")
        if is_too_bright:
            if quality_status != "analysis_unavailable":
                quality_status = "unreliable"
            reasons.append("Surexposition ou éblouissement lumineux")

        is_reliable = bool(quality_status == "valid")
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
    """Analyse faciale géométrique complémentaire (validation croisée Swin)."""

    def __init__(self, eye_cascade):
        self.eye_cascade = eye_cascade

    def cross_validate(
        self,
        face_crop: np.ndarray,
        predicted_class: str,
        scores: Dict[str, float],
    ) -> Tuple[str, Dict[str, float], Optional[str]]:
        fh, fw = face_crop.shape[:2]
        if fh < 35 or fw < 35:
            return predicted_class, scores, None

        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)

        # 1. Analyse oculaire (20% à 55% de la hauteur du visage)
        eye_roi = gray[int(0.20 * fh):int(0.55 * fh), :]
        eyes = self.eye_cascade.detectMultiScale(
            eye_roi, scaleFactor=1.1, minNeighbors=2, minSize=(16, 16)
        )
        two_eyes_open = bool(len(eyes) >= 2)

        # 2. Analyse buccale (60% à 95% de la hauteur, 20% à 80% largeur)
        mouth_roi = gray[int(0.60 * fh):int(0.95 * fh), int(0.20 * fw):int(0.80 * fw)]
        mouth_open_ratio = 0.0
        if mouth_roi.size > 0:
            # Détection de la cavité buccale sombre ouverte
            _, mouth_thresh = cv2.threshold(mouth_roi, 50, 255, cv2.THRESH_BINARY_INV)
            mouth_open_ratio = float(np.count_nonzero(mouth_thresh) / mouth_roi.size)

        is_mouth_open = bool(mouth_open_ratio > 0.20)
        override_note = None
        new_scores = dict(scores)

        # Règle 1 : Swin prédit Yawn mais la bouche est fermée (faux positif dû aux mains sur le volant)
        if predicted_class == "Yawn" and not is_mouth_open:
            override_note = (
                f"Contradiction visuelle : bouche fermée (ratio cavité={mouth_open_ratio:.2f}) "
                f"alors que Swin prédit Yawn -> probabilité Yawn pondérée"
            )
            new_scores["Yawn"] *= 0.15
            total = sum(new_scores.values())
            new_scores = {k: round(v / total, 5) for k, v in new_scores.items()}
            predicted_class = max(new_scores, key=new_scores.get)

        # Règle 2 : Swin prédit SleepyDriving mais 2 yeux grands ouverts sont détectés
        elif predicted_class == "SleepyDriving" and two_eyes_open:
            override_note = (
                "Contradiction visuelle : 2 yeux ouverts détectés alors que Swin prédit SleepyDriving "
                "-> probabilité SleepyDriving pondérée"
            )
            new_scores["SleepyDriving"] *= 0.20
            total = sum(new_scores.values())
            new_scores = {k: round(v / total, 5) for k, v in new_scores.items()}
            predicted_class = max(new_scores, key=new_scores.get)

        return predicted_class, new_scores, override_note


class TemporalTracker:
    """Filtre temporel à fenêtre glissante et lissage exponentiel (EMA)."""

    def __init__(self, window_size: int = 5, alpha: float = 0.40):
        self.window_size = window_size
        self.alpha = alpha
        self.history = deque(maxlen=window_size)
        self.ema_scores: Optional[Dict[str, float]] = None
        self._lock = Lock()

    def update(
        self,
        raw_scores: Dict[str, float],
        raw_predicted_class: str,
        is_reliable: bool,
        timestamp: float,
    ) -> Dict[str, Any]:
        with self._lock:
            self.history.append({
                "timestamp": timestamp,
                "scores": raw_scores,
                "class": raw_predicted_class,
                "is_reliable": is_reliable,
            })

            # Initialisation ou mise à jour de l'EMA
            if self.ema_scores is None:
                self.ema_scores = dict(raw_scores)
            else:
                # Si la frame est non-fiable (flou, mauvaise luminosité), on atténue fortement son impact
                effective_alpha = self.alpha if is_reliable else (self.alpha * 0.15)
                for k in raw_scores:
                    self.ema_scores[k] = (
                        effective_alpha * raw_scores[k]
                        + (1.0 - effective_alpha) * self.ema_scores[k]
                    )

            # Normalisation
            total = sum(self.ema_scores.values())
            if total > 0:
                for k in self.ema_scores:
                    self.ema_scores[k] = round(self.ema_scores[k] / total, 5)

            smoothed_top_class = max(self.ema_scores, key=self.ema_scores.get)
            smoothed_top_conf = float(self.ema_scores[smoothed_top_class])

            # Analyse de consensus sur les frames fiables récentes
            reliable_history = [h for h in self.history if h["is_reliable"]]
            recent_classes = [h["class"] for h in reliable_history]

            # 1. Détection d'oscillation (forte instabilité sans consensus)
            if len(recent_classes) >= 3 and len(set(recent_classes[-3:])) == 3:
                consolidated_state = "Uncertain"
            # 2. Persistance obligatoire pour les alertes critiques (SleepyDriving / DangerousDriving)
            elif smoothed_top_class in ("SleepyDriving", "DangerousDriving"):
                consecutive_count = sum(
                    1 for c in recent_classes[-2:] if c == smoothed_top_class
                )
                if consecutive_count >= 2 and smoothed_top_conf >= 0.45:
                    consolidated_state = smoothed_top_class
                else:
                    consolidated_state = "Uncertain"
            else:
                consolidated_state = smoothed_top_class

            return {
                "consolidated_state": consolidated_state,
                "smoothed_confidence": smoothed_top_conf,
                "smoothed_scores": dict(self.ema_scores),
                "history_frames_count": len(self.history),
            }

    def reset(self):
        with self._lock:
            self.history.clear()
            self.ema_scores = None


class SwinInferenceService:
    """Service singleton thread-safe pour l'inférence Swin Transformer ONNX avec pipeline fiabilisé."""

    _instance = None
    _init_lock = Lock()

    def __init__(self, model_path: Path = MODEL_PATH, labels_path: Path = LABELS_PATH):
        self.model_path = Path(model_path).resolve()
        self.labels_path = Path(labels_path).resolve()

        if not self.model_path.exists():
            raise FileNotFoundError(f"Modèle ONNX introuvable : {self.model_path}")

        if not self.labels_path.exists():
            raise FileNotFoundError(f"Fichier de labels introuvable : {self.labels_path}")

        # Chargement des labels et vérification stricte de l'ordre
        self.labels = json.loads(self.labels_path.read_text(encoding="utf-8"))
        if self.labels != EXPECTED_CLASSES:
            logger.warning(
                f"Ordre des classes différent de l'attendu : {self.labels} vs {EXPECTED_CLASSES}"
            )

        # Configuration de la session ONNX Runtime CPU
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        logger.info(f"Chargement du modèle Swin ONNX depuis {self.model_path.name}...")
        t0 = time.perf_counter()
        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        t_load = (time.perf_counter() - t0) * 1000
        logger.info(f"Modèle Swin ONNX chargé avec succès en {t_load:.1f} ms.")

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        # Cascades OpenCV
        cascade_dir = Path(cv2.data.haarcascades)
        self.face_cascade = cv2.CascadeClassifier(
            str(cascade_dir / "haarcascade_frontalface_default.xml")
        )
        self.face_alt_cascade = cv2.CascadeClassifier(
            str(cascade_dir / "haarcascade_frontalface_alt2.xml")
        )
        self.profile_cascade = cv2.CascadeClassifier(
            str(cascade_dir / "haarcascade_profileface.xml")
        )
        self.eye_cascade = cv2.CascadeClassifier(
            str(cascade_dir / "haarcascade_eye.xml")
        )

        # Modules d'amélioration Phase 6F
        self.quality_analyzer = ImageQualityAnalyzer(
            self.face_cascade,
            self.face_alt_cascade,
            self.profile_cascade,
            self.eye_cascade,
        )
        self.facial_analyzer = ComplementaryFacialAnalyzer(self.eye_cascade)
        self.temporal_tracker = TemporalTracker(window_size=5, alpha=0.40)

        self._infer_lock = Lock()

    @classmethod
    def get_instance(cls) -> "SwinInferenceService":
        if cls._instance is None:
            with cls._init_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @staticmethod
    def detect_orientation(cap: cv2.VideoCapture) -> int:
        """Détecte l'orientation Android stockée dans les métadonnées MP4 (0, 90, 180, 270)."""
        rotation = 0
        try:
            if hasattr(cv2, "CAP_PROP_ORIENTATION_META"):
                meta_rot = cap.get(cv2.CAP_PROP_ORIENTATION_META)
                if meta_rot in (90, 180, 270):
                    rotation = int(meta_rot)
        except Exception as e:
            logger.debug(f"Impossible de lire CAP_PROP_ORIENTATION_META: {e}")
        return rotation

    @staticmethod
    def apply_rotation(frame: np.ndarray, rotation_degrees: int) -> np.ndarray:
        """Applique la rotation pour redresser la frame dans le sens normal de lecture."""
        if rotation_degrees == 90:
            return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif rotation_degrees == 180:
            return cv2.rotate(frame, cv2.ROTATE_180)
        elif rotation_degrees == 270:
            return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return frame

    def check_driver_presence(self, frame_bgr: np.ndarray) -> bool:
        """Vérifie la présence effective d'un visage / conducteur via l'analyseur qualité."""
        quality = self.quality_analyzer.analyze(frame_bgr)
        return quality["face_detected"]

    def extract_face_roi(
        self,
        frame_bgr: np.ndarray,
        face_box: Optional[Tuple[int, int, int, int]]
    ) -> np.ndarray:
        """
        Extrait la zone de la tête avec marge calibrée (1.6x largeur, 1.8x hauteur),
        reproduisant le cadrage DMD face d'entraînement où Swin atteint sa précision maximale.
        """
        if face_box is None:
            return frame_bgr

        h, w = frame_bgr.shape[:2]
        fx, fy, fw, fh = face_box

        # Marge adaptée pour inclure la tête, les yeux et la bouche
        cx = fx + fw // 2
        cy = fy + fh // 2
        crop_w = int(fw * 1.6)
        crop_h = int(fh * 1.8)

        x1 = max(0, cx - crop_w // 2)
        y1 = max(0, cy - crop_h // 2)
        x2 = min(w, x1 + crop_w)
        y2 = min(h, y1 + crop_h)

        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0 or crop.shape[0] < 20 or crop.shape[1] < 20:
            return frame_bgr
        return crop

    def preprocess_frame(
        self,
        frame_bgr: np.ndarray,
        rotation_degrees: int = 0,
        face_box: Optional[Tuple[int, int, int, int]] = None,
        use_face_crop: bool = True,
    ) -> np.ndarray:
        """
        Prétraitement rigoureusement conforme à preprocessing.json :
        1. Correction de la rotation Android si nécessaire.
        2. Recadrage centré tête/visage si détecté (alignement DMD), sinon pleine image.
        3. Conversion BGR vers RGB.
        4. Redimensionnement direct 224x224 (cv2.INTER_LINEAR).
        5. Normalisation ImageNet RGB : (pixel / 255.0 - mean) / std.
        6. Format NCHW float32.
        """
        if rotation_degrees != 0:
            frame_bgr = self.apply_rotation(frame_bgr, rotation_degrees)

        img_to_process = frame_bgr
        if use_face_crop and face_box is not None:
            img_to_process = self.extract_face_roi(frame_bgr, face_box)

        rgb = cv2.cvtColor(img_to_process, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_LINEAR)

        arr = resized.astype(np.float32) / 255.0
        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD

        tensor = np.ascontiguousarray(arr.transpose(2, 0, 1)[None, :])
        return tensor

    def predict_single_tensor(self, tensor_nchw: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Exécute l'inférence ONNX sur un tenseur [1, 3, 224, 224].
        Retourne (logits [6], scores_softmax [6], classe_index).
        """
        with self._infer_lock:
            logits = self.session.run([self.output_name], {self.input_name: tensor_nchw})[0][0]

        # Softmax numériquement stable
        exp_logits = np.exp(logits - np.max(logits))
        scores = exp_logits / np.sum(exp_logits)
        class_idx = int(np.argmax(scores))
        return logits, scores, class_idx

    def predict_video(
        self,
        video_path: Path | str,
        num_frames: int = 5,
    ) -> Dict[str, Any]:
        """
        Analyse une vidéo MP4 par échantillonnage uniforme de frames avec Swin ONNX.
        """
        path = Path(video_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Fichier vidéo introuvable : {path}")

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise ValueError(f"Impossible d'ouvrir le conteneur vidéo : {path}")

        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            raw_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            raw_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            rotation = self.detect_orientation(cap)

            if rotation in (90, 270):
                eff_w, eff_h = raw_h, raw_w
            else:
                eff_w, eff_h = raw_w, raw_h

            duration_sec = round(total_frames / fps, 2) if (fps > 0 and total_frames > 0) else 0.0

            if total_frames < num_frames:
                raise ValueError(
                    f"Vidéo trop courte : au moins {num_frames} frames requises (trouvées : {total_frames})."
                )

            target_indices = [
                min(int((i + 0.5) * total_frames / num_frames), total_frames - 1)
                for i in range(num_frames)
            ]

            t_start_total = time.perf_counter()
            preproc_total_ms = 0.0
            infer_total_ms = 0.0

            frame_results: List[Dict[str, Any]] = []
            all_scores_matrix = []
            driver_detected_count = 0

            for idx in target_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue

                ts_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                if ts_ms <= 0 and fps > 0:
                    ts_ms = (idx / fps) * 1000.0

                t0_p = time.perf_counter()
                quality = self.quality_analyzer.analyze(
                    self.apply_rotation(frame, rotation) if rotation != 0 else frame
                )
                if quality["face_detected"]:
                    driver_detected_count += 1

                tensor = self.preprocess_frame(
                    frame,
                    rotation_degrees=rotation,
                    face_box=quality["face_box"],
                    use_face_crop=quality["is_reliable"],
                )
                preproc_total_ms += (time.perf_counter() - t0_p) * 1000

                t0_i = time.perf_counter()
                logits, scores, class_idx = self.predict_single_tensor(tensor)
                infer_total_ms += (time.perf_counter() - t0_i) * 1000

                all_scores_matrix.append(scores)

                frame_results.append({
                    "frame_index": idx,
                    "timestamp_ms": round(float(ts_ms), 2),
                    "predicted_class": self.labels[class_idx],
                    "predicted_class_index": class_idx,
                    "driver_detected": quality["face_detected"],
                    "scores": {
                        label: round(float(scores[c_idx]), 5)
                        for c_idx, label in enumerate(self.labels)
                    },
                    "logits": {
                        label: round(float(logits[c_idx]), 5)
                        for c_idx, label in enumerate(self.labels)
                    },
                })

            total_elapsed_ms = (time.perf_counter() - t_start_total) * 1000

            if not frame_results:
                raise ValueError("Aucune frame n'a pu être décodée pour l'analyse Swin.")

            scores_arr = np.array(all_scores_matrix)
            mean_scores = scores_arr.mean(axis=0)
            dominant_class_idx = int(np.argmax(mean_scores))
            dominant_class = self.labels[dominant_class_idx]

            votes = [r["predicted_class"] for r in frame_results]
            vote_counts = {label: votes.count(label) for label in self.labels if votes.count(label) > 0}
            agreement_ratio = round(vote_counts.get(dominant_class, 0) / len(frame_results), 2)

            eps = 1e-12
            entropy = -float(np.sum(mean_scores * np.log(mean_scores + eps)))

            warnings: List[str] = [
                "Analyse expérimentale Swin Transformer — non validée pour les alertes de sécurité.",
                "Les scores Softmax représentent une confiance relative du modèle et non des probabilités certifiées."
            ]
            if driver_detected_count < len(frame_results) * 0.5:
                warnings.append(
                    "Présence conducteur faible ou incertaine : la caméra semble mal cadrée ou le visage est masqué."
                )

            return {
                "model": "driver_swin.onnx",
                "framework": "ONNX Runtime (CPU)",
                "analyzed_frames_count": len(frame_results),
                "dominant_prediction": {
                    "class": dominant_class,
                    "class_index": dominant_class_idx,
                    "mean_score": round(float(mean_scores[dominant_class_idx]), 5),
                    "agreement_ratio": agreement_ratio,
                    "frame_votes": vote_counts,
                },
                "aggregated_scores": {
                    label: round(float(mean_scores[i]), 5)
                    for i, label in enumerate(self.labels)
                },
                "uncertainty_metrics": {
                    "shannon_entropy": round(entropy, 4),
                    "driver_detected_ratio": round(driver_detected_count / len(frame_results), 2),
                },
                "video_metadata": {
                    "filename": path.name,
                    "duration_seconds": duration_sec,
                    "fps": round(fps, 2),
                    "resolution": f"{eff_w}x{eff_h}",
                    "rotation_degrees": rotation,
                    "total_frames": total_frames,
                },
                "performance": {
                    "preprocessing_time_ms": round(preproc_total_ms, 2),
                    "inference_time_ms": round(infer_total_ms, 2),
                    "total_time_ms": round(total_elapsed_ms, 2),
                    "avg_inference_per_frame_ms": round(infer_total_ms / len(frame_results), 2),
                },
                "frame_details": frame_results,
                "warnings": warnings,
            }

        finally:
            cap.release()

    def predict_image(
        self,
        image_bytes: bytes,
        rotation_degrees: int = 0,
    ) -> Dict[str, Any]:
        """
        Analyse une frame image avec pipeline complet Phase 6F :
        1. Décodage mémoire sans écriture disque.
        2. Contrôle qualité d'image (flou, luminosité, détection visage/cadrage).
        3. Recadrage visage centré avec marge calibrée.
        4. Inférence Swin Transformer ONNX.
        5. Analyse faciale complémentaire (cross-validation EAR/MAR).
        6. Stabilisation temporelle EMA et logique de consensus anti-oscillation.
        """
        if not image_bytes:
            raise ValueError("L'image fournie est vide (0 octet).")

        t_start = time.perf_counter()

        # 1. Décodage mémoire
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame_bgr is None or frame_bgr.size == 0:
            raise ValueError("Impossible de décoder les données de l'image (format corrompu ou non supporté).")

        h, w = frame_bgr.shape[:2]

        # 2. Contrôle qualité
        t0_p = time.perf_counter()
        rotated_frame = (
            self.apply_rotation(frame_bgr, rotation_degrees)
            if rotation_degrees != 0
            else frame_bgr
        )
        quality = self.quality_analyzer.analyze(rotated_frame)

        # 3. Prétraitement avec cadrage visage si disponible
        tensor = self.preprocess_frame(
            rotated_frame,
            rotation_degrees=0,
            face_box=quality["face_box"],
            use_face_crop=quality["is_reliable"],
        )
        preproc_ms = (time.perf_counter() - t0_p) * 1000

        # 4. Inférence ONNX
        t0_i = time.perf_counter()
        logits, scores, class_idx = self.predict_single_tensor(tensor)
        infer_ms = (time.perf_counter() - t0_i) * 1000

        scores_dict = {
            label: round(float(scores[i]), 5)
            for i, label in enumerate(self.labels)
        }
        raw_predicted_label = self.labels[class_idx]

        # 5. Validation faciale complémentaire (EAR/MAR)
        face_roi = self.extract_face_roi(rotated_frame, quality["face_box"])
        validated_label, validated_scores, contradiction_note = (
            self.facial_analyzer.cross_validate(
                face_roi, raw_predicted_label, scores_dict
            )
        )

        # 6. Stabilisation temporelle EMA
        now_ts = time.time()
        temporal_res = self.temporal_tracker.update(
            raw_scores=validated_scores,
            raw_predicted_class=validated_label,
            is_reliable=quality["is_reliable"],
            timestamp=now_ts,
        )

        total_elapsed_ms = (time.perf_counter() - t_start) * 1000

        # Métriques d'incertitude
        eps = 1e-9
        raw_entropy = float(-np.sum(scores * np.log(scores + eps)))
        confidence = float(validated_scores[validated_label])

        warnings: List[str] = []
        if not quality["is_reliable"]:
            warnings.append(f"Image dégradée : {quality['quality_reason']}.")
        if contradiction_note:
            warnings.append(contradiction_note)
        if temporal_res["consolidated_state"] == "Uncertain":
            warnings.append("Comportement incertain : les prédictions récentes oscillent sans consensus.")

        return {
            "model": "driver_swin.onnx",
            "framework": "ONNX Runtime (CPU)",
            "prediction": {
                "class": validated_label,
                "class_index": self.labels.index(validated_label),
                "confidence": round(confidence, 5),
                "is_reliable": quality["is_reliable"],
                "quality_status": quality["quality_status"],
                "quality_reason": quality["quality_reason"],
                "consolidated_state": temporal_res["consolidated_state"],
                "smoothed_confidence": temporal_res["smoothed_confidence"],
            },
            "scores": validated_scores,
            "smoothed_scores": temporal_res["smoothed_scores"],
            "driver_detected": quality["face_detected"],
            "driver_confidence": 1.0 if quality["face_detected"] else 0.0,
            "framing_status": quality["framing_status"],
            "quality_metrics": {
                "laplacian_variance": quality["laplacian_variance"],
                "mean_brightness": quality["mean_brightness"],
                "face_area_ratio": quality["face_area_ratio"],
                "is_blurry": quality["laplacian_variance"] < 45.0,
                "quality_status": quality["quality_status"],
            },
            "cross_validation": {
                "contradiction_detected": contradiction_note is not None,
                "note": contradiction_note,
            },
            "image_metadata": {
                "width": w,
                "height": h,
                "size_bytes": len(image_bytes),
                "channels": 3,
            },
            "uncertainty_metrics": {
                "shannon_entropy": round(raw_entropy, 4),
            },
            "performance": {
                "preprocessing_time_ms": round(preproc_ms, 2),
                "inference_time_ms": round(infer_ms, 2),
                "total_time_ms": round(total_elapsed_ms, 2),
            },
            "warnings": warnings,
            "disclaimer": "Analyse expérimentale — non validée pour les alertes de sécurité routière",
        }
