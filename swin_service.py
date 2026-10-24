import json
import logging
import time
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


class SwinInferenceService:
    """Service singleton thread-safe pour l'inférence Swin Transformer ONNX."""

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

        # Vérification des spécifications d'entrée et sortie
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        # Cascade OpenCV pour la détection de présence conducteur
        cascade_dir = Path(cv2.data.haarcascades)
        self.face_cascade = cv2.CascadeClassifier(
            str(cascade_dir / "haarcascade_frontalface_default.xml")
        )
        self.profile_cascade = cv2.CascadeClassifier(
            str(cascade_dir / "haarcascade_profileface.xml")
        )

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
        """Vérifie sommairement la présence effective d'un visage / conducteur."""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=3, minSize=(35, 35)
        )
        if len(faces) > 0:
            return True
        profiles = self.profile_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=3, minSize=(35, 35)
        )
        return len(profiles) > 0

    def preprocess_frame(self, frame_bgr: np.ndarray, rotation_degrees: int = 0) -> np.ndarray:
        """
        Prétraitement rigoureusement conforme à preprocessing.json :
        1. Correction de la rotation Android si applicable.
        2. BGR OpenCV vers RGB.
        3. Redimensionnement direct 224x224 (cv2.INTER_LINEAR) sans crop ni letterboxing.
        4. Normalisation ImageNet RGB : (pixel / 255.0 - mean) / std.
        5. Format NCHW float32.
        """
        if rotation_degrees != 0:
            frame_bgr = self.apply_rotation(frame_bgr, rotation_degrees)

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_LINEAR)

        arr = resized.astype(np.float32) / 255.0
        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD

        # Transposition HWC -> CHW, puis ajout dimension batch [1, 3, 224, 224]
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
        Analyse une vidéo MP4 par échantillonnage uniforme de frames :
        1. Lit la vidéo, extrait les métadonnées et la rotation.
        2. Sélectionne `num_frames` frames réparties uniformément.
        3. Prétraite et classifie chaque frame individuellement avec Swin ONNX.
        4. Calcule un résumé agrégé avec métriques de confiance et avertissements d'incertitude.
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

            # Calcul des indices temporels équidistants
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

                # 1. Prétraitement
                t0_p = time.perf_counter()
                tensor = self.preprocess_frame(frame, rotation_degrees=rotation)
                has_driver = self.check_driver_presence(
                    self.apply_rotation(frame, rotation) if rotation != 0 else frame
                )
                if has_driver:
                    driver_detected_count += 1
                preproc_total_ms += (time.perf_counter() - t0_p) * 1000

                # 2. Inférence ONNX
                t0_i = time.perf_counter()
                logits, scores, class_idx = self.predict_single_tensor(tensor)
                infer_total_ms += (time.perf_counter() - t0_i) * 1000

                all_scores_matrix.append(scores)

                frame_results.append({
                    "frame_index": idx,
                    "timestamp_ms": round(float(ts_ms), 2),
                    "predicted_class": self.labels[class_idx],
                    "predicted_class_index": class_idx,
                    "driver_detected": has_driver,
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

            # 3. Agrégation statistique sur les frames
            scores_arr = np.array(all_scores_matrix)  # (N, 6)
            mean_scores = scores_arr.mean(axis=0)  # (6,)
            dominant_class_idx = int(np.argmax(mean_scores))
            dominant_class = self.labels[dominant_class_idx]

            # Vote majoritaire des frames
            votes = [r["predicted_class"] for r in frame_results]
            vote_counts = {label: votes.count(label) for label in self.labels if votes.count(label) > 0}
            agreement_ratio = round(vote_counts.get(dominant_class, 0) / len(frame_results), 2)

            # Entropie de Shannon comme indicateur d'incertitude
            eps = 1e-12
            entropy = -float(np.sum(mean_scores * np.log(mean_scores + eps)))

            # Alertes et mises en garde
            warnings: List[str] = [
                "Analyse expérimentale Swin Transformer — non validée pour les alertes de sécurité.",
                "Les scores Softmax représentent une confiance relative du modèle et non des probabilités cliniques ou certifiées."
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
        Analyse une image fixe unique (JPEG / PNG / WEBP) capturée pendant un trajet actif :
        1. Décode les octets bruts en BGR via cv2.imdecode sans écriture disque.
        2. Détecte la présence du conducteur via Haar cascade.
        3. Prétraite conformément à preprocessing.json (RGB, resize 224x224, ImageNet norm, NCHW float32).
        4. Exécute l'inférence Swin Transformer ONNX avec thread-safety.
        5. Calcule les scores softmax des 6 classes, l'entropie et les alertes de cadrage.
        """
        if not image_bytes:
            raise ValueError("L'image fournie est vide (0 octet).")

        t_start = time.perf_counter()

        # 1. Décodage mémoire sans écriture disque
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame_bgr is None or frame_bgr.size == 0:
            raise ValueError("Impossible de décoder les données de l'image (format corrompu ou non supporté).")

        h, w = frame_bgr.shape[:2]

        # 2. Détection de présence conducteur et prétraitement
        t0_p = time.perf_counter()
        has_driver = self.check_driver_presence(frame_bgr)
        tensor = self.preprocess_frame(frame_bgr, rotation_degrees=rotation_degrees)
        preproc_ms = (time.perf_counter() - t0_p) * 1000

        # 3. Inférence ONNX
        t0_i = time.perf_counter()
        logits, scores, class_idx = self.predict_single_tensor(tensor)
        infer_ms = (time.perf_counter() - t0_i) * 1000

        total_elapsed_ms = (time.perf_counter() - t_start) * 1000

        # 4. Métriques et avertissements de cadrage
        eps = 1e-9
        entropy = float(-np.sum(scores * np.log(scores + eps)))
        confidence = float(scores[class_idx])
        predicted_label = self.labels[class_idx]

        warnings: List[str] = []
        is_reliable = True

        if not has_driver:
            warnings.append(
                "Visage du conducteur non détecté : cadrage potentiellement inadapté ou obscurité."
            )
            is_reliable = False

        if confidence < 0.35 or entropy > 1.6:
            warnings.append(
                "Niveau de confiance faible ou forte ambiguïté sur la prédiction."
            )
            is_reliable = False

        scores_dict = {
            label: round(float(scores[i]), 5)
            for i, label in enumerate(self.labels)
        }

        return {
            "model": "driver_swin.onnx",
            "framework": "ONNX Runtime (CPU)",
            "prediction": {
                "class": predicted_label,
                "class_index": class_idx,
                "confidence": round(confidence, 5),
                "is_reliable": is_reliable,
            },
            "scores": scores_dict,
            "driver_detected": has_driver,
            "framing_status": "good" if has_driver else "uncertain",
            "image_metadata": {
                "width": w,
                "height": h,
                "size_bytes": len(image_bytes),
                "channels": 3,
            },
            "uncertainty_metrics": {
                "shannon_entropy": round(entropy, 4),
            },
            "performance": {
                "preprocessing_time_ms": round(preproc_ms, 2),
                "inference_time_ms": round(infer_ms, 2),
                "total_time_ms": round(total_elapsed_ms, 2),
            },
            "warnings": warnings,
            "disclaimer": "Analyse expérimentale — non validée pour les alertes de sécurité routière",
        }

