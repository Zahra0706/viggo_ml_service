from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging
import cv2
import numpy as np
import torch

# Configuration du logger avec anonymisation (aucune donnée biométrique dans les logs)
logger = logging.getLogger("viggo.preprocessor")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [PREPROCESSOR] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Constantes alignées sur preprocessing.py et model_architecture.py
NUM_SEGMENTS = 8
TARGET_SIZE = (224, 224)  # (largeur, hauteur)
MEAN_RGB_BGR_ORDER = (0.485, 0.456, 0.406)  # Appliqué sur BGR comme dans le notebook original
STD_RGB_BGR_ORDER = (0.229, 0.224, 0.225)


@dataclass
class VideoMetadata:
    """Métadonnées techniques extraites du conteneur vidéo."""
    duration_seconds: float
    fps: float
    width: int
    height: int
    frame_count: int
    rotation_degrees: int
    is_portrait: bool


@dataclass
class DetectionSummary:
    """Résultat de détection d'une région anatomique dans la vidéo."""
    zone_name: str
    is_available: bool
    detection_rate: float  # Proportion de frames (sur 8) où la zone est détectée
    detected_frames_count: int
    bounding_box_sample: Optional[Tuple[int, int, int, int]] = None  # (x, y, w, h) sur la 1ère frame détectée
    reason: str = ""


@dataclass
class VisibilityReport:
    """Rapport d'analyse de présence des zones anatomiques requises."""
    face: DetectionSummary
    body: DetectionSummary
    hands: DetectionSummary
    available_views: List[str] = field(default_factory=list)
    missing_views: List[str] = field(default_factory=list)
    is_dmd_compatible: bool = False
    compatibility_verdict: str = ""
    warnings: List[str] = field(default_factory=list)


@dataclass
class PreprocessedStreamResult:
    """Résultat du prétraitement d'un flux vidéo."""
    metadata: VideoMetadata
    sampled_frame_indices: List[int]
    sampled_timestamps_ms: List[float]
    tensor: torch.Tensor  # Forme (1, 8, 3, 224, 224)
    visibility: VisibilityReport
    experimental_crops: Dict[str, Optional[torch.Tensor]] = field(default_factory=dict)


class VideoPreprocessor:
    """
    Module indépendant de prétraitement vidéo pour VIGGO.
    Permet de convertir une vidéo MP4 en tenseurs prêts pour l'inférence
    tout en auditant la visibilité réelle des composantes anatomiques.
    """

    def __init__(
        self,
        num_segments: int = NUM_SEGMENTS,
        target_size: Tuple[int, int] = TARGET_SIZE,
    ):
        self.num_segments = num_segments
        self.target_size = target_size

        # Initialisation des classificateurs en cascade OpenCV
        cascade_dir = Path(cv2.data.haarcascades)
        self.face_cascade_path = cascade_dir / "haarcascade_frontalface_default.xml"
        self.profile_cascade_path = cascade_dir / "haarcascade_profileface.xml"
        self.upperbody_cascade_path = cascade_dir / "haarcascade_upperbody.xml"

        self.face_cascade = cv2.CascadeClassifier(str(self.face_cascade_path))
        self.profile_cascade = cv2.CascadeClassifier(str(self.profile_cascade_path))
        self.upperbody_cascade = cv2.CascadeClassifier(str(self.upperbody_cascade_path))

    @staticmethod
    def detect_orientation(cap: cv2.VideoCapture) -> int:
        """
        Détecte l'orientation Android stockée dans les métadonnées MP4.
        Retourne l'angle en degrés : 0, 90, 180 ou 270.
        """
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
        """Applique la rotation requise pour remettre la frame dans le sens de lecture normal."""
        if rotation_degrees == 90:
            return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif rotation_degrees == 180:
            return cv2.rotate(frame, cv2.ROTATE_180)
        elif rotation_degrees == 270:
            return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return frame

    def compute_sample_indices(self, total_frames: int) -> List[int]:
        """
        Calcule les indices des 8 frames réparties temporellement.
        Reproduction exacte de la stratégie DMD (preprocessing.py) :
        indices = [min(int((i + 0.5) * total / 8), total - 1) for i in range(8)]
        Garantit un ordre chronologique strict.
        """
        if total_frames < self.num_segments:
            raise ValueError(
                f"Vidéo trop courte : au moins {self.num_segments} frames sont nécessaires "
                f"(trouvées : {total_frames})."
            )

        indices = [
            min(int((i + 0.5) * total_frames / self.num_segments), total_frames - 1)
            for i in range(self.num_segments)
        ]

        # Vérification stricte de l'ordre chronologique
        for i in range(len(indices) - 1):
            if indices[i] > indices[i + 1]:
                raise ValueError(
                    f"Incohérence temporelle détectée dans l'échantillonnage : "
                    f"frame[{i}]={indices[i]} > frame[{i+1}]={indices[i+1]}"
                )

        return indices

    def extract_raw_frames(
        self, video_path: Path
    ) -> Tuple[List[np.ndarray], List[int], List[float], VideoMetadata]:
        """
        Ouvre la vidéo, inspecte ses propriétés, extrait les 8 frames échantillonnées
        et applique la rotation nécessaire.
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Impossible d'ouvrir le fichier vidéo : {video_path}")

        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            raw_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            raw_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            rotation = self.detect_orientation(cap)

            # Dimensions effectives après rotation
            if rotation in (90, 270):
                eff_w, eff_h = raw_h, raw_w
            else:
                eff_w, eff_h = raw_w, raw_h

            is_portrait = eff_h > eff_w
            duration_sec = round(total_frames / fps, 2) if (fps > 0 and total_frames > 0) else 0.0

            metadata = VideoMetadata(
                duration_seconds=duration_sec,
                fps=round(fps, 2) if fps > 0 else 0.0,
                width=eff_w,
                height=eff_h,
                frame_count=total_frames,
                rotation_degrees=rotation,
                is_portrait=is_portrait,
            )

            target_indices = self.compute_sample_indices(total_frames)

            # Extraction robuste par parcours séquentiel / saut d'index
            frames: List[np.ndarray] = []
            timestamps_ms: List[float] = []

            # Pour garantir la précision sur tous les codecs vidéo (H.264/H.265 avec GOP longs),
            # nous lisons jusqu'aux indices cibles
            current_frame_idx = 0
            target_set = set(target_indices)
            collected_by_idx: Dict[int, Tuple[np.ndarray, float]] = {}

            # Tentative d'accès indexé via CAP_PROP_POS_FRAMES
            can_seek = True
            for target_idx in target_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
                ret, frame = cap.read()
                pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
                # Si le décodeur ne supporte pas le seek précis, on basculera en séquentiel
                if not ret or frame is None or abs(pos - target_idx) > 2:
                    can_seek = False
                    break
                # Correction d'orientation
                if rotation != 0:
                    frame = self.apply_rotation(frame, rotation)
                ts_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                if ts_ms <= 0 and fps > 0:
                    ts_ms = (target_idx / fps) * 1000.0
                collected_by_idx[target_idx] = (frame, ts_ms)

            if not can_seek or len(collected_by_idx) < len(target_indices):
                # Fallback en lecture séquentielle déterministe
                collected_by_idx.clear()
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                current_idx = 0
                max_target = max(target_indices)

                while current_idx <= max_target:
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        break
                    if current_idx in target_set:
                        if rotation != 0:
                            frame = self.apply_rotation(frame, rotation)
                        ts_ms = (current_idx / fps * 1000.0) if fps > 0 else float(current_idx)
                        collected_by_idx[current_idx] = (frame, ts_ms)
                    current_idx += 1

            if len(collected_by_idx) < len(target_indices):
                raise ValueError(
                    f"Échec de l'extraction des {self.num_segments} frames cibles. "
                    f"Obtenues : {len(collected_by_idx)}/{len(target_indices)}."
                )

            # Assemblage dans l'ordre chronologique strict
            for idx in target_indices:
                f, t = collected_by_idx[idx]
                frames.append(f)
                timestamps_ms.append(round(t, 2))

            return frames, target_indices, timestamps_ms, metadata

        finally:
            cap.release()

    def normalize_to_tensor(self, frames: List[np.ndarray]) -> torch.Tensor:
        """
        Transforme une liste de 8 frames (BGR numpy arrays) en tenseur PyTorch
        strictement conforme au prétraitement de l'entraînement :
        - Resize vers (224, 224) en interpolation bilinéaire.
        - Conservation de l'ordre BGR.
        - Division par 255.0.
        - Centrage et réduction ImageNet (mean=0.485, 0.456, 0.406 / std=0.229, 0.224, 0.225).
        - Sortie de forme (1, 8, 3, 224, 224).
        """
        resized_frames = []
        for im in frames:
            if im is None:
                raise ValueError("Frame nulle rencontrée lors du redimensionnement.")
            r = cv2.resize(im, self.target_size, interpolation=cv2.INTER_LINEAR)
            resized_frames.append(r)

        # Transposition (T, H, W, C) -> (T, C, H, W)
        arr = np.stack(resized_frames).transpose(0, 3, 1, 2).copy()
        data = torch.from_numpy(arr).float().div_(255.0)

        mean = torch.tensor(MEAN_RGB_BGR_ORDER).view(1, 3, 1, 1)
        std = torch.tensor(STD_RGB_BGR_ORDER).view(1, 3, 1, 1)

        normalized = (data - mean) / std
        return normalized.unsqueeze(0)  # (1, 8, 3, 224, 224)

    def detect_face_in_frame(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Détecte un visage dans une frame (frontal ou profil)."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40)
        )
        if len(faces) > 0:
            # Retourne le visage avec la plus grande surface
            return tuple(sorted(faces, key=lambda b: b[2] * b[3], reverse=True)[0])

        # Essai avec le profil
        profiles = self.profile_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40)
        )
        if len(profiles) > 0:
            return tuple(sorted(profiles, key=lambda b: b[2] * b[3], reverse=True)[0])

        return None

    def detect_upperbody_in_frame(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Détecte le haut du corps / buste dans une frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bodies = self.upperbody_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=3, minSize=(60, 60)
        )
        if len(bodies) > 0:
            return tuple(sorted(bodies, key=lambda b: b[2] * b[3], reverse=True)[0])
        return None

    @staticmethod
    def detect_hands_in_frame(frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Vérifie la présence des mains sur la frame.
        Dans une caméra frontale Android de smartphone :
        - Les mains sont positionnées sur le volant dans le tiers inférieur de l'image (si visible).
        - Analyse de la région basse par segmentation de teinte de peau (YCrCb / HSV)
          et détection de composantes connexes significatives.
        - Si aucune composante compatible avec une main n'est détectée dans la zone basse,
          retourne None (zone indisponible).
        """
        h, w = frame.shape[:2]
        # Focalisation sur les 40% inférieurs de l'image (zone volant / mains)
        lower_y = int(h * 0.60)
        roi = frame[lower_y:h, 0:w]

        # Conversion YCrCb pour segmentation couleur de peau robuste
        ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)
        # Plages usuelles peau : Cr in [133, 173], Cb in [77, 127]
        skin_mask = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))

        # Nettoyage morphologique
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_hand_area = (w * h) * 0.008  # Au moins 0.8% de la frame totale

        valid_boxes = []
        for c in contours:
            area = cv2.contourArea(c)
            if area >= min_hand_area:
                bx, by, bw, bh = cv2.boundingRect(c)
                # Remettre dans les coordonnées de l'image complète
                valid_boxes.append((bx, lower_y + by, bw, bh))

        if valid_boxes:
            # Retourne la plus grande boîte
            return sorted(valid_boxes, key=lambda b: b[2] * b[3], reverse=True)[0]

        return None

    def analyze_visibility(self, frames: List[np.ndarray]) -> VisibilityReport:
        """
        Évalue la présence effective du visage, du haut du corps et des mains
        sur les 8 frames échantillonnées.
        """
        n = len(frames)
        face_hits = 0
        body_hits = 0
        hands_hits = 0

        first_face_box = None
        first_body_box = None
        first_hands_box = None

        for frame in frames:
            fb = self.detect_face_in_frame(frame)
            if fb:
                face_hits += 1
                if first_face_box is None:
                    first_face_box = fb

            bb = self.detect_upperbody_in_frame(frame)
            if bb:
                body_hits += 1
                if first_body_box is None:
                    first_body_box = bb

            hb = self.detect_hands_in_frame(frame)
            if hb:
                hands_hits += 1
                if first_hands_box is None:
                    first_hands_box = hb

        face_rate = face_hits / n
        body_rate = body_hits / n
        hands_rate = hands_hits / n

        # Heuristique : présence confirmée si détecté sur au moins 37.5% (3/8) des frames
        threshold = 0.35

        face_detected = face_rate >= threshold
        body_detected = body_rate >= threshold
        hands_detected = hands_rate >= threshold

        face_summary = DetectionSummary(
            zone_name="face",
            is_available=face_detected,
            detection_rate=round(face_rate, 2),
            detected_frames_count=face_hits,
            bounding_box_sample=first_face_box,
            reason="Visage clairement identifié" if face_detected else "Visage non détecté ou trop instable"
        )

        body_summary = DetectionSummary(
            zone_name="body",
            is_available=body_detected,
            detection_rate=round(body_rate, 2),
            detected_frames_count=body_hits,
            bounding_box_sample=first_body_box,
            reason="Buste identifié" if body_detected else "Haut du corps non résolu par le capteur"
        )

        hands_summary = DetectionSummary(
            zone_name="hands",
            is_available=hands_detected,
            detection_rate=round(hands_rate, 2),
            detected_frames_count=hands_hits,
            bounding_box_sample=first_hands_box,
            reason="Mains visibles en zone volant" if hands_detected else "Mains hors champ (champ de vision smartphone trop étroit)"
        )

        available = []
        missing = []
        for name, detected in [("face", face_detected), ("body", body_detected), ("hands", hands_detected)]:
            if detected:
                available.append(name)
            else:
                missing.append(name)

        # Vérification stricte de compatibilité avec le modèle DMD multi-vues :
        # Le modèle RGBDriverFusionNet exige 3 caméras physiques distinctes synchronisées !
        # Une seule caméra Android frontale NE PEUT PAS fournir 3 flux indépendants réels.
        is_dmd_compatible = False
        warnings = [
            "Vidéo issue d'un capteur mono-caméra frontal Android : incompatible nativement "
            "avec l'architecture multi-vues DMD (qui exige 3 flux physiques indépendants : body, face, hands)."
        ]

        if "hands" in missing:
            warnings.append(
                "Le flux 'hands' est indisponible : les mains et le volant sont hors champ "
                "de la caméra frontale du smartphone. Ne pas inventer ni dupliquer cette vue."
            )

        if not face_detected:
            warnings.append(
                "Le flux 'face' n'a pas pu être détecté de façon stable sur les 8 frames."
            )

        verdict = (
            "INCOMPATIBLE_MULTI_VIEW: La vidéo smartphone ne fournit qu'un seul point de vue partiel. "
            f"Vues détectées : {available}. Vues manquantes : {missing}. "
            "L'inférence directe sur RGBDriverFusionNet fausserait les poids du modèle."
        )

        return VisibilityReport(
            face=face_summary,
            body=body_summary,
            hands=hands_summary,
            available_views=available,
            missing_views=missing,
            is_dmd_compatible=is_dmd_compatible,
            compatibility_verdict=verdict,
            warnings=warnings
        )

    def extract_experimental_crops(
        self,
        frames: List[np.ndarray],
        visibility: VisibilityReport
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        ÉTUDE EXPÉRIMENTALE DE FAISABILITÉ :
        Tente de découper des sous-régions (crops) 'face' et 'body' à partir de l'unique
        flux vidéo de la caméra frontale si les zones correspondantes ont été détectées.

        IMPORTANT :
        - Les flux ainsi générés NE SONT PAS des caméras indépendantes.
        - Si 'hands' est absent, 'rgb_hands' reste None (aucune invention de pixels).
        - Ces crops sont fournis à des fins d'analyse comparative uniquement.
        """
        results: Dict[str, Optional[torch.Tensor]] = {
            "rgb_body": None,
            "rgb_face": None,
            "rgb_hands": None,
        }

        # 1. Région Face (si détectée)
        if visibility.face.is_available and visibility.face.bounding_box_sample is not None:
            face_crops = []
            for frame in frames:
                fb = self.detect_face_in_frame(frame) or visibility.face.bounding_box_sample
                x, y, w, h = fb
                # Marge de 20% autour du visage
                pad_x = int(w * 0.2)
                pad_y = int(h * 0.2)
                img_h, img_w = frame.shape[:2]
                x1 = max(0, x - pad_x)
                y1 = max(0, y - pad_y)
                x2 = min(img_w, x + w + pad_x)
                y2 = min(img_h, y + h + pad_y)
                crop = frame[y1:y2, x1:x2]
                face_crops.append(crop)
            results["rgb_face"] = self.normalize_to_tensor(face_crops)

        # 2. Région Body (si détectée, ou 70% supérieur si le buste est présent)
        if visibility.body.is_available and visibility.body.bounding_box_sample is not None:
            body_crops = []
            for frame in frames:
                bb = self.detect_upperbody_in_frame(frame) or visibility.body.bounding_box_sample
                x, y, w, h = bb
                img_h, img_w = frame.shape[:2]
                crop = frame[max(0, y):min(img_h, y + h), max(0, x):min(img_w, x + w)]
                body_crops.append(crop)
            results["rgb_body"] = self.normalize_to_tensor(body_crops)

        # 3. Région Hands : STRICTEMENT NONE si indisponible
        # Pas d'invention de pixels ni de duplication !
        if visibility.hands.is_available and visibility.hands.bounding_box_sample is not None:
            hands_crops = []
            for frame in frames:
                hb = self.detect_hands_in_frame(frame) or visibility.hands.bounding_box_sample
                x, y, w, h = hb
                img_h, img_w = frame.shape[:2]
                crop = frame[max(0, y):min(img_h, y + h), max(0, x):min(img_w, x + w)]
                hands_crops.append(crop)
            results["rgb_hands"] = self.normalize_to_tensor(hands_crops)
        else:
            results["rgb_hands"] = None

        return results

    def process(self, video_path: Path | str) -> PreprocessedStreamResult:
        """
        Point d'entrée principal du module de prétraitement :
        1. Lit la vidéo MP4 et en extrait les métadonnées.
        2. Échantillonne 8 frames ordonnées chronologiquement.
        3. Applique la correction de rotation Android.
        4. Construit le tenseur global normalisé (1, 8, 3, 224, 224).
        5. Analyse la visibilité anatomique (visage, buste, mains).
        6. Génère l'évaluation de compatibilité et l'extraction expérimentale éventuelle.
        """
        path = Path(video_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Fichier introuvable : {path}")

        # 1-3. Extraction des 8 frames corrigées
        frames, indices, timestamps, metadata = self.extract_raw_frames(path)

        # 4. Normalisation conforme à preprocessing.py
        tensor = self.normalize_to_tensor(frames)

        # 5. Détection de visibilité et conformité
        visibility = self.analyze_visibility(frames)

        # 6. Étude expérimentale des crops
        crops = self.extract_experimental_crops(frames, visibility)

        logger.info(
            f"Traitement termine: {path.name} | "
            f"Dims: {metadata.width}x{metadata.height} | "
            f"FPS: {metadata.fps} | "
            f"Vues dispo: {visibility.available_views} | "
            f"Vues manquantes: {visibility.missing_views}"
        )

        return PreprocessedStreamResult(
            metadata=metadata,
            sampled_frame_indices=indices,
            sampled_timestamps_ms=timestamps,
            tensor=tensor,
            visibility=visibility,
            experimental_crops=crops
        )
