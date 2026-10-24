import os
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch

from video_preprocessor import (
    VideoPreprocessor,
    VideoMetadata,
    VisibilityReport,
    PreprocessedStreamResult,
    NUM_SEGMENTS,
    TARGET_SIZE,
    MEAN_RGB_BGR_ORDER,
    STD_RGB_BGR_ORDER,
)


class TestVideoPreprocessor(unittest.TestCase):
    """Tests unitaires du préprocesseur vidéo VIGGO."""

    @classmethod
    def setUpClass(cls):
        cls.preprocessor = VideoPreprocessor()
        cls.test_dir = Path(tempfile.mkdtemp(prefix="viggo_test_preproc_"))

    @classmethod
    def tearDownClass(cls):
        if cls.test_dir.exists():
            shutil.rmtree(cls.test_dir, ignore_errors=True)

    def _create_synthetic_video(
        self,
        filename: str,
        width: int,
        height: int,
        fps: float,
        duration_sec: float,
        draw_face: bool = False,
        draw_body: bool = False,
        draw_hands: bool = False,
    ) -> Path:
        """Génère une vidéo MP4 synthétique pour les tests et retourne son chemin."""
        video_path = self.test_dir / filename
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        total_frames = max(1, int(fps * duration_sec))
        out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))

        if not out.isOpened():
            raise RuntimeError(f"Impossible d'initialiser cv2.VideoWriter pour {video_path}")

        try:
            for idx in range(total_frames):
                frame = np.full((height, width, 3), 40, dtype=np.uint8)  # Fond gris sombre

                # Dessin géométrique simulant un buste
                if draw_body:
                    body_center_x = width // 2
                    body_y = int(height * 0.6)
                    cv2.ellipse(
                        frame,
                        (body_center_x, body_y),
                        (int(width * 0.3), int(height * 0.35)),
                        0,
                        0,
                        360,
                        (80, 70, 60),
                        -1,
                    )

                # Dessin géométrique simulant un visage avec yeux et bouche
                if draw_face:
                    face_center_x = width // 2
                    face_center_y = int(height * 0.35)
                    face_radius = min(width, height) // 6
                    # Couleur peau BGR ~ (150, 180, 210)
                    cv2.circle(
                        frame,
                        (face_center_x, face_center_y),
                        face_radius,
                        (150, 180, 210),
                        -1,
                    )
                    # Yeux
                    eye_offset = face_radius // 3
                    cv2.circle(
                        frame,
                        (face_center_x - eye_offset, face_center_y - eye_offset // 2),
                        max(2, face_radius // 8),
                        (30, 30, 30),
                        -1,
                    )
                    cv2.circle(
                        frame,
                        (face_center_x + eye_offset, face_center_y - eye_offset // 2),
                        max(2, face_radius // 8),
                        (30, 30, 30),
                        -1,
                    )

                # Dessin simulant des mains sur le volant (dans le tiers inférieur)
                if draw_hands:
                    hand_y = int(height * 0.85)
                    hand_radius = min(width, height) // 10
                    # Main gauche
                    cv2.circle(
                        frame,
                        (int(width * 0.3), hand_y),
                        hand_radius,
                        (140, 170, 205),
                        -1,
                    )
                    # Main droite
                    cv2.circle(
                        frame,
                        (int(width * 0.7), hand_y),
                        hand_radius,
                        (140, 170, 205),
                        -1,
                    )

                # Horodatage textuel pour différencier les frames
                cv2.putText(
                    frame,
                    f"F:{idx:03d}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

                out.write(frame)
        finally:
            out.release()

        return video_path

    # =========================================================================
    # TEST 1 : Calcul des 8 indices temporels et ordre chronologique
    # =========================================================================
    def test_sample_indices_computation_and_chronology(self):
        """Vérifie la répartition temporelle exacte et l'ordre chronologique strict."""
        test_cases = [
            (240, "8s @ 30 FPS"),
            (480, "8s @ 60 FPS"),
            (120, "8s @ 15 FPS"),
            (60, "2s @ 30 FPS"),
            (8, "Minimum requis 8 frames"),
        ]

        for total_frames, description in test_cases:
            with self.subTest(case=description):
                indices = self.preprocessor.compute_sample_indices(total_frames)

                # Exactement 8 frames
                self.assertEqual(len(indices), NUM_SEGMENTS)

                # Tous les indices sont valides
                for idx in indices:
                    self.assertGreaterEqual(idx, 0)
                    self.assertLess(idx, total_frames)

                # Ordre chronologique strict
                for i in range(len(indices) - 1):
                    self.assertLess(
                        indices[i],
                        indices[i + 1],
                        f"Non chronologique pour {description}: {indices[i]} >= {indices[i+1]}",
                    )

                # Vérification avec la formule exacte du notebook DMD / preprocessing.py
                expected = [
                    min(int((i + 0.5) * total_frames / NUM_SEGMENTS), total_frames - 1)
                    for i in range(NUM_SEGMENTS)
                ]
                self.assertEqual(indices, expected)

    def test_sample_indices_raises_on_too_short_video(self):
        """Une vidéo de moins de 8 frames doit lever une ValueError explicite."""
        with self.assertRaises(ValueError) as ctx:
            self.preprocessor.compute_sample_indices(7)
        self.assertIn("au moins 8 frames", str(ctx.exception))

    # =========================================================================
    # TEST 2 : Équivalence mathématique avec preprocessing.py
    # =========================================================================
    def test_tensor_normalization_matches_preprocessing_py(self):
        """
        Vérifie que la normalisation de video_preprocessor produit un tenseur
        strictement identique à celle de preprocessing.py.
        """
        # Création de 8 frames BGR aléatoires fixes
        np.random.seed(42)
        raw_frames = [
            np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
            for _ in range(NUM_SEGMENTS)
        ]

        # 1. Calcul via video_preprocessor
        preproc_tensor = self.preprocessor.normalize_to_tensor(raw_frames)

        # 2. Reproduction exacte de preprocessing.py load_stream
        expected_frames = [
            cv2.resize(im, TARGET_SIZE, interpolation=cv2.INTER_LINEAR)
            for im in raw_frames
        ]
        arr = np.stack(expected_frames).transpose(0, 3, 1, 2).copy()
        data = torch.from_numpy(arr).float().div_(255.0)
        mean = torch.tensor(MEAN_RGB_BGR_ORDER).view(1, 3, 1, 1)
        std = torch.tensor(STD_RGB_BGR_ORDER).view(1, 3, 1, 1)
        expected_tensor = ((data - mean) / std).unsqueeze(0)

        # Vérification des formes
        self.assertEqual(preproc_tensor.shape, (1, 8, 3, 224, 224))
        self.assertEqual(expected_tensor.shape, (1, 8, 3, 224, 224))

        # Vérification de l'égalité numérique parfaite (tolérance 1e-6)
        diff = torch.max(torch.abs(preproc_tensor - expected_tensor)).item()
        self.assertLess(
            diff,
            1e-6,
            f"Différence numérique avec preprocessing.py: {diff}",
        )

    # =========================================================================
    # TEST 3 : Extraction multi-résolutions et multi-FPS
    # =========================================================================
    def test_extraction_multiple_resolutions_and_fps(self):
        """
        Vérifie l'extraction des 8 frames sur diverses résolutions (1080p, 720p, 480p)
        et cadences d'images (15, 24, 30, 60 FPS).
        """
        configurations = [
            ("1080p_30fps.mp4", 1920, 1080, 30.0, 8.0),
            ("720p_60fps.mp4", 1280, 720, 60.0, 8.0),
            ("480p_15fps.mp4", 640, 480, 15.0, 8.0),
            ("portrait_24fps.mp4", 720, 1280, 24.0, 8.0),
            ("short_2s_30fps.mp4", 640, 480, 30.0, 2.0),
        ]

        for fname, w, h, fps, dur in configurations:
            with self.subTest(file=fname):
                vpath = self._create_synthetic_video(
                    fname, w, h, fps, dur, draw_face=True
                )
                try:
                    result = self.preprocessor.process(vpath)

                    # 1. Vérification métadonnées
                    self.assertAlmostEqual(result.metadata.fps, fps, places=1)
                    self.assertEqual(len(result.sampled_frame_indices), 8)
                    self.assertEqual(len(result.sampled_timestamps_ms), 8)

                    # 2. Vérification ordre chronologique des timestamps
                    for i in range(len(result.sampled_timestamps_ms) - 1):
                        self.assertLess(
                            result.sampled_timestamps_ms[i],
                            result.sampled_timestamps_ms[i + 1],
                        )

                    # 3. Vérification forme du tenseur PyTorch
                    self.assertEqual(result.tensor.shape, (1, 8, 3, 224, 224))
                    self.assertEqual(result.tensor.dtype, torch.float32)

                    # 4. Vérification que le tenseur n'est ni NaN ni Inf
                    self.assertFalse(torch.isnan(result.tensor).any().item())
                    self.assertFalse(torch.isinf(result.tensor).any().item())

                finally:
                    if vpath.exists():
                        vpath.unlink()

    # =========================================================================
    # TEST 4 : Rotation Android et orientation
    # =========================================================================
    def test_android_rotation_handling(self):
        """Vérifie la correction d'orientation sur 0°, 90°, 180° et 270°."""
        sample_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Point repère en haut à gauche
        sample_frame[0:50, 0:50] = (255, 0, 0)

        # 0° : pas de changement
        rot0 = self.preprocessor.apply_rotation(sample_frame, 0)
        self.assertEqual(rot0.shape, (480, 640, 3))

        # 90° : inversion hauteur/largeur
        rot90 = self.preprocessor.apply_rotation(sample_frame, 90)
        self.assertEqual(rot90.shape, (640, 480, 3))

        # 180° : conservation dimensions
        rot180 = self.preprocessor.apply_rotation(sample_frame, 180)
        self.assertEqual(rot180.shape, (480, 640, 3))

        # 270° : inversion hauteur/largeur
        rot270 = self.preprocessor.apply_rotation(sample_frame, 270)
        self.assertEqual(rot270.shape, (640, 480, 3))

    # =========================================================================
    # TEST 5 : Détection des zones anatomiques et signalement d'indisponibilité
    # =========================================================================
    def test_zone_availability_and_no_pixel_invention(self):
        """
        Vérifie :
        1. Détection de visage quand présent.
        2. Signalement formel que les mains sont indisponibles quand absentes du champ.
        3. Aucune invention de pixels ni duplication dans rgb_hands.
        4. Verdict de non-compatibilité avec le modèle multi-caméras DMD.
        """
        # Vidéo simulant une caméra frontale classique (visage seul visible, pas de mains)
        fname = "android_face_no_hands.mp4"
        vpath = self._create_synthetic_video(
            fname,
            640,
            480,
            30.0,
            4.0,
            draw_face=True,
            draw_body=False,
            draw_hands=False,
        )

        try:
            result = self.preprocessor.process(vpath)

            # Le rapport de visibilité doit marquer la non-compatibilité DMD multi-vues
            self.assertFalse(result.visibility.is_dmd_compatible)
            self.assertIn("INCOMPATIBLE_MULTI_VIEW", result.visibility.compatibility_verdict)

            # Mains expressément manquantes
            self.assertIn("hands", result.visibility.missing_views)
            self.assertFalse(result.visibility.hands.is_available)

            # RÈGLE D'OR : Aucune invention de pixels !
            # experimental_crops['rgb_hands'] doit être strictement None
            self.assertIsNone(
                result.experimental_crops["rgb_hands"],
                "rgb_hands ne doit PAS être synthétisé ou dupliqué lorsque les mains sont absentes !",
            )

        finally:
            if vpath.exists():
                vpath.unlink()

    def test_blank_video_flags_all_views_missing(self):
        """Sur une vidéo sans sujet, toutes les vues doivent être déclarées indisponibles."""
        fname = "blank_video.mp4"
        vpath = self._create_synthetic_video(
            fname,
            640,
            480,
            30.0,
            2.0,
            draw_face=False,
            draw_body=False,
            draw_hands=False,
        )

        try:
            result = self.preprocessor.process(vpath)

            self.assertFalse(result.visibility.face.is_available)
            self.assertFalse(result.visibility.body.is_available)
            self.assertFalse(result.visibility.hands.is_available)
            self.assertEqual(len(result.visibility.available_views), 0)
            self.assertEqual(set(result.visibility.missing_views), {"face", "body", "hands"})
            self.assertIsNone(result.experimental_crops["rgb_body"])
            self.assertIsNone(result.experimental_crops["rgb_face"])
            self.assertIsNone(result.experimental_crops["rgb_hands"])

        finally:
            if vpath.exists():
                vpath.unlink()

    # =========================================================================
    # TEST 6 : Gestion des erreurs
    # =========================================================================
    def test_non_existent_file_raises_filenotfound(self):
        """Un fichier inexistant doit lever FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            self.preprocessor.process(self.test_dir / "non_existent.mp4")


if __name__ == "__main__":
    unittest.main(verbosity=2)
