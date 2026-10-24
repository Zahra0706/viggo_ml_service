import unittest
import numpy as np
import cv2

from swin_service import (
    SwinInferenceService,
    ImageQualityAnalyzer,
    ComplementaryFacialAnalyzer,
    TemporalTracker,
    EXPECTED_CLASSES,
)


class TestPhase6FQualityAndTemporal(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.service = SwinInferenceService.get_instance()

    def test_quality_analyzer_blurry_image(self):
        """Une image très floue doit être marquée unreliable."""
        # Création d'une image floue uniforme
        blurry = np.full((240, 320, 3), 128, dtype=np.uint8)
        cv2.GaussianBlur(blurry, (25, 25), 0, dst=blurry)
        q = self.service.quality_analyzer.analyze(blurry)

        self.assertFalse(q["is_reliable"])
        self.assertIn("Flou", q["quality_reason"])
        self.assertTrue(q["laplacian_variance"] < 45.0)

    def test_quality_analyzer_dark_image(self):
        """Une image trop sombre (< 35 moyenne) doit être marquée unreliable."""
        dark = np.full((240, 320, 3), 15, dtype=np.uint8)
        q = self.service.quality_analyzer.analyze(dark)

        self.assertFalse(q["is_reliable"])
        self.assertIn("sombre", q["quality_reason"].lower())

    def test_quality_analyzer_no_driver(self):
        """Une image sans aucun conducteur doit avoir le statut analysis_unavailable."""
        empty_scene = np.full((240, 320, 3), 120, dtype=np.uint8)
        # Texture sans visage
        cv2.rectangle(empty_scene, (50, 50), (200, 200), (80, 80, 80), -1)
        q = self.service.quality_analyzer.analyze(empty_scene)

        self.assertFalse(q["face_detected"])
        self.assertEqual(q["quality_status"], "analysis_unavailable")
        self.assertEqual(q["framing_status"], "face_not_detected")

    def test_facial_cross_validation_yawn_override(self):
        """Si Swin prédit Yawn mais que la bouche est fermée, le score Yawn est rétrogradé."""
        analyzer = self.service.facial_analyzer
        # Face synthétique avec bouche fermée (pas de cavité sombre)
        dummy_face = np.full((120, 100, 3), 160, dtype=np.uint8)
        initial_scores = {
            "DangerousDriving": 0.05,
            "Distracted": 0.05,
            "Drinking": 0.05,
            "SafeDriving": 0.35,
            "SleepyDriving": 0.05,
            "Yawn": 0.45,
        }

        new_class, new_scores, note = analyzer.cross_validate(
            dummy_face, "Yawn", initial_scores
        )

        self.assertIsNotNone(note)
        self.assertIn("Contradiction visuelle", note)
        self.assertNotEqual(new_class, "Yawn")
        self.assertLess(new_scores["Yawn"], 0.20)

    def test_temporal_tracker_avoids_single_blink_alert(self):
        """Un clignement de paupière isolé (1 frame) ne doit pas basculer en SleepyDriving."""
        tracker = TemporalTracker(window_size=5, alpha=0.40)
        tracker.reset()

        # Frame 1 : SafeDriving stable
        res1 = tracker.update(
            {"SafeDriving": 0.90, "SleepyDriving": 0.02, "DangerousDriving": 0.02, "Distracted": 0.02, "Drinking": 0.02, "Yawn": 0.02},
            "SafeDriving", True, 1000.0
        )
        self.assertEqual(res1["consolidated_state"], "SafeDriving")

        # Frame 2 : Faux clignement isolé vers SleepyDriving
        res2 = tracker.update(
            {"SafeDriving": 0.20, "SleepyDriving": 0.75, "DangerousDriving": 0.01, "Distracted": 0.01, "Drinking": 0.01, "Yawn": 0.02},
            "SleepyDriving", True, 1003.0
        )
        # La bascule immédiate en alerte somnolence est évitée
        self.assertNotEqual(res2["consolidated_state"], "SleepyDriving")

        # Frame 3 : Retour à la normale
        res3 = tracker.update(
            {"SafeDriving": 0.88, "SleepyDriving": 0.03, "DangerousDriving": 0.02, "Distracted": 0.02, "Drinking": 0.02, "Yawn": 0.03},
            "SafeDriving", True, 1006.0
        )
        self.assertEqual(res3["consolidated_state"], "SafeDriving")

    def test_temporal_tracker_sustained_danger_confirmed(self):
        """Une somnolence persistante (au moins 2 frames successives) est bien confirmée."""
        tracker = TemporalTracker(window_size=5, alpha=0.40)
        tracker.reset()

        # Frame 1 : Début somnolence
        tracker.update(
            {"SafeDriving": 0.10, "SleepyDriving": 0.85, "DangerousDriving": 0.01, "Distracted": 0.01, "Drinking": 0.01, "Yawn": 0.02},
            "SleepyDriving", True, 1000.0
        )
        # Frame 2 : Confirmation somnolence
        res = tracker.update(
            {"SafeDriving": 0.10, "SleepyDriving": 0.85, "DangerousDriving": 0.01, "Distracted": 0.01, "Drinking": 0.01, "Yawn": 0.02},
            "SleepyDriving", True, 1003.0
        )
        self.assertEqual(res["consolidated_state"], "SleepyDriving")


if __name__ == "__main__":
    unittest.main()
