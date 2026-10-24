import cv2
import numpy as np
import unittest
from fastapi.testclient import TestClient

from api import app
from swin_service import SwinInferenceService, EXPECTED_CLASSES


def create_dummy_jpeg(width: int = 320, height: int = 240, color=(120, 100, 80)) -> bytes:
    """Crée une image JPEG de test en mémoire."""
    img = np.full((height, width, 3), color, dtype=np.uint8)
    cv2.circle(img, (width // 2, height // 2), min(width, height) // 4, (200, 200, 200), -1)
    success, buffer = cv2.imencode(".jpg", img)
    assert success
    return buffer.tobytes()


class TestSwinImageService(unittest.TestCase):
    def test_swin_predict_image_unit(self):
        """Vérifie l'inférence Swin unitaire sur une image en mémoire."""
        service = SwinInferenceService.get_instance()
        jpeg_bytes = create_dummy_jpeg(320, 240)

        result = service.predict_image(jpeg_bytes)

        self.assertEqual(result["model"], "driver_swin.onnx")
        self.assertIn("prediction", result)
        self.assertIn(result["prediction"]["class"], EXPECTED_CLASSES)
        self.assertTrue(0 <= result["prediction"]["class_index"] < len(EXPECTED_CLASSES))
        self.assertTrue(0.0 <= result["prediction"]["confidence"] <= 1.0)

        # Vérification des 6 classes dans les scores
        scores = result["scores"]
        self.assertEqual(len(scores), len(EXPECTED_CLASSES))
        for c in EXPECTED_CLASSES:
            self.assertIn(c, scores)
            self.assertTrue(0.0 <= scores[c] <= 1.0)

        # Somme des probabilités softmax ~ 1.0
        total_prob = sum(scores.values())
        self.assertAlmostEqual(total_prob, 1.0, places=2)

        # Métadonnées
        self.assertEqual(result["image_metadata"]["width"], 320)
        self.assertEqual(result["image_metadata"]["height"], 240)
        self.assertGreater(result["performance"]["total_time_ms"], 0)

    def test_swin_predict_image_empty_and_corrupt(self):
        """Vérifie que des octets vides ou corrompus lèvent une ValueError claire."""
        service = SwinInferenceService.get_instance()

        with self.assertRaises(ValueError):
            service.predict_image(b"")

        with self.assertRaises(ValueError):
            service.predict_image(b"ceci n'est pas une image")

    def test_api_predict_image_swin_endpoint(self):
        """Vérifie l'endpoint FastAPI POST /image/predict-swin."""
        client = TestClient(app)
        jpeg_bytes = create_dummy_jpeg(400, 300)

        response = client.post(
            "/image/predict-swin",
            files={"file": ("frame.jpg", jpeg_bytes, "image/jpeg")},
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["success"])
        self.assertIn("data", data)
        self.assertIn(data["data"]["prediction"]["class"], EXPECTED_CLASSES)

    def test_api_predict_image_invalid_signature(self):
        """Vérifie le rejet HTTP 400 si les magic bytes ne correspondent pas à JPEG/PNG/WEBP."""
        client = TestClient(app)

        response = client.post(
            "/image/predict-swin",
            files={"file": ("fake.jpg", b"Texte non binaire", "image/jpeg")},
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("Format d'image non supporté", response.json()["detail"])


if __name__ == "__main__":
    unittest.main()
