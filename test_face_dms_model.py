import hashlib
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

from face_dms_model import (
    FaceOnlyDMSNet,
    load_face_dms_pretrained_weights,
    NUM_SEGMENTS,
    PROJ_DIM,
    FEAT_DIM,
    FACE_DMS_CLASSES,
)
from preprocessing import load_stream


class TestFaceOnlyDMSNet(unittest.TestCase):
    """Tests unitaires et de transfert de poids pour FaceOnlyDMSNet."""

    @classmethod
    def setUpClass(cls):
        cls.base_dir = Path(__file__).resolve().parent
        cls.ckpt_path = cls.base_dir / "models" / "rgb_best_model.pth"
        cls.sample_face_dir = cls.base_dir / "samples" / "session2" / "frames_rgb_face"

        # Empreinte SHA256 pour vérifier l'intégrité du checkpoint
        if cls.ckpt_path.exists():
            with open(cls.ckpt_path, "rb") as f:
                cls.original_ckpt_hash = hashlib.sha256(f.read()).hexdigest()

    def test_model_instantiation_and_structure(self):
        """Vérifie l'instanciation et les dimensions internes du modèle."""
        model = FaceOnlyDMSNet(num_segments=NUM_SEGMENTS)
        self.assertEqual(model.feat_dim, FEAT_DIM)  # 1408
        self.assertIn("drowsiness", model.task_heads)
        self.assertIn("gaze", model.task_heads)

        # Vérification des classes
        self.assertEqual(len(FACE_DMS_CLASSES["drowsiness"]), 3)
        self.assertEqual(len(FACE_DMS_CLASSES["gaze"]), 2)

    def test_selective_weight_loading_and_audit(self):
        """Vérifie le chargement sélectif et la complétude des paramètres transférés."""
        if not self.ckpt_path.exists():
            self.skipTest("models/rgb_best_model.pth non disponible.")

        model = FaceOnlyDMSNet()
        report = load_face_dms_pretrained_weights(model, self.ckpt_path, strict_check=True)

        # Vérifications quantitatives de l'audit
        self.assertEqual(report["total_source_keys"], 1080)
        self.assertEqual(report["total_target_keys"], 531)
        self.assertEqual(report["transferred_keys_count"], 531)
        self.assertEqual(report["ignored_source_keys_count"], 549)
        self.assertEqual(report["target_parameters_coverage"], 100.0)
        self.assertTrue(report["coverage_complete"])

        # Détail des catégories
        cats = report["transferred_categories"]
        self.assertEqual(cats["face_backbone"], 506)
        self.assertEqual(cats["face_consensus"], 7)
        self.assertEqual(cats["task_heads.drowsiness"], 9)
        self.assertEqual(cats["task_heads.gaze"], 9)

    def test_forward_pass_synthetic_tensors(self):
        """Vérifie la passe avant avec des tenseurs synthétiques B=1 et B=2."""
        model = FaceOnlyDMSNet()
        model.eval()

        for batch_size in [1, 2]:
            with self.subTest(batch_size=batch_size):
                dummy_input = torch.randn(batch_size, NUM_SEGMENTS, 3, 224, 224)
                with torch.no_grad():
                    outputs = model(dummy_input)

                self.assertIn("drowsiness", outputs)
                self.assertIn("gaze", outputs)

                # Formes attendues
                self.assertEqual(outputs["drowsiness"].shape, (batch_size, 3))
                self.assertEqual(outputs["gaze"].shape, (batch_size, 2))

                # Absence de NaN ou Inf
                self.assertFalse(torch.isnan(outputs["drowsiness"]).any().item())
                self.assertFalse(torch.isinf(outputs["drowsiness"]).any().item())
                self.assertFalse(torch.isnan(outputs["gaze"]).any().item())
                self.assertFalse(torch.isinf(outputs["gaze"]).any().item())

                # Vérification des probabilités Softmax
                p_drow = F.softmax(outputs["drowsiness"], dim=1)
                p_gaze = F.softmax(outputs["gaze"], dim=1)
                self.assertTrue(torch.allclose(p_drow.sum(dim=1), torch.ones(batch_size), atol=1e-5))
                self.assertTrue(torch.allclose(p_gaze.sum(dim=1), torch.ones(batch_size), atol=1e-5))

    def test_forward_pass_on_real_dmd_face_sample(self):
        """Vérifie la passe avant sur de véritables frames DMD de visage."""
        if not self.sample_face_dir.exists():
            self.skipTest("samples/session2/frames_rgb_face non disponible.")

        # Chargement via preprocessing.py (conforme BGR / ImageNet)
        face_tensor = load_stream(self.sample_face_dir, "rgb")  # (1, 8, 3, 224, 224)

        model = FaceOnlyDMSNet()
        load_face_dms_pretrained_weights(model, self.ckpt_path, strict_check=True)
        model.eval()

        with torch.no_grad():
            outputs = model(face_tensor)

        p_drow = F.softmax(outputs["drowsiness"], dim=1)[0]
        p_gaze = F.softmax(outputs["gaze"], dim=1)[0]

        best_drow = FACE_DMS_CLASSES["drowsiness"][int(p_drow.argmax().item())]
        best_gaze = FACE_DMS_CLASSES["gaze"][int(p_gaze.argmax().item())]

        # Les sorties doivent être des prédictions valides
        self.assertIn(best_drow, FACE_DMS_CLASSES["drowsiness"])
        self.assertIn(best_gaze, FACE_DMS_CLASSES["gaze"])

    def test_invalid_input_shapes_raise_value_error(self):
        """Vérifie la détection d'erreurs sur des entrées mal dimensionnées."""
        model = FaceOnlyDMSNet()

        # Dimension invalide : 4D au lieu de 5D
        with self.assertRaises(ValueError):
            model(torch.randn(1, 3, 224, 224))

        # Segments invalides : 7 frames au lieu de 8
        with self.assertRaises(ValueError):
            model(torch.randn(1, 7, 3, 224, 224))

    def test_original_checkpoint_integrity_not_modified(self):
        """Garantit formellement que rgb_best_model.pth n'a pas été modifié."""
        if hasattr(self, "original_ckpt_hash") and self.ckpt_path.exists():
            with open(self.ckpt_path, "rb") as f:
                current_hash = hashlib.sha256(f.read()).hexdigest()
            self.assertEqual(
                current_hash,
                self.original_ckpt_hash,
                "ALERTE : rgb_best_model.pth a été altéré ou écrasé !",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
