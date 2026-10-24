from typing import Dict, List, Optional, Tuple, Any
import math
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

logger = logging.getLogger("viggo.face_dms")

NUM_SEGMENTS = 8
DROPOUT = 0.35
FEAT_DIM = 1408
PROJ_DIM = 512

FACE_DMS_CLASSES = {
    "drowsiness": ["eyes_open", "eyes_closed", "yawning"],
    "gaze": ["looking_road", "not_looking_road"],
}


class TemporalShift(nn.Module):
    """
    TSM (Temporal Shift Module) : décale une fraction des canaux (1/8)
    vers l'avant et l'arrière dans le temps sans introduire de paramètres apprenables.
    Reproduit exactement l'implémentation de model_architecture.py.
    """
    def __init__(self, n_segment: int = NUM_SEGMENTS, n_div: int = 8):
        super().__init__()
        self.n_segment = n_segment
        self.n_div = n_div

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        BT, C, H, W = x.shape
        T = self.n_segment
        B = BT // T
        x = x.view(B, T, C, H, W)
        fold = max(1, C // self.n_div)
        out = x.clone()
        out[:, 1:, :fold] = x[:, :-1, :fold]
        out[:, :-1, fold:2 * fold] = x[:, 1:, fold:2 * fold]
        return out.view(BT, C, H, W)


class TaskHead(nn.Module):
    """
    Tête de classification multi-tâches à deux étages.
    Identique structurellement à TaskHead de model_architecture.py.
    """
    def __init__(self, in_dim: int = PROJ_DIM, num_classes: int = 2, dropout: float = DROPOUT):
        super().__init__()
        hidden = in_dim // 2  # 256
        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class FaceOnlyDMSNet(nn.Module):
    """
    Réseau neuronal profond mono-caméra dédié à la surveillance du visage.
    Conçu spécifiquement pour le cas d'usage smartphone Android VIGGO :
    un seul capteur orienté vers le conducteur, surveillant somnolence et regard.
    """

    def __init__(
        self,
        num_segments: int = NUM_SEGMENTS,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.num_segments = num_segments
        self.dropout = dropout

        # 1. Module temporel
        self.tsm = TemporalShift(n_segment=num_segments, n_div=8)

        # 2. Extracteur spatial EfficientNet-B2
        self.face_backbone = timm.create_model(
            "efficientnet_b2",
            pretrained=False,
            in_chans=3,
            num_classes=0,
            global_pool="avg"
        )
        self.feat_dim = self.face_backbone.num_features  # 1408

        # 3. Consensus temporel et projection (identique à face_consensus de RGBDriverFusionNet)
        self.face_consensus = nn.Sequential(
            nn.Linear(self.feat_dim, PROJ_DIM),
            nn.BatchNorm1d(PROJ_DIM),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 4. Têtes de classification adaptées aux deux tâches du visage
        self.task_heads = nn.ModuleDict({
            task: TaskHead(PROJ_DIM, len(classes), dropout)
            for task, classes in FACE_DMS_CLASSES.items()
        })

    def _freeze_early_backbone_layers(self):
        """Gèle les premières couches du backbone pour stabiliser le fine-tuning."""
        for name, param in self.face_backbone.named_parameters():
            if (name.startswith("conv_stem") or name.startswith("bn1")
                or name.startswith("blocks.0.") or name.startswith("blocks.1.")):
                param.requires_grad = False

    @staticmethod
    def _temporal_pool(feat: torch.Tensor) -> torch.Tensor:
        """Max pooling adaptatif sur la dimension temporelle : (B, T, D) -> (B, D)."""
        return F.adaptive_max_pool1d(feat.permute(0, 2, 1), 1).squeeze(-1)

    def forward(self, rgb_face: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Passe avant du modèle :
        Args:
            rgb_face: Tenseur de forme (B, 8, 3, 224, 224)
        Returns:
            Dict contenant les logits bruts pour 'drowsiness' (B, 3) et 'gaze' (B, 2)
        """
        if rgb_face.dim() != 5:
            raise ValueError(
                f"Forme d'entrée attendue : (B, {self.num_segments}, 3, 224, 224), "
                f"reçue : {list(rgb_face.shape)}"
            )

        B, T, C, H, W = rgb_face.shape
        if T != self.num_segments:
            raise ValueError(
                f"Nombre de segments temporel incorrect : {T} (attendu : {self.num_segments})"
            )

        # Aplatissement temporel pour traitement par batch (B*T, C, H, W)
        flat = rgb_face.view(B * T, C, H, W)

        # Décalage temporel inter-frames
        flat = self.tsm(flat)

        # Extraction spatiale via EfficientNet-B2
        feat = self.face_backbone(flat)  # (B*T, 1408)

        # Restauration de la dimension temporelle (B, T, 1408)
        feat = feat.view(B, T, -1)

        # Pooling temporel (Max Pooling sur les 8 segments)
        pooled = self._temporal_pool(feat)  # (B, 1408)

        # Projection consensus visage (B, 512)
        face_repr = self.face_consensus(pooled)

        # Évaluation des deux têtes de tâches
        return {
            "drowsiness": self.task_heads["drowsiness"](face_repr),
            "gaze": self.task_heads["gaze"](face_repr),
        }

    def get_trainable_param_groups(self, lr_backbone: float = 1e-4, lr_head: float = 1e-3) -> List[Dict]:
        """Prépare des groupes de paramètres différentiés pour le fine-tuning."""
        backbone_params = [p for p in self.face_backbone.parameters() if p.requires_grad]
        head_params = (
            list(self.face_consensus.parameters())
            + list(self.task_heads.parameters())
        )
        return [
            {"params": backbone_params, "lr": lr_backbone, "weight_decay": 1e-4},
            {"params": head_params, "lr": lr_head, "weight_decay": 1e-3},
        ]


def load_face_dms_pretrained_weights(
    model: FaceOnlyDMSNet,
    checkpoint_path: Path | str,
    strict_check: bool = True
) -> Dict[str, Any]:
    """
    Transfère sélectivement les poids compatibles depuis le checkpoint RGB multi-vues (rgb_best_model.pth).
    
    Couches transférées (531 clés) :
    - face_backbone.* (506 clés) : EfficientNet-B2 complet pour le visage
    - face_consensus.* (7 clés) : Linéaire 1408->512 et BatchNorm
    - task_heads.drowsiness.* (9 clés) : Tête somnolence (warm-start)
    - task_heads.gaze.* (9 clés) : Tête regard (warm-start)
    
    Couches ignorées (549 clés non pertinentes pour mono-caméra) :
    - body_backbone.* (506 clés)
    - body_hands_fuse.* (4 clés)
    - body_consensus.* (7 clés)
    - cross_attn.* (14 clés)
    - task_heads.distraction.* (9 clés)
    - task_heads.hands.* (9 clés)
    """
    ckpt_path = Path(checkpoint_path).resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint introuvable : {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if "model_state_dict" not in checkpoint:
        raise ValueError(f"Format de checkpoint invalide (clé 'model_state_dict' manquante) : {ckpt_path}")

    source_sd = checkpoint["model_state_dict"]
    target_sd = model.state_dict()

    transferred_keys = []
    ignored_keys = []

    for key, param in source_sd.items():
        if key.startswith("face_backbone."):
            if key in target_sd and target_sd[key].shape == param.shape:
                target_sd[key] = param.clone()
                transferred_keys.append(key)
        elif key.startswith("face_consensus."):
            if key in target_sd and target_sd[key].shape == param.shape:
                target_sd[key] = param.clone()
                transferred_keys.append(key)
        elif key.startswith("task_heads.drowsiness."):
            if key in target_sd and target_sd[key].shape == param.shape:
                target_sd[key] = param.clone()
                transferred_keys.append(key)
        elif key.startswith("task_heads.gaze."):
            if key in target_sd and target_sd[key].shape == param.shape:
                target_sd[key] = param.clone()
                transferred_keys.append(key)
        else:
            ignored_keys.append(key)

    # Chargement strict dans le modèle FaceOnlyDMSNet
    model.load_state_dict(target_sd, strict=True)

    coverage_ratio = len(transferred_keys) / len(target_sd)
    report = {
        "source_checkpoint": str(ckpt_path.name),
        "total_source_keys": len(source_sd),
        "total_target_keys": len(target_sd),
        "transferred_keys_count": len(transferred_keys),
        "ignored_source_keys_count": len(ignored_keys),
        "target_parameters_coverage": round(coverage_ratio * 100.0, 2),
        "coverage_complete": coverage_ratio == 1.0,
        "transferred_categories": {
            "face_backbone": len([k for k in transferred_keys if k.startswith("face_backbone")]),
            "face_consensus": len([k for k in transferred_keys if k.startswith("face_consensus")]),
            "task_heads.drowsiness": len([k for k in transferred_keys if "drowsiness" in k]),
            "task_heads.gaze": len([k for k in transferred_keys if "gaze" in k]),
        }
    }

    if strict_check and not report["coverage_complete"]:
        raise RuntimeError(
            f"Transfert incomplet : {len(transferred_keys)}/{len(target_sd)} clés cibles chargées."
        )

    return report
