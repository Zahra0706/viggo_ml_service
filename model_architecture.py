NUM_SEGMENTS = 8
DROPOUT = 0.35
TASK_N_CLASSES = {"distraction": 8, "drowsiness": 3, "gaze": 2, "hands": 3}
ACTIVE_TASKS = list(TASK_N_CLASSES)
TASK_CAMERA = {"distraction": "body", "drowsiness": "face", "gaze": "face", "hands": "body"}
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
import timm


class TemporalShift(nn.Module):
    """TSM: shifts a small part of channels forward/backward across time."""
    def __init__(self, n_segment: int, n_div: int = 8):
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
        out[:, :-1, fold:2*fold] = x[:, 1:, fold:2*fold]
        return out.view(BT, C, H, W)


class CrossViewAttention(nn.Module):
    """Gated interaction between body and face features after TSN pooling."""
    def __init__(self, dim: int, proj_dim: int = 128):
        super().__init__()
        self.scale = 1.0 / math.sqrt(proj_dim)
        self.b2f_q = nn.Linear(dim, proj_dim)
        self.b2f_k = nn.Linear(dim, proj_dim)
        self.b2f_v = nn.Linear(dim, dim)
        self.b2f_gate = nn.Parameter(torch.zeros(1))
        self.f2b_q = nn.Linear(dim, proj_dim)
        self.f2b_k = nn.Linear(dim, proj_dim)
        self.f2b_v = nn.Linear(dim, dim)
        self.f2b_gate = nn.Parameter(torch.zeros(1))

    def forward(self, body: torch.Tensor, face: torch.Tensor):
        q_b, k_f, v_f = self.b2f_q(body), self.b2f_k(face), self.b2f_v(face)
        att_b = torch.sigmoid((q_b * k_f).sum(dim=-1, keepdim=True) * self.scale)
        body_out = body + torch.tanh(self.b2f_gate) * (att_b * v_f)

        q_f, k_b, v_b = self.f2b_q(face), self.f2b_k(body), self.f2b_v(body)
        att_f = torch.sigmoid((q_f * k_b).sum(dim=-1, keepdim=True) * self.scale)
        face_out = face + torch.tanh(self.f2b_gate) * (att_f * v_b)
        return body_out, face_out


class TaskHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, dropout: float = 0.35):
        super().__init__()
        hidden = in_dim // 2
        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        return self.head(x)


class RGBDriverFusionNet(nn.Module):
    """
    RGB-only multi-view fusion model.

    body  : rgb_body  → body EfficientNet
    hands : rgb_hands → same body EfficientNet, then fused with body
    face  : rgb_face  → face EfficientNet

    Improvements kept from v2:
    - EfficientNet-B2 pretrained backbones
    - TSM temporal shift before backbone
    - body+hands fusion for distraction/hands
    - gated body-face interaction
    - task-specific heads
    """
    PROJ_DIM = 512

    def __init__(self, num_segments: int = NUM_SEGMENTS, dropout: float = DROPOUT):
        super().__init__()
        self.num_segments = num_segments

        self.body_backbone = timm.create_model(
            "efficientnet_b2", pretrained=False, in_chans=3, num_classes=0, global_pool="avg"
        )
        self.face_backbone = timm.create_model(
            "efficientnet_b2", pretrained=False, in_chans=3, num_classes=0, global_pool="avg"
        )
        self.feat_dim = self.body_backbone.num_features

        self._freeze_early_layers(self.body_backbone)
        self._freeze_early_layers(self.face_backbone)
        self.tsm = TemporalShift(num_segments, n_div=8)

        self.body_hands_fuse = nn.Sequential(
            nn.Linear(self.feat_dim * 2, self.feat_dim),
            nn.LayerNorm(self.feat_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )

        self.body_consensus = nn.Sequential(
            nn.Linear(self.feat_dim, self.PROJ_DIM),
            nn.BatchNorm1d(self.PROJ_DIM),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.face_consensus = nn.Sequential(
            nn.Linear(self.feat_dim, self.PROJ_DIM),
            nn.BatchNorm1d(self.PROJ_DIM),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.cross_attn = CrossViewAttention(self.PROJ_DIM, proj_dim=128)
        self.task_heads = nn.ModuleDict({
            task: TaskHead(self.PROJ_DIM, n_cls, dropout)
            for task, n_cls in TASK_N_CLASSES.items()
        })

    @staticmethod
    def _freeze_early_layers(backbone: nn.Module):
        for name, param in backbone.named_parameters():
            if (name.startswith("conv_stem") or name.startswith("bn1")
                or name.startswith("blocks.0.") or name.startswith("blocks.1.")):
                param.requires_grad = False

    def _backbone_forward(self, backbone: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if self.training and torch.is_grad_enabled():
            x = x.requires_grad_(True)
            return cp.checkpoint(backbone, x, use_reentrant=False)
        return backbone(x)

    def _extract_features(self, backbone: nn.Module, x: torch.Tensor) -> torch.Tensor:
        B, N, C, H, W = x.shape
        flat = x.reshape(B * N, C, H, W)
        flat = self.tsm(flat)
        feat = self._backbone_forward(backbone, flat)
        return feat.view(B, N, -1)

    @staticmethod
    def _temporal_pool(feat: torch.Tensor) -> torch.Tensor:
        # Max pooling works well for sparse discriminative actions; average is also possible.
        return F.adaptive_max_pool1d(feat.permute(0, 2, 1), 1).squeeze(-1)

    def forward(self, streams: dict) -> dict:
        body_feat  = self._extract_features(self.body_backbone, streams["rgb_body"])
        hands_feat = self._extract_features(self.body_backbone, streams["rgb_hands"])
        face_feat  = self._extract_features(self.face_backbone, streams["rgb_face"])

        body_combined = self.body_hands_fuse(torch.cat([body_feat, hands_feat], dim=-1))
        body_pooled = self.body_consensus(self._temporal_pool(body_combined))
        face_pooled = self.face_consensus(self._temporal_pool(face_feat))
        body_out, face_out = self.cross_attn(body_pooled, face_pooled)

        return {
            task: self.task_heads[task](body_out if TASK_CAMERA[task] == "body" else face_out)
            for task in ACTIVE_TASKS
        }

    def get_backbone_params(self) -> list:
        return list(self.body_backbone.parameters()) + list(self.face_backbone.parameters())

    def get_head_params(self) -> list:
        backbone_ids = {id(p) for p in self.get_backbone_params()}
        return [p for p in self.parameters() if id(p) not in backbone_ids]

class IRDDriverFusionNet(nn.Module):
    """
    IR + Depth multi-view fusion model (no RGB).

    Modality encoders (shared per modality across all camera views):
      ir_backbone    : EfficientNet-B2 — encodes ir_body, ir_face, ir_hands
      depth_backbone : EfficientNet-B2 — encodes depth_body, depth_face, depth_hands

    Fusion stages:
      1. ir_depth_fuse  : merges IR + Depth feature vectors per camera view
      2. body_hands_fuse: merges body + hands fused features
      3. CrossViewAttention: gated body ↔ face interaction
      4. Task-specific classification heads

    TSM temporal shift is applied before each backbone forward pass.
    """
    PROJ_DIM = 512

    def __init__(self, num_segments: int = NUM_SEGMENTS, dropout: float = DROPOUT):
        super().__init__()
        self.num_segments = num_segments

        # ── Modality-specific backbones ────────────────────────────────────
        # Each backbone is shared across all camera views for its modality.
        self.ir_backbone = timm.create_model(
            "efficientnet_b2", pretrained=False, in_chans=3, num_classes=0, global_pool="avg"
        )
        self.depth_backbone = timm.create_model(
            "efficientnet_b2", pretrained=False, in_chans=3, num_classes=0, global_pool="avg"
        )
        self.feat_dim = self.ir_backbone.num_features

        self._freeze_early_layers(self.ir_backbone)
        self._freeze_early_layers(self.depth_backbone)
        self.tsm = TemporalShift(num_segments, n_div=8)

        # ── Modality fusion: IR + Depth → fused per camera view ───────────
        self.ir_depth_fuse = nn.Sequential(
            nn.Linear(self.feat_dim * 2, self.feat_dim),
            nn.LayerNorm(self.feat_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )

        # ── Spatial fusion: body + hands (after modality fusion) ──────────
        self.body_hands_fuse = nn.Sequential(
            nn.Linear(self.feat_dim * 2, self.feat_dim),
            nn.LayerNorm(self.feat_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )

        # ── Temporal consensus + cross-view attention ──────────────────────
        self.body_consensus = nn.Sequential(
            nn.Linear(self.feat_dim, self.PROJ_DIM),
            nn.BatchNorm1d(self.PROJ_DIM),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.face_consensus = nn.Sequential(
            nn.Linear(self.feat_dim, self.PROJ_DIM),
            nn.BatchNorm1d(self.PROJ_DIM),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.cross_attn = CrossViewAttention(self.PROJ_DIM, proj_dim=128)
        self.task_heads = nn.ModuleDict({
            task: TaskHead(self.PROJ_DIM, n_cls, dropout)
            for task, n_cls in TASK_N_CLASSES.items()
        })

    @staticmethod
    def _freeze_early_layers(backbone: nn.Module):
        for name, param in backbone.named_parameters():
            if (name.startswith("conv_stem") or name.startswith("bn1")
                or name.startswith("blocks.0.") or name.startswith("blocks.1.")):
                param.requires_grad = False

    def _backbone_forward(self, backbone: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if self.training and torch.is_grad_enabled():
            x = x.requires_grad_(True)
            return cp.checkpoint(backbone, x, use_reentrant=False)
        return backbone(x)

    def _extract_features(self, backbone: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """(B, N, C, H, W) → (B, N, feat_dim) with TSM applied before backbone."""
        B, N, C, H, W = x.shape
        flat = x.reshape(B * N, C, H, W)
        flat = self.tsm(flat)
        feat = self._backbone_forward(backbone, flat)
        return feat.view(B, N, -1)

    @staticmethod
    def _temporal_pool(feat: torch.Tensor) -> torch.Tensor:
        return F.adaptive_max_pool1d(feat.permute(0, 2, 1), 1).squeeze(-1)

    def forward(self, streams: dict) -> dict:
        # ── Step 1: encode each modality × camera with the shared backbone ──
        ir_body_feat    = self._extract_features(self.ir_backbone,    streams["ir_body"])
        ir_hands_feat   = self._extract_features(self.ir_backbone,    streams["ir_hands"])
        ir_face_feat    = self._extract_features(self.ir_backbone,    streams["ir_face"])

        depth_body_feat  = self._extract_features(self.depth_backbone, streams["depth_body"])
        depth_hands_feat = self._extract_features(self.depth_backbone, streams["depth_hands"])
        depth_face_feat  = self._extract_features(self.depth_backbone, streams["depth_face"])

        # ── Step 2: fuse IR + Depth per camera view ──────────────────────────
        body_feat  = self.ir_depth_fuse(torch.cat([ir_body_feat,  depth_body_feat],  dim=-1))
        hands_feat = self.ir_depth_fuse(torch.cat([ir_hands_feat, depth_hands_feat], dim=-1))
        face_feat  = self.ir_depth_fuse(torch.cat([ir_face_feat,  depth_face_feat],  dim=-1))

        # ── Step 3: fuse body + hands, then apply cross-view attention ───────
        body_combined = self.body_hands_fuse(torch.cat([body_feat, hands_feat], dim=-1))
        body_pooled   = self.body_consensus(self._temporal_pool(body_combined))
        face_pooled   = self.face_consensus(self._temporal_pool(face_feat))
        body_out, face_out = self.cross_attn(body_pooled, face_pooled)

        # ── Step 4: task-specific heads ──────────────────────────────────────
        return {
            task: self.task_heads[task](
                body_out if TASK_CAMERA[task] == "body" else face_out
            )
            for task in ACTIVE_TASKS
        }

    def get_backbone_params(self) -> list:
        return list(self.ir_backbone.parameters()) + list(self.depth_backbone.parameters())

    def get_head_params(self) -> list:
        backbone_ids = {id(p) for p in self.get_backbone_params()}
        return [p for p in self.parameters() if id(p) not in backbone_ids]