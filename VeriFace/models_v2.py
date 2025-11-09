"""
VeriFace/models_v2.py

FusionClassifier: processes a short sequence of frames + mel spectrogram and
returns a single logit per example (shape (B,1)). Also exposes build_model()
and ModelLoader for compatibility.

FrameEncoder prefers torchvision's ResNet-18 backbone (pretrained optional).
If torchvision/resnet18 isn't available, we fall back to a small custom CNN.
"""
from typing import Optional
import os
import torch
import torch.nn as nn

# Try to import resnet18 from torchvision; if it's not available we'll use fallback.
try:
    from torchvision.models import resnet18
    _HAS_TORCHVISION = True
except Exception:
    _HAS_TORCHVISION = False


class FrameEncoder(nn.Module):
    """Per-frame encoder.
    Accepts input shape (B*N, 3, H, W) and returns (B*N, out_dim).
    Uses ResNet-18 (without final fc) when available.
    """
    def __init__(self, out_dim: int = 128, pretrained: bool = False):
        super().__init__()
        self.out_dim = out_dim

        if _HAS_TORCHVISION:
            # Use resnet18 backbone (drop the final FC).
            backbone = resnet18(pretrained=pretrained)
            modules = list(backbone.children())[:-1]  # drop the final fc
            self.backbone = nn.Sequential(*modules)   # outputs (B, 512, 1, 1)
            self.proj = nn.Linear(512, out_dim)
        else:
            # Fallback small CNN (keeps API stable)
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 32, 3, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.proj = nn.Linear(64, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B*N, 3, H, W)
        h = self.backbone(x)          # (B*N, C, 1, 1)
        h = h.view(h.size(0), -1)     # (B*N, C)
        return self.proj(h)           # (B*N, out_dim)


class AudioEncoder(nn.Module):
    """Simple 1D CNN for Mel spectrograms: input (B, n_mels, T) -> (B, out_dim)."""
    def __init__(self, n_mels: int = 128, out_dim: int = 128):
        super().__init__()
        # Conv1d expects (B, channels=n_mels, T)
        self.conv = nn.Sequential(
            nn.Conv1d(n_mels, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(128, out_dim)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        # mel: (B, n_mels, T)
        h = self.conv(mel)            # (B, 128, 1)
        h = h.view(h.size(0), -1)     # (B, 128)
        return self.fc(h)             # (B, out_dim)


class FusionClassifier(nn.Module):
    """
    Combines FrameEncoder and AudioEncoder.
    frames: (B, N, 3, H, W)
    mel:    (B, n_mels, T)
    returns logits: (B, 1)
    """
    def __init__(self, embed_dim: int = 128, pretrained_frame_backbone: bool = False):
        super().__init__()
        self.frame_enc = FrameEncoder(out_dim=embed_dim, pretrained=pretrained_frame_backbone)
        self.audio_enc = AudioEncoder(out_dim=embed_dim)
        self.fc = nn.Sequential(
            nn.Linear(embed_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1),  # single logit per example (use BCEWithLogitsLoss)
        )

    def forward(self, frames: torch.Tensor, mel: torch.Tensor) -> torch.Tensor:
        # frames: (B, N, 3, H, W), mel: (B, n_mels, T)
        B, N, C, H, W = frames.shape
        # merge batch and frames to run through frame encoder
        frames_ = frames.view(B * N, C, H, W)
        f_emb = self.frame_enc(frames_)          # (B*N, embed)
        f_emb = f_emb.view(B, N, -1).mean(dim=1) # (B, embed) - temporal avg
        a_emb = self.audio_enc(mel)              # (B, embed)
        fused = torch.cat([f_emb, a_emb], dim=1) # (B, 2*embed)
        logits = self.fc(fused)                  # (B,1)
        return logits


def build_model(device: Optional[torch.device] = None,
                embed_dim: int = 128,
                pretrained_frame_backbone: bool = False) -> torch.nn.Module:
    model = FusionClassifier(embed_dim=embed_dim,
                             pretrained_frame_backbone=pretrained_frame_backbone)
    if device is not None:
        model.to(device)
    return model


class ModelLoader:
    """Simple loader for saving/loading state_dicts."""
    def __init__(self, model_cls=FusionClassifier):
        self.model_cls = model_cls

    def create(self, device: Optional[torch.device] = None) -> torch.nn.Module:
        return build_model(device)

    def save(self, model: torch.nn.Module, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(model.state_dict(), path)

    def load(self, path: str, device: Optional[torch.device] = None) -> torch.nn.Module:
        device = device or torch.device("cpu")
        model = self.model_cls()
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device)
        model.eval()
        return model