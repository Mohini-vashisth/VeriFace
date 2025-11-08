# VeriFace/models_v2.py
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class FrameBackbone(nn.Module):
    def __init__(self, out_dim=256, backbone_name='resnet18', pretrained=True):
        super().__init__()
        if backbone_name == 'resnet18':
            base = models.resnet18(pretrained=pretrained)
            feat_dim = base.fc.in_features
            base.fc = nn.Identity()
        elif backbone_name == 'resnet50':
            base = models.resnet50(pretrained=pretrained)
            feat_dim = base.fc.in_features
            base.fc = nn.Identity()
        else:
            raise ValueError(backbone_name)
        self.backbone = base
        self.fc = nn.Linear(feat_dim, out_dim)

    def forward(self, frames):
        # frames: (B, N, C, H, W)  -> process per-frame
        B, N, C, H, W = frames.shape
        frames = frames.view(B * N, C, H, W)
        feats = self.backbone(frames)  # (B*N, feat_dim)
        feats = self.fc(feats)         # (B*N, out_dim)
        feats = feats.view(B, N, -1)   # (B, N, out_dim)
        # temporal pooling: mean
        pooled = feats.mean(dim=1)     # (B, out_dim)
        return pooled

class AudioEncoderSmall(nn.Module):
    def __init__(self, n_mels=128, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_mels, 64, 3, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(128, out_dim)

    def forward(self, mel):  # mel: (B, n_mels, T)
        h = self.net(mel).squeeze(-1)  # (B, 128)
        return self.fc(h)

class FusionClassifier(nn.Module):
    def __init__(self, frame_dim=256, audio_dim=256):
        super().__init__()
        self.frame_enc = FrameBackbone(out_dim=frame_dim, backbone_name='resnet18', pretrained=True)
        self.audio_enc = AudioEncoderSmall(n_mels=128, out_dim=audio_dim)
        self.classifier = nn.Sequential(
            nn.Linear(frame_dim + audio_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, frames, mel):
        # frames: (B,N,3,H,W), mel: (B,n_mels,T)
        f = self.frame_enc(frames)
        a = self.audio_enc(mel)
        x = torch.cat([f, a], dim=1)
        logit = self.classifier(x)
        return logit.squeeze(1)