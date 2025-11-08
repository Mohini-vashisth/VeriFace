# train/train_video.py
"""
Train script for VeriFace (video + audio fusion).
Usage:
  python train/train_video.py --csv data/labels.csv --outdir checkpoints/v2 --epochs 20 --batch-size 8 --frames 8
"""

import argparse
import os
import random
import time
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# try to import your feature extraction helpers
try:
    from VeriFace.feature_extraction import extract_frames, extract_mel_spectrogram
except Exception:
    extract_frames = None
    extract_mel_spectrogram = None

# import the new model (models_v2) if present, otherwise fallback to models.py
try:
    from VeriFace.models_v2 import FusionClassifier as ModelClass
except Exception:
    try:
        from VeriFace.models import DeepfakeClassifier as ModelClass
    except Exception:
        ModelClass = None


# -----------------------
# Utilities
# -----------------------
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_csv(path: str) -> List[Tuple[str, int]]:
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 2:
                continue
            video_path = parts[0].strip()
            label = int(parts[1].strip())
            rows.append((video_path, label))
    return rows


# -----------------------
# Dataset
# -----------------------
class VideoAudioDataset(Dataset):
    def __init__(self, rows: List[Tuple[str, int]], frames: int = 8, frame_size: int = 224,
                 n_mels: int = 128, transform=None, cache_audio: bool = True):
        self.rows = rows
        self.frames = frames
        self.frame_size = frame_size
        self.n_mels = n_mels
        self.transform = transform
        self.cache_audio = cache_audio
        self._audio_cache_dir = Path(".audio_cache")
        if self.cache_audio:
            self._audio_cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self):
        return len(self.rows)

    def _extract_audio_cached(self, video_path: str) -> str:
        # cache WAV by hashing path name
        import hashlib
        h = hashlib.md5(video_path.encode()).hexdigest()
        wav_path = str(self._audio_cache_dir / f"{h}.wav")
        if not os.path.exists(wav_path):
            # use ffmpeg to extract wav
            cmd = ["ffmpeg", "-y", "-i", video_path, "-vn", "-ar", "22050", "-ac", "1", wav_path]
            try:
                import subprocess
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception:
                # if ffmpeg fails, try fallback using extract_mel_spectrogram directly on file
                wav_path = video_path
        return wav_path

    def __getitem__(self, idx):
        video_path, label = self.rows[idx]
        # frames: (N, H, W, 3) float32 [0..1]
        frames_np = None
        if extract_frames is not None:
            try:
                frames_np = extract_frames(video_path, num_frames=self.frames, resize=(self.frame_size, self.frame_size))
            except Exception:
                frames_np = None

        # fallback: simple OpenCV sampling if extract_frames isn't available or failed
        if frames_np is None:
            import cv2
            cap = cv2.VideoCapture(video_path)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
            idxs = np.linspace(0, total - 1, self.frames).astype(int).tolist()
            imgs = []
            for i in idxs:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
                ok, frame = cap.read()
                if not ok or frame is None:
                    # repeat last frame if missing
                    if len(imgs) > 0:
                        imgs.append(imgs[-1].copy())
                    else:
                        imgs.append(np.zeros((self.frame_size, self.frame_size, 3), dtype=np.float32))
                    continue
                frame = cv2.resize(frame, (self.frame_size, self.frame_size))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                imgs.append(frame)
            cap.release()
            frames_np = np.stack(imgs[:self.frames], axis=0).astype(np.float32)

        # frames to torch (N, C, H, W) later we'll reshape to (C, N, H, W) or model expects (B, N, C, H, W)
        frames_t = torch.from_numpy(frames_np).permute(0, 3, 1, 2).contiguous()  # (N,C,H,W)

        # audio / mel
        audio_source = video_path
        if self.cache_audio:
            audio_source = self._extract_audio_cached(video_path)
        mel = None
        if extract_mel_spectrogram is not None:
            try:
                mel_np = extract_mel_spectrogram(audio_source, sr=22050, n_mels=self.n_mels)
                # ensure shape (n_mels, T)
                mel = torch.from_numpy(mel_np).float()
            except Exception:
                mel = None
        if mel is None:
            # fallback: create zeros
            mel = torch.zeros(self.n_mels, max(10, self.frames*4), dtype=torch.float32)

        sample = {
            "frames": frames_t,  # (N,C,H,W)
            "mel": mel,          # (n_mels, T)
            "label": torch.tensor(label, dtype=torch.float32)
        }
        return sample


# -----------------------
# Collate fn
# -----------------------
def collate_fn(batch):
    # batch: list of dicts
    frames = [b["frames"] for b in batch]
    # stack as (B, N, C, H, W)
    frames = torch.stack(frames, dim=0)
    mels = [b["mel"] for b in batch]
    # pad mels to same T
    maxT = max(m.shape[1] for m in mels)
    mels_padded = []
    for m in mels:
        if m.shape[1] < maxT:
            pad = torch.zeros(m.shape[0], maxT - m.shape[1])
            mels_padded.append(torch.cat([m, pad], dim=1))
        else:
            mels_padded.append(m[:, :maxT])
    mels = torch.stack(mels_padded, dim=0)  # (B, n_mels, T)
    labels = torch.stack([b["label"] for b in batch], dim=0)
    return frames, mels, labels


# -----------------------
# Train / Eval
# -----------------------
def train_epoch(model, loader, optimizer, device):
    model.train()
    losses = []
    all_scores = []
    all_labels = []
    criterion = nn.BCEWithLogitsLoss()
    for frames, mels, labels in tqdm(loader, desc="train", leave=False):
        # frames: (B,N,C,H,W) -> our model (models_v2) expects this shape
        frames = frames.to(device)
        mels = mels.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(frames, mels)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_scores.extend(probs.tolist())
        all_labels.extend(labels.detach().cpu().numpy().tolist())

    avg_loss = float(np.mean(losses)) if losses else 0.0
    auc = roc_auc_score(all_labels, all_scores) if len(set(all_labels)) > 1 else 0.5
    return avg_loss, auc


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    losses = []
    all_scores = []
    all_labels = []
    criterion = nn.BCEWithLogitsLoss()
    for frames, mels, labels in tqdm(loader, desc="eval", leave=False):
        frames = frames.to(device)
        mels = mels.to(device)
        labels = labels.to(device)
        logits = model(frames, mels)
        loss = criterion(logits, labels)
        losses.append(loss.item())
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_scores.extend(probs.tolist())
        all_labels.extend(labels.detach().cpu().numpy().tolist())

    avg_loss = float(np.mean(losses)) if losses else 0.0
    auc = roc_auc_score(all_labels, all_scores) if len(set(all_labels)) > 1 else 0.5
    return avg_loss, auc


# -----------------------
# Main
# -----------------------
def main(args):
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu")
    print("Device:", device)

    rows = read_csv(args.csv)
    if len(rows) == 0:
        raise RuntimeError("No data entries found in CSV")

    # split simple (shuffle then split)
    random.shuffle(rows)
    n_val = max(1, int(len(rows) * args.val_split))
    val_rows = rows[:n_val]
    train_rows = rows[n_val:]

    train_ds = VideoAudioDataset(train_rows, frames=args.frames, frame_size=args.frame_size, n_mels=args.n_mels,
                                 cache_audio=not args.no_audio_cache)
    val_ds = VideoAudioDataset(val_rows, frames=args.frames, frame_size=args.frame_size, n_mels=args.n_mels,
                               cache_audio=not args.no_audio_cache)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                              collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                            collate_fn=collate_fn, pin_memory=True)

    # model
    if ModelClass is None:
        raise RuntimeError("No model class found. Ensure VeriFace/models_v2.py or VeriFace/models.py exists.")
    model = ModelClass().to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_auc = -1.0
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss, train_auc = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_auc = eval_epoch(model, val_loader, device)
        print(f" train_loss: {train_loss:.4f} train_auc: {train_auc:.4f}")
        print(f"  val_loss: {val_loss:.4f}  val_auc: {val_auc:.4f}")

        # save checkpoint
        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
            "val_auc": val_auc
        }
        torch.save(ckpt, outdir / f"ckpt_epoch{epoch}.pth")

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(ckpt, outdir / "best.pth")
            print("  -> new best saved")

        scheduler.step()

        # fast debug exit
        if args.debug:
            break

    elapsed = time.time() - start_time
    print("Training finished in %.2f min" % (elapsed / 60.0))
    print("Best val AUC:", best_auc)


# -----------------------
# CLI
# -----------------------
def get_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, required=True, help="CSV with video_path,label")
    p.add_argument("--outdir", type=str, default="checkpoints/v2")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--frames", type=int, default=8)
    p.add_argument("--frame-size", type=int, default=224)
    p.add_argument("--n-mels", type=int, default=128)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--val-split", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--debug", action="store_true")
    p.add_argument("--force-cpu", dest="force_cpu", action="store_true")
    p.add_argument("--no-audio-cache", dest="no_audio_cache", action="store_true")
    return p


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)