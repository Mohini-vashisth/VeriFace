# VeriFace/feature_extraction.py
"""
Simple, dependency-minimal feature extraction utilities for VeriFace.
Functions:
 - extract_frames(video_path, num_frames=8, resize=(224,224)) -> ndarray (N,H,W,3) float32 [0..1]
 - extract_mel_spectrogram(audio_path, sr=22050, n_mels=128, n_fft=2048, hop_length=512) -> ndarray (n_mels, T) float32
These are small, well-documented building blocks and unit-testable.
"""

from typing import Tuple
import numpy as np
import subprocess
import os
import tempfile
from pathlib import Path

def extract_frames(video_path: str, num_frames: int = 8, resize: Tuple[int,int]=(224,224)) -> np.ndarray:
    """
    Extract evenly spaced frames from a video using ffmpeg (fast, no heavy deps).
    Returns frames as float32 array shape (num_frames, H, W, 3) in range [0,1].
    """
    video_path = str(video_path)
    assert Path(video_path).exists(), f"video not found: {video_path}"
    h, w = resize
    with tempfile.TemporaryDirectory() as td:
        out_pattern = os.path.join(td, "frame-%03d.jpg")
        # Try to extract roughly num_frames using ffmpeg thumbnail or fps trick
        cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-vf", f"scale={w}:{h},thumbnail={num_frames}",
            "-frames:v", str(num_frames),
            out_pattern
        ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            # fallback: extract sequential frames if thumbnail filter fails
            cmd2 = [
                "ffmpeg", "-y", "-i", video_path,
                "-vf", f"scale={w}:{h},fps=1",
                "-frames:v", str(num_frames),
                out_pattern
            ]
            subprocess.run(cmd2, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        imgs = []
        files = sorted([p for p in Path(td).glob("frame-*.jpg")])
        for f in files[:num_frames]:
            from PIL import Image
            im = Image.open(f).convert("RGB").resize((w,h))
            arr = np.asarray(im, dtype=np.float32) / 255.0
            imgs.append(arr)
        if len(imgs) == 0:
            raise RuntimeError("No frames extracted")
        # pad if fewer frames than requested
        while len(imgs) < num_frames:
            imgs.append(imgs[-1].copy())
        return np.stack(imgs[:num_frames], axis=0).astype(np.float32)


def extract_mel_spectrogram(audio_path: str, sr: int = 22050, n_mels: int = 128, n_fft: int = 2048, hop_length: int = 512) -> np.ndarray:
    """
    Use librosa if present; otherwise fall back to a scipy+soundfile approach via ffmpeg.
    Returns (n_mels, T) float32 (log-scaled dB).
    """
    audio_path = str(audio_path)
    assert Path(audio_path).exists(), f"audio not found: {audio_path}"
    try:
        import librosa
        y, _ = librosa.load(audio_path, sr=sr, mono=True)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
        S_db = librosa.power_to_db(S, ref=np.max)
        return S_db.astype(np.float32)
    except Exception:
        # fallback: call ffmpeg to produce a WAV and use scipy/soundfile
        import tempfile
        from scipy import signal
        import soundfile as sf

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpwav:
            tmpname = tmpwav.name
        try:
            subprocess.run(["ffmpeg", "-y", "-i", audio_path, "-ar", str(sr), "-ac", "1", tmpname],
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            y, sr_read = sf.read(tmpname, dtype='float32')
            # compute STFT magnitude
            f, t, Zxx = signal.stft(y, fs=sr_read, nperseg=n_fft, noverlap=n_fft-hop_length, boundary=None)
            S = np.abs(Zxx)
            # coarse mel-like pooling: uniform bins
            S_mel = np.zeros((n_mels, S.shape[1]), dtype=np.float32)
            bins = np.array_split(np.arange(S.shape[0]), n_mels)
            for i, b in enumerate(bins):
                if len(b) == 0:
                    continue
                S_mel[i] = S[b].mean(axis=0)
            # log-scale (dB)
            S_db = 20 * np.log10(np.maximum(S_mel, 1e-10))
            return S_db.astype(np.float32)
        finally:
            try:
                os.remove(tmpname)
            except Exception:
                pass
