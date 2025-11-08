# VeriFace/preprocess.py
import os
import tempfile
from typing import List
import numpy as np
import cv2
import subprocess
import librosa

def extract_frames(video_path: str, max_frames: int = 16, target_size=(224,224)) -> List[np.ndarray]:
    """
    Extract up to max_frames evenly spaced RGB frames from video_path.
    Returns list of HxWxC numpy arrays (RGB, float32 normalized to [0,1]).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if total_frames == 0:
        cap.release()
        return []
    step = max(1, total_frames // max_frames)
    frames = []
    idx = 0
    grabbed = 0
    while grabbed < max_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        # convert BGR -> RGB, resize, normalize
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, target_size)
        frame = frame.astype("float32") / 255.0
        frames.append(frame)
        grabbed += 1
        idx += step
    cap.release()
    return frames

def extract_mel_spectrogram(video_path: str, sr: int = 16000, n_mels: int = 128) -> np.ndarray:
    """
    Extract audio from video using ffmpeg -> compute log-mel spectrogram with librosa.
    Returns np.ndarray shaped (n_mels, T).
    """
    tmpwav = None
    try:
        tmpdir = tempfile.gettempdir()
        tmpwav = os.path.join(tmpdir, f"vf_tmp_{os.getpid()}.wav")
        cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-ar", str(sr), "-ac", "1", "-vn", tmpwav
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        y, _ = librosa.load(tmpwav, sr=sr, mono=True)
        if len(y) == 0:
            return np.zeros((n_mels, 0), dtype=np.float32)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
        log_mel = librosa.power_to_db(mel, ref=np.max)
    finally:
        if tmpwav and os.path.exists(tmpwav):
            try:
                os.remove(tmpwav)
            except Exception:
                pass
    return log_mel.astype(np.float32)
