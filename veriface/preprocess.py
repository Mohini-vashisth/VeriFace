import os
import subprocess
import tempfile
from pathlib import Path
import numpy as np

# lazy imports (so module import stays fast)
def _lazy_imports():
    global cv2, sf, librosa
    import cv2
    import soundfile as sf
    import librosa
    return cv2, sf, librosa

def extract_frames(video_path, max_frames=8, target_size=(224,224)):
    """
    Extract up to max_frames uniformly from the video and resize to target_size.
    Returns numpy array shape (max_frames, H, W, 3) float32 normalized [0,1].
    """
    cv2, _, _ = _lazy_imports()
    video_path = str(video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:
        # fallback: try to read until EOF
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
    else:
        indices = np.linspace(0, max(total-1,0), num=min(max_frames, total), dtype=int)
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret:
                continue
            frames.append(frame)
        cap.release()

    # normalize / resize
    out = []
    for f in frames:
        f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        h, w = target_size
        f = cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA)
        out.append(f.astype(np.float32) / 255.0)

    # ensure fixed length
    if len(out) == 0:
        # return zeros if no frames
        return np.zeros((max_frames, target_size[0], target_size[1], 3), dtype=np.float32)
    if len(out) < max_frames:
        # pad by repeating last frame
        last = out[-1]
        while len(out) < max_frames:
            out.append(last.copy())
    out = np.stack(out[:max_frames], axis=0)
    return out

import tempfile
import subprocess
import numpy as np
import os
import warnings
import librosa
from typing import Optional

def _has_audio_stream(path: str) -> bool:
    """Return True if ffprobe sees an audio stream in the file."""
    try:
        res = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a",
             "-show_entries", "stream=codec_type",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            capture_output=True, text=True, check=False)
        return bool(res.stdout.strip())
    except FileNotFoundError:
        # ffprobe not available — be conservative and attempt normal load
        return True

def _file_duration_seconds(path: str) -> float:
    """Return duration in seconds (float) using ffprobe, or 1.0 if unknown."""
    try:
        res = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            capture_output=True, text=True, check=True)
        s = res.stdout.strip()
        return float(s) if s else 1.0
    except Exception:
        return 1.0

def extract_mel_spectrogram(video_path: str,
                            sr: int = 16000,
                            n_mels: int = 128,
                            n_fft: int = 1024,
                            hop_length: int = 256,
                            duration: Optional[float] = None) -> np.ndarray:
    """
    Robustly extract a mel-spectrogram from a video file.
    - If the file has no audio stream, returns zeros with shape (n_mels, T)
      where T is estimated from the video's duration.
    - Tries librosa.load first, then falls back to ffmpeg -> temp WAV and soundfile/scipy/librosa.
    """
    # 1) Quick audio-stream check
    if not _has_audio_stream(video_path):
        # estimate number of frames in mel-time axis from video duration
        dur = duration if duration is not None else _file_duration_seconds(video_path)
        T = max(1, int(np.ceil((dur * sr) / hop_length)))
        return np.zeros((n_mels, T), dtype=np.float32)

    # 2) Try direct load (librosa / soundfile)
    try:
        audio, file_sr = librosa.load(video_path, sr=sr, mono=True, duration=duration)
    except Exception as e:
        warnings.warn(f"PySoundFile/librosa.load failed; falling back to ffmpeg→wav ({e})")
        tmpf = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmpf = tmp.name
            cmd = ["ffmpeg", "-y", "-i", video_path, "-ar", str(sr), "-ac", "1", "-vn", tmpf]
            try:
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            except subprocess.CalledProcessError as cpe:
                # If ffmpeg failed, maybe the file truly had no audio or ffmpeg couldn't decode:
                # estimate duration and return zeros
                dur = duration if duration is not None else _file_duration_seconds(video_path)
                T = max(1, int(np.ceil((dur * sr) / hop_length)))
                return np.zeros((n_mels, T), dtype=np.float32)

            # Try reading the WAV with soundfile, then scipy, then librosa
            try:
                import soundfile as sf
                audio, file_sr = sf.read(tmpf, dtype="float32")
                if audio.ndim > 1:
                    audio = np.mean(audio, axis=1)
                if file_sr != sr:
                    audio = librosa.resample(audio, file_sr, sr)
                    file_sr = sr
            except Exception:
                try:
                    from scipy.io import wavfile
                    file_sr, audio = wavfile.read(tmpf)
                    # normalize integer types
                    if audio.dtype == np.int16:
                        audio = audio.astype("float32") / 32768.0
                    elif audio.dtype == np.int32:
                        audio = audio.astype("float32") / 2147483648.0
                    elif audio.dtype == np.uint8:
                        audio = (audio.astype("float32") - 128.0) / 128.0
                    if audio.ndim > 1:
                        audio = np.mean(audio, axis=1)
                    if file_sr != sr:
                        audio = librosa.resample(audio, file_sr, sr)
                        file_sr = sr
                except Exception:
                    audio, file_sr = librosa.load(tmpf, sr=sr, mono=True, duration=duration)
        finally:
            if tmpf and os.path.exists(tmpf):
                try:
                    os.remove(tmpf)
                except Exception:
                    pass

    # Normalize to float32 and ensure mono
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    # Compute mel spectrogram (power) -> dB
    S = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=n_fft,
                                       hop_length=hop_length,
                                       n_mels=n_mels, power=2.0)
    mel_db = librosa.power_to_db(S, ref=np.max).astype(np.float32)
    return mel_db

# small helper for batch formatting (optional)
def frames_to_tensor(frames):
    """
    Convert frames numpy (N,H,W,3) float32 -> torch tensor (N,3,H,W) normalized.
    """
    import torch
    x = np.transpose(frames, (0,3,1,2))
    return torch.from_numpy(x).float()
