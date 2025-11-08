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

def extract_mel_spectrogram(video_path, sr=16000, n_mels=128, n_fft=2048, hop_length=512, T_fixed=64):
    """
    Robust mel spectrogram extraction:
    1) Try using ffmpeg to dump audio to a temp wav and read with soundfile.
    2) If ffmpeg fails, try librosa.load(video_path).
    3) If everything fails or no audio stream, return zeros shape (n_mels, T_fixed).
    Returns numpy array (n_mels, T_fixed) float32 (dB scaled).
    """
    cv2, sf, librosa = _lazy_imports()
    video_path = str(video_path)

    tmp_wav = None
    audio = None
    file_sr = None

    # Try ffmpeg -> temporary wav
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_wav = tmp.name
        cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-ar", str(sr), "-ac", "1", "-vn", tmp_wav
        ]
        # run and allow errors to surface in stderr if needed (but we swallow)
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        # load audio
        audio, file_sr = sf.read(tmp_wav, dtype='float32')
        if audio is None:
            raise RuntimeError("No audio read from temp wav")
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
    except Exception:
        # cleanup temp if created
        try:
            if tmp_wav and os.path.exists(tmp_wav):
                os.remove(tmp_wav)
        except Exception:
            pass
        # fallback to librosa load (it sometimes can read odd containers)
        try:
            audio, file_sr = librosa.load(video_path, sr=sr, mono=True, duration=None)
        except Exception:
            # give safe default zeros
            return np.zeros((n_mels, T_fixed), dtype=np.float32)

    # if tmp_wav exists, remove it
    try:
        if tmp_wav and os.path.exists(tmp_wav):
            os.remove(tmp_wav)
    except Exception:
        pass

    # ensure sampling rate
    try:
        if file_sr is None:
            file_sr = sr
    except Exception:
        file_sr = sr

    # if audio length is too small, we still compute mel but will pad/truncate later
    try:
        S = librosa.feature.melspectrogram(y=audio.astype(float), sr=file_sr, n_fft=n_fft,
                                           hop_length=hop_length, n_mels=n_mels)
        S_db = librosa.power_to_db(S, ref=np.max)
    except Exception:
        return np.zeros((n_mels, T_fixed), dtype=np.float32)

    # pad or truncate time axis to T_fixed
    if S_db.shape[1] < T_fixed:
        pad_width = T_fixed - S_db.shape[1]
        S_db = np.pad(S_db, ((0,0),(0,pad_width)), mode='constant', constant_values=(S_db.min(),))
    elif S_db.shape[1] > T_fixed:
        S_db = S_db[:, :T_fixed]

    return S_db.astype(np.float32)

# small helper for batch formatting (optional)
def frames_to_tensor(frames):
    """
    Convert frames numpy (N,H,W,3) float32 -> torch tensor (N,3,H,W) normalized.
    """
    import torch
    x = np.transpose(frames, (0,3,1,2))
    return torch.from_numpy(x).float()
