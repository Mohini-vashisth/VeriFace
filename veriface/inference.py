"""
VeriFace: Inference Pipeline
----------------------------
Loads the trained model, runs preprocessing on input video/audio,
and returns deepfake detection probabilities.
"""

import os
import torch
import numpy as np
import cv2
import librosa
from VeriFace.models import build_model
from VeriFace.feature_extraction import extract_frames, extract_mel_spectrogram


class VeriFaceInference:
    def __init__(self, model_path=None, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = build_model(self.device)

        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"✅ Loaded model weights from {model_path}")
        else:
            print("⚠️ No trained weights found — using untrained model for dry-run.")

        self.model.eval()

    def preprocess_video(self, video_path, num_frames=8):
        """Extracts a small batch of frames from a video and preprocesses them."""
        frames = extract_frames(video_path, num_frames=8)
        if not frames.size:
            raise ValueError(f"No frames extracted from {video_path}")
        # Normalize and convert to torch tensor
        frames = torch.tensor(frames.transpose(0, 3, 1, 2), dtype=torch.float32)
        return frames.to(self.device)

    def preprocess_audio(self, audio_path):
        """Extracts Mel spectrogram features from the audio track."""
        mel = extract_mel_spectrogram(audio_path)
        mel = torch.tensor(mel[np.newaxis, :, :], dtype=torch.float32)  # (1, n_mels, T)
        return mel.to(self.device)

    @torch.inference_mode()
    def predict(self, video_path, audio_path):
        """Runs full inference on given video + audio."""
        frames = self.preprocess_video(video_path)
        mel = self.preprocess_audio(audio_path)

        # Only take first frame batch if multiple
        frame_batch = frames[:1]
        mel_batch = mel[:1]

        output = self.model(frame_batch, mel_batch)
        prob = output.item()
        label = "FAKE" if prob > 0.5 else "REAL"

        return {"label": label, "confidence": float(prob)}

    def dry_run(self):
        """Quick test with dummy tensors."""
        dummy_frames = torch.randn(1, 3, 224, 224).to(self.device)
        dummy_mel = torch.randn(1, 128, 64).to(self.device)
        prob = self.model(dummy_frames, dummy_mel).item()
        print(f"Dry run output: {prob:.4f}")


if __name__ == "__main__":
    # Example usage
    infer = VeriFaceInference(model_path="checkpoints/deepfake_model.pt")
    infer.dry_run()

    # Example prediction (replace with actual paths)
    # result = infer.predict("data/sample_video.mp4", "data/sample_audio.wav")
    # print("Inference result:", result)