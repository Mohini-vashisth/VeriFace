# tests/test_preprocess.py
import os
import subprocess
from VeriFace.preprocess import extract_frames, extract_mel_spectrogram

SAMPLE = "tests/sample_test_video.mp4"

def make_sample_video(path):
    """Create a 2-second sample video (color bars + tone) with ffmpeg."""
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "smptebars=size=320x240:rate=25",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
        "-c:v", "libx264", "-t", "2", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-shortest", path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

if __name__ == "__main__":
    if not os.path.exists("tests"):
        os.makedirs("tests", exist_ok=True)
    print("Generating test video...")
    make_sample_video(SAMPLE)
    print("Extracting frames...")
    frames = extract_frames(SAMPLE, max_frames=8)
    print("Frames extracted:", len(frames))
    if frames:
        print("Frame shape:", frames[0].shape, "dtype:", frames[0].dtype)
    print("Extracting mel spectrogram...")
    mel = extract_mel_spectrogram(SAMPLE)
    print("Mel shape:", mel.shape, "dtype:", mel.dtype)
    print("✅ Test preprocess done.")
