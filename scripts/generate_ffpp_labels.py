#!/usr/bin/env python3
import csv, os
from pathlib import Path

ROOT = Path("data/ffpp_c23")
OUT = ROOT / "labels.csv"

FAKE_FOLDERS = ["DeepFakeDetection", "Deepfakes", "Face2Face", "FaceShifter", "FaceSwap", "NeuralTextures"]
REAL_FOLDERS = ["original"]

def collect_videos():
    rows = []
    for folder in FAKE_FOLDERS + REAL_FOLDERS:
        dir_path = ROOT / folder
        if not dir_path.exists():
            continue
        label = 1 if folder in FAKE_FOLDERS else 0
        for path in dir_path.rglob("*.mp4"):
            rows.append((str(path), label))
    return rows

if __name__ == "__main__":
    videos = collect_videos()
    print(f"Found {len(videos)} videos total.")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["video_path", "label"])
        for v, l in videos:
            writer.writerow([v, l])
    print(f"Wrote {len(videos)} entries to {OUT}")
