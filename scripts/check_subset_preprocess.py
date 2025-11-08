#!/usr/bin/env python3
import os
import random
from VeriFace.preprocess import extract_frames, extract_mel_spectrogram

subset_root = "data/ffpp_c23_subset"
real_dir = os.path.join(subset_root, "real")
fake_dir = os.path.join(subset_root, "fake")

real_samples = os.listdir(real_dir)
fake_samples = os.listdir(fake_dir)

print(f"🧾 Total: real={len(real_samples)}, fake={len(fake_samples)}")

# pick one random real and one fake
real_sample = os.path.join(real_dir, random.choice(real_samples))
fake_sample = os.path.join(fake_dir, random.choice(fake_samples))

print("Testing on:", real_sample)
frames = extract_frames(real_sample, max_frames=8)
mel = extract_mel_spectrogram(real_sample)

print(f"Frames shape: {frames.shape}, dtype={frames.dtype}")
print(f"Mel shape: {mel.shape}, dtype={mel.dtype}")

print("✅ Preprocess sanity check complete.")
