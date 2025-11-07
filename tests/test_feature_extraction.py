import os
import numpy as np
import tempfile
from VeriFace.feature_extraction import extract_frames, extract_mel_spectrogram

def test_extract_frames():
    # Create a 1-second dummy video with ffmpeg (solid color)
    with tempfile.TemporaryDirectory() as td:
        video_path = os.path.join(td, "test.mp4")
        os.system(f"ffmpeg -y -f lavfi -i color=c=blue:s=64x64:d=1 -vf fps=4 {video_path} > /dev/null 2>&1")
        arr = extract_frames(video_path, num_frames=4, resize=(32,32))
        assert arr.shape == (4,32,32,3)
        assert np.all(arr >= 0) and np.all(arr <= 1)

def test_extract_mel_spectrogram():
    # Create a dummy sine wave audio
    import soundfile as sf
    import numpy as np
    import math
    sr = 22050
    t = np.linspace(0,1,sr,False)
    y = 0.5*np.sin(2*math.pi*440*t)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, y, sr)
        spec = extract_mel_spectrogram(tmp.name)
        assert isinstance(spec, np.ndarray)
        assert spec.ndim == 2
        assert spec.shape[0] == 128
        os.remove(tmp.name)

if __name__ == "__main__":
    test_extract_frames()
    test_extract_mel_spectrogram()
    print("✅ All feature_extraction tests passed.")
