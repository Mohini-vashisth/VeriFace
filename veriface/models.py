# VeriFace/models.py
import os
from typing import Any, Optional

def _safe_import_torch():
    try:
        import torch
        return torch
    except Exception:
        return None

def _safe_import_tf():
    try:
        import tensorflow as tf
        return tf
    except Exception:
        return None

class ModelLoader:
    """
    Minimal loader wrapper.
    - If you add real model artifacts under `models/` this will load them.
    - Uses lazy imports so missing heavy libs won't break the app.
    """
    def __init__(self, model_dir: str = "models"):
        self.model_dir = model_dir

    def load_pytorch(self, filename: str, model_class: Optional[Any] = None, map_location='cpu'):
        torch = _safe_import_torch()
        path = os.path.join(self.model_dir, filename)
        if os.path.exists(path):
            if torch is None:
                raise RuntimeError("torch is required to load PyTorch models but is not installed")
            if model_class is None:
                return torch.jit.load(path, map_location=map_location)
            else:
                model = model_class()
                state = torch.load(path, map_location=map_location)
                model.load_state_dict(state)
                model.eval()
                return model
        else:
            return None  # caller handles absence

    def load_tf(self, foldername: str):
        tf = _safe_import_tf()
        path = os.path.join(self.model_dir, foldername)
        if os.path.exists(path):
            if tf is None:
                raise RuntimeError("tensorflow is required to load TF models but is not installed")
            return tf.keras.models.load_model(path)
        else:
            return None
