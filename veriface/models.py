# VeriFace/models.py — stable API that forwards to models_v2
from typing import Optional

# forward to the implementation in models_v2
from .models_v2 import build_model, ModelLoader

__all__ = ["build_model", "ModelLoader"]
