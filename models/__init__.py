import os
import importlib
from os import path as osp
from litsr.utils.registry import ModelRegistry
from litsr import create_model, load_model

# Import all models

__all__ = ["create_model", "load_model", "ModelRegistry"]

model_folder = osp.dirname(osp.abspath(__file__))
for root, dirs, files in os.walk(model_folder):
    for f in files:
        if f.endswith("_model.py"):
            importlib.import_module(f"models.{osp.splitext(f)[0]}")
