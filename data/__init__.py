import importlib
import os
from os import path as osp

from litsr import create_data_module

# Import all datamodules
folder = osp.dirname(osp.abspath(__file__))
for root, dirs, files in os.walk(folder):
    for f in files:
        if f.endswith("_datamodule.py"):
            importlib.import_module(f"data.{osp.splitext(f)[0]}")
