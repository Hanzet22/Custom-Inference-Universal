# detector.py
from pathlib import Path  # <-- FIX: Tambahkan ini
import json
import os
import re
import zipfile  # <-- Tambahkan ini juga (dipakai di inspect_zip_contents)
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

try:
    import cv2
except:
    cv2 = None
try:
    import h5py
except:
    h5py = None
try:
    import onnx
except:
    onnx = None
try:
    import tensorflow as tf
except:
    tf = None
try:
    import coremltools as ct
except:
    ct = None
try:
    from openvino.runtime import Core as OVCore
except:
    OVCore = None
try:
    from safetensors.torch import load_file as safetensors_load
except:
    safetensors_load = None

from config import MODEL_DIR, INPUT_DIR, RESULT_DIR
from utils import detect_extension, SUPPORTED_EXTENSIONS, is_zipfile

@dataclass
class ModelReport:
    file_name: str
    path: str
    ext: str
    kind: str
    runtime_hint: str
    family_hint: str
    task_hints: List[str] = field(default_factory=list)
    media_hints: List[str] = field(default_factory=list)
    problem_hints: List[str] = field(default_factory=list)
    tensor_count: int = 0
    param_count: int = 0
    top_keys: List[str] = field(default_factory=list)
    shapes: List[Tuple[str, Tuple[int, ...]]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

def ensure_dirs():
    MODEL_DIR.mkdir(exist_ok=True)
    INPUT_DIR.mkdir(exist_ok=True)
    RESULT_DIR.mkdir(exist_ok=True)

def scan_models(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    items = []
    for p in sorted(folder.iterdir()):
        if p.is_file() and detect_extension(p) in SUPPORTED_EXTENSIONS:
            items.append(p)
    return items

def strip_common_prefixes(state_dict: Dict[str, Any]) -> OrderedDict:
    prefixes = ("module.", "model.", "net.", "net_g.", "network_g.", "generator.", "ema.")
    cleaned = OrderedDict()
    for k, v in state_dict.items():
        new_k = k
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if new_k.startswith(prefix):
                    new_k = new_k[len(prefix):]
                    changed = True
        cleaned[new_k] = v
    return cleaned

def is_tensor_dict(obj: Any) -> bool:
    return isinstance(obj, dict) and len(obj) > 0 and all(torch.is_tensor(v) for v in obj.values())

def unwrap_torch_payload(obj: Any) -> Tuple[Any, List[str]]:
    top_keys = []
    if isinstance(obj, dict):
        top_keys = list(obj.keys())
        for key in ("params_ema", "params", "state_dict", "model", "net", "network_g", "generator"):
            if key in obj and isinstance(obj[key], dict):
                return obj[key], top_keys
        return obj, top_keys
    return obj, top_keys

def count_params(state_dict: Dict[str, Any]) -> int:
    total = 0
    for v in state_dict.values():
        if torch.is_tensor(v):
            total += int(v.numel())
    return total

def collect_shapes(state_dict: Dict[str, Any], limit: int = 20) -> List[Tuple[str, Tuple[int, ...]]]:
    out = []
    for k, v in state_dict.items():
        if torch.is_tensor(v):
            out.append((k, tuple(v.shape)))
        if len(out) >= limit:
            break
    return out

def family_by_keys(keys: Iterable[str]) -> str:
    keys = [k.lower() for k in keys]
    def has(*parts: str) -> bool:
        return any(any(p in k for p in parts) for k in keys)
    if has("gfpgan", "codeformer", "face", "facial", "landmark", "mouth", "eye"):
        return "Face restoration"
    if has("rrdb", "conv_first", "body.0", "dense"):
        return "RRDBNet / ESRGAN-like"
    if has("srvgg", "vgg", "body.1", "compact"):
        return "SRVGG / compact image model"
    if has("swin", "window", "relative_position_bias", "attn_mask", "qkv"):
        return "Transformer restoration"
    if has("hat", "omnisr", "dat", "natten", "attention"):
        return "Modern attention-based restoration"
    if has("unet", "encoder", "decoder", "downsample", "upsample"):
        return "Encoder-decoder / U-Net-like"
    if has("lora"):
        return "LoRA adapter"
    if has("ncnn", "param", "bin"):
        return "NCNN-style deployment"
    if has("caffe", "prototxt", "caffemodel"):
        return "Caffe-style deployment"
    return "Unknown / custom architecture"

def runtime_hint_by_ext(ext: str) -> str:
    table = {
        ".safetensors": "safetensors / Hugging Face",
        ".pth": "PyTorch",
        ".pt": "PyTorch / TorchScript / export variant",
        ".ckpt": "PyTorch checkpoint",
        ".bin": "PyTorch / OpenVINO / mixed",
        ".param": "NCNN",
        ".onnx": "ONNX Runtime",
        ".engine": "TensorRT",
        ".trt": "TensorRT",
        ".plan": "TensorRT",
        ".xml": "OpenVINO IR",
        ".h5": "TensorFlow / Keras",
        ".hdf5": "TensorFlow / Keras",
        ".pb": "TensorFlow graph / SavedModel asset",
        ".msgpack": "Framework-specific archive",
        ".mlmodel": "Core ML",
        ".mlpackage": "Core ML",
        ".tflite": "TensorFlow Lite / LiteRT",
        ".json": "Metadata / config",
        ".gguf": "GGUF runtime (LLM inference)",
        ".ggml": "Legacy GGML-family runtime",
        ".ggmf": "Legacy GGML-family runtime",
        ".ggjt": "Legacy GGML-family runtime",
        ".npy": "NumPy",
        ".npz": "NumPy",
        ".keras": "Keras v3",
        ".tfrecords": "TensorFlow input pipeline",
        ".mar": "TorchServe",
        ".pte": "ExecuTorch",
        ".pt2": "PyTorch export / graph artifact",
        ".ptl": "PyTorch Lightning / checkpoint variant",
        ".pkl": "Pickle object",
        ".nc": "Ambiguous / deployment-specific",
        ".mleap": "MLeap",
        ".coreml": "Core ML alias",
        ".surml": "Unknown alias / custom",
        ".llamafile": "Llama.cpp-family packaged runtime",
        ".caffemodel": "Caffe",
        ".prototxt": "Caffe",
        ".dlc": "SNPE / deployment artifact",
        ".model": "Generic / ambiguous",
        ".exl2": "ExLlama quantized format",
        ".lora": "LoRA adapter",
        ".prompt": "Text prompt / metadata",
        ".weights": "Generic weights",
        ".gptq": "GPTQ quantized",
        ".awq": "AWQ quantized",
        ".vae": "VAE component",
        ".unet": "UNet component",
        ".embedding": "Embedding",
        ".hypernetwork": "Hypernetwork",
        ".pbtxt": "TensorFlow text proto",
        ".index": "Index file",
    }
    return table.get(ext, "Unknown")

def kind_by_ext(ext: str) -> str:
    if ext in {".json", ".prompt", ".pbtxt"}:
        return "config_or_metadata"
    if ext in {".lora", ".embedding", ".hypernetwork"}:
        return "adapter"
    if ext in {".npy", ".npz"}:
        return "array_container"
    if ext in {".gguf", ".ggml", ".ggmf", ".ggjt", ".llamafile"}:
        return "llm_container"
    if ext in {".engine", ".trt", ".plan", ".xml", ".mlmodel", ".mlpackage", ".tflite", ".pte", ".mar", ".coreml", ".dlc"}:
        return "runtime_artifact"
    if ext in {".prototxt"}:
        return "graph_definition"
    if ext in {".caffemodel", ".pth", ".pt", ".ckpt", ".bin", ".safetensors", ".param", ".h5", ".hdf5", ".keras", ".pb", ".pt2", ".ptl", ".weights", ".vae", ".unet"}:
        return "weight_or_graph_model"
    return "unknown"

def keyword_task_hints(text: str) -> List[str]:
    t = text.lower()
    hints = []
    # ... (isi lengkap seperti asli)
    return list(dict.fromkeys(hints))

def detect_problem_hints(name: str) -> List[str]:
    t = name.lower()
    hints = []
    # ... (isi lengkap)
    return hints

def load_json_meta(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def inspect_numpy(path: Path) -> Tuple[str, int, List[str]]:
    obj = np.load(str(path), allow_pickle=False)
    notes = []
    if isinstance(obj, np.lib.npyio.NpzFile):
        keys = list(obj.files)
        return "npz_archive", len(keys), keys[:20]
    arr = np.asarray(obj)
    notes.append(f"dtype={arr.dtype}, shape={tuple(arr.shape)}")
    return "npy_array", 1, notes

def inspect_caffe_prototxt(path: Path) -> Tuple[List[str], List[str]]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    keys = []
    if "layer" in text.lower():
        keys.append("layer blocks")
    if "input" in text.lower():
        keys.append("input definition")
    if "convolution" in text.lower():
        keys.append("convolution ops")
    if "relu" in text.lower():
        keys.append("activation ops")
    return keys, [f"lines={text.count(os.linesep) + 1}"]

def inspect_zip_contents(path: Path) -> List[str]:
    items = []
    try:
        with zipfile.ZipFile(path, "r") as zf:
            items = zf.namelist()[:20]
    except Exception:
        pass
    return items

# ------------------------------------------------------------------
# Semua fungsi inspect_* (safetensors, torch, onnx, tflite, json, numpy, openvino, caffe, zip, generic)
# Sama persis seperti di inference.py asli.
# ------------------------------------------------------------------
def inspect_file(path: Path, unsafe_torchload: bool = False) -> ModelReport:
    ext = detect_extension(path)
    # ... (isi lengkap, panggil fungsi-fungsi di atas)
    return ModelReport(...)  # placeholder

def print_report(r: ModelReport) -> None:
    print("=" * 78)
    print(f"FILE         : {r.file_name}")
    # ... dst.
