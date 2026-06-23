# loader.py
import torch
import numpy as np
import onnxruntime as ort
import tensorflow as tf
from pathlib import Path
from safetensors.torch import load_file as safetensors_load
from utils import detect_extension
from architectures import detect_architecture_from_state_dict

def load_model_universal(model_path: Path):
    """
    Memuat model dari berbagai format dan mengembalikan objek model + scale.
    Untuk PyTorch: load state_dict, deteksi arsitektur, buat model, load weights.
    Untuk ONNX: load dengan onnxruntime (return inference session).
    Untuk TensorRT: load engine.
    Untuk TFLite: load interpreter.
    """
    ext = detect_extension(model_path)
    if ext in ['.pth', '.pt', '.ckpt', '.bin', '.pt2', '.ptl']:
        # PyTorch
        state_dict = torch.load(str(model_path), map_location='cpu')
        # unwrap jika ada wrapper
        if isinstance(state_dict, dict):
            for key in ['state_dict', 'model', 'net', 'params']:
                if key in state_dict and isinstance(state_dict[key], dict):
                    state_dict = state_dict[key]
                    break
        model, scale = detect_architecture_from_state_dict(state_dict)
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        return model, scale
    elif ext == '.safetensors':
        state_dict = safetensors_load(str(model_path), device='cpu')
        model, scale = detect_architecture_from_state_dict(state_dict)
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        return model, scale
    elif ext == '.onnx':
        # ONNX Runtime session
        session = ort.InferenceSession(str(model_path))
        return session, None  # scale perlu diatur manual atau dari metadata
    elif ext in ['.engine', '.trt', '.plan']:
        # TensorRT (butuh tensorrt lib)
        try:
            import tensorrt as trt
            with open(str(model_path), 'rb') as f:
                runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
                engine = runtime.deserialize_cuda_engine(f.read())
            return engine, None
        except ImportError:
            raise ImportError("TensorRT tidak terinstall untuk load .engine")
    elif ext == '.tflite':
        interpreter = tf.lite.Interpreter(model_path=str(model_path))
        interpreter.allocate_tensors()
        return interpreter, None
    else:
        raise ValueError(f"Format {ext} belum didukung untuk inference.")
