# utils.py
import zipfile
from pathlib import Path

SUPPORTED_EXTENSIONS = {
    ".safetensors", ".pth", ".pt", ".ckpt", ".bin", ".param", ".onnx",
    ".engine", ".trt", ".plan", ".xml", ".h5", ".hdf5", ".pb", ".msgpack",
    ".mlmodel", ".mlpackage", ".tflite", ".json", ".gguf", ".ggml", ".ggmf",
    ".ggjt", ".npy", ".npz", ".keras", ".tfrecords", ".mar", ".pte", ".pt2",
    ".ptl", ".pkl", ".nc", ".mleap", ".coreml", ".surml", ".llamafile",
    ".caffemodel", ".prototxt", ".dlc", ".model", ".exl2", ".lora", ".prompt",
    ".weights", ".gptq", ".awq", ".vae", ".unet", ".embedding", ".hypernetwork",
    ".data-00000-of-00001", ".index", ".pbtxt", ".plan"
}

def detect_extension(path: Path) -> str:
    name = path.name.lower()
    for suffix in [".mlpackage", ".torchscript", ".safetensors", ".tfrecords", ".caffemodel", ".prototxt", ".llamafile", ".hdf5"]:
        if name.endswith(suffix):
            return suffix
    return path.suffix.lower()

def is_zipfile(path: Path) -> bool:
    return zipfile.is_zipfile(path)

def normalize_name(path: Path) -> str:
    return path.name.lower()
