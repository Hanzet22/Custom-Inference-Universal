# upscaler.py
import cv2
import os
import glob
import numpy as np
from pathlib import Path
from realesrgan import RealESRGANer
from config import DEFAULT_TILE, DEFAULT_SCALE, DEFAULT_FACE_ENHANCE
from telegram_notifier import send_error_to_telegram

def create_upsampler(model, scale, tile=0, tile_pad=10, pre_pad=0, half=True, gpu_id=None):
    """Bungkus model dengan RealESRGANer."""
    return RealESRGANer(
        scale=scale,
        model_path=None,  # sudah lewat model
        model=model,
        tile=tile,
        tile_pad=tile_pad,
        pre_pad=pre_pad,
        half=half,
        gpu_id=gpu_id
    )

def upscale_image(model, image_path, output_path, scale=4, tile=0, face_enhance=False, gpu_id=None):
    try:
        img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError("Gagal membaca gambar.")
        # Deteksi mode RGBA
        img_mode = 'RGBA' if (len(img.shape)==3 and img.shape[2]==4) else None
        # Buat upsampler
        upsampler = create_upsampler(model, scale, tile=tile, gpu_id=gpu_id)
        if face_enhance:
            from gfpgan import GFPGANer
            face_enhancer = GFPGANer(
                model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
                upscale=scale,
                arch='clean',
                channel_multiplier=2,
                bg_upsampler=upsampler
            )
            _, _, output = face_enhancer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
        else:
            output, _ = upsampler.enhance(img, outscale=scale)
        # Simpan
        output_path.parent.mkdir(parents=True, exist_ok=True)
        ext = output_path.suffix[1:] or 'png'
        if img_mode == 'RGBA':
            ext = 'png'
        cv2.imwrite(str(output_path), output)
        return True
    except Exception as e:
        send_error_to_telegram(f"Error upscale {image_path}: {e}")
        raise e

def upscale_video(model, video_path, output_path, scale=4, tile=0, face_enhance=False, gpu_id=None):
    """Proses video frame per frame."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError("Tidak bisa membuka video.")
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # Upscale ukuran output
    out_width = int(width * scale)
    out_height = int(height * scale)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (out_width, out_height))
    upsampler = create_upsampler(model, scale, tile=tile, gpu_id=gpu_id)
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Upscale frame
        if face_enhance:
            # Face enhance per frame (bisa lambat, opsional)
            from gfpgan import GFPGANer
            face_enhancer = GFPGANer(
                model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
                upscale=scale,
                arch='clean',
                channel_multiplier=2,
                bg_upsampler=upsampler
            )
            _, _, output = face_enhancer.enhance(frame, has_aligned=False, only_center_face=False, paste_back=True)
        else:
            output, _ = upsampler.enhance(frame, outscale=scale)
        out.write(output)
        frame_count += 1
        if frame_count % 10 == 0:
            print(f"Diproses {frame_count} frame")
    cap.release()
    out.release()
    return True
