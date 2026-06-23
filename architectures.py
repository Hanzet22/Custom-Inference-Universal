# architectures.py
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan.archs.srvgg_arch import SRVGGNetCompact

def detect_architecture_from_state_dict(state_dict, scale_hint=4):
    """
    Kembalikan tuple (model_obj, netscale) berdasarkan keys state_dict.
    Referensi: inference_realesrgan.py
    """
    keys = list(state_dict.keys())
    # RRDBNet (RealESRGAN x4plus, x2plus, anime)
    if any("rrdb" in k.lower() for k in keys) or "conv_first" in keys:
        # Estimasi num_block, num_feat, dll. Bisa di-hardcode atau deteksi dari shape
        # Biasanya x4: num_block=23, x2: num_block=23, anime: num_block=6
        # Cek scale dari shape conv_first.weight
        scale = 4  # fallback
        # Coba deteksi scale dari input/output channels atau dari nama file
        if "x2" in str(state_dict.get("_name", "")):
            scale = 2
        num_block = 23
        if "anime" in str(state_dict.get("_name", "")):
            num_block = 6
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=num_block, num_grow_ch=32, scale=scale)
        return model, scale
    # SRVGGNetCompact (realesr-animevideov3, realesr-general-x4v3)
    elif "body.0" in keys or "body.1" in keys:
        num_conv = 16  # default untuk animevideov3
        if "general" in str(state_dict.get("_name", "")):
            num_conv = 32
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=num_conv, upscale=4, act_type='prelu')
        return model, 4
    # Tambahan: SwinIR, HAT, dll. bisa ditambahkan di sini
    else:
        raise ValueError("Arsitektur tidak dikenali. Pastikan model adalah RRDBNet atau SRVGG.")
