# Models/ESRGAN/arch.py
from realesrgan.archs.rrdbnet_arch import RRDBNet
from realesrgan.archs.srvgg_arch import SRVGGNetCompact

# Ekspor class biar gampang di-import oleh handler
__all__ = ['RRDBNet', 'SRVGGNetCompact']

# Factory functions (biar handler gampang manggilnya)
def create_rrdbnet(scale=4, num_block=23):
    return RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=num_block, num_grow_ch=32, scale=scale)

def create_srvgg(upscale=4, num_conv=16):
    return SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=num_conv, upscale=upscale, act_type='prelu')
