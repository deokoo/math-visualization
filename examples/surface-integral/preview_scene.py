"""半分辨率预览入口。"""
from surface_integral_scene import SurfaceIntegral as _Base
import surface_integral_scene as _m

_m.config.pixel_width = 540
_m.config.pixel_height = 676  # 偶数，libx264/yuv420p 要求


class SurfaceIntegral(_Base):
    pass
