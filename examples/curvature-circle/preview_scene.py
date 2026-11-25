"""半分辨率预览入口。"""
from curvature_scene import CurvatureCircle as _Base
import curvature_scene as _m

_m.config.pixel_width = 540
_m.config.pixel_height = 676  # 偶数，libx264/yuv420p 要求


class CurvatureCircle(_Base):
    pass
