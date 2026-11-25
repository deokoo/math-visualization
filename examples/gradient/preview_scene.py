"""半分辨率预览入口（540x676，偶数边长满足 libx264/yuv420p）。"""
from gradient_scene import Gradient as _Base
import gradient_scene as _m

_m.config.pixel_width = 540
_m.config.pixel_height = 676


class Gradient(_Base):
    pass
