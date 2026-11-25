"""半分辨率预览入口。"""
from divergence_scene import DivergenceTheorem as _Base
import divergence_scene as _m

_m.config.pixel_width = 540
_m.config.pixel_height = 676  # 偶数，libx264/yuv420p 要求


class DivergenceTheorem(_Base):
    pass
