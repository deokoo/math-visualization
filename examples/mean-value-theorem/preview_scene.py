"""半分辨率预览入口。"""
from mean_value_scene import MeanValueTheorem as _Base
import mean_value_scene as _m

_m.config.pixel_width = 540
_m.config.pixel_height = 676  # 偶数，libx264/yuv420p 要求


class MeanValueTheorem(_Base):
    pass
