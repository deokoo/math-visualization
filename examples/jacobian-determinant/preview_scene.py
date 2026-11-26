"""半分辨率预览入口。"""
from jacobian_scene import JacobianChange as _Base
import jacobian_scene as _m

_m.config.pixel_width = 540
_m.config.pixel_height = 676  # 偶数，libx264/yuv420p 要求


class JacobianChange(_Base):
    pass
