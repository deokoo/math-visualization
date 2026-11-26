"""数值预检：T_s 的雅可比行列式、面积比与网格投影范围。"""
import math
import numpy as np

from jacobian_scene import (DURATION, PHASES, VMAX,
                            patch_params, morph_s, map_uv, right_px, left_px)

# 1) s=1 时 J=u，像面积 / 原面积 = u0，与 v0、Δ 无关
for u0 in (0.35, 0.72, 0.90, 1.55, 2.10):
    for v0 in (0.1, 0.82, 1.9, 4.7):
        for du, dv in ((0.44, 0.62), (0.12, 0.18)):
            jac = u0
            # 解析积分 ∫∫ u du dv /(du dv) = u0
            num = ((u0 + du / 2) ** 2 - (u0 - du / 2) ** 2) / 2 * dv
            assert abs(num / (du * dv) - jac) < 1e-10, (u0, v0, du, dv)
print("J = u = r，像面积 / 原面积 = u0（与 v0、Δ 无关）  ✓")

# 2) 偏导定义核对：∂x/∂u=cos v, ∂y/∂v=u cos v, ∂x/∂v=-u sin v, ∂y/∂u=sin v
for u in (0.4, 1.1, 2.3):
    for v in (0.2, 1.4, 3.0, 5.5):
        e = 1e-6
        dxdu = (map_uv(u + e, v, 1)[0] - map_uv(u - e, v, 1)[0]) / (2 * e)
        dydu = (map_uv(u + e, v, 1)[1] - map_uv(u - e, v, 1)[1]) / (2 * e)
        dxdv = (map_uv(u, v + e, 1)[0] - map_uv(u, v - e, 1)[0]) / (2 * e)
        dydv = (map_uv(u, v + e, 1)[1] - map_uv(u, v - e, 1)[1]) / (2 * e)
        assert abs(dxdu - math.cos(v)) < 1e-6
        assert abs(dydu - math.sin(v)) < 1e-6
        assert abs(dxdv + u * math.sin(v)) < 1e-6
        assert abs(dydv - u * math.cos(v)) < 1e-6
        assert abs(dxdu * dydv - dxdv * dydu - u) < 1e-6
print("四个偏导数与 det J = u 的数值微分核对通过  ✓")

# 3) 时间轴连续
assert abs(PHASES[0][1]) < 1e-9
assert abs(PHASES[-1][2] - DURATION) < 1e-9
for a, b in zip(PHASES, PHASES[1:]):
    assert abs(a[2] - b[1]) < 1e-9
print("时间轴阶段连续，总时长", DURATION)

# 4) 最终网格与微元落在右栏画布内
pts = [right_px(u, v, 1.0)
       for u in (0, 1.3, 2.6) for v in np.linspace(0, VMAX, 97)]
us = [p[0] for p in pts]
ws = [p[1] for p in pts]
print(f"右栏投影 u=[{min(us):.0f},{max(us):.0f}] w=[{min(ws):.0f},{max(ws):.0f}] (1000x560)")
assert 505 < min(us) < max(us) < 990
assert 26 < min(ws) < max(ws) < 520
for t in (5.5, 10.0, 12.5, 15.0, 18.8):
    u0, v0, du, dv, rev = patch_params(t)
    pp = [right_px(u0 + a * du / 2, v0 + b * dv / 2, 1.0)
          for a in (-1, 1) for b in (-1, 1)]
    assert all(510 < p[0] < 990 and 30 < p[1] < 520 for p in pp), (t, pp)
print("微元投影全部在画布内  ✓")
print("预检通过")
