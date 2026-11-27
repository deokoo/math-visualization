"""数值预检：割线斜率、中值点、阶段时间轴与投影边界。"""
import math

import numpy as np

from mean_value_scene import (A, B, XI, K_MEAN, DURATION, PHASES,
                              fval, fpval, left_px, right_px, tangent_points)

# 1) 中值点：cos(xi) = k，xi 严格在区间内
assert A < XI < B, (A, XI, B)
assert abs(fpval(XI) - K_MEAN) < 1e-12, (fpval(XI), K_MEAN)
print(f"k = {K_MEAN:.6f}, xi = {XI:.6f}, cos(xi) - k = {fpval(XI)-K_MEAN:.2e}  ✓")

# 2) 割线端点一致性
for x in np.linspace(A, B, 21):
    y = fval(A) + K_MEAN * (x - A)
    assert abs(y - (fval(A) + (fval(B) - fval(A)) * (x - A) / (B - A))) < 1e-12
print("割线方程与两点式一致  ✓")

# 3) 时间轴连续
assert abs(PHASES[0][1]) < 1e-9
assert abs(PHASES[-1][2] - DURATION) < 1e-9
for a, b in zip(PHASES, PHASES[1:]):
    assert abs(a[2] - b[1]) < 1e-9
print(f"时间轴阶段连续，总时长 {DURATION}s  ✓")

# 4) 整段滑动中，切线与导函数游标都落在面板内
for c in np.linspace(A, B, 61):
    pts = tangent_points(float(c))
    assert len(pts) > 2, c
    assert all(56 <= p[0] <= 630 and 40 <= p[1] <= 505 for p in pts), c
    q = right_px(float(c), fpval(float(c)))
    assert 700 <= q[0] <= 970 and 70 <= q[1] <= 480, (c, q)
print("切线与导函数游标全程落在画布内  ✓")
print("预检通过")
