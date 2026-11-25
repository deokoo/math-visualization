"""数值预检：散度恒等式、逐面通量求和与投影范围。"""
import math
import numpy as np

from divergence_scene import (A_FACES_0, A_FIXED, P_FAR, P_START,
                              face_fluxes, project, vertex, PHASES, DURATION)

for a in (A_FACES_0, 0.9, A_FIXED):
    for p in (P_START, P_FAR, (0.0, 0.0, 0.0), (-0.7, 0.9, -0.4)):
        qs = face_fluxes(a, p)
        lhs = sum(qs)
        rhs = 24.0 * a ** 3
        assert abs(lhs - rhs) < 1e-10, (a, p, lhs, rhs)
        print(f"a={a:.2f} p={tuple(round(v, 2) for v in p)} "
              f"ΣF·n={lhs:.6f}  24a³={rhs:.6f}  ✓")

# 阶段覆盖完整时长且首尾相接
assert abs(PHASES[0][1] - 0.0) < 1e-9
assert abs(PHASES[-1][2] - DURATION) < 1e-9
for prev, nxt in zip(PHASES, PHASES[1:]):
    assert abs(prev[2] - nxt[1]) < 1e-9
print("时间轴阶段连续，总时长", DURATION)

# 最大立方体的投影需落在画布内（含红色源点在外的情况）
pts = [vertex(A_FIXED, *v) for v in
       [(sx, sy, sz) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]]
us = [q[0] for q in pts]
ws = [q[1] for q in pts]
print(f"投影范围 u=[{min(us):.0f},{max(us):.0f}] w=[{min(ws):.0f},{max(ws):.0f}] "
      f"(画布 900x470)")
assert min(us) > 10 and max(us) < 570
assert min(ws) > 15 and max(ws) < 455
print("预检通过")
