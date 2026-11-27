# -*- coding: utf-8 -*-
"""Numeric sanity check for all chaos solvers (no rendering)."""
import numpy as np
from chaos_scene import (Sim, Lorenz, Logistic, VDP, DoublePendulum,
                         Rossler, Henon, Duffing, W, H, FPS, NFRAMES)

def chk(name, arr):
    a = np.asarray(arr, dtype=float)
    finite = np.isfinite(a).all()
    print("%-16s shape=%-14s finite=%s  min=% .3f  max=% .3f" %
          (name, str(a.shape), finite, np.nanmin(a), np.nanmax(a)))
    assert finite, name + " has NaN/Inf"
    return a

s = Sim()
chk("lorenz", s.lorenz.pts)
chk("logistic", s.logistic.pts)
for i, t in enumerate(s.vdp.trajs):
    chk("vdp[%d]" % i, t)
chk("double_pendulum", s.pend.s)
chk("rossler", s.rossler.pts)
chk("henon", s.henon.pts)
chk("duffing", s.duffing.s)

# double pendulum angles are unwrapped (the arm can spin many turns) -- check
# that sin/cos stay bounded, which is what the renderer actually uses.
assert np.abs(np.sin(s.pend.s[:, 0])).max() <= 1.0001
assert np.abs(np.cos(s.pend.s[:, 1])).max() <= 1.0001
print("time=%.1fs frames=%d size=%dx%d" % (NFRAMES/FPS, NFRAMES, W, H))
print("PRECHECK PASS")
