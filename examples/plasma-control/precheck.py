# -*- coding: utf-8 -*-
import numpy as np
from plasma_scene import Sim, NFRAMES, FPS, W, H

s = Sim()
for name in ("ne_true", "Te_true", "z_ne", "ekf", "sigma", "Kgain",
             "lstm", "alpha", "fused", "ref", "actual", "ctrl", "Rest"):
    a = np.asarray(getattr(s, name))
    assert np.isfinite(a).all(), "%s has NaN/inf" % name
    print("%-9s min=%8.4f max=%8.4f mean=%8.4f" % (name, a.min(), a.max(), a.mean()))

# EKF covariance shrinks: late sigma < early sigma
assert s.sigma[400:].mean() < s.sigma[:100].mean(), "EKF covariance not shrinking"
# fusion best
assert s.rmse[3] < s.rmse[1] < s.rmse[0], "fusion/EKF/open-loop RMSE ordering wrong"
# alpha decreases (LSTM weight grows)
assert s.alpha[-1] < s.alpha[0], "alpha not decreasing"
# closed loop bounded
assert np.abs(s.actual).max() < 6.0
print("time=%.1fs frames=%d size=%dx%d" % (NFRAMES/FPS, NFRAMES, W, H))
print("RMSE open=%.4f ekf=%.4f lstm=%.4f fuse=%.4f" % tuple(s.rmse))
print("sigma early=%.4f late=%.4f" % (s.sigma[:100].mean(), s.sigma[400:].mean()))
print("PRECHECK OK")
