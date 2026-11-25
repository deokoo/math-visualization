"""预检查：mathtext 公式能否渲染 + 梯度数值自洽性。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gradient_scene as m

EXPRS = [
    r"$f(x,y)\quad(\mathrm{height\ on\ the\ map})$",
    r"$\nabla f=\left(\frac{\partial f}{\partial x},\ \frac{\partial f}{\partial y}\right)$",
    r"$D_{\mathbf{u}}f=\nabla f\cdot\mathbf{u}\quad(|\mathbf{u}|=1)$",
    r"$D_{\mathbf{u}}f=|\nabla f|\cos\theta$",
    r"$\theta=0:\max\quad \theta=\frac{\pi}{2}:0\quad \theta=\pi:\min$",
    r"$\times\ \cos\theta\ =$",
]

for expr in EXPRS:
    try:
        im = m._math_img(expr, 30, (236, 232, 220))
        print("OK  ", im.size, expr)
    except Exception as exc:
        print("FAIL", type(exc).__name__, str(exc)[:300], expr)

print(f"P=({m.PX},{m.PY}) grad=({m.GX:.6f},{m.GY:.6f}) "
      f"|grad|={m.GMAG:.6f} num=({m.NUM_GX:.6f},{m.NUM_GY:.6f})")
print(f"contour segments={len(m.CONTOUR_SEGS)} base={m.BASE_IMG.size}")
for t in (0.5, 2.0, 6.0, 7.8, 10.0, 12.2, 14.0, 16.0, 17.6, 18.5, 20.5):
    ph, pg = m.phase_info(t)
    th = m.theta_of(t) * 180 / 3.141592653589793 % 360
    print(f"t={t:5.1f} phase={ph:6s} prog={pg:.2f} theta={th:6.1f}deg "
          f"Duf={m.GMAG * __import__('math').cos(m.theta_of(t)):+.3f}")
