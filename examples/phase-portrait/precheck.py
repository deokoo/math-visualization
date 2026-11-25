import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase_portrait_scene as m

EXPRS = [
    r"$\tau=$", r"$\Delta=$", r"$\tau^2-4\Delta=$", r"$\lambda_{1,2}=$",
    r"$\dot{x}=ax+by,\quad \dot{y}=cx+dy$",
    r"$\tau=a+d,\quad \Delta=ad-bc$",
    r"$\det(A-\lambda I)=\lambda^2-\tau\lambda+\Delta=0$",
    r"$\lambda_{1,2}=\frac{\tau\pm\sqrt{\tau^2-4\Delta}}{2}$",
    r"$\Delta<0:\ \mathrm{saddle}\quad \tau^2<4\Delta:\ \mathrm{focus/center}$",
]

for expr in EXPRS:
    try:
        im = m._math_img(expr, 30, (236, 232, 220))
        print("OK  ", im.size, expr)
    except Exception as exc:
        print("FAIL", type(exc).__name__, str(exc)[:200], expr)
