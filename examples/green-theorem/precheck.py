import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import green_theorem_scene as m

EXPRS = [
    r"$C=\partial D\quad(\mathrm{positive\ orientation})$",
    r"$\mathbf{F}=(P,Q),\quad \oint_C P\,dx+Q\,dy$",
    r"$\frac{\partial Q}{\partial x}-\frac{\partial P}{\partial y}=(\nabla\times\mathbf{F})\cdot\mathbf{k}$",
    r"$\iint_D\left(\frac{\partial Q}{\partial x}-\frac{\partial P}{\partial y}\right)dA$",
    r"$\oint_C P\,dx+Q\,dy=\iint_D\left(\frac{\partial Q}{\partial x}-\frac{\partial P}{\partial y}\right)dA$",
]

for expr in EXPRS:
    try:
        im = m._math_img(expr, 30, (236, 232, 220))
        print("OK  ", im.size, expr)
    except Exception as exc:
        print("FAIL", type(exc).__name__, str(exc)[:300], expr)

print(f"line={m.TOTAL_LINE:.6f} area(D)={m.AREA_D:.6f} cells={len(m.CELLS)} cancel={len(m.CANCEL_MARKS)}")
