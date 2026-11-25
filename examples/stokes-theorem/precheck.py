import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stokes_scene as m

EXPRS = [
    r"$C=\partial\Sigma:\ x^2+y^2=R^2,\quad z=z_0$",
    r"$\mathbf{F}=(-y,\ x,\ xy),\quad \nabla\times\mathbf{F}=(x,\ -y,\ 2)$",
    r"$\oint_C\mathbf{F}\cdot d\mathbf{r}=\int_0^{2\pi}R^2\,d\theta=2\pi R^2$",
    r"$(\nabla\times\mathbf{F})\cdot\mathbf{n}\,dS=\left(2+\frac{2h(x^2-y^2)}{R^2}\right)dx\,dy$",
    r"$\oint_{\partial\Sigma}\mathbf{F}\cdot d\mathbf{r}=\iint_\Sigma(\nabla\times\mathbf{F})\cdot\mathbf{n}\,dS$",
]
HEIGHTS = [0.29, 0.29, 0.50, 0.48, 0.42]
for expr, h in zip(EXPRS, HEIGHTS):
    try:
        im = m._math_img(expr, 28, (236, 232, 220))
        w = im.width / im.height * h
        print(f"OK size={im.size} display_w={w:.2f} (limit 7.6) height={h}  {expr[:42]}")
    except Exception as exc:
        print("FAIL", type(exc).__name__, str(exc)[:300], expr)

print(f"R={m.R} H={m.H} TOTAL(analytic 2πR²)={m.TOTAL:.6f}")
print(f"flux(h=H) = {m.flux_numeric(m.H):.6f}")
print(f"flux(h=0) = {m.flux_numeric(0.0):.6f}")
print(f"patches={len(m.PATCHES)} cum_end={m.CUM[-1]:.6f}")
