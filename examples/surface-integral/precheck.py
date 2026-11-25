import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import surface_integral_scene as m

EXPRS = [
    r"$D=[-a,a]^2,\quad \Sigma:\ \mathbf{r}(u,v)=(u,v,f(u,v))$",
    r"$\mathbf{r}_u=(1,0,f_u),\quad \mathbf{r}_v=(0,1,f_v)$",
    r"$\mathbf{r}_u\times\mathbf{r}_v=(-f_u,-f_v,1)$",
    r"$\mathbf{F}\cdot\mathbf{n}\,dS=\mathbf{F}(\mathbf{r})\cdot(\mathbf{r}_u\times\mathbf{r}_v)\,du\,dv$",
    r"$\Phi=\iint_\Sigma\mathbf{F}\cdot\mathbf{n}\,dS=\iint_D\mathbf{F}(\mathbf{r})\cdot(\mathbf{r}_u\times\mathbf{r}_v)\,du\,dv$",
]

for expr in EXPRS:
    try:
        im = m._math_img(expr, 30, (236, 232, 220))
        print("OK  ", im.size, expr)
    except Exception as exc:
        print("FAIL", type(exc).__name__, str(exc)[:300], expr)

print(f"TOTAL={m.TOTAL:.6f} patches={len(m.PATCHES)} cum_end={m.CUM[-1]:.6f}")
