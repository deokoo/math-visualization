import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import binary_search_scene as m

EXPRS = [
    r"$A[0]\leq A[1]\leq\cdots\leq A[n-1]$",
    r"$\mathrm{mid}=\left\lfloor\frac{lo+hi}{2}\right\rfloor$",
    r"$A[\mathrm{mid}]<T\ \Rightarrow\ lo\leftarrow\mathrm{mid}+1$",
    r"$A[\mathrm{mid}]>T\ \Rightarrow\ hi\leftarrow\mathrm{mid}-1$",
    r"$\mathrm{steps}\leq\left\lceil\log_2 n\right\rceil=29\quad(n=5\times10^8)$",
    r"$\mathrm{step}=$", r"$\mathrm{remain}=$", r"$\mathrm{mid}=$", r"$T=$",
]

for expr in EXPRS:
    try:
        im = m._math_img(expr, 30, (236, 232, 220))
        print("OK  ", im.size, expr)
    except Exception as exc:
        print("FAIL", type(exc).__name__, str(exc)[:200], expr)
