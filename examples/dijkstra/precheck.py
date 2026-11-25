import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dijkstra_scene as m

EXPRS = [
    r"$d(s)=0,\quad d(v)=+\infty\ (v\neq s)$",
    r"$u=\mathrm{argmin}_{v\notin S}\,d(v)$",
    r"$d(v)\leftarrow\min\{\,d(v),\,d(u)+w(u,v)\,\}$",
    r"$S\leftarrow S\cup\{u\}$",
    r"$\mathrm{at\ end}:\ d(v)=\mathrm{shortest\ distance}$",
    r"$|S|=$", r"$\mathrm{relax}=$", r"$d(H)=$", r"$\mathrm{updates}=$",
]

for expr in EXPRS:
    try:
        im = m._math_img(expr, 30, (236, 232, 220))
        print("OK  ", im.size, expr)
    except Exception as exc:
        print("FAIL", type(exc).__name__, str(exc)[:200], expr)
