"""公式 PNG 预生成 + 拉普拉斯积分数值预检。"""
import math
import laplace_scene as m

FORMULAS = {
    "f1": r"$F(s)=\mathcal{L}\{f(t)\}=\int_0^{\infty}f(t)e^{-st}\,dt$",
    "f2": r"$f(t)=e^{-at}\ \Rightarrow\ f(t)e^{-st}=e^{-(s+a)t}$",
    "f3": r"$\int_0^{T}e^{-(s+a)t}\,dt=\frac{1-e^{-(s+a)T}}{s+a}$",
    "f4": r"$T\to\infty,\ s+a>0:\quad F(s)=\frac{1}{s+a}$",
    "f5": r"$\mathrm{ROC}:\ \mathrm{Re}(s)>-a\quad(\mathrm{pole}\ s=-a)$",
}

print("[公式 PNG 原始尺寸]")
for name, expr in FORMULAS.items():
    im = m._math_img(expr, 27, m.C_FORMULA)
    im.save(m.FORMULA_DIR / f"lp_{name}.png")
    print(f"  {name}: {im.width} x {im.height}  (aspect {im.width / im.height:.2f})")

print("\n[数值积分 vs 闭式解]")
for sigma in (1.20, 0.50, 0.00, -0.50):
    q = sigma + m.A
    if q <= 0:
        print(f"  s={sigma:5.2f}: 发散（s+a={q:.2f} <= 0）")
        continue
    for T in (1.0, 3.0, 6.0):
        num = m.partial(sigma, T)
        target = 1.0 / q
        print(f"  s={sigma:5.2f} T={T:.1f}: F_T={num:.6f} -> 1/(s+a)={target:.6f}")
    assert abs(m.partial(sigma, 60.0) - 1.0 / q) < 1e-6

print("\n[相位边界]")
for name, t0, t1 in m.PHASES:
    print(f"  {name:9s} [{t0:5.2f}, {t1:5.2f})  -> 成片约 [{t0+1:.1f}, {t1+1:.1f}]")
print("预检通过")
