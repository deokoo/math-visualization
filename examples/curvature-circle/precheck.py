"""公式 PNG 预生成 + 数值/几何预检。"""
import math
import curvature_scene as m

FORMULAS = {
    "f1": r"$\kappa=\lim_{\Delta s\to 0}\left|\frac{\Delta\alpha}{\Delta s}\right|=\left|\frac{d\alpha}{ds}\right|$",
    "f2": r"$y=f(x):\quad\kappa=\frac{|y''|}{(1+y'^2)^{3/2}}$",
    "f3": r"$R=\frac{1}{\kappa}=\frac{(1+y'^2)^{3/2}}{|y''|}$",
    "f4": r"$C=(a,b)=\left(x-\frac{y'(1+y'^2)}{y''},\ y+\frac{1+y'^2}{y''}\right)$",
    "f5": r"$(x-a)^2+(y-b)^2=R^2\quad\Rightarrow\quad\mathrm{second\ order\ contact}$",
}

print("[公式 PNG 原始尺寸]")
for name, expr in FORMULAS.items():
    im = m._math_img(expr, 27, m.C_FORMULA)
    im.save(m.FORMULA_DIR / f"cv_{name}.png")
    print(f"  {name}: {im.width} x {im.height}  (aspect {im.width / im.height:.2f})")

print("\n[曲线上的关键数值]")
for x in (2.25, 2.05, 1.55, math.pi / 2, 1.20, 0.85):
    a, b = m.center(x)
    r = m.radius(x)
    bottom = b - r
    print(f"  x={x:5.3f}  κ={m.kappa(x):.4f}  R={r:6.3f}  C=({a:6.3f},{b:6.3f})  圆底y={bottom:.4f} ≥ Y_MIN={m.Y_MIN}")
    assert bottom >= m.Y_MIN

print("\n[圆心到曲线上点的距离 == R]")
for x in (2.25, 1.55, math.pi / 2, 0.85):
    a, b = m.center(x)
    dist = math.hypot(x - a, m.f(x) - b)
    assert abs(dist - m.radius(x)) < 1e-9
    print(f"  x={x:5.3f}: {dist:.6f}")

print("\n[相位边界]")
for name, t0, t1 in m.PHASES:
    print(f"  {name:8s} [{t0:5.2f}, {t1:5.2f})  -> 成片约 [{t0+1:.1f}, {t1+1:.1f}]")
print("预检通过")
