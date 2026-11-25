#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Manim 数学动画：泰勒展开 —— 用多项式逼近 sin x（高等数学）

视觉规范与高斯积分 / 复指数映射 / 傅里叶级数成片一致：
  - 1080x1350 竖屏、30fps、约 22s
  - 公式全程固定，只动参数、网格与动画
  - 无 LaTeX：公式用 matplotlib mathtext 预渲染为透明 PNG
  - 暗色底 #101113、中灰清晰线条、青/红边缘曲线、纯内存动态数字

画面：
  顶部 = 3D 泰勒多项式"瀑布"曲面（青=sin x 目标，红=当前 P_n）
  中部 = 2D 收敛图 + 收敛区间指示 + 动态数值行
  底部 = 固定公式链

运行：
    manim -pql taylor_scene.py TaylorSeries
    manim -pqh taylor_scene.py TaylorSeries
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PIL import Image, ImageDraw, ImageFont

from manim import *

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
FORMULA_DIR = HERE / "formulas"
FORMULA_DIR.mkdir(exist_ok=True)

config.pixel_width = 1080
config.pixel_height = 1350
config.frame_width = 8
config.frame_height = 10
config.background_color = "#101113"
config.frame_rate = 30

DURATION = 21.26

C_TEXT = (236, 232, 220)
C_FORMULA = (236, 232, 220)
C_GRID = (104, 110, 114)
C_GRID_DIM = (66, 70, 74)
C_CYAN = (74, 214, 200)
C_RED = (232, 96, 82)
C_ORANGE = (238, 148, 72)

M_TEXT = "#ECE8DC"
M_FORMULA = "#ECE8DC"
M_GRID = "#767C80"
M_GRID_DIM = "#5C6165"
M_CYAN = "#4AD6C8"
M_RED = "#E86052"
M_ORANGE = "#EE9448"

_SERIF = str(Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSerif.ttf")

# ---------------------------------------------------------------------------
# 无 LaTeX 公式预渲染 + 纯内存动态数字
# ---------------------------------------------------------------------------
_MATH_CACHE: dict[tuple, Image.Image] = {}


def _math_img(expr: str, size: int, color) -> Image.Image:
    # 统一颜色为 RGB 元组（mi() 可能传入 M_* 字符串常量）
    if isinstance(color, str):
        color = tuple(int(c) for c in ManimColor(color).to_rgb() * 255)
    key = (expr, size, color)
    if key in _MATH_CACHE:
        return _MATH_CACHE[key]
    fig = Figure(figsize=(16.0, 2.0), dpi=200)
    fig.patch.set_alpha(0.0)
    canvas = FigureCanvasAgg(fig)
    fig.text(0.01, 0.5, expr, fontsize=size,
             color="#%02x%02x%02x" % color, ha="left", va="center")
    canvas.draw()
    arr = np.asarray(canvas.buffer_rgba()).copy()
    im = Image.fromarray(arr, mode="RGBA")
    bbox = im.getchannel("A").getbbox()
    im = im.crop(bbox) if bbox else Image.new("RGBA", (1, 1), (0, 0, 0, 0))
    _MATH_CACHE[key] = im
    return im


def mi(expr: str, name: str, color=C_FORMULA, height=0.25, size=32) -> ImageMobject:
    path = FORMULA_DIR / f"ts_{name}.png"
    if not path.exists():
        _math_img(expr, size, color).save(path)
    m = ImageMobject(str(path))
    if m.height > 0:
        m.scale(height / m.height)
    return m


def text_img(text: str, px: int, color=C_TEXT, height=0.3) -> ImageMobject:
    img = Image.new("RGBA", (1400, 200), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    font = ImageFont.truetype(_SERIF, px)
    d.text((20, 20), text, font=font, fill=(*color, 255))
    bbox = img.getchannel("A").getbbox()
    img = img.crop(bbox) if bbox else img
    m = ImageMobject(np.asarray(img))
    if m.height > 0:
        m.scale(height / m.height)
    return m


class PixelNumber(ImageMobject):
    """PIL 渲染到固定 RGBA 数组，set_value 原地更新，零文件 IO。"""

    _FONT = str(Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf")

    def __init__(self, value: float = 0.0, places: int = 2,
                 color=M_TEXT, px: int = 40, **kwargs):
        self.places = places
        self.value = value
        self._color = tuple(int(c) for c in ManimColor(color).to_rgb() * 255) \
            if not isinstance(color, tuple) else color
        self._font = ImageFont.truetype(self._FONT, px)
        nchars = 1 + 1 + places + (1 if places else 0)
        self._cw, self._ch = int(px * (nchars * 0.62 + 0.4)), int(px * 1.6)
        canvas = np.zeros((self._ch, self._cw, 4), dtype=np.uint8)
        super().__init__(canvas, **kwargs)
        self.set_value(value)

    def set_value(self, value: float):
        self.value = value
        txt = f"{abs(value):.{self.places}f}"
        if value < 0:
            txt = "-" + txt
        img = Image.new("RGBA", (self._cw, self._ch), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        box = d.textbbox((0, 0), txt, font=self._font)
        tw, th = box[2] - box[0], box[3] - box[1]
        d.text(((self._cw - tw) / 2 - box[0], (self._ch - th) / 2 - box[1]),
               txt, font=self._font, fill=(*self._color, 255))
        self.pixel_array[:] = np.asarray(img)


def poly(points, color, width=1.2, opacity=1.0):
    m = VMobject(color=color, stroke_width=width, stroke_opacity=opacity)
    m.set_points_as_corners(list(points))
    return m


def set_poly(mob, points):
    mob.set_points_as_corners(list(points))


def visible_run(pts, ymin, ymax):
    """取折线中纵坐标落在 [ymin, ymax] 内的最长连续段，避免曲线越界压到公式。"""
    mask = [ymin <= p[1] <= ymax for p in pts]
    best, cur = [], []
    for p, ok in zip(pts, mask):
        if ok:
            cur.append(p)
        else:
            if len(cur) > len(best):
                best = cur
            cur = []
    if len(cur) > len(best):
        best = cur
    return best if len(best) >= 2 else [pts[0], pts[1]]


def clamp_y(v, ymin, ymax):
    return min(ymax, max(ymin, v))


def clamp01(v):
    return float(np.clip(v, 0.0, 1.0))


def ease(v):
    v = clamp01(v)
    return v * v * (3 - 2 * v)


def tri(v):
    """三角波：0->1->0，周期 2。"""
    f = v % 2.0
    return f if f < 1.0 else 2.0 - f


# ---------------------------------------------------------------------------
# 数学：sin x 的泰勒多项式
# ---------------------------------------------------------------------------
NMAX = 5                       # n = 0..5，最高次 2n+1 = 11
XS = np.linspace(-math.pi, math.pi, 121)
_SINX = np.sin(XS)


def P_int(x, n: int):
    """n 阶（含 k=0..n，共 n+1 项）麦克劳林多项式，支持 numpy 数组。"""
    x = np.asarray(x, dtype=float)
    s = np.zeros_like(x)
    for k in range(n + 1):
        s = s + ((-1) ** k) * x ** (2 * k + 1) / math.factorial(2 * k + 1)
    return s


def P_cont(x, nf: float):
    """连续阶数 nf（0..NMAX）上的插值多项式，供动画平滑变形。"""
    k0 = int(math.floor(nf))
    k0 = max(0, min(NMAX - 1, k0))
    w = clamp01(nf - k0)
    return (1 - w) * P_int(x, k0) + w * P_int(x, k0 + 1)


def approx_interval(nf: float, tol: float = 0.12) -> float:
    """返回 [-a, a] 中 |P_n - sin x| <= tol 的最大 a。"""
    grid = np.linspace(0, math.pi, 96)
    err = np.abs(P_cont(grid, nf) - np.sin(grid))
    ok = err <= tol
    if not ok.any():
        return 0.0
    # 取连续为 True 的前缀（多项式在 0 附近最准）
    idx = 0
    while idx < len(ok) and ok[idx]:
        idx += 1
    return float(grid[max(0, idx - 1)])


# ---------------------------------------------------------------------------
# 3D 瀑布曲面投影：depth = 阶数 n，height = P_n(x)
# ---------------------------------------------------------------------------
C3 = np.array([-0.15, 2.48, 0.0])
SX3 = 0.92          # x 缩放
DX3 = 0.155         # 每阶深度水平位移
DY3 = 0.165         # 每阶深度竖直位移
VS3 = 0.38          # 函数值高度缩放


def proj3(x, nf, val):
    return np.array([C3[0] + SX3 * x + DX3 * nf,
                     C3[1] + DY3 * nf + VS3 * float(val), 0.0])


# 2D 收敛图
YC2 = -0.32
SX2 = 3.42 / math.pi     # x -> 屏幕
A2 = 0.72                # 幅值缩放
AX2_Y = YC2


def x2(x):
    return SX2 * x


def y2(v):
    return YC2 + A2 * v


# ---------------------------------------------------------------------------
# 场景
# ---------------------------------------------------------------------------
class TaylorSeries(Scene):
    def construct(self):
        # 顶部标题
        title = text_img("AIMaker-Lab", 52, C_TEXT, 0.28).move_to([0, 4.35, 0])
        cursor = Line(UP * 0.16, DOWN * 0.16, color=M_TEXT,
                      stroke_width=2.5).move_to([0, 4.35, 0])
        cursor.set_opacity(0)

        # ------------------------------------------------------------------
        # 3D 泰勒多项式瀑布曲面（静态线框）
        # ------------------------------------------------------------------
        nf_grid = np.linspace(0, NMAX, 10)
        surf_cross = [
            poly([proj3(x, nf, P_cont(x, nf)) for x in np.linspace(-math.pi, math.pi, 52)],
                 M_GRID, 1.35, 0.78)
            for nf in nf_grid
        ]
        surf_long = [
            poly([proj3(xv, nf, P_cont(xv, nf)) for nf in np.linspace(0, NMAX, 28)],
                 M_GRID_DIM, 1.15, 0.70)
            for xv in np.linspace(-math.pi, math.pi, 19)
        ]
        # 底部深度基线（网格地面感）
        floor_lines = [
            poly([proj3(x, 0, -1.35), proj3(x, NMAX, -1.35)],
                 M_GRID_DIM, 1.0, 0.42)
            for x in np.linspace(-math.pi, math.pi, 13)
        ]
        floor_side = poly([proj3(-math.pi, nf, -1.35) for nf in np.linspace(0, NMAX, 20)] +
                          [proj3(math.pi, nf, -1.35) for nf in np.linspace(NMAX, 0, 20)],
                          M_GRID_DIM, 1.0, 0.42)
        # 青色目标：sin x 位于最深处（n=NMAX 的"极限墙"）
        edge_sin = poly([proj3(x, NMAX, math.sin(x)) for x in XS], M_CYAN, 4.0, 1.0)
        lbl_sin = mi(r"$\sin x$", "lbl_sin", C_CYAN, 0.17, 27).move_to(
            proj3(2.15, NMAX, math.sin(2.15)) + [0.20, 0.20, 0])
        lbl_n0 = mi(r"$n=0$", "lbl_n0", C_GRID, 0.15, 26).move_to(
            proj3(-3.45, 0, -1.35) + [-0.12, -0.14, 0])
        lbl_n5 = mi(r"$n=5$", "lbl_n5", C_GRID, 0.15, 26).move_to(
            proj3(-3.45, NMAX, -1.35) + [0.02, 0.04, 0])
        surf_static = Group(*floor_lines, floor_side, *surf_long, *surf_cross,
                            edge_sin, lbl_sin, lbl_n0, lbl_n5)

        # 3D 动态：当前阶数 P_n 红色曲线 + 扫描点 + 竖直高度线
        edge_cur = VMobject(color=M_RED, stroke_width=4.2, stroke_opacity=1.0)
        cur_dot = Dot(radius=0.055, color=M_ORANGE)
        cur_halo = Dot(radius=0.13, color=M_ORANGE, fill_opacity=0.22)
        cur_drop = VMobject(color=M_ORANGE, stroke_width=2.2, stroke_opacity=0.75)
        lbl_pn = mi(r"$P_n(x)$", "lbl_pn", C_RED, 0.17, 27)

        # ------------------------------------------------------------------
        # 2D 收敛图（静态网格 + 目标 sin x）
        # ------------------------------------------------------------------
        plot = Group()
        for xv in [-math.pi, -math.pi / 2, 0, math.pi / 2, math.pi]:
            plot.add(Line([x2(xv), AX2_Y - A2 * 1.15, 0],
                          [x2(xv), AX2_Y + A2 * 1.15, 0],
                          color=M_GRID_DIM, stroke_width=1.2, stroke_opacity=0.85))
        for vv in [-1, 0, 1]:
            plot.add(Line([x2(-math.pi), y2(vv), 0], [x2(math.pi), y2(vv), 0],
                          color=M_GRID_DIM if vv else M_GRID,
                          stroke_width=1.3 if vv else 1.8, stroke_opacity=0.95))
        lbl_pi = mi(r"$\pi$", "pi", C_GRID, 0.14, 26).move_to(
            [x2(math.pi) + 0.10, AX2_Y - 0.10, 0])
        lbl_mpi = mi(r"$-\pi$", "mpi", C_GRID, 0.14, 26).move_to(
            [x2(-math.pi) - 0.14, AX2_Y - 0.10, 0])
        lbl_0 = mi(r"$0$", "zero", C_GRID, 0.13, 26).move_to(
            [x2(0) - 0.10, AX2_Y - 0.12, 0])
        lbl_x = mi(r"$x$", "xaxis", C_GRID, 0.14, 26).move_to(
            [x2(math.pi) + 0.26, AX2_Y - 0.02, 0])
        plot.add(lbl_pi, lbl_mpi, lbl_0, lbl_x)
        sin2 = poly([[x2(x), y2(math.sin(x)), 0] for x in XS], M_CYAN, 3.4, 1.0)
        lbl_sin2 = mi(r"$\sin x$", "sin2", C_CYAN, 0.15, 26).move_to(
            [x2(-2.55), y2(math.sin(-2.55)) - 0.20, 0])

        # 2D 动态：P_n 曲线 + 误差竖线 + 扫描点 + 收敛区间
        cur2 = VMobject(color=M_RED, stroke_width=3.6, stroke_opacity=1.0)
        err_segs = [VMobject(color=M_ORANGE, stroke_width=1.5, stroke_opacity=0.0)
                    for _ in range(13)]
        dot2_sin = Dot(radius=0.045, color=M_CYAN)
        dot2_cur = Dot(radius=0.055, color=M_RED)
        conv_seg = VMobject(color=M_CYAN, stroke_width=3.0, stroke_opacity=0.9)
        conv_ticks = [VMobject(color=M_CYAN, stroke_width=2.0, stroke_opacity=0.9)
                      for _ in range(2)]
        lbl_conv = mi(r"$|R_n|\leq 0.12$", "conv", C_CYAN, 0.14, 25).move_to(
            [0, YC2 - A2 * 1.15 - 0.20, 0])

        # ------------------------------------------------------------------
        # 动态数值行
        # ------------------------------------------------------------------
        def num_row(expr, name, color, value, lab_h=0.16, places=2, px=28):
            lab = mi(expr, name, color, lab_h, 25)
            val = PixelNumber(value, places, color, px).set(height=0.20)
            return Group(lab, val).arrange(RIGHT, buff=0.05)

        row_n = num_row(r"$n=$", "lab_n", M_CYAN, 0, 0.16, 0)
        row_x = num_row(r"$x_0=$", "lab_x", M_RED, -math.pi, 0.16, 2)
        row_p = num_row(r"$P_n(x_0)=$", "lab_p", M_CYAN, 0, 0.16, 2)
        row_e = num_row(r"$|R_n(x_0)|=$", "lab_e", M_ORANGE, 0, 0.16, 3)
        row_n.move_to([-2.85, 1.08, 0])
        row_x.move_to([-0.95, 1.08, 0])
        row_p.move_to([0.85, 1.08, 0])
        row_e.move_to([2.85, 1.08, 0])
        num_n, num_x, num_p, num_e = row_n[1], row_x[1], row_p[1], row_e[1]

        # ------------------------------------------------------------------
        # 底部固定公式链
        # ------------------------------------------------------------------
        # 统一字号：按 PNG 原始高度反比设定显示高度（字形大小 ≈ 常数）
        f1 = mi(r"$\sin x=x-\frac{x^{3}}{3!}+\frac{x^{5}}{5!}-\frac{x^{7}}{7!}+\cdots$",
                "f1", C_FORMULA, 0.42, 30).move_to([0, -1.92, 0])
        f2 = mi(r"$P_{n}(x)=\sum_{k=0}^{n}(-1)^{k}\frac{x^{2k+1}}{(2k+1)!}$",
                "f2", C_FORMULA, 0.76, 30).move_to([0, -2.58, 0])
        f3 = mi(r"$\sin x=P_{n}(x)+R_{n}(x)$",
                "f3", C_FORMULA, 0.27, 30).move_to([0, -3.20, 0])
        f4 = mi(r"$|R_{n}(x)|\leq\frac{|x|^{2n+3}}{(2n+3)!}\rightarrow 0$",
                "f4", C_FORMULA, 0.48, 30).move_to([0, -3.82, 0])
        f5 = mi(r"$\lim_{n\to\infty}P_{n}(x)=\sin x\quad(-\infty<x<\infty)$",
                "f5", C_FORMULA, 0.39, 29).move_to([0, -4.46, 0])
        f5_ul = Line([-1.85, -4.74, 0], [1.85, -4.74, 0],
                     color=M_CYAN, stroke_width=2.4, stroke_opacity=0.0)
        formulas = Group(f1, f2, f3, f4, f5)

        # ------------------------------------------------------------------
        # 图层
        # ------------------------------------------------------------------
        self.add(surf_static, plot, sin2, lbl_sin2, lbl_conv, formulas)
        dynamic = Group(
            title, cursor,
            edge_cur, cur_dot, cur_halo, cur_drop, lbl_pn,
            cur2, *err_segs, dot2_sin, dot2_cur, conv_seg, *conv_ticks,
            row_n, row_x, row_p, row_e, f5_ul,
        )
        self.add(dynamic)

        # ------------------------------------------------------------------
        # 逐帧驱动
        # ------------------------------------------------------------------
        def apply(t):
            if t < 16.15:
                nf = NMAX * ease((t - 1.6) / 13.0)
                x0 = -math.pi + 2 * math.pi * tri((t - 1.2) / 7.0)
            else:
                u = clamp01((t - 16.15) / 5.0)
                nf = NMAX * ease(u / 0.72)
                x0 = -math.pi + 2 * math.pi * tri(u * 1.6)

            pv = P_cont(XS, nf)
            y0 = float(P_cont(np.array([x0]), nf)[0])
            s0 = math.sin(x0)
            err0 = abs(s0 - y0)

            # 3D 当前多项式红曲线（只绘制 0..NMAX 的完整深度范围内的当前切片）
            set_poly(edge_cur, [proj3(x, nf, v) for x, v in zip(XS, pv)])
            cp = proj3(x0, nf, y0)
            cur_dot.move_to(cp)
            cur_halo.move_to(cp)
            set_poly(cur_drop, [proj3(x0, nf, y0), proj3(x0, nf, -1.35)])
            lbl_pn.move_to(proj3(-2.72, nf, float(P_cont(np.array([-2.72]), nf)[0])) +
                           [-0.05, 0.20, 0])

            # 2D 当前多项式（裁剪到绘图区，避免低阶直线越界压到公式）
            ymin2, ymax2 = YC2 - A2 * 1.15, YC2 + A2 * 1.15
            pts2 = [[x2(x), y2(v), 0] for x, v in zip(XS, pv)]
            set_poly(cur2, visible_run(pts2, ymin2, ymax2))
            # 误差竖线（在若干采样点上，同样裁剪）
            err_x = np.linspace(-math.pi, math.pi, len(err_segs))
            for seg, ex in zip(err_segs, err_x):
                ev = float(P_cont(np.array([ex]), nf)[0])
                sv = math.sin(ex)
                e = abs(ev - sv)
                op = clamp01((e - 0.02) / 0.35) * 0.85
                set_poly(seg, [[x2(ex), clamp_y(y2(sv), ymin2, ymax2), 0],
                               [x2(ex), clamp_y(y2(ev), ymin2, ymax2), 0]])
                seg.set_stroke(M_ORANGE, width=1.5, opacity=op)
            dot2_sin.move_to([x2(x0), y2(s0), 0])
            dot2_cur.move_to([x2(x0), y2(y0), 0])
            dot2_cur.set_opacity(1.0 if ymin2 <= y2(y0) <= ymax2 else 0.0)

            # 收敛区间指示
            a = approx_interval(nf)
            yy = YC2 - A2 * 1.15 - 0.06
            conv_seg.set_stroke(M_CYAN, width=3.0, opacity=0.9)
            set_poly(conv_seg, [[x2(-a), yy, 0], [x2(a), yy, 0]])
            for tick, sgn in zip(conv_ticks, (-1, 1)):
                tick.set_stroke(M_CYAN, width=2.0, opacity=0.9)
                set_poly(tick, [[x2(sgn * a), yy - 0.06, 0],
                                [x2(sgn * a), yy + 0.06, 0]])

            # 动态数值
            num_n.set_value(round(nf))
            num_x.set_value(x0)
            num_p.set_value(y0)
            num_e.set_value(err0)

            # 末行高亮：收敛区间足够大后出现
            op = clamp01((a - 2.0) / 0.8)
            f5_ul.set_stroke(M_CYAN, width=2.4, opacity=0.85 * op)

            # 循环重置瞬间标题闪烁
            blink = 16.02 <= t <= 16.36
            title.set_opacity(1)
            cursor.set_opacity(0)

        apply(0)
        self.play(FadeIn(formulas), run_time=0.7)
        self.play(FadeIn(title), run_time=0.3)

        class Timeline(Animation):
            def __init__(self, mobject, **kw):
                super().__init__(mobject, run_time=DURATION, **kw)

            def interpolate_mobject(self, alpha):
                apply(alpha * DURATION)

        self.play(Timeline(dynamic), rate_func=linear)
        apply(DURATION - 1 / config.frame_rate)
        self.wait(0.08)
