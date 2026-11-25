#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Manim 数学动画：最速降线（Brachistochrone）—— 变分法寻找最快下滑曲线

视觉规范与高斯积分 / 复指数映射 / 傅里叶级数 / 泰勒展开成片一致：
  - 1080x1350 竖屏、30fps、约 22s
  - 公式全程固定，只动参数、网格与动画
  - 无 LaTeX：公式用 matplotlib mathtext 预渲染为透明 PNG
  - 暗色底 #101113、中灰清晰线条、青/红边缘曲线、纯内存动态数字

画面：
  顶部 = 3D 摆线族"瀑布"曲面（深度 = 参数 a，青 = 过终点 B 的最优摆线，红 = 当前候选）
  中部 = 2D 四轨道竞速（直线 / 折线 / 抛物线 / 摆线） + 到达时间条 + 动态数值行
  底部 = 固定五行动画公式链

数学主线（A 在原点，y 轴向下为正，v = sqrt(2gy)）：
  T = ∫ ds/v = ∫ sqrt(1+y'^2)/sqrt(2gy) dx
  L = sqrt(1+y'^2)/sqrt(2y)；Beltrami: L - y' L_y' = 1/sqrt(2y(1+y'^2)) = C
  => y(1+y'^2) = 2a，其解为摆线 x=a(θ-sinθ), y=a(1-cosθ)
  沿摆线 T(θ) = θ sqrt(a/g)，故摆线即最速降线

运行：
    manim -pql brachistochrone_scene.py Brachistochrone
    manim -pqh brachistochrone_scene.py Brachistochrone
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
RESET_T = 16.15

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
    path = FORMULA_DIR / f"bs_{name}.png"
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


def clamp01(v):
    return float(np.clip(v, 0.0, 1.0))


def ease(v):
    v = clamp01(v)
    return v * v * (3 - 2 * v)


# ---------------------------------------------------------------------------
# 物理：摆线参数求解与四条轨道的时间表（A=(0,0), B=(6,4), g=9.81, y 向下为正）
# ---------------------------------------------------------------------------
G = 9.81
XB = 6.0
YB = 4.0


def cyclo_xy(a, theta):
    th = np.asarray(theta, dtype=float)
    return a * (th - np.sin(th)), a * (1 - np.cos(th))


def theta_at_y(a, y=YB):
    c = float(np.clip(1.0 - y / a, -1.0, 1.0))
    return math.acos(c)


def x_end(a):
    th = theta_at_y(a)
    return a * (th - math.sin(th))


def solve_astar():
    lo, hi = YB / 2 + 1e-5, 12.0     # x_end 随 a 单调减小
    for _ in range(90):
        mid = 0.5 * (lo + hi)
        if x_end(mid) > XB:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


ASTAR = solve_astar()
THB = theta_at_y(ASTAR)
TCYC = THB * math.sqrt(ASTAR / G)


def _build_smooth_track(kind):
    """用 x = XB*u^2 换元消除 x=0 处 dy/dx 奇性，返回 (x, y, t) 数组。"""
    n = 1400
    u = np.linspace(0.0, 1.0, n)
    x = XB * u * u
    if kind == "line":
        y = YB * x / XB
        dy = YB / XB * np.ones_like(x)
    elif kind == "parabola":
        r = x / XB
        y = YB * (1.5 * r - 0.5 * r * r)
        dy = YB * (1.5 - x / XB) / XB
    else:
        raise ValueError(kind)
    w = 2 * XB * u * np.sqrt(1 + dy * dy) / np.sqrt(2 * G * np.maximum(y, 1e-12))
    dt = np.empty(n)
    dt[0] = 0.0
    dt[1:] = 0.5 * (w[1:] + w[:-1]) * (u[1] - u[0])
    t = np.cumsum(dt)
    return x, y, t


def _build_broken():
    n = 500
    y = np.linspace(0.0, YB, n)
    tv = np.sqrt(2 * y / G)                       # 竖直段 v = sqrt(2gy)
    x1 = np.zeros(n)
    t1 = tv
    x2 = np.linspace(0.0, XB, n)
    vh = math.sqrt(2 * G * YB)                     # 水平段匀速
    t2 = t1[-1] + x2 / vh
    return (np.concatenate([x1, x2[1:]]),
            np.concatenate([y, np.full(n - 1, YB)]),
            np.concatenate([t1, t2[1:]]))


def _build_cyclo():
    n = 1000
    th = np.linspace(0.0, THB, n)
    x, y = cyclo_xy(ASTAR, th)
    t = th * math.sqrt(ASTAR / G)
    return x, y, t


TRACK_DATA = {
    "cycloid": _build_cyclo(),
    "parabola": _build_smooth_track("parabola"),
    "broken": _build_broken(),
    "line": _build_smooth_track("line"),
}
TRACK_T = {k: float(v[2][-1]) for k, v in TRACK_DATA.items()}
TR_MAX = 1.8


def track_pos(kind, t):
    """按物理时间返回轨道上的 (x, y)，越界则停在终点。"""
    x, y, ta = TRACK_DATA[kind]
    if t <= 0:
        return 0.0, 0.0
    if t >= ta[-1]:
        return float(x[-1]), float(y[-1])
    i = int(np.searchsorted(ta, t))
    i = max(1, min(i, len(ta) - 1))
    f = (t - ta[i - 1]) / max(1e-12, ta[i] - ta[i - 1])
    return float(x[i - 1] + f * (x[i] - x[i - 1])), float(y[i - 1] + f * (y[i] - y[i - 1]))


# ---------------------------------------------------------------------------
# 3D 摆线族瀑布：depth = 参数 nf（0 = 最优 a*，1 = 偏离的 a）
# ---------------------------------------------------------------------------
AMAX = 3.8
C3 = np.array([-3.45, 3.98, 0.0])
SX3 = 6.18
DX3 = 0.10
DY3 = -0.075
VS3 = 1.55
FLOOR_V = 1.14


def a_of(nf):
    return ASTAR + clamp01(nf) * (AMAX - ASTAR)


def proj3(u, nf, v):
    return np.array([C3[0] + SX3 * u + DX3 * nf,
                     C3[1] + DY3 * nf - VS3 * v, 0.0])


def cyclo_v(a, theta):
    """给定参数 a 与角度 theta，返回归一化 (u=x/XB, v=y/YB)。"""
    x, y = cyclo_xy(a, theta)
    return x / XB, y / YB


def proj_cyclo(a, theta, nf):
    """把摆线上一点投到 3D：注意 proj3 的参数次序是 (u, nf, v)。"""
    u, v = cyclo_v(a, theta)
    return proj3(u, nf, v)


def cyclo_curve3(nf, n=90):
    a = a_of(nf)
    th = np.linspace(0.0, theta_at_y(a), n)
    return [proj_cyclo(a, t, nf) for t in th]


# 2D 竞速场（A 在左上，B 在右下）
AX, AY = -3.30, 0.85
BX, BY = 3.30, -0.85


def arena(u, v):
    return np.array([AX + (BX - AX) * u, AY + (BY - AY) * v, 0.0])


def track_pts2(kind):
    x, y, _ = TRACK_DATA[kind]
    return [arena(xi / XB, yi / YB) for xi, yi in zip(x[::6], y[::6])]


# ---------------------------------------------------------------------------
# 场景
# ---------------------------------------------------------------------------
class Brachistochrone(Scene):
    def construct(self):
        # 顶部标题（AIMaker-Lab 系列标识 + 闪烁光标）
        title = text_img("AIMaker-Lab", 52, C_TEXT, 0.28).move_to([0, 4.35, 0])
        cursor = Line(UP * 0.16, DOWN * 0.16, color=M_TEXT,
                      stroke_width=2.5).move_to([0, 4.35, 0])
        cursor.set_opacity(0)

        # ------------------------------------------------------------------
        # 顶部 3D 摆线族瀑布（静态灰线框 + 青色最优边线）
        # ------------------------------------------------------------------
        nf_grid = np.linspace(0, 1, 10)
        cross = [poly(cyclo_curve3(nf), M_GRID, 1.3, 0.58) for nf in nf_grid]
        longi = []
        for s in np.linspace(0.06, 0.96, 11):
            pts = []
            for nf in np.linspace(0, 1, 26):
                a = a_of(nf)
                th = s * theta_at_y(a)
                pts.append(proj_cyclo(a, th, nf))
            longi.append(poly(pts, M_GRID_DIM, 1.1, 0.55))
        floor_lines = [poly([proj3(u, 0, FLOOR_V), proj3(u, 1, FLOOR_V)],
                            M_GRID_DIM, 1.0, 0.42)
                       for u in np.linspace(0, 1, 9)]
        floor_side = poly([proj3(0, nf, FLOOR_V) for nf in np.linspace(0, 1, 24)] +
                          [proj3(1, nf, FLOOR_V) for nf in np.linspace(1, 0, 24)],
                          M_GRID_DIM, 1.0, 0.42)
        edge_opt = poly(cyclo_curve3(0.0), M_CYAN, 4.0, 1.0)
        lbl_a0 = mi(r"$a=a^*$", "a0", C_CYAN, 0.15, 26).move_to(
            proj3(0.74, 0, FLOOR_V) + [0.06, -0.10, 0])
        lbl_amax = mi(r"$a=3.8$", "amax", C_GRID, 0.15, 26).move_to(
            proj3(0.38, 1, FLOOR_V) + [-0.16, -0.10, 0])
        lbl_3d = mi(r"$x=a(\theta-\sin\theta),\;y=a(1-\cos\theta)$",
                    "lbl3d", C_GRID, 0.14, 25).move_to([1.15, 1.86, 0])
        dot_a = Dot(radius=0.05, color=M_CYAN).move_to(proj3(1, 0, 1))
        dot_a0 = Dot(radius=0.05, color=M_CYAN).move_to(proj3(0, 0, 0))
        surf_static = Group(*floor_lines, floor_side, *longi, *cross,
                            edge_opt, lbl_a0, lbl_amax, lbl_3d, dot_a, dot_a0)

        # 3D 动态：当前候选摆线（红）+ 探针点 + 垂线
        edge_cur3 = VMobject(color=M_RED, stroke_width=4.2, stroke_opacity=1.0)
        probe3 = Dot(radius=0.055, color=M_ORANGE)
        halo3 = Dot(radius=0.13, color=M_ORANGE, fill_opacity=0.22)
        drop3 = VMobject(color=M_ORANGE, stroke_width=2.2, stroke_opacity=0.75)

        # ------------------------------------------------------------------
        # 中部 2D 竞速场（静态四轨道 + A/B 标记 + 时间条底轴）
        # ------------------------------------------------------------------
        arena_static = Group()
        # 淡网格
        for u in np.linspace(0, 1, 7):
            arena_static.add(Line(arena(u, 0), arena(u, 1),
                                  color=M_GRID_DIM, stroke_width=1.0,
                                  stroke_opacity=0.38))
        for v in [0, 0.5, 1]:
            arena_static.add(Line(arena(0, v), arena(1, v),
                                  color=M_GRID_DIM, stroke_width=1.0,
                                  stroke_opacity=0.42))
        trk_line = poly(track_pts2("line"), M_GRID, 2.0, 0.85)
        trk_broken = poly(track_pts2("broken"), M_ORANGE, 2.0, 0.72)
        trk_parab = poly(track_pts2("parabola"), M_RED, 2.2, 0.78)
        trk_cycl = poly(track_pts2("cycloid"), M_CYAN, 3.6, 1.0)
        lbl_A = mi(r"$A$", "A", C_TEXT, 0.15, 26).move_to(arena(0, 0) + [-0.17, 0.16, 0])
        lbl_B = mi(r"$B$", "B", C_TEXT, 0.15, 26).move_to(arena(1, 1) + [0.16, -0.14, 0])
        arena_static.add(trk_line, trk_broken, trk_parab, trk_cycl, lbl_A, lbl_B)

        # 候选曲线（搜寻阶段在 2D 场中同步变形）
        cur2 = VMobject(color=M_RED, stroke_width=2.2, stroke_opacity=0.0)
        probe2 = Dot(radius=0.055, color=M_ORANGE)

        # 竞速小球（四个）+ 光晕
        beads, halos = {}, {}
        for key, col in [("cycloid", M_CYAN), ("parabola", M_RED),
                         ("broken", M_ORANGE), ("line", M_GRID)]:
            beads[key] = Dot(radius=0.062, color=col)
            halos[key] = Dot(radius=0.14, color=col, fill_opacity=0.20)

        # 时间条（每条轨道一行，长度 = 当前物理时间）
        bar_y = {"cycloid": -1.02, "parabola": -1.15, "broken": -1.28, "line": -1.41}
        bar_col = {"cycloid": M_CYAN, "parabola": M_RED,
                   "broken": M_ORANGE, "line": M_GRID}
        bars = {}
        for key in bar_y:
            bars[key] = Line([AX, bar_y[key], 0], [AX, bar_y[key], 0],
                             color=bar_col[key], stroke_width=4.0,
                             stroke_opacity=0.92)
        bar_axis = Line([AX, -1.52, 0], [BX, -1.52, 0],
                        color=M_GRID_DIM, stroke_width=1.2, stroke_opacity=0.8)
        bar_t0 = mi(r"$0$", "bar0", C_GRID, 0.12, 24).move_to([AX, -1.65, 0])
        bar_t1 = mi(r"$1.8\,\mathrm{s}$", "bar18", C_GRID, 0.12, 24).move_to([BX, -1.65, 0])
        time_static = Group(bar_axis, bar_t0, bar_t1)

        # ------------------------------------------------------------------
        # 动态数值行（搜寻阶段 / 竞速阶段各一组）
        # ------------------------------------------------------------------
        def num_row(expr, name, color, value, lab_h=0.16, places=2, px=28):
            lab = mi(expr, name, color, lab_h, 25)
            val = PixelNumber(value, places, color, px).set(height=0.20)
            return Group(lab, val).arrange(RIGHT, buff=0.05)

        seek_rows = [
            num_row(r"$a=$", "lab_a", M_RED, AMAX, 0.16, 3),
            num_row(r"$\theta_B=$", "lab_th", M_CYAN, 0, 0.16, 2),
            num_row(r"$x_{\mathrm{end}}=$", "lab_xe", M_RED, 0, 0.16, 2),
            num_row(r"$|\Delta x|=$", "lab_dx", M_ORANGE, 0, 0.16, 2),
        ]
        race_rows = [
            num_row(r"$t=$", "lab_t", M_ORANGE, 0, 0.16, 2),
            num_row(r"$T_{\mathrm{cyc}}=$", "lab_tc", M_CYAN, 0, 0.16, 2),
            num_row(r"$T_{\mathrm{line}}=$", "lab_tl", C_GRID, 0, 0.16, 2),
            num_row(r"$\Delta=$", "lab_dt", M_CYAN, 0, 0.16, 1),
        ]
        for rows in (seek_rows, race_rows):
            for row, x in zip(rows, [-2.85, -0.95, 0.85, 2.85]):
                row.move_to([x, 1.35, 0])
        num_a, num_th, num_xe, num_dx = [r[1] for r in seek_rows]
        num_t, num_tc, num_tl, num_dt = [r[1] for r in race_rows]
        seek_group = Group(*seek_rows)
        race_group = Group(*race_rows)

        # ------------------------------------------------------------------
        # 底部固定公式链
        # ------------------------------------------------------------------
        # 统一字号：显示高度 ∝ PNG 原始像素高度（字形大小 ≈ 0.11 世界单位）
        f1 = mi(r"$T=\int_{A}^{B}\frac{ds}{v}=\int_{0}^{x_B}\frac{\sqrt{1+y'^2}}{\sqrt{2gy}}\,dx$",
                "f1", C_FORMULA, 0.704, 30).move_to([0, -1.98, 0])
        f2 = mi(r"$L(y,y')=\frac{\sqrt{1+y'^2}}{\sqrt{2gy}}\quad(v=\sqrt{2gy})$",
                "f2", C_FORMULA, 0.664, 30).move_to([0, -2.66, 0])
        f3 = mi(r"$L-y'\frac{\partial L}{\partial y'}=\frac{1}{\sqrt{2y(1+y'^2)}}=C$",
                "f3", C_FORMULA, 0.539, 30).move_to([0, -3.29, 0])
        f4 = mi(r"$y(1+y'^2)=2a\;\Rightarrow\;x=a(\theta-\sin\theta),\;y=a(1-\cos\theta)$",
                "f4", C_FORMULA, 0.348, 30).move_to([0, -3.78, 0])
        f5 = mi(r"$T_{\min}=\theta_B\sqrt{a/g}\quad(\mathrm{cycloid})$",
                "f5", C_FORMULA, 0.417, 29).move_to([0, -4.32, 0])
        f5_ul = Line([-1.85, -4.62, 0], [1.85, -4.62, 0],
                     color=M_CYAN, stroke_width=2.4, stroke_opacity=0.0)
        formulas = Group(f1, f2, f3, f4, f5)

        # ------------------------------------------------------------------
        # 图层
        # ------------------------------------------------------------------
        self.add(surf_static, arena_static, time_static, formulas)
        dynamic = Group(
            title, cursor, edge_cur3, probe3, halo3, drop3,
            cur2, probe2,
            beads["cycloid"], beads["parabola"], beads["broken"], beads["line"],
            halos["cycloid"], halos["parabola"], halos["broken"], halos["line"],
            *bars.values(), f5_ul, seek_group, race_group,
        )
        self.add(dynamic)

        # ------------------------------------------------------------------
        # 逐帧驱动
        # ------------------------------------------------------------------
        def apply(t):
            seeking = t < RESET_T
            if seeking:
                if t < 1.6:
                    nf = 1.0
                    tr = 0.0
                elif t < 7.2:
                    nf = 1.0 - ease((t - 1.6) / 5.0)     # a: 3.8 -> a*
                    tr = 0.0
                elif t < 14.4:
                    nf = 0.0
                    tr = clamp01((t - 7.2) / 6.6) * 1.72  # 四球竞速
                else:
                    nf, tr = 0.0, 1.72                      # 定格
            else:
                u = clamp01((t - RESET_T) / (DURATION - RESET_T))
                nf = 1.0 - ease(u / 0.34)
                tr = clamp01((u - 0.30) / 0.52) * 1.72

            a = a_of(nf)
            th_b = theta_at_y(a)
            xe = x_end(a)
            err = abs(xe - XB)

            # 3D 当前候选摆线 + 探针点（沿曲线往返扫描）+ 垂线
            pts3 = cyclo_curve3(nf)
            set_poly(edge_cur3, pts3)
            edge_cur3.set_stroke(M_RED, width=4.2,
                                 opacity=0.92 if seeking and t < 7.4 else 0.0)
            scan = (t % 2.4) / 2.4
            scan = scan if scan < 0.5 else 1.0 - scan
            th_scan = 2 * scan * th_b
            pp = np.array(proj_cyclo(a, th_scan, nf))
            probe3.move_to(pp)
            halo3.move_to(pp)
            floor_pp = np.array([pp[0], C3[1] + DY3 * nf - VS3 * FLOOR_V, 0.0])
            set_poly(drop3, [pp, floor_pp])
            op3 = 1.0 if seeking and t < 7.4 else 0.0
            probe3.set_opacity(op3)
            halo3.set_opacity(0.22 * op3)
            drop3.set_stroke(M_ORANGE, width=2.2, opacity=0.75 * op3)

            # 2D 候选曲线（搜寻阶段）与探针
            a2 = a
            th2 = np.linspace(0.0, theta_at_y(a2), 120)
            x2, y2 = cyclo_xy(a2, th2)
            set_poly(cur2, [arena(xi / XB, yi / YB) for xi, yi in zip(x2, y2)])
            cur2.set_stroke(M_RED, width=2.2,
                            opacity=0.75 if seeking and t < 7.4 else 0.0)
            probe2.move_to(arena(float(x2[int(scan * (len(x2) - 1))]) / XB,
                                 float(y2[int(scan * (len(y2) - 1))]) / YB))
            probe2.set_opacity(0.9 if seeking and t < 7.4 else 0.0)

            # 竞速小球与光晕
            for key in beads:
                bx, by = track_pos(key, tr)
                pos = arena(bx / XB, by / YB)
                beads[key].move_to(pos)
                halos[key].move_to(pos)
                vis = 1.0 if tr > 0.02 else 0.0
                beads[key].set_opacity(vis)
                halos[key].set_opacity(0.20 * vis)

            # 时间条
            for key in bars:
                frac = clamp01(min(tr, TRACK_T[key]) / TR_MAX)
                set_poly(bars[key], [[AX, bar_y[key], 0],
                                     [AX + (BX - AX) * frac, bar_y[key], 0]])

            # 数值行：互斥显示用「移出画面」实现（PixelNumber 的 alpha 由 set_value 全量重写）
            num_a.set_value(a)
            num_th.set_value(th_b)
            num_xe.set_value(xe)
            num_dx.set_value(err)
            num_t.set_value(tr)
            num_tc.set_value(TRACK_T["cycloid"])
            num_tl.set_value(TRACK_T["line"])
            num_dt.set_value(100 * (TRACK_T["line"] - TRACK_T["cycloid"]) /
                             TRACK_T["line"])
            seek_y = 1.35 if (seeking and t < 7.4) else 50.0
            race_y = 1.35 if t >= 7.2 else 50.0
            seek_group.move_to([0, seek_y, 0])
            race_group.move_to([0, race_y, 0])

            # 最优解出现后末行高亮
            op = clamp01((0.35 - err) / 0.35)
            f5_ul.set_stroke(M_CYAN, width=2.4, opacity=0.85 * op)

            # 循环重置瞬间：标题闪烁
            blink = RESET_T - 0.13 <= t <= RESET_T + 0.21
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
