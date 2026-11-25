#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Manim 数学动画：复指数映射 w = e^z（共形映射）

画面继承高斯积分复刻片的视觉规范：
  - 公式全程固定，只动参数、网格与动画
  - 顶部标题 + 3D 线框曲面 + 左右双平面 + 底部固定公式链
  - 无 LaTeX：公式用 matplotlib mathtext 预渲染为透明 PNG

运行：
    manim -pql complex_exp_scene.py ComplexExponential
    manim -pqh complex_exp_scene.py ComplexExponential
输出：
    media/videos/complex_exp_scene/<quality>/ComplexExponential.mp4
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
# 配置（1080x1350 竖屏，与高斯积分成片一致）
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
    path = FORMULA_DIR / f"cx_{name}.png"
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
    """PIL 渲染到固定 RGBA 数组，set_value 原地更新，无临时文件 IO。"""

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


def poly(points, color, width=1.2, opacity=1.0, fill_opacity=None):
    m = VMobject(color=color, stroke_width=width, stroke_opacity=opacity)
    m.set_points_as_corners(points)
    if fill_opacity is not None:
        m.set_fill(color, opacity=fill_opacity)
    return m


def bracket(x, y_top, y_bot, side="left", color=M_TEXT):
    tick = 0.07 if side == "left" else -0.07
    return VGroup(
        Line([x + tick, y_top, 0], [x, y_top, 0], color=color, stroke_width=1.9),
        Line([x, y_top, 0], [x, y_bot, 0], color=color, stroke_width=1.9),
        Line([x, y_bot, 0], [x + tick, y_bot, 0], color=color, stroke_width=1.9),
    )


def set_poly(mob, points, closed=False):
    pts = points + [points[0]] if closed else points
    mob.set_points_as_corners(pts)


def clamp01(v):
    return float(np.clip(v, 0.0, 1.0))


def ease(v):
    v = clamp01(v)
    return v * v * (3 - 2 * v)


# ---------------------------------------------------------------------------
# 3D 螺旋面：w = e^z 的像（对数螺线面），手动正交投影
# ---------------------------------------------------------------------------
RHO_SCALE = 0.72
Z_SCALE = 1.00
C3 = np.array([0.0, 2.42, 0.0])
PHI3 = math.radians(30)
SX3, SY3 = 0.95, 0.85
TILT3, ZS3 = 0.45, 0.95


def helicoid(x, y):
    rho = RHO_SCALE * math.exp(x)
    return np.array([rho * math.cos(y), rho * math.sin(y), Z_SCALE * x])


def proj3(p):
    c, s = math.cos(PHI3), math.sin(PHI3)
    x3 = p[0] * c - p[1] * s
    y3 = p[0] * s + p[1] * c
    return C3 + np.array([SX3 * x3, SY3 * (TILT3 * y3 + ZS3 * p[2]), 0.0])


# z 平面（直角网格）与 w 平面（极坐标网格）
ZC = np.array([-2.15, 0.10, 0.0])
WC = np.array([2.15, 0.10, 0.0])
ZSX, ZSY = 0.90, 0.92
WRS = 0.55


def z_point(x, y):
    """z=x+iy 在左平面的显示坐标；x∈[-1,1], y∈[0,2π]"""
    return ZC + np.array([ZSX * x, ZSY * (y / TAU * 2 - 1), 0.0])


def w_point(x, y):
    """w=e^z 在右平面的显示坐标；rho=e^x, phi=y"""
    rho = WRS * math.exp(x)
    return WC + np.array([rho * math.cos(y), rho * math.sin(y), 0.0])


def z_as_w_point(x, y):
    """把 z 平面的直角网格按 w 平面尺度摆放，作为变形的起点（不跨平面平移）"""
    return WC + np.array([WRS * 0.95 * x, WRS * 0.95 * (y / TAU * 2 - 1), 0.0])


# ---------------------------------------------------------------------------
# 场景
# ---------------------------------------------------------------------------
class ComplexExponential(Scene):
    def construct(self):
        # ------------------------------------------------------------------
        # 顶部标题
        # ------------------------------------------------------------------
        title = text_img("AIMaker-Lab", 52, C_TEXT, 0.28).move_to([0, 4.35, 0])
        cursor = Line(UP * 0.16, DOWN * 0.16, color=M_TEXT,
                      stroke_width=2.5).move_to([0, 4.35, 0])
        cursor.set_opacity(0)

        # ------------------------------------------------------------------
        # 3D 螺旋面静态线框
        # ------------------------------------------------------------------
        xs = np.linspace(-1.0, 1.0, 15)
        ys = np.linspace(0.0, TAU, 40)
        surf_radial = [
            poly([proj3(helicoid(x, y)) for x in np.linspace(-1, 1, 24)],
                 M_GRID, 1.5, 0.92)
            for y in np.linspace(0, TAU, 13, endpoint=False)
        ]
        surf_circles = [
            poly([proj3(helicoid(x, y)) for y in ys], M_GRID, 1.55, 1.0)
            for x in xs
        ]
        edge_min = poly([proj3(helicoid(-1.0, y)) for y in ys], M_CYAN, 4.0)
        edge_max = poly([proj3(helicoid(1.0, y)) for y in ys], M_RED, 4.0)
        edge_start = poly([proj3(helicoid(x, 0.0)) for x in np.linspace(-1, 1, 30)],
                          M_ORANGE, 3.2, 0.85)
        edge_end = poly([proj3(helicoid(x, TAU)) for x in np.linspace(-1, 1, 30)],
                        M_ORANGE, 3.2, 0.85)

        lbl_z = mi(r"$z$", "ax_z", C_GRID, 0.15, 28).move_to(
            proj3(helicoid(-1.0, 0.0)) + [-0.16, -0.12, 0])
        lbl_w = mi(r"$w=e^z$", "ax_w", C_CYAN, 0.17, 28).move_to(
            proj3(helicoid(1.0, math.pi * 0.55)) + [0.18, 0.16, 0])

        surf_static = Group(*surf_radial, *surf_circles,
                            edge_min, edge_max, edge_start, edge_end,
                            lbl_z, lbl_w)

        # 曲面上 current (x, y) 的高亮：青色圆弧 + 红色径向线 + 光点
        surf_arc = VMobject(color=M_CYAN, stroke_width=3.4, stroke_opacity=0.98)
        surf_ray = VMobject(color=M_RED, stroke_width=3.0, stroke_opacity=0.95)
        surf_dot = Dot(radius=0.055, color=M_ORANGE)
        surf_dot_halo = Dot(radius=0.13, color=M_ORANGE, fill_opacity=0.22)

        # ------------------------------------------------------------------
        # 左平面：z 直角网格（静态）+ 高亮 x 竖线 / y 横线
        # ------------------------------------------------------------------
        z_grid = Group()
        for k in np.linspace(-1, 1, 9):
            z_grid.add(Line(z_point(k, 0), z_point(k, TAU),
                            color=M_GRID_DIM, stroke_width=1.3, stroke_opacity=0.95))
        for k in np.linspace(0, TAU, 13, endpoint=False):
            z_grid.add(Line(z_point(-1, k), z_point(1, k),
                            color=M_GRID_DIM, stroke_width=1.3, stroke_opacity=0.95))
        z_grid.add(Line(z_point(-1, 0), z_point(1, 0),
                        color=M_GRID, stroke_width=1.9, stroke_opacity=1.0))
        z_grid.add(Line(z_point(0, 0), z_point(0, TAU),
                        color=M_GRID, stroke_width=1.9, stroke_opacity=1.0))
        lbl_zx = mi(r"$x$", "zx", C_GRID, 0.14, 26).move_to(z_point(1, 0) + [0.12, -0.08, 0])
        lbl_zy = mi(r"$iy$", "zy", C_GRID, 0.14, 26).move_to(z_point(0, TAU) + [-0.16, 0.10, 0])
        z_grid.add(lbl_zx, lbl_zy)

        z_vert = VMobject(color=M_CYAN, stroke_width=3.0, stroke_opacity=0.98)
        z_horiz = VMobject(color=M_RED, stroke_width=3.0, stroke_opacity=0.98)

        # ------------------------------------------------------------------
        # 右平面：w 极坐标网格（静态）+ 高亮圆 rho / 射线 phi
        # ------------------------------------------------------------------
        w_grid = Group()
        for k in range(1, 7):
            w_grid.add(Circle(radius=WRS * math.exp(-1 + k / 3),
                              color=M_GRID_DIM, stroke_width=1.3,
                              stroke_opacity=0.95).move_to(WC))
        for k in range(12):
            a = TAU * k / 12
            d = np.array([math.cos(a), math.sin(a), 0])
            w_grid.add(Line(WC, WC + WRS * math.exp(1) * d,
                            color=M_GRID_DIM, stroke_width=1.3, stroke_opacity=0.95))
        w_grid.add(Line(WC + LEFT * (WRS * math.exp(1) + 0.08),
                        WC + RIGHT * (WRS * math.exp(1) + 0.08),
                        color=M_GRID, stroke_width=1.9, stroke_opacity=1.0))
        w_grid.add(Line(WC + DOWN * (WRS * math.exp(1) + 0.08),
                        WC + UP * (WRS * math.exp(1) + 0.08),
                        color=M_GRID, stroke_width=1.9, stroke_opacity=1.0))
        lbl_wr = mi(r"$\rho$", "wr", C_GRID, 0.14, 26).move_to(
            WC + RIGHT * (WRS * math.exp(1) + 0.18))
        lbl_wp = mi(r"$\varphi$", "wp", C_GRID, 0.14, 26).move_to(
            WC + UP * (WRS * math.exp(1) + 0.16))
        w_grid.add(lbl_wr, lbl_wp)

        w_circle = VMobject(color=M_CYAN, stroke_width=3.0, stroke_opacity=0.98)
        w_ray = VMobject(color=M_RED, stroke_width=3.0, stroke_opacity=0.98)

        # ------------------------------------------------------------------
        # 网格变形线：z 网格点 -> w 网格点的连续插值
        # ------------------------------------------------------------------
        MORPH_X = np.linspace(-1, 1, 7)
        MORPH_Y = np.linspace(0, TAU, 7, endpoint=False)
        morph_v = [
            VMobject(color=M_CYAN, stroke_width=2.2, stroke_opacity=0.0)
            for _ in MORPH_X
        ]
        morph_h = [
            VMobject(color=M_RED, stroke_width=2.2, stroke_opacity=0.0)
            for _ in MORPH_Y
        ]

        # 保角演示：z 平面小圆 -> w 平面小圆
        cst_z = Circle(radius=0.17, color=M_ORANGE, stroke_width=2.6,
                       stroke_opacity=0.0)
        cst_w = Circle(radius=0.17, color=M_ORANGE, stroke_width=2.6,
                       stroke_opacity=0.0)

        # ------------------------------------------------------------------
        # 动态数值：每组「标签 + 数值」整体排布，避免横向挤压重叠
        # ------------------------------------------------------------------
        def num_row(expr, name, color, value, lab_h=0.20):
            lab = mi(expr, name, color, lab_h, 26)
            val = PixelNumber(value, 2, color, 28).set(height=0.20)
            return Group(lab, val).arrange(RIGHT, buff=0.05)

        row_x = num_row(r"$x=$", "lab_x", M_CYAN, -1.0, 0.15).move_to([-2.15, -1.52, 0])
        row_y = num_row(r"$y=\varphi=$", "lab_y", M_RED, 0.0).move_to([0.0, -1.52, 0])
        row_rho = num_row(r"$\rho=e^x=$", "lab_rho", M_CYAN, 0.37).move_to([2.15, -1.52, 0])
        row_det = num_row(r"$\det J=e^{2x}=$", "lab_det", M_RED, 0.14).move_to([0.0, 0.58, 0])
        num_x, num_y, num_rho, num_det = row_x[1], row_y[1], row_rho[1], row_det[1]

        # ------------------------------------------------------------------
        # 底部固定公式链
        # ------------------------------------------------------------------
        f1 = mi(r"$z=x+iy$", "f1", C_FORMULA, 0.30, 32).move_to([0, -1.88, 0])
        f2 = mi(r"$w=e^{z}=e^{x}(\cos y+i\sin y)$", "f2",
                C_FORMULA, 0.32, 30).move_to([0, -2.52, 0])
        f3 = mi(r"$u=e^{x}\cos y,\quad v=e^{x}\sin y$", "f3",
                C_FORMULA, 0.30, 30).move_to([0, -3.14, 0])

        f4a = mi(r"$J=\mathrm{det}$", "f4a", C_FORMULA, 0.26, 28).move_to([-2.55, -3.84, 0])
        b1 = bracket(-1.88, -3.60, -4.10, "left", M_FORMULA)
        b2 = bracket(-0.10, -3.60, -4.10, "right", M_FORMULA)
        m00 = mi(r"$e^{x}\cos y$", "m00", C_FORMULA, 0.19, 24).move_to([-1.42, -3.71, 0])
        m01 = mi(r"$-e^{x}\sin y$", "m01", C_FORMULA, 0.19, 24).move_to([-0.68, -3.71, 0])
        m10 = mi(r"$e^{x}\sin y$", "m10", C_FORMULA, 0.19, 24).move_to([-1.42, -3.99, 0])
        m11 = mi(r"$e^{x}\cos y$", "m11", C_FORMULA, 0.19, 24).move_to([-0.68, -3.99, 0])
        f4c = mi(r"$=e^{2x}$", "f4c", C_FORMULA, 0.28, 30).move_to([0.65, -3.84, 0])

        f5 = mi(r"$f'(z)=e^{z}\neq 0\quad\Rightarrow\quad\mathrm{conformal}$",
                "f5", C_FORMULA, 0.31, 30).move_to([0, -4.52, 0])
        f5_ul = Line([-1.55, -4.76, 0], [1.55, -4.76, 0],
                     color=M_CYAN, stroke_width=2.4, stroke_opacity=0.0)

        formulas = Group(f1, f2, f3, f4a, b1, b2, m00, m01, m10, m11, f4c, f5)

        # ------------------------------------------------------------------
        # 图层
        # ------------------------------------------------------------------
        self.add(surf_static, z_grid, w_grid, formulas)
        dynamic = Group(
            title, cursor,
            surf_arc, surf_ray, surf_dot, surf_dot_halo,
            z_vert, z_horiz, w_circle, w_ray,
            *morph_v, *morph_h, cst_z, cst_w,
            row_x, row_y, row_rho, row_det,
            f5_ul,
        )
        self.add(dynamic)

        # ------------------------------------------------------------------
        # 逐帧驱动
        # ------------------------------------------------------------------
        def apply(t):
            # 时间轴：x 扫描 -> y 扫描 -> 网格变形；16.15s 重置后快速重演
            if t < 16.15:
                x = -1.0 + 2.0 * ease((t - 1.8) / 5.2)
                y = TAU * ease((t - 7.0) / 5.4)
                m = ease((t - 12.4) / 3.7)
                mv = ease((t - 12.4) / 3.0)   # 竖线先弯成同心圆
                mh = ease((t - 13.6) / 3.0)   # 横线后弯成射线
            else:
                u = clamp01((t - 16.15) / 5.0)
                x = -1.0 + 2.0 * ease(u / 0.42)
                y = TAU * ease((u - 0.24) / 0.52)
                m = ease((u - 0.56) / 0.38)
                mv = ease((u - 0.56) / 0.30)
                mh = ease((u - 0.66) / 0.30)
            rho = math.exp(x)

            # 3D 曲面：当前位置的青弧 + 红径向线 + 光点
            set_poly(surf_arc, [proj3(helicoid(x, yy))
                                for yy in np.linspace(0, y, 36)] if y > 0.02
                     else [proj3(helicoid(x, 0.0))] * 2)
            set_poly(surf_ray, [proj3(helicoid(xx, y))
                                for xx in np.linspace(-1, x, 24)])
            surf_dot.move_to(proj3(helicoid(x, y)))
            surf_dot_halo.move_to(proj3(helicoid(x, y)))

            # z 平面高亮线
            z_vert.set_points_as_corners([z_point(x, 0), z_point(x, TAU)])
            z_horiz.set_points_as_corners([z_point(-1, y), z_point(1, y)])

            # w 平面高亮圆与射线
            set_poly(w_circle, [WC + WRS * rho * np.array([math.cos(a), math.sin(a), 0])
                                for a in np.linspace(0, TAU, 72)])
            w_ray.set_points_as_corners([WC, WC + WRS * rho *
                                         np.array([math.cos(y), math.sin(y), 0])])

            # 直角网格 -> 极坐标网格的连续变形
            if m < 0.002:
                for mob in morph_v + morph_h:
                    mob.set_stroke(M_CYAN, width=2.2, opacity=0.0)
            else:
                for mob, xv in zip(morph_v, MORPH_X):
                    yy = np.linspace(0, TAU, 44)
                    pts = [(1 - mv) * z_as_w_point(xv, a) + mv * w_point(xv, a)
                           for a in yy]
                    set_poly(mob, pts)
                    fade = 1.0 - ease((mv - 0.88) / 0.12)
                    mob.set_stroke(M_CYAN, width=2.2,
                                   opacity=(0.25 + 0.60 * math.sin(math.pi * mv)) * fade)
                for mob, yv in zip(morph_h, MORPH_Y):
                    xx = np.linspace(-1, 1, 30)
                    pts = [(1 - mh) * z_as_w_point(xa, yv) + mh * w_point(xa, yv)
                           for xa in xx]
                    set_poly(mob, pts)
                    fade = 1.0 - ease((mh - 0.88) / 0.12)
                    mob.set_stroke(M_RED, width=2.2,
                                   opacity=(0.25 + 0.60 * math.sin(math.pi * mh)) * fade)

            # 保角小圆：局部放大后仍是圆
            op = clamp01((m - 0.55) / 0.35)
            cst_z.move_to(z_point(0.32, 1.15))
            cst_z.set_width(0.34)
            cst_z.set_stroke(M_ORANGE, width=2.6, opacity=0.9 * op)
            cst_w.move_to(w_point(0.32, 1.15))
            cst_w.set_width(0.34 * rho * 0.9)
            cst_w.set_stroke(M_ORANGE, width=2.6, opacity=0.9 * op)

            # 动态数值
            num_x.set_value(x)
            num_rho.set_value(rho)
            num_y.set_value(y)
            num_det.set_value(rho * rho)
            f5_ul.set_stroke(M_CYAN, width=2.4, opacity=0.85 * op)

            # 循环重置瞬间的标题闪烁
            blink = 16.02 <= t <= 16.36
            title.set_opacity(0 if blink else 1)
            cursor.set_opacity(1 if blink else 0)

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
