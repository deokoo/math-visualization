#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Manim 数学动画：傅里叶级数合成方波（高等数学）

视觉规范与高斯积分 / 复指数映射成片一致：
  - 公式全程固定，只动参数、圆链与波形
  - 无 LaTeX：公式用 matplotlib mathtext 预渲染为透明 PNG

运行：
    manim -pql fourier_scene.py FourierSeries
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
    path = FORMULA_DIR / f"cf_{name}.png"
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
    m.set_points_as_corners(points)
    return m


def clamp01(v):
    return float(np.clip(v, 0.0, 1.0))


def ease(v):
    v = clamp01(v)
    return v * v * (3 - 2 * v)


# ---------------------------------------------------------------------------
# 几何参数
# ---------------------------------------------------------------------------
C_CENTER = np.array([-2.35, 2.40, 0.0])   # 旋转圆链起点
CY = 2.40                                  # 波形基线
WX0 = -0.30                                # 波形屏幕起点 x
SX = 3.80 / TAU                            # 相位 -> 屏幕 x 缩放
A = 0.85                                   # 幅值显示缩放
N_TERMS = 5                                # 最多 5 个谐波


def s_partial(phase: float, prog: float) -> float:
    """带渐入权重的傅里叶部分和 s_N(phase)，收敛到幅值 ±1 的方波"""
    total = 0.0
    for n in range(N_TERMS):
        w = clamp01((prog - n) / 0.7)
        if w > 0:
            k = 2 * n + 1
            total += (4 / math.pi) * w * math.sin(k * phase) / k
    return total


# ---------------------------------------------------------------------------
# 场景
# ---------------------------------------------------------------------------
class FourierSeries(Scene):
    def construct(self):
        # 顶部标题
        title = text_img("AIMaker-Lab", 52, C_TEXT, 0.28).move_to([0, 4.35, 0])
        cursor = Line(UP * 0.16, DOWN * 0.16, color=M_TEXT,
                      stroke_width=2.5).move_to([0, 4.35, 0])
        cursor.set_opacity(0)

        # 参考水平线：目标方波幅值 ±1
        ref_top = Line([WX0, CY + A, 0], [WX0 + TAU * SX, CY + A, 0],
                       color=M_GRID_DIM, stroke_width=1.3, stroke_opacity=0.7)
        ref_bot = Line([WX0, CY - A, 0], [WX0 + TAU * SX, CY - A, 0],
                       color=M_GRID_DIM, stroke_width=1.3, stroke_opacity=0.7)
        axis = Line([WX0, CY, 0], [WX0 + TAU * SX, CY, 0],
                    color=M_GRID, stroke_width=1.6, stroke_opacity=0.9)
        lbl_top = mi(r"$+1$", "ref_top", C_GRID, 0.14, 26).move_to([WX0 - 0.20, CY + A, 0])
        lbl_bot = mi(r"$-1$", "ref_bot", C_GRID, 0.14, 26).move_to([WX0 - 0.20, CY - A, 0])
        lbl_t = mi(r"$x$", "axis_x", C_GRID, 0.14, 26).move_to([WX0 + TAU * SX + 0.14, CY, 0])

        # 旋转圆链
        circles = [Circle(radius=1, color=M_GRID, stroke_width=1.8,
                          stroke_opacity=0.0) for _ in range(N_TERMS)]
        radii = [VMobject(color=M_CYAN, stroke_width=2.6,
                          stroke_opacity=0.0) for _ in range(N_TERMS)]
        for i in range(N_TERMS):
            circles[i].set_stroke(M_CYAN if i == 0 else M_GRID,
                                  width=2.0 if i == 0 else 1.5, opacity=0.0)
        endpoint = Dot(radius=0.055, color=M_RED)
        endpoint_halo = Dot(radius=0.13, color=M_RED, fill_opacity=0.22)
        connector = VMobject(color=M_CYAN, stroke_width=2.4, stroke_opacity=0.9)

        # 部分和波形
        wave = VMobject(color=M_CYAN, stroke_width=3.2, stroke_opacity=1.0)
        wave_glow = VMobject(color=M_CYAN, stroke_width=6.0, stroke_opacity=0.12)

        # 吉布斯超调标注
        overshoot_dot = Dot(radius=0.05, color=M_ORANGE)
        overshoot_lbl = mi(r"$1.18$", "overshoot", C_ORANGE, 0.16, 26)

        # 动态数值
        lab_n = mi(r"$N=$", "lab_n", C_CYAN, 0.20, 28).move_to([-3.05, 0.78, 0])
        num_n = PixelNumber(1, 0, M_CYAN, 30).set(height=0.24).move_to([-2.55, 0.78, 0])
        lab_ph = mi(r"$\varphi=$", "lab_ph", C_RED, 0.20, 28).move_to([-0.35, 0.78, 0])
        num_ph = PixelNumber(0.0, 2, M_RED, 30).set(height=0.24).move_to([0.35, 0.78, 0])
        lab_ov = mi(r"$\mathrm{overshoot}=$", "lab_ov", C_ORANGE, 0.20, 28).move_to([1.55, 0.78, 0])
        num_ov = PixelNumber(1.18, 2, M_ORANGE, 30).set(height=0.24).move_to([3.15, 0.78, 0])

        # 底部固定公式链
        f1 = mi(r"$f(x)=\frac{4}{\pi}\sum_{n=0}^{\infty}"
                r"\frac{\sin((2n+1)x)}{2n+1}$",
                "f1", C_FORMULA, 0.62, 30).move_to([0, -0.38, 0])
        f2 = mi(r"$s_N(x)=\frac{4}{\pi}\sum_{n=0}^{N-1}"
                r"\frac{\sin((2n+1)x)}{2n+1}$",
                "f2", C_FORMULA, 0.62, 30).move_to([0, -1.32, 0])
        f3 = mi(r"$s_N(x)\ \longrightarrow\ \mathrm{square\ wave}\quad(N\to\infty)$",
                "f3", C_FORMULA, 0.31, 29).move_to([0, -2.24, 0])
        f4 = mi(r"$\mathrm{Gibbs}:\quad \max s_N\approx 1.18$",
                "f4", C_FORMULA, 0.31, 29).move_to([0, -3.16, 0])
        f5 = mi(r"$\lim_{N\to\infty}s_N(x)=1\quad(0<x<\pi)$",
                "f5", C_FORMULA, 0.32, 30).move_to([0, -4.08, 0])
        f5_ul = Line([-1.15, -4.36, 0], [1.15, -4.36, 0],
                     color=M_CYAN, stroke_width=2.4, stroke_opacity=0.0)
        formulas = Group(f1, f2, f3, f4, f5)

        # 图层
        self.add(ref_top, ref_bot, axis, lbl_top, lbl_bot, lbl_t, formulas)
        dynamic = Group(
            title, cursor,
            *circles, *radii, endpoint, endpoint_halo, connector,
            wave_glow, wave, overshoot_dot, overshoot_lbl,
            lab_n, num_n, lab_ph, num_ph, lab_ov, num_ov, f5_ul,
        )
        self.add(dynamic)

        # 逐帧驱动
        def apply(t):
            if t < 16.15:
                prog = 0.8 + 4.2 * ease((t - 1.5) / 10.5)
                phase = TAU * (t - 1.5) / 7.0
            else:
                u = clamp01((t - 16.15) / 5.0)
                prog = 0.8 + 4.2 * ease(u / 0.50)
                phase = TAU * u * 1.8

            # 圆链：半径带渐入权重，角度为 (2n+1)*phase
            center = C_CENTER.copy()
            for n in range(N_TERMS):
                w = clamp01((prog - n) / 0.7)
                k = 2 * n + 1
                r = (4 / math.pi) * w * A / k
                theta = k * phase
                direction = np.array([math.cos(theta), math.sin(theta), 0])
                end = center + r * direction
                circ = circles[n]
                if w > 0.001:
                    circ.move_to(center).set_width(max(0.01, 2 * r))
                    circ.set_stroke(M_CYAN if n == 0 else M_GRID,
                                    width=2.4 if n == 0 else 1.6,
                                    opacity=0.95 if n == 0 else 0.75 * w)
                    radii[n].set_points_as_corners([center, end])
                    radii[n].set_stroke(M_CYAN if n == 0 else M_GRID,
                                        width=2.6, opacity=0.95 * w)
                else:
                    circ.set_stroke(M_CYAN, width=1.5, opacity=0.0)
                    radii[n].set_stroke(M_CYAN, width=1.5, opacity=0.0)
                center = end

            # 波形：相位从当前 phase 开始向右推进一个周期
            pts = []
            for u in np.linspace(0, TAU, 200):
                val = s_partial(phase + u, prog)
                pts.append([WX0 + u * SX, CY + A * val, 0])
            wave.set_points_as_corners(pts)
            wave_glow.set_points_as_corners(pts)

            # 圆链末端 -> 波形起点
            connector.set_points_as_corners([center, [WX0, pts[0][1], 0]])
            endpoint.move_to(center)
            endpoint_halo.move_to(center)

            # 吉布斯超调标注（谐波足够多后出现）
            op = clamp01((prog - 3.0) / 1.0)
            peak_u = 0.16
            peak_val = s_partial(phase + peak_u, prog)
            peak_pts = [WX0 + u * SX, CY + A * s_partial(phase + u, prog), 0]
            overshoot_dot.move_to(peak_pts)
            overshoot_dot.set_opacity(op)
            overshoot_lbl.move_to(np.array(peak_pts) + [0.10, 0.20, 0])
            overshoot_lbl.set_opacity(op)
            f5_ul.set_stroke(M_CYAN, width=2.4, opacity=0.85 * op)

            # 数值
            num_n.set_value(sum(1 for n in range(N_TERMS)
                                if clamp01((prog - n) / 0.7) > 0.5))
            num_ph.set_value(phase % TAU)

            # 循环重置瞬间标题闪烁
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
