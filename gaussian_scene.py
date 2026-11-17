#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Manim 复刻：架构师研究会 高斯积分科普短片
I = ∫e^(-x²)dx = √π  —— 升维 / 极坐标 / 雅可比行列式

环境说明：
    本机无 LaTeX，MathTex 不可用。公式用 matplotlib mathtext 预渲染为透明 PNG，
    由 Manim ImageMobject 承载；3D 曲面/坐标轴/曲线为 Manim 原生对象，
    通过手动 rotate 陈列斜视角，避免 ThreeDScene 相机对 2D 层的投影干扰。
    动画（弹出/旋转/数值递增）全部由 Manim 驱动。

运行：
    manim -pql gaussian_scene.py GaussianIntegral      # 低质量
    manim -pqh gaussian_scene.py GaussianIntegral      # 高质量
输出：
    media/videos/gaussian_scene/<quality>/GaussianIntegral.mp4
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

# ============================================================================
# 竖屏配置（匹配原视频 1080x1364）
# ============================================================================
config.pixel_width = 1080
config.pixel_height = 1364
config.frame_width = 9.0
config.frame_height = 9.0 * 1364 / 1080
config.frame_rate = 30

HERE = Path(__file__).resolve().parent
FORMULA_DIR = HERE / "formulas"
FORMULA_DIR.mkdir(exist_ok=True)

WHITE_C = (240, 241, 238)
CYAN_C = (75, 214, 200)
RED_C = (232, 96, 82)
ORANGE_C = (255, 168, 72)

CYAN_M = "#4BD6C8"
RED_M = "#E86052"
GRAY_M = "#697078"
DIM_M = "#969CA2"
ORANGE_M = "#FFA848"
PINK_M = "#E896F0"


# ============================================================================
# 公式 PNG 预渲染（matplotlib mathtext）
# ============================================================================
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


def formula(name: str, expr: str, size: int = 40, color=WHITE_C,
            height: float = 0.42) -> ImageMobject:
    path = FORMULA_DIR / f"{name}.png"
    if not path.exists():
        _math_img(expr, size, color).save(path)
    m = ImageMobject(str(path))
    if m.height > 0:
        m.scale(height / m.height)
    return m


class PixelNumber(ImageMobject):
    """纯内存数值显示：PIL 渲染到固定 RGBA 数组，set_value 原地更新 pixel_array。
    完全不写临时文件，规避 Windows 下 safe-delete 对 SVG 清理的拦截。"""
    _FONT = str(Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf")

    def __init__(self, value: float = 0.0, places: int = 2,
                 color=WHITE, px: int = 40, **kwargs):
        self.places = places
        self.value = value
        self._color = tuple(int(c) for c in ManimColor(color).to_rgb() * 255) \
            if not isinstance(color, tuple) else color
        self._font = ImageFont.truetype(self._FONT, px)
        # 固定画布：按小数位数给足宽度（整数位+符号位+小数点+小数位）
        nchars = 1 + 1 + places + (1 if places else 0)  # 符号 + 整数 + 小数
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
        x = (self._cw - tw) / 2 - box[0]
        y = (self._ch - th) / 2 - box[1]
        d.text((x, y), txt, font=self._font,
               fill=(self._color[0], self._color[1], self._color[2], 255))
        self.pixel_array[:] = np.asarray(img)


class ValueAnimation(Animation):
    """显式驱动一组 (对象, 更新函数) 的动画，绕开 updater 在 Group 内不触发的问题。"""
    def __init__(self, items, run_time=10, **kwargs):
        # items: list of (getter_from_t, setter)
        self.items = items
        super().__init__(Group(), run_time=run_time, **kwargs)

    def interpolate_mobject(self, alpha):
        for getter, setter in self.items:
            setter(getter(alpha))


class LiveNumber(Text):
    """用 Pango Text 实现的动态数值（不依赖 LaTeX 的 DecimalNumber）。"""
    def __init__(self, value: float = 0.0, places: int = 2,
                 color=WHITE, font_size: int = 30, **kwargs):
        self.value = value
        self.places = places
        sign = "-" if value < 0 else ""
        init = sign + f"{abs(value):.{places}f}" if places > 0 else f"{value:.0f}"
        super().__init__(init, color=color, font_size=font_size, **kwargs)

    def _fmt(self, value: float) -> str:
        if self.places <= 0:
            return f"{value:.0f}"
        sign = "-" if value < 0 else ""
        return sign + f"{abs(value):.{self.places}f}"

    def set_value(self, value: float):
        self.value = value
        self.set_text(self._fmt(value))


def make_brackets(h: float):
    """返回一对行列式方括号（左 '[' 与右 ']'）。"""
    lbr = VGroup(
        Line(UP * h, DOWN * h, color=WHITE, stroke_width=3),
        Line(UP * h, UP * h + RIGHT * 0.08, color=WHITE, stroke_width=3),
        Line(DOWN * h, DOWN * h + RIGHT * 0.08, color=WHITE, stroke_width=3),
    )
    rbr = VGroup(
        Line(UP * h, DOWN * h, color=WHITE, stroke_width=3),
        Line(UP * h, UP * h + LEFT * 0.08, color=WHITE, stroke_width=3),
        Line(DOWN * h, DOWN * h + LEFT * 0.08, color=WHITE, stroke_width=3),
    )
    return lbr, rbr


def matrix_group(elems) -> VGroup:
    """2x2 行列式括号矩阵。"""
    a, b, c, d = elems
    rows = Group(
        Group(a, b).arrange(RIGHT, buff=0.32),
        Group(c, d).arrange(RIGHT, buff=0.32),
    ).arrange(DOWN, buff=0.14)
    h = rows.height / 2 + 0.07
    lbr, rbr = make_brackets(h)
    rows.next_to(lbr, RIGHT, buff=0.10)
    rbr.next_to(rows, RIGHT, buff=0.10)
    return Group(lbr, rows, rbr)


# ============================================================================
# 场景
# ============================================================================
class GaussianIntegral(Scene):
    def construct(self):
        self.camera.background_color = "#131416"

        # 全局时间轴（驱动数值动画）
        T = ValueTracker(0.0)

        # ================================================================
        # 第一幕：顶部呼吸光标
        # ================================================================
        cursor = Line(UP * 0.16, DOWN * 0.16, color=WHITE, stroke_width=5)
        cursor.move_to([0, config.frame_height / 2 - 0.45, 0])
        self.play(FadeIn(cursor), run_time=0.4)
        self.wait(0.4)
        self.play(FadeOut(cursor), run_time=0.5)

        # ================================================================
        # 第二幕：3D 钟形曲面（线框）+ XYZ 三轴 + 背板墙 + 青红边缘曲线
        # ================================================================
        def gauss(u, v):
            return np.array([u, v, 1.45 * math.exp(-0.95 * (u * u + v * v))])

        LIM = 3.0
        geo = Group()

        # 曲面经纬线框
        for k in np.linspace(-LIM, LIM, 19):
            geo.add(ParametricFunction(
                lambda t, k=k: gauss(k, (2 * t - 1) * LIM),
                t_range=[0, 1, 0.02], color=GRAY_M, stroke_width=1.0,
                stroke_opacity=0.6))
            geo.add(ParametricFunction(
                lambda t, k=k: gauss((2 * t - 1) * LIM, k),
                t_range=[0, 1, 0.02], color=GRAY_M, stroke_width=1.0,
                stroke_opacity=0.45))

        # 地面网格
        for k in np.linspace(-LIM, LIM, 11):
            geo.add(Line([k, -LIM, 0], [k, LIM, 0],
                         color=GRAY_M, stroke_width=0.8, stroke_opacity=0.3))
            geo.add(Line([-LIM, k, 0], [LIM, k, 0],
                         color=GRAY_M, stroke_width=0.8, stroke_opacity=0.3))

        # 背板墙
        wall_h = 2.0
        for k in np.linspace(-LIM, LIM, 11):
            geo.add(Line([k, -LIM, 0], [k, -LIM, wall_h],
                         color=GRAY_M, stroke_width=0.8, stroke_opacity=0.22))
            geo.add(Line([LIM, k, 0], [LIM, k, wall_h],
                         color=GRAY_M, stroke_width=0.8, stroke_opacity=0.22))
        for z in np.linspace(0, wall_h, 6):
            geo.add(Line([-LIM, -LIM, z], [LIM, -LIM, z],
                         color=GRAY_M, stroke_width=0.8, stroke_opacity=0.22))
            geo.add(Line([LIM, -LIM, z], [LIM, LIM, z],
                         color=GRAY_M, stroke_width=0.8, stroke_opacity=0.22))

        # XYZ 三轴（Manim ThreeDAxes）
        axes = ThreeDAxes(
            x_range=[-3.6, 3.6, 1], y_range=[-3.6, 3.6, 1], z_range=[0, 2.4, 1],
            x_length=6.6, y_length=6.6, z_length=2.6,
            axis_config={"color": DIM_M, "stroke_width": 1.8,
                         "include_ticks": False},
        )
        geo.add(axes)

        # 青红边缘分布曲线
        cyan_curve = ParametricFunction(
            lambda t: gauss((2 * t - 1) * LIM, -LIM),
            t_range=[0, 1, 0.005], color=CYAN_M, stroke_width=5)
        red_curve = ParametricFunction(
            lambda t: gauss(LIM, (2 * t - 1) * LIM),
            t_range=[0, 1, 0.005], color=RED_M, stroke_width=5)
        geo.add(cyan_curve, red_curve)

        # 峰顶高光
        peak = Dot3D(gauss(0, 0), radius=0.07, color=PINK_M)
        peak_halo = Dot3D(gauss(0, 0), radius=0.16, color=PINK_M,
                          fill_opacity=0.25)
        geo.add(peak, peak_halo)

        # 统一旋转到等轴测俯视角：先在 xy 平面内转方位角，再绕 x 轴俯视
        geo.rotate(-45 * DEGREES, axis=OUT)
        geo.rotate(-58 * DEGREES, axis=RIGHT)
        geo.scale(0.80).shift(UP * 2.95)

        self.play(Create(axes), run_time=1.2)
        self.play(Create(VGroup(*[m for m in geo.submobjects if m not in
                                  [axes, cyan_curve, red_curve, peak, peak_halo]])),
                  run_time=2.0)
        self.play(Create(cyan_curve), Create(red_curve), run_time=1.4)
        self.play(FadeIn(peak), FadeIn(peak_halo), run_time=0.6)

        # 轴标签（2D，放到曲面端点附近）
        lbl_x = formula("ax_x", r"$x$", 44, (150, 156, 162), 0.26)
        lbl_y = formula("ax_y", r"$y$", 44, (150, 156, 162), 0.26)
        lbl_z = formula("ax_z", r"$z$", 44, (150, 156, 162), 0.26)
        lbl_x.move_to([3.55, 2.35, 0])
        lbl_y.move_to([-3.55, 2.55, 0])
        lbl_z.move_to([-1.55, 5.05, 0])
        lbl_ex = formula("ex", r"$e^{-x^2}$", 38, CYAN_C, 0.26)
        lbl_ey = formula("ey", r"$e^{-y^2}$", 38, RED_C, 0.26)
        lbl_ex.move_to([-1.85, 4.05, 0])
        lbl_ey.move_to([0.95, 3.15, 0])
        self.play(FadeIn(lbl_x), FadeIn(lbl_y), FadeIn(lbl_z),
                  FadeIn(lbl_ex), FadeIn(lbl_ey), run_time=0.7)

        # 曲面缓慢自转（用整体微旋转表现）
        self.play(
            geo.animate.rotate(0.12, axis=OUT),
            run_time=6, rate_func=linear,
        )

        # ================================================================
        # 第三幕：数值雅可比矩阵 + I² 估计值 + 数轴（Manim 原生动态数值）
        # ================================================================
        theta_trk = ValueTracker(0.25)
        r_trk = ValueTracker(0.35)

        def dec(initial, color=WHITE, fs=30):
            return LiveNumber(initial, places=2, color=color, font_size=fs)

        # 数值文本：PixelNumber 纯内存渲染，无临时文件 IO
        def dec(initial, places=2, color=WHITE, fs=30):
            return PixelNumber(initial, places=places, color=color, px=fs)

        n00, n01, n10, n11 = dec(0.97), dec(-0.09), dec(0.25), dec(0.34)

        lb, rb = make_brackets(0.40)
        inner = Group(Group(n00, n01).arrange(RIGHT, buff=0.12),
                      Group(n10, n11).arrange(RIGHT, buff=0.12)).arrange(DOWN, buff=0.10)
        inner.next_to(lb, RIGHT, buff=0.10)
        rb.next_to(inner, RIGHT, buff=0.10)
        jac = Group(lb, inner, rb)

        f_J = formula("fJ", r"$J=\mathrm{det}$", 38, WHITE_C, 0.34)
        f_eq = formula("feq", r"$=$", 38, WHITE_C, 0.34)
        det_val = dec(0.35, 2, RED, 32)
        row_num = Group(f_J, jac, f_eq, det_val).arrange(RIGHT, buff=0.16)
        row_num.scale(0.82).move_to([1.05, 1.45, 0])
        self.play(FadeIn(row_num, shift=UP * 0.15), run_time=0.7)

        f_I2 = formula("fI2", r"$I^2\approx$", 40, WHITE_C, 0.36)
        est = dec(0.0, 4, RED, 34)
        row_est = Group(f_I2, est).arrange(RIGHT, buff=0.12)
        row_est.move_to([0.55, 0.75, 0])
        self.play(FadeIn(row_est, shift=UP * 0.12), run_time=0.6)

        # 数轴
        axline = Line(LEFT * 2.4, RIGHT * 2.4, color=DIM_M, stroke_width=2)
        axline.move_to([1.0, 0.18, 0])
        pitick = Line(UP * 0.10, DOWN * 0.10, color=DIM_M, stroke_width=2)
        pitick.move_to(axline.get_end())
        lbl_pi = formula("pi", r"$\pi$", 34, (150, 156, 162), 0.26)
        lbl_pi.next_to(axline.get_end(), DOWN, buff=0.06)
        cdot = Dot(radius=0.07, color=RED)
        cdot.move_to(axline.get_start())
        self.play(Create(axline), FadeIn(pitick), FadeIn(lbl_pi),
                  FadeIn(cdot), run_time=0.6)

        # 显式动画驱动所有数值（interpolate_mobject 已验证可靠）
        def apply_values(theta, r, tval):
            theta_trk.set_value(theta); r_trk.set_value(r); T.set_value(tval)
            n00.set_value(math.cos(theta))
            n01.set_value(-r * math.sin(theta))
            n10.set_value(math.sin(theta))
            n11.set_value(r * math.cos(theta))
            det_val.set_value(r)
            est.set_value(math.pi * (1 - math.exp(-tval / 12)))
            cdot.move_to(axline.get_start() + RIGHT * axline.get_length() *
                         min(1.0, 1 - math.exp(-tval / 12)))

        class NumbersAnim(Animation):
            def __init__(self, apply, mob, run_time=10, **kw):
                self.apply = apply
                super().__init__(mob, run_time=run_time, **kw)
            def interpolate_mobject(self, alpha):
                self.apply(0.25 + alpha * 2 * math.pi,
                           0.35 + alpha * 0.95, alpha * 30)

        self.play(NumbersAnim(apply_values, cdot), rate_func=linear)

        # ================================================================
        # 第四幕：极坐标网 + 微元（左下）
        # ================================================================
        polar = VGroup()
        R = 1.15
        for i in range(1, 8):
            polar.add(Circle(radius=R * i / 7, color=GRAY_M,
                             stroke_width=1, stroke_opacity=0.5))
        for deg in range(0, 360, 30):
            ang = deg * DEGREES
            polar.add(Line(ORIGIN, R * np.array([math.cos(ang), math.sin(ang), 0]),
                           color=GRAY_M, stroke_width=1, stroke_opacity=0.45))
        polar.add(Line(LEFT * (R + 0.12), RIGHT * (R + 0.12),
                       color=GRAY_M, stroke_width=2, stroke_opacity=0.7))
        polar.add(Line(UP * (R + 0.12), DOWN * (R + 0.12),
                       color=GRAY_M, stroke_width=2, stroke_opacity=0.7))
        polar.move_to([-3.0, 0.55, 0])

        sector = AnnularSector(inner_radius=0.34, outer_radius=0.70,
                               angle=22 * DEGREES, start_angle=30 * DEGREES,
                               color=ORANGE_M, fill_opacity=0.35,
                               stroke_color=ORANGE_M, stroke_width=3)
        sector.move_to(polar.get_center())
        sector.add_updater(lambda m, dt: m.rotate(dt * 0.5,
                           about_point=polar.get_center()))
        self.play(Create(polar), FadeIn(sector), run_time=1.5)

        lbl_dr = formula("dr", r"$dr$", 30, ORANGE_C, 0.22)
        lbl_dth = formula("dth", r"$d\theta$", 30, ORANGE_C, 0.22)
        lbl_dr.next_to(polar.get_center() + 0.65 * RIGHT + 0.42 * DOWN, RIGHT, buff=0.04)
        lbl_dth.next_to(polar.get_center() + 0.75 * RIGHT + 0.22 * UP, RIGHT, buff=0.04)
        self.play(FadeIn(lbl_dr), FadeIn(lbl_dth), run_time=0.5)

        # ================================================================
        # 第五幕：四行推导公式（逐个弹出）
        # ================================================================
        f_double = formula("double",
                           r"$I^2=\int_{-\infty}^{\infty}\int_{-\infty}^{\infty}"
                           r"e^{-(x^2+y^2)}\,dx\,dy$", 42, WHITE_C, 0.42)
        f_double.move_to([0, -1.35, 0])

        e_xxr = formula("dxxr", r"$\frac{\partial x}{\partial r}$", 34, WHITE_C, 0.22)
        e_xxt = formula("dxxt", r"$\frac{\partial x}{\partial\theta}$", 34, WHITE_C, 0.22)
        e_xyr = formula("dxyr", r"$\frac{\partial y}{\partial r}$", 34, WHITE_C, 0.22)
        e_xyt = formula("dxyt", r"$\frac{\partial y}{\partial\theta}$", 34, WHITE_C, 0.22)
        e_cos = formula("cos", r"$\cos\theta$", 34, WHITE_C, 0.22)
        e_rsin = formula("rsin", r"$-r\sin\theta$", 34, WHITE_C, 0.22)
        e_sin = formula("sin", r"$\sin\theta$", 34, WHITE_C, 0.22)
        e_rcos = formula("rcos", r"$r\cos\theta$", 34, WHITE_C, 0.22)
        row_jac = Group(
            formula("J2", r"$J=\mathrm{det}$", 34, WHITE_C, 0.28),
            matrix_group([e_xxr, e_xxt, e_xyr, e_xyt]),
            formula("eqdet", r"$=\mathrm{det}$", 34, WHITE_C, 0.28),
            matrix_group([e_cos, e_rsin, e_sin, e_rcos]),
            formula("eqr", r"$=r$", 38, WHITE_C, 0.30),
        ).arrange(RIGHT, buff=0.14).scale(0.88).move_to([0, -2.55, 0])

        f_pi = formula("polarint",
                       r"$I^2=\int_0^{2\pi}\int_0^{\infty}e^{-r^2}r\,dr\,d\theta"
                       r"=\int_0^{2\pi}\frac{1}{2}\,d\theta=\pi$",
                       36, WHITE_C, 0.36)
        f_pi.move_to([0, -3.70, 0])

        f_final = formula("final",
                          r"$I=\int_{-\infty}^{\infty}e^{-x^2}dx=\sqrt{\pi}$",
                          44, WHITE_C, 0.44)
        f_final.move_to([0, -4.75, 0])

        for m in (f_double, row_jac, f_pi, f_final):
            self.play(FadeIn(m, shift=UP * 0.12, scale=0.85), run_time=0.8)
            self.wait(0.25)

        # ================================================================
        # 第六幕：结尾停留（数值继续逼近 π）+ 淡出
        # ================================================================
        class FinalAnim(Animation):
            def __init__(self, apply, mob, run_time=8, **kw):
                self.apply = apply
                super().__init__(mob, run_time=run_time, **kw)
            def interpolate_mobject(self, alpha):
                self.apply(0.25 + (2 + 2 * alpha) * math.pi,
                           1.30 + alpha * 0.05, 30 + alpha * 20)

        self.play(FinalAnim(apply_values, cdot), rate_func=linear)
        self.wait(1.5)
        self.play(*[FadeOut(m) for m in list(self.mobjects)], run_time=1.5)
