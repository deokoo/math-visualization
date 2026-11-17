#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
逐帧对齐版：架构师研究会《高斯积分》视频号原片复刻。

与原片的关键对齐（基于 original_frames/643 帧逐帧读取）：
  * 公式全程静止，从第一帧就存在，没有弹出动画；
  * 数学内容只占原录屏中部，比例为 576:720 = 4:5；
  * 动态元素只有：3D 钟形线框曲面旋转、极坐标扇形/白线、
    雅可比矩阵四个数值与 det、I² 数值与进度条；
  * 16.15s 左右发生一次重置（I² 归零、标题消失一瞬）。
  * I²(t) 直接由原片每帧红色进度条像素反解，非估算。

运行：
    manim -pql gaussian_scene_exact.py GaussianExact
输出：
    media/videos/gaussian_scene_exact/480p30/GaussianExact.mp4
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib

from manim import *

# 复用已验证的 mathtext PNG 与内存数值组件
import gaussian_scene as gs

# ============================================================================
# 画布：按原片数学内容区域 576x720 (4:5) 放大到 1080x1350
# ============================================================================
config.pixel_width = 1080
config.pixel_height = 1350
config.frame_width = 8.0
config.frame_height = 10.0
config.frame_rate = 30

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PARAM_CSV = ROOT / "frame_analysis" / "params.csv"
DURATION = 21.26
FPS_SRC = 30.34

BG = "#141414"
C_TITLE = (185, 185, 185)
C_TEXT = (205, 205, 205)
C_GRID = (92, 98, 102)
C_GRID_DIM = (72, 78, 82)
C_CYAN = (86, 200, 192)
C_RED = (201, 80, 74)
C_FORMULA = (138, 84, 80)
C_SECTOR = (190, 82, 76)
C_WHITE_LINE = (215, 215, 215)

# Manim 原生几何用字符串颜色，避免元组解析后的意外偏色
S_GRID = "#5C6266"
S_GRID_DIM = "#484E52"
S_CYAN = "#56C8C0"
S_RED = "#C9504A"
S_SECTOR = "#BE524C"
S_WHITE_LINE = "#D7D7D7"

# ============================================================================
# 逐帧参数
# ============================================================================
def load_params():
    rows = []
    with PARAM_CSV.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append({
                "t": float(row["t"]),
                "i2": float(row["i2"]),
                "theta_deg": float(row["theta"]) * 180.0 / math.pi,
                "r_max": float(row["r_max"]),
            })
    return rows


PARAMS = load_params()


def sync_points(dst, src):
    """只同步几何点，绕开 Group 没有 interpolate_color 的限制。"""
    for d, s in zip(dst.get_family(), src.get_family()):
        d.points = np.array(s.points)
    return dst


def param_at(t, key):
    x = max(0.0, min(float(len(PARAMS) - 1), t * FPS_SRC))
    i = int(x)
    j = min(i + 1, len(PARAMS) - 1)
    a = x - i
    return PARAMS[i][key] * (1 - a) + PARAMS[j][key] * a


# 从关键帧人工读数得到的真实 r / theta，用于精确匹配矩阵
_R_T = np.array([0.00, 3.26, 9.85, 16.15, 21.16])
_R_V = np.array([2.19, 0.36, 2.53, 0.35, 1.12])
_TH_V = np.radians([108.6, 167.4, 298.1, 345.5, 417.2])


def r_at(t):
    return float(np.interp(t, _R_T, _R_V))


def theta_at(t):
    return float(np.interp(t, _R_T, _TH_V))


# ============================================================================
# 文本组件
# ============================================================================
_SERIF = str(Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSerif.ttf")
_SANS = str(Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSerif.ttf")


def serif(text: str, px: int, color=C_TEXT, height=None) -> ImageMobject:
    """用 PIL 直接渲染文字，不依赖 LaTeX。"""
    img = Image.new("RGBA", (1400, 180), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    font = ImageFont.truetype(_SERIF, px)
    d.text((20, 20), text, font=font, fill=(*color, 255))
    bbox = img.getchannel("A").getbbox()
    img = img.crop(bbox) if bbox else img
    m = ImageMobject(np.asarray(img))
    if height is not None and m.height > 0:
        m.scale(height / m.height)
    return m


def math_img(expr: str, name: str, size=34, color=C_FORMULA, height=0.22) -> ImageMobject:
    m = gs.formula(name, expr, size, color, height)
    return m


class SignedNumber(gs.PixelNumber):
    """像原片一样给正数显示 '+' 号。"""

    def set_value(self, value: float):
        self.value = value
        txt = ("+" if value >= 0 else "-") + f"{abs(value):.{self.places}f}"
        img = Image.new("RGBA", (self._cw, self._ch), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        box = d.textbbox((0, 0), txt, font=self._font)
        tw, th = box[2] - box[0], box[3] - box[1]
        x = (self._cw - tw) / 2 - box[0]
        y = (self._ch - th) / 2 - box[1]
        d.text((x, y), txt, font=self._font,
               fill=(self._color[0], self._color[1], self._color[2], 255))
        self.pixel_array[:] = np.asarray(img)


class PlainNumber(gs.PixelNumber):
    """无符号数值（用于 I²，原片不显示 +）。"""

    def set_value(self, value: float):
        self.value = value
        txt = f"{abs(value):.{self.places}f}"
        img = Image.new("RGBA", (self._cw, self._ch), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        box = d.textbbox((0, 0), txt, font=self._font)
        tw, th = box[2] - box[0], box[3] - box[1]
        d.text(((self._cw - tw) / 2 - box[0], (self._ch - th) / 2 - box[1]),
               txt, font=self._font,
               fill=(self._color[0], self._color[1], self._color[2], 255))
        self.pixel_array[:] = np.asarray(img)


# ============================================================================
# 场景
# ============================================================================
class GaussianExact(Scene):
    def construct(self):
        self.camera.background_color = BG
        T = ValueTracker(0.0)
        self.add(T)

        # ------------------------------------------------------------------
        # 标题（原片 16.15s 附近短暂消失）
        # ------------------------------------------------------------------
        title = serif("matheorems", 26, C_TITLE, height=0.17)
        title.move_to([0, 3.06, 0])
        cursor = Line(UP * 0.09, DOWN * 0.09, color=C_TITLE, stroke_width=3)
        cursor.move_to([0, 3.06, 0]).set_opacity(0)
        self.add(title, cursor)

        # ------------------------------------------------------------------
        # 3D 钟形线框曲面 + 两侧背板墙 + 青红边缘曲线
        # ------------------------------------------------------------------
        def gauss(u, v):
            return np.array([u, v, 1.25 * math.exp(-0.95 * (u * u + v * v))])

        LIM = 3.0
        base = Group()
        # 曲面经纬线
        for k in np.linspace(-LIM, LIM, 17):
            base.add(ParametricFunction(
                lambda t, k=k: gauss(k, (2 * t - 1) * LIM),
                t_range=[0, 1, 0.025], color=S_GRID, stroke_width=1.0,
                stroke_opacity=0.72))
            base.add(ParametricFunction(
                lambda t, k=k: gauss((2 * t - 1) * LIM, k),
                t_range=[0, 1, 0.025], color=S_GRID, stroke_width=1.0,
                stroke_opacity=0.58))
        # 地面网格
        for k in np.linspace(-LIM, LIM, 9):
            base.add(Line([k, -LIM, 0], [k, LIM, 0],
                          color=S_GRID_DIM, stroke_width=0.8, stroke_opacity=0.55))
            base.add(Line([-LIM, k, 0], [LIM, k, 0],
                          color=S_GRID_DIM, stroke_width=0.8, stroke_opacity=0.55))
        # 背板墙
        wall_h = 1.65
        for k in np.linspace(-LIM, LIM, 9):
            base.add(Line([k, -LIM, 0], [k, -LIM, wall_h],
                          color=S_GRID_DIM, stroke_width=0.8, stroke_opacity=0.42))
            base.add(Line([LIM, k, 0], [LIM, k, wall_h],
                          color=S_GRID_DIM, stroke_width=0.8, stroke_opacity=0.42))
        for z in np.linspace(0, wall_h, 5):
            base.add(Line([-LIM, -LIM, z], [LIM, -LIM, z],
                          color=S_GRID_DIM, stroke_width=0.8, stroke_opacity=0.42))
            base.add(Line([LIM, -LIM, z], [LIM, LIM, z],
                          color=S_GRID_DIM, stroke_width=0.8, stroke_opacity=0.42))
        # 边缘分布曲线
        base.add(ParametricFunction(
            lambda t: gauss((2 * t - 1) * LIM, -LIM),
            t_range=[0, 1, 0.005], color=S_CYAN, stroke_width=3.4))
        base.add(ParametricFunction(
            lambda t: gauss(LIM, (2 * t - 1) * LIM),
            t_range=[0, 1, 0.005], color=C_RED, stroke_width=3.4))

        def plot_at(angle):
            p = base.copy()
            p.rotate(angle, axis=OUT)
            p.rotate(-58 * DEGREES, axis=RIGHT)
            p.scale_to_fit_height(1.52)
            p.scale(1.35)
            p.move_to([-0.35, 2.05, 0])
            return p

        plot = plot_at(0)
        self.add(plot)

        # 轴/曲线标注（固定位置，贴近原片）
        lbl_x = math_img(r"$x$", "ex_x", 34, C_GRID, 0.13).move_to([0.75, 1.35, 0])
        lbl_y = math_img(r"$y$", "ex_y", 34, C_GRID, 0.13).move_to([-1.55, 1.35, 0])
        lbl_ex = math_img(r"$e^{-x^2}$", "ex_ex", 30, C_CYAN, 0.13).move_to([-1.30, 2.45, 0])
        lbl_ey = math_img(r"$e^{-y^2}$", "ex_ey", 30, C_RED, 0.13).move_to([0.45, 2.40, 0])
        self.add(lbl_x, lbl_y, lbl_ex, lbl_ey)

        # ------------------------------------------------------------------
        # 左下：极坐标网 + 旋转扇形 + 白线
        # ------------------------------------------------------------------
        PC = np.array([-1.65, -0.30, 0.0])
        PR = 1.03
        polar = Group()
        for i in range(1, 8):
            polar.add(Circle(radius=PR * i / 7, color=S_GRID,
                             stroke_width=0.9, stroke_opacity=0.62).move_to(PC))
        for deg in range(0, 360, 30):
            ang = deg * DEGREES
            polar.add(Line(PC, PC + PR * np.array([math.cos(ang), math.sin(ang), 0]),
                           color=S_GRID, stroke_width=0.8, stroke_opacity=0.58))
        polar.add(Line(PC + LEFT * (PR + 0.10), PC + RIGHT * (PR + 0.10),
                       color=S_GRID, stroke_width=1.4, stroke_opacity=0.85))
        polar.add(Line(PC + UP * (PR + 0.10), PC + DOWN * (PR + 0.10),
                       color=S_GRID, stroke_width=1.4, stroke_opacity=0.85))
        self.add(polar)

        # 角度刻度文字
        lbl_p2 = math_img(r"$\frac{\pi}{2}$", "p2", 24, C_GRID, 0.11).move_to(PC + UP * (PR + 0.20))
        lbl_pi = math_img(r"$\pi$", "pi_ang", 24, C_GRID, 0.11).move_to(PC + RIGHT * (PR + 0.22))
        lbl_p3 = math_img(r"$\frac{3\pi}{2}$", "p3", 24, C_GRID, 0.11).move_to(PC + DOWN * (PR + 0.20))
        self.add(lbl_p2, lbl_pi, lbl_p3)

        # 中心红色光晕
        glow = Group(*[
            Circle(radius=r, color=C_RED, stroke_width=0,
                   fill_opacity=0.05).move_to(PC)
            for r in np.linspace(0.12, 0.62, 6)
        ])
        self.add(glow)

        sector = AnnularSector(
            inner_radius=0.32, outer_radius=0.56, angle=18 * DEGREES,
            start_angle=0, color=S_SECTOR, fill_opacity=0.38,
            stroke_color=S_SECTOR, stroke_width=1.8).move_to(PC)
        white_line = Line(PC, PC + PR * RIGHT, color=S_WHITE_LINE, stroke_width=2.2)
        self.add(sector, white_line)

        lbl_r = math_img(r"$r$", "lbl_r", 26, C_SECTOR, 0.11)
        lbl_dr = math_img(r"$dr$", "lbl_dr", 26, C_SECTOR, 0.11)
        lbl_rdth = math_img(r"$r\,d\theta$", "lbl_rdth", 26, C_SECTOR, 0.11)
        self.add(lbl_r, lbl_dr, lbl_rdth)

        # ------------------------------------------------------------------
        # 雅可比矩阵数值行
        # ------------------------------------------------------------------
        def sdec(v, px=26):
            return SignedNumber(v, places=2, color=C_TEXT, px=px)

        n00, n01, n10, n11 = sdec(0), sdec(0), sdec(0), sdec(0)
        lb, rb = gs.make_brackets(0.24)
        inner = Group(
            Group(n00, n01).arrange(RIGHT, buff=0.07),
            Group(n10, n11).arrange(RIGHT, buff=0.07),
        ).arrange(DOWN, buff=0.06)
        inner.next_to(lb, RIGHT, buff=0.06)
        rb.next_to(inner, RIGHT, buff=0.06)
        jac = Group(lb, inner, rb)

        f_J = math_img(r"$J=\mathrm{det}$", "ex_fJ", 34, C_TEXT, 0.22)
        f_eq = math_img(r"$=$", "ex_feq", 34, C_TEXT, 0.22)
        det_val = sdec(0, 26)
        det_val._color = (*C_RED, 255)
        det_val.set_value(0)
        row_jac = Group(f_J, jac, f_eq, det_val).arrange(RIGHT, buff=0.09)
        row_jac.move_to([-0.35, 0.20, 0])
        self.add(row_jac)

        # ------------------------------------------------------------------
        # I² 数值 + 进度条
        # ------------------------------------------------------------------
        f_i2 = math_img(r"$I^2\approx$", "ex_i2", 34, C_TEXT, 0.22)
        est = PlainNumber(0.0, places=4, color=C_RED, px=25)
        row_i2 = Group(f_i2, est).arrange(RIGHT, buff=0.07)
        row_i2.move_to([-0.75, -0.31, 0])
        self.add(row_i2)

        bar_x0, bar_x1, bar_y = -0.62, 1.91, -0.47
        axis_line = Line([bar_x0, bar_y, 0], [bar_x1, bar_y, 0],
                         color=S_GRID, stroke_width=1.4)
        fill_line = Line([bar_x0, bar_y, 0], [bar_x0 + 0.001, bar_y, 0],
                         color=C_RED, stroke_width=3.0)
        pi_tick = Line([bar_x1, bar_y + 0.07, 0], [bar_x1, bar_y - 0.07, 0],
                       color=S_GRID, stroke_width=1.4)
        lbl_pi_end = math_img(r"$\pi$", "ex_pi_end", 26, C_GRID, 0.12).move_to(
            [bar_x1 + 0.08, bar_y - 0.10, 0])
        self.add(axis_line, fill_line, pi_tick, lbl_pi_end)

        # ------------------------------------------------------------------
        # 四行静止公式（原片第一帧即存在，无弹出）
        # ------------------------------------------------------------------
        f1 = math_img(
            r"$I^2=\int_{-\infty}^{\infty}\int_{-\infty}^{\infty}"
            r"e^{-(x^2+y^2)}\,dx\,dy$",
            "ex_f1", 34, C_FORMULA, 0.24).move_to([-0.75, -0.76, 0])

        def frac(a, b):
            return Group(
                math_img(a, "ex_" + a.replace("\\", "").replace("{", "").replace("}", "")[:12],
                         22, C_FORMULA, 0.12),
                math_img(b, "ex_b_" + b.replace("\\", "").replace("{", "").replace("}", "")[:12],
                         22, C_FORMULA, 0.12),
            ).arrange(DOWN, buff=0.01)

        m1 = Group(
            Group(frac(r"$\partial x$", r"$\partial\theta$"),
                  frac(r"$\partial x$", r"$\partial r$")).arrange(RIGHT, buff=0.10),
            Group(frac(r"$\partial y$", r"$\partial\theta$"),
                  frac(r"$\partial y$", r"$\partial r$")).arrange(RIGHT, buff=0.10),
        ).arrange(DOWN, buff=0.07)
        m2 = Group(
            Group(math_img(r"$\cos\theta$", "ex_cos", 24, C_FORMULA, 0.12),
                  math_img(r"$-r\sin\theta$", "ex_rsin", 24, C_FORMULA, 0.12)).arrange(RIGHT, buff=0.10),
            Group(math_img(r"$\sin\theta$", "ex_sin", 24, C_FORMULA, 0.12),
                  math_img(r"$r\cos\theta$", "ex_rcos", 24, C_FORMULA, 0.12)).arrange(RIGHT, buff=0.10),
        ).arrange(DOWN, buff=0.07)

        def bracketed(m):
            lb_, rb_ = gs.make_brackets(m.height / 2 + 0.03)
            m.next_to(lb_, RIGHT, buff=0.05)
            rb_.next_to(m, RIGHT, buff=0.05)
            return Group(lb_, m, rb_)

        f2 = Group(
            math_img(r"$J=\mathrm{det}$", "ex_f2a", 24, C_FORMULA, 0.17),
            bracketed(m1),
            math_img(r"$=\mathrm{det}$", "ex_f2b", 24, C_FORMULA, 0.17),
            bracketed(m2),
            math_img(r"$=r$", "ex_f2c", 24, C_FORMULA, 0.17),
        ).arrange(RIGHT, buff=0.06).move_to([-0.75, -1.36, 0])

        f3 = math_img(
            r"$I^2=\int_0^{2\pi}\int_0^{\infty}e^{-r^2}r\,dr\,d\theta"
            r"=\int_0^{2\pi}\frac{1}{2}\,d\theta=\pi$",
            "ex_f3", 28, C_FORMULA, 0.20).move_to([-0.75, -1.92, 0])
        f4 = math_img(
            r"$I=\int_{-\infty}^{\infty}e^{-x^2}dx=\sqrt{\pi}$",
            "ex_f4", 30, C_FORMULA, 0.22).move_to([-0.75, -2.40, 0])
        self.add(f1, f2, f3, f4)

        # 原片数学内容在 576x1280 录屏里占 720px 高；按内容区 4:5 放大到本画布
        root = Group(*[m for m in self.mobjects if m is not T])
        self.clear()
        self.add(root.scale(1.45), T)

        # ------------------------------------------------------------------
        # 逐帧时间轴驱动
        # ------------------------------------------------------------------
        def apply(t):
            theta = theta_at(t)
            rr = r_at(t)
            i2 = param_at(t, "i2")

            # 3D 曲面随 theta 缓慢转动
            new_plot = plot_at(theta * 0.35)
            sync_points(plot, new_plot)

            # 极坐标白线 / 扇形
            a = theta % TAU
            white_line.set_points_by_ends(PC, PC + PR * np.array([math.cos(a), math.sin(a), 0]))
            rpx = param_at(t, "r_max")
            inner = float(np.clip(rpx / 135.0 * 0.80, 0.10, 0.82))
            outer = inner + 0.20
            new_sector = AnnularSector(
                inner_radius=inner, outer_radius=outer, angle=18 * DEGREES,
                start_angle=a, color=S_SECTOR, fill_opacity=0.38,
                stroke_color=S_SECTOR, stroke_width=1.8).move_to(PC)
            sync_points(sector, new_sector)

            mid = (inner + outer) / 2
            lbl_r.move_to(PC + mid * np.array([math.cos(a), math.sin(a), 0]))
            lbl_dr.move_to(PC + (outer + 0.10) * np.array([math.cos(a), math.sin(a), 0]))
            off = a + 13 * DEGREES
            lbl_rdth.move_to(PC + (outer + 0.04) * np.array([math.cos(off), math.sin(off), 0]))

            # 矩阵数值
            n00.set_value(math.cos(theta))
            n01.set_value(-rr * math.sin(theta))
            n10.set_value(math.sin(theta))
            n11.set_value(rr * math.cos(theta))
            det_val.set_value(rr)

            # I² 数值与进度条
            est.set_value(i2)
            frac_fill = float(np.clip(i2 / math.pi, 0.0, 1.0))
            fill_line.set_points_by_ends(
                [bar_x0, bar_y, 0], [bar_x0 + (bar_x1 - bar_x0) * frac_fill, bar_y, 0])

            # 标题重置瞬间的闪烁
            title.set_opacity(0 if 16.05 <= t <= 16.40 else 1)
            cursor.set_opacity(1 if 16.05 <= t <= 16.40 else 0)

        # 只有被动画持有的对象才会逐帧重绘，否则会被 Manim 当作静态层烘焙
        dynamic = Group(
            plot, title, cursor, sector, white_line,
            lbl_r, lbl_dr, lbl_rdth,
            n00, n01, n10, n11, det_val, est, fill_line,
        )

        class Timeline(Animation):
            def __init__(self, run_time=DURATION, **kw):
                super().__init__(dynamic, run_time=run_time, **kw)

            def interpolate_mobject(self, alpha):
                apply(alpha * DURATION)

        self.play(Timeline(), rate_func=linear)

        # 末尾保持最后一帧
        apply(DURATION - 1 / config.frame_rate)
        self.wait(0.1)
