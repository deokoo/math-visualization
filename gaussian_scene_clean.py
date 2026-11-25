from pathlib import Path
import csv
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib

from manim import *

import gaussian_scene as gs

ROOT = Path(__file__).resolve().parent
PARAMS_CSV = ROOT / "params.csv"

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
C_SECTOR = (214, 82, 76)
C_ORANGE = (238, 148, 72)
C_BG = (10, 11, 12)

# VMobject 需要 hex 颜色；tuple(0-255) 会被 Manim 当成 0-1 浮点，导致全变蓝
M_TEXT = "#ECE8DC"
M_FORMULA = "#ECE8DC"
M_GRID = "#767C80"
M_GRID_DIM = "#5C6165"
M_CYAN = "#4AD6C8"
M_RED = "#E86052"
M_SECTOR = "#D6524C"
M_ORANGE = "#EE9448"

_SERIF = str(Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSerif.ttf")


# ---------------------------------------------------------------------------
# 原片逐帧参数
# ---------------------------------------------------------------------------
with PARAMS_CSV.open(newline="", encoding="utf-8") as fh:
    PARAMS = [
        {
            "frame": int(row["frame"]),
            "t": float(row["t"]),
            "i2": float(row["i2"]),
        }
        for row in csv.DictReader(fh)
    ]

TS = np.array([row["t"] for row in PARAMS])
I2S = np.array([row["i2"] for row in PARAMS])


def param_at(t, key):
    # 按原片 CSV 里的真实时间戳插值，避免 16.15s 重置点对不上
    values = I2S if key == "i2" else np.array([row[key] for row in PARAMS])
    return float(np.interp(t, TS, values))


_R_T = np.array([0.00, 3.26, 9.85, 16.15, 21.16])
_R_V = np.array([2.19, 0.36, 2.53, 0.35, 1.12])
_TH_V = np.radians([108.6, 167.4, 298.1, 345.5, 417.2])


def r_at(t):
    return float(np.interp(t, _R_T, _R_V))


def theta_at(t):
    # 用原片关键帧反解的 θ：矩阵四值与白线角度逐帧贴近原片；I² 进度仍按 I²=θ/2 的累积量显示
    return float(np.interp(t, _R_T, _TH_V) % TAU)


# ---------------------------------------------------------------------------
# 渲染小工具：无 LaTeX 公式、纯内存动态数字、折线
# ---------------------------------------------------------------------------
def mi(expr, name, color=C_FORMULA, height=0.25, size=32):
    return gs.formula("clean_" + name, expr, size, color, height)


def text_img(text, px, color=C_TEXT, height=0.3):
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


class SignedNumber(gs.PixelNumber):
    def set_value(self, value):
        self.value = value
        txt = ("+" if value >= 0 else "-") + f"{abs(value):.{self.places}f}"
        img = Image.new("RGBA", (self._cw, self._ch), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        box = d.textbbox((0, 0), txt, font=self._font)
        tw, th = box[2] - box[0], box[3] - box[1]
        x = (self._cw - tw) / 2 - box[0]
        y = (self._ch - th) / 2 - box[1]
        d.text((x, y), txt, font=self._font, fill=(*self._color, 255))
        self.pixel_array[:] = np.asarray(img)


class PlainNumber(gs.PixelNumber):
    def set_value(self, value):
        self.value = value
        txt = f"{abs(value):.{self.places}f}"
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


# ---------------------------------------------------------------------------
# 3D 线框曲面：手动正交投影，避免 Manim 3D 相机把曲面压扁
# ---------------------------------------------------------------------------
LIM = 1.6
WALL_H = 1.28
PHI0 = math.radians(35)

PROJ_CENTER = np.array([0.0, 1.92, 0.0])
SX = 0.86
SY = 0.86
ZS = 0.95
TILT = 0.48


def surf_z(u, v):
    return 1.18 * math.exp(-1.80 * (u * u + v * v))


def proj(u, v, z, phi):
    x3 = u * math.cos(phi) - v * math.sin(phi)
    y3 = u * math.sin(phi) + v * math.cos(phi)
    return PROJ_CENTER + np.array([SX * x3, SY * (TILT * y3 + ZS * z), 0.0])


def samples(n=22, a=-LIM, b=LIM):
    return np.linspace(a, b, n)


class GaussianClean(Scene):
    def construct(self):
        # ------------------------------------------------------------------
        # 静态标题
        # ------------------------------------------------------------------
        title = text_img("AIMaker-Lab", 52, C_TEXT, 0.28).move_to([0, 4.35, 0])
        cursor = Line(UP * 0.16, DOWN * 0.16, color=M_TEXT, stroke_width=2.5).move_to([0, 4.35, 0])
        cursor.set_opacity(0)

        # ------------------------------------------------------------------
        # 3D 线框曲面（动态折线，先建模板）
        # ------------------------------------------------------------------
        phi0 = PHI0
        us = samples(18)
        vs = samples(18)

        floor_u = [poly([proj(u, v, 0, phi0) for v in vs], M_GRID_DIM, 1.3, 0.95) for u in us[::2]]
        floor_v = [poly([proj(u, v, 0, phi0) for u in us], M_GRID_DIM, 1.3, 0.95) for v in vs[::2]]

        wall_a_v = [poly([proj(u, LIM, z, phi0) for z in np.linspace(0, WALL_H, 12)],
                         M_GRID_DIM, 1.2, 0.85) for u in us[::2]]
        wall_a_h = [poly([proj(u, LIM, z, phi0) for u in us],
                         M_GRID_DIM, 1.2, 0.85) for z in np.linspace(0, WALL_H, 7)]
        wall_b_v = [poly([proj(LIM, v, z, phi0) for z in np.linspace(0, WALL_H, 12)],
                         M_GRID_DIM, 1.2, 0.85) for v in vs[::2]]
        wall_b_h = [poly([proj(LIM, v, z, phi0) for v in vs],
                         M_GRID_DIM, 1.2, 0.85) for z in np.linspace(0, WALL_H, 7)]

        surf_u = [poly([proj(u, v, surf_z(u, v), phi0) for v in vs], M_GRID, 1.55, 1.0) for u in us]
        surf_v = [poly([proj(u, v, surf_z(u, v), phi0) for u in us], M_GRID, 1.55, 1.0) for v in vs]

        curve_x = [proj(u, LIM, 1.05 * math.exp(-1.60 * u * u), phi0) for u in samples(90)]
        curve_y = [proj(LIM, v, 1.05 * math.exp(-1.60 * v * v), phi0) for v in samples(90)]
        edge_cyan = poly(curve_x, M_CYAN, 4.0)
        edge_red = poly(curve_y, M_RED, 4.0)

        # 原片里曲面不旋转，而是随积分角度扫过一圈彩色网格
        SWEEP_ANG = 22
        SWEEP_RAD = 7
        sweep_segs = [
            VMobject(color=M_CYAN, stroke_width=1.7, stroke_opacity=0.0)
            for _ in range(SWEEP_ANG * (SWEEP_RAD + 1))
        ]
        sweep_group = Group(*sweep_segs)

        patch_pts = [
            proj(-0.82 + 0.20 * math.cos(a), -0.30 + 0.13 * math.sin(a), 0.006, phi0)
            for a in np.linspace(0, TAU, 30)
        ]
        patch = poly(patch_pts + [patch_pts[0]], M_SECTOR, 1.0, 0.9, 0.48)

        lbl_x = mi(r"$x$", "px", C_GRID, 0.15, 28).move_to(proj(LIM, -LIM, 0, phi0) + [0.13, -0.08, 0])
        lbl_y = mi(r"$y$", "py", C_GRID, 0.15, 28).move_to(proj(-LIM, LIM, 0, phi0) + [-0.12, -0.08, 0])
        lbl_ex = mi(r"$e^{-x^2}$", "pex", C_CYAN, 0.15, 28).move_to(proj(-1.45, LIM, 1.30, phi0) + [-0.10, 0.10, 0])
        lbl_ey = mi(r"$e^{-y^2}$", "pey", C_RED, 0.15, 28).move_to(proj(LIM, 1.20, 1.30, phi0) + [0.10, 0.10, 0])

        plot_group = Group(
            *floor_u, *floor_v,
            *wall_a_v, *wall_a_h, *wall_b_v, *wall_b_h,
            *surf_u, *surf_v,
            edge_cyan, edge_red, patch,
            lbl_x, lbl_y, lbl_ex, lbl_ey,
        )

        # ------------------------------------------------------------------
        # 左下极坐标网（静态网格 + 动态白线/红扇区）
        # ------------------------------------------------------------------
        PC = np.array([-2.75, -0.42, 0.0])
        PR = 0.98
        polar_static = VGroup()
        for k in range(1, 8):
            polar_static.add(Circle(radius=PR * k / 7, color=M_GRID_DIM,
                                    stroke_width=1.3, stroke_opacity=0.95).move_to(PC))
        for k in range(12):
            a = TAU * k / 12
            d = np.array([math.cos(a), math.sin(a), 0])
            polar_static.add(Line(PC, PC + PR * d, color=M_GRID_DIM,
                                  stroke_width=1.3, stroke_opacity=0.95))
        polar_static.add(Line(PC + LEFT * PR, PC + RIGHT * PR, color=M_GRID, stroke_width=1.7, stroke_opacity=1.0))
        polar_static.add(Line(PC + DOWN * PR, PC + UP * PR, color=M_GRID, stroke_width=1.7, stroke_opacity=1.0))

        lbl_pr = mi(r"$r$", "pr", C_GRID, 0.12, 24).move_to(PC + UP * (PR + 0.12))
        lbl_pt = mi(r"$\theta$", "ptheta", C_GRID, 0.12, 24).move_to(PC + RIGHT * (PR + 0.14))

        white_line = Line(PC, PC + RIGHT * PR, color=M_TEXT, stroke_width=2.8)
        sector = poly([PC], M_SECTOR, 1.2, 0.95, 0.42)
        lbl_r = mi(r"$r$", "dynr", C_SECTOR, 0.12, 24)
        lbl_dr = mi(r"$dr$", "dyndr", C_SECTOR, 0.12, 24)
        lbl_rdth = mi(r"$r\,d\theta$", "dynrdth", C_SECTOR, 0.12, 24)

        polar_dyn = Group(white_line, sector, lbl_r, lbl_dr, lbl_rdth)

        # ------------------------------------------------------------------
        # 右上：雅可比数值矩阵 + I² 进度条
        # ------------------------------------------------------------------
        f_J = mi(r"$J=\mathrm{det}$", "mjdet", C_TEXT, 0.24, 30).move_to([-1.02, 0.22, 0])
        b1 = bracket(-0.34, 0.42, -0.02, "left", M_TEXT)
        b2 = bracket(0.72, 0.42, -0.02, "right", M_TEXT)

        n00 = SignedNumber(0, 2, C_TEXT, 28).set(height=0.19).move_to([-0.02, 0.31, 0])
        n01 = SignedNumber(0, 2, C_TEXT, 28).set(height=0.19).move_to([0.48, 0.31, 0])
        n10 = SignedNumber(0, 2, C_TEXT, 28).set(height=0.19).move_to([-0.02, 0.09, 0])
        n11 = SignedNumber(0, 2, C_TEXT, 28).set(height=0.19).move_to([0.48, 0.09, 0])
        # 用两条细线画等号，避免 mathtext/PIL 把 "=" 糊成白块
        eq_top = Line([0.94, 0.27, 0], [1.08, 0.27, 0], color=M_TEXT, stroke_width=2.8)
        eq_bot = Line([0.94, 0.17, 0], [1.08, 0.17, 0], color=M_TEXT, stroke_width=2.8)
        det_val = PlainNumber(0, 2, C_RED, 30).set(height=0.25).move_to([1.40, 0.22, 0])

        matrix_dyn = Group(n00, n01, n10, n11, det_val)

        f_i2 = mi(r"$I^2\approx$", "mi2", C_TEXT, 0.25, 30).move_to([-0.68, -0.56, 0])
        est = PlainNumber(0, 4, C_RED, 30).set(height=0.25).move_to([0.02, -0.56, 0])

        bar_x0, bar_x1, bar_y = -1.22, 3.48, -0.92
        bar_base = Line([bar_x0, bar_y, 0], [bar_x1, bar_y, 0], color=M_GRID, stroke_width=1.8)
        bar_fill = Line([bar_x0, bar_y, 0], [bar_x0, bar_y, 0], color=M_RED, stroke_width=4.2)
        lbl_pi_end = mi(r"$\pi$", "mpiend", C_GRID, 0.14, 26).move_to([bar_x1, bar_y - 0.15, 0])

        # ------------------------------------------------------------------
        # 底部固定四行公式
        # ------------------------------------------------------------------
        f1 = mi(
            r"$I^2=\int_{-\infty}^{\infty}\int_{-\infty}^{\infty}"
            r"e^{-(x^2+y^2)}\,dx\,dy$",
            "f1", C_FORMULA, 0.33, 30).move_to([0.10, -2.08, 0])

        f2a = mi(r"$J=\mathrm{det}$", "f2a", C_FORMULA, 0.26, 28).move_to([-2.55, -2.99, 0])
        f2b1 = bracket(-1.88, -2.74, -3.24, "left", M_FORMULA)
        f2b2 = bracket(-0.16, -2.74, -3.24, "right", M_FORMULA)

        f2_00 = mi(r"$\frac{\partial x}{\partial r}$", "f200", C_FORMULA, 0.19, 24).move_to([-1.48, -2.85, 0])
        f2_01 = mi(r"$\frac{\partial x}{\partial \theta}$", "f201", C_FORMULA, 0.19, 24).move_to([-0.76, -2.85, 0])
        f2_10 = mi(r"$\frac{\partial y}{\partial r}$", "f210", C_FORMULA, 0.19, 24).move_to([-1.48, -3.13, 0])
        f2_11 = mi(r"$\frac{\partial y}{\partial \theta}$", "f211", C_FORMULA, 0.19, 24).move_to([-0.76, -3.13, 0])

        f2c = mi(r"$=\mathrm{det}$", "f2c", C_FORMULA, 0.26, 28).move_to([0.48, -2.99, 0])
        f2b3 = bracket(1.22, -2.74, -3.24, "left", M_FORMULA)
        f2b4 = bracket(2.70, -2.74, -3.24, "right", M_FORMULA)
        f2_20 = mi(r"$\cos\theta$", "f220", C_FORMULA, 0.19, 24).move_to([1.60, -2.85, 0])
        f2_21 = mi(r"$-r\sin\theta$", "f221", C_FORMULA, 0.19, 24).move_to([2.30, -2.85, 0])
        f2_30 = mi(r"$\sin\theta$", "f230", C_FORMULA, 0.19, 24).move_to([1.60, -3.13, 0])
        f2_31 = mi(r"$r\cos\theta$", "f231", C_FORMULA, 0.19, 24).move_to([2.30, -3.13, 0])
        f2d = mi(r"$=r$", "f2d", C_FORMULA, 0.26, 28).move_to([3.15, -2.99, 0])

        f3 = mi(
            r"$I^2=\int_0^{2\pi}\int_0^{\infty}e^{-r^2}r\,dr\,d\theta"
            r"=\int_0^{2\pi}\frac{1}{2}\,d\theta=\pi$",
            "f3", C_FORMULA, 0.30, 28).move_to([0.08, -3.78, 0])

        f4 = mi(
            r"$I=\int_{-\infty}^{\infty}e^{-x^2}\,dx=\sqrt{\pi}$",
            "f4", C_FORMULA, 0.31, 30).move_to([0.08, -4.44, 0])

        # ------------------------------------------------------------------
        # 图层：静态先加，动态对象统一放进 Timeline 持有的 Group
        # ------------------------------------------------------------------
        self.add(
            title, lbl_pr, lbl_pt, f_J, b1, b2, eq_top, eq_bot, f_i2,
            bar_base, lbl_pi_end,
            f1, f2a, f2b1, f2b2, f2b3, f2b4,
            f2_00, f2_01, f2_10, f2_11,
            f2c, f2_20, f2_21, f2_30, f2_31, f2d, f3, f4,
            polar_static,
        )

        dynamic = Group(
            title, cursor,
            *plot_group,
            sweep_group,
            *polar_dyn,
            *matrix_dyn,
            est, bar_fill,
        )
        self.add(dynamic)

        # ------------------------------------------------------------------
        # 逐帧驱动
        # ------------------------------------------------------------------
        def set_poly(mob, points, closed=False):
            pts = points + [points[0]] if closed else points
            mob.set_points_as_corners(pts)

        def apply(t):
            theta = theta_at(t)
            rr = r_at(t)
            i2 = param_at(t, "i2")
            if 16.05 <= t <= 16.40:
                i2 = 0.0
            phi = PHI0

            # 3D 线框
            for mob, u in zip(floor_u, us[::2]):
                set_poly(mob, [proj(u, v, 0, phi) for v in vs])
            for mob, v in zip(floor_v, vs[::2]):
                set_poly(mob, [proj(u, v, 0, phi) for u in us])

            for mob, u in zip(wall_a_v, us[::2]):
                set_poly(mob, [proj(u, LIM, z, phi) for z in np.linspace(0, WALL_H, 12)])
            for mob, z in zip(wall_a_h, np.linspace(0, WALL_H, 7)):
                set_poly(mob, [proj(u, LIM, z, phi) for u in us])
            for mob, v in zip(wall_b_v, vs[::2]):
                set_poly(mob, [proj(LIM, v, z, phi) for z in np.linspace(0, WALL_H, 12)])
            for mob, z in zip(wall_b_h, np.linspace(0, WALL_H, 7)):
                set_poly(mob, [proj(LIM, v, z, phi) for v in vs])

            for mob, u in zip(surf_u, us):
                set_poly(mob, [proj(u, v, surf_z(u, v), phi) for v in vs])
            for mob, v in zip(surf_v, vs):
                set_poly(mob, [proj(u, v, surf_z(u, v), phi) for u in us])

            set_poly(edge_cyan, [proj(u, LIM, 1.05 * math.exp(-1.60 * u * u), phi) for u in samples(90)])
            set_poly(edge_red, [proj(LIM, v, 1.05 * math.exp(-1.60 * v * v), phi) for v in samples(90)])
            set_poly(
                patch,
                [proj(-0.82 + 0.20 * math.cos(a), -0.30 + 0.13 * math.sin(a), 0.006, phi)
                 for a in np.linspace(0, TAU, 30)],
                closed=True,
            )

            lbl_x.move_to(proj(LIM, -LIM, 0, phi) + [0.13, -0.08, 0])
            lbl_y.move_to(proj(-LIM, LIM, 0, phi) + [-0.12, -0.08, 0])
            lbl_ex.move_to(proj(-1.45, LIM, 1.30, phi) + [-0.10, 0.10, 0])
            lbl_ey.move_to(proj(LIM, 1.20, 1.30, phi) + [0.10, 0.10, 0])

            # 曲面彩色扫描：随积分进度扫过一圈，重置后回到灰色线框
            sweep = TAU * min(t / 12.0, 1.0) if t < 15.2 else 0.0
            k = 0
            if sweep > 0.02:
                for j in range(SWEEP_ANG):
                    a0 = sweep * j / SWEEP_ANG
                    a1 = sweep * (j + 1) / SWEEP_ANG
                    am = (a0 + a1) / 2
                    col = M_CYAN if math.sin(am) >= 0 else M_ORANGE

                    rad_pts = []
                    for rr0 in np.linspace(0.05, LIM, 10):
                        uu, vv = rr0 * math.cos(a0), rr0 * math.sin(a0)
                        rad_pts.append(proj(uu, vv, surf_z(uu, vv), phi))
                    set_poly(sweep_segs[k], rad_pts)
                    sweep_segs[k].set_stroke(col, width=1.75, opacity=1.0)
                    k += 1

                    for rr0 in np.linspace(0.20, LIM, SWEEP_RAD):
                        arc_pts = []
                        for aa in np.linspace(a0, a1, 4):
                            uu, vv = rr0 * math.cos(aa), rr0 * math.sin(aa)
                            arc_pts.append(proj(uu, vv, surf_z(uu, vv), phi))
                        set_poly(sweep_segs[k], arc_pts)
                        sweep_segs[k].set_stroke(col, width=1.5, opacity=0.95)
                        k += 1
            else:
                for mob in sweep_segs:
                    mob.set_stroke(M_CYAN, width=1.5, opacity=0.0)

            # 极坐标：θ 与 I² 同源，det 与 r 同源
            a = theta
            direction = np.array([math.cos(a), math.sin(a), 0])
            white_line.set_points_by_ends(PC, PC + PR * direction)

            inner = float(np.clip(0.15 + 0.18 * rr, 0.14, 0.72))
            outer = min(inner + 0.16, 0.88)
            n_arc = 14
            sector_pts = []
            for ang in np.linspace(a - 0.13, a + 0.13, n_arc):
                sector_pts.append(PC + inner * np.array([math.cos(ang), math.sin(ang), 0]))
            for ang in np.linspace(a + 0.13, a - 0.13, n_arc):
                sector_pts.append(PC + outer * np.array([math.cos(ang), math.sin(ang), 0]))
            set_poly(sector, sector_pts, closed=True)

            mid = (inner + outer) / 2
            lbl_r.move_to(PC + mid * direction)
            lbl_dr.move_to(PC + (outer + 0.12) * direction)
            off = a + 0.32
            lbl_rdth.move_to(PC + (outer + 0.05) * np.array([math.cos(off), math.sin(off), 0]))

            # 雅可比数值：矩阵 det 必须等于当前 r
            n00.set_value(math.cos(a))
            n01.set_value(-rr * math.sin(a))
            n10.set_value(math.sin(a))
            n11.set_value(rr * math.cos(a))
            det_val.set_value(rr)

            # I² 进度
            est.set_value(i2)
            frac = float(np.clip(i2 / math.pi, 0, 1))
            bar_fill.set_points_by_ends([bar_x0, bar_y, 0], [bar_x0 + (bar_x1 - bar_x0) * frac, bar_y, 0])

            # 16.15s 循环重置瞬间的标题闪烁
            blink = 16.05 <= t <= 16.38
            title.set_opacity(0 if blink else 1)
            cursor.set_opacity(1 if blink else 0)

        apply(0)

        class Timeline(Animation):
            def __init__(self, mobject, **kw):
                super().__init__(mobject, run_time=DURATION, **kw)

            def interpolate_mobject(self, alpha):
                apply(alpha * DURATION)

        self.play(Timeline(dynamic), rate_func=linear)
        apply(DURATION - 1 / config.frame_rate)
        self.wait(0.08)
