#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Manim 数学动画：高斯散度定理 —— 边界通量 = 内部源强总和

与前作同风格：
  - 1080x1350 竖屏、30fps、约 22.3s
  - 公式全程固定，只动参数 / 面元 / 箭头 / 动画
  - 无 LaTeX：公式用 matplotlib mathtext 预渲染为透明 PNG
  - 暗色底 #101113、中灰清晰线条、青/红边缘曲线、纯内存动态数字
  - 标题双语，整体淡入后常亮，无打字机竖线

画面：
  左 = 斜投影立方体 Ω=[-a,a]^3，红色源点 p，六个面按 F·n 着色（青=流出，红=流入），
       可见三个面上画出穿出的箭头；
  右 = 六个面的通量条形图（±x/±y/±z），负通量向左延伸；
  下方 = 两个动态读数「面通量总和」「体积分 ∭∇·F dV」最终相等；
  底部 = 五行固定公式链 + 随阶段切换的中文讲述行。

运行：
    python precheck.py
    manim -pql preview_scene.py DivergenceTheorem
    python check_frames.py
    manim -pqh divergence_scene.py DivergenceTheorem
    python add_bgm.py
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
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
M_CYAN = "#4AD6C8"
M_RED = "#E86052"
M_ORANGE = "#EE9448"

_SERIF = str(Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSerif.ttf")

# ---------------------------------------------------------------------------
# 无 LaTeX 公式预渲染 + 纯内存动态数字（沿用系列规范）
# ---------------------------------------------------------------------------
_MATH_CACHE: dict[tuple, Image.Image] = {}


def _math_img(expr: str, size: int, color) -> Image.Image:
    if isinstance(color, str):
        color = tuple(int(c) for c in ManimColor(color).to_rgb() * 255)
    key = (expr, size, color)
    if key in _MATH_CACHE:
        return _MATH_CACHE[key]
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    fig = Figure(figsize=(16.0, 2.4), dpi=200)
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


def mi(expr: str, name: str, color=C_FORMULA, height=0.30, size=27) -> ImageMobject:
    path = FORMULA_DIR / f"dv_{name}.png"
    if not path.exists():
        _math_img(expr, size, color).save(path)
    m = ImageMobject(str(path))
    if m.height > 0:
        m.scale(height / m.height)
    return m


def text_img(text: str, px: int, color=C_TEXT, height=0.3) -> ImageMobject:
    img = Image.new("RGBA", (1600, 200), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    font = ImageFont.truetype(_SERIF, px)
    d.text((20, 20), text, font=font, fill=(*color, 255))
    bbox = img.getchannel("A").getbbox()
    img = img.crop(bbox) if bbox else img
    m = ImageMobject(np.asarray(img))
    if m.height > 0:
        m.scale(height / m.height)
    return m


def cn_text(text: str, px: int, color=C_TEXT, height=0.3) -> ImageMobject:
    img = Image.new("RGBA", (1800, 220), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", px)
    d.text((20, 16), text, font=font, fill=(*color, 255))
    bbox = img.getchannel("A").getbbox()
    img = img.crop(bbox) if bbox else img
    m = ImageMobject(np.asarray(img))
    if m.height > 0:
        m.scale(height / m.height)
    return m


class PixelNumber(ImageMobject):
    _FONT = str(Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf")

    def __init__(self, value=0.0, places: int = 3, color=M_TEXT, px: int = 40,
                 nchars: int | None = None, **kwargs):
        self.places = places
        self.value = value
        self._color = tuple(int(c) for c in ManimColor(color).to_rgb() * 255) \
            if not isinstance(color, tuple) else color
        self._font = ImageFont.truetype(self._FONT, px)
        n = nchars if nchars is not None else (1 + 1 + places + (1 if places else 0))
        self._cw, self._ch = int(px * (n * 0.62 + 0.4)), int(px * 1.6)
        super().__init__(np.zeros((self._ch, self._cw, 4), dtype=np.uint8), **kwargs)
        self.set_value(value)

    def set_value(self, value):
        self.value = value
        txt = value if isinstance(value, str) else f"{abs(value):.{self.places}f}"
        img = Image.new("RGBA", (self._cw, self._ch), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        box = d.textbbox((0, 0), txt, font=self._font)
        tw, th = box[2] - box[0], box[3] - box[1]
        d.text(((self._cw - tw) / 2 - box[0], (self._ch - th) / 2 - box[1]),
               txt, font=self._font, fill=(*self._color, 255))
        self.pixel_array[:] = np.asarray(img)


# ---------------------------------------------------------------------------
# 数学设置
#   F(r) = r - p,  ∇·F = 3
#   Ω = [-a,a]^3,  ∭_Ω 3 dV = 3(2a)^3 = 24 a^3
#   单个面（法向 n_i = ±e_i）通量 q_i = (±a - p_i)(2a)^2
#   Σ q_i = 24 a^3，与源点 p 的位置无关
# ---------------------------------------------------------------------------
P_START = (0.25, 0.15, 0.20)
P_FAR = (1.30, -0.30, 0.35)
A_FACES_0, A_FIXED = 0.75, 1.05

RES_W, RES_H = 900, 470
BG_RGB = (16, 17, 19)
PIX_FONT = ImageFont.truetype(_SERIF, 17)
PIX_FONT_B = ImageFont.truetype(_SERIF, 19)

CX, CY, PROJ = 300, 246, 96.0
BAR_X, BAR_BAR0, BAR_W = 636, 741, 176
FACE_NAMES = ["+x", "-x", "+y", "-y", "+z", "-z"]


def face_fluxes(a: float, p) -> list[float]:
    px, py, pz = p
    s = (2.0 * a) ** 2
    return [(a - px) * s, (a + px) * s,
            (a - py) * s, (a + py) * s,
            (a - pz) * s, (a + pz) * s]


def project(v, a: float):
    x, y, z = v
    u = CX + PROJ * a * (x - 0.56 * y)
    w = CY + PROJ * a * (0.30 * x + 0.34 * y - 0.92 * z)
    return (u, w)


def vertex(a: float, sx: int, sy: int, sz: int):
    return project((sx * a, sy * a, sz * a), a)


FACE_VERTS = {
    "+x": [(1, -1, -1), (1, 1, -1), (1, 1, 1), (1, -1, 1)],
    "-x": [(-1, -1, -1), (-1, 1, -1), (-1, 1, 1), (-1, -1, 1)],
    "+y": [(-1, 1, -1), (1, 1, -1), (1, 1, 1), (-1, 1, 1)],
    "-y": [(-1, -1, -1), (1, -1, -1), (1, -1, 1), (-1, -1, 1)],
    "+z": [(-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)],
    "-z": [(-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1)],
}
VISIBLE = ["+x", "-y", "+z"]
NORMAL = {"+x": (1, 0, 0), "-x": (-1, 0, 0), "+y": (0, 1, 0),
          "-y": (0, -1, 0), "+z": (0, 0, 1), "-z": (0, 0, -1)}


PHASES = [
    ("intro",   0.00,  1.40),
    ("field",   1.40,  5.40),
    ("faces",   5.40, 10.80),
    ("volume", 10.80, 15.80),
    ("move",   15.80, 19.40),
    ("final",  19.40, 20.40),
    ("replay", 20.40, DURATION),
]


def phase_info(t: float):
    for name, t0, t1 in PHASES:
        if t < t1:
            return name, (0.0 if name == "intro"
                          else float(np.clip((t - t0) / (t1 - t0), 0.0, 1.0)))
    return "replay", 1.0


def box_size(t: float) -> float:
    phase, prog = phase_info(t)
    if phase in ("intro", "field"):
        return A_FACES_0
    if phase == "faces":
        return A_FACES_0 + (A_FIXED - A_FACES_0) * prog
    return A_FIXED


def source_pos(t: float):
    phase, prog = phase_info(t)
    if phase in ("intro", "field", "faces", "volume"):
        return P_START
    if phase == "move":
        # 往复移动，末段回到起点：各面通量重分配而总和不变
        w = 0.5 - 0.5 * math.cos(2 * math.pi * prog)
        return tuple(P_START[i] + (P_FAR[i] - P_START[i]) * w for i in range(3))
    return P_START


def reveal_fraction(t: float) -> float:
    phase, prog = phase_info(t)
    if phase in ("intro", "field"):
        return 0.0
    if phase == "faces":
        return float(np.clip(prog * 1.08, 0, 1))
    return 1.0


def volume_fraction(t: float) -> float:
    phase, prog = phase_info(t)
    if phase in ("intro", "field", "faces"):
        return 0.0
    if phase == "volume":
        return float(np.clip(prog * 1.05, 0, 1))
    return 1.0


def make_base() -> Image.Image:
    img = Image.new("RGBA", (RES_W, RES_H), (*BG_RGB, 255))
    d = ImageDraw.Draw(img, "RGBA")
    # 左侧立方体区网格
    for gx in range(60, 560, 50):
        d.line([(gx, 20), (gx, RES_H - 20)], fill=(*C_GRID_DIM, 65), width=1)
    for gy in range(30, RES_H - 20, 50):
        d.line([(20, gy), (566, gy)], fill=(*C_GRID_DIM, 65), width=1)
    d.text((30, 18), "VECTOR FIELD  F = r - p     DOMAIN  [-a,a]^3",
           font=PIX_FONT, fill=(*C_GRID, 255))
    # 右侧条形图区
    d.line([(590, 28), (590, RES_H - 28)], fill=(*C_GRID_DIM, 130), width=1)
    d.text((606, 18), "FLUX ON EACH FACE  F.n * Area",
           font=PIX_FONT, fill=(*C_GRID, 255))
    d.line([(BAR_BAR0, 48), (BAR_BAR0, 408)], fill=(*C_GRID, 190), width=2)
    for i, name in enumerate(FACE_NAMES):
        y = 88 + i * 55
        d.text((BAR_X - 50, y - 10), name, font=PIX_FONT_B, fill=(*C_GRID, 255))
        d.line([(BAR_BAR0 - BAR_W - 14, y + 10), (BAR_BAR0 + BAR_W + 14, y + 10)],
               fill=(*C_GRID_DIM, 90), width=1)
    d.text((BAR_X + 8, 400), "cyan: outward    red: inward",
           font=PIX_FONT, fill=(*C_GRID_DIM, 255))
    return img


BASE_IMG = make_base()


def draw_arrow(d, p0, p1, color, width=2, head=7.0):
    d.line([p0, p1], fill=(*color, 255), width=width)
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    n = math.hypot(dx, dy)
    if n < 1e-6:
        return
    ux, uy = dx / n, dy / n
    px, py = -uy, ux
    a = (p1[0] - head * ux + 0.42 * head * px, p1[1] - head * uy + 0.42 * head * py)
    b = (p1[0] - head * ux - 0.42 * head * px, p1[1] - head * uy - 0.42 * head * py)
    d.polygon([p1, a, b], fill=(*color, 255))


class DivergenceField(ImageMobject):
    def __init__(self, disp_height=3.45):
        super().__init__(np.asarray(BASE_IMG).copy())
        self.scale(disp_height / self.height)

    def redraw(self, t: float):
        phase, prog = phase_info(t)
        arr = self.pixel_array
        arr[:] = np.asarray(BASE_IMG)
        img = Image.fromarray(arr, mode="RGBA")
        d = ImageDraw.Draw(img, "RGBA")
        a = box_size(t)
        p = source_pos(t)
        qs = face_fluxes(a, p)
        rev = reveal_fraction(t)
        volfrac = volume_fraction(t)

        # --- 立方体：先画隐蔽面轮廓 ---
        for name in ("-x", "+y", "-z"):
            pts = [vertex(a, *v) for v in FACE_VERTS[name]]
            d.line(pts + [pts[0]], fill=(*C_GRID_DIM, 150), width=1)

        # --- 体积阶段的均匀源强点阵 ∇·F = 3 ---
        if volfrac > 0:
            for ix in (-0.55, 0, 0.55):
                for iy in (-0.55, 0, 0.55):
                    for iz in (-0.55, 0, 0.55):
                        pt = project((ix * a, iy * a, iz * a), a)
                        alpha = int(120 * volfrac)
                        d.ellipse([pt[0] - 2, pt[1] - 2, pt[0] + 2, pt[1] + 2],
                                  fill=(*C_ORANGE, alpha))

        # --- 可见面：按 F·n 着色（青=流出，红=流入）---
        qmax = max(1.0, 8.0 * a ** 3)
        for name in VISIBLE:
            idx = FACE_NAMES.index(name)
            q = qs[idx]
            col = C_CYAN if q >= 0 else C_RED
            alpha = int(min(1.0, abs(q) / qmax) * 88 * rev)
            pts = [vertex(a, *v) for v in FACE_VERTS[name]]
            d.polygon(pts, fill=(*col, alpha))
            d.line(pts + [pts[0]], fill=(*col, int(210 * max(rev, 0.2))), width=2)

        # --- 面上通量箭头：沿外法向穿出 ---
        if rev > 0.05:
            for name in VISIBLE:
                idx = FACE_NAMES.index(name)
                q = qs[idx]
                col = C_CYAN if q >= 0 else C_RED
                nv = NORMAL[name]
                strength = np.clip(abs(q) / qmax, 0.15, 1.0)
                for s1, s2 in ((-0.48, -0.42), (0.0, 0.10), (0.48, 0.48)):
                    if name == "+x":
                        base = (a, s1 * a, s2 * a)
                    elif name == "-y":
                        base = (s1 * a, -a, s2 * a)
                    else:
                        base = (s1 * a, s2 * a, a)
                    p0 = project(base, a)
                    p1 = project(tuple(base[i] + 0.34 * strength * nv[i]
                                       for i in range(3)), a)
                    if q < 0:  # 流入：箭头反向
                        p0, p1 = p1, p0
                    draw_arrow(d, p0, p1, col, width=2, head=6.5)

        # --- 立方体棱 ---
        edges = [((-1, -1, -1), (1, -1, -1)), ((1, -1, -1), (1, 1, -1)),
                 ((1, 1, -1), (-1, 1, -1)), ((-1, 1, -1), (-1, -1, -1)),
                 ((-1, -1, 1), (1, -1, 1)), ((1, -1, 1), (1, 1, 1)),
                 ((1, 1, 1), (-1, 1, 1)), ((-1, 1, 1), (-1, -1, 1)),
                 ((-1, -1, -1), (-1, -1, 1)), ((1, -1, -1), (1, -1, 1)),
                 ((1, 1, -1), (1, 1, 1)), ((-1, 1, -1), (-1, 1, 1))]
        ecol = (*C_GRID, 215) if phase in ("intro", "field") else (*C_GRID, 245)
        for v0, v1 in edges:
            d.line([vertex(a, *v0), vertex(a, *v1)], fill=ecol, width=2)

        # --- 源点 p 与内部短箭头 ---
        if phase != "intro":
            sp = project(p, a)
            d.ellipse([sp[0] - 6, sp[1] - 6, sp[0] + 6, sp[1] + 6],
                      fill=(*C_RED, 255))
            d.text((sp[0] + 10, sp[1] - 10), "p", font=PIX_FONT_B,
                   fill=(*C_RED, 255))
            if phase in ("field", "faces"):
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    p0 = (sp[0] + 12 * dx, sp[1] + 12 * dy)
                    p1 = (sp[0] + 38 * dx, sp[1] + 30 * dy)
                    draw_arrow(d, p0, p1, C_RED, width=2, head=6)

        # --- 六个面的通量条 ---
        for i, q in enumerate(qs):
            y = 88 + i * 55
            frac = np.clip(q / (2.0 * qmax), -1.0, 1.0)
            end = BAR_BAR0 + BAR_W * frac
            col = C_CYAN if q >= 0 else C_RED
            if rev > 0 or phase == "field":
                alpha = 235 if phase != "field" else 60
                d.rectangle([min(BAR_BAR0, end), y - 6, max(BAR_BAR0, end), y + 6],
                            fill=(*col, alpha))
                d.line([(end, y - 10), (end, y + 10)], fill=(*col, 255), width=2)

        self.pixel_array[:] = np.asarray(img)


class DivergenceTheorem(Scene):
    def construct(self):
        # 标题：双语、无光标竖线，整体淡入后常亮
        cn_title = cn_text("高斯散度定理", 50, C_TEXT, 0.30).move_to([0, 4.68, 0])
        en_title = text_img("D I V E R G E N C E   T H E O R E M   /   F L U X   =   S O U R C E",
                            19, C_CYAN, 0.11).move_to([0, 4.37, 0])
        watermark = text_img("AIMaker-Lab", 17, C_GRID, 0.09).move_to([3.05, 4.68, 0])
        title = Group(cn_title, en_title, watermark)

        leg = Group(
            Group(Line(LEFT * 0.13, RIGHT * 0.13, color=M_RED, stroke_width=3),
                  text_img("FIELD  F = r - p", 23, C_RED, 0.14)).arrange(RIGHT, buff=0.06),
            Group(Line(LEFT * 0.13, RIGHT * 0.13, color=M_CYAN, stroke_width=3),
                  text_img("FACE FLUX  F.n", 23, C_CYAN, 0.14)).arrange(RIGHT, buff=0.06),
            Group(Line(LEFT * 0.13, RIGHT * 0.13, color=M_ORANGE, stroke_width=3),
                  text_img("SOURCE  div F = 3", 23, C_ORANGE, 0.14)).arrange(RIGHT, buff=0.06),
        ).arrange(RIGHT, buff=0.16).move_to([0, 4.02, 0])

        plane = DivergenceField(disp_height=3.45).move_to([0, 1.95, 0])

        lbl_s = cn_text("边界通量总和", 29, C_CYAN, 0.22).move_to([-2.22, -0.10, 0])
        lbl_v = cn_text("内部体积分", 29, C_ORANGE, 0.22).move_to([2.20, -0.10, 0])
        num_s = PixelNumber(0.0, 3, M_CYAN, 40, nchars=7).set(height=0.30).move_to([-2.22, -0.52, 0])
        num_v = PixelNumber(0.0, 3, M_ORANGE, 40, nchars=7).set(height=0.30).move_to([2.20, -0.52, 0])

        NARR = [
            ("intro",  "闭合曲面上的通量，和内部的源有什么关系？", C_TEXT),
            ("field",  "取向量场 F = r − p：它处处发散，散度等于 3", C_RED),
            ("faces",  "六个面逐个累加 F·n：红面流入，青面流出", C_CYAN),
            ("volume", "内部每一点的源强都是 3，体积分给出同一个数", C_ORANGE),
            ("move",   "移动源点 p：各面通量重新分配，总和始终不变", C_ORANGE),
            ("final",  "边界通量 = 内部源强总和：这就是高斯散度定理", C_CYAN),
            ("replay", "看边界，也看内部：两条路算出同一个答案", C_TEXT),
        ]
        narr_pos = [0, -1.05, 0]
        narr_lbls = {}
        for key, s, col in NARR:
            lb = cn_text(s, 27, col, 0.23).move_to(narr_pos)
            lb.move_to([9, 9, 0])
            narr_lbls[key] = lb

        f1 = mi(r"$\oint_{\partial\Omega}\mathbf{F}\cdot\mathbf{n}\,dS"
                r"=\iiint_{\Omega}\nabla\cdot\mathbf{F}\,dV$",
                "f1", C_FORMULA, 0.42, 27).move_to([0, -1.70, 0])
        f2 = mi(r"$\mathbf{F}=\mathbf{r}-\mathbf{p}"
                r"=(x-p_x,\ y-p_y,\ z-p_z)$",
                "f2", C_FORMULA, 0.30, 27).move_to([0, -2.30, 0])
        f3 = mi(r"$\nabla\cdot\mathbf{F}=1+1+1=3$",
                "f3", C_FORMULA, 0.32, 27).move_to([0, -2.90, 0])
        f4 = mi(r"$\Omega=[-a,a]^3:\quad \iiint_{\Omega}3\,dV=24a^3$",
                "f4", C_FORMULA, 0.34, 27).move_to([0, -3.50, 0])
        f5 = mi(r"$\sum_{\mathrm{faces}}(\pm a-p_i)(2a)^2=24a^3$",
                "f5", C_FORMULA, 0.30, 27).move_to([0, -4.15, 0])
        formulas = Group(f1, f2, f3, f4, f5)
        FORMULA_LIST = [f1, f2, f3, f4, f5]
        phase_formula = {"intro": 0, "field": 2, "faces": 1, "volume": 3,
                         "move": 4, "final": 0, "replay": 0}

        self.add(leg, lbl_s, lbl_v)
        dynamic = Group(title, plane, num_s, num_v, *narr_lbls.values(), formulas)
        self.add(dynamic)

        def apply(t):
            phase, prog = phase_info(t)
            a = box_size(t)
            p = source_pos(t)
            qs = face_fluxes(a, p)
            total = sum(qs)
            rev = reveal_fraction(t)
            vf = volume_fraction(t)

            # 逐面累加（按 ±x ±y ±z 顺序点亮）
            k = rev * len(qs)
            surf = sum(qs[:int(k)]) + (k - int(k)) * qs[int(k)] if int(k) < len(qs) else total
            num_s.set_value(surf)
            num_v.set_value("—" if vf <= 0.0 else total * vf)

            plane.redraw(t)
            for key, lb in narr_lbls.items():
                lb.move_to(narr_pos if key == phase else [9, 9, 0])
            hi = phase_formula[phase]
            for i, fm in enumerate(FORMULA_LIST):
                fm.set_opacity(1.0 if i == hi else 0.48)

        apply(0)
        self.play(FadeIn(title), run_time=0.7)
        self.play(FadeIn(formulas), run_time=0.3)

        class Timeline(Animation):
            def __init__(self, mobject, **kw):
                super().__init__(mobject, run_time=DURATION, **kw)

            def interpolate_mobject(self, alpha):
                apply(alpha * DURATION)

        self.play(Timeline(dynamic), rate_func=linear)
        apply(DURATION - 1 / config.frame_rate)
        self.wait(0.08)
