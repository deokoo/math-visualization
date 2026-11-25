#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Manim 数学动画：格林公式 —— 沿边界绕一圈的环流 = 区域内部旋度之和

视觉规范与前作一致：
  - 1080x1350 竖屏、30fps、约 22.3s
  - 公式全程固定，只动参数 / 网格 / 动画
  - 无 LaTeX：公式用 matplotlib mathtext 预渲染为透明 PNG
  - 暗色底 #101113、中灰清晰线条、青/红边缘曲线、纯内存动态数字

画面：
  中央 = 闭合曲线 C 围成区域 D，内部向量场（灰青小箭头）
         ① 红点沿 C 逆时针走一圈，青色轨迹生长，左侧「边界环流 ∮」累加
         ② 区域切成小格，小环流沿公共边两两抵消（红色 × 标记）
         ③ 旋度热图自内浮现，右侧「区域旋度和 ∬」逐格累加
         ④ 两个数值收敛到同一值 10.401，公式链整体点亮
  底部 = 5 行固定公式链；中文讲述行随阶段切换

运行：
    manim -pql green_theorem_scene.py GreenTheorem
    manim -pqh green_theorem_scene.py GreenTheorem
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
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
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
    path = FORMULA_DIR / f"gt_{name}.png"
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
    """中文标题/讲述行：微软雅黑渲染（衬线字体缺中文字形）。"""
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
    """PIL 渲染到固定 RGBA 数组，set_value 原地更新，零文件 IO。"""

    _FONT = str(Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf")

    def __init__(self, value=0.0, places: int = 2, color=M_TEXT, px: int = 40,
                 nchars: int | None = None, **kwargs):
        self.places = places
        self.value = value
        self._color = tuple(int(c) for c in ManimColor(color).to_rgb() * 255) \
            if not isinstance(color, tuple) else color
        self._font = ImageFont.truetype(self._FONT, px)
        n = nchars if nchars is not None else (1 + 1 + places + (1 if places else 0))
        self._cw, self._ch = int(px * (n * 0.62 + 0.4)), int(px * 1.6)
        canvas = np.zeros((self._ch, self._cw, 4), dtype=np.uint8)
        super().__init__(canvas, **kwargs)
        self.set_value(value)

    def set_value(self, value):
        self.value = value
        if isinstance(value, str):
            txt = value
        else:
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


# ---------------------------------------------------------------------------
# 数学对象：区域 D、向量场 F、旋度
#   r(θ) = 1.28 + 0.16 sin3θ + 0.09 cos5θ      （逆时针闭合曲线 C）
#   F = (P, Q) = (−y + 0.35xy, x + 0.25xy)
#   curl = ∂Q/∂x − ∂P/∂y = 2 − 0.35x + 0.25y
#   数值校验：∮_C Pdx+Qdy = 10.400876，∬_D curl dA = 10.400870（差 5.7e−6）
# ---------------------------------------------------------------------------
def radius(theta):
    return 1.28 + 0.16 * np.sin(3 * theta) + 0.09 * np.cos(5 * theta)


def field_P(x, y):
    return -np.asarray(y) + 0.35 * np.asarray(x) * np.asarray(y)


def field_Q(x, y):
    return np.asarray(x) + 0.25 * np.asarray(x) * np.asarray(y)


def curl(x, y):
    return 2.0 - 0.35 * np.asarray(x) + 0.25 * np.asarray(y)


# 边界高精度采样与环流累积（预计算，逐帧只查表）
TH = np.linspace(0.0, 2 * np.pi, 4001)
RR = radius(TH)
BX, BY = RR * np.cos(TH), RR * np.sin(TH)
BDX, BDY = np.gradient(BX, TH), np.gradient(BY, TH)
BND = field_P(BX, BY) * BDX + field_Q(BX, BY) * BDY
CUM_LINE = np.concatenate([[0.0], np.cumsum(0.5 * (BND[:-1] + BND[1:]) * np.diff(TH))])
TOTAL_LINE = float(CUM_LINE[-1])
AREA_D = float(0.5 * np.trapezoid(BX * BDY - BY * BDX, TH))

# 网格单元（10×10，中心在 D 内者保留），权重归一使逐格累积精确收敛到 TOTAL_LINE
CELL = 0.32
CELL_XS = np.arange(-1.6, 1.6, CELL) + CELL / 2
CELLS = []
for cx in CELL_XS:
    for cy in CELL_XS:
        # 边界单元也纳入（+0.16 扩张），再按区域蒙版裁剪，避免热图出现矩形缺口
        if math.hypot(cx, cy) <= radius(math.atan2(cy, cx)) + 0.16:
            CELLS.append((float(cx), float(cy), float(curl(cx, cy))))
_raw = sum(c * CELL * CELL for _, _, c in CELLS)
CELLS = [(x, y, c * TOTAL_LINE / _raw) for x, y, c in CELLS]
# 按极角排序：红色热图沿边界推进方向扫过区域
CELLS.sort(key=lambda e: math.atan2(e[1], e[0]) % (2 * np.pi))
CUM_AREA = np.concatenate([[0.0], np.cumsum([w for _, _, w in CELLS])])

# 公共边抵消标记：取相邻单元对的中点（最多 18 个，避免拥挤）
CANCEL_MARKS = []
_pos = {(round(x / CELL), round(y / CELL)): (x, y, w) for x, y, w in CELLS}
for (ix, iy) in list(_pos):
    for dx, dy in ((1, 0), (0, 1)):
        if (ix + dx, iy + dy) in _pos:
            x, y, _ = _pos[(ix, iy)]
            CANCEL_MARKS.append((x + dx * CELL / 2, y + dy * CELL / 2))
            if len(CANCEL_MARKS) >= 18:
                break
    if len(CANCEL_MARKS) >= 18:
        break

# ---------------------------------------------------------------------------
# 区域位图：网格 + 区域填充 + 向量场 + 边界（静态底图）；每帧叠加动态层
# ---------------------------------------------------------------------------
RES = 480
WORLD = 1.76
BG_RGB = (16, 17, 19)


def w2p(x, y):
    return ((x + WORLD) / (2 * WORLD) * RES,
            (WORLD - y) / (2 * WORLD) * RES)


def make_base() -> Image.Image:
    img = Image.new("RGBA", (RES, RES), (*BG_RGB, 255))
    d = ImageDraw.Draw(img, "RGBA")

    # 淡网格
    g = -1.6
    while g <= 1.6001:
        gx, _ = w2p(g, 0)
        _, gy = w2p(0, g)
        d.line([(gx, 0), (gx, RES)], fill=(*C_GRID_DIM, 120), width=1)
        d.line([(0, gy), (RES, gy)], fill=(*C_GRID_DIM, 120), width=1)
        g += 0.4

    # 区域 D 填充（略亮于背景）
    poly = [w2p(float(bx), float(by)) for bx, by in zip(BX[::5], BY[::5])]
    d.polygon(poly, fill=(24, 29, 31, 255))

    # 向量场小箭头（只画在 D 内）
    gx = np.arange(-1.5, 1.501, 0.375)
    X, Y = np.meshgrid(gx, gx)
    for xx, yy in zip(X.ravel(), Y.ravel()):
        if math.hypot(xx, yy) > radius(math.atan2(yy, xx)) - 0.10:
            continue
        vx, vy = float(field_P(xx, yy)), float(field_Q(xx, yy))
        nrm = math.hypot(vx, vy)
        if nrm < 1e-6:
            continue
        ux, uy = vx / nrm, vy / nrm
        ax, ay = xx - 0.105 * ux, yy - 0.105 * uy
        bx, by = xx + 0.105 * ux, yy + 0.105 * uy
        ps, pe = w2p(ax, ay), w2p(bx, by)
        d.line([ps, pe], fill=(88, 122, 124, 210), width=2)
        ang = math.atan2(pe[1] - ps[1], pe[0] - ps[0])
        hlen = 0.052 * RES / (2 * WORLD)
        for da in (math.pi - 0.48, math.pi + 0.48):
            d.line([pe, (pe[0] + hlen * math.cos(ang + da),
                         pe[1] + hlen * math.sin(ang + da))],
                   fill=(88, 122, 124, 210), width=2)

    # 坐标轴（细、暗）
    cx, cy = w2p(0, 0)
    d.line([(cx, 0), (cx, RES)], fill=(*C_GRID_DIM, 180), width=1)
    d.line([(0, cy), (RES, cy)], fill=(*C_GRID_DIM, 180), width=1)

    # 边界 C：青色闭合曲线
    pts = [w2p(float(bx), float(by)) for bx, by in zip(BX[::3], BY[::3])]
    pts.append(pts[0])
    d.line(pts, fill=(*C_CYAN, 255), width=3, joint="curve")
    return img


BASE_IMG = make_base()


def make_region_mask() -> Image.Image:
    """区域 D 的灰度蒙版：热图严格贴合边界曲线，不留方块缺口。"""
    m = Image.new("L", (RES, RES), 0)
    d = ImageDraw.Draw(m)
    pts = [w2p(float(bx), float(by)) for bx, by in zip(BX[::3], BY[::3])]
    d.polygon(pts, fill=255)
    return m


REGION_MASK = make_region_mask()


class GreenField(ImageMobject):
    """numpy/PIL 整块位图：静态底图 + 逐帧动态层，原位更新 pixel_array。"""

    def __init__(self, disp_height=3.90):
        super().__init__(np.asarray(BASE_IMG).copy())
        self.scale(disp_height / self.height)

    def redraw(self, t: float):
        # 每帧必须先清空，否则动态层会累积成糊块
        arr = self.pixel_array
        arr[:] = np.asarray(BASE_IMG)
        img = Image.fromarray(arr, mode="RGBA")
        d = ImageDraw.Draw(img, "RGBA")

        phase, prog = phase_info(t)

        # 单元网格（阶段②起）
        if phase in ("cells", "area", "equal", "replay"):
            for x0 in np.arange(-1.6, 1.6, CELL):
                for y0 in np.arange(-1.6, 1.6, CELL):
                    if math.hypot(x0 + CELL / 2, y0 + CELL / 2) > \
                            radius(math.atan2(y0 + CELL / 2, x0 + CELL / 2)) - 0.04:
                        continue
                    p0, p1 = w2p(x0, y0 + CELL), w2p(x0 + CELL, y0)
                    d.rectangle([p0, p1], outline=(110, 118, 122, 150), width=1)

        # 旋度热图：按单元逐个揭示（阶段③起）
        reveal = 0
        if phase in ("area", "equal", "replay"):
            reveal = len(CELLS) if phase != "area" else max(1, int(prog * len(CELLS)))
        if phase == "cells":
            reveal = 0
        if reveal:
            heat = Image.new("RGBA", (RES, RES), (0, 0, 0, 0))
            hd = ImageDraw.Draw(heat, "RGBA")
            for x, y, w in CELLS[:reveal]:
                c = curl(x, y)
                k = float(np.clip((c - 1.2) / 1.2, 0, 1))
                col = (int(58 + 150 * k), int(24 + 44 * k), int(24 + 34 * k), 130)
                p0, p1 = w2p(x - CELL / 2, y + CELL / 2), w2p(x + CELL / 2, y - CELL / 2)
                hd.rectangle([p0, p1], fill=col)
            img.paste(heat, (0, 0), REGION_MASK)   # 严格裁到曲线 C 内部

        # 小环流与公共边抵消（阶段②/③）
        if phase in ("cells", "area"):
            for x, y, _ in CELLS[::3]:
                px, py = w2p(x, y)
                rad = 0.075 * RES / (2 * WORLD)
                d.arc([px - rad, py - rad, px + rad, py + rad],
                      start=35, end=300, fill=(*C_RED, 215), width=2)
                # 箭头
                a = math.radians(300)
                hx, hy = px + rad * math.cos(a), py + rad * math.sin(a)
                d.line([(hx, hy), (hx - 5, hy - 1), (hx - 2, hy + 5)],
                       fill=(*C_RED, 215), width=2)
            for mx, my in CANCEL_MARKS:
                px, py = w2p(mx, my)
                s = 4
                d.line([(px - s, py - s), (px + s, py + s)], fill=(238, 190, 90, 235), width=2)
                d.line([(px - s, py + s), (px + s, py - s)], fill=(238, 190, 90, 235), width=2)

        # 边界行走与轨迹（阶段①/回放）
        line_prog = 0.0
        if phase == "line":
            line_prog = prog
        elif phase == "replay":
            line_prog = min(1.0, prog / 0.62)
        if line_prog > 0:
            idx = max(1, int(line_prog * (len(TH) - 1)))
            step = max(1, idx // 700)
            pts = [w2p(float(BX[k]), float(BY[k])) for k in range(0, idx + 1, step)]
            pts.append(w2p(float(BX[idx]), float(BY[idx])))
            d.line(pts, fill=(*C_RED, 70), width=9, joint="curve")
            d.line(pts, fill=(255, 128, 112, 255), width=4, joint="curve")
            px, py = pts[-1]
            d.ellipse([px - 7, py - 7, px + 7, py + 7], fill=(255, 176, 160, 255))
            d.ellipse([px - 3, py - 3, px + 3, py + 3], fill=(255, 245, 235, 255))

        self.pixel_array[:] = np.asarray(img)


# ---------------------------------------------------------------------------
# 阶段表
# ---------------------------------------------------------------------------
PHASES = [
    ("intro",  0.00,  1.30),
    ("line",   1.30,  5.60),
    ("cells",  5.60,  9.80),
    ("area",   9.80, 14.60),
    ("equal", 14.60, 17.80),
    ("replay",17.80, DURATION),
]


def phase_info(t: float):
    for name, t0, t1 in PHASES:
        if t < t1:
            if name == "intro":
                return name, 0.0
            return name, float(np.clip((t - t0) / (t1 - t0), 0.0, 1.0))
    return "replay", 1.0


def smoothstep(v):
    v = float(np.clip(v, 0.0, 1.0))
    return v * v * (3 - 2 * v)


# ---------------------------------------------------------------------------
# 场景
# ---------------------------------------------------------------------------
class GreenTheorem(Scene):
    def construct(self):
        # 双语标题（新约定）
        cn_title = cn_text("格林公式", 50, C_TEXT, 0.30).move_to([0, 4.68, 0])
        en_title = text_img("G R E E N ' S   T H E O R E M   /   C I R C U L A T I O N   &   C U R L",
                            20, C_CYAN, 0.11).move_to([0, 4.37, 0])
        watermark = text_img("AIMaker-Lab", 17, C_GRID, 0.09).move_to([3.05, 4.68, 0])
        title = Group(cn_title, en_title, watermark)
        cursor = Line(UP * 0.16, DOWN * 0.16, color=M_TEXT, stroke_width=2.5).move_to([0, 4.68, 0])
        cursor.set_opacity(0)

        # 图例
        leg = Group(
            Group(Line(LEFT * 0.13, RIGHT * 0.13, color=M_CYAN, stroke_width=3),
                  text_img("BOUNDARY C", 24, C_CYAN, 0.14)).arrange(RIGHT, buff=0.06),
            Group(Line(LEFT * 0.13, RIGHT * 0.13, color=M_RED, stroke_width=3),
                  text_img("CIRCULATION", 24, C_RED, 0.14)).arrange(RIGHT, buff=0.06),
            Group(Line(LEFT * 0.13, RIGHT * 0.13, color=M_ORANGE, stroke_width=3),
                  text_img("CURL DENSITY", 24, C_ORANGE, 0.14)).arrange(RIGHT, buff=0.06),
        ).arrange(RIGHT, buff=0.20).move_to([0, 4.02, 0])

        # 中央区域
        plane = GreenField(disp_height=3.90).move_to([0, 1.85, 0])

        # 两个计数的标签与数值
        lbl_circ = cn_text("边界环流", 30, C_CYAN, 0.21).move_to([-2.30, -0.22, 0])
        lbl_area = cn_text("区域旋度和", 30, C_ORANGE, 0.21).move_to([2.30, -0.22, 0])
        num_circ = PixelNumber(0.0, 3, M_CYAN, 40, nchars=7).set(height=0.30).move_to([-2.30, -0.66, 0])
        num_area = PixelNumber(0.0, 3, M_ORANGE, 40, nchars=7).set(height=0.30).move_to([2.30, -0.66, 0])
        eq_sign = text_img("=", 46, C_TEXT, 0.32).move_to([0, -0.66, 0])
        eq_sign.set_opacity(0)

        # 中文讲述行（预建，靠 move_to 移出画面切换）
        NARR = [
            ("intro",  "闭合曲线 C 围住区域 D", C_TEXT),
            ("line",   "沿边界走一圈：累加环流 ∮Pdx+Qdy", C_CYAN),
            ("cells",  "切成小格：内部环流沿公共边抵消", C_RED),
            ("area",   "只剩外边界：区域内部旋度之和", C_ORANGE),
            ("equal",  "格林公式：边界积分 = 区域积分", C_TEXT),
            ("replay", "同一枚硬币的两面", C_TEXT),
        ]
        narr_pos = [0, -1.30, 0]
        narr_lbls = {}
        for key, s, col in NARR:
            lb = cn_text(s, 27, col, 0.23).move_to(narr_pos)
            lb.move_to([9, 9, 0])
            narr_lbls[key] = lb

        # 底部固定公式链
        f1 = mi(r"$C=\partial D\quad(\mathrm{positive\ orientation})$",
                "f1", C_FORMULA, 0.27, 29).move_to([0, -1.98, 0])
        f2 = mi(r"$\mathbf{F}=(P,Q),\quad \oint_C P\,dx+Q\,dy$",
                "f2", C_FORMULA, 0.31, 30).move_to([0, -2.60, 0])
        f3 = mi(r"$\frac{\partial Q}{\partial x}-\frac{\partial P}{\partial y}=(\nabla\times\mathbf{F})\cdot\mathbf{k}$",
                "f3", C_FORMULA, 0.40, 29).move_to([0, -3.22, 0])
        f4 = mi(r"$\iint_D\left(\frac{\partial Q}{\partial x}-\frac{\partial P}{\partial y}\right)dA$",
                "f4", C_FORMULA, 0.40, 29).move_to([0, -3.84, 0])
        f5 = mi(r"$\oint_C P\,dx+Q\,dy=\iint_D\left(\frac{\partial Q}{\partial x}-\frac{\partial P}{\partial y}\right)dA$",
                "f5", C_FORMULA, 0.36, 29).move_to([0, -4.48, 0])
        formulas = Group(f1, f2, f3, f4, f5)
        FORMULA_COLORS = [f1, f2, f3, f4, f5]
        phase_formula = {"intro": 0, "line": 1, "cells": 2, "area": 3,
                         "equal": 4, "replay": 4}

        self.add(leg, lbl_circ, lbl_area)
        dynamic = Group(title, cursor, plane, num_circ, num_area, eq_sign,
                        *narr_lbls.values(), formulas)
        self.add(dynamic)

        def apply(t):
            phase, prog = phase_info(t)
            plane.redraw(t)

            # 计数：环流随行走累加；旋度和随热图揭示累加
            if phase in ("intro", "line"):
                iv = 0 if phase == "intro" else int(prog * (len(TH) - 1))
                num_circ.set_value(float(CUM_LINE[iv]))
                num_area.set_value(0.0)
            elif phase == "cells":
                num_circ.set_value(TOTAL_LINE)
                num_area.set_value(0.0)
            elif phase == "area":
                num_circ.set_value(TOTAL_LINE)
                num_area.set_value(float(CUM_AREA[max(1, int(prog * len(CELLS)))]))
            else:  # equal / replay
                num_circ.set_value(TOTAL_LINE)
                num_area.set_value(TOTAL_LINE)

            # 等号：阶段④后半闪烁出现，回放结尾稳定
            if phase == "equal":
                eq_sign.set_opacity(0.45 + 0.55 * min(1.0, prog * 2.4))
            elif phase == "replay" and prog > 0.62:
                eq_sign.set_opacity(1.0)
            else:
                eq_sign.set_opacity(0.0)

            # 中文讲述行切换
            for k, lb in narr_lbls.items():
                lb.move_to(narr_pos if k == phase else [9, 9, 0])

            # 公式链：当前行高亮，其余微暗（位置始终固定）
            hi = phase_formula[phase]
            for i, fm in enumerate(FORMULA_COLORS):
                fm.set_opacity(1.0 if i == hi else 0.48)

            # 片头与回放切换处标题闪烁
            blink = (t < 0.28) or (17.72 <= t <= 18.06)
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
