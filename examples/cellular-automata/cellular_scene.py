#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Manim 数学动画：元胞自动机 —— 简单局部规则，复杂全局涌现

视觉规范与高斯积分 / 复指数映射 / 傅里叶级数 / 泰勒展开成片一致：
  - 1080x1350 竖屏、30fps、约 21.3s
  - 公式全程固定，只动参数、网格与动画
  - 无 LaTeX：公式用 matplotlib mathtext 预渲染为透明 PNG
  - 暗色底 #101113、中灰清晰线条、青/红边缘、纯内存动态数字

画面：
  顶部 = 一维初等元胞自动机时空图（规则 30 / 90 / 110 / 184 依次生长）
  中部 = 康威生命游戏高斯帕枪（持续发射滑翔机）
  数值行 = 世代 / 规则号 / 当前行活细胞数 / 生命游戏代数
  底部 = 5 行固定公式链

运行：
    manim -pql cellular_scene.py CellularAutomata
    manim -pqh cellular_scene.py CellularAutomata
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
    path = FORMULA_DIR / f"ca_{name}.png"
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


# ---------------------------------------------------------------------------
# 元胞自动机数学
# ---------------------------------------------------------------------------
RULES = [30, 90, 110, 184]
RULE_DESC = [
    "RULE 30  -  CHAOTIC",
    "RULE 90  -  FRACTAL",
    "RULE 110  -  UNIVERSAL",
    "RULE 184  -  TRAFFIC",
]
CA_ROWS, CA_COLS = 28, 61

LIFE_ROWS, LIFE_COLS = 18, 40
LIFE_GENS = 90

# 高斯帕滑翔机枪（标准 36x9 图案）
_GUN = [
    "........................O",
    "......................O.O",
    "............OO......OO............OO",
    "...........O...O....OO............OO",
    "OO........O.....O...OO",
    "OO........O...O.OO....O.O",
    "..........O.....O.......O",
    "...........O...O",
    "............OO",
]


def ca_step(row: np.ndarray, rule: int) -> np.ndarray:
    """一维初等 CA：固定边界（两侧补 0），三邻居编码 0..7 查规则位。"""
    padded = np.concatenate(([0], row.astype(np.int64), [0]))
    trip = 4 * padded[:-2] + 2 * padded[1:-1] + padded[2:]
    return ((rule >> trip) & 1).astype(bool)


def ca_space_time(rule: int) -> np.ndarray:
    grid = np.zeros((CA_ROWS, CA_COLS), dtype=bool)
    grid[0, CA_COLS // 2] = True
    for i in range(1, CA_ROWS):
        grid[i] = ca_step(grid[i - 1], rule)
    return grid


_CA_GRIDS = {r: ca_space_time(r) for r in RULES}


def life_step(grid: np.ndarray) -> np.ndarray:
    """康威生命游戏：固定边界，B3/S23。"""
    p = np.pad(grid.astype(np.uint8), 1, mode="constant")
    n = (p[:-2, :-2] + p[:-2, 1:-1] + p[:-2, 2:] +
         p[1:-1, :-2] + p[1:-1, 2:] +
         p[2:, :-2] + p[2:, 1:-1] + p[2:, 2:])
    return ((grid & ((n == 2) | (n == 3))) | (~grid & (n == 3)))


def life_frames() -> list[np.ndarray]:
    g = np.zeros((LIFE_ROWS, LIFE_COLS), dtype=bool)
    for r, line in enumerate(_GUN):
        for c, ch in enumerate(line):
            if ch == "O":
                g[4 + r, 1 + c] = True
    frames = []
    for _ in range(LIFE_GENS):
        frames.append(g.copy())
        g = life_step(g)
    return frames


_LIFE_FRAMES = life_frames()


# ---------------------------------------------------------------------------
# 内存位图：高效绘制元胞网格（青/红活细胞 + 中灰死细胞 + 细网格缝）
# ---------------------------------------------------------------------------
class GridBitmap(ImageMobject):
    def __init__(self, rows, cols, cell, alive_rgb, gap_rgb, dead_rgb,
                 head_rgb, disp_height):
        self.rows, self.cols, self.cell = rows, cols, cell
        self.alive = np.array(alive_rgb, dtype=np.uint8)
        self.dead = np.array(dead_rgb, dtype=np.uint8)
        self.gap = np.array(gap_rgb, dtype=np.uint8)
        self.head = np.array(head_rgb, dtype=np.uint8)
        core = np.zeros((cell, cell), dtype=bool)
        core[1:, 1:] = True                       # 每格左上留 1px 缝 → 网格感
        self._tile = np.tile(core, (rows, cols))
        arr = np.zeros((rows * cell, cols * cell, 4), dtype=np.uint8)
        arr[..., :3] = self.gap
        arr[self._tile, :3] = self.dead
        arr[..., 3] = 255
        super().__init__(arr)
        self.scale(disp_height / self.height)

    def update_grid(self, grid: np.ndarray, reveal_row: int = -1):
        big = np.repeat(np.repeat(grid, self.cell, axis=0), self.cell, axis=1) & self._tile
        arr = self.pixel_array
        arr[..., :3] = self.gap
        arr[self._tile, :3] = self.dead
        arr[big, :3] = self.alive
        if 0 <= reveal_row < self.rows:
            r = reveal_row
            sub_big = np.repeat(
                np.repeat(grid[r:r + 1], self.cell, axis=0), self.cell, axis=1
            ) & np.tile(self._core_local(), (1, self.cols))
            sub = arr[r * self.cell:(r + 1) * self.cell]
            sub[sub_big, :3] = self.head
        arr[..., 3] = 255

    def _core_local(self):
        core = np.zeros((self.cell, self.cell), dtype=bool)
        core[1:, 1:] = True
        return core


def clamp01(v):
    return float(np.clip(v, 0.0, 1.0))


def ease(v):
    v = clamp01(v)
    return v * v * (3 - 2 * v)


# ---------------------------------------------------------------------------
# 场景
# ---------------------------------------------------------------------------
class CellularAutomata(Scene):
    def construct(self):
        # ------------------------------------------------------------------
        # 标题
        # ------------------------------------------------------------------
        title = text_img("AIMaker-Lab", 52, C_TEXT, 0.28).move_to([0, 4.58, 0])
        cursor = Line(UP * 0.16, DOWN * 0.16, color=M_TEXT,
                      stroke_width=2.5).move_to([0, 4.58, 0])
        cursor.set_opacity(0)

        # ------------------------------------------------------------------
        # 顶部：一维 CA 时空图
        # ------------------------------------------------------------------
        ca = GridBitmap(CA_ROWS, CA_COLS, 11,
                        C_CYAN, (16, 17, 19), (34, 40, 44), C_ORANGE,
                        disp_height=3.03)
        ca.move_to([0, 2.40, 0])
        ca_frame = Rectangle(width=7.15, height=3.35, color=M_GRID_DIM,
                             stroke_width=1.6).move_to([0, 2.40, 0])
        rule_lbls = [text_img(d, 40, C_CYAN, 0.20) for d in RULE_DESC]
        for i, lb in enumerate(rule_lbls):
            lb.move_to([0, 4.18, 0]).set_opacity(0)
        lbl_axis = text_img("generations  ↓", 30, C_GRID, 0.145).move_to(
            [-3.55, 2.40, 0]).rotate(PI / 2)

        # ------------------------------------------------------------------
        # 中部：康威生命游戏高斯帕枪
        # ------------------------------------------------------------------
        life = GridBitmap(LIFE_ROWS, LIFE_COLS, 10,
                          C_RED, (16, 17, 19), (34, 40, 44), C_ORANGE,
                          disp_height=1.70)
        life.move_to([0, -0.87, 0])
        life_frame = Rectangle(width=4.15, height=1.94, color=M_GRID_DIM,
                               stroke_width=1.6).move_to([0, -0.87, 0])
        lbl_life = text_img("CONWAY'S LIFE · B3/S23", 34, C_RED, 0.16).move_to(
            [0, 0.16, 0])
        lbl_gun = text_img("GOSPER GLIDER GUN", 28, C_GRID, 0.135).move_to(
            [0, -1.90, 0])

        # ------------------------------------------------------------------
        # 动态数值行
        # ------------------------------------------------------------------
        def num_row(expr, name, color, value, lab_h=0.16, places=0, px=28):
            lab = mi(expr, name, color, lab_h, 25)
            val = PixelNumber(value, places, color, px).set(height=0.20)
            return Group(lab, val).arrange(RIGHT, buff=0.05)

        row_t = num_row(r"$t=$", "lab_t", M_CYAN, 0, 0.16, 0)
        row_r = num_row(r"$rule=$", "lab_r", M_CYAN, 30, 0.16, 0)
        row_n = num_row(r"$N_{live}=$", "lab_n", M_ORANGE, 1, 0.16, 0)
        row_g = num_row(r"$gen=$", "lab_g", M_RED, 0, 0.16, 0)
        row_t.move_to([-2.85, 0.56, 0])
        row_r.move_to([-0.95, 0.56, 0])
        row_n.move_to([0.95, 0.56, 0])
        row_g.move_to([2.85, 0.56, 0])
        num_t, num_r, num_n, num_g = row_t[1], row_r[1], row_n[1], row_g[1]

        # ------------------------------------------------------------------
        # 底部固定公式链（按 PNG 原始高度反比设定显示高度，统一字号）
        # ------------------------------------------------------------------
        # 主体字形大小统一：含上下标的公式主体占比小，显示高度需相应放大
        f1 = mi(r"$s_i^{(t+1)}=f\left(s_{i-1}^{(t)},s_i^{(t)},s_{i+1}^{(t)}\right)$",
                "f1", C_FORMULA, 0.46, 30).move_to([0, -2.30, 0])
        f2 = mi(r"$r=\sum_{a,b,c\in\{0,1\}}f(a,b,c)\,2^{4a+2b+c}$",
                "f2", C_FORMULA, 0.66, 30).move_to([0, -2.95, 0])
        f3 = mi(r"$B3/S23:\ n=3\Rightarrow\mathrm{birth},\ n=2,3\Rightarrow\mathrm{survive}$",
                "f3", C_FORMULA, 0.30, 27).move_to([0, -3.55, 0])
        f4 = mi(r"$\mathrm{Rule\ 110}:\ \mathrm{Turing\ complete}\Rightarrow\mathrm{universal\ computation}$",
                "f4", C_FORMULA, 0.27, 25).move_to([0, -4.08, 0])
        f5 = mi(r"$\mathrm{emergence}:\ \mathrm{local\ rule}\longrightarrow\mathrm{global\ complexity}$",
                "f5", C_FORMULA, 0.30, 26).move_to([0, -4.60, 0])
        formulas = Group(f1, f2, f3, f4, f5)

        # ------------------------------------------------------------------
        # 图层
        # ------------------------------------------------------------------
        self.add(ca_frame, life_frame, lbl_axis, lbl_life, lbl_gun, formulas)
        dynamic = Group(
            title, cursor,
            ca, *rule_lbls,
            life,
            row_t, row_r, row_n, row_g,
        )
        self.add(dynamic)

        # ------------------------------------------------------------------
        # 逐帧驱动
        # ------------------------------------------------------------------
        prev_seg = [-1]

        def apply(t):
            if t < 16.15:
                u = clamp01((t - 1.5) / 14.0)
            else:
                u = clamp01((t - 16.15) / 4.6)

            p = u * 4.0
            seg = min(3, int(p))
            local = p - seg
            rule = RULES[seg]
            reveal = max(1, int(local * CA_ROWS))

            grid = _CA_GRIDS[rule].copy()
            grid[reveal:] = False
            ca.update_grid(grid, reveal_row=reveal - 1)

            lv = _LIFE_FRAMES[min(LIFE_GENS - 1, int(u * LIFE_GENS))]
            life.update_grid(lv)

            for i, lb in enumerate(rule_lbls):
                lb.set_opacity(1.0 if i == seg else 0.0)

            num_t.set_value(reveal)
            num_r.set_value(rule)
            num_n.set_value(int(grid[reveal - 1].sum()))
            num_g.set_value(int(u * LIFE_GENS))

            # 规则切换瞬间：标题闪烁（呼应前面成片的循环"呼吸"）
            blank = (seg != prev_seg[0])
            if blank:
                prev_seg[0] = seg
            blink = (local < 0.022) or (16.02 <= t <= 16.36)
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
