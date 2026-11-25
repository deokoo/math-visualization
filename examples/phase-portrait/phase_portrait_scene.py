#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Manim 数学动画：微分方程相图 —— 特征值如何决定平衡点的类型

视觉规范与前作一致：
  - 1080x1350 竖屏、30fps、约 21.3s
  - 公式全程固定，只动参数 / 网格 / 动画
  - 无 LaTeX：公式用 matplotlib mathtext 预渲染为透明 PNG
  - 暗色底 #101113、中灰清晰线条、青/红边缘曲线、纯内存动态数字

画面：
  中央 = 相平面：向量场（灰青小箭头）+ 12 条解轨迹（青色，沿特征向量的两条红色）
         红点 = 平衡点 x=0；参数矩阵 A(t) 在 4 种平衡点类型间连续演化：
         鞍点 SADDLE → 稳定结点 STABLE NODE → 稳定焦点 STABLE SPIRAL → 中心 CENTER
  左侧 = 2x2 参数矩阵 A(t) 的 a,b,c,d 实时数值
  右侧 = 当前平衡点类型 + 稳定性
  数值行 = τ=tr A / Δ=det A / τ²−4Δ / λ₁,₂（判别式<0 时显示共轭复根）
  底部 = 5 行固定公式链

运行：
    manim -pql phase_portrait_scene.py PhasePortrait
    manim -pqh phase_portrait_scene.py PhasePortrait
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
    path = FORMULA_DIR / f"pp_{name}.png"
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
    """中文标题：微软雅黑渲染。"""
    img = Image.new("RGBA", (1600, 220), (0, 0, 0, 0))
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
# 相平面位图：向量场 + 解轨迹（numpy 积分 + PIL 整块绘制，避免上千图元）
# ---------------------------------------------------------------------------
RES = 480          # 贴近成片中相平面的实际像素高度，避免缩小时细线锯齿糊成一片
WORLD = 2.08
BG_RGB = (16, 17, 19)


def w2p(x, y):
    """世界坐标 → 像素坐标（PIL y 轴向下）"""
    return ((x + WORLD) / (2 * WORLD) * RES,
            (WORLD - y) / (2 * WORLD) * RES)


def integrate(A: np.ndarray, p0: np.ndarray, T=3.2, dt=0.06, sign=1):
    """向量化 RK4：一次推进 M 条初值轨迹，越界者冻结。返回 (K,2,M)。"""
    P = np.asarray(p0, dtype=float).copy()  # 已是 (2, M)，不要再转置
    M = sign * A
    hist = [P.copy()]
    alive = np.ones(P.shape[1], dtype=bool)
    lim = WORLD - 0.18
    for _ in range(int(T / dt)):
        k1 = M @ P
        k2 = M @ (P + 0.5 * dt * k1)
        k3 = M @ (P + 0.5 * dt * k2)
        k4 = M @ (P + dt * k3)
        Pn = P + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        rad = np.linalg.norm(Pn, axis=0)
        new_alive = alive & np.isfinite(rad) & (rad <= lim)
        P[:, new_alive] = Pn[:, new_alive]
        alive = new_alive
        hist.append(P.copy())
        if not alive.any():
            break
    return np.stack(hist, axis=0)


def init_points(A: np.ndarray) -> np.ndarray:
    """圆上均匀初值；若特征值为实，则前两条沿特征向量（红色高亮）。"""
    ang = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    P = 1.32 * np.vstack([np.cos(ang), np.sin(ang)])
    try:
        w, V = np.linalg.eig(A)
        if np.all(np.isreal(w)):
            v = np.real(V[:, 0])
            nrm = np.linalg.norm(v)
            if nrm > 1e-9:
                v = v / nrm
                P[:, 0] = 1.32 * v
                P[:, 1] = -1.32 * v
    except Exception:
        pass
    return P


class PhaseField(ImageMobject):
    def __init__(self, disp_height=4.0):
        arr = np.zeros((RES, RES, 4), dtype=np.uint8)
        arr[..., :3] = BG_RGB
        arr[..., 3] = 255
        super().__init__(arr)
        self.scale(disp_height / self.height)

    def redraw(self, A: np.ndarray):
        # 必须先清空为纯背景，否则每帧都叠在上一帧上，几百帧轨迹会累积成实心色块
        arr = self.pixel_array
        arr[..., :3] = BG_RGB
        arr[..., 3] = 255
        img = Image.fromarray(arr, mode="RGBA")
        d = ImageDraw.Draw(img)

        # 网格
        g = -2.0
        while g <= 2.0001:
            gx, _ = w2p(g, 0)
            _, gy = w2p(0, g)
            d.line([(gx, 0), (gx, RES)], fill=(*C_GRID_DIM, 255), width=1)
            d.line([(0, gy), (RES, gy)], fill=(*C_GRID_DIM, 255), width=1)
            g += 0.5

        # 解轨迹（先画，位于向量场与坐标轴之下；只取少量初值与较短正/反向时间，
        # 避免轨迹铺满画面形成实心圆遮挡向量场）
        P0 = init_points(A)
        fwd = integrate(A, P0, T=2.6, dt=0.06, sign=+1)
        bwd = integrate(A, P0, T=0.9, dt=0.06, sign=-1)
        M = P0.shape[1]
        for i in range(M):
            curve = np.vstack([bwd[::-1, :, i], fwd[:, :, i]])
            pts = [w2p(curve[j, 0], curve[j, 1]) for j in range(curve.shape[0])]
            if i < 2:
                col, wdt = (206, 84, 74, 255), 2   # 关键轨迹（红）
            else:
                col, wdt = (54, 165, 158, 255), 2   # 一般解轨迹（青）
            d.line(pts, fill=col, width=wdt)

        # 向量场箭头（9x9）——画在轨迹之上，保证始终清晰
        gx = np.arange(-2.0, 2.01, 0.5)
        X, Y = np.meshgrid(gx, gx)
        G = np.vstack([X.ravel(), Y.ravel()])
        V = A @ G
        nrm = np.linalg.norm(V, axis=0)
        scale = RES / (2 * WORLD)
        for k in range(G.shape[1]):
            if nrm[k] < 1e-4:
                continue
            u = V[:, k] / nrm[k]
            p = G[:, k]
            s, e = p - 0.11 * u, p + 0.11 * u
            ps, pe = w2p(s[0], s[1]), w2p(e[0], e[1])
            col = (96, 158, 154, 255)
            d.line([ps, pe], fill=col, width=2)
            ang = math.atan2(pe[1] - ps[1], pe[0] - ps[0])
            hlen = 0.055 * scale
            for da in (math.pi - 0.5, math.pi + 0.5):
                d.line([pe, (pe[0] + hlen * math.cos(ang + da),
                             pe[1] + hlen * math.sin(ang + da))], fill=col, width=2)

        # 坐标轴（最上层）
        cx, cy = w2p(0, 0)
        d.line([(cx, 0), (cx, RES)], fill=(*C_GRID, 255), width=2)
        d.line([(0, cy), (RES, cy)], fill=(*C_GRID, 255), width=2)

        self.pixel_array[:] = np.asarray(img)


# ---------------------------------------------------------------------------
# 参数演化：4 种平衡点类型
# ---------------------------------------------------------------------------
STAGES = [
    ("SADDLE",        np.array([[0.90,  0.00], [0.00, -0.90]]), C_RED),
    ("NODE",          np.array([[-1.40, 0.00], [0.00, -0.60]]), C_CYAN),
    ("SPIRAL",        np.array([[-0.28, -1.00], [1.00, -0.28]]), C_ORANGE),
    ("CENTER",        np.array([[0.00, -1.00], [1.00,  0.00]]), C_CYAN),
]
LBL_COLOR = {name: col for name, _, col in STAGES}
LBL_COLOR["DEGENERATE"] = C_GRID


def smoothstep(v):
    v = float(np.clip(v, 0.0, 1.0))
    return v * v * (3 - 2 * v)


def system_at(t: float) -> np.ndarray:
    """两段：1.5~16.15s 正演；16.15~21.26s 快速重演。"""
    if t < 1.5:
        return STAGES[0][1].copy()
    if t <= 16.15:
        prog = (t - 1.5) / (16.15 - 1.5)
    else:
        prog = (t - 16.15) / (DURATION - 16.15)
    x = prog * (len(STAGES) - 1)
    i = min(int(x), len(STAGES) - 2)
    f = smoothstep(x - i)
    return (1 - f) * STAGES[i][1] + f * STAGES[i + 1][1]


def classify(A: np.ndarray):
    a, b, c, d = A.ravel()
    tau, det = a + d, a * d - b * c
    disc = tau * tau - 4 * det
    if det < -1e-6:
        name, stab = "SADDLE", "UNSTABLE"
    elif abs(tau) < 2e-3 and det > 0:
        name, stab = "CENTER", "MARGINAL"
    elif disc < -1e-6:
        name, stab = "SPIRAL", ("STABLE" if tau < 0 else "UNSTABLE")
    elif det > 1e-6:
        name, stab = "NODE", ("STABLE" if tau < 0 else "UNSTABLE")
    else:
        name, stab = "DEGENERATE", "BIFURCATION"
    # 特征值显示串
    if disc >= 0:
        s = math.sqrt(max(0.0, disc))
        ls = f"{(tau + s) / 2:+.2f},{(tau - s) / 2:+.2f}"
    else:
        re, im = tau / 2, math.sqrt(-disc) / 2
        ls = f"{re:+.2f}±{im:.2f}i"
    return name, stab, tau, det, disc, ls


def clamp01(v):
    return float(np.clip(v, 0.0, 1.0))


# ---------------------------------------------------------------------------
# 场景
# ---------------------------------------------------------------------------
class PhasePortrait(Scene):
    def construct(self):
        # 标题：中文主题名 + 英文数学标题（新约定：每部动画写明算法/数学标题）
        cn_title = cn_text("微分方程相图", 50, C_TEXT, 0.30).move_to([0, 4.68, 0])
        en_title = text_img("P H A S E   P O R T R A I T   /   L I N E A R   S Y S T E M S",
                            23, C_CYAN, 0.11).move_to([0, 4.37, 0])
        watermark = text_img("AIMaker-Lab", 17, C_GRID, 0.09).move_to([3.05, 4.68, 0])
        title = Group(cn_title, en_title, watermark)
        cursor = Line(UP * 0.16, DOWN * 0.16, color=M_TEXT, stroke_width=2.5).move_to([0, 4.68, 0])
        cursor.set_opacity(0)

        # 图例
        leg = Group(
            Group(Line(LEFT * 0.13, RIGHT * 0.13, color=M_CYAN, stroke_width=3),
                  text_img("SOLUTION", 26, C_CYAN, 0.14)).arrange(RIGHT, buff=0.06),
            Group(Line(LEFT * 0.13, RIGHT * 0.13, color=M_RED, stroke_width=3),
                  text_img("HIGHLIGHT", 26, C_RED, 0.14)).arrange(RIGHT, buff=0.06),
            Group(Line(LEFT * 0.13, RIGHT * 0.13, color=M_GRID, stroke_width=3),
                  text_img("VECTOR FIELD", 26, C_GRID, 0.14)).arrange(RIGHT, buff=0.06),
        ).arrange(RIGHT, buff=0.22).move_to([0, 4.02, 0])

        # --- 中央相平面 ---
        plane = PhaseField(disp_height=4.0).move_to([0, 1.85, 0])
        eq_dot = Dot(color=M_RED, radius=0.060).move_to([0, 1.85, 0.01])
        pulse = Circle(radius=0.13, color=M_RED, stroke_width=2.4)
        pulse.set_fill(opacity=0).move_to([0, 1.85, 0.01])

        # --- 左侧参数矩阵 A(t) ---
        mat_lbl = text_img("A(t)=", 28, C_TEXT, 0.18).move_to([-2.98, 3.02, 0])
        bracket = Group(
            Line([-3.46, 1.53, 0], [-3.46, 2.83, 0], color=M_GRID, stroke_width=2),
            Line([-3.46, 2.83, 0], [-3.32, 2.83, 0], color=M_GRID, stroke_width=2),
            Line([-3.46, 1.53, 0], [-3.32, 1.53, 0], color=M_GRID, stroke_width=2),
            Line([-2.20, 1.53, 0], [-2.20, 2.83, 0], color=M_GRID, stroke_width=2),
            Line([-2.20, 2.83, 0], [-2.34, 2.83, 0], color=M_GRID, stroke_width=2),
            Line([-2.20, 1.53, 0], [-2.34, 1.53, 0], color=M_GRID, stroke_width=2),
        )
        mat_nums = []
        for k, (xx, yy) in enumerate([(-3.22, 2.45), (-2.46, 2.45),
                                      (-3.22, 1.90), (-2.46, 1.90)]):
            pn = PixelNumber(0.0, 2, M_TEXT, 30).set(height=0.19).move_to([xx, yy, 0])
            mat_nums.append(pn)

        # --- 右侧类型标签（预建，靠 move_to 移出画面切换） ---
        type_pos = [3.02, 2.55, 0]
        type_lbls = {}
        for nm, col in LBL_COLOR.items():
            lb = text_img(nm, 27, col, 0.17).move_to(type_pos)
            lb.move_to([9, 9, 0])
            type_lbls[nm] = lb
        stab_pos = [3.02, 2.05, 0]
        stab_lbls = {}
        for nm, col in [("STABLE", C_CYAN), ("UNSTABLE", C_RED), ("MARGINAL", C_ORANGE),
                        ("BIFURCATION", C_GRID)]:
            lb = text_img(nm, 22, col, 0.13).move_to(stab_pos)
            lb.move_to([9, 9, 0])
            stab_lbls[nm] = lb

        # --- 数值行 ---
        def num_row(expr, name, color, value, lab_h=0.16, places=2, px=28, nchars=None):
            lab = mi(expr, name, color, lab_h, 24)
            val = PixelNumber(value, places, color, px, nchars=nchars).set(height=0.19)
            return Group(lab, val).arrange(RIGHT, buff=0.05)

        row_tau = num_row(r"$\tau=$", "lab_tau", M_CYAN, 0.0)
        row_det = num_row(r"$\Delta=$", "lab_det", M_ORANGE, 0.0)
        row_dis = num_row(r"$\tau^2-4\Delta=$", "lab_dis", M_RED, 0.0, 0.16, 2, 27)
        row_lam = num_row(r"$\lambda_{1,2}=$", "lab_lam", M_TEXT,
                          "+0.00,-0.00", 0.15, 2, 22, 14)
        for row, x in zip((row_tau, row_det, row_dis, row_lam),
                          (-2.85, -0.95, 0.90, 2.85)):
            row.move_to([x, -0.62, 0])
        num_tau, num_det, num_dis, num_lam = (r[1] for r in
                                               (row_tau, row_det, row_dis, row_lam))
        num_row_grp = Group(row_tau, row_det, row_dis, row_lam)

        # --- 底部固定公式链 ---
        f1 = mi(r"$\dot{x}=ax+by,\quad \dot{y}=cx+dy$",
                "f1", C_FORMULA, 0.29, 31).move_to([0, -1.50, 0])
        f2 = mi(r"$\tau=a+d,\quad \Delta=ad-bc$",
                "f2", C_FORMULA, 0.29, 31).move_to([0, -2.18, 0])
        f3 = mi(r"$\det(A-\lambda I)=\lambda^2-\tau\lambda+\Delta=0$",
                "f3", C_FORMULA, 0.30, 30).move_to([0, -2.86, 0])
        # \frac 使 PNG 主体被撑高，按原始高度反比放大显示高度以统一字号
        f4 = mi(r"$\lambda_{1,2}=\frac{\tau\pm\sqrt{\tau^2-4\Delta}}{2}$",
                "f4", C_FORMULA, 0.46, 31).move_to([0, -3.48, 0])
        f5 = mi(r"$\Delta<0:\ \mathrm{saddle}\quad \tau^2<4\Delta:\ \mathrm{focus/center}$",
                "f5", C_FORMULA, 0.26, 27).move_to([0, -4.25, 0])
        formulas = Group(f1, f2, f3, f4, f5)

        # 静态图层
        self.add(formulas, leg, bracket, mat_lbl)

        # 动态图层（必须全部放进 Timeline 持有的 Group，否则被烘焙成静态层不更新）
        dynamic = Group(title, cursor, plane, eq_dot, pulse,
                        *mat_nums, *type_lbls.values(), *stab_lbls.values(),
                        num_row_grp)
        self.add(dynamic)

        last_A = None

        def apply(t):
            nonlocal last_A
            A = system_at(t)
            nm, stab, tau, det, disc, lam = classify(A)

            # 相平面每帧重绘（向量场 + 解轨迹）
            if last_A is None or not np.allclose(A, last_A, atol=1e-6):
                plane.redraw(A)
                last_A = A.copy()

            # 参数矩阵
            vals = [A[0, 0], A[0, 1], A[1, 0], A[1, 1]]
            for pn, v in zip(mat_nums, vals):
                pn.set_value(v)

            # 类型标签切换
            for k, lb in type_lbls.items():
                lb.move_to(type_pos if k == nm else [9, 9, 0])
            for k, lb in stab_lbls.items():
                lb.move_to(stab_pos if k == stab else [9, 9, 0])

            # 数值行
            num_tau.set_value(tau)
            num_det.set_value(det)
            num_dis.set_value(disc)
            num_lam.set_value(lam)

            # 平衡点脉动
            ph = (t * 1.4) % 1.0
            pulse.set_stroke(M_RED, width=2.4, opacity=0.6 * (1 - ph))
            pulse.scale_to_fit_width(0.24 + 0.16 * ph)

            # 循环切换点闪烁
            blink = (t < 0.28) or (16.02 <= t <= 16.36)
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
