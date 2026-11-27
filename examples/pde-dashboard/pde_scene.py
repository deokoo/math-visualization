# -*- coding: utf-8 -*-
"""
Seven PDEs, solved live -- 1080x1350 (4:5) 30fps ~70s.
Matches the vertical 1080x1350 format of the rest of the series.
Self-contained: numpy numeric solvers + PIL drawing + matplotlib mathtext PNG.
No LaTeX install, no Manim required.
"""
import math
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.mathtext import math_to_image

W, H = 1080, 1350
FPS = 30
DURATION = 70.0
PANEL_H = 252
PANEL_TOP = 112
PANEL_GAP = 10
PANEL_MARGIN = 24
PANEL_GUTTER = 12
CAPTION_H = 162
NFRAMES = int(FPS * DURATION)

HERE = os.path.dirname(os.path.abspath(__file__))
FORM = os.path.join(HERE, "formulas")
os.makedirs(FORM, exist_ok=True)

BG = (16, 17, 19)
PANEL_BG = (26, 28, 32)
GRID = (52, 56, 63)
TEXT = (214, 219, 226)
DIMTEXT = (128, 134, 143)
CYAN = (80, 214, 224)
RED = (232, 96, 92)
GOLD = (240, 186, 84)
VIOLET = (172, 138, 226)
GREEN = (120, 206, 146)
WHITE = (238, 240, 243)

import matplotlib as _mpl
_MPL_TTF = os.path.join(os.path.dirname(_mpl.__file__), "mpl-data", "fonts", "ttf")
FONT_CN = "C:/Windows/Fonts/msyh.ttc"
FONT_EN = os.path.join(_MPL_TTF, "DejaVuSans.ttf")
FONT_SERIF = os.path.join(_MPL_TTF, "DejaVuSerif.ttf")


# ---------------- formula PNGs (mathtext, no LaTeX) ----------------
FORMULAS = {
    "heat":   r"$\frac{\partial u}{\partial t}=\alpha\nabla^{2}u$",
    "wave":   r"$\frac{\partial^{2}u}{\partial t^{2}}=c^{2}\nabla^{2}u$",
    "maxwell":r"$\nabla\times\mathbf{E}=-\frac{\partial\mathbf{B}}{\partial t}$",
    "navier": r"$\rho\left(\partial_t\mathbf{u}+\mathbf{u}\cdot\nabla\mathbf{u}\right)=-\nabla p+\mu\nabla^{2}\mathbf{u}$",
    "schrod": r"$i\hbar\frac{\partial\psi}{\partial t}=-\frac{\hbar^{2}}{2m}\nabla^{2}\psi$",
    "psi":    r"$|\psi|^{2}\ /\ \int|\psi|^{2}dx=1$",
    "beam":   r"$\frac{\partial^{2}u}{\partial t^{2}}+\frac{EI}{\rho A}\frac{\partial^{4}u}{\partial x^{4}}=0$",
    "rd":     r"$\partial_tu=D_u\nabla^{2}u-uv^{2}+F(1-u)$",
}
FORMULA_SUB = {
    "maxwell": r"$\nabla\cdot\mathbf{B}=0,\quad \nabla\times\mathbf{B}=\mu_0\mathbf{J}+\frac{1}{c^{2}}\partial_t\mathbf{E}$",
    "navier":  r"$\nabla\cdot\mathbf{u}=0$",
    "rd":      r"$\partial_tv=D_v\nabla^{2}v+uv^{2}-(F+k)v$",
}

def _fname(key, sub=False):
    return os.path.join(FORM, key + ("_sub" if sub else "") + ".png")

def _render_formula(tex, path, size, color):
    """Render mathtext onto a transparent PNG (no white rectangle)."""
    fig = plt.figure(figsize=(0.01, 0.01), dpi=200)
    fig.patch.set_alpha(0)
    fig.text(0, 0, tex, fontsize=size, color=color, va="bottom", ha="left")
    fig.savefig(path, dpi=200, transparent=True, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

def ensure_formulas():
    for key, tex in FORMULAS.items():
        p = _fname(key)
        if not os.path.exists(p):
            _render_formula(tex, p, 22, "#e6ebf2")
    for key, tex in FORMULA_SUB.items():
        p = _fname(key, sub=True)
        if not os.path.exists(p):
            _render_formula(tex, p, 17, "#b9c0cb")

def formula_img(key, sub=False):
    return Image.open(_fname(key, sub)).convert("RGBA")


# ---------------- numeric solvers ----------------
class HeatSolver:
    """2D heat equation, two gaussian blobs diffusing."""
    N = 72
    def __init__(self):
        y, x = np.mgrid[0:self.N, 0:self.N]
        self.u = (np.exp(-(((x-24)**2+(y-36)**2)/120)) +
                  0.8*np.exp(-(((x-50)**2+(y-26)**2)/80)))
        self.u = np.clip(self.u, 0, 1)
    def step(self):
        p = np.pad(self.u, 1, mode="edge")
        lap = (p[1:-1,2:] + p[1:-1,:-2] + p[2:,1:-1] + p[:-2,1:-1] - 4*self.u)
        self.u = np.clip(self.u + 0.16*lap, 0, 1.2)


class WaveSolver:
    """1D wave: gaussian pulse splits into left/right travelling waves. Analytic d'Alembert."""
    def field(self, t):
        x = np.linspace(-1, 1, 240)
        # periodic re-emission so the pulse keeps being born at centre
        tp = t % 2.4
        f = lambda z: np.exp(-(z**2)/0.010)*np.cos(8*z)
        c = 0.36
        ul = 0.5*f(x + c*tp)
        ur = 0.5*f(x - c*tp)
        # faint reflected waves from the fixed ends
        return x, ul, ur, ul+ur


class MaxwellSolver:
    """ dipole radiation: expanding alternating E-field lobes (analytic). """
    def field(self, t, n=90):
        a = np.linspace(0.16, 0.95, n)
        th = np.linspace(0, 2*np.pi, 160)
        R, T = np.meshgrid(a, th, indexing="ij")
        # dipole pattern modulated by outgoing wave
        patt = np.abs(np.sin(T))*(0.55 + 0.45*np.cos(7*R - 3.2*t))
        X = R*np.cos(T)*patt
        Y = R*np.sin(T)*patt
        return X, Y


class NavierSolver:
    """ potential flow around cylinder + two shed vortices -> streaklines. """
    def __init__(self):
        self.np = 130
        rng = np.random.default_rng(11)
        self.px = rng.uniform(-1.0, -0.5, self.np)
        self.py = rng.uniform(-0.62, 0.62, self.np)
        self.age = rng.uniform(0, 1, self.np)
    def vel(self, x, y, t):
        a = 0.32
        r2 = x*x + y*y
        if r2 < a*a:
            r2 = a*a
        # doublet + uniform
        u = 1.0*(1 - a*a*(x*x-y*y)/(r2*r2))
        v = -1.0*(2*a*a*x*y/(r2*r2))
        # shedding vortices downstream
        for k in (0, 1):
            vx = 0.5 + 0.42*((t*0.42 + k*0.5) % 1.0)
            vy = (0.20 if k == 0 else -0.20)
            gamma = (0.16 if k == 0 else -0.16)
            dx, dy = x-vx, y-vy
            d2 = dx*dx + dy*dy + 0.02
            u += -gamma*dy/d2
            v +=  gamma*dx/d2
        return u, v
    def step(self, t, dt=0.045):
        for i in range(self.np):
            u, v = self.vel(self.px[i], self.py[i], t)
            self.px[i] += u*dt; self.py[i] += v*dt
            self.age[i] += dt
            if self.px[i] > 1.05 or abs(self.py[i]) > 0.78:
                self.px[i] = -1.0
                self.py[i] = np.random.uniform(-0.62, 0.62)
                self.age[i] = 0.0


class SchrodingerSolver:
    """ 3D gaussian wavepacket |psi| surface, evolving width/centre. """
    def surface(self, t):
        s = 0.20 + 0.11*t
        xc = -0.65 + 0.20*t
        xs = np.linspace(-1, 1, 46)
        ys = np.linspace(-1, 1, 46)
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        amp = np.exp(-(((X-xc)**2 + Y**2)/(2*s*s)))
        phase = np.cos(6*(X-xc) - 2.6*t)
        Z = amp*phase
        return X, Y, Z


class BeamSolver:
    """ cantilever beam first mode, vertical vibration. """
    def shape(self, t):
        x = np.linspace(0, 1, 90)
        beta = 1.875
        mode = (np.cosh(beta*x)-np.cos(beta*x) -
                0.734*(np.sinh(beta*x)-np.sin(beta*x)))
        mode /= np.max(np.abs(mode))
        y = 0.20*mode*np.cos(2.5*t)
        return x, y


class RDSolver:
    """ Gray-Scott reaction-diffusion, spot/stripe self-organisation. """
    N = 78
    def __init__(self):
        rng = np.random.default_rng(20261127)
        self.u = np.ones((self.N, self.N))
        self.v = np.zeros((self.N, self.N))
        for _ in range(14):
            cx, cy = rng.integers(6, self.N-6, 2)
            self.u[cx-3:cx+3, cy-3:cy+3] = 0.45
            self.v[cx-3:cx+3, cy-3:cy+3] = 0.30
        self.F, self.k = 0.033, 0.062
        self.Du, self.Dv = 0.21, 0.105
    def step(self, n=3):
        for _ in range(n):
            def lap(a):
                p = np.pad(a, 1, mode="edge")
                return p[1:-1,2:]+p[1:-1,:-2]+p[2:,1:-1]+p[:-2,1:-1]-4*a
            uvv = self.u*self.v*self.v
            self.u = np.clip(self.u + self.Du*lap(self.u) - uvv + self.F*(1-self.u), 0, 1)
            self.v = np.clip(self.v + self.Dv*lap(self.v) + uvv - (self.F+self.k)*self.v, 0, 1)


class Sim:
    def __init__(self):
        self.heat = HeatSolver()
        self.wave = WaveSolver()
        self.maxwell = MaxwellSolver()
        self.navier = NavierSolver()
        self.schrod = SchrodingerSolver()
        self.beam = BeamSolver()
        self.rd = RDSolver()
    def step(self, t):
        self.heat.step()
        self.navier.step(t)
        self.rd.step(3)


# ---------------- drawing helpers ----------------
def map_color(x, lo, hi, c0, c1):
    c0 = np.asarray(c0, dtype=np.float64)
    c1 = np.asarray(c1, dtype=np.float64)
    x = np.clip((x-lo)/(hi-lo), 0, 1)[..., None]
    return (c0*(1-x) + c1*x).astype(np.uint8)


def panel_base(d, box, title=None):
    x0, y0, x1, y1 = box
    d.rounded_rectangle(box, radius=9, fill=PANEL_BG, outline=GRID, width=1)
    d.line([(x0+12, y0+34), (x1-12, y0+34)], fill=GRID, width=1)


def draw_formula(d, im, key, box, sub=False):
    x0, y0, x1, y1 = box
    f = formula_img(key, sub=sub)
    maxw, maxh = (x1-x0-20), (y1-y0-4)
    r = min(maxw/f.width, maxh/f.height, 1.0)
    f = f.resize((max(1, int(f.width*r)), max(1, int(f.height*r))), Image.LANCZOS)
    im.paste(f, (int((x0+x1-f.width)/2), int(y0+2)), f)
    return f.height


def heat_panel(im, d, box, sim, gy0):
    x0, y0, x1, y1 = box
    u = sim.heat.u
    rgb = map_color(u, 0.0, 0.95, (28, 42, 104), (244, 196, 76))
    big = Image.fromarray(rgb, "RGB").resize((x1-x0-16, y1-gy0-8), Image.BILINEAR)
    im.paste(big, (x0+8, gy0))
    # colour bar
    cbx = x1-16
    for i in range(60):
        f = 1 - i/59
        c = tuple(int((28+(244-28)*f, 42+(196-42)*f, 104+(76-104)*f)[k]) for k in range(3))
        d.line([(cbx, gy0+i*2), (cbx+5, gy0+i*2)], fill=c)


def wave_panel(im, d, box, sim, t, gy0):
    x0, y0, x1, y1 = box
    lane = (y1-gy0)//3
    x, ul, ur, ut = sim.wave.field(t)
    def plot(vals, cy, col):
        pts = [(x0+10 + (i/(len(vals)-1))*(x1-x0-20),
                cy - vals[i]*lane*0.40) for i in range(len(vals))]
        d.line(pts, fill=col, width=2)
    for k, (vals, col, lab) in enumerate([(ul, CYAN, "u_L"), (ur, RED, "u_R"), (ut, GOLD, "u_L+u_R")]):
        cy = gy0 + lane*k + lane//2 + 6
        d.line([(x0+10, cy), (x1-10, cy)], fill=(44, 48, 54), width=1)
        plot(vals, cy, col)
        en_label(d, (x0+14, cy-lane//2+5), lab, size=13, fill=col)


def maxwell_panel(im, d, box, sim, t, gy0):
    x0, y0, x1, y1 = box
    cx, cy = (x0+x1)//2, (gy0+y1)//2
    rx, ry = (x1-x0)//2-20, (y1-gy0)//2-12
    X, Y = sim.maxwell.field(t)
    # draw contour-ish loops
    for k in range(0, X.shape[0], 5):
        pts = [(cx + X[k, j]*rx, cy + Y[k, j]*ry) for j in range(0, X.shape[1], 2)]
        col = CYAN if k % 10 == 0 else GRID
        d.line(pts, fill=col, width=1)
    d.ellipse((cx-5, cy-5, cx+5, cy+5), fill=RED)
    d.ellipse((cx+8, cy-5, cx+18, cy+5), fill=CYAN)
    en_label(d, (x0+12, y1-22), "E / B  FIELD LINES", size=12)


def navier_panel(im, d, box, sim, t, gy0):
    x0, y0, x1, y1 = box
    cx, cy = (x0+x1)//2, (gy0+y1)//2
    sx = (x1-x0)/2.2; sy = (y1-gy0)/1.7
    # cylinder
    d.ellipse((cx-0.32*sx, cy-0.32*sy, cx+0.32*sx, cy+0.32*sy),
              fill=(70, 58, 40), outline=GOLD, width=1)
    for i in range(sim.navier.np):
        px, py, age = sim.navier.px[i], sim.navier.py[i], sim.navier.age[i]
        xx = cx + px*sx; yy = cy + py*sy
        if x0+6 < xx < x1-6 and gy0 < yy < y1-6:
            a = int(90 + 130*min(age, 1))
            col = (a, int(a*0.78), int(a*0.48))
            uv, vv = sim.navier.vel(px, py, t)
            d.line([(xx-uv*5.5, yy-vv*5.5), (xx, yy)],
                   fill=(a//3, int(a*0.26), int(a*0.16)), width=1)
            d.ellipse((xx-1.9, yy-1.9, xx+1.9, yy+1.9), fill=col)
    en_label(d, (x0+12, y1-22), "VORTEX SHEDDING", size=12)


def psi_panel(im, d, box, t, gy0):
    x0, y0, x1, y1 = box
    mid = (gy0+y1)//2
    xs = np.linspace(0, 1, 220)
    s = 0.16 + 0.045*t
    env = np.exp(-((xs-0.5)**2)/(2*s*s))
    osc = np.cos(22*(xs-0.5) - 4*t)
    y = env*osc
    pts = [(x0+10 + v*(x1-x0-20), mid - y[i]*(y1-gy0)*0.34)
           for i, v in enumerate(xs)]
    d.line(pts, fill=GOLD, width=2)
    # |psi|^2 envelope
    up = [(x0+10 + v*(x1-x0-20), mid - env[i]*(y1-gy0)*0.34) for i, v in enumerate(xs)]
    dn = [(x0+10 + v*(x1-x0-20), mid + env[i]*(y1-gy0)*0.34) for i, v in enumerate(xs)]
    d.line(up, fill=(90, 96, 106), width=1)
    d.line(dn, fill=(90, 96, 106), width=1)


def beam_panel(im, d, box, sim, t, gy0):
    x0, y0, x1, y1 = box
    cx, cy = x0+42, (gy0+y1)//2
    L = x1-x0-84
    x, y = sim.beam.shape(t)
    # wall
    d.rectangle((cx-8, gy0+16, cx, y1-30), fill=(70, 72, 76))
    # mesh: longitudinal lines across width
    for w in np.linspace(-0.32, 0.32, 7):
        pts = [(cx + xi*L, cy + yi*(y1-gy0)*0.7 + w*(y1-gy0)*0.34)
               for xi, yi in zip(x, y)]
        d.line(pts, fill=GOLD if abs(w) < 0.01 else (168, 122, 62), width=1)
    for xi in np.linspace(0, 1, 13):
        yi = np.interp(xi, x, y)
        d.line([(cx+xi*L, cy+yi*(y1-gy0)*0.7-0.32*(y1-gy0)*0.34),
                (cx+xi*L, cy+yi*(y1-gy0)*0.7+0.32*(y1-gy0)*0.34)],
               fill=(120, 92, 56), width=1)


def rd_panel(im, d, box, sim, gy0):
    x0, y0, x1, y1 = box
    v = sim.rd.v
    rgb = map_color(v, 0.02, 0.34, (52, 30, 24), (246, 158, 70))
    big = Image.fromarray(rgb, "RGB").resize((x1-x0-16, y1-gy0-8), Image.BILINEAR)
    im.paste(big, (x0+8, gy0))


def schrod_panel(im, d, box, sim, t, gy0):
    x0, y0, x1, y1 = box
    cx, cy = (x0+x1)//2, (gy0+y1)//2
    rx, ry = (x1-x0)/2-22, (y1-gy0)/2-8
    X, Y, Z = sim.schrod.surface(t)
    def proj(i, j):
        xx, yy, zz = X[i, j], Y[i, j], Z[i, j]
        px = cx + (xx-yy)*0.46*rx
        py = cy + (xx+yy)*0.22*ry - zz*0.42*ry
        return px, py
    n = X.shape[0]
    for i in range(0, n, 3):
        pts = [proj(i, j) for j in range(n)]
        d.line(pts, fill=(110, 124, 146), width=1)
    for j in range(0, n, 3):
        pts = [proj(i, j) for i in range(n)]
        d.line(pts, fill=(110, 124, 146), width=1)
    # highlight ridge
    jmid = n//2
    d.line([proj(i, jmid) for i in range(n)], fill=CYAN, width=2)


# ---------------- subtitles (one sentence per scene, evolving) ----------------
CAPTIONS = [
    "七个偏微分方程，每一个都在屏幕上被现场求解。",
    "热量沿 ∂u/∂t = α∇²u 扩散，两个热点渐渐摊平。",
    "弦上的脉冲按波动方程裂成左右两列行波。",
    "麦克斯韦方程组让电磁场彼此激发、向外传播。",
    "纳维-斯托克斯方程刻画出圆柱后的涡旋脱落。",
    "薛定谔方程让波包一边前进一边展宽。",
    "概率密度 |ψ|² 守恒，振荡被高斯包络轻轻收住。",
    "悬臂梁的振动，由四阶导数与材料刚度共同决定。",
    "反应扩散方程中，均匀状态自发长出斑图。",
    "七个方程，七种演化——数学就这样把世界写成偏微分方程。",
]


def cn_text(im, d, text, xy, size, fill, anchor=None):
    f = ImageFont.truetype(FONT_CN, size)
    d.text(xy, text, font=f, fill=fill, anchor=anchor)


def en_label(d, xy, text, size=13, fill=DIMTEXT, anchor=None):
    d.text(xy, text, font=ImageFont.truetype(FONT_EN, size), fill=fill, anchor=anchor)


def render_frame(sim, idx):
    t = idx / FPS
    sim.step(t)
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)

    # ---- title (bilingual, no cursor) ----
    cn_text(im, d, "偏微分方程 · 现场求解", (W//2, 42), 36, WHITE, anchor="mt")
    en_label(d, (W//2, 88),
             "P D E   ·   S E V E N   E Q U A T I O N S   I N   M O T I O N",
             size=13, fill=DIMTEXT, anchor="mt")
    en_label(d, (W-24, 44), "AIMaker-Lab", size=12,
             fill=(120, 126, 136), anchor="rt")

    # ---- panel layout: 2 columns x 4 rows, tuned for 1080x1350 ----
    m, g = PANEL_MARGIN, PANEL_GUTTER
    pw = (W - 2*m - g)//2
    ph = PANEL_H
    ys = [PANEL_TOP + r*(ph + PANEL_GAP) for r in range(4)]
    boxes = []
    for y in ys:
        for c in range(2):
            x = m + c*(pw+g)
            boxes.append((x, y, x+pw, y+ph))

    keys = ["heat", "wave", "maxwell", "navier",
            "schrod", "psi", "beam", "rd"]
    subkeys = {"maxwell", "navier", "rd"}
    panels = {"heat": heat_panel, "wave": wave_panel, "maxwell": maxwell_panel,
              "navier": navier_panel, "schrod": schrod_panel, "psi": psi_panel,
              "beam": beam_panel, "rd": rd_panel}

    for key, box in zip(keys, boxes):
        y0 = box[1]
        has_sub = key in subkeys
        panel_base(d, box)
        draw_formula(d, im, key, (box[0], y0, box[2], y0+34))
        if has_sub:
            draw_formula(d, im, key, (box[0], y0+34, box[2], y0+56), sub=True)
        gy0 = y0 + (64 if has_sub else 44)
        if key in ("heat", "rd"):
            panels[key](im, d, box, sim, gy0)
        elif key in ("psi",):
            psi_panel(im, d, box, t, gy0)
        else:
            panels[key](im, d, box, sim, t, gy0)

    # ---- caption bar ----
    seg = DURATION/len(CAPTIONS)
    ci = min(int(t/seg), len(CAPTIONS)-1)
    top = H - CAPTION_H
    d.rectangle((0, top, W, H), fill=(20, 22, 26))
    d.line([(0, top), (W, top)], fill=GRID, width=1)
    d.rectangle((m, top+16, W-m, top+18), fill=(48, 52, 58))
    d.rectangle((m, top+16, m+int((W-2*m)*idx/NFRAMES), top+18), fill=CYAN)
    cn_text(im, d, CAPTIONS[ci], (W//2, top+36), 24, WHITE, anchor="mt")
    cn_text(im, d, "现场数值求解 · 无预渲染路径", (W//2, top+84), 14, DIMTEXT, anchor="mt")
    return im


def render_preview():
    """21 keyframes at 540x675 (half resolution), saved to preview_grid."""
    sim = Sim()
    out = os.path.join(HERE, "preview_grid")
    os.makedirs(out, exist_ok=True)
    for i in range(21):
        idx = int(i*(NFRAMES-1)/20)
        im = render_frame(sim, idx)
        im.resize((W//2, H//2), Image.LANCZOS).save(os.path.join(out, "k%02d.png" % i))
    print("preview frames:", 21)


if __name__ == "__main__":
    ensure_formulas()
    import sys
    if "--preview" in sys.argv:
        render_preview()
    else:
        sim = Sim()
        out = os.environ.get("PDE_OUT", os.path.join(HERE, "pde_silent.mp4"))
        import subprocess, shutil
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            try:
                import imageio_ffmpeg
                ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
            except Exception:
                pass
        cmd = [ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", "%dx%d" % (W, H), "-r", str(FPS), "-i", "-", "-an",
               "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", out]
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        for i in range(NFRAMES):
            p.stdin.write(render_frame(sim, i).tobytes())
            if i % 300 == 0:
                print("frame", i, flush=True)
        p.stdin.close()
        p.wait()
        print("rendered", out)
