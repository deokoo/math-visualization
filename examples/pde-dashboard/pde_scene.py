# -*- coding: utf-8 -*-
"""
Seven PDEs, solved live -- faithful replica of the reference layout.
1080x1350 (4:5) 30fps ~70s. Layout: row1 = 3 panels, row2 = 3 panels, row3 = 2 panels.
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

TOP = 112                 # title band height
ROW_H = [340, 340, 352]   # three rows
ROW_Y = [TOP, TOP + ROW_H[0] + 12, TOP + ROW_H[0] + ROW_H[1] + 24]
CAPTION_H = 162
MARGIN = 20
GUTTER = 12
NFRAMES = int(FPS * DURATION)

HERE = os.path.dirname(os.path.abspath(__file__))
FORM = os.path.join(HERE, "formulas")
os.makedirs(FORM, exist_ok=True)

BG = (16, 17, 19)
PANEL_BG = (22, 24, 28)
GRID = (52, 56, 63)
TEXT = (214, 219, 226)
DIMTEXT = (130, 136, 145)
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
    "beam":   r"$\omega_2/\omega_1=(\beta_2/\beta_1)^2=6.267$",
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
    fig = plt.figure(figsize=(0.01, 0.01), dpi=200)
    fig.patch.set_alpha(0)
    fig.text(0, 0, tex, fontsize=size, color=color, va="bottom", ha="left")
    fig.savefig(path, dpi=200, transparent=True, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

def ensure_formulas():
    for key, tex in FORMULAS.items():
        if not os.path.exists(_fname(key)):
            _render_formula(tex, _fname(key), 22, "#e6ebf2")
    for key, tex in FORMULA_SUB.items():
        if not os.path.exists(_fname(key, sub=True)):
            _render_formula(tex, _fname(key, sub=True), 17, "#b9c0cb")

def formula_img(key, sub=False):
    return Image.open(_fname(key, sub)).convert("RGBA")


# ---------------- numeric solvers ----------------
class HeatSolver:
    N = 80
    def __init__(self):
        y, x = np.mgrid[0:self.N, 0:self.N]
        self.u = (np.exp(-(((x-26)**2+(y-42)**2)/150)) +
                  0.75*np.exp(-(((x-56)**2+(y-30)**2)/90)))
        self.u = np.clip(self.u, 0, 1)
    def step(self):
        p = np.pad(self.u, 1, mode="edge")
        lap = p[1:-1,2:]+p[1:-1,:-2]+p[2:,1:-1]+p[:-2,1:-1]-4*self.u
        self.u = np.clip(self.u + 0.14*lap, 0, 1.2)


class WaveSolver:
    """1D wave: gaussian pulse splits into left/right travelling waves."""
    def field(self, t):
        x = np.linspace(-1, 1, 300)
        tp = t % 2.6
        f = lambda z: np.exp(-(z**2)/0.011)*np.cos(7*z)
        c = 0.34
        ul = 0.5*f(x + c*tp)     # left-going
        ur = 0.5*f(x - c*tp)     # right-going
        return x, ul, ur, ul+ur


class MaxwellSolver:
    """ analytic electric-dipole field lines r = r0 sin^2(theta), animated phase. """
    def field(self, t, n_lines=13, n_th=150):
        th = np.linspace(0, 2*np.pi, n_th)
        lines = []
        for k in range(n_lines):
            r0 = 0.12 + 0.82*(k/(n_lines-1))
            r = r0*np.cos(th)**2   # left-right lobes around the dipole axis
            # phase-driven slight breathing
            r = r*(0.94 + 0.06*np.cos(3.0*t - 2*r0*math.pi))
            lines.append((r*np.cos(th), r*np.sin(th)))
        return lines


class NavierSolver:
    """ cylinder/airfoil in potential flow + tracer particles. """
    def __init__(self):
        self.np = 90
        rng = np.random.default_rng(11)
        self.px = rng.uniform(-1.0, -0.5, self.np)
        self.py = rng.uniform(-0.72, 0.72, self.np)
        self.age = rng.uniform(0, 1, self.np)
    def vel(self, x, y, t):
        a = 0.24
        r2 = max(x*x + y*y, a*a)
        u = 1.0*(1 - a*a*(x*x-y*y)/(r2*r2))
        v = -1.0*(2*a*a*x*y/(r2*r2))
        return u, v
    def step(self, t, dt=0.045):
        for i in range(self.np):
            u, v = self.vel(self.px[i], self.py[i], t)
            self.px[i] += u*dt; self.py[i] += v*dt
            self.age[i] += dt
            if self.px[i] > 1.05 or abs(self.py[i]) > 0.82:
                self.px[i] = -1.0
                self.py[i] = np.random.uniform(-0.72, 0.72)
                self.age[i] = 0.0


class SchrodingerSolver:
    """ 3D gaussian wavepacket |psi| surface, evolving width/centre. """
    def surface(self, t, n=42):
        s = 0.22 + 0.10*(t % 6.0)
        xc = -0.35 + 0.14*(t % 6.0)
        xs = np.linspace(-1, 1, n)
        ys = np.linspace(-1, 1, n)
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        amp = np.exp(-(((X-xc)**2 + Y**2)/(2*s*s)))
        phase = np.cos(7*(X-xc) - 2.4*t)
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
        y = 0.20*mode*np.cos(2.3*t)
        return x, y


class RDSolver:
    N = 84
    def __init__(self):
        rng = np.random.default_rng(20261127)
        self.u = np.ones((self.N, self.N))
        self.v = np.zeros((self.N, self.N))
        for _ in range(12):
            cx, cy = rng.integers(6, self.N-6, 2)
            self.u[cx-4:cx+4, cy-4:cy+4] = 0.45
            self.v[cx-4:cx+4, cy-4:cy+4] = 0.30
        self.F, self.k = 0.033, 0.062
        self.Du, self.Dv = 0.21, 0.105
    def step(self, n=3):
        for _ in range(n):
            def lap(a):
                p = np.pad(a, 1, mode="edge")
                return p[1:-1,2:]+p[1:-1,:-2]+p[2:,1:-1]+p[:-2,1:-1]-4*a
            uvv = self.u*self.v*self.v
            self.u = np.clip(self.u + self.Du*lap(self.u)-uvv+self.F*(1-self.u), 0, 1)
            self.v = np.clip(self.v + self.Dv*lap(self.v)+uvv-(self.F+self.k)*self.v, 0, 1)


class Sim:
    def __init__(self):
        self.heat = HeatSolver(); self.wave = WaveSolver(); self.maxwell = MaxwellSolver()
        self.navier = NavierSolver(); self.schrod = SchrodingerSolver()
        self.beam = BeamSolver(); self.rd = RDSolver()
    def step(self, t):
        self.heat.step(); self.navier.step(t); self.rd.step(3)


# ---------------- drawing helpers ----------------
def map_color(x, lo, hi, c0, c1):
    c0 = np.asarray(c0, dtype=np.float64); c1 = np.asarray(c1, dtype=np.float64)
    x = np.clip((x-lo)/(hi-lo), 0, 1)[..., None]
    return (c0*(1-x)+c1*x).astype(np.uint8)

def panel_base(d, box):
    d.rounded_rectangle(box, radius=8, fill=PANEL_BG, outline=(44, 48, 54), width=1)

def draw_formula(d, im, key, box, sub=False):
    x0, y0, x1, y1 = box
    f = formula_img(key, sub=sub)
    maxw, maxh = x1-x0-12, y1-y0-2
    r = min(maxw/f.width, maxh/f.height, 1.0)
    f = f.resize((max(1, int(f.width*r)), max(1, int(f.height*r))), Image.LANCZOS)
    im.paste(f, (int((x0+x1-f.width)/2), int(y0+1)), f)
    return f.height


def cn_text(im, d, text, xy, size, fill, anchor=None):
    d.text(xy, text, font=ImageFont.truetype(FONT_CN, size), fill=fill, anchor=anchor)

def en_label(d, xy, text, size=12, fill=DIMTEXT, anchor=None):
    d.text(xy, text, font=ImageFont.truetype(FONT_EN, size), fill=fill, anchor=anchor)


# ---------------- panels (each draws inside [x0,y0,x1,y1]; gy0 = graph top) ----------------
def heat_panel(im, d, box, sim, gy0):
    x0, y0, x1, y1 = box
    rgb = map_color(sim.heat.u, 0.0, 0.9, (26, 52, 132), (250, 198, 70))
    big = Image.fromarray(rgb, "RGB").resize((x1-x0-16, y1-gy0-8), Image.BILINEAR)
    im.paste(big, (x0+8, gy0))
    cbx = x1-15
    for i in range(56):
        f = 1-i/55
        col = (int(26+(250-26)*f), int(52+(198-52)*f), int(132+(70-132)*f))
        d.line([(cbx, gy0+2+i*2), (cbx+5, gy0+2+i*2)], fill=col)


def wave_panel(im, d, box, sim, t, gy0):
    x0, y0, x1, y1 = box
    lane = (y1-gy0)//3
    x, ul, ur, ut = sim.wave.field(t)
    def plot(vals, cy, col):
        pts = [(x0+12+(i/(len(vals)-1))*(x1-x0-24), cy-vals[i]*lane*0.42)
               for i in range(len(vals))]
        d.line(pts, fill=col, width=2)
    # top: right-going pulse (gold) + arrow; middle: left-going (blue) + arrow; bottom: sum (white) + walls
    for k, (vals, col, lab) in enumerate([(ur, GOLD, "+"), (ul, (110,150,225), "-"), (ut, WHITE, "")]):
        cy = gy0+lane*k+lane//2+4
        d.line([(x0+12, cy), (x1-12, cy)], fill=(46, 50, 56), width=1)
        plot(vals, cy, col)
        if k == 0:
            en_label(d, (x0+16, cy-lane//2+2), "+", 14, GOLD)
            d.line([(x1-46, cy-8), (x1-16, cy-8)], fill=GOLD, width=1)
            d.line([(x1-22, cy-12), (x1-16, cy-8), (x1-22, cy-4)], fill=GOLD, width=1)
        elif k == 1:
            en_label(d, (x1-24, cy-lane//2+2), "-", 14, (110,150,225))
            d.line([(x0+16, cy+8), (x0+46, cy+8)], fill=(110,150,225), width=1)
            d.line([(x0+22, cy+4), (x0+16, cy+8), (x0+22, cy+12)], fill=(110,150,225), width=1)
        else:
            d.line([(x0+12, cy-lane//2+4), (x0+12, cy+lane//2-4)], fill=(120,126,134), width=2)
            d.line([(x1-12, cy-lane//2+4), (x1-12, cy+lane//2-4)], fill=(120,126,134), width=2)


def maxwell_panel(im, d, box, sim, t, gy0):
    x0, y0, x1, y1 = box
    cx, cy = (x0+x1)//2, (gy0+y1)//2
    rx, ry = (x1-x0)/2-14, (y1-gy0)/2-12
    lines = sim.maxwell.field(t)
    for k, (X, Y) in enumerate(lines):
        pts = [(cx+X[j]*rx, cy+Y[j]*ry) for j in range(0, len(X), 2)]
        col = (128, 134, 142) if k % 2 == 0 else (104, 110, 118)
        d.line(pts, fill=col, width=1)
    d.ellipse((cx-9, cy-6, cx+1, cy+4), fill=RED)
    d.ellipse((cx+3, cy-6, cx+13, cy+4), fill=CYAN)
    en_label(d, (x0+12, y1-20), "E / B  FIELD LINES", 11)


def navier_panel(im, d, box, sim, t, gy0):
    x0, y0, x1, y1 = box
    X0, X1 = x0+12, x1-12
    Y0, Y1 = gy0+10, y1-18
    midy = (Y0+Y1)/2
    a = 0.23
    def flow_y(x, y0line):
        return y0line + a*a*y0line/(x*x+y0line*y0line+0.05)*math.exp(-(x*x)/0.42)
    def px(x): return X0+(x+1)/2*(X1-X0)
    def py(yv): return midy-yv*(Y1-Y0)/2
    # curved streamline grid
    for y0line in np.linspace(-0.85, 0.85, 12):
        pts = [(px(x), py(flow_y(x, y0line))) for x in np.linspace(-1, 1, 64)]
        d.line(pts, fill=(96, 102, 110), width=1)
    for xv in np.linspace(-0.92, 0.92, 10):
        pts = [(px(xv), py(flow_y(xv, y0line))) for y0line in np.linspace(-0.85, 0.85, 26)]
        d.line(pts, fill=(74, 80, 88), width=1)
    # airfoil
    acx, acy = px(0), midy
    d.ellipse((acx-14, acy-24, acx+32, acy+8), fill=(112, 116, 122), outline=(176, 180, 186), width=1)
    # tracers
    for i in range(sim.navier.np):
        pxx, pyy, age = sim.navier.px[i], sim.navier.py[i], sim.navier.age[i]
        xx, yy = px(pxx), py(flow_y(pxx, pyy))
        if X0+2 < xx < X1-2 and Y0+2 < yy < Y1-2:
            col = (118, 200, 140) if (i % 3 == 0) else (224, 158, 84)
            d.ellipse((xx-1.6, yy-1.6, xx+1.6, yy+1.6), fill=col)
    en_label(d, (x0+12, y1-18), "FLOW AROUND AIRFOIL", 11)


def schrod_panel(im, d, box, sim, t, gy0):
    """3D coloured wave-packet surface inside a circular viewport."""
    x0, y0, x1, y1 = box
    cx, cy = (x0+x1)//2, (gy0+y1)//2
    R = int(min((x1-x0)/2-12, (y1-gy0)/2-10))
    # clip everything (surface + circle) through one layer so nothing escapes the panel
    layer = Image.new("RGBA", (x1-x0, y1-gy0), (0, 0, 0, 0))
    dl = ImageDraw.Draw(layer)
    lx, ly = cx-x0, cy-gy0
    ang = 0.4 + 0.30*t
    ca, sa = math.cos(ang), math.sin(ang)
    X, Y, Z = sim.schrod.surface(t)
    n = X.shape[0]
    def proj(i, j):
        xx, yy, zz = X[i, j], Y[i, j], Z[i, j]
        xr = xx*ca - yy*sa
        yr = xx*sa + yy*ca
        return (lx+(xr-yr)*0.42*R, ly+(xr+yr)*0.20*R - zz*0.40*R)
    def zcol(z):
        q = max(0, min(1, (z+0.8)/1.6))
        return (int(70+130*q), int(150-40*q), int(225-70*q), 210)
    for i in range(0, n, 2):
        cols = [zcol(Z[i, j]) for j in range(n)]
        for j in range(n-1):
            dl.line([proj(i, j), proj(i, j+1)], fill=cols[j], width=2)
    for j in range(0, n, 3):
        for i in range(n-1):
            dl.line([proj(i, j), proj(i+1, j)], fill=zcol(Z[i, j]), width=1)
    # small coordinate triad at centre
    dl.line([(lx, ly), (lx+22, ly)], fill=(200, 206, 214), width=2)
    dl.line([(lx, ly), (lx, ly-22)], fill=(200, 206, 214), width=2)
    dl.line([(lx, ly), (lx-16, ly+10)], fill=(200, 206, 214), width=2)
    dl.ellipse((lx-R, ly-R, lx+R, ly+R), outline=(104, 110, 118), width=1)
    # paste the whole circular viewport through a circular clip
    mask = Image.new("L", (x1-x0, y1-gy0), 0)
    dm = ImageDraw.Draw(mask)
    dm.ellipse((lx-R, ly-R, lx+R, ly+R), fill=255)
    im.paste(layer, (x0, gy0), mask)
    en_label(d, (x0+12, y1-19), "Re ψ  SURFACE", 11)


def psi_panel(im, d, box, t, gy0):
    x0, y0, x1, y1 = box
    xs = np.linspace(-1, 1, 300)
    xc = 0.0
    s = 0.085 + 0.015*(t % 5)
    env = np.exp(-((xs-xc)**2)/(2*s*s))
    wave = env*np.cos(23*(xs-xc) - 3.0*t)
    base = gy0 + (y1-gy0)*0.55
    amp = (y1-gy0)*0.30
    d.line([(x0+12, base), (x1-12, base)], fill=(88, 94, 102), width=1)
    pts = [(x0+12+v*(x1-x0-24), base-wave[i]*amp)
           for i, v in enumerate(xs)]
    d.line(pts, fill=GOLD, width=2)
    # envelope
    up = [(x0+12+v*(x1-x0-24), base-env[i]*amp) for i, v in enumerate(xs)]
    dn = [(x0+12+v*(x1-x0-24), base+env[i]*amp) for i, v in enumerate(xs)]
    d.line(up, fill=(110, 96, 64), width=1); d.line(dn, fill=(110, 96, 64), width=1)
    en_label(d, (x1-18, gy0+6), "|ψ|²", 13, GOLD, anchor="rt")


def beam_panel(im, d, box, sim, t, gy0):
    """Solid 3D rectangular cantilever box beam: orange top, gridded grey faces, fixed wall."""
    x0, y0, x1, y1 = box
    ax0, ax1 = x0+44, x1-38
    cy = (gy0+y1)//2
    L = ax1-ax0
    x, y = sim.beam.shape(t)
    amp = (y1-gy0)*0.50
    depth = min(24, (y1-gy0)*0.18)
    dx, dy = max(10, int(L*0.032)), max(6, int(depth*0.6))
    half = depth/2

    def cl(xi):
        return cy - np.interp(xi, x, y)*amp

    def verts(xi):
        yy = cl(xi); px = ax0 + xi*L
        return ((px, yy-half), (px, yy+half),
                (px+dx, yy-half-dy), (px+dx, yy+half-dy))

    # fixed wall with hatching
    d.rectangle((ax0-16, gy0+18, ax0+2, y1-26), fill=(88, 90, 94))
    for hy in range(gy0+18, y1-26, 10):
        d.line([(ax0-16, hy), (ax0+2, hy+7)], fill=(134, 138, 144), width=1)

    # ghost bending poses (motion trail)
    for dt, a in ((0.55, 74), (0.28, 104)):
        _, yh = sim.beam.shape(t-dt)
        d.line([(ax0+xi*L, cy-np.interp(xi, x, yh)*amp) for xi in x],
               fill=(a, int(a*0.74), int(a*0.42)), width=2)

    # TOP face: orange near clamp fading to light grey toward tip, drawn as thin quads
    ns = 44
    for i in range(ns):
        xi0, xi1 = i/ns, (i+1)/ns
        ft0, _, bt0, _ = verts(xi0)
        ft1, _, bt1, _ = verts(xi1)
        f = xi0
        if f < 0.42:
            k = 1 - f/0.42
            col = (int(162+80*k), int(160-14*k), int(162-70*k))
        else:
            k = (f-0.42)/0.58
            col = (int(162+34*k), int(158+38*k), int(160+30*k))
        d.polygon([ft0, ft1, bt1, bt0], fill=col)
    # top-face cross-grid
    for xi in np.linspace(0, 1, 20):
        ft, _, bt, _ = verts(xi)
        d.line([ft, bt], fill=(228, 214, 196), width=1)

    # FRONT face: cool grey with fine grid
    ft_list = [verts(xi)[0] for xi in x]
    fb_list = [verts(xi)[1] for xi in x]
    d.polygon(ft_list + fb_list[::-1], fill=(170, 174, 182))
    for xi in np.linspace(0, 1, 20):
        ft, fb, _, _ = verts(xi)
        d.line([ft, fb], fill=(112, 116, 124), width=1)
    for frac in (0.33, 0.66):
        d.line([(ax0+xi*L, cl(xi)-half+frac*depth) for xi in x],
               fill=(126, 130, 138), width=1)

    # TIP end face
    ft, fb, bt, bb = verts(1)
    d.polygon([ft, bt, bb, fb], fill=(124, 128, 136))
    d.line([ft, bt], fill=(206, 210, 216), width=1)

    # outlines
    d.line(ft_list, fill=(232, 230, 224), width=1)
    d.line(fb_list, fill=(104, 108, 116), width=1)

    cn_text(im, d, "悬臂梁 · 模态振动", (x1-14, y1-18), 13, DIMTEXT, anchor="ra")


def rd_panel(im, d, box, sim, gy0):
    x0, y0, x1, y1 = box
    rgb = map_color(sim.rd.v, 0.02, 0.34, (176, 58, 32), (255, 202, 96))
    big = Image.fromarray(rgb, "RGB").resize((x1-x0-16, y1-gy0-8), Image.BILINEAR)
    im.paste(big, (x0+8, gy0))


# ---------------- subtitle script ----------------
CAPTIONS = [
    "七个偏微分方程，每一个都在屏幕上被现场求解。",
    "热量沿 ∂u/∂t = α∇²u 扩散，两个热斑渐渐摊平。",
    "弦上的脉冲按波动方程分裂成向左和向右的两列行波。",
    "麦克斯韦方程组让电磁场线彼此激发、向外传播。",
    "纳维-斯托克斯方程刻画出流体绕过物体时的弯曲流线。",
    "薛定谔方程让波包的概率幅在复平面上起伏演化。",
    "概率密度 |ψ|² 守恒，高频振荡被高斯包络收住。",
    "悬臂梁的振动，由四阶导数与材料刚度共同决定。",
    "反应扩散方程中，均匀状态自发长出斑图。",
]


# ---------------- frame ----------------
def render_frame(sim, idx):
    t = idx/FPS
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)

    cn_text(im, d, "偏微分方程 · 现场求解", (W//2, 38), 36, WHITE, anchor="mt")
    en_label(d, (W//2, 82), "P D E   ·   S E V E N   E Q U A T I O N S   I N   M O T I O N",
             13, DIMTEXT, anchor="mt")
    en_label(d, (W-22, 40), "AIMaker-Lab", 12, (120, 126, 136), anchor="rt")

    specs = [
        # (key, row, col_in_row, cols_in_row, has_sub, painter)
        ("heat",    0, 0, 3, False, "graph"),
        ("wave",    0, 1, 3, False, "graph"),
        ("maxwell", 0, 2, 3, True,  "graph"),
        ("navier",  1, 0, 3, True,  "graph"),
        ("schrod",  1, 1, 3, False, "graph"),
        ("psi",     1, 2, 3, False, "graph"),
        ("beam",    2, 0, 2, False, "graph"),
        ("rd",      2, 1, 2, True,  "graph"),
    ]
    painters = {"heat": heat_panel, "wave": wave_panel, "maxwell": maxwell_panel,
                "navier": navier_panel, "schrod": schrod_panel, "psi": psi_panel,
                "beam": beam_panel, "rd": rd_panel}
    for key, row, col, ncols, has_sub, _ in specs:
        y0 = ROW_Y[row]; rh = ROW_H[row]
        total = W-2*MARGIN-(ncols-1)*GUTTER
        pw = total//ncols
        x0 = MARGIN+col*(pw+GUTTER)
        box = (x0, y0, x0+pw, y0+rh)
        panel_base(d, box)
        draw_formula(d, im, key, (x0, y0, x0+pw, y0+32))
        if has_sub:
            draw_formula(d, im, key, (x0, y0+32, x0+pw, y0+54), sub=True)
        gy0 = y0 + (62 if has_sub else 42)
        p = painters[key]
        if key in ("heat", "rd"):
            p(im, d, box, sim, gy0)
        elif key == "psi":
            p(im, d, box, t, gy0)
        else:
            p(im, d, box, sim, t, gy0)

    seg = DURATION/len(CAPTIONS)
    ci = min(int(t/seg), len(CAPTIONS)-1)
    top = H-CAPTION_H
    d.rectangle((0, top, W, H), fill=(20, 22, 26))
    d.line([(0, top), (W, top)], fill=GRID, width=1)
    d.rectangle((MARGIN, top+16, W-MARGIN, top+18), fill=(48, 52, 58))
    d.rectangle((MARGIN, top+16, MARGIN+int((W-2*MARGIN)*idx/NFRAMES), top+18), fill=CYAN)
    cn_text(im, d, CAPTIONS[ci], (W//2, top+36), 23, WHITE, anchor="mt")
    cn_text(im, d, "现场数值求解 · 无预渲染路径", (W//2, top+82), 13, DIMTEXT, anchor="mt")
    return im


def render_preview():
    sim = Sim()
    out = os.path.join(HERE, "preview_grid")
    os.makedirs(out, exist_ok=True)
    for i in range(21):
        idx = int(i*(NFRAMES-1)/20)
        im = render_frame(sim, idx)
        im.resize((540, 675), Image.LANCZOS).save(os.path.join(out, "k%02d.png" % i))
    print("preview frames:", 21)


def render_full():
    ensure_formulas()
    sim = Sim()
    import subprocess, imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    out = os.path.join(HERE, "pde_silent.mp4")
    cmd = [ff, "-y", "-loglevel", "error", "-f", "rawvideo", "-vcodec", "rawvideo",
           "-s", "%dx%d" % (W, H), "-pix_fmt", "rgb24", "-r", str(FPS), "-i", "-",
           "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
           "-preset", "veryfast", "-crf", "20", out]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for idx in range(NFRAMES):
        sim.step(idx/FPS)
        proc.stdin.write(render_frame(sim, idx).tobytes())
        if idx % 300 == 0:
            print("frame", idx)
    proc.stdin.close(); proc.wait()
    print("rendered", out)


if __name__ == "__main__":
    ensure_formulas()
    import sys
    if "--preview" in sys.argv:
        render_preview()
    else:
        render_full()
