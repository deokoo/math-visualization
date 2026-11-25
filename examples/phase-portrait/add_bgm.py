#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""微分方程相图成片专用配乐 —— 「PHASOR / F#-Dorian」

系列铁律：每部动画的配乐必须彼此不同（调性 / 音色 / 结构至少换两样）。
本片 vs 已用配乐的区别（三样全换）：
  - 泰勒 / 最速降线：D 小调五声、滑升正弦 pad + 加法铃音 + 心跳
  - 元胞自动机：A 小调五声、锯齿 pad + FM 金属铃音 + 粒子脉冲
  - Dijkstra：E 小调五声、拨弦包络 + 三角波合唱 + 木质敲击 + 7/8
  - 本片：F# 多利亚调式（F#2=92.50Hz）、**减法滤波锯齿扫频 pad +
          钢片琴(glockenspiel)非谐波泛音 + 低通噪声打击**，
          结构为 **5/4 拍 + 黄金比间隔音序**（音符间隔按 φ 幂次轮换）。

结构（全程不循环、持续演进）：
  - pad：12 个锯齿谐波，低通截止频率随时间指数上行 + 独立慢漂移，
        亮度持续变化，无稳定和弦循环
  - 钢片琴：非整数比泛音 [1, 2.76, 5.40, 8.93]，按黄金比幂次间隔触发，
        音高沿多利亚音阶随机游走，力度/衰减随进度演进
  - 噪声打击：5/4 拍（每 5 个四分音符一小节），第 1 拍重击，带微节奏漂移
  - 低频 F#1 呼吸底，独立漂移；收尾渐出；无版权依赖

用法：python add_bgm.py [input.mp4] [output.mp4]
"""
import re
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
_OLD = HERE.parent
_WS = HERE.parent.parent / "2026-11-17-10-23-42"
_CANDIDATES = [
    HERE / "ffmpeg.exe",
    _WS / "manim_gaussian" / "ffmpeg.exe",
]
FFMPEG = next((p for p in _CANDIDATES if p.exists()), None)
if FFMPEG is None:
    try:
        import imageio_ffmpeg
        FFMPEG = Path(imageio_ffmpeg.get_ffmpeg_exe())
    except ImportError:
        FFMPEG = Path("ffmpeg")
BGM = HERE / "bgm.wav"

SR = 44100
ROOT = 92.4986                   # F#2 —— 区别于 D2/D3、A2、E2
SCALE = [0, 2, 3, 5, 7, 9, 10]   # F# 多利亚（F# G# A B C# D# E）
SEED = 201120
PHI = (1 + 5 ** 0.5) / 2
QUARTER = 0.305                  # 四分音符 → 5/4 小节 ≈ 1.525s

OLD_BGMS = [
    _OLD / "manim_taylor" / "bgm.wav",
    _OLD / "manim_brachistochrone" / "bgm.wav",
    _OLD / "manim_cellular" / "bgm.wav",
    _OLD / "manim_dijkstra" / "bgm.wav",
    _OLD / "manim_binary_search" / "bgm.wav",
]


def make_bgm(duration: float, path: Path) -> None:
    n = int(SR * duration)
    t = np.arange(n) / SR
    audio = np.zeros(n, dtype=np.float64)

    # ---- 1) 减法滤波锯齿扫频 pad：12 谐波 + 时变低通截止 ----
    cutoff = 260.0 * np.exp(2.15 * t / duration) * (1.0 + 0.09 * np.sin(2 * np.pi * 0.037 * t))
    width = 170.0 + 55.0 * np.sin(2 * np.pi * 0.023 * t + 1.1)
    pad = np.zeros(n, dtype=np.float64)
    for k in range(1, 13):
        fk = ROOT * k
        amp = (1.0 / k) / (1.0 + np.exp(np.clip((fk - cutoff) / width, -40, 40)))
        ph = 2 * np.pi * fk * (t + 0.0045 * t * t / duration)
        pad += amp * np.sin(ph + 0.31 * k)
    swell = np.clip((t - 0.9) / 6.0, 0, 1)
    audio += 0.034 * swell * pad

    # ---- 2) 钢片琴：黄金比间隔音序 + 非谐波泛音，多利亚音阶随机游走 ----
    rng = np.random.default_rng(SEED)
    gaps = [0.24 * PHI ** (k - 2) for k in range(5)]  # 0.09 / 0.15 / 0.24 / 0.39 / 0.63
    tpos = 0.95
    step = 0
    degree = 3
    while tpos < duration - 0.45:
        degree = int(np.clip(degree + rng.choice([-2, -1, 0, 1, 1, 2]), 0, 13))
        semi = SCALE[degree % len(SCALE)] + 12 * (degree // len(SCALE))
        f0 = ROOT * 4 * 2 ** (semi / 12)     # 中高音区钢片琴
        s0 = int(tpos * SR)
        life = 1.05 + 0.25 * rng.random()
        s1 = min(int((tpos + life) * SR), n)
        if s1 <= s0:
            break
        local = np.arange(s1 - s0) / SR
        body = (np.sin(2 * np.pi * f0 * local) * np.exp(-local * 3.1)
                + 0.52 * np.sin(2 * np.pi * 2.76 * f0 * local) * np.exp(-local * 5.4)
                + 0.28 * np.sin(2 * np.pi * 5.40 * f0 * local) * np.exp(-local * 9.0)
                + 0.11 * np.sin(2 * np.pi * 8.93 * f0 * local) * np.exp(-local * 14.0))
        accent = 0.85 if step % 5 == 0 else (0.62 if step % 3 == 1 else 0.46)
        audio[s0:s1] += 0.070 * accent * body * np.clip(local / 0.003, 0, 1)
        # 黄金比幂次轮换 + 慢漂移，避免固定小节循环
        gap = gaps[step % 5] * (1.0 + 0.16 * np.sin(0.31 * tpos + 0.7))
        tpos += gap
        step += 1

    # ---- 3) 低通噪声打击：5/4 拍 ----
    tpos = 0.80
    step = 0
    while tpos < duration - 0.2:
        s0 = int(tpos * SR)
        ln = int(0.085 * SR)
        s1 = min(s0 + ln, n)
        if s1 <= s0:
            break
        noise = rng.normal(size=s1 - s0)
        y = np.empty(s1 - s0, dtype=np.float64)
        y[0] = noise[0]
        for i in range(1, len(y)):          # 一阶低通 → 柔和鼓刷
            y[i] = 0.55 * y[i - 1] + 0.45 * noise[i]
        local = np.arange(s1 - s0) / SR
        beat = step % 5
        accent = 1.0 if beat == 0 else (0.55 if beat == 2 else 0.30)
        audio[s0:s1] += 0.055 * accent * y * np.exp(-local * 55)
        tpos += QUARTER * (1.0 + 0.012 * np.sin(step * 0.7))
        step += 1

    # ---- 4) 低频呼吸底：F#1，独立漂移 ----
    fsub = ROOT / 2
    sub_ph = 2 * np.pi * fsub * (t + 0.006 * t * t / duration)
    sub = np.sin(sub_ph + 0.42 * np.sin(2 * np.pi * 0.029 * t))
    audio += 0.017 * np.clip((t - 0.7) / 5.0, 0, 1) * sub

    # ---- 5) 收尾与归一化 ----
    rng2 = np.random.default_rng(SEED + 7)
    audio += 0.0020 * rng2.normal(size=n)
    fi, fo = int(1.0 * SR), int(2.0 * SR)
    audio[:fi] *= np.linspace(0, 1, fi)
    audio[-fo:] *= np.linspace(1, 0, fo)
    audio /= max(1e-9, float(np.max(np.abs(audio))))
    audio *= 0.80
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes((audio * 32767).astype("<i2").tobytes())
    print(f"BGM（PHASOR / F#-Dorian，专用签名）: {path} ({duration:.1f}s)")


def _load_mono(path: Path, limit_s=6.0):
    with wave.open(str(path), "rb") as wf:
        n = min(wf.getnframes(), int(SR * limit_s))
        raw = wf.readframes(n)
    return np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0


def verify(path: Path):
    """波形余弦相似度（与旧片 <0.1）+ 自身自相关峰值（<0.8，无循环）。"""
    step = 8
    x = _load_mono(path)[::step]
    print("\n[配乐差异检验]")
    for old in OLD_BGMS:
        if not old.exists():
            continue
        y = _load_mono(old)[::step]
        m = min(len(x), len(y))
        a, b = x[:m] - x[:m].mean(), y[:m] - y[:m].mean()
        cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
        print(f"  vs {old.parent.name:20s}: {cos:+.4f}")
    z = x - x.mean()
    ac = np.fft.irfft(np.abs(np.fft.rfft(z)) ** 2)[: len(z)]
    ac /= ac[0] + 1e-12
    lo, hi = int(0.4 * SR / step), len(ac) // 2
    peak = float(np.max(ac[lo:hi]))
    print(f"  自身自相关峰（跳过 <0.4s）: {peak:.3f}  → {'无循环' if peak < 0.8 else '需检查'}")


def get_duration(video: Path) -> float:
    out = subprocess.run([str(FFMPEG), "-hide_banner", "-i", str(video)],
                         capture_output=True, text=True, errors="ignore")
    m = re.search(r"Duration: (\d+):(\d+):(\d+\.\d+)", out.stderr)
    if not m:
        raise RuntimeError("无法读取视频时长")
    h, mi, s = m.groups()
    return int(h) * 3600 + int(mi) * 60 + float(s)


def main():
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else \
        HERE / "media/videos/phase_portrait_scene/1350p30/PhasePortrait.mp4"
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else \
        HERE.parent / "phase_portrait_final.mp4"
    if not src.exists():
        raise SystemExit(f"找不到输入视频: {src}")
    dur = get_duration(src)
    print(f"视频时长: {dur:.2f}s")
    make_bgm(dur, BGM)
    verify(BGM)
    cmd = [str(FFMPEG), "-y", "-loglevel", "error",
           "-i", str(src), "-i", str(BGM),
           "-map", "0:v:0", "-map", "1:a:0",
           "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", str(dst)]
    subprocess.run(cmd, check=True)
    print(f"完成: {dst} ({dst.stat().st_size / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
