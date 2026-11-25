#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""二分查找成片专用配乐 —— 「BISECT / C-Lydian」

系列铁律：每部动画的配乐必须彼此不同（调性 / 音色 / 结构至少换两样）。
本片 vs 已用配乐的区别（三样全换）：
  - 泰勒 / 最速降线：D 小调五声、滑升正弦 pad + 加法铃音 + 心跳
  - 元胞自动机：A 小调五声、锯齿 pad + FM 金属铃音 + 粒子脉冲
  - Dijkstra：E 小调五声、拨弦包络 + 三角波合唱 + 木质敲击 + 7/8
  - 相图：F# 多利亚、减法锯齿扫频 pad + 钢片琴 + 低通噪声 + 5/4 黄金比
  - 本片：C 利底亚（C2=65.41Hz，含 #4=F#）、**方波 PWM pad +
          玻璃谐波铃音 + 高通噪声 ticks**，
          结构为 **11/8 拍 + 递减间隔收缩音序**（音符间隔逐次减半，
          呼应二分查找"每步砍一半"；到 29 步命中时收束）。

结构（全程不循环、持续演进）：
  - PWM pad：方波（两反相锯齿比较生成可变占空比），占空比随进度渐变，
            5 声部按利底亚音程叠置，独立慢漂移，无稳定和弦循环
  - 玻璃谐波铃：非整数比泛音 [1, 2.01, 3.97]，音高沿利底亚音阶向下收敛，
                间隔按 0.7^k 递减（砍半感），力度随接近目标渐强
  - 高通噪声 ticks：11/8 拍（每 11 个八分音符一小节），第 1 拍重击，
                    对应"比较一次"的机械节拍，带微节奏漂移
  - 低频 C1 呼吸底，独立漂移；29 步命中处一次清亮泛音收束；尾渐出；无版权依赖

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
ROOT = 65.4064                   # C2 —— 区别于 D2/D3、A2、E2、F#2
SCALE = [0, 2, 4, 6, 7, 9, 11]   # C 利底亚（C D E F# G A B）
SEED = 221120
EIGHTH = 0.272                   # 八分音符 → 11/8 小节 ≈ 2.99s
HIT_T = 16.0                     # 画面命中收束时刻

OLD_BGMS = [
    _OLD / "manim_taylor" / "bgm.wav",
    _OLD / "manim_brachistochrone" / "bgm.wav",
    _OLD / "manim_cellular" / "bgm.wav",
    _OLD / "manim_dijkstra" / "bgm.wav",
    _OLD / "manim_phase_portrait" / "bgm.wav",
]


def make_bgm(duration: float, path: Path) -> None:
    n = int(SR * duration)
    t = np.arange(n) / SR
    audio = np.zeros(n, dtype=np.float64)

    # ---- 1) 方波 PWM pad：反相锯齿比较生成可变占空比方波，利底亚 5 声部 ----
    # 占空比 0.5 -> 0.24 缓慢渐变，亮度持续变化；每声部独立慢漂移
    duty = 0.50 - 0.26 * np.clip(t / duration, 0, 1) \
        + 0.035 * np.sin(2 * np.pi * 0.031 * t)
    degrees = [0, 2, 4, 6, 7]   # 利底亚音阶级数（7 = 高八度根音）
    for vi, deg in enumerate(degrees):
        semi = SCALE[deg % len(SCALE)] + 12 * (deg // len(SCALE))
        f0 = ROOT * 2 * 2 ** (semi / 12)
        # 两路锯齿相位比较 → PWM 方波（±1）
        ph = (f0 * (t + 0.0032 * vi * t * t / duration)) % 1.0
        pwm = np.where(ph < duty, 1.0, -1.0)
        # 一阶低通软化边缘（保留方波谐波但去刺耳高频）
        soft = np.empty(n)
        soft[0] = pwm[0]
        a = 0.62 - 0.10 * vi / len(degrees)
        for i in range(1, n):
            soft[i] = a * soft[i - 1] + (1 - a) * pwm[i]
        amp = 0.016 / (1 + 0.18 * vi)
        audio += amp * soft * (0.75 + 0.25 * np.sin(2 * np.pi * (0.021 + 0.004 * vi) * t))

    # ---- 2) 玻璃谐波铃：递减间隔收缩音序，音高沿利底亚向下收敛 ----
    rng = np.random.default_rng(SEED)
    tpos = 1.0
    step = 0
    degree = 13                   # 起点最高，逐步向目标收敛
    while tpos < duration - 0.5:
        # 递减间隔：0.62 * 0.87^step，下限 0.10（呼应每步砍一半，但不会无限快）
        gap = max(0.10, 0.62 * (0.87 ** step)) * (1.0 + 0.05 * np.sin(step * 1.7))
        semi = SCALE[degree % len(SCALE)] + 12 * (degree // len(SCALE))
        f0 = ROOT * 8 * 2 ** (semi / 12)   # 中高音区玻璃铃
        s0 = int(tpos * SR)
        life = 0.95 + 0.30 * rng.random()
        s1 = min(int((tpos + life) * SR), n)
        if s1 <= s0:
            break
        local = np.arange(s1 - s0) / SR
        # 玻璃谐波：近整数比 + 极轻微失谐，长衰减
        body = (np.sin(2 * np.pi * f0 * local) * np.exp(-local * 2.4)
                + 0.34 * np.sin(2 * np.pi * 2.01 * f0 * local) * np.exp(-local * 4.2)
                + 0.13 * np.sin(2 * np.pi * 3.97 * f0 * local) * np.exp(-local * 7.5))
        accent = 0.55 + 0.40 * (step / 29.0)   # 越接近目标越响
        audio[s0:s1] += 0.058 * accent * body * np.clip(local / 0.004, 0, 1)
        degree = max(0, degree - 1) if step % 3 else min(15, degree + 2)
        tpos += gap
        step += 1

    # ---- 3) 高通噪声 ticks：11/8 拍，机械的"比较一次"节拍 ----
    tpos = 0.9
    step = 0
    while tpos < duration - 0.2:
        s0 = int(tpos * SR)
        ln = int(0.055 * SR)
        s1 = min(s0 + ln, n)
        if s1 <= s0:
            break
        noise = rng.normal(size=s1 - s0)
        y = np.empty(s1 - s0, dtype=np.float64)
        y[0] = noise[0]
        for i in range(1, len(y)):            # 一阶高通（差值）→ 短促 tick
            y[i] = noise[i] - 0.72 * noise[i - 1]
        local = np.arange(s1 - s0) / SR
        beat = step % 11
        accent = 1.0 if beat == 0 else (0.5 if beat in (4, 7) else 0.22)
        audio[s0:s1] += 0.040 * accent * y * np.exp(-local * 90)
        tpos += EIGHTH * (1.0 + 0.010 * np.sin(step * 0.9))
        step += 1

    # ---- 4) 低频 C1 呼吸底，独立漂移 ----
    fsub = ROOT / 2
    sub_ph = 2 * np.pi * fsub * (t + 0.005 * t * t / duration)
    sub = np.sin(sub_ph + 0.38 * np.sin(2 * np.pi * 0.027 * t))
    audio += 0.015 * np.clip((t - 0.6) / 5.0, 0, 1) * sub

    # ---- 5) 29 步命中处清亮泛音收束 ----
    s0 = int(HIT_T * SR)
    if s0 < n:
        local = np.arange(min(n - s0, int(2.0 * SR))) / SR
        chime = (np.sin(2 * np.pi * ROOT * 16 * local)
                 + 0.5 * np.sin(2 * np.pi * ROOT * 24 * local)
                 + 0.25 * np.sin(2 * np.pi * ROOT * 32 * local))
        audio[s0:s0 + len(local)] += 0.045 * chime * np.exp(-local * 1.6)

    # ---- 6) 收尾与归一化 ----
    rng2 = np.random.default_rng(SEED + 7)
    audio += 0.0018 * rng2.normal(size=n)
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
    print(f"BGM（BISECT / C-Lydian，专用签名）: {path} ({duration:.1f}s)")


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
        print(f"  vs {old.parent.name:22s}: {cos:+.4f}")
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
        HERE / "media/videos/binary_search_scene/1350p30/BinarySearch.mp4"
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else \
        HERE.parent / "binary_search_final.mp4"
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
