#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""格林公式成片专用配乐 —— 「VORTEX / C#-WholeTone」

系列铁律：每部动画配乐必须彼此不同（调性 / 音色 / 结构至少换两样）。
本片 vs 已用配乐（三样全换）：
  - 泰勒 / 最速降线：D 小调五声、滑升正弦 pad + 加法铃音 + 心跳
  - 元胞自动机：A 小调五声、锯齿 pad + FM 金属铃音 + 粒子脉冲
  - Dijkstra：E 小调五声、拨弦包络 + 三角波合唱 + 木质敲击 + 7/8
  - 相图：F# 多利亚、减法滤波锯齿扫频 + 钢片琴 + 5/4 黄金比间隔
  - 二分查找：C 利底亚、方波 PWM + 玻璃谐波 + 11/8 递减间隔
  - 本片：**C# 全音阶（无半音、无稳定主和弦）**，
           **formant 人声共振 pad + 簧片风琴 + 金属板非谐波共振**，
           **13/8 拍 + 素数间隔音序（2/3/5/7/11 半音跳进）**。

结构（全程不循环、持续演进）：
  - formant pad：16 谐波经时变共振峰（F1/F2/F3 缓慢漂移）加权，像人声在哼唱
  - 簧片风琴：奇次谐波 + 微失谐拍频 + 颤音，沿全音阶游走的长音
  - 金属板：非整数比泛音 [1, 1.87, 3.41, 5.93, 8.21]，按素数间隔跳进触发
  - 13/8 拍微漂移；低频 C#1 呼吸；收尾渐出；无版权依赖

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
ROOT = 138.5913                  # C#3 —— 区别于 D/A/E/F#/C 各前作
SCALE = [0, 2, 4, 6, 8, 10]      # 全音阶：C# D# F G A B（无半音）
SEED = 211121
EIGHTH = 0.196                   # 13/8 一小节 ≈ 2.55s
PRIMES = [2, 3, 5, 7, 11]        # 素数间隔音序

OLD_BGMS = [
    _OLD / "manim_taylor" / "bgm.wav",
    _OLD / "manim_brachistochrone" / "bgm.wav",
    _OLD / "manim_cellular" / "bgm.wav",
    _OLD / "manim_dijkstra" / "bgm.wav",
    _OLD / "manim_phase_portrait" / "bgm.wav",
    _OLD / "manim_binary_search" / "bgm.wav",
]


def make_bgm(duration: float, path: Path) -> None:
    n = int(SR * duration)
    t = np.arange(n) / SR
    audio = np.zeros(n, dtype=np.float64)
    U = t / duration

    # ---- 1) formant 人声 pad：共振峰随进度从「啊」漂移到「哦/咿」 ----
    f0 = ROOT / 2                       # C#2 低哼
    f1 = 480 + 260 * U + 40 * np.sin(2 * np.pi * 0.041 * t)
    f2 = 1180 - 210 * U + 90 * np.sin(2 * np.pi * 0.027 * t + 0.8)
    f3 = 2540 + 180 * np.sin(2 * np.pi * 0.019 * t + 1.6)
    pad = np.zeros(n, dtype=np.float64)
    for k in range(1, 17):
        fk = k * f0
        gaus = (np.exp(-((fk - f1) / 190) ** 2)
                + 0.72 * np.exp(-((fk - f2) / 260) ** 2)
                + 0.38 * np.exp(-((fk - f3) / 400) ** 2))
        amp = (0.55 + 1.35 * gaus) / (k ** 1.15)
        ph = 2 * np.pi * fk * (t + 0.0038 * t * t / duration) + 0.23 * k
        pad += amp * np.sin(ph + 0.18 * np.sin(2 * np.pi * 0.033 * t + k))
    swell = np.clip((t - 0.45) / 4.2, 0, 1)
    audio += 0.030 * swell * pad / 6.0

    # ---- 2) 簧片风琴：奇次谐波 + 微失谐拍频 + 颤音，全音阶长音游走 ----
    rng = np.random.default_rng(SEED)
    degree = 2
    tpos = 0.60
    while tpos < duration - 1.0:
        degree = int(np.clip(degree + rng.choice([-2, -1, 1, 1, 2]), 0, 11))
        semi = SCALE[degree % 6] + 12 * (degree // 6)
        f = ROOT * 2 ** (semi / 12) / 2
        life = 1.35 + 0.55 * rng.random()
        s0, s1 = int(tpos * SR), min(int((tpos + life) * SR), n)
        if s1 <= s0:
            break
        loc = np.arange(s1 - s0) / SR
        env = np.clip(loc / 0.09, 0, 1) * np.clip((life - loc) / 0.55, 0, 1)
        vib = 1.0 + 0.0034 * np.sin(2 * np.pi * 4.6 * loc)
        body = np.zeros_like(loc)
        for k in range(1, 9):
            if k % 2 == 1:
                body += (1.0 / k) * np.sin(2 * np.pi * f * k * vib * loc + 0.3 * k)
        body += 0.45 * np.sin(2 * np.pi * f * 1.006 * vib * loc)   # 拍频
        audio[s0:s1] += 0.036 * env * body
        tpos += life * (0.82 + 0.10 * np.sin(0.37 * tpos))

    # ---- 3) 金属板共振：素数间隔跳进 + 13/8 拍 ----
    tpos, step, deg = 0.85, 0, 0
    while tpos < duration - 0.55:
        s0, s1 = int(tpos * SR), min(int((tpos + 1.05) * SR), n)
        if s1 <= s0:
            break
        loc = np.arange(s1 - s0) / SR
        deg += PRIMES[step % len(PRIMES)]
        semi = SCALE[deg % 6] + 12 * (deg // 6)
        f = ROOT * 2 ** (semi / 12) * 2
        partials = [(1.00, 3.0), (1.87, 4.6), (3.41, 6.2), (5.93, 8.4), (8.21, 11.0)]
        plate = np.zeros_like(loc)
        for ratio, decay in partials:
            plate += np.sin(2 * np.pi * f * ratio * loc) * np.exp(-decay * loc)
        accent = 1.0 if (step % 13) in (0, 3, 7, 11) else 0.42
        audio[s0:s1] += 0.052 * accent * plate * np.clip(loc / 0.002, 0, 1)
        tpos += EIGHTH * (1 + 0.014 * np.sin(0.29 * step)) * (13 / [1, 1, 2, 1, 3][step % 5])
        step += 1

    # ---- 4) 低频呼吸 C#1 ----
    fsub = f0 / 2
    audio += 0.016 * np.clip((t - 0.6) / 5.0, 0, 1) * \
        np.sin(2 * np.pi * fsub * (t + 0.005 * t * t / duration)
               + 0.35 * np.sin(2 * np.pi * 0.023 * t))

    # ---- 5) 收尾与归一化 ----
    rng2 = np.random.default_rng(SEED + 7)
    audio += 0.0016 * rng2.normal(size=n)
    fi, fo = int(0.9 * SR), int(2.0 * SR)
    audio[:fi] *= np.linspace(0, 1, fi)
    audio[-fo:] *= np.linspace(1, 0, fo)
    audio /= max(1e-9, float(np.max(np.abs(audio))))
    audio *= 0.80
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes((audio * 32767).astype("<i2").tobytes())
    print(f"BGM（VORTEX / C#-WholeTone，专用签名）: {path} ({duration:.1f}s)")


def _load_mono(path: Path, limit_s=6.0):
    with wave.open(str(path), "rb") as wf:
        n = min(wf.getnframes(), int(SR * limit_s))
        raw = wf.readframes(n)
    return np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0


def verify(path: Path):
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
        HERE / "media/videos/green_theorem_scene/1350p30/GreenTheorem.mp4"
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else \
        HERE.parent / "green_theorem_final.mp4"
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
