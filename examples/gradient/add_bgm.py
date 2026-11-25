#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""梯度与方向导数成片专用配乐 —— 「CONTOUR / G-Hijaz」

系列铁律：每部动画配乐必须彼此不同（调性 / 音色 / 结构至少换两样）。
本片 vs 已用配乐（三样全换）：
  - 泰勒 / 最速降线：D 小调五声、滑升正弦 pad + 加法铃音 + 心跳
  - 元胞自动机：A 小调五声、锯齿 pad + FM 金属铃音 + 粒子脉冲
  - Dijkstra：E 小调五声、拨弦包络 + 三角波合唱 + 7/8
  - 相图：F# 多利亚、减法锯齿扫频 + 钢片琴 + 5/4 黄金比
  - 二分查找：C 利底亚、方波 PWM + 玻璃谐波 + 11/8 递减间隔
  - 格林公式：C# 全音阶、formant 人声 + 簧片风琴 + 金属板 + 13/8 素数间隔
  - 本片：**G 阿拉伯音阶 Hijaz（增二度色彩）**，
           **特雷门琴连续滑音 + 钢鼓非谐波敲击 + 磁带嘶声/抖晃**，
           **17/8 拍 + 斐波那契间隔音序（1/1/2/3/5/8 半音跳进）**。

结构（全程不循环、持续演进）：
  - 特雷门琴：沿 Hijaz 音级连续滑音（无阶跃），5.3Hz 颤音 + 呼吸包络
  - 钢鼓：[1, 1.99, 2.98, 4.12, 5.43] 非整数泛音，起音微降滑，指数衰减
  - 磁带层：平滑噪声 + 两个不可公度调制频率造成的 wow/flutter
  - G1 低频呼吸；17/8 微漂移；首尾淡入淡出；无版权依赖

用法：python add_bgm.py [input.mp4] [output.mp4]
"""
import re
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
_WS = HERE.parent
_CANDIDATES = [
    HERE / "ffmpeg.exe",
    _WS / "manim_green_theorem" / "ffmpeg.exe",
    _WS / "manim_gaussian" / "ffmpeg.exe",
    Path("C:/Users/admin/WorkBuddy/2026-11-17-10-23-42/manim_gaussian/ffmpeg.exe"),
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
ROOT = 98.0                          # G2 —— 区别于 D/A/E/F#/C/C# 各前作
SCALE = [0, 1, 4, 5, 7, 8, 10]       # G Hijaz: G Ab B C D Eb F
SEED = 221122
STEP = 0.178                         # 17/8 拍基准音距
FIB = [1, 1, 2, 3, 5, 8]             # 斐波那契间隔音序

OLD_BGMS = [
    _WS / "manim_taylor" / "bgm.wav",
    _WS / "manim_brachistochrone" / "bgm.wav",
    _WS / "manim_cellular" / "bgm.wav",
    _WS / "manim_dijkstra" / "bgm.wav",
    _WS / "manim_phase_portrait" / "bgm.wav",
    _WS / "manim_binary_search" / "bgm.wav",
    _WS / "manim_green_theorem" / "bgm.wav",
]


def make_bgm(duration: float, path: Path) -> None:
    n = int(SR * duration)
    t = np.arange(n) / SR
    audio = np.zeros(n, dtype=np.float64)
    rng = np.random.default_rng(SEED)

    # ---- 1) 特雷门琴：Hijaz 音级间连续滑音（相位积分，无爆音） ----
    seg_t = [0.55]
    seg_d = [2]
    d = 2
    while seg_t[-1] < duration + 1.0:
        d = int(np.clip(d + rng.choice([-2, -1, 1, 1, 2]), 0, 6))
        seg_d.append(d)
        seg_t.append(seg_t[-1] + 0.85 + 0.55 * rng.random())
    semi = np.zeros(n, dtype=np.float64)
    for i in range(len(seg_t) - 1):
        s0, s1 = int(seg_t[i] * SR), min(int(seg_t[i + 1] * SR), n)
        if s1 <= s0:
            break
        w = np.linspace(0, 1, s1 - s0)
        w = w * w * (3 - 2 * w)
        semi[s0:s1] = SCALE[seg_d[i]] * (1 - w) + SCALE[seg_d[i + 1]] * w
    vib = 1.0 + 0.0042 * np.sin(2 * np.pi * 5.3 * t)
    freq = ROOT * 2.0 ** (semi / 12.0) * 2.0 * vib      # G3 附近
    phase = 2 * np.pi * np.cumsum(freq) / SR
    ther = (np.sin(phase) + 0.26 * np.sin(2 * phase + 0.35)
            + 0.07 * np.sin(3 * phase + 0.9))
    env1 = np.clip((t - 0.5) / 3.4, 0, 1) * \
        (0.66 + 0.34 * np.sin(2 * np.pi * 0.037 * t + 0.7))
    audio += 0.034 * env1 * ther

    # ---- 2) 钢鼓：斐波那契间隔 + 非谐波泛音 + 17/8 微漂移 ----
    partials = [(1.00, 3.0), (1.99, 4.0), (2.98, 5.1),
                (4.12, 6.4), (5.43, 7.8)]
    tpos, step, deg = 0.80, 0, 0
    while tpos < duration - 0.45:
        deg += FIB[step % len(FIB)]
        sm = SCALE[deg % 7] + 12 * (deg // 7)
        f = ROOT * 2.0 ** (sm / 12.0) * 4.0
        life = 1.25
        s0, s1 = int(tpos * SR), min(int((tpos + life) * SR), n)
        if s1 <= s0:
            break
        loc = np.arange(s1 - s0) / SR
        hit = np.zeros_like(loc)
        for ratio, decay in partials:
            ff = f * ratio * (1.0 + 0.010 * np.exp(-loc * 20.0))
            hit += np.sin(2 * np.pi * ff * loc) * np.exp(-decay * loc)
        accent = 1.0 if (step % 17) in (0, 4, 9, 14) else 0.46
        audio[s0:s1] += 0.055 * accent * hit * np.clip(loc / 0.003, 0, 1)
        tpos += STEP * (1.0 + 0.055 * np.sin(0.137 * step + 0.4)
                        + 0.028 * np.sin(0.071 * step + 1.9))
        step += 1

    # ---- 3) 磁带嘶声 + wow/flutter（两个不可公度调制） ----
    hiss = rng.normal(size=n)
    hiss_s = np.convolve(hiss, np.ones(26) / 26.0, mode="same")
    wow = (0.5 + 0.5 * np.sin(2 * np.pi * 0.058 * t + 1.1)
           * np.sin(2 * np.pi * 0.013 * t + 0.3))
    audio += 0.0022 * (0.72 * hiss_s + 0.28 * hiss) * (0.35 + 0.65 * wow)

    # ---- 4) G1 低频呼吸 ----
    audio += 0.013 * np.clip((t - 0.7) / 4.8, 0, 1) * \
        np.sin(2 * np.pi * (ROOT / 2) * (t + 0.004 * t * t / duration)
               + 0.32 * np.sin(2 * np.pi * 0.021 * t))

    # ---- 5) 收尾与归一化 ----
    audio += 0.0012 * rng.normal(size=n)
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
    print(f"BGM（CONTOUR / G-Hijaz，专用签名）: {path} ({duration:.1f}s)")


def _load_mono(path: Path, limit_s=6.0):
    with wave.open(str(path), "rb") as wf:
        n = min(wf.getnframes(), int(SR * limit_s))
        raw = wf.readframes(n)
    return np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0


def verify(path: Path):
    step = 8
    x = _load_mono(path)[::step]
    print("\n[配乐差异检验]")
    worst = 0.0
    for old in OLD_BGMS:
        if not old.exists():
            continue
        y = _load_mono(old)[::step]
        m = min(len(x), len(y))
        a, b = x[:m] - x[:m].mean(), y[:m] - y[:m].mean()
        cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
        worst = max(worst, abs(cos))
        print(f"  vs {old.parent.name:22s}: {cos:+.4f}")
    print(f"  最大绝对相似度: {worst:.4f}  → {'通过' if worst < 0.1 else '需重做'}")
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
        HERE / "media/videos/gradient_scene/1350p30/Gradient.mp4"
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else \
        HERE.parent / "gradient_final.mp4"
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
