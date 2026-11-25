#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""为 Manim 成片混入「不循环、持续演进」的程序化背景音乐（无版权依赖）。

沿用每日数学动画系列 BGM 规范（区别于旧的 4 和弦 ×3 循环）：
  1) 底层 pad 三条声部指数滑升，相位解析积分，无周期调制；
  2) 低音心跳间隔逐渐缩短、力度起伏；
  3) 铃音琶音用固定 seed 的不重复音序 + 随机间隔抖动；
  4) 张力层随"逼近最优摆线"进程渐强，16.15s 画面重置处同步呼吸；
  5) 首尾淡入淡出，无重复乐句。

用法：
    python add_bgm.py [input.mp4] [output.mp4]
"""
import re
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
_CANDIDATES = [
    HERE / "ffmpeg.exe",
    HERE.parent / "manim_taylor" / "ffmpeg.exe",
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
RESET_T = 16.15


def make_bgm(duration: float, path: Path) -> None:
    n = int(SR * duration)
    t = np.arange(n) / SR
    audio = np.zeros(n, dtype=np.float64)

    # ---- 1) 持续演进的 pad：三条声部全程缓慢上行 ----
    voices = [(146.83, 0.0, 2.0),    # D3 -> E3
              (146.83, 3.0, 5.0),    # F3 -> G3
              (146.83, 7.0, 9.0)]    # A3 -> B3
    for vi, (base, semi0, semi1) in enumerate(voices):
        f0 = base * 2 ** (semi0 / 12)
        f1 = base * 2 ** (semi1 / 12)
        k = np.log(f1 / f0) / duration
        phase = 2 * np.pi * f0 * (np.exp(k * t) - 1) / k
        for det in (-1.6, 1.6):
            breathe = 0.86 + 0.14 * np.sin(2 * np.pi * (0.09 + 0.02 * vi) * t + vi)
            audio += 0.027 * breathe * np.sin(phase + 2 * np.pi * det * t + vi)

    # ---- 2) 低音心跳：间隔逐渐缩短，力度起伏 ----
    tpos, beat_i = 0.0, 0
    while tpos < duration:
        gap = 1.06 - 0.24 * (tpos / duration)
        s0 = int(tpos * SR)
        s1 = min(int((tpos + 0.30) * SR), n)
        if s1 <= s0:
            break
        local = np.arange(s1 - s0) / SR
        bf = 146.83 * 2 ** ((2.0 * tpos / duration) / 12) / 2
        amp = 0.13 * (0.85 + 0.15 * np.sin(beat_i * 0.7))
        audio[s0:s1] += amp * np.sin(2 * np.pi * bf * local) * np.exp(-local * 8.5)
        tpos += gap
        beat_i += 1

    # ---- 3) 铃音琶音：不重复音高序列 + 不规则间隔 + 随时间渐密 ----
    rng = np.random.default_rng(191119)
    scale = [0, 3, 5, 7, 10]          # D 小调五声音阶
    tpos = 0.8
    while tpos < duration - 0.25:
        degree = int(scale[rng.integers(0, len(scale))]) + \
                 (12 if rng.random() < 0.35 else 0)
        f = 146.83 * 2 ** (degree / 12) * 4.0
        s0 = int(tpos * SR)
        s1 = min(int((tpos + 0.26) * SR), n)
        if s1 <= s0:
            break
        local = np.arange(s1 - s0) / SR
        env = np.exp(-local * 12.0)
        audio[s0:s1] += 0.058 * env * (
            np.sin(2 * np.pi * f * local) +
            0.40 * np.sin(4 * np.pi * f * local) +
            0.16 * np.sin(6 * np.pi * f * local))
        gap = (0.30 - 0.10 * (tpos / duration)) + 0.08 * rng.random()
        tpos += gap

    # ---- 4) 张力层：随收敛渐强，重置处音乐呼吸 ----
    ten = np.sin(2 * np.pi * 440.0 * t + 0.4 * np.sin(2 * np.pi * 0.07 * t))
    ramp = np.clip((t - 2.0) / 12.0, 0, 1)
    audio += 0.016 * ramp * ten
    breath = np.exp(-((t - RESET_T) / 0.32) ** 2)
    audio *= (1.0 - 0.62 * breath)

    # ---- 5) 收尾与归一化 ----
    rng2 = np.random.default_rng(19)
    audio += 0.0030 * rng2.normal(size=n)
    fi, fo = int(1.4 * SR), int(2.2 * SR)
    audio[:fi] *= np.linspace(0, 1, fi)
    audio[-fo:] *= np.linspace(1, 0, fo)
    audio /= max(1e-9, float(np.max(np.abs(audio))))
    audio *= 0.80
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes((audio * 32767).astype("<i2").tobytes())
    print(f"BGM（不循环版）: {path} ({duration:.1f}s)")


def get_duration(video: Path) -> float:
    out = subprocess.run(
        [str(FFMPEG), "-hide_banner", "-i", str(video)],
        capture_output=True, text=True, errors="ignore")
    m = re.search(r"Duration: (\d+):(\d+):(\d+\.\d+)", out.stderr)
    if not m:
        raise RuntimeError("无法读取视频时长")
    h, mi, s = m.groups()
    return int(h) * 3600 + int(mi) * 60 + float(s)


def main():
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else \
        HERE / "media/videos/brachistochrone_scene/1350p30/Brachistochrone.mp4"
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else \
        HERE.parent / "brachistochrone_final.mp4"
    if not src.exists():
        raise SystemExit(f"找不到输入视频: {src}")
    dur = get_duration(src)
    print(f"视频时长: {dur:.2f}s")
    make_bgm(dur, BGM)
    cmd = [str(FFMPEG), "-y", "-loglevel", "error",
           "-i", str(src), "-i", str(BGM),
           "-map", "0:v:0", "-map", "1:a:0",
           "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
           "-shortest", str(dst)]
    subprocess.run(cmd, check=True)
    print(f"完成: {dst} ({dst.stat().st_size / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
