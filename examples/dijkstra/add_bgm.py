#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dijkstra 最短路成片专用配乐 —— 「LATTICE / E-minor」

系列铁律：每部动画的音乐必须彼此不同（调性 / 音色 / 结构至少换两样）。
本片音乐签名须登记到 github_publish/BGM_REGISTRY.md 防止撞车。

本片 vs 已用配乐的区别（三样全换）：
  - 泰勒 / 最速降线：D 小调五声、滑升正弦 pad + 加法铃音 + 心跳
  - 元胞自动机：A 小调五声、锯齿谐波 pad + FM 金属铃音 + 粒子脉冲
  - 本片：E 小调五声（E2=82.41Hz）、**拨弦(plucked)包络 + 三角波合唱 +
          短促木质敲击**，节奏结构为 **7/8 奇数拍逐跳递进**
          ——每个"跳"(松弛步)一记拨弦，随 |S| 锁定数增加音高阶梯上行

结构（全程不循环、持续演进）：
  - 拨弦音序：7/8 拍（7 个八分音符为一小节），每拍触发一记拨弦，
    音高沿 E 小调五声按随机游走上行，模拟最短路"逐跳松弛"；无重复乐句
  - 三角波合唱层：三条极缓慢上行的三角波，音量随进度渐强（张力累积）
  - 木质敲击：在"锁定新节点"的近似时刻落下短促泛音敲击
  - 低频 E 根音以呼吸式包络铺底，带独立漂移防准周期
  - 收尾渐出；无版权依赖

用法：python add_bgm.py [input.mp4] [output.mp4]
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
    HERE.parent / "manim_gaussian" / "ffmpeg.exe",
    HERE.parent.parent / "2026-11-17-10-23-42" / "manim_gaussian" / "ffmpeg.exe",
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
ROOT = 82.41                      # E2 —— 区别于前作 D2/D3、A2
SCALE = [0, 3, 5, 7, 10]          # E 小调五声（E G A B D）
SEED = 221119                     # 本片专用种子
EIGHTH = 0.128                    # 一个八分音符时长 → 7/8 小节 ≈ 0.896s
BAR = 7 * EIGHTH


def make_bgm(duration: float, path: Path) -> None:
    n = int(SR * duration)
    t = np.arange(n) / SR
    audio = np.zeros(n, dtype=np.float64)

    # ---- 1) 拨弦音序：7/8 奇数拍，每拍一记拨弦，随机游走上行 ----
    # Karplus-Strong 风格：短噪声激励 + 快速衰减 + 少量高次谐波，木质拨弦感
    rng = np.random.default_rng(SEED)
    step = 0
    tpos = 0.85
    degree = 0
    while tpos < duration - 0.25:
        # 7/8 拍：每小节第 1 拍略重，其余轻
        beat_in_bar = step % 7
        accent = 1.0 if beat_in_bar == 0 else (0.62 if beat_in_bar in (3, 5) else 0.42)
        # 音高随机游走（偏向阶梯上行，模拟最短路距离层层确定）
        move = rng.choice([-1, 0, 0, 1, 1, 2])
        degree = int(np.clip(degree + move, 0, 14))
        semi = SCALE[degree % len(SCALE)] + 12 * (degree // len(SCALE))
        f0 = ROOT * 2 ** (semi / 12) * 2.0     # 拨弦在中高音区
        s0 = int(tpos * SR)
        life = 0.42 + 0.10 * rng.random()
        s1 = min(int((tpos + life) * SR), n)
        if s1 <= s0:
            break
        local = np.arange(s1 - s0) / SR
        # 激励噪声（前 8ms）+ 谐波衰减体，包络 exp 快衰减 → 拨弦音色
        env = np.exp(-local * (9.0 + 3.0 * degree / 14))
        body = (np.sin(2 * np.pi * f0 * local) +
                0.5 * np.sin(2 * np.pi * 2 * f0 * local) * np.exp(-local * 8) +
                0.25 * np.sin(2 * np.pi * 3 * f0 * local) * np.exp(-local * 14))
        attack = np.clip(local / 0.006, 0, 1)
        audio[s0:s1] += 0.075 * accent * env * body * attack
        step += 1
        tpos += EIGHTH

    # ---- 2) 三角波合唱层：三条缓慢上行，音量随进度渐强 ----
    tri_voices = [(1.0, 0.020, +0.005), (1.498, 0.015, -0.008), (2.0, 0.012, +0.011)]
    for vi, (ratio, amp, drift) in enumerate(tri_voices):
        f0 = ROOT * ratio
        ph = 2 * np.pi * f0 * (t + 0.5 * drift * t * t / duration)
        # 三角波：asin(sin) 近似，区别于正弦 pad 与锯齿 pad
        tri = (2 / np.pi) * np.arcsin(np.sin(ph + 0.5 * vi))
        swell = np.clip((t - 2.0) / 12.0, 0, 1) * (0.85 + 0.15 * np.sin(2 * np.pi * 0.06 * t + vi))
        audio += amp * swell * tri

    # ---- 3) 木质敲击：近似"锁定新节点"时刻的短促泛音 ----
    # 8 次锁定（含源点），间隔逐渐拉长（越到后面节点越少、越"确定"）
    lock_times = np.linspace(1.4, 15.2, 8)
    for k, lt in enumerate(lock_times):
        s0 = int(lt * SR)
        s1 = min(int((lt + 0.20) * SR), n)
        if s1 <= s0:
            continue
        local = np.arange(s1 - s0) / SR
        fk = ROOT * 4 * 2 ** (SCALE[k % len(SCALE)] / 12)
        click = np.sin(2 * np.pi * fk * local) * np.exp(-local * 30)
        click += 0.4 * np.sin(2 * np.pi * 2.76 * fk * local) * np.exp(-local * 45)
        audio[s0:s1] += 0.055 * click

    # ---- 4) 低频呼吸底：E 根音，带独立漂移防准周期 ----
    fsub = ROOT / 2
    sub_ph = 2 * np.pi * fsub * (t + 0.5 * (-0.007) * t * t / duration)
    sub = np.sin(sub_ph + 0.5 * np.sin(2 * np.pi * 0.041 * t))
    audio += 0.018 * np.clip((t - 1.0) / 7.0, 0, 1) * sub

    # ---- 5) 收尾与归一化 ----
    rng2 = np.random.default_rng(SEED + 7)
    audio += 0.0022 * rng2.normal(size=n)
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
    print(f"BGM（LATTICE / E-minor，专用签名）: {path} ({duration:.1f}s)")


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
        HERE / "media/videos/dijkstra_scene/1350p30/Dijkstra.mp4"
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else \
        HERE.parent / "dijkstra_final.mp4"
    if not src.exists():
        raise SystemExit(f"找不到输入视频: {src}")
    dur = get_duration(src)
    print(f"视频时长: {dur:.2f}s")
    make_bgm(dur, BGM)
    cmd = [str(FFMPEG), "-y", "-loglevel", "error",
           "-i", str(src), "-i", str(BGM),
           "-map", "0:v:0", "-map", "1:a:0",
           "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", str(dst)]
    subprocess.run(cmd, check=True)
    print(f"完成: {dst} ({dst.stat().st_size / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
