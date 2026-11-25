#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""元胞自动机成片专用配乐 —— 「EMERGENCE / A-minimal」

系列约定：每部动画的音乐必须彼此不同（调性 / 音色 / 结构至少换两样），
本片音乐签名须登记到 github_publish/BGM_REGISTRY.md 防止后续撞车。

本片 vs 已用配乐的区别：
  - 泰勒展开 / 最速降线：D 小调五声、滑升正弦 pad、加法合成铃音、心跳脉冲
  - 本片：A 小调五声（A2=110Hz）、锯齿谐波 pad + 低频滤波扫描、
          FM 金属铃音（调制指数随段变化）、四段式粒子脉冲
          ——分别对应规则 30(混沌) / 90(分形) / 110(复杂) / 184(车流) 的切换点

四段音乐结构（与画面规则切换时刻对齐）：
  SEG1 1.5-5.0s   RULE 30  混沌：不规则密集脉冲、FM 铃音调制指数大
  SEG2 5.0-8.5s   RULE 90  分形：脉冲按 4:2:1 递比、对称音型
  SEG3 8.5-12.0s  RULE 110 复杂：两层复节奏（0.27s × 0.4s）叠加
  SEG4 12.0-16.15 RULE 184 车流：均匀脉冲 + 偶发"堵塞"停顿
  SEG5 16.15-end  循环重演：密度回升、整体收束

全程不循环、无版权依赖。用法：python add_bgm.py [input.mp4] [output.mp4]
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
ROOT = 110.0                      # A2 —— 区别于前作的 D2/D3
SCALE = [0, 3, 5, 7, 10]          # A 小调五声（A C D E G）
SEED = 201119                     # 本片专用种子（泰勒 181118 / 最速降线 191119）
SITE = [1.5, 5.0, 8.5, 12.0, 16.15]   # 画面规则切换点


def seg_at(t):
    for i, s in enumerate(SITE):
        if t < s:
            return max(0, i - 1)
    return 4


def make_bgm(duration: float, path: Path) -> None:
    n = int(SR * duration)
    t = np.arange(n) / SR
    audio = np.zeros(n, dtype=np.float64)

    # ---- 1) 锯齿谐波 pad：A 根音 + E 五度 + C 小三度，缓慢滤波扫描 ----
    # 用有限谐波近似锯齿并做一阶低通感（谐波幅度随频率递减），音色明显不同于正弦 pad
    # 每个声部加极缓慢的独立漂移（互质系数），使波形在 22s 内不精确重复 → 消除准周期
    chord = [(1.0, 0.045, +0.004),   # A2 根音稳住
             (1.5, 0.030, -0.007),   # E3
             (1.2, 0.026, +0.011)]   # C3（小三度）
    for vi, (f0_ratio, amp, drift) in enumerate(chord):
        f0 = ROOT * f0_ratio
        # f(t)=f0*(1+drift*t/T) 的解析相位，避免任何整周期复现
        ph = 2 * np.pi * f0 * (t + 0.5 * drift * t * t / duration)
        sway = 0.90 + 0.10 * np.sin(2 * np.pi * (0.055 + 0.013 * vi) * t + vi)
        for h in range(1, 7):
            audio += amp * sway / (h ** 1.35) * np.sin(h * ph + 0.3 * vi)

    # ---- 2) 四段式粒子脉冲：规则切换处节奏型同步变化 ----
    rng = np.random.default_rng(SEED)
    tpos = 0.55
    pattern_i = 0
    while tpos < duration - 0.2:
        seg = seg_at(tpos)
        if seg == 0:             # 混沌：不规则密集
            gap = 0.17 + 0.15 * rng.random()
        elif seg == 1:           # 分形：4:2:1 递比循环（对称感）
            gap = [0.44, 0.22, 0.11, 0.22][pattern_i % 4]
        elif seg == 2:           # 复杂：两层复节奏
            gap = [0.27, 0.13, 0.40, 0.27][pattern_i % 4]
        elif seg == 3:           # 车流：均匀 + 偶发堵塞停顿
            gap = 0.50 if rng.random() > 0.18 else 0.95
        else:                    # 重演：密度回升
            gap = 0.19 + 0.10 * rng.random()
        pattern_i += 1
        s0 = int(tpos * SR)
        s1 = min(int((tpos + 0.16) * SR), n)
        if s1 <= s0:
            break
        local = np.arange(s1 - s0) / SR
        deg = SCALE[int(rng.integers(0, len(SCALE)))] + (12 if rng.random() < 0.3 else 0)
        pf = ROOT * 2 ** (deg / 12) * 2.0
        pulse = (np.sin(2 * np.pi * pf * local) +
                 0.33 * np.sin(6 * np.pi * pf * local)) * np.exp(-local * 22.0)
        audio[s0:s1] += 0.055 * pulse
        tpos += gap

    # ---- 3) FM 金属铃音：调制指数随段变化，音色区别于前作的加法铃音 ----
    rng2 = np.random.default_rng(SEED + 1)
    tpos = 1.0
    while tpos < duration - 0.3:
        seg = seg_at(tpos)
        carrier_deg = SCALE[int(rng2.integers(0, len(SCALE)))] + 24
        fc = ROOT * 2 ** (carrier_deg / 12)
        fm = 2.4 if seg in (0, 2) else 1.2       # 混沌/复杂段金属感强
        ratio = 1.41 if seg % 2 == 0 else 2.0    # 非整数比 → 金属/玻璃质感
        s0 = int(tpos * SR)
        s1 = min(int((tpos + 0.42) * SR), n)
        if s1 <= s0:
            break
        local = np.arange(s1 - s0) / SR
        env = np.exp(-local * (7.5 if seg in (1, 3) else 5.0))
        mod = fm * np.sin(2 * np.pi * fc * ratio * local) * np.exp(-local * 6.0)
        audio[s0:s1] += 0.042 * env * np.sin(2 * np.pi * fc * local + mod)
        tpos += (0.62 - 0.10 * seg) + 0.30 * rng2.random()

    # ---- 4) 低频呼吸层 + 规则切换点的短促"相位重置"（加漂移防准周期） ----
    fsub = ROOT / 2
    sub_ph = 2 * np.pi * fsub * (t + 0.5 * (-0.009) * t * t / duration)
    sub = np.sin(sub_ph + 0.7 * np.sin(2 * np.pi * 0.045 * t))
    audio += 0.020 * np.clip((t - 1.0) / 8.0, 0, 1) * sub
    for st in SITE[1:]:
        dip = np.exp(-((t - st) / 0.14) ** 2)
        audio *= (1.0 - 0.38 * dip)

    # ---- 5) 收尾与归一化 ----
    rng3 = np.random.default_rng(119)
    audio += 0.0025 * rng3.normal(size=n)
    fi, fo = int(1.2 * SR), int(2.0 * SR)
    audio[:fi] *= np.linspace(0, 1, fi)
    audio[-fo:] *= np.linspace(1, 0, fo)
    audio /= max(1e-9, float(np.max(np.abs(audio))))
    audio *= 0.80
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes((audio * 32767).astype("<i2").tobytes())
    print(f"BGM（EMERGENCE / A-minimal，专用签名）: {path} ({duration:.1f}s)")


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
        HERE / "media/videos/cellular_scene/1350p30/CellularAutomata.mp4"
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else \
        HERE.parent / "cellular_automata_final.mp4"
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
