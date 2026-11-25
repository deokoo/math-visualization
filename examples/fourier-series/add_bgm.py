#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""为 Manim 成片混入程序化背景音乐（无版权依赖）。

用法：
    python add_bgm.py [input.mp4] [output.mp4]
默认输入 media/videos/complex_exp_scene/1350p30/ComplexExponential.mp4
默认输出 complex_exponential_final.mp4
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
    HERE.parent.parent / "manim_gaussian" / "ffmpeg.exe",
]
FFMPEG = next((p for p in _CANDIDATES if p.exists()), Path("ffmpeg"))
BGM = HERE / "bgm.wav"

SR = 44100


def make_bgm(duration: float, path: Path) -> None:
    n = int(SR * duration)
    audio = np.zeros(n, dtype=np.float64)
    # 新风格：明亮科技脉冲 —— Dm - Bb - F - C，4 秒一换，低音心跳 + 铃音琶音
    prog = [(146.83, "min"), (116.54, "maj"),
            (174.61, "maj"), (130.81, "maj")]
    bar = 4.0
    for cycle in range(3):
        for i, (root, kind) in enumerate(prog):
            start = cycle * 4 * bar + i * bar
            if start >= duration:
                break
            end = min(start + bar, duration)
            i0, i1 = int(start * SR), int(end * SR)
            tt = np.arange(i1 - i0) / SR
            intervals = [0, 3, 7] if kind == "min" else [0, 4, 7]

            # 柔和 pad 铺底
            for semi in intervals:
                f = root * 2 ** (semi / 12)
                for det in (-1.5, 1.5):
                    audio[i0:i1] += 0.030 * np.sin(
                        2 * np.pi * (f + det) * tt + 0.3 * i)

            # 低音脉冲：每 1 秒一次，像心跳一样推动画面
            t = 0.0
            while t < end - start:
                s0 = int((start + t) * SR)
                s1 = min(int((start + t + 0.28) * SR), n)
                if s1 <= s0:
                    break
                local = np.arange(s1 - s0) / SR
                audio[s0:s1] += 0.15 * np.sin(
                    2 * np.pi * (root / 2) * local) * np.exp(-local * 9.0)
                t += 1.0

            # 铃音琶音：每 0.25 秒一颗，带二次/三次泛音
            k = 0
            while k * 0.25 < end - start:
                ts = k * 0.25
                s0 = int((start + ts) * SR)
                s1 = min(int((start + ts + 0.22) * SR), n)
                if s1 <= s0:
                    break
                local = np.arange(s1 - s0) / SR
                f = root * 2 ** (intervals[k % 3] / 12) * 4
                env = np.exp(-local * 13.0)
                audio[s0:s1] += 0.075 * (
                    np.sin(2 * np.pi * f * local) +
                    0.45 * np.sin(4 * np.pi * f * local) +
                    0.18 * np.sin(6 * np.pi * f * local)) * env
                k += 1

    rng = np.random.default_rng(23)
    audio += 0.0035 * rng.normal(size=n)
    fi, fo = int(1.5 * SR), int(2.5 * SR)
    audio[:fi] *= np.linspace(0, 1, fi)
    audio[-fo:] *= np.linspace(1, 0, fo)
    audio /= max(1e-9, float(np.max(np.abs(audio))))
    audio *= 0.80
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes((audio * 32767).astype("<i2").tobytes())
    print(f"BGM: {path} ({duration:.1f}s)")


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
        HERE / "media/videos/fourier_scene/1350p30/FourierSeries.mp4"
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else \
        HERE / "fourier_series_final.mp4"
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
