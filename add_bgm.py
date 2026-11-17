#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""为 Manim 成片混入程序化背景音乐（无版权依赖）。

用法：
    python add_bgm.py [input.mp4] [output.mp4]
默认输入 media/videos/gaussian_scene/1364p30/GaussianIntegral.mp4
默认输出 gaussian_integral_manim.mp4
"""
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FFMPEG = HERE / "ffmpeg.exe"
BGM = HERE / "bgm.wav"

SR = 44100


def make_bgm(duration: float, path: Path) -> None:
    n = int(SR * duration)
    audio = np.zeros(n, dtype=np.float64)
    # 和弦进行：Am - F - C - G，柔和铺底
    prog = [(220.00, "min"), (174.61, "maj"),
            (261.63, "maj"), (196.00, "maj")]
    bar = 8.0
    for cycle in range(2):
        for i, (root, kind) in enumerate(prog):
            start = cycle * 4 * bar + i * bar
            if start >= duration:
                break
            end = min(start + bar, duration)
            i0, i1 = int(start * SR), int(end * SR)
            tt = np.arange(i1 - i0) / SR
            intervals = [0, 3, 7] if kind == "min" else [0, 4, 7]
            for semi in intervals:
                f = root * 2 ** (semi / 12)
                for det in (-2.0, 2.0):
                    audio[i0:i1] += 0.042 * np.sin(
                        2 * np.pi * (f + det) * tt + 0.4 * i)
            audio[i0:i1] += 0.085 * np.sin(2 * np.pi * (root / 2) * tt)
            arp = [root * 2 ** (s / 12) * 2 for s in intervals]
            k = 0
            while k * 0.5 < end - start:
                ts = k * 0.5
                s0 = int((start + ts) * SR)
                s1 = min(int((start + ts + 0.44) * SR), n)
                if s1 <= s0:
                    break
                env = np.exp(-np.arange(s1 - s0) / SR * 7.5)
                f = arp[k % 3]
                local = np.arange(s1 - s0) / SR
                audio[s0:s1] += 0.085 * (
                    np.sin(2 * np.pi * f * local) +
                    0.30 * np.sin(4 * np.pi * f * local)) * env
                k += 1
    rng = np.random.default_rng(11)
    audio += 0.0045 * rng.normal(size=n)
    fi, fo = int(2 * SR), int(3 * SR)
    audio[:fi] *= np.linspace(0, 1, fi)
    audio[-fo:] *= np.linspace(1, 0, fo)
    audio /= max(1e-9, float(np.max(np.abs(audio))))
    audio *= 0.82
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
    import re
    m = re.search(r"Duration: (\d+):(\d+):(\d+\.\d+)", out.stderr)
    if not m:
        raise RuntimeError("无法读取视频时长")
    h, mi, s = m.groups()
    return int(h) * 3600 + int(mi) * 60 + float(s)


def main():
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else \
        HERE / "media/videos/gaussian_scene/1364p30/GaussianIntegral.mp4"
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else \
        HERE / "gaussian_integral_manim.mp4"
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
