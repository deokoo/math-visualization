# -*- coding: utf-8 -*-
"""BGM for Chaos & Dynamical Systems — C# minor / sync-saw + ring-mod bell
+ reversed tape collage. Signature: COLLAGE-CSharpSync  seed 271127b.
Non-looping, continuously evolving collage events; no metre, no loop.
"""
import math
import os
import subprocess
import numpy as np

SR = 44100
SEED = 2711277
BGM_SIG = "COLLAGE-CSharpSync"
ROOT = 138.59  # C#3

HERE = os.path.dirname(os.path.abspath(__file__))
G = os.path.dirname(HERE)
LIB = os.path.join(G, "2026-11-17-10-23-42", "manim_gaussian")


def adsr(n, a=0.08, d=0.6, s=0.45, r=0.7):
    n = int(n)
    env = np.ones(n, dtype=np.float64)
    tot = a + d + r
    a, d, r = a/tot, d/tot, r/tot
    ai = max(1, int(n*a)); di = max(1, int(n*d)); ri = max(1, int(n*r))
    ai = min(ai, n-2); di = min(di, n-ai-1); ri = min(ri, n-ai-di)
    env[:ai] = np.linspace(0, 1, ai)
    env[ai:ai+di] = np.linspace(1, s, di)
    env[-ri:] = np.linspace(s, 0, ri)
    return env


_R = np.random.default_rng(SEED)


def sync_saw(freq, dur, t0, sr=SR):
    """Sync-saw: two saws phase-locked with a slowly swept ratio -> metallic drive."""
    n = int(dur*sr)
    t = np.arange(n)/sr
    out = np.zeros(n)
    for det, amp in [(-4.0, 0.55), (0.0, 0.75), (5.3, 0.5)]:
        ph = 2*np.pi*(freq*2**(det/1200.0))*t + _R.random()*2*np.pi
        saw = 2*((ph/(2*np.pi)) % 1.0) - 1
        # gentle resonant colouring via a fixed lowpass spectrum
        spec = np.fft.rfft(saw)
        fr = np.fft.rfftfreq(n, 1/sr)
        filt = 1/(1 + (fr/1650.0)**2)
        out += amp*np.fft.irfft(spec*filt, n)
    return out*adsr(n, a=0.4, d=1.0, s=0.55, r=2.6)/2.4


def ring_bell(freq, dur, t0, sr=SR):
    """Ring-modulated bell: carrier*modulator at an incommensurate ratio."""
    n = int(dur*sr)
    t = np.arange(n)/sr
    carrier = np.sin(2*np.pi*freq*t)
    mod = np.sin(2*np.pi*freq*1.4142*t + 0.7)*np.exp(-t*1.4)
    out = carrier*mod
    for h, a in [(2.31, 0.28), (3.87, 0.15)]:
        out += a*np.sin(2*np.pi*freq*h*t)*np.exp(-t*2.2)*np.sin(2*np.pi*freq*1.618*t)
    return out*adsr(n, a=0.006, d=0.25, s=0.2, r=0.7)*0.6


def tape_collage(dur, t0, sr=SR):
    """Reversed, wobbled noise fragment — a tape-splice 'breath' event."""
    n = int(dur*sr)
    t = np.arange(n)/sr
    # band-limited noise
    noise = _R.normal(0, 1, n)
    spec = np.fft.rfft(noise)
    fr = np.fft.rfftfreq(n, 1/sr)
    band = np.exp(-((fr-900)/1100)**2)
    frag = np.fft.irfft(spec*band, n)
    frag = frag[::-1]                                  # reversed splice
    # wow/flutter pitch wobble via variable-rate resample approximation
    wobble = 1 + 0.04*np.sin(2*np.pi*0.6*t) + 0.02*np.sin(2*np.pi*1.7*t)
    env = np.sin(np.pi*np.clip(np.cumsum(wobble)/n, 0, 1))
    return frag*env*0.5


def sub_tick(freq, dur, sr=SR):
    n = int(dur*sr); t = np.arange(n)/sr
    return np.tanh(1.5*np.sin(2*np.pi*freq*t))*np.exp(-t/0.4)*0.4


def safe_add(buf, seg, i0, gain):
    n = len(buf)
    if i0 >= n or i0 + len(seg) <= 0:
        return
    s0 = max(0, -i0); b0 = max(0, i0)
    m = min(len(seg)-s0, n-b0)
    if m > 0:
        buf[b0:b0+m] += seg[s0:s0+m]*gain


def make_bgm(duration, out_path, sr=SR):
    global _R
    _R = np.random.default_rng(SEED)
    rng = np.random.default_rng(SEED+11)
    n = int(duration*sr)
    buf = np.zeros(n)

    # C# natural minor: 0 2 3 5 7 8 10 semitones from C#
    scale = [0, 2, 3, 5, 7, 8, 10]

    # sync-saw chord beds: successive transpositions, never a repeating cycle
    t = 0.0
    ci = 0
    while t < duration:
        deg = scale[(ci*2 + ci//3) % 7]
        f0 = ROOT*2**(deg/12)*2**((ci//7)/12)
        dur = min(11.0, duration-t)
        seg = sync_saw(f0, dur, t)
        safe_add(buf, seg, int(t*sr), 0.34 + 0.22*(0.5+0.5*math.sin(ci*1.31)))
        seg2 = sync_saw(f0*1.3348, dur, t)
        safe_add(buf, seg2, int(t*sr), 0.16 + 0.09*math.cos(ci*0.77))
        t += 6.1 + 0.9*math.sin(ci*0.6)
        ci += 1

    # ring-mod bell collage: non-metric event stream, gap from a wandering walk
    t = 0.25
    k = 0
    gap_state = 0.0
    while t < duration-1.2:
        deg = scale[(k*3 + (k//5)) % 7]
        octv = 12*((k//4) % 3)
        f = ROOT*2**((deg+octv)/12)*2
        dur = min(2.8, duration-t)
        safe_add(buf, ring_bell(f, dur, t), int(t*sr), 0.38)
        # wandering, non-periodic gap (no fixed metre)
        gap_state += 0.37*(rng.random()-0.5)
        gap_state = max(-0.6, min(0.6, gap_state))
        t += 0.44 + 0.22*gap_state + 0.05*(k % 3)
        k += 1

    # reversed tape-collage breaths at asymmetric, drifted boundaries
    t = 1.7
    j = 0
    while t < duration-2.0:
        dur = min(1.6 + 0.3*(j % 4), duration-t)
        safe_add(buf, tape_collage(dur, t), int(t*sr), 0.26 + 0.08*math.sin(j*0.9))
        t += 9.0 + 2.7*math.sin(j*1.13) + rng.random()
        j += 1

    # sparse sub ticks at chaotic-ish intervals
    t = 0.0
    while t < duration:
        safe_add(buf, sub_tick(ROOT/2, 1.0), int(t*sr), 0.45)
        t += 7.0 + 2.0*rng.random()

    fade = np.ones(n)
    fi = int(1.4*sr); fo = int(2.4*sr)
    fade[:fi] = np.linspace(0, 1, fi)
    fade[-fo:] = np.linspace(1, 0, fo)
    buf *= fade

    spec = np.fft.rfft(buf)
    fr = np.fft.rfftfreq(n, 1/sr)
    shelf = 1/(1 + (fr/4800.0)**2)
    out = np.fft.irfft(spec*shelf, n)
    tt = np.arange(n)/sr
    drift = 0.80 + 0.20*np.sin(2*np.pi*0.043*tt + 1.1)*np.sin(2*np.pi*0.117*tt+0.3)
    out *= drift
    out = np.tanh(out/np.max(np.abs(out)+1e-12)*1.7)
    out = out/(np.max(np.abs(out))+1e-12)*0.78

    audio = np.stack([out, out], axis=1)
    pcm = (np.clip(audio, -1, 1)*32767).astype("<i2")
    import wave
    with wave.open(str(out_path), "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return out_path


def verify(path, threshold=0.1):
    import wave
    def load(p):
        with wave.open(str(p), "rb") as w:
            nf = min(w.getnframes(), SR*8)
            nch = w.getnchannels()
            raw = w.readframes(nf)
        x = np.frombuffer(raw, dtype="<i2").astype(np.float64)
        if nch == 2:
            x = x[:nf*2].reshape(-1, 2).mean(axis=1)
        else:
            x = x[:nf]
        return x[::12]
    a = load(path)
    a = a - a.mean(); a /= (np.linalg.norm(a)+1e-12)
    worst = (0.0, "")
    pats = []
    for root_dir in (G, LIB):
        if not os.path.isdir(root_dir):
            continue
        for d in os.listdir(root_dir):
            if d.startswith("manim_") and d != "manim_chaos_dashboard":
                cand = os.path.join(root_dir, d, "bgm.wav")
                if os.path.exists(cand):
                    pats.append(cand)
    pats += [os.path.join(HERE, p) for p in os.listdir(HERE) if p.endswith("_old.wav")]
    for p in pats:
        try:
            b = load(p)
            if b.size < 1000: continue
            m = min(a.size, b.size)
            b = b[:m] - b[:m].mean()
            a2 = a[:m] - a[:m].mean()
            nb = np.linalg.norm(b)+1e-12; na = np.linalg.norm(a2)+1e-12
            c = abs(float(np.dot(a2, b)/(na*nb)))
            if c > worst[0]: worst = (c, os.path.basename(p))
        except Exception:
            pass
    with wave.open(str(path), "rb") as w:
        nch = w.getnchannels()
        raw = w.readframes(w.getnframes())
    full = np.frombuffer(raw, dtype="<i2").astype(np.float64)
    if nch == 2:
        full = full.reshape(-1, 2).mean(axis=1)
    full = full[::12]
    full = full - full.mean()
    corr = np.correlate(full, full, mode="full")
    corr = corr[len(full)-1:]
    den = corr[0]+1e-12
    minlag = int(2.0*SR/12)
    peak = float(np.max(corr[minlag:])/den)
    print("BGM signature:", BGM_SIG)
    print("max |cos| vs library:", round(worst[0], 4), "vs", worst[1])
    print("self-correlation peak:", round(peak, 4))
    ok = worst[0] < threshold and peak < 0.8
    print("VERIFY:", "PASS" if ok else "FAIL")
    return ok


def mix(video, bgm, out):
    import imageio_ffmpeg
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [ffmpeg, "-y", "-loglevel", "error", "-i", video, "-i", bgm,
           "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "128k",
           "-filter:a", "volume=0.85", "-shortest", out]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    import sys
    wav = os.path.join(HERE, "chaos_bgm.wav")
    dur = float(sys.argv[1]) if len(sys.argv) > 1 else 70.0
    make_bgm(dur, wav)
    ok = verify(wav)
    if not ok:
        print("BGM verification failed; not mixing.")
        sys.exit(1)
    video = os.path.join(HERE, "chaos_silent.mp4")
    final = os.path.join(HERE, "chaos_final.mp4")
    mix(video, wav, final)
    print("final:", final, os.path.getsize(final))
