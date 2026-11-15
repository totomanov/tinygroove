"""Milestone 2: turn raw teacher renders into clean training clips.

    uv run --with numpy python -m tinygroove.data.process_dataset renders/dataset_v4

Reads  <dir>/meta.jsonl and the raw wavs (left untouched).
Writes <dir>/clean/<category>/<id>.wav   mono, 44.1 kHz, 16-bit, exactly 1 s
       <dir>/clean/meta.jsonl            kept clips (raw metadata + length_ms)
       <dir>/clean/rejects.jsonl         dropped clips with the reason

Per clip: drop if clipped / silent / more than one hit; mix to mono; trim so the hit starts
at time zero; cut the noise floor after the tail with a short fade; peak-normalize to -1 dBFS;
pad to 1 s. No EQ, compression or denoising.
"""
import json, sys, wave
from collections import Counter
from pathlib import Path
import numpy as np

SR, LEN = 44100, 44100
PEAK = 10 ** (-1 / 20)


def read(path):
    with wave.open(str(path)) as w:
        x = np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float32) / 32768
        return x.reshape(-1, w.getnchannels())


def hits(m):
    """Count separate hits: a fresh >=10 dB jump in a 10 ms envelope, >=80 ms after the last one,
    reaching within 20 dB of the peak. Flams inside a clap (<80 ms apart) count as one hit."""
    hop, win = 441, 1764      # 40 ms window: longer than a sub-bass cycle, so 808 tails don't ripple
    env = np.sqrt(np.convolve(m ** 2, np.ones(win) / win, "same")[::hop] + 1e-12)
    db = 20 * np.log10(env / env.max())
    n, last = 1, 0
    for i in range(3, len(db)):
        if db[i] - db[i - 3] >= 10 and db[i] > -20 and i - last >= 8 and i >= 8:
            n += 1
        if db[i] - db[i - 3] >= 10 and db[i] > -20:
            last = i
    return n


def process(x):
    if (np.abs(x) >= 0.999).sum() > 8:
        return None, "clipped"
    m = x.mean(axis=1)
    peak = np.abs(m).max()
    if peak < 0.01:
        return None, "silent"
    on = max(0, int(np.argmax(np.abs(m) > 0.02 * peak)) - 44)     # keep 1 ms before the attack
    m = m[on:]
    if hits(m) > 1:
        return None, "multi-hit"
    loud = np.where(np.abs(m) > 0.001 * peak)[0]                   # last sample above -60 dB
    end = min(len(m), loud[-1] + int(0.02 * SR))
    m = m[:end].copy()
    f = min(len(m), int(0.02 * SR))
    m[-f:] *= np.linspace(1, 0, f) ** 2                            # fade the tail to zero
    if len(m) < 0.02 * SR:
        return None, "too-short"
    m = m[:LEN] * (PEAK / np.abs(m).max())
    return np.pad(m, (0, LEN - len(m))), round(1000 * end / SR)


def main():
    d = Path(sys.argv[1])
    out = d / "clean"
    out.mkdir(exist_ok=True)
    kept, rej = [], []
    for line in open(d / "meta.jsonl"):
        j = json.loads(line)
        y, info = process(read(d / j["file"]))
        if y is None:
            rej.append({**j, "reason": info})
            continue
        p = out / j["file"]
        p.parent.mkdir(exist_ok=True)
        with wave.open(str(p), "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
            w.writeframes((y * 32767).astype("<i2").tobytes())
        kept.append({**j, "length_ms": info})
    (out / "meta.jsonl").write_text("".join(json.dumps(k) + "\n" for k in kept))
    (out / "rejects.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rej))

    total = Counter(j["category"] for j in kept + rej)
    bad = Counter((r["category"], r["reason"]) for r in rej)
    print(f"{'category':12} kept  clipped silent multi-hit too-short  median length")
    for c in total:
        k = [x["length_ms"] for x in kept if x["category"] == c]
        med = f"{sorted(k)[len(k) // 2]} ms" if k else "-"
        print(f"{c:12} {len(k):3}/{total[c]:<3} " + " ".join(
            f"{bad[(c, r)]:^{len(r)}}" for r in ["clipped", "silent", "multi-hit", "too-short"]) + f"   {med}")
    print(f"\nkept {len(kept)}/{len(kept) + len(rej)} -> {out}")


if __name__ == "__main__":
    main()
