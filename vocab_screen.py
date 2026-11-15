"""Measure what every vocabulary entry does to the sound.

    PY=stable-audio-3/optimized/mlx/.venv/bin/python
    $PY vocab_screen.py jobs      # write the job list
    $PY gen_batch.py --jobs renders/vocab_screen/jobs.jsonl --out renders/vocab_screen
    $PY process_dataset.py renders/vocab_screen
    $PY vocab_screen.py report    # -> vocab_catalogue.csv

Design: for each category and each of SEEDS seeds, render a neutral base prompt ("kick drum")
and the same prompt with exactly one entry added, at the same seed. Everything a clip differs
from its base by is caused by that entry. Filler words ("nice", "sample") give the floor: how
much any extra word shifts the sound.
"""
import csv, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from vocab import BASE, PRE, entries

HERE = Path(__file__).resolve().parent
OUT = HERE / "renders" / "vocab_screen"
SEEDS = [5000, 5001, 5002, 5003]
SR = 44100
PREFIX, SUFFIX = "TrackType: SFX,", "one shot, single hit, no loop, dry, no reverb, isolated"


def prompt(cat, axis=None, text=None):
    if text is None:
        return f"{PREFIX} {BASE[cat]}, {SUFFIX}"
    if axis in PRE:
        return f"{PREFIX} {text} {BASE[cat]}, {SUFFIX}"
    return f"{PREFIX} {BASE[cat]}, {text}, {SUFFIX}"


def jobs():
    out = []
    for cat in BASE:
        for s in SEEDS:
            out.append(dict(id=f"{cat}_base_{s}", category=cat, axis="base", word="", prompt=prompt(cat), seed=s, seconds=1))
    for i, (axis, text, cats) in enumerate(entries()):
        for cat in cats:
            for s in SEEDS:
                out.append(dict(id=f"{cat}_w{i:03d}_{s}", category=cat, axis=axis, word=text,
                                prompt=prompt(cat, axis, text), seed=s, seconds=1))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "jobs.jsonl").write_text("".join(json.dumps(j) + "\n" for j in out))
    print(len(out), "jobs ->", OUT / "jobs.jsonl")


def feats(path):
    from process_dataset import read
    m = read(path)[:, 0]
    seg = m[: SR // 2]
    S = np.abs(np.array([np.fft.rfft(seg[i:i + 1024] * np.hanning(1024)) for i in range(0, len(seg) - 1024, 512)])) ** 2
    f = np.fft.rfftfreq(1024, 1 / SR)
    tot = S.sum(0) + 1e-12
    edges = np.geomspace(30, 20000, 33)
    bands = np.array([S[:, (f >= a) & (f < b)].sum(1) for a, b in zip(edges[:-1], edges[1:])]).T
    emb = np.log10(bands + 1e-8)
    emb = np.maximum(emb, emb.max() - 6).ravel()                      # 60 dB floor
    loud = S[S.sum(1) > 0.01 * S.sum(1).max()] + 1e-12
    flat = np.exp(np.log(loud).mean(1)) / loud.mean(1)                # spectral flatness = noisiness
    return dict(emb=emb,
                bright=np.log2((f * tot).sum() / tot.sum()),          # octaves
                low=tot[f < 120].sum() / tot.sum(),                   # share of energy under 120 Hz
                attack=(seg[:882] ** 2).sum() / (m ** 2).sum(),       # share of energy in first 20 ms
                noisy=float(flat.mean()))


SIG = [("bright", "brighter", "darker"), ("length", "longer", "shorter"), ("low", "more sub", "less sub"),
       ("attack", "snappier", "softer attack"), ("noisy", "noisier", "more tonal")]


def report():
    R = {}
    for l in open(OUT / "clean" / "meta.jsonl"):
        j = json.loads(l)
        j.update(feats(OUT / "clean" / j["file"]))
        j["length"] = np.log2(j["length_ms"])
        R[(j["category"], j["word"], j["seed"])] = j
    rejected = defaultdict(int)
    for l in open(OUT / "clean" / "rejects.jsonl"):
        j = json.loads(l)
        rejected[j["word"]] += 1

    # per (word, category): distance from base and feature shifts, paired by seed
    eff, sig, n_total = defaultdict(list), defaultdict(lambda: defaultdict(list)), defaultdict(int)
    words = {}
    for axis, text, cats in entries():
        words[text] = (axis, cats)
        n_total[text] = len(cats) * len(SEEDS)
        for c in cats:
            for s in SEEDS:
                a, b = R.get((c, text, s)), R.get((c, "", s))
                if a is None or b is None:
                    continue
                eff[(text, c)].append(np.linalg.norm(a["emb"] - b["emb"]))
                for k, _, _ in SIG:
                    sig[text][k].append(a[k] - b[k])
    ctrl = [w for w, (a, _) in words.items() if a == "control"]
    floor = {c: np.mean([np.mean(eff[(w, c)]) for w in ctrl if eff[(w, c)]]) for c in BASE}
    sd0 = {k: np.std([d for w in ctrl for d in sig[w][k]]) + 1e-9 for k, _, _ in SIG}

    rows = []
    for w, (axis, cats) in words.items():
        per = {c: np.mean(eff[(w, c)]) / floor[c] for c in cats if eff[(w, c)]}
        if not per:
            rows.append(dict(word=w, axis=axis, strength=0, does="", reject_pct=100, verdict="unsafe"))
            continue
        does = []
        for k, up, down in SIG:
            d = np.array(sig[w][k])
            if len(d) >= 3 and abs(d.mean()) > 1.5 * sd0[k] and (np.sign(d) == np.sign(d.mean())).mean() >= 0.75:
                does.append(up if d.mean() > 0 else down)
        strength = float(np.mean(list(per.values())))
        rej = 100 * rejected[w] / n_total[w]
        verdict = "control" if axis == "control" else "unsafe" if rej >= 25 else "inert" if strength < 1.25 else \
                  "strong" if strength >= 2 else "works"
        rows.append(dict(word=w, axis=axis, strength=round(strength, 2), does=", ".join(does), reject_pct=round(rej),
                         verdict=verdict, **{f"x_{c}": round(per[c], 2) for c in per}))
    rows.sort(key=lambda r: (r["axis"], -r["strength"]))
    cols = ["word", "axis", "verdict", "strength", "does", "reject_pct"] + [f"x_{c}" for c in BASE]
    with open(HERE / "vocab_catalogue.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, cols)
        w.writeheader()
        w.writerows(rows)
    print("floor (distance a filler word causes):", {c: round(v, 1) for c, v in floor.items()})
    for axis in dict.fromkeys(r["axis"] for r in rows):
        print(f"\n== {axis} ==")
        for r in [r for r in rows if r["axis"] == axis]:
            print(f"  {r['word'][:30]:30} {r['verdict']:7} x{r['strength']:<5} rej {r['reject_pct']:>3}%  {r['does']}")
    from collections import Counter
    print("\nverdicts:", dict(Counter(r["verdict"] for r in rows)))


if __name__ == "__main__":
    {"jobs": jobs, "report": report}[sys.argv[1]]()
