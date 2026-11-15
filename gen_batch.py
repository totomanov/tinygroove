"""Milestone 1: in-process Stable Audio 3 batch generator for drum one-shots.

Loads the text encoder, DiT and decoder once and loops over a prompt grid, instead of
reloading everything per clip like ./sa3 does.

    stable-audio-3/optimized/mlx/.venv/bin/python gen_batch.py --n 500 --out renders/dataset_v0
    ... --only kick,clap      # subset of categories
    ... --n 50000             # extends the same plan: clips already rendered are kept and skipped
    ... --list                # print the planned prompts, render nothing

Resumable: clips already on disk are skipped. Output:
    <out>/<category>/<category>_<index>.wav     44.1 kHz stereo
    <out>/meta.jsonl                            category, tags, prompt, seed, seconds per clip
"""
import argparse, json, math, os, random, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SA3 = HERE / "stable-audio-3" / "optimized" / "mlx"

# category: (seconds, weight, [phrasings]). Weight = share of the dataset; core house sounds get more.
# Every clip is 1 s. A phrasing can be (text, seconds) to override that; unused for now.
CATEGORIES = {
    "kick":       (1, 6, ["kick drum", "house kick drum", "bass drum hit", "four-on-the-floor kick",
                          "tight short kick drum", "punchy drum machine kick",
                          "bass drum, long sub boom", "deep sub kick with long tail",
                          "kick drum, long sine sub decay"]),
    "clap":       (1, 3, ["hand clap", "house clap", "layered clap", "drum machine clap"]),
    "snare":      (1, 3, ["snare drum hit", "house snare", "snappy snare drum"]),
    "hat_closed": (1, 4, ["closed hi-hat", "tight closed hi-hat tick", "short closed hat"]),
    "hat_open":   (1, 3, ["open hi-hat cymbal, metallic ringing", "open hi-hat, shimmering metallic decay",
                          "offbeat open hi-hat, bright metallic ring"]),
    "rim":        (1, 1, ["rimshot", "side stick click", "rim click"]),
    "snap":       (1, 1, ["finger snap", "single finger snap, crisp click"]),
    "shaker":     (1, 1, ["single shaker shake", "single maraca hit"]),
    "tambourine": (1, 1, ["tambourine hit", "single tambourine shake, jingles"]),
    "ride":       (1, 1, ["ride cymbal hit", "ride cymbal ping", "ride bell hit"]),
    "crash":      (1, 1, ["crash cymbal hit", "splash cymbal hit"]),
    "tom":        (1, 1, ["tom drum hit", "low tom", "high tom", "electronic tom"]),
    "conga":      (1, 1, ["conga slap", "conga open tone", "low conga hit"]),
    "bongo":      (1, 1, ["bongo hit", "high bongo slap", "timbale hit"]),
    "cowbell":    (1, 1, ["cowbell hit", "drum machine cowbell"]),
    "clave":      (1, 1, ["clave hit", "woodblock hit"]),
}
CLEAN_METAL = {"hat_closed", "hat_open", "ride", "crash"}   # no lo-fi / saturated: they turn to noise
# --- vocabulary: only entries that vocab_screen.py measured as working (see vocab_catalogue.csv) ---
PRE_AXES = ("source", "genre")
POST_AXES = ("processing", "transient", "body", "tail", "tone", "size", "texture", "pitch",
             "onomatopoeia", "performance")
# kept off hats and cymbals: these turn metal into hiss
NOISY = {"sampled from vinyl", "lo-fi", "noisy", "analog warmth", "high-pass filtered", "dusty", "crackly",
         "noisy tail", "distorted", "overdriven", "bitcrushed", "fuzzy", "dirty", "trashy", "gritty"}
PREFIX = "TrackType: SFX,"
SUFFIX = "one shot, single hit, no loop, dry, no reverb, isolated"
P_LONG = 0.0          # every clip is 1 s (a deliberate choice). >0 would render that share at 2 s for longer tails
P_MATCHED = 0.03      # chance a slot starts a matched set (-> about 15% of clips)


def load_vocab(cat):
    """{axis: [(text, weight)]} of proven entries for one category. Weight = measured strength."""
    import csv
    from vocab import entries
    applies = {text: cats for _, text, cats in entries()}
    core = cat in ("kick", "clap", "snare", "hat_closed", "hat_open")
    out = {}
    for r in csv.DictReader(open(HERE / "vocab_catalogue.csv")):
        w = r["word"]
        if r["verdict"] not in ("works", "strong") or (cat in CLEAN_METAL and w in NOISY):
            continue
        if core:
            if cat not in applies[w] or float(r[f"x_{cat}"] or 0) < 1.25:
                continue
            strength = float(r[f"x_{cat}"])
        else:                                   # other instruments: only the general-purpose entries
            if len(applies[w]) < 5:
                continue
            strength = float(r["strength"])
        out.setdefault(r["axis"], []).append((w, strength))
    return out


def _prompt(phrase, pre=(), post=()):
    return ", ".join([f"{PREFIX} {' '.join([*pre, phrase])}", *post, SUFFIX])


def _stream(cat, grid_seed):
    """Endless, deterministic stream of jobs for one category. Each category has its own RNG, so
    asking for more clips later extends the stream without changing the earlier ones."""
    rng = random.Random(f"{grid_seed}/{cat}")
    base_secs, _, phrases = CATEGORIES[cat]
    V = load_vocab(cat)
    post_axes = [a for a in POST_AXES if a in V]
    pick = lambda axis: rng.choices([w for w, _ in V[axis]], [s for _, s in V[axis]])[0]
    seen, k, fam = set(), 0, 0

    def job(prompt, secs, seed, **tags):
        nonlocal k
        k += 1
        return dict(id=f"{cat}_{k - 1:05d}", category=cat, prompt=prompt, seed=seed, seconds=secs, **tags)

    while True:
        phrase = rng.choice(phrases)
        secs = base_secs
        if not isinstance(phrase, str):
            phrase, secs = phrase
        elif base_secs == 1 and rng.random() < P_LONG:
            secs = 2
        seed = rng.randrange(2**31)
        if rng.random() < P_MATCHED:        # matched set: same seed, one entry added per clip
            fam += 1
            tags = dict(matched=f"{cat}_m{fam:04d}")
            yield job(_prompt(phrase), secs, seed, words={}, **tags)
            for axis in rng.sample([*PRE_AXES, *post_axes], 5):
                w = pick(axis)
                yield job(_prompt(phrase, pre=[w]) if axis in PRE_AXES else _prompt(phrase, post=[w]),
                          secs, seed, words={axis: w}, **tags)
            continue
        style = rng.choice(["none", "source", "source", "genre", "genre", "both"])
        words = {a: pick(a) for a in PRE_AXES if style in (a, "both")}
        for axis in rng.sample(post_axes, rng.choice([1, 1, 2, 2, 3])):
            words[axis] = pick(axis)
        prompt = _prompt(phrase, pre=[words[a] for a in PRE_AXES if a in words],
                         post=[words[a] for a in words if a not in PRE_AXES])
        if (prompt, secs) in seen:           # every clip gets its own prompt
            continue
        seen.add((prompt, secs))
        yield job(prompt, secs, seed, words=words)


def plan(n, only, grid_seed):
    """Deterministic job list. Category shares follow the weights in CATEGORIES."""
    cats = [c for c in CATEGORIES if not only or c in only]
    total = sum(CATEGORIES[c][1] for c in cats)
    jobs = []
    for c in cats:
        gen = _stream(c, grid_seed)
        jobs += [next(gen) for _ in range(round(n * CATEGORIES[c][1] / total))]
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--out", default=str(HERE / "renders" / "dataset_v3"))
    ap.add_argument("--only", default="")
    ap.add_argument("--dit", default="sm-sfx", choices=["sm-sfx", "medium"])
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--grid-seed", type=int, default=0)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--jobs", default="", help="render an explicit job list (jsonl) instead of the grid")
    a = ap.parse_args()

    jobs = plan(a.n, set(filter(None, a.only.split(","))), a.grid_seed)
    if a.jobs:
        jobs = [json.loads(l) for l in open(a.jobs)]
    if a.list:
        for j in jobs:
            print(j["id"], "|", j["prompt"])
        return
    out = Path(a.out).resolve()
    todo = [j for j in jobs if not (out / j["category"] / f"{j['id']}.wav").exists()]
    print(f"{len(jobs)} planned, {len(jobs) - len(todo)} already done, {len(todo)} to render")
    if not todo:
        return

    # --- SA3 runtime (imported late so --list works with any python) ---
    os.chdir(SA3)
    sys.path.insert(0, str(SA3 / "scripts"))
    import numpy as np
    import mlx.core as mx
    import sa3_mlx as S

    decoder_name = "same-s" if a.dit == "sm-sfx" else "same-l"
    dtype = mx.float16
    t_load = time.time()
    enc = S.T5Gemma.from_npz(str(S.ensure_local(S.T5GEMMA_NPZ_REL)))
    padding_emb, secs_embedder = S.load_conditioner_from_npz(
        str(S.ensure_local(S.DIT_CHOICES[a.dit]["ckpt"])), prefix="cond.")
    decoder, chunk_fn, (chunk, ovl) = S.load_decoder(decoder_name, mx.float32)
    sigmas = S.build_pingpong_schedule(a.steps, sigma_max=1.0, use_logsnr_shift=True)
    print(f"text encoder + decoder loaded in {time.time() - t_load:.1f}s")

    def decode(lat, T_lat):  # same dispatch as sa3_mlx.py
        if T_lat > chunk + 2 * ovl:
            return chunk_fn(decoder, lat, chunk, ovl)
        if T_lat % 2 == 0:
            return decoder(lat)
        if T_lat > 6:
            return chunk_fn(decoder, lat, 2, 2)
        return decoder(mx.concatenate([lat, lat[..., -1:]], axis=-1))[..., : T_lat * 16]

    out.mkdir(parents=True, exist_ok=True)
    meta = open(out / "meta.jsonl", "a")
    times, t_all = [], time.time()
    for secs in sorted({j["seconds"] for j in todo}):       # one DiT build per duration
        bucket = [j for j in todo if j["seconds"] == secs]
        T_lat = max(1, math.ceil(secs * S.SAMPLE_RATE / S.SAMPLES_PER_LATENT))
        t0 = time.time()
        dit, _ = S.load_dit(a.dit, T_lat=T_lat, dtype=dtype, num_steps=a.steps)
        seconds_embed = secs_embedder(float(secs)).astype(dtype)
        global_cond = seconds_embed[:, 0, :]
        print(f"[{secs}s] DiT loaded in {time.time() - t0:.1f}s, {len(bucket)} clips")
        for k, j in enumerate(bucket):
            t0 = time.time()
            embeds, mask = enc.encode([j["prompt"]], max_len=256)
            embeds = S.apply_prompt_padding(embeds.astype(dtype), mask, padding_emb.astype(dtype))
            cross = mx.concatenate([embeds, seconds_embed], axis=1)
            noise = mx.random.normal((1, 256, T_lat), dtype=dtype, key=mx.random.key(j["seed"]))
            lat = S.sample_flow_pingpong(lambda x, t: dit(x, t, cross, global_cond, local_add_cond=None),
                                         noise, sigmas, seed=j["seed"] + 1)
            audio = S.patched_decode(decode(lat.astype(mx.float32), T_lat), patch_size=256, channels=2)
            audio = np.array(audio.astype(mx.float32))[0][..., : int(round(secs * S.SAMPLE_RATE))]
            path = out / j["category"] / f"{j['id']}.wav"
            path.parent.mkdir(exist_ok=True)
            S.save_wav(str(path), audio)
            meta.write(json.dumps({**j, "file": f"{j['category']}/{j['id']}.wav", "model": a.dit,
                                   "steps": a.steps, "peak": round(float(np.abs(audio).max()), 3)}) + "\n")
            meta.flush()
            times.append(time.time() - t0)
            if (k + 1) % 50 == 0 or k + 1 == len(bucket):
                print(f"  {k + 1}/{len(bucket)}  avg {sum(times[-50:]) / len(times[-50:]):.2f}s/clip", flush=True)
        del dit
    total = time.time() - t_all
    print(f"\n{len(times)} clips in {total:.0f}s -> {total / len(times):.2f}s/clip "
          f"(median {sorted(times)[len(times) // 2]:.2f}s) -> {out}")


if __name__ == "__main__":
    main()
