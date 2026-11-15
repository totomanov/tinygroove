"""Render the teacher dataset on Modal GPUs (PyTorch reference runtime of Stable Audio 3).

One-time setup (yours to do, in your own terminal):
    modal token new                                   # log this machine in to Modal
    modal secret create huggingface HF_TOKEN=hf_...   # read token of the account with SA3 access

Run:
    modal run modal_gen.py --n 600  --out renders/modal_test          # smoke test
    modal run modal_gen.py --n 10000 --out renders/dataset_v4         # first pass
    modal run modal_gen.py --n 50000 --out renders/dataset_v4         # extends it; done chunks are skipped

The plan (prompts, seeds, matched sets) comes from gen_batch.plan(), the same as the Mac path.
Clips are rendered in batches on the GPU, stored per chunk on a Modal volume, then pulled back
and unpacked to <out>/<category>/<id>.wav + <out>/meta.jsonl, ready for process_dataset.py.

Seeds: the GPU renders a whole batch from one seed, so a clip is reproduced by (batch_seed, row),
both saved in meta.jsonl. Matched sets are batched so that every member of a set sits in the same
row of batches that share a seed: same starting noise, one word changed.
Sounds differ from the Mac (MLX) renders of the same plan: different random numbers.
"""
import modal

app = modal.App("tinygroove-gen")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libsndfile1")
    .pip_install("torch==2.7.1", "torchaudio==2.7.1", index_url="https://download.pytorch.org/whl/cu126")
    .pip_install("git+https://github.com/Stability-AI/stable-audio-3.git")
    .env({"HF_HOME": "/cache/hf"})
)
cache = modal.Volume.from_name("tinygroove-cache", create_if_missing=True)
data = modal.Volume.from_name("tinygroove-data", create_if_missing=True)
BATCH = 64
UNITS_PER_CHUNK = 8          # 8 batches of 64 = up to 512 clips per stored chunk


@app.cls(image=image, gpu=["L40S", "A10G", "L4"], volumes={"/cache": cache, "/data": data},
         secrets=[modal.Secret.from_name("huggingface")], max_containers=12, timeout=3600)
class Teacher:
    @modal.enter()
    def load(self):
        from stable_audio_3 import StableAudioModel
        self.model = StableAudioModel.from_pretrained("small-sfx")

    @modal.method()
    def render(self, run: str, chunk_id: str, units: list, steps: int, padding: float):
        """units: [{seed, jobs: [{id, prompt, ...}]}]. Writes /data/<run>/<chunk_id>.npz."""
        import io, json, os, time
        import numpy as np
        path = f"/data/{run}/{chunk_id}.npz"
        if os.path.exists(path):
            return dict(chunk=chunk_id, clips=0, seconds=0.0, skipped=True)
        t0, audio, meta = time.time(), [], []
        for u in units:
            jobs = u["jobs"]
            out = self.model.generate(prompt=[j["prompt"] for j in jobs], duration=1.0, steps=steps,
                                      cfg_scale=1.0, batch_size=len(jobs), seed=u["seed"],
                                      duration_padding_sec=padding)
            x = out.float().cpu().numpy()[:, :, :44100]
            audio.append((np.clip(x, -1, 1) * 32767).astype(np.int16))
            meta += [{**j, "batch_seed": u["seed"], "row": r, "seed": None} for r, j in enumerate(jobs)]
        buf = io.BytesIO()
        np.savez_compressed(buf, audio=np.concatenate(audio), meta=json.dumps(meta))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(buf.getvalue())
        data.commit()
        return dict(chunk=chunk_id, clips=len(meta), seconds=time.time() - t0, skipped=False)


def make_units(jobs):
    """Group jobs into GPU batches. Matched sets are laid out row-aligned (see module docstring)."""
    singles = [j for j in jobs if "matched" not in j]
    units = [dict(seed=c[0]["seed"], jobs=c) for c in (singles[i:i + BATCH] for i in range(0, len(singles), BATCH))]
    sets = {}
    for j in jobs:
        if "matched" in j:
            sets.setdefault(j["matched"], []).append(j)
    by_len = {}
    for s in sets.values():
        by_len.setdefault(len(s), []).append(s)
    for n, group in sorted(by_len.items()):
        for i in range(0, len(group), BATCH):
            block = group[i:i + BATCH]
            for m in range(n):                       # member m of every set, same seed for all m
                units.append(dict(seed=block[0][0]["seed"], jobs=[s[m] for s in block]))
    return units


@app.local_entrypoint()
def main(n: int = 600, out: str = "renders/modal_test", steps: int = 8, padding: float = 6.0,
         only: str = "", fetch_only: bool = False):
    import io, json, time, wave
    from pathlib import Path
    import numpy as np
    from gen_batch import plan

    out_dir = Path(out).resolve()
    run = out_dir.name
    jobs = plan(n, set(filter(None, only.split(","))), 0)
    units = make_units(jobs)
    # chunk ids are stable for a given plan prefix only within one n: name them by content
    import hashlib
    chunks = []
    for i in range(0, len(units), UNITS_PER_CHUNK):
        us = units[i:i + UNITS_PER_CHUNK]
        h = hashlib.sha1(json.dumps([[j["id"] for j in u["jobs"]] + [u["seed"]] for u in us]).encode()).hexdigest()[:12]
        chunks.append((h, us))
    print(f"{len(jobs)} clips in {len(units)} batches, {len(chunks)} chunks -> volume tinygroove-data/{run}")

    if not fetch_only:
        t0, done, gpu_s = time.time(), 0, 0.0
        for r in Teacher().render.starmap([(run, h, us, steps, padding) for h, us in chunks], order_outputs=False):
            done += r["clips"]; gpu_s += r["seconds"]
            tag = "skipped (already on volume)" if r["skipped"] else f"{r['clips']} clips in {r['seconds']:.1f}s gpu"
            print(f"  chunk {r['chunk']}: {tag}")
        print(f"rendered {done} new clips in {time.time() - t0:.0f}s wall, {gpu_s:.0f}s of GPU render time "
              f"({done / max(gpu_s, 1e-9):.1f} clips/s per GPU)")

    # pull back and unpack
    have = set()
    meta_path = out_dir / "meta.jsonl"
    if meta_path.exists():
        have = {json.loads(l)["id"] for l in open(meta_path)}
    out_dir.mkdir(parents=True, exist_ok=True)
    new = 0
    with open(meta_path, "a") as mf:
        for h, us in chunks:
            ids = {j["id"] for u in us for j in u["jobs"]}
            if ids <= have:
                continue
            z = np.load(io.BytesIO(b"".join(data.read_file(f"{run}/{h}.npz"))))
            for a, j in zip(z["audio"], json.loads(str(z["meta"]))):
                if j["id"] in have:
                    continue
                p = out_dir / j["category"] / f"{j['id']}.wav"
                p.parent.mkdir(exist_ok=True)
                with wave.open(str(p), "wb") as w:
                    w.setnchannels(2); w.setsampwidth(2); w.setframerate(44100)
                    w.writeframes(a.T.astype("<i2").tobytes())
                mf.write(json.dumps({**j, "file": f"{j['category']}/{j['id']}.wav", "model": "sm-sfx",
                                     "runtime": "torch", "steps": steps, "padding": padding,
                                     "peak": round(float(np.abs(a).max()) / 32767, 3)}) + "\n")
                have.add(j["id"]); new += 1
    print(f"unpacked {new} clips -> {out_dir}")
