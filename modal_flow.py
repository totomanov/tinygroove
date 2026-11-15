"""Train the flow model on Modal, on the latents of a trained autoencoder.

    modal run modal_flow.py                                   # 100,000 steps (stops at 40 min if slower)
    modal run modal_flow.py --name flow_v1 --fetch-only       # pull an earlier run's results again

Everything it needs is already on the Modal volume: <ae>/latents.npz and <ae>/ae.pt from modal_ae.py,
and ae_data/ (clean clips and metadata). Results land in ./<name>/: flow.pt (best EMA weights plus
tag list, categories, latent scale and knob statistics), log.json and samples/ for listening.
"""
import modal

app = modal.App("tinygroove-flow")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.7.1", index_url="https://download.pytorch.org/whl/cu126")
    .pip_install("numpy")
    .add_local_python_source("ae_model", "ae_train", "process_dataset", "flow_model", "flow_train", "flow_eval")
)
data = modal.Volume.from_name("tinygroove-data")
PREP = "/data/ae_data"


@app.function(image=image, gpu=["H100", "A100-80GB", "A100-40GB", "L40S"], cpu=8, volumes={"/data": data}, timeout=2 * 3600)
def train(name: str, ae_name: str, steps: int, max_min: float):
    import json, os
    import numpy as np
    import torch
    from ae_model import AE
    from ae_train import brightness
    from flow_eval import evaluate
    from flow_train import pitch, train as run
    dev = "cuda"
    torch.set_float32_matmul_precision("high")
    meta = json.load(open(f"{PREP}/meta.json"))
    L = np.load(f"/data/{ae_name}/latents.npz")
    assert list(L["ids"]) == [m["id"] for m in meta]
    audio = torch.from_numpy(np.load(f"{PREP}/clean.npy")).to(dev)
    names = sorted({m["category"] for m in meta})
    vocab = sorted({f"{k}:{v}" for m in meta for k, v in m["words"].items()})
    where = {v: i for i, v in enumerate(vocab)}
    cats = torch.tensor([names.index(m["category"]) for m in meta], device=dev)
    tags = torch.full((len(meta), max(len(m["words"]) for m in meta)), len(vocab))
    for i, m in enumerate(meta):
        for j, (k, v) in enumerate(m["words"].items()):
            tags[i, j] = where[f"{k}:{v}"]
    tags = tags.to(dev)
    raw = torch.stack([torch.log2(torch.tensor([float(m["length_ms"]) for m in meta], device=dev)),
                       brightness(audio, dev), pitch(audio, dev)], 1)
    kmean, kstd = raw.mean(0), raw.std(0)
    knobs = (raw - kmean) / kstd
    z = torch.from_numpy(L["z"]).to(dev).float()
    scale = float(z.std())
    z = z / scale
    print(f"{len(meta)} clips, {len(vocab)} tags, latent scale {scale:.3f}, gpu {torch.cuda.get_device_name(0)}", flush=True)

    r = run(z, cats, tags, knobs, len(vocab), steps, max_min * 60, dev, log=lambda s: print(s, flush=True))
    flow = r["ema"]
    a = torch.load(f"/data/{ae_name}/ae.pt", map_location=dev)
    ae = AE(len(names), 2).to(dev).eval()
    ae.load_state_dict(a["state"])
    stats, clips = evaluate(flow, ae, scale, z, cats, tags, knobs, kmean, kstd, names, vocab,
                            r["val_idx"], r["train_idx"], dev)
    q = torch.tensor([0.05, 0.5, 0.95], device=dev)
    out = f"/data/{name}"
    os.makedirs(out, exist_ok=True)
    torch.save(dict(state=flow.state_dict(), categories=names, tags=vocab, scale=scale, ae=ae_name,
                    knob_mean=kmean.tolist(), knob_std=kstd.tolist(),
                    knob_range={n: torch.quantile(knobs[cats == i], q, dim=0).tolist() for i, n in enumerate(names)}),
               f"{out}/flow.pt")
    res = dict(stats=stats, steps=r["steps"], epochs=round(r["epochs"], 1), best_val=r["best_val"],
               last_val=r["last_val"], gpu=torch.cuda.get_device_name(0))
    json.dump(dict(res, hist=r["hist"]), open(f"{out}/log.json", "w"))
    np.savez_compressed(f"{out}/samples.npz", names=np.array(list(clips)),
                        audio=(torch.stack(list(clips.values())).clamp(-1, 1) * 32767).short().cpu().numpy())
    data.commit()
    return res


@app.local_entrypoint()
def main(name: str = "flow_v1", ae: str = "ae_v2", steps: int = 100000, max_min: float = 40.0, fetch_only: bool = False):
    import json, wave
    from pathlib import Path
    import numpy as np
    if not fetch_only:
        print("train result:", json.dumps(train.remote(name, ae, steps, max_min), indent=1))
    out = Path(name)
    out.mkdir(exist_ok=True)
    for f in ["flow.pt", "log.json", "samples.npz"]:
        (out / f).write_bytes(b"".join(data.read_file(f"{name}/{f}")))
    s = np.load(out / "samples.npz")
    for n, a in zip(s["names"], s["audio"]):
        p = out / "samples" / f"{n}.wav"
        p.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(p), "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(44100)
            w.writeframes(a.astype("<i2").tobytes())
    print("fetched ->", out.resolve())
