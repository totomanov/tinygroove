"""Train the whole student on Modal, start to finish.

    modal run --detach -m tinygroove.model.modal_train               # the full run (about 75 minutes)
    TG_GPU=T4 modal run -m tinygroove.model.modal_train --smoke      # minutes-long dry run of every stage on a cheap GPU
    modal run -m tinygroove.model.modal_train --fetch-only           # pull the results of a finished run

Stages (each is its own job; the orchestrator runs in the cloud too, so this machine can go to sleep):
  prep       gate the raw renders of dataset_v4 and dataset_v5 on the Modal volume and join them
  stage 1    the autoencoder, spectral loss only: this fixes the sound code
  stage 2    on that frozen code, in parallel: the decoder with adversarial losses, and the generator
             (a 16-step teacher, then the half-size one-step student distilled from it)
  evaluate   one table for the teacher and the student at several step counts, plus listening sets
Results land in runs/<run>/. `export_onnx.py` turns them into the files the browser loads.
"""
import os
import modal

GPU = os.environ.get("TG_GPU", "H100")
app = modal.App("tinygroove-train")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.7.1", index_url="https://download.pytorch.org/whl/cu126")
    .pip_install("numpy")
    .add_local_python_source("tinygroove")
)
data = modal.Volume.from_name("tinygroove-data")
RAW, PREP, MODE = ("/data/dataset_v4", "/data/dataset_v5"), "/data/ae_data_v5", "tail"
gpu_job = dict(image=image, gpu=GPU, cpu=8, memory=49152, volumes={"/data": data}, timeout=3 * 3600)


def _prep_chunk(path):
    import json
    import numpy as np
    from tinygroove.data.process_dataset import process
    z = np.load(path)
    audio, meta = [], []
    for a, j in zip(z["audio"], json.loads(str(z["meta"]))):
        y, info = process(a.T.astype(np.float32) / 32768)
        if y is not None:
            audio.append((y * 32767).astype(np.int16))
            meta.append(dict(id=j["id"], category=j["category"], words=j.get("words", {}), matched=j.get("matched"),
                             prompt=j["prompt"], length_ms=info, src=path.split("/")[-2].split("_")[-1]))
    return np.stack(audio), meta


@app.function(image=image, cpu=16, memory=65536, volumes={"/data": data}, timeout=3600)
def prep():
    import glob, json, os
    from collections import Counter
    from multiprocessing import Pool
    import numpy as np
    data.reload()
    if not os.path.exists(f"{PREP}/meta.json"):
        with Pool(16) as pool:
            parts = pool.map(_prep_chunk, [p for d in RAW for p in sorted(glob.glob(f"{d}/*.npz"))])
        os.makedirs(PREP, exist_ok=True)
        np.save(f"{PREP}/clean.npy", np.concatenate([a for a, _ in parts]))
        json.dump([m for _, ms in parts for m in ms], open(f"{PREP}/meta.json", "w"))
        data.commit()
    meta = json.load(open(f"{PREP}/meta.json"))
    return dict(clips=len(meta), by_category=dict(Counter(m["category"] for m in meta)))


def _load(smoke):
    """Clips, metadata and a held-out set, on the GPU."""
    import json
    import numpy as np
    import torch
    data.reload()
    meta = json.load(open(f"{PREP}/meta.json"))
    audio = np.load(f"{PREP}/clean.npy", mmap_mode="r")
    g = torch.Generator().manual_seed(0)
    if smoke:                                                    # a random 6,000, so that every drum type is present
        sub = torch.randperm(len(meta), generator=g)[:6000].sort().values
        meta, audio = [meta[i] for i in sub.tolist()], audio[sub.numpy()]
    val_idx = torch.randperm(len(meta), generator=g)[:128 if smoke else 1024].sort().values.cuda()
    names = sorted({m["category"] for m in meta})
    cats = torch.tensor([names.index(m["category"]) for m in meta]).cuda()
    length = torch.tensor([float(m["length_ms"]) for m in meta]).cuda()
    return torch.from_numpy(np.ascontiguousarray(audio)).cuda(), meta, names, cats, length, val_idx


def _say(s):
    print(s, flush=True)


@app.function(**gpu_job)
def stage1(run: str, minutes: float, smoke: bool):
    import os
    import numpy as np
    import torch
    from tinygroove.model.ae_train import encode_all, train_ae
    torch.backends.cudnn.benchmark = True
    audio, meta, names, cats, length, val_idx = _load(smoke)
    _say(f"stage 1: {len(meta)} clips, {len(val_idx)} held out, gpu {torch.cuda.get_device_name(0)}")
    r = train_ae(audio, cats, length, val_idx, MODE, minutes * 60, "cuda", log=_say)
    out = f"/data/{run}"; os.makedirs(out, exist_ok=True)
    np.save(f"{out}/latents.npy", encode_all(r["ema"], audio, length, "cuda").cpu().numpy())
    res = dict(val_spec=r["val_spec"], val_by_cat={names[c]: round(v, 4) for c, v in r["val_by_cat"].items()}, steps=r["steps"])
    torch.save(dict(state=r["ema"].state_dict(), mode=MODE, categories=names, hist=r["hist"], **res), f"{out}/ae.pt")
    data.commit()
    return res


@app.function(**gpu_job)
def stage2(run: str, minutes: float, smoke: bool):
    import numpy as np
    import torch
    from tinygroove.model.ae_train import train_decoder
    torch.backends.cudnn.benchmark = True
    audio, meta, names, cats, length, val_idx = _load(smoke)
    out = f"/data/{run}"
    z = torch.from_numpy(np.load(f"{out}/latents.npy")).cuda()
    s = torch.load(f"{out}/ae.pt", map_location="cuda")["state"]
    init = {k[4:]: v for k, v in s.items() if k.startswith("dec.")}            # start from the stage 1 decoder
    r = train_decoder(audio, cats, length, z, val_idx, MODE, 160, minutes * 60, "cuda", init=init, log=_say)
    res = dict(width=160, val_spec=r["val_spec"], steps=r["steps"], gan_failed=r["gan_failed"], gap_last=round(r["gap_last"], 4),
               params=sum(p.numel() for p in r["ema"].parameters()))
    torch.save(dict(state=r["ema"].state_dict(), mode=MODE, hist=r["hist"], **res), f"{out}/decoder.pt")
    data.commit()
    return res


def _flow_data(run, smoke):
    """Latents (+ length token), tags and knobs for the generator, all on the GPU."""
    import numpy as np
    import torch
    from tinygroove.model.ae_model import D, K
    from tinygroove.model.ae_train import brightness
    from tinygroove.model.flow_eval import pitch
    audio, meta, names, cats, length, val_idx = _load(smoke)
    vocab = sorted({f"{k}:{v}" for m in meta for k, v in m["words"].items()})
    where = {v: i for i, v in enumerate(vocab)}
    tags = torch.full((len(meta), max(len(m["words"]) for m in meta)), len(vocab))
    for i, m in enumerate(meta):
        for j, (k, v) in enumerate(m["words"].items()):
            tags[i, j] = where[f"{k}:{v}"]
    raw = torch.stack([torch.log2(length), brightness(audio, "cuda"), pitch(audio, "cuda")], 1)
    kmean, kstd = raw.mean(0), raw.std(0)
    z = torch.from_numpy(np.load(f"/data/{run}/latents.npy")).cuda().float()
    scale = float(z.std())
    x = torch.zeros(len(meta), K + 1, D, device="cuda")
    x[:, :K] = z / scale
    x[:, K, 0] = (raw[:, 0] - kmean[0]) / kstd[0]                # the length token: the decay knob's own scale
    return dict(audio=audio, meta=meta, names=names, cats=cats, length=length, val_idx=val_idx, vocab=vocab, tags=tags.cuda(),
                raw=raw, kmean=kmean, kstd=kstd, knobs=(raw - kmean) / kstd, scale=scale, x=x)


@app.function(**gpu_job)
def generator(run: str, teacher_minutes: float, student_minutes: float, smoke: bool):
    import torch
    from tinygroove.model.flow import distil, train_flow
    torch.set_float32_matmul_precision("high")
    d = _flow_data(run, smoke)
    names, vocab, cats, knobs = d["names"], d["vocab"], d["cats"], d["knobs"]
    q = torch.tensor([0.05, 0.5, 0.95], device="cuda")
    common = dict(categories=names, tags=vocab, scale=d["scale"], mode=MODE, knob_mean=d["kmean"].tolist(), knob_std=d["kstd"].tolist(),
                  knob_range={n: torch.quantile(knobs[cats == i], q, dim=0).tolist() for i, n in enumerate(names)})
    _say(f"teacher: {len(d['meta'])} clips, {len(vocab)} tags, latent scale {d['scale']:.3f}")
    t = train_flow(d["x"], cats, d["tags"], knobs, len(vocab), d["val_idx"], teacher_minutes * 60, "cuda", log=_say)
    torch.save(dict(state=t["ema"].state_dict(), d=224, layers=5, hist=t["hist"], **common), f"/data/{run}/flow_teacher.pt")
    _say("student: distilling")
    s = distil(t["ema"], cats, d["tags"], knobs, len(vocab), student_minutes * 60, "cuda", log=_say)
    torch.save(dict(state=s["ema"].state_dict(), d=160, layers=5, hist=s["hist"], **common), f"/data/{run}/flow_student.pt")
    data.commit()
    count = lambda m: sum(p.numel() for p in m.parameters())
    return dict(teacher=dict(steps=t["steps"], best_val=t["best_val"], last_val=t["last_val"], params=count(t["ema"])),
                student=dict(steps=s["steps"], loss=s["last_loss"], params=count(s["ema"])))


@app.function(**gpu_job)
def evaluate(run: str, smoke: bool):
    import json
    import numpy as np
    import torch
    from tinygroove.model.ae_model import AE, Decoder, N, frames_of
    from tinygroove.model.flow import Flow
    from tinygroove.model.flow_eval import judge_with, maker, score, to16
    d = _flow_data(run, smoke)
    out, dev = f"/data/{run}", "cuda"
    names, vocab, audio, cats, val_idx = d["names"], d["vocab"], d["audio"], d["cats"], d["val_idx"]
    ae = AE(MODE, n_categories=len(names)).to(dev).eval(); ae.load_state_dict(torch.load(f"{out}/ae.pt", map_location=dev)["state"])
    p = torch.load(f"{out}/decoder.pt", map_location=dev)
    decs = {"stage 1": ae.dec, "final": Decoder(p["width"], MODE).to(dev).eval()}
    decs["final"].load_state_dict(p["state"])
    flows = {}
    for name in ("teacher", "student"):
        p = torch.load(f"{out}/flow_{name}.pt", map_location=dev)
        flows[name] = Flow(len(names), len(vocab), p["d"], p["layers"]).to(dev).eval(); flows[name].load_state_dict(p["state"])
    make = lambda who, steps, dec: maker(flows[who], decs[dec], d["scale"], d["kmean"][0], d["kstd"][0], steps, who == "student", len(vocab), dev)
    settings = {f"teacher, {n} step(s), final decoder": make("teacher", n, "final") for n in (1, 4, 16)}
    settings.update({f"student, {n} step(s), final decoder": make("student", n, "final") for n in (1, 2, 4)})
    settings["student, 1 step, stage 1 decoder"] = make("student", 1, "stage 1")
    judge, table = judge_with(ae, len(names), dev), {}
    for name, mk in settings.items():
        table[name] = score(mk, judge, audio, cats, d["raw"], d["kmean"], d["kstd"], val_idx, names, dev, per_cat=8 if smoke else 64)
        _say(f"{name}: {json.dumps(table[name])}")
    json.dump(table, open(f"{out}/eval.json", "w"), indent=1)

    clips = {}                                                   # listening sets
    g = torch.Generator(device=dev).manual_seed(7)
    with torch.no_grad():
        for k, cat in enumerate(names):
            idx = val_idx[cats[val_idx] == k][:3]
            if len(idx) == 0:
                continue
            x, fr = audio[idx, :N].float() / 32768, frames_of(d["length"][idx])
            y = decs["final"](ae.encode(x, fr), fr)
            for j, i in enumerate(idx.tolist()):
                clips[f"recon/{cat}/{d['meta'][i]['id']}_teacher"] = to16(x[j:j + 1])[0]
                clips[f"recon/{cat}/{d['meta'][i]['id']}_decoded"] = to16(y[j:j + 1])[0]
        c = torch.arange(len(names), device=dev).repeat_interleave(6)
        zero = torch.zeros(len(c), 3, device=dev)
        for name, mk in {"student_1_step": make("student", 1, "final"), "teacher_16_steps": make("teacher", 16, "final")}.items():
            y = to16(mk(c, zero, zero, g))
            for j in range(len(c)):
                clips[f"generated_{name}/{names[c[j]]}_{j % 6}"] = y[j]
    np.savez_compressed(f"{out}/samples.npz", names=np.array(list(clips)), audio=torch.stack(list(clips.values())).cpu().numpy())
    data.commit()
    return table


@app.function(image=image, volumes={"/data": data}, timeout=5 * 3600)
def pipeline(run: str, smoke: bool, s1: float, s2: float, teacher: float, student: float):
    import json
    summary = dict(run=run, smoke=smoke, data=prep.remote()); _say(json.dumps(summary["data"]))
    summary["stage1"] = stage1.remote(run, s1, smoke); _say(json.dumps(summary["stage1"]))
    jobs = [stage2.spawn(run, s2, smoke), generator.spawn(run, teacher, student, smoke)]
    summary["decoder"], summary["generator"] = (j.get() for j in jobs)
    _say(json.dumps(dict(decoder=summary["decoder"], generator=summary["generator"])))
    summary["table"] = evaluate.remote(run, smoke)
    data.reload()
    json.dump(summary, open(f"/data/{run}/summary.json", "w"), indent=1)
    data.commit()
    return summary


@app.local_entrypoint()
def main(run: str = "v3", smoke: bool = False, fetch_only: bool = False, s1: float = 30, s2: float = 28,
         teacher: float = 10, student: float = 20):
    import json, wave
    from pathlib import Path
    import numpy as np
    if smoke:
        run, s1, s2, teacher, student = run + "_smoke", 1.0, 1.5, 0.6, 0.6
    if not fetch_only:
        print(json.dumps(pipeline.remote(run, smoke, s1, s2, teacher, student), indent=1)[:6000])
    out = Path(__file__).resolve().parents[2] / "runs" / run
    out.mkdir(parents=True, exist_ok=True)
    for f in ["summary.json", "eval.json"] + ([] if smoke else ["samples.npz", "ae.pt", "decoder.pt", "flow_teacher.pt", "flow_student.pt"]):
        (out / f).write_bytes(b"".join(data.read_file(f"{run}/{f}")))
    if not smoke:
        s = np.load(out / "samples.npz")
        for n, a in zip(s["names"], s["audio"]):
            p = out / "samples" / f"{n}.wav"
            p.parent.mkdir(parents=True, exist_ok=True)
            with wave.open(str(p), "wb") as w:
                w.setnchannels(1); w.setsampwidth(2); w.setframerate(44100)
                w.writeframes(a.astype("<i2").tobytes())
    print("fetched ->", out)
