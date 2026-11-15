"""Train the drum autoencoder on Modal.

    modal run modal_ae.py                                  # 40,000 steps (stops at 50 min if slower)
    modal run modal_ae.py --steps 80000 --max-min 100      # longer run; no other limit applies
    modal run modal_ae.py --name ae_v1 --fetch-only        # pull an earlier run's results again

Data: the raw 50k render is already on the Modal volume (tinygroove-data/dataset_v4). `prep`
applies the same gate and cleaning as process_dataset.py there, so nothing is uploaded.
Results land in ./<name>/ (default ae_v2; ae_v1 is the first baseline run): ae.pt (EMA weights), log.json, latents.npz (every clip encoded) and
recon/<category>/ with original, reconstruction and a second decode of the same latent.
"""
import modal

app = modal.App("tinygroove-ae")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.7.1", index_url="https://download.pytorch.org/whl/cu126")
    .pip_install("numpy")
    .add_local_python_source("ae_model", "ae_train", "process_dataset")
)
data = modal.Volume.from_name("tinygroove-data")
SRC = "/data/dataset_v4"
PREP = "/data/ae_data"


def _prep_chunk(path):
    import json
    import numpy as np
    from process_dataset import process
    z = np.load(path)
    audio, meta = [], []
    for a, j in zip(z["audio"], json.loads(str(z["meta"]))):
        y, info = process(a.T.astype(np.float32) / 32768)
        if y is not None:
            audio.append((y * 32767).astype(np.int16))
            meta.append(dict(id=j["id"], category=j["category"], words=j.get("words", {}),
                             matched=j.get("matched"), prompt=j["prompt"], length_ms=info))
    return np.stack(audio), meta


@app.function(image=image, cpu=16, memory=32768, volumes={"/data": data}, timeout=1800)
def prep():
    import glob, json, os
    from multiprocessing import Pool
    import numpy as np
    if os.path.exists(f"{PREP}/meta.json"):
        return len(json.load(open(f"{PREP}/meta.json")))
    with Pool(16) as pool:
        parts = pool.map(_prep_chunk, sorted(glob.glob(f"{SRC}/*.npz")))
    os.makedirs(PREP, exist_ok=True)
    np.save(f"{PREP}/clean.npy", np.concatenate([a for a, _ in parts]))
    meta = [m for _, ms in parts for m in ms]
    json.dump(meta, open(f"{PREP}/meta.json", "w"))
    data.commit()
    return len(meta)


@app.function(image=image, gpu=["H100", "A100-80GB", "A100-40GB", "L40S"], volumes={"/data": data}, timeout=4 * 3600)
def train(name: str, steps: int, max_min: float):
    import io, json, os
    import numpy as np
    import torch
    from ae_train import encode_all, train as run
    dev = "cuda"
    torch.backends.cudnn.benchmark = True
    meta = json.load(open(f"{PREP}/meta.json"))
    audio = torch.from_numpy(np.load(f"{PREP}/clean.npy")).to(dev)
    names = sorted({m["category"] for m in meta})
    cats = torch.tensor([names.index(m["category"]) for m in meta])
    length = torch.tensor([m["length_ms"] for m in meta])
    print(f"{len(meta)} clips, {len(names)} categories, gpu {torch.cuda.get_device_name(0)}", flush=True)
    out = f"/data/{name}"
    os.makedirs(out, exist_ok=True)

    def ckpt(ema, step):                      # so a late failure cannot lose the run
        torch.save(dict(state=ema.state_dict(), categories=names, step=step), f"{out}/ae_ckpt.pt")
        data.commit()

    r = run(audio, cats, length, steps, max_min * 60, dev, log=lambda s: print(s, flush=True), on_ckpt=ckpt)
    ema = r["ema"]
    torch.save(dict(state=ema.state_dict(), categories=names), f"{out}/ae.pt")
    val = dict(r["val"], spec_by_cat={names[c]: round(v, 4) for c, v in r["val"]["spec_by_cat"].items()})
    json.dump(dict(val=val, hist=r["hist"], steps=r["steps"], epochs=r["epochs"],
                   gpu=torch.cuda.get_device_name(0)), open(f"{out}/log.json", "w"))
    np.savez_compressed(f"{out}/latents.npz", z=encode_all(ema, audio, dev).numpy(),
                        ids=np.array([m["id"] for m in meta]))
    # listening set: 6 held-out clips per category: original, reconstruction, second decode
    picks = []
    for c in range(len(names)):
        picks += [int(i) for i in r["val_idx"] if cats[i] == c][:6]
    with torch.no_grad():
        x = audio[picks, :44032].float() / 32768
        z = ema.encode(x)
        ya, yb = ema.decode(z), ema.decode(z)
    to16 = lambda t: (t.clamp(-1, 1) * 32767).short().cpu().numpy()
    np.savez_compressed(f"{out}/recon.npz", orig=to16(x), recon=to16(ya), recon_b=to16(yb),
                        ids=np.array([meta[i]["id"] for i in picks]),
                        cats=np.array([meta[i]["category"] for i in picks]))
    data.commit()
    return dict(val=val, steps=r["steps"], epochs=round(r["epochs"], 1), gpu=torch.cuda.get_device_name(0),
                its=r["hist"][-1]["its"] if r["hist"] else None)


@app.local_entrypoint()
def main(name: str = "ae_v2", steps: int = 40000, max_min: float = 50.0, fetch_only: bool = False):
    import io, json, wave
    from pathlib import Path
    import numpy as np
    if not fetch_only:
        print("prep: clean clips on volume:", prep.remote())
        r = train.remote(name, steps, max_min)
        print("train result:", json.dumps(r, indent=1))
    out = Path(name)
    out.mkdir(exist_ok=True)
    for f in ["ae.pt", "log.json", "latents.npz", "recon.npz"]:
        (out / f).write_bytes(b"".join(data.read_file(f"{name}/{f}")))
    z = np.load(out / "recon.npz")
    for i, (cid, cat) in enumerate(zip(z["ids"], z["cats"])):
        d = out / "recon" / str(cat)
        d.mkdir(parents=True, exist_ok=True)
        for kind in ["orig", "recon", "recon_b"]:
            with wave.open(str(d / f"{cid}_{kind}.wav"), "wb") as w:
                w.setnchannels(1); w.setsampwidth(2); w.setframerate(44100)
                w.writeframes(z[kind][i].astype("<i2").tobytes())
    print("fetched ->", out.resolve())
