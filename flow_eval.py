"""Simple checks and a listening set for a trained flow model.

Numbers: does a generated sound land in the asked category, does it pass the dataset gate, is it
as varied as the teacher's sounds, is it a copy of a training sound, and do the knobs steer it.
"""
import numpy as np
import torch
from ae_train import brightness
from flow_model import KNOBS
from flow_train import pitch
from process_dataset import process

SWEEP_CATS = ("kick", "clap", "snare", "hat_closed", "hat_open", "tom")
TAG_CATS = ("kick", "clap", "snare", "hat_closed")
QS = (0.05, 0.275, 0.5, 0.725, 0.95)


def _corr(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-9))


@torch.no_grad()
def evaluate(flow, ae, scale, z, cats, tags, knobs, kmean, kstd, names, vocab, val_idx, train_idx, device,
             per_cat=64, steps=24, cfg=2.0, seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    n_cat = len(names)
    decode = lambda x: ae.decode((x * scale).clamp(-1, 1), generator=g)
    stats, clips = {}, {}

    # A. category only
    c = torch.arange(n_cat, device=device).repeat_interleave(per_cat)
    x = flow.sample(c, steps=steps, cfg=cfg, generator=g)
    y = decode(x)
    pred = ae.aux((x * scale).clamp(-1, 1).flatten(1))[:, :n_cat].argmax(1)
    stats["category_acc"] = float((pred == c).float().mean())
    stats["category_acc_by_cat"] = {names[k]: round(float((pred == c)[c == k].float().mean()), 3) for k in range(n_cat)}
    reasons = [process(w[:, None])[1] for w in y.cpu().numpy()]
    stats["gate_pass"] = float(np.mean([not isinstance(r, str) for r in reasons]))
    stats["gate_rejects"] = {r: reasons.count(r) for r in set(reasons) if isinstance(r, str)}
    div = {}
    zt = z[train_idx.to(device)].float()
    ct = cats[train_idx.to(device)]
    for k in range(n_cat):
        real = zt[ct == k][:per_cat].flatten(1)
        gen = x[c == k].flatten(1)
        div[names[k]] = round(float(torch.pdist(gen).mean() / torch.pdist(real).mean()), 3)
    stats["diversity_vs_teacher_by_cat"] = div
    stats["diversity_vs_teacher"] = float(np.mean(list(div.values())))
    nn_gen = torch.cdist(x.flatten(1), zt.flatten(1)).min(1).values
    nn_val = torch.cdist(z[val_idx.to(device)].float().flatten(1), zt.flatten(1)).min(1).values
    stats["nearest_train_dist"] = dict(generated=float(nn_gen.median()), held_out_real=float(nn_val.median()),
                                       generated_min=float(nn_gen.min()))
    for k in range(n_cat):
        for i in range(8):
            clips[f"category/{names[k]}_{i}"] = y[k * per_cat + i]

    # B. knobs: ask for the knob values of real held-out sounds, measure what comes out
    idx = val_idx.to(device)
    x = flow.sample(cats[idx], knobs=knobs[idx], kmask=torch.ones(len(idx), 3, device=device),
                    steps=steps, cfg=cfg, generator=g)
    y = decode(x)
    y16 = (y.clamp(-1, 1) * 32767).short()
    length = [process(w[:, None])[1] for w in y.cpu().numpy()]
    ok = torch.tensor([not isinstance(l, str) for l in length], device=device)
    dec = torch.tensor([np.log2(l) if not isinstance(l, str) else 0.0 for l in length], device=device)
    got = torch.stack([dec.float(), brightness(y16, device), pitch(y16, device)], 1)
    got = (got - kmean) / kstd
    stats["knob_corr"] = {n: round(_corr(knobs[idx][ok, i], got[ok, i]), 3) for i, n in enumerate(KNOBS)}

    # C. listening: knob sweeps and tags, each row from one fixed noise
    for name in SWEEP_CATS:
        k = names.index(name)
        q = torch.quantile(knobs[cats == k].float(), torch.tensor(QS, device=device), dim=0)    # [5, 3]
        for n in range(2):
            noise = torch.randn(1, *x.shape[1:], device=device, generator=g).expand(len(QS), -1, -1)
            for i, kn in enumerate(KNOBS):
                kv, km = torch.zeros(len(QS), 3, device=device), torch.zeros(len(QS), 3, device=device)
                kv[:, i], km[:, i] = q[:, i], 1
                w = decode(flow.sample(torch.full((len(QS),), k, device=device), knobs=kv, kmask=km,
                                       steps=steps, cfg=cfg, noise=noise))
                for j in range(len(QS)):
                    clips[f"sweep_{kn}/{name}_n{n}_{j + 1}of{len(QS)}"] = w[j]
    for name in TAG_CATS:
        k = names.index(name)
        src = [i for i, v in enumerate(vocab) if v.startswith("source:")]
        used = tags[cats == k]
        top = sorted(src, key=lambda i: -int((used == i).sum()))[:6]
        noise = torch.randn(1, *x.shape[1:], device=device, generator=g).expand(len(top) + 1, -1, -1)
        tg = torch.tensor([[len(vocab)]] + [[i] for i in top], device=device)
        w = decode(flow.sample(torch.full((len(tg),), k, device=device), tags=tg, steps=steps, cfg=cfg, noise=noise))
        for j, label in enumerate(["no tag"] + [vocab[i].split(":", 1)[1] for i in top]):
            clips[f"tags/{name}_{j}_{label.replace(' ', '_').replace('/', '-')}"] = w[j]
    return stats, clips
