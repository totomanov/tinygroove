"""Training loop for the flow model: plain flow matching on autoencoder latents.

x_t = (1 - t) * noise + t * latent; the model predicts (latent - noise). Conditions are dropped at
random so that each is optional at sampling time. The best EMA weights by held-out loss are kept,
as a guard against memorising the 48k training sounds.
"""
import copy, json, math, time
import torch
import torch.nn.functional as F
from tinygroove.model.ae_model import N, SR
from tinygroove.model.flow_model import Flow


def pitch(audio, device, bs=1024):
    """log2 of the strongest frequency between 30 Hz and 16 kHz, per clip."""
    out = []
    f = torch.fft.rfftfreq(N, 1 / SR).to(device)
    ok = (f >= 30) & (f <= 16000)
    for i in range(0, len(audio), bs):
        P = torch.fft.rfft(audio[i:i + bs, :N].to(device).float() / 32768).abs()
        out.append(torch.log2(f[(P * ok).argmax(1)]))
    return torch.cat(out)


def drop_conditions(cat, tags, n_cat, n_tags, g=None):
    """Random condition dropout for one batch. Returns (cat, tags, kmask)."""
    B, dev = len(cat), cat.device
    r = lambda *s: torch.rand(*s, device=dev, generator=g)
    none = r(B) < 0.1                                           # everything dropped: the cfg branch
    cat = torch.where(none, torch.full_like(cat, n_cat), cat)
    drop_tag = none[:, None] | (r(B, 1) < 0.3) | (r(*tags.shape) < 0.2)
    tags = torch.where(drop_tag, torch.full_like(tags, n_tags), tags)
    kmask = ((r(B, 3) < 0.5) & ~none[:, None]).float()
    return cat, tags, kmask


def train(z, cats, tags, knobs, n_tags, steps, max_s, device, batch=512, lr=5e-4, log=print, n_val=1024, seed=0):
    """z [M,K,D] latents already divided by their std; cats [M]; tags [M,n] padded with n_tags;
    knobs [M,3] standardized. Stops at `steps` or `max_s` seconds, whichever comes first."""
    torch.manual_seed(seed)
    M, n_cat = len(z), int(cats.max()) + 1
    perm = torch.randperm(M, device=device)
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    model = Flow(n_cat, n_tags).to(device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr, betas=(0.9, 0.99), weight_decay=0.01)

    def loss_on(m, idx, g=None):
        x1 = z[idx].float()
        x0 = torch.randn(x1.shape, device=device, generator=g)
        t = torch.sigmoid(torch.randn(len(idx), device=device, generator=g))        # logit-normal
        c, tg, km = drop_conditions(cats[idx], tags[idx], n_cat, n_tags, g)
        xt = (1 - t[:, None, None]) * x0 + t[:, None, None] * x1
        return F.mse_loss(m(xt, t, c, tg, knobs[idx], km), x1 - x0)

    @torch.no_grad()
    def validate():
        g = torch.Generator(device=device).manual_seed(1)       # same noise and times every time
        return sum(loss_on(ema, val_idx, g).item() for _ in range(8)) / 8

    t0, step, hist, best, best_state, run = time.time(), 0, [], 1e9, None, 0.0
    while True:
        p = max(step / steps, (time.time() - t0) / max_s)
        if p >= 1:
            break
        cur = lr * min(1.0, (step + 1) / 500) * (0.5 * (1 + math.cos(math.pi * p)) * 0.98 + 0.02)
        for g in opt.param_groups:
            g["lr"] = cur
        loss = loss_on(model, train_idx[torch.randint(len(train_idx), (batch,), device=device)])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            decay = min(0.9995, (1 + step) / (10 + step))
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.lerp_(pm, 1 - decay)
        step += 1
        run = loss.item() if step == 1 else 0.98 * run + 0.02 * loss.item()
        if step % 2000 == 0 or step == 1:
            v = validate()
            if v < best:
                best, best_state = v, copy.deepcopy(ema.state_dict())
            rec = dict(step=step, minutes=round((time.time() - t0) / 60, 2), its=round(step / (time.time() - t0), 1),
                       train=round(run, 4), val=round(v, 4), best_val=round(best, 4), lr=round(cur, 6))
            hist.append(rec)
            log(json.dumps(rec))
    v = validate()
    if v < best:
        best, best_state = v, copy.deepcopy(ema.state_dict())
    last_val = v
    ema.load_state_dict(best_state)
    return dict(ema=ema, hist=hist, steps=step, epochs=step * batch / len(train_idx), best_val=best,
                last_val=last_val, val_idx=val_idx.cpu(), train_idx=train_idx.cpu())
