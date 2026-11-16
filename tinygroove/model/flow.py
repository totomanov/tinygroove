"""tinygroove generator: a flow-matching teacher, then a smaller student that needs one step.

The thing generated is the autoencoder's 20 tokens plus one extra token whose first number is the sound's
length, so the model also decides how long a drum is when nobody asks for a decay.

Teacher (`train_flow`): plain flow matching. No guidance: measured, it narrowed the range of sounds and bought no obedience.
Student (`distil`): half the width budget, and it takes the step size as an input. It is trained on the
teacher's own 16-step paths to make any jump along them (1/16, 1/8, 1/4, 1/2 or the whole way) in one go,
so the same weights can be run with 1, 2, 4, 8 or 16 steps. One step holds up (by measurement and by ear) and is what ships.
"""
import copy, json, math, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tinygroove.model.ae_model import Attention, D, K

TOKENS = K + 1          # the autoencoder's tokens + one that carries the length
KNOBS = ("decay", "brightness", "pitch")


class Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.n1 = nn.LayerNorm(d, elementwise_affine=False)
        self.n2 = nn.LayerNorm(d, elementwise_affine=False)
        self.attn = Attention(d, heads)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.film = nn.Linear(d, 4 * d)
        nn.init.zeros_(self.film.weight); nn.init.zeros_(self.film.bias)

    def forward(self, x, c):
        s1, b1, s2, b2 = self.film(c)[:, None].chunk(4, -1)
        x = x + self.attn(self.n1(x) * (1 + s1) + b1)
        return x + self.mlp(self.n2(x) * (1 + s2) + b2)


class Flow(nn.Module):
    def __init__(self, n_categories, n_tags, d=224, layers=5, heads=4):
        super().__init__()
        self.n_categories, self.n_tags, self.d = n_categories, n_tags, d
        self.inp = nn.Linear(D, d)
        self.pos = nn.Parameter(torch.randn(TOKENS, d) * 0.02)
        self.time = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
        self.jump = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))        # step size; unused by the teacher
        self.cat = nn.Embedding(n_categories + 1, d)
        self.tag = nn.Embedding(n_tags + 1, d, padding_idx=n_tags)
        self.knob = nn.Sequential(nn.Linear(2 * len(KNOBS), d), nn.SiLU(), nn.Linear(d, d))
        self.blocks = nn.ModuleList([Block(d, heads) for _ in range(layers)])
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, D)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def _wave(self, v):
        half = self.d // 2
        f = torch.exp(-math.log(1000.0) * torch.arange(half, device=v.device) / half)
        a = v[:, None] * 1000.0 * f
        return torch.cat([a.sin(), a.cos()], 1)

    def forward(self, x, t, dt, cat, tags, knobs, kmask):
        """x [B,TOKENS,D]; t [B] in 0..1 (0 = noise); dt [B] = size of the step about to be taken.
        Returns the average velocity over that step, so x_next = x + dt * output."""
        c = self.time(self._wave(t)) + self.jump(self._wave(dt)) + self.cat(cat) + self.tag(tags).sum(1) \
            + self.knob(torch.cat([knobs * kmask, kmask], 1))
        h = self.inp(x) + self.pos
        for b in self.blocks:
            h = b(h, c)
        return self.out(self.norm(h))


@torch.no_grad()
def sample(model, cat, tags, knobs, kmask, steps, noise, student=True):
    """The student is told its step size; the teacher was only ever trained with step size 0."""
    x, B = noise.clone(), len(cat)
    dt = torch.full((B,), 1 / steps if student else 0.0, device=cat.device)
    for i in range(steps):
        x = x + model(x, torch.full((B,), i / steps, device=cat.device), dt, cat, tags, knobs, kmask) / steps
    return x


def drop_conditions(cat, tags, n_cat, n_tags, g=None):
    """Random condition dropout, so every condition is optional at run time. Returns (cat, tags, kmask)."""
    B, dev = len(cat), cat.device
    r = lambda *s: torch.rand(*s, device=dev, generator=g)
    none = r(B) < 0.05
    cat = torch.where(none, torch.full_like(cat, n_cat), cat)
    drop_tag = none[:, None] | (r(B, 1) < 0.3) | (r(*tags.shape) < 0.2)
    tags = torch.where(drop_tag, torch.full_like(tags, n_tags), tags)
    kmask = ((r(B, 3) < 0.5) & ~none[:, None]).float()
    return cat, tags, kmask


def _ema(ema, model, step, top=0.9995):
    with torch.no_grad():
        d = min(top, (1 + step) / (10 + step))
        for pe, pm in zip(ema.parameters(), model.parameters()):
            pe.lerp_(pm, 1 - d)


def train_flow(x, cats, tags, knobs, n_tags, val_idx, max_s, device, d=224, layers=5, steps=10 ** 9, batch=512, lr=5e-4,
               log=print, seed=0):
    """Teacher. x [M,TOKENS,D] unit-variance latents (+ length token). Keeps the EMA weights with the best held-out loss."""
    torch.manual_seed(seed)
    M, n_cat = len(x), int(cats.max()) + 1
    keep = torch.ones(M, dtype=torch.bool, device=device); keep[val_idx] = False
    train_idx = torch.nonzero(keep)[:, 0]
    model = Flow(n_cat, n_tags, d, layers).to(device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr, betas=(0.9, 0.99), weight_decay=0.01)

    def loss_on(m, idx, g=None):
        x1 = x[idx].float()
        x0 = torch.randn(x1.shape, device=device, generator=g)
        t = torch.sigmoid(torch.randn(len(idx), device=device, generator=g))
        c, tg, km = drop_conditions(cats[idx], tags[idx], n_cat, n_tags, g)
        xt = (1 - t[:, None, None]) * x0 + t[:, None, None] * x1
        return F.mse_loss(m(xt, t, torch.zeros_like(t), c, tg, knobs[idx], km), x1 - x0)

    @torch.no_grad()
    def validate():
        g = torch.Generator(device=device).manual_seed(1)
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
        opt.zero_grad(set_to_none=True); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); _ema(ema, model, step)
        step += 1
        run = loss.item() if step == 1 else 0.98 * run + 0.02 * loss.item()
        if step % 2000 == 0 or step == 1:
            v = validate()
            if v < best:
                best, best_state = v, copy.deepcopy(ema.state_dict())
            rec = dict(step=step, minutes=round((time.time() - t0) / 60, 2), its=round(step / (time.time() - t0), 1),
                       train=round(run, 4), val=round(v, 4), best_val=round(best, 4))
            hist.append(rec); log(json.dumps(rec))
    v = validate()
    if v < best:
        best, best_state = v, copy.deepcopy(ema.state_dict())
    ema.load_state_dict(best_state)
    return dict(ema=ema, hist=hist, steps=step, best_val=best, last_val=v)


JUMPS = (1, 1, 1, 2, 2, 4, 8, 16)        # how many equal jumps the path is cut into; 1 and 2 are drawn most often


def distil(teacher, cats, tags, knobs, n_tags, max_s, device, d=160, layers=5, steps=10 ** 9, batch=512, lr=5e-4,
           reuse=4, log=print, seed=0):
    """Student. Each round: the teacher walks `batch` fresh noises to sounds in 16 steps (conditions drawn from the
    data); the student is then trained `reuse` times to reproduce randomly chosen jumps along those paths."""
    torch.manual_seed(seed)
    M, n_cat = len(cats), int(cats.max()) + 1
    model = Flow(n_cat, n_tags, d, layers).to(device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr, betas=(0.9, 0.99), weight_decay=0.01)
    jumps = torch.tensor(JUMPS, device=device)
    rows = torch.arange(batch, device=device)
    t0, step, hist, run = time.time(), 0, [], 0.0
    while True:
        p = max(step / steps, (time.time() - t0) / max_s)
        if p >= 1:
            break
        cur = lr * min(1.0, (step + 1) / 500) * (0.5 * (1 + math.cos(math.pi * p)) * 0.98 + 0.02)
        for g in opt.param_groups:
            g["lr"] = cur
        idx = torch.randint(M, (batch,), device=device)
        c, tg, km = drop_conditions(cats[idx], tags[idx], n_cat, n_tags)
        kv = knobs[idx]
        with torch.no_grad():
            xs = [torch.randn(batch, TOKENS, D, device=device)]
            zero = torch.zeros(batch, device=device)
            for s in range(16):
                xs.append(xs[-1] + teacher(xs[-1], torch.full((batch,), s / 16, device=device), zero, c, tg, kv, km) / 16)
            path = torch.stack(xs)                                             # [17, B, TOKENS, D]
        for _ in range(reuse):
            n = jumps[torch.randint(len(jumps), (batch,), device=device)]
            m = 16 // n                                                        # jump length in sixteenths
            a = (torch.rand(batch, device=device) * n).long() * m              # where the jump starts
            xa, xb = path[a, rows], path[a + m, rows]
            dt = m.float() / 16
            loss = F.mse_loss(model(xa, a.float() / 16, dt, c, tg, kv, km), (xb - xa) / dt[:, None, None])
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); _ema(ema, model, step)
            step += 1
            run = loss.item() if step == 1 else 0.98 * run + 0.02 * loss.item()
            if step % 2000 == 0 or step == 1:
                rec = dict(step=step, minutes=round((time.time() - t0) / 60, 2), its=round(step / (time.time() - t0), 1), loss=round(run, 4))
                hist.append(rec); log(json.dumps(rec))
    return dict(ema=ema, hist=hist, steps=step, last_loss=run)


if __name__ == "__main__":
    n = lambda mod: sum(p.numel() for p in mod.parameters())
    print(f"teacher {n(Flow(16, 172)) / 1e6:.2f}M  student {n(Flow(16, 172, 160, 5)) / 1e6:.2f}M")
