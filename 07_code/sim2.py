
"""sim2.py - revised simulator for the institutional-transfer benchmark (revision round 1).
Adds: two planner surrogates (M1, M2) with an alternative specification, payment-weighted surplus, boundary and reset
diagnostics, optional linear extrapolation beyond the strategy grid, alternative reset convention, and per-seat records."""
import numpy as np, commons, zlib
CAP, P, GROWTH = 200.0, 4, 0.4
CE = commons.CEILING
PAYW = (40 - np.arange(40)) / 40.0          # P(randomly chosen payment round >= t), t = 1..40
GR, GLS, GQ, GY = commons.GRID_R, commons.GRID_LS, commons.GRID_Q, commons.GRID_Y
def seed_of(*a): return zlib.crc32(repr(a).encode()) & 0x7fffffff

def _axis_ext(x, g):
    i = np.where(x < g[1], 0, 1)
    w = (x - g[i]) / (g[i + 1] - g[i])
    return i, w
def table_eval_ext(T, R, s, q, y):
    """Multilinear interpolation with linear extrapolation outside the grid (result later clipped to [0,1])."""
    ls = np.log2(np.maximum(s, 1e-6))
    ax = [_axis_ext(R, GR), _axis_ext(ls, GLS), _axis_ext(q, GQ), _axis_ext(y, GY)]
    out = 0.0
    for corner in range(16):
        bits = [(corner >> k) & 1 for k in range(4)]
        flat = np.zeros(np.shape(R), dtype=np.int64); w = np.ones(np.shape(R))
        for k, (i, wk) in enumerate(ax):
            flat = flat * 3 + (i + bits[k]); w = w * (wk if bits[k] else (1 - wk))
        out = out + np.take_along_axis(T, flat[..., None], axis=-1)[..., 0] * w
    return out

class Surrogate:
    FE = ['R', 'Rprev', 'gret', 'cshare', 'yprev', 'oshare', 'cshare2', 'rank', 'tt']
    FE_ALT = ['R', 'Rprev', 'gret', 'cshare', 'yprev', 'tt']
    GF = ['R', 'Rprev', 'gret', 'tt']
    def __init__(self, df, alt=False, seed=0):
        from sklearn.ensemble import HistGradientBoostingRegressor as HGB
        self.alt = alt; fe = self.FE_ALT if alt else self.FE
        kw = dict(max_iter=200, learning_rate=0.1, max_depth=4, random_state=seed) if alt else dict(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, random_state=seed)
        self.ms = HGB(**kw).fit(df[fe].values, df.share.values)
        grp = df.groupby(['gid', 't']).first().reset_index()
        self.ma = HGB(max_iter=300, learning_rate=0.05, random_state=seed).fit(grp[self.GF].values, grp.alloc.values)
        X = df[fe].values
        self.lo = np.quantile(X, 0.005, axis=0); self.hi = np.quantile(X, 0.995, axis=0)
        self.fe = fe
    def __call__(self, R, c_prev, state, first):
        G = R.shape[0]
        if state['t'] == 0:
            return np.repeat(R[:, None] / 4, 4, 1)
        o_prev, c2, Rp = state['o_prev'], state['c_prev2'], state['R_prev']
        sc = c_prev.sum(1, keepdims=True); so = o_prev.sum(1, keepdims=True); s2 = c2.sum(1, keepdims=True)
        cshare = np.where(sc > 0, c_prev / np.maximum(sc, 1e-9), .25)
        yprev = np.where(o_prev >= 1, c_prev / np.maximum(o_prev, 1e-9), 0.)
        oshare = np.where(so > 0, o_prev / np.maximum(so, 1e-9), .25)
        cshare2 = np.where(s2 > 0, c2 / np.maximum(s2, 1e-9), .25)
        rank = (c_prev[:, :, None] > c_prev[:, None, :]).sum(2) + 0.5 * ((c_prev[:, :, None] == c_prev[:, None, :]).sum(2) - 1)
        gret = np.where(so[:, 0] > 0, sc[:, 0] / np.maximum(so[:, 0], 1e-9), 0.)
        tt = np.full(G, state['t'] / 40)
        full = dict(R=np.repeat((R / 200)[:, None], 4, 1), Rprev=np.repeat((Rp / 200)[:, None], 4, 1), gret=np.repeat(gret[:, None], 4, 1),
                    cshare=cshare, yprev=yprev, oshare=oshare, cshare2=cshare2, rank=rank, tt=np.repeat(tt[:, None], 4, 1))
        X = np.stack([full[f] for f in self.fe], -1).reshape(-1, len(self.fe))
        if state.get('diag') is not None:
            live = np.repeat(R >= 1, 4)
            out = ((X < self.lo) | (X > self.hi)).any(1)
            state['diag']['sur_n'] += int(live.sum()); state['diag']['sur_out'] += int((out & live).sum())
        sh = np.clip(self.ms.predict(X).reshape(G, 4), 0, None)
        sh = sh / np.maximum(sh.sum(1, keepdims=True), 1e-9)
        sh = np.where(sh.sum(1, keepdims=True) > 0, sh, .25)
        al = np.clip(self.ma.predict(np.stack([R / 200, Rp / 200, gret, tt], -1)), 0, 1)
        al = np.where(R < 1, 0, al)
        return R[:, None] * al[:, None] * sh

def weights_for(mech, R, params):
    if mech == 'equal': return np.ones_like(R)
    if mech == 'mixed': return np.full_like(R, 0.5)
    if mech == 'prop': return np.zeros_like(R)
    if mech == 'interp': return commons.w_interp(R, 22.0, True)
    if mech == 'interpk': return commons.w_interp(R, params['k'], params.get('rounding', False))
    if mech == 'step': return (R >= params.get('theta', 194.0)).astype(float)
    raise ValueError(mech)

def simulate2(mech, T, F, U=None, E=None, rng=None, params=None, policy=None, extrap=False, reset='half',
              integer=True, override=None, noise_mask=None, diag=False, n_rounds=40, keep_seat=False, drift=False):
    rng = rng or np.random.default_rng(0); params = params or {}
    G = T.shape[0]
    R = np.full(G, CAP); c_prev = np.zeros((G, P)); o_prev = np.zeros((G, P)); c_prev2 = np.zeros((G, P)); R_prev = np.full(G, CAP)
    surplus = np.zeros((G, P)); surplus_pw = np.zeros((G, P)); offers_tot = np.zeros((G, P)); active = np.zeros(G); Rsum = np.zeros(G)
    U = np.zeros((G, P)) if U is None else U
    nm = np.ones((G, P)) if noise_mask is None else noise_mask
    dg = dict(n=0, R_lo=0, s_out=0, q_reset=0, y_reset=0, w0=0, w1=0, rounds=0, sur_n=0, sur_out=0) if diag else None
    dr = np.zeros((3, 20)) if drift else None
    for t in range(n_rounds):
        first = np.full(G, t == 0)
        if mech in ('m1', 'm2', 'm1alt', 'm2alt'):
            st = dict(o_prev=o_prev, c_prev2=c_prev2, R_prev=R_prev, t=t, diag=dg)
            o = policy(R, c_prev, st, first)
        else:
            w = weights_for(mech, R, params); w = np.where(first, 1.0, w)
            sc = c_prev.sum(1, keepdims=True)
            prop = np.where(sc > 0, c_prev / np.maximum(sc, 1e-12), 0.25)
            o = R[:, None] * (w[:, None] / 4 + (1 - w[:, None]) * prop)
            if diag and t > 0:
                live = R >= 1; dg['rounds'] += int(live.sum()); dg['w0'] += int(((w == 0) & live).sum()); dg['w1'] += int(((w == 1) & live).sum())
        s = o / np.maximum(R[:, None] / 4, 1e-9)
        so = o_prev.sum(1, keepdims=True) - o_prev
        qr = so < 1; yr = o_prev < 1
        q = np.where(qr, 0.5, (c_prev.sum(1, keepdims=True) - c_prev) / np.maximum(so, 1e-9))
        yp = np.where(yr, 0.5 if reset == 'half' else 0.0, c_prev / np.maximum(o_prev, 1e-9))
        if t == 0:
            y = F.copy()
        else:
            Rb = np.broadcast_to(R[:, None], (G, P))
            y = table_eval_ext(T, Rb, s, q, yp) if extrap else commons.table_eval(T, Rb, s, q, yp)
        y = y + U * nm
        if E is not None: y = y + rng.choice(E, size=y.shape) * nm
        if override is not None: y[:, 0] = override(R, o[:, 0], s[:, 0], q[:, 0], yp[:, 0], t)
        y = np.clip(y, 0, 1)
        c = y * o
        if integer: c = np.floor(c + 0.5)
        c = np.where(o < 1, 0.0, np.minimum(c, np.floor(o + 1e-9) if integer else o))
        if diag and t > 0:
            act = o >= 1
            dg['n'] += int(act.sum()); dg['R_lo'] += int((act & (R[:, None] < 40)).sum())
            dg['s_out'] += int((act & ((s < 0.5) | (s > 2))).sum()); dg['q_reset'] += int((act & qr).sum()); dg['y_reset'] += int((act & yr).sum())
        if drift and t >= 10:
            live = R >= 1; b = np.clip((R / 10).astype(int), 0, 19)
            np.add.at(dr[0], b[live], c.sum(1)[live]); np.add.at(dr[1], b[live], o.sum(1)[live]); np.add.at(dr[2], b[live], 1)
        surplus += o - c; surplus_pw += (o - c) * PAYW[t]; offers_tot += o
        active += (o >= 1).sum(1)
        if t >= 10: Rsum += R
        Rn = np.minimum(CAP, R - o.sum(1) + (1 + GROWTH) * c.sum(1)); Rn = np.maximum(Rn, 0.0)
        c_prev2, c_prev, o_prev, R_prev = c_prev, c, o, R
        R = np.where(Rn < 1, 0.0, Rn)
    out = dict(eff=surplus.sum(1) / CE, eff_pw=surplus_pw.sum(1) / (CE * PAYW.mean()), gini=commons.gini_rows(surplus),
               sustained=(R >= 1).astype(float), active=active / n_rounds, meanR=Rsum / (n_rounds - 10))
    if keep_seat: out['seat_surplus'] = surplus; out['seat_offers'] = offers_tot; out['seat_surplus_pw'] = surplus_pw
    if diag: out['diag'] = dg
    if drift: out['drift'] = dr
    return out

def summ(o):
    r = {k: float(np.mean(o[k])) for k in ['eff', 'eff_pw', 'gini', 'sustained', 'active', 'meanR']}
    r['eff_se'] = float(np.std(o['eff']) / np.sqrt(len(o['eff'])))
    if 'diag' in o:
        d = o['diag']; n = max(d['n'], 1)
        r.update(frac_R_lo=d['R_lo'] / n, frac_s_out=d['s_out'] / n, frac_q_reset=d['q_reset'] / n, frac_y_reset=d['y_reset'] / n,
                 frac_w0=d['w0'] / max(d['rounds'], 1), frac_w1=d['w1'] / max(d['rounds'], 1), frac_sur_out=d['sur_out'] / max(d['sur_n'], 1) if d['sur_n'] else np.nan)
    return r
