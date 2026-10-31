
"""commons.py - simulator and strategy-table machinery for the Koster et al. (2025) common-pool trust game.
Game: p=4 players, pool R0=200 (cap), growth r=0.4. Each round the mechanism offers o_i (sum <= R);
player i re-invests c_i in [0, o_i] (integer coins; offers < 1 force c_i = 0);
R_{t+1} = min(200, R_t - sum(o) + (1+r) * sum(c)). Surplus s_i = o_i - c_i.
"""
import numpy as np

CAP = 200.0
GROWTH = 0.4
P = 4
# strategy-table grid (shared by human-fitted and LLM-elicited tables)
GRID_R = np.array([40.0, 120.0, 200.0])       # pool size
GRID_LS = np.array([-1.0, 0.0, 1.0])          # log2 of own offer relative to equal share (0.5x, 1x, 2x)
GRID_Q = np.array([0.2, 0.5, 0.8])            # others' previous re-investment fraction
GRID_Y = np.array([0.2, 0.5, 0.8])            # own previous re-investment fraction
SHAPE = (3, 3, 3, 3)
NCELL = 81

def cell_list():
    """Enumerate the 81 cells in canonical order (R, s, q, yprev)."""
    out = []
    for a, R in enumerate(GRID_R):
        for b, ls in enumerate(GRID_LS):
            for c, q in enumerate(GRID_Q):
                for d, y in enumerate(GRID_Y):
                    out.append((a, b, c, d, R, 2.0 ** ls, q, y))
    return out

def _axis(x, grid):
    x = np.clip(x, grid[0], grid[-1])
    i = np.clip(np.searchsorted(grid, x, side='right') - 1, 0, len(grid) - 2)
    w = (x - grid[i]) / (grid[i + 1] - grid[i])
    return i, w

def interp_weights(R, s, q, y):
    """Return (flat corner indices [n,16], weights [n,16]) for multilinear interpolation."""
    ls = np.log2(np.maximum(s, 1e-6))
    idx = []; wts = []
    ax = [_axis(R, GRID_R), _axis(ls, GRID_LS), _axis(q, GRID_Q), _axis(y, GRID_Y)]
    n = np.shape(R)
    for corner in range(16):
        bits = [(corner >> k) & 1 for k in range(4)]
        flat = np.zeros(n, dtype=np.int64); w = np.ones(n)
        for k, (i, wk) in enumerate(ax):
            ii = i + bits[k]
            flat = flat * 3 + ii
            w = w * (wk if bits[k] else (1 - wk))
        idx.append(flat); wts.append(w)
    return np.stack(idx, -1), np.stack(wts, -1)

def table_eval(T, R, s, q, y):
    """T: array [..., 81] (per-agent tables broadcast to R's shape + (81,))."""
    idx, w = interp_weights(R, s, q, y)
    vals = np.take_along_axis(T, idx, axis=-1)
    return (vals * w).sum(-1)

# ---------------------------------------------------------------- mechanisms
def w_interp(R, k=22.0, rounding=True):
    w = (np.clip(R, 0, CAP) / CAP) ** k
    return np.round(w, 1) if rounding else w

def allocate(mech, R, c_prev, first, params=None, m2=None, state=None):
    """Return offers [G,4]. mech in {'equal','mixed','prop','interp','step','interpk','m2'}."""
    G = R.shape[0]
    sc = c_prev.sum(1, keepdims=True)
    prop = np.where(sc > 0, c_prev / np.maximum(sc, 1e-12), 0.25)
    if mech == 'equal': w = np.ones(G)
    elif mech == 'mixed': w = np.full(G, 0.5)
    elif mech == 'prop': w = np.zeros(G)
    elif mech == 'interp': w = w_interp(R, 22.0, True)
    elif mech == 'interpk': w = w_interp(R, params['k'], False)
    elif mech == 'step': w = (R >= params.get('theta', 194.0)).astype(float)
    elif mech == 'm2':
        return m2(R, c_prev, state, first)
    elif mech == 'reserve':
        w = w_interp(R, params.get('k', 22.0), params.get('rounding', True))
        w = np.where(first, 1.0, w)
        a = np.where(first, 1.0, params['a'])
        return (a * R)[:, None] * (w[:, None] / 4 + (1 - w[:, None]) * prop)
    else: raise ValueError(mech)
    w = np.where(first, 1.0, w)
    offers = R[:, None] * (w[:, None] / 4 + (1 - w[:, None]) * prop)
    return offers

def simulate(mech, agents_T, first_val, u=None, eps_sd=0.0, n_rounds=40, rng=None, params=None,
             m2=None, integer=True, growth=GROWTH, return_traj=False, eps_pool=None, override=None, noise_mask=None):
    """agents_T: [G,4,81] tables; first_val: [G,4] first-round re-investment fraction;
    u: [G,4] additive individual offsets; eps_sd: Gaussian noise sd; eps_pool: 1-d array of empirical
    residuals resampled instead of Gaussian noise; override: f(R,o,s,q,yp,t)->[G] re-investment fraction
    for seat 0 (a deviant); noise_mask: [G,4] 1 where noise applies.
    Returns dict of per-game outcomes."""
    rng = rng or np.random.default_rng(0)
    G = agents_T.shape[0]
    R = np.full(G, CAP)
    c_prev = np.zeros((G, P)); o_prev = np.zeros((G, P)); c_prev2 = np.zeros((G, P)); R_prev = np.full(G, CAP)
    surplus = np.zeros((G, P)); active = np.zeros(G); depl_round = np.full(G, n_rounds)
    u = np.zeros((G, P)) if u is None else u
    nm = np.ones((G, P)) if noise_mask is None else noise_mask
    yrec = []; Rrec = []
    for t in range(n_rounds):
        first = np.full(G, t == 0)
        state = dict(o_prev=o_prev, c_prev2=c_prev2, R_prev=R_prev, t=t)
        o = allocate(mech, R, c_prev, first, params, m2, state)
        s = o / np.maximum(R[:, None] / 4, 1e-9)
        so = o_prev.sum(1, keepdims=True) - o_prev
        q = np.where(so >= 1, (c_prev.sum(1, keepdims=True) - c_prev) / np.maximum(so, 1e-9), 0.5)
        yp = np.where(o_prev >= 1, c_prev / np.maximum(o_prev, 1e-9), 0.5)
        if t == 0:
            y = first_val.copy()
        else:
            Rb = np.broadcast_to(R[:, None], (G, P))
            y = table_eval(agents_T, Rb, s, q, yp)
        y = y + u * nm
        if eps_pool is not None:
            y = y + rng.choice(eps_pool, size=y.shape) * nm
        elif eps_sd > 0:
            y = y + rng.normal(0, eps_sd, y.shape) * nm
        if override is not None:
            y[:, 0] = override(R, o[:, 0], s[:, 0], q[:, 0], yp[:, 0], t)
        y = np.clip(y, 0, 1)
        c = y * o
        if integer: c = np.floor(c + 0.5)
        c = np.where(o < 1, 0.0, np.minimum(c, np.floor(o)))
        surplus += o - c
        active += (o >= 1).sum(1)
        Rn = np.minimum(CAP, R - o.sum(1) + (1 + growth) * c.sum(1))
        Rn = np.maximum(Rn, 0.0)
        newly = (Rn < 1) & (depl_round == n_rounds)
        depl_round[newly] = t + 1
        if return_traj: yrec.append(np.where(o >= 1, c / np.maximum(o, 1e-9), np.nan)); Rrec.append(R.copy())
        c_prev2, c_prev, o_prev, R_prev = c_prev, c, o, R
        R = np.where(Rn < 1, 0.0, Rn)
    tot = surplus.sum(1)
    out = dict(surplus=tot, gini=gini_rows(surplus), sustained=(R >= 1).astype(float),
               active=active / n_rounds, depletion_round=depl_round, player_surplus=surplus, final_pool=R)
    if return_traj: out['y'] = np.stack(yrec, 1); out['R'] = np.stack(Rrec, 1)
    return out

def gini_rows(x):
    x = np.sort(np.asarray(x, float), axis=1); n = x.shape[1]
    cum = np.cumsum(x, 1); tot = cum[:, -1]
    g = (n + 1 - 2 * (cum.sum(1) / np.maximum(tot, 1e-12))) / n
    return np.where(tot > 0, g, 0.0)

CEILING = 40 * CAP * GROWTH / (1 + GROWTH)   # sustainable surplus ceiling (2285.7)
