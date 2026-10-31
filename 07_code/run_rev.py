
"""run_rev.py - all revision simulations. Usage: python3 run_rev.py <stage> ; stages: prep main boot loo valid tune mech exploit all"""
import os, sys, json, time, numpy as np, pandas as pd, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import commons, sim2
from sim2 import simulate2, summ, seed_of
from multiprocessing import Pool
import scipy.sparse as sp
from scipy.sparse.linalg import lsqr
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN = os.path.join(ROOT, '08_derived_data', 'inputs')
OUT = os.environ.get('INSTITUTION_RESULTS_DIR', os.path.join(ROOT, 'output', 'simulations')); os.makedirs(OUT, exist_ok=True)
T0 = time.time()
def log(*a): print('[%6.0fs]' % (time.time() - T0), *a, flush=True)
AG = json.load(open(os.path.join(IN, 'agents_v2.json')))
DP = json.load(open(os.path.join(IN, 'direct_contrasts.json')))
PEN = np.load(os.path.join(IN, 'pen.npy'))
HDF = pd.read_csv(os.path.join(IN, 'human_decisions_full.csv.gz'))
HFIRST = pd.read_csv(os.path.join(IN, 'human_first.csv.gz'))
MODELS = ['claude-haiku-4-5-20251001', 'claude-sonnet-4-5-20250929', 'claude-sonnet-4-6', 'claude-opus-4-6', 'claude-sonnet-5', 'claude-opus-5']
DEP_MODELS = ['claude-sonnet-4-6', 'claude-opus-5']
DEP_CONDS = ['base_think', 'persona', 'neutral', 'neutral_mixed', 'expl_prop', 'expl_interp', 'expl_rl']
INST = ['equal', 'mixed', 'prop', 'interp', 'step', 'm1', 'm2']
SUR = {}
def get_sur(name):
    if name not in SUR:
        p = os.path.join(OUT, 'sur_%s.pkl' % name)
        SUR[name] = pickle.load(open(p, 'rb'))
    return SUR[name]
# ---------------------------------------------------------------- human table fits
def fit_table(df, lam=5.0, weights=None):
    idx, w = commons.interp_weights(df.R.values, df.s.values, df.q.values, df.yprev.values)
    n = len(df); A = sp.csr_matrix((w.ravel(), (np.repeat(np.arange(n), 16), idx.ravel())), shape=(n, 81)); y = df.y.values
    if weights is not None:
        sw = np.sqrt(weights); A = sp.diags(sw) @ A; y = y * sw
    A2 = sp.vstack([A, sp.csr_matrix(PEN * np.sqrt(lam))]); b = np.r_[y, np.zeros(len(PEN))]
    return lsqr(A2, b, atol=1e-10, btol=1e-10, iter_lim=20000)[0]
def human_model(df, first_df, weights=None):
    T = fit_table(df, weights=weights)
    idx, w = commons.interp_weights(df.R.values, df.s.values, df.q.values, df.yprev.values)
    res = df.y.values - (T[idx] * w).sum(1)
    d = pd.DataFrame(dict(pid=df.pid.values, res=res))
    pm = d.groupby('pid').res.agg(['mean', 'count'])
    within = (d.res - d.groupby('pid').res.transform('mean')).values; s2w = within.var()
    s2u = max(pm['mean'].var() - s2w / pm['count'].mean(), 1e-6)
    u = (pm['mean'] * pm['count'] / (pm['count'] + s2w / s2u)).values
    return dict(T=T, U=u, E=within.astype(np.float32), F=first_df.y.values)
GIDX = None
def resample_games(df, first_df, rng):
    global GIDX
    if GIDX is None:
        GIDX = (df.groupby('game').indices, first_df.groupby('game').indices)
    games = np.array(sorted(GIDX[0])); gs = rng.choice(games, len(games))
    di = np.concatenate([GIDX[0][g] for g in gs]); fi = np.concatenate([GIDX[1][g] for g in gs])
    tag = np.concatenate([np.full(len(GIDX[0][g]), k) for k, g in enumerate(gs)])
    a = df.iloc[di].copy(); a['pid'] = a.pid.values + '_b' + tag.astype(str)
    return a, first_df.iloc[fi]
def boot_model(b):
    rng = np.random.default_rng(seed_of('hboot', b))
    df, fd = resample_games(HDF, HFIRST, rng); hm = human_model(df, fd)
    hm['E'] = np.random.default_rng(b).choice(hm['E'], 8000).astype(np.float32)
    return hm
# ---------------------------------------------------------------- populations
def tables(cond, model):
    sel = [a for a in AG if a['cond'] == cond and (model == 'pooled' or a['model'] == model)]
    return np.array([a['T'] for a in sel]), np.array([a['F'] for a in sel]), np.array([a['model'] for a in sel])
def decompose(T):
    G = T.reshape(-1, 3, 3, 3, 3); mu = G.mean(axis=(1, 2, 3, 4), keepdims=True)
    mains = []
    for ax in range(4):
        axes = tuple(1 + i for i in range(4) if i != ax)
        mains.append(G.mean(axis=axes, keepdims=True) - mu)
    return G, mu, mains
HUMAN_FULL = None
def transform(Ts, Fs, var, model=None, rng=None):
    """Return transformed tables, first-round values, and a flag for offsets/noise."""
    Th = HUMAN_FULL['T'].reshape(1, 3, 3, 3, 3)
    G, mu, mains = decompose(Ts); _, muh, mainsh = decompose(HUMAN_FULL['T'][None])
    parts = set(var.split('+')) if var else set()
    H = G.copy()
    if 'allmain' in parts:
        H = H - mu + muh
        for k in range(4): H = H - mains[k] + mainsh[k]
    if 'mu' in parts: H = H - mu + muh
    for k, nm in enumerate(['R', 's', 'q', 'y']):
        if 'hum_' + nm in parts: H = H - mains[k] + mainsh[k]
        if 'flat_' + nm in parts: H = H - mains[k]
    if 'Ronly' in parts:
        H = np.broadcast_to(mu + mains[0], H.shape).copy()
    if 'dpadj' in parts:
        d = DP[model]
        for k, nm in enumerate(['R', 's', 'q', 'y']):
            delta = -d['R'] if nm == 'R' else d[nm]          # hi minus lo along the axis
            prof = np.array([-delta / 2, 0, delta / 2]); shape = [1, 1, 1, 1, 1]; shape[k + 1] = 3
            H = H - mains[k] + prof.reshape(shape)
    Fo = Fs.copy()
    if 'first' in parts: Fo = None
    return np.clip(H.reshape(len(Ts), 81), 0, 1), Fo
def make_pop(spec, G, rng):
    """spec: dict(kind='human'|'machine', cond, model, var, sigma, off ('human'|None), noise (bool), agent_idx, hm (human model), mean_match)"""
    if spec['kind'] == 'human':
        hm = spec.get('hm') or HUMAN_FULL
        T = np.broadcast_to(hm['T'], (G, 4, 81)).copy(); F = np.clip(rng.choice(hm['F'], (G, 4)), 0, 1)
        U = rng.choice(hm['U'], (G, 4)); return T, F, U, hm['E']
    Ts, Fs, ms = tables(spec['cond'], spec['model'])
    if spec.get('agent_idx') is not None: Ts, Fs, ms = Ts[spec['agent_idx']], Fs[spec['agent_idx']], ms[spec['agent_idx']]
    var = spec.get('var') or ''
    if var:
        Ts, Fo = transform(Ts, Fs, var, spec['model'])
        Fs = Fo if Fo is not None else None
    ix = rng.integers(0, len(Ts), (G, 4)); T = Ts[ix]
    F = Fs[ix] if Fs is not None else np.clip(rng.choice(HUMAN_FULL['F'], (G, 4)), 0, 1)
    U = None
    if spec.get('sigma'):
        U = rng.normal(0, spec['sigma'], (G, 4))
        if spec.get('mean_match'):
            # shift offsets so that the mean of clipped re-investment over the grid equals the unperturbed mean
            base = T.mean(); lo, hi = -0.3, 0.3
            for _ in range(30):
                mid = (lo + hi) / 2
                m = np.clip(T + (U + mid)[..., None], 0, 1).mean()
                lo, hi = (mid, hi) if m < base else (lo, mid)
            U = U + (lo + hi) / 2
    if spec.get('off') == 'human': U = rng.choice(HUMAN_FULL['U'], (G, 4)) if U is None else U + rng.choice(HUMAN_FULL['U'], (G, 4))
    E = HUMAN_FULL['E'] if spec.get('noise') else None
    return T, F, U, E
def policy_for(inst):
    return get_sur(inst) if inst in ('m1', 'm2', 'm1alt', 'm2alt') else None
def run(spec, inst, G, seed, params=None, **kw):
    rng = np.random.default_rng(seed)
    T, F, U, E = make_pop(spec, G, rng)
    p = params or {'theta': 194.0}
    return simulate2(inst, T, F, U=U, E=E, rng=rng, params=p, policy=policy_for(inst), **kw)
def stratified_idx(models_arr, rng):
    idx = []
    for m in np.unique(models_arr):
        w = np.where(models_arr == m)[0]; idx.extend(rng.choice(w, len(w)))
    return np.array(idx)
# ---------------------------------------------------------------- populations list
def pop_specs():
    P = {'human': dict(kind='human')}
    for m in MODELS: P['base|' + m] = dict(kind='machine', cond='base', model=m)
    P['base|pooled'] = dict(kind='machine', cond='base', model='pooled')
    for m in DEP_MODELS:
        for c in DEP_CONDS: P[c + '|' + m] = dict(kind='machine', cond=c, model=m)
    for m in MODELS: P['dpadj|' + m] = dict(kind='machine', cond='base', model=m, var='dpadj')
    return P
# ---------------------------------------------------------------- tasks
def t_main(a):
    name, inst, variant = a
    spec = pop_specs()[name]; kw = {}
    if variant == 'extrap': kw['extrap'] = True
    if variant == 'reset0': kw['reset'] = 'zero'
    if variant == 'continuous': kw['integer'] = False
    if variant == 'altsur':
        inst = inst + 'alt'
    o = run(spec, inst, 6000, seed_of('main', name), diag=True, drift=(variant == 'none'), **kw)
    r = summ(o); r.update(pop=name, inst=inst.replace('alt', ''), variant=variant)
    if 'drift' in o: r['drift'] = o['drift'].tolist()
    return r
def t_boot(a):
    name, b = a
    spec = dict(pop_specs()[name]); rng = np.random.default_rng(seed_of('boot', name, b))
    if spec['kind'] == 'human':
        spec['hm'] = BOOT_HM[b]
    else:
        Ts, Fs, ms = tables(spec['cond'], spec['model'])
        spec['agent_idx'] = stratified_idx(ms, rng)
    out = dict(pop=name, b=b)
    for inst in INST:
        o = run(spec, inst, 1000, seed_of('bootsim', name, b))
        for k in ['eff', 'eff_pw', 'sustained', 'gini']: out[inst + '_' + k] = float(np.mean(o[k]))
    return out
def t_loo(a):
    name, j = a
    spec = dict(pop_specs()[name]); Ts, Fs, ms = tables(spec['cond'], spec['model'])
    spec['agent_idx'] = np.array([i for i in range(len(Ts)) if i != j])
    out = dict(pop=name, left_out=j)
    for inst in INST:
        o = run(spec, inst, 3000, seed_of('loo', name)); out[inst] = float(np.mean(o['eff']))
    return out
def t_valid(a):
    label, hmkey, inst = a
    spec = dict(kind='human', hm=VALID_HM[hmkey])
    o = run(spec, inst, 6000, seed_of('valid', hmkey))
    r = summ(o); r.update(label=label, fit=hmkey, inst=inst); return r
KG = [0.0, 0.5, 1, 2, 3, 4, 6, 8, 11, 16, 22, 32, 45, 64, 128, 256]
THG = [100, 120, 140, 150, 160, 170, 180, 185, 190, 194, 197, 200]
def t_tune(a):
    name, fam, v = a
    spec = pop_specs()[name]
    if fam == 'k': o = run(spec, 'interpk', 4000, seed_of('tune', name), params={'k': v, 'rounding': False})
    elif fam == 'kr': o = run(spec, 'interpk', 4000, seed_of('tune', name), params={'k': v, 'rounding': True})
    else: o = run(spec, 'step', 4000, seed_of('tune', name), params={'theta': v})
    r = summ(o); r.update(pop=name, fam=fam, v=v); return r
MECH_VARS = ['', 'sig15', 'sig15mm', 'flat_s', 'flat_s+sig15', 'hum_s', 'hum_s+sig15', 'humoff', 'mu', 'hum_R', 'hum_q', 'hum_y',
             'allmain', 'allmain+first', 'allmain+first+humoff', 'allmain+first+noise', 'allmain+first+humoff+noise', 'allmain+humoff+noise',
             'Ronly', 'Ronly_cont', 'dpadj', 'dpadj+sig15']
def mech_spec(model, v):
    parts = v.split('+') if v else []
    spec = dict(kind='machine', cond='base', model=model)
    tv = [p for p in parts if p not in ('sig15', 'sig15mm', 'humoff', 'noise', 'Ronly_cont')]
    if 'Ronly_cont' in parts: tv.append('Ronly')
    spec['var'] = '+'.join(tv)
    if 'sig15' in parts: spec['sigma'] = 0.15
    if 'sig15mm' in parts: spec['sigma'] = 0.15; spec['mean_match'] = True
    if 'humoff' in parts: spec['off'] = 'human'
    if 'noise' in parts: spec['noise'] = True
    return spec, ('Ronly_cont' in parts)
def t_mech_point(a):
    model, v, inst = a
    spec, cont = mech_spec(model, v)
    rng = np.random.default_rng(seed_of('mech', model, v))
    T, F, U, E = make_pop(spec, 4000, rng)
    o = simulate2(inst, T, F, U=U, E=E, rng=rng, params={'theta': 194.0}, policy=policy_for(inst), integer=not cont, keep_seat=True)
    r = summ(o); r.update(model=model, var=v, inst=inst)
    # realized propensity diagnostics and selection slope
    prop = np.clip(T + (U[..., None] if U is not None else 0), 0, 1).mean(-1)       # [G,4]
    r['prop_mean'] = float(prop.mean()); r['prop_sd'] = float(prop.std())
    r['hit_frac'] = float(((T + (U[..., None] if U is not None else 0)) < 0).mean() + ((T + (U[..., None] if U is not None else 0)) > 1).mean())
    sh = o['seat_offers'] / np.maximum(o['seat_offers'].sum(1, keepdims=True), 1e-9)
    pc = prop - prop.mean(1, keepdims=True); shc = sh - 0.25
    r['sel_slope'] = float((pc * shc).sum() / max((pc ** 2).sum(), 1e-12))
    return r
def t_mech_boot(a):
    model, v, b = a
    spec, cont = mech_spec(model, v); rng = np.random.default_rng(seed_of('mboot', model, v, b))
    Ts, Fs, ms = tables('base', model); spec['agent_idx'] = rng.choice(len(Ts), len(Ts))
    out = dict(model=model, var=v, b=b)
    for inst in ['equal', 'interp']:
        rr = np.random.default_rng(seed_of('mbootsim', model, v, b))
        T, F, U, E = make_pop(spec, 1000, rr)
        o = simulate2(inst, T, F, U=U, E=E, rng=rr, params={'theta': 194.0}, integer=not cont)
        out[inst] = float(np.mean(o['eff']))
    return out
# ---- exploitability
DEV = [('const', p, None, None) for p in np.round(np.linspace(0, 1, 21), 2)] + \
      [('thresh', lo, hi, th) for lo in [0, .2, .4, .6, .8, 1] for hi in [0, .2, .4, .6, .8, 1] for th in [100, 150, 190] if lo != hi]
def dev_fn(j):
    kind, a, b, c = DEV[j]
    if kind == 'const': return lambda R, o, s, q, yp, t: np.full(R.shape, a)
    return lambda R, o, s, q, yp, t: np.where(R >= c, b, a)
EX_POPS = ['human'] + ['base|' + m for m in MODELS] + ['base|pooled']
def ex_eval(spec, inst, j, seedtag, G=3000, noise=False, dev_table=None):
    sp_ = dict(spec)
    if noise and sp_['kind'] == 'machine': sp_['noise'] = True
    rng = np.random.default_rng(seed_of('ex', seedtag, sp_.get('model', 'h'), inst))
    T, F, U, E = make_pop(sp_, G, rng)
    mask = np.ones((G, 4)); kw = {}
    if j == -1:      # baseline: unperturbed game with the same draws
        pass
    elif j == -2:    # prompted strategic table in seat 0
        T = T.copy(); F = F.copy(); T[:, 0, :] = dev_table[0]; F[:, 0] = dev_table[1]; mask[:, 0] = 0
    else:
        kw['override'] = dev_fn(j); mask[:, 0] = 0
    o = simulate2(inst, T, F, U=U, E=E, rng=rng, params={'theta': 194.0}, policy=policy_for(inst), noise_mask=mask, keep_seat=True, **kw)
    ss, so_, sp = o['seat_surplus'], o['seat_offers'], o['seat_surplus_pw']
    tot = ss.sum(1)
    return dict(dev=float(ss[:, 0].mean()), others=float(ss[:, 1:].mean()), group=float(tot.mean()),
                dev_pw=float(sp[:, 0].mean()), others_pw=float(sp[:, 1:].mean()),
                dev_offer_share=float((so_[:, 0] / np.maximum(so_.sum(1), 1e-9)).mean()),
                dev_surplus_share=float((ss[:, 0] / np.maximum(tot, 1e-9)).mean()), gini=float(np.mean(commons.gini_rows(ss))),
                seat0_base=float(ss[:, 0].mean()), eff=float(tot.mean() / sim2.CE))
def t_exploit(a):
    name, inst, j, seedtag, noise = a
    r = ex_eval(pop_specs()[name], inst, j, seedtag, noise=noise); r.update(pop=name, inst=inst, j=j, seedtag=seedtag, noise=noise); return r
def t_prompted(a):
    name, inst, k = a
    strat = [x for x in AG if x['cond'] == 'strategic' and x['mech'] == inst][k]
    r = ex_eval(pop_specs()[name], inst, -2, 'B', dev_table=(np.array(strat['T']), strat['F']))
    r.update(pop=name, inst=inst, j=-2, model=strat['model'], rep=strat['rep']); return r
def t_exboot(a):
    name, inst, j, b = a
    spec = dict(pop_specs()[name]); rng = np.random.default_rng(seed_of('exb', name, inst, b))
    if spec['kind'] == 'human': spec['hm'] = BOOT_HM[b % len(BOOT_HM)]
    else:
        Ts, Fs, ms = tables(spec['cond'], spec['model']); spec['agent_idx'] = stratified_idx(ms, rng)
    base = ex_eval(spec, inst, -1, 'C%d' % b, G=1000); dev = ex_eval(spec, inst, j, 'C%d' % b, G=1000)
    return dict(pop=name, inst=inst, j=j, b=b, gain=(dev['dev'] - base['others']) / (sim2.CE / 4),
                others_change=dev['others'] / max(base['others'], 1e-9) - 1, group_change=dev['group'] / max(base['group'], 1e-9) - 1,
                d_others_abs=dev['others'] - base['others'], d_group_abs=dev['group'] - base['group'])
# ---------------------------------------------------------------- stage driver
BOOT_HM = None; VALID_HM = None
def prep():
    import pandas as pd
    from sim2 import Surrogate
    for nm in ['m1', 'm2']:
        df = pd.read_csv(os.path.join(IN, '%s_train.csv.gz' % nm))
        pickle.dump(Surrogate(df), open(os.path.join(OUT, 'sur_%s.pkl' % nm), 'wb'))
        pickle.dump(Surrogate(df, alt=True, seed=1), open(os.path.join(OUT, 'sur_%salt.pkl' % nm), 'wb'))
        # held-out evaluation: 20 human games out
        hg = df[df.src == 'human'].gid.unique(); rs = np.random.default_rng(0); test = set(rs.choice(hg, 20, replace=False))
        tr = df[~df.gid.isin(test)]; te = df[df.gid.isin(test)].copy()
        s = Surrogate(tr); X = te[s.fe].values
        te['p'] = np.clip(s.ms.predict(X), 0, None); te['pn'] = te.p / te.groupby(['gid', 't']).p.transform('sum')
        g = te.groupby(['gid', 't']).first().reset_index(); g['alloc_hat'] = np.clip(s.ma.predict(g[s.GF].values), 0, 1)
        te[['gid', 't', 'player', 'R', 'share', 'pn', 'alloc']].to_csv(os.path.join(OUT, 'sur_%s_holdout_shares.csv' % nm), index=False)
        g[['gid', 't', 'R', 'alloc', 'alloc_hat']].to_csv(os.path.join(OUT, 'sur_%s_holdout_alloc.csv' % nm), index=False)
    log('surrogates')
def build_human_models(nboot):
    global HUMAN_FULL, BOOT_HM, VALID_HM
    HUMAN_FULL = human_model(HDF, HFIRST)
    p = os.path.join(OUT, 'human_models.pkl')
    if os.path.exists(p):
        d = pickle.load(open(p, 'rb')); BOOT_HM, VALID_HM = d['boot'], d['valid']; return
    with Pool(int(os.environ.get('NPROC', '96'))) as pool:
        BOOT_HM = pool.map(boot_model, range(nboot), chunksize=4)
    VALID_HM = {'full': HUMAN_FULL}
    games = HDF.game.unique(); rs = np.random.default_rng(7); fold = dict(zip(games, rs.integers(0, 5, len(games))))
    for k in range(5):
        tr = HDF[HDF.game.map(fold) != k]; fr = HFIRST[HFIRST.game.map(fold) != k]; VALID_HM['fold%d' % k] = human_model(tr, fr)
    VALID_HM['exp1'] = human_model(HDF[HDF.exp == 1], HFIRST[HFIRST.exp == 1])
    VALID_HM['exp23'] = human_model(HDF[HDF.exp > 1], HFIRST[HFIRST.exp > 1])
    m = ~HDF.mech_name.str.contains('M1'); VALID_HM['noM1'] = human_model(HDF[m], HFIRST[~HFIRST.mech_name.str.contains('M1')])
    w = 1.0 / HDF.groupby('game').game.transform('size').values; w = w / w.mean(); VALID_HM['groupw'] = human_model(HDF, HFIRST, weights=w)
    json.dump({'fold': {str(k): int(v) for k, v in fold.items()}}, open(os.path.join(OUT, 'folds.json'), 'w'))
    pickle.dump(dict(boot=BOOT_HM, valid=VALID_HM), open(p, 'wb'))
def init_worker():
    global HUMAN_FULL, BOOT_HM, VALID_HM
    HUMAN_FULL = human_model(HDF, HFIRST)
    d = pickle.load(open(os.path.join(OUT, 'human_models.pkl'), 'rb')); BOOT_HM, VALID_HM = d['boot'], d['valid']
if __name__ == '__main__':
    stage = sys.argv[1]; NPROC = int(os.environ.get('NPROC', '96'))
    if stage in ('prep', 'all'): prep(); build_human_models(int(os.environ.get('NBOOT_H', '1000'))); log('human models')
    if stage == 'prep': sys.exit(0)
    build_human_models(int(os.environ.get('NBOOT_H', '1000')))
    P = pop_specs()
    with Pool(NPROC, initializer=init_worker) as pool:
        if stage in ('main', 'all'):
            tasks = [(n, i, 'none') for n in P for i in INST]
            core = ['human'] + ['base|' + m for m in MODELS] + ['base|pooled']
            tasks += [(n, i, v) for n in core for i in INST for v in ['extrap', 'reset0', 'continuous'] if not (v == 'continuous' and False)]
            tasks += [(n, i, 'altsur') for n in core for i in ['m1', 'm2']]
            res = pool.map(t_main, tasks, chunksize=1); json.dump(res, open(os.path.join(OUT, 'main.json'), 'w')); log('main', len(res))
        if stage in ('boot', 'all'):
            NB = int(os.environ.get('NBOOT', '1000'))
            core = ['human'] + ['base|' + m for m in MODELS] + ['base|pooled']
            dep = [c + '|' + m for m in DEP_MODELS for c in DEP_CONDS]
            tasks = [(n, b) for n in core for b in range(NB)] + [(n, b) for n in dep for b in range(NB // 2)]
            res = pool.map(t_boot, tasks, chunksize=4); pd.DataFrame(res).to_csv(os.path.join(OUT, 'boot.csv'), index=False); log('boot', len(res))
        if stage in ('loo', 'all'):
            tasks = []
            for n in ['base|' + m for m in MODELS] + [c + '|' + m for m in DEP_MODELS for c in DEP_CONDS]:
                c, m = n.split('|'); k = len(tables(c, m)[0]); tasks += [(n, j) for j in range(k)]
            res = pool.map(t_loo, tasks, chunksize=1); pd.DataFrame(res).to_csv(os.path.join(OUT, 'loo.csv'), index=False); log('loo', len(res))
        if stage in ('valid', 'all'):
            tasks = [(k, k, i) for k in VALID_HM for i in INST]
            res = pool.map(t_valid, tasks, chunksize=1); pd.DataFrame(res).to_csv(os.path.join(OUT, 'valid.csv'), index=False); log('valid', len(res))
        if stage in ('tune', 'all'):
            core = ['human'] + ['base|' + m for m in MODELS] + ['base|pooled']
            tasks = [(n, f, v) for n in core for f in ['k', 'kr'] for v in KG] + [(n, 'th', v) for n in core for v in THG]
            res = pool.map(t_tune, tasks, chunksize=1); pd.DataFrame(res).to_csv(os.path.join(OUT, 'tune.csv'), index=False); log('tune', len(res))
        if stage in ('mech', 'all'):
            tasks = [(m, v, i) for m in MODELS for v in MECH_VARS for i in ['equal', 'prop', 'interp', 'm2']]
            res = pool.map(t_mech_point, tasks, chunksize=1); pd.DataFrame(res).to_csv(os.path.join(OUT, 'mech_point.csv'), index=False); log('mech point', len(res))
            NBM = int(os.environ.get('NBOOT_M', '200'))
            tasks = [(m, v, b) for m in MODELS for v in MECH_VARS for b in range(NBM)]
            res = pool.map(t_mech_boot, tasks, chunksize=4); pd.DataFrame(res).to_csv(os.path.join(OUT, 'mech_boot.csv'), index=False); log('mech boot', len(res))
        if stage in ('exploit', 'all'):
            tasks = [(n, i, j, s, False) for n in EX_POPS for i in INST for j in [-1] + list(range(len(DEV))) for s in ['A', 'B']]
            tasks += [(n, i, j, 'B', True) for n in EX_POPS for i in INST for j in [-1] + list(range(len(DEV)))]
            res = pool.map(t_exploit, tasks, chunksize=2); EXR = pd.DataFrame(res); EXR.to_csv(os.path.join(OUT, 'exploit.csv'), index=False); log('exploit', len(res))
            nstr = {i: len([x for x in AG if x['cond'] == 'strategic' and x['mech'] == i]) for i in INST}
            tasks = [(n, i, k) for n in EX_POPS for i in INST for k in range(nstr.get(i, 0))]
            res = pool.map(t_prompted, tasks, chunksize=1); pd.DataFrame(res).to_csv(os.path.join(OUT, 'prompted.csv'), index=False); log('prompted', len(res))
            # selected on A, bootstrap uncertainty for the selected strategy
            A = EXR[(EXR.seedtag == 'A') & (~EXR.noise)]; base = A[A.j == -1].set_index(['pop', 'inst']).others
            A = A[A.j >= 0].copy(); A['gain'] = (A.dev - A.set_index(['pop', 'inst']).index.map(base).values)
            sel = A.loc[A.groupby(['pop', 'inst']).gain.idxmax(), ['pop', 'inst', 'j']]
            NBX = int(os.environ.get('NBOOT_X', '200'))
            tasks = [(r.pop, r.inst, int(r.j), b) for r in sel.itertuples() for b in range(NBX)]
            res = pool.map(t_exboot, tasks, chunksize=2); pd.DataFrame(res).to_csv(os.path.join(OUT, 'exploit_boot.csv'), index=False); log('exploit boot', len(res))
    log('DONE', stage)
