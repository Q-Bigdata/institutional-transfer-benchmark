# fig_rev.py - revised main figures (revision round 1). Requires kernel objects built in the analysis session
# (PT, LV, BOOT2, OBS, CV, vp, FE2, hF, SD_U, MPT, DC, MB, T5, SEL, PG, MAIN2, mn); values are exported to Source_Data.xlsx.
import matplotlib as mpl, matplotlib.pyplot as plt, numpy as np, pandas as pd, os
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
os.makedirs('figs_rev', exist_ok=True)
MCOL = {'Human': '#1a1a1a', 'Haiku 4.5': '#8c6bb1', 'Sonnet 4.5': '#d95f02', 'Sonnet 4.6': '#e6ab02',
        'Opus 4.6': '#1b9e77', 'Sonnet 5': '#4a90c2', 'Opus 5': '#08306b', 'All models': '#8a8a8a'}
INSTL = {'equal': 'Equal', 'mixed': 'Mixed', 'prop': 'Prop.', 'interp': 'Interp.', 'step': 'Step', 'm1': 'RL M1', 'm2': 'RL M2'}
POPS = ['human'] + ['base|' + m for m in MODELS] + ['base|pooled']
def plab(p):
    if p == 'human': return 'Human'
    if p.endswith('pooled'): return 'All models'
    return SHORT[p.split('|')[1]]
def save(fig, name):
    fig.savefig(f'figs_rev/{name}.pdf', bbox_inches='tight'); fig.savefig(f'figs_rev/{name}.png', dpi=300, bbox_inches='tight')
    fig.savefig(f'figs_rev/{name}.tif', dpi=300, bbox_inches='tight', pil_kwargs={'compression': 'tiff_lzw'})
def overlaps(fig):
    r = fig.canvas.get_renderer()
    T = [(t, t.get_window_extent(r)) for t in fig.findobj(mpl.text.Text) if t.get_text().strip() and t.get_visible()]
    return [(a.get_text()[:25], b.get_text()[:25]) for i, (a, ba) in enumerate(T) for b, bb in T[i + 1:] if ba.overlaps(bb)]
def box(ax, x, y, w, h, txt, fc='#f2f2f2', ec='#555555', fs=6.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.008,rounding_size=0.015', fc=fc, ec=ec, lw=0.7))
    ax.text(x + w / 2, y + h / 2, txt, ha='center', va='center', fontsize=fs)
def arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>', mutation_scale=7, lw=0.7, color='#555555'))
def ci_of(pop, inst):
    b = BOOT2[BOOT2['pop'] == pop][inst + '_eff']; return b.quantile(.025), b.quantile(.975)

# ------------------------------------------------------------------ Figure 1
def fig1():
    fig = plt.figure(figsize=(7.2, 4.8))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.62, 1], hspace=0.30, wspace=0.95, width_ratios=[1, 0.95])
    ax = fig.add_subplot(gs[0, :]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
    ax.text(0.0, 0.97, 'Human evidence', fontsize=7, fontweight='bold', va='top')
    box(ax, 0.00, 0.56, 0.19, 0.28, '440 human groups\n(Koster et al., Exps 1-3)', fc='#ececec')
    box(ax, 0.25, 0.56, 0.19, 0.28, 'Fitted human\nstrategy table', fc='#ececec')
    arrow(ax, 0.19, 0.70, 0.25, 0.70)
    ax.text(0.0, 0.47, 'Machine evidence', fontsize=7, fontweight='bold', va='top')
    box(ax, 0.00, 0.06, 0.19, 0.28, '6 language models,\n8 prompt conditions', fc='#e3eef8')
    box(ax, 0.25, 0.06, 0.19, 0.28, '213 elicited tables,\nchecked by direct-play\nprobes', fc='#e3eef8')
    arrow(ax, 0.19, 0.20, 0.25, 0.20)
    box(ax, 0.51, 0.20, 0.20, 0.50, 'Shared simulator\n40 rounds, 4 seats,\n7 institutions,\n6,000 games per cell', fc='#fdf0dd')
    arrow(ax, 0.44, 0.70, 0.51, 0.58); arrow(ax, 0.44, 0.20, 0.51, 0.32)
    tests = ['Validation against observed\ninstitutions (b)', 'Leverage, Interp. - Equal\n(c; Figs 3, 4)',
             'Mechanism factorial (Fig. 5)', 'Best tested deviation (Fig. 6)']
    for k, t in enumerate(tests):
        y = 0.76 - k * 0.235; box(ax, 0.78, y, 0.22, 0.19, t, fc='#e8f1e4', fs=6); arrow(ax, 0.71, 0.45, 0.78, y + 0.095)
    panel_letter(ax, 'a')
    # b validation
    ax = fig.add_subplot(gs[1, 0]); order = ['equal', 'mixed', 'prop', 'interp', 'm1', 'm2']; x = np.arange(len(order))
    o = OBS.loc[order]
    ax.errorbar(x - 0.13, o.eff, yerr=[o.eff - o.lo, o.hi - o.eff], fmt='o', color=MCOL['Human'], ms=4, lw=1, capsize=0, label='Observed groups')
    ax.plot(x + 0.13, CV.loc[order, 'cv_pred_eff'], 's', mfc='white', mec=MCOL['Human'], ms=4, label='Simulator, held-out games')
    ax.set_xticks(x); ax.set_xticklabels([INSTL[i] for i in order], rotation=30, ha='right')
    ax.set_ylabel('Efficiency (share of maximum surplus)'); ax.set_ylim(0, 0.75)
    ax.set_title('Simulator reproduces the human ranking', loc='left')
    ax.legend(frameon=False, loc='upper left', fontsize=6, handletextpad=0.3)
    ax.text(0.98, 0.04, 'r = 0.99, Kendall $\\tau$ = 0.87', transform=ax.transAxes, ha='right', fontsize=6)
    panel_letter(ax, 'b')
    # c human leverage evidence
    ax = fig.add_subplot(gs[1, 1])
    rows = [('Observed, cross-experiment', 0.404, 0.352, 0.455, 'o'),
            ('Observed, bridged via Prop.', 0.402, 0.276, 0.529, 'o'),
            ('Simulated humans (primary)', LV.loc['human', 'lev'], LV.loc['human', 'lev_lo'], LV.loc['human', 'lev_hi'], 's'),
            ('Simulated, Exp 1 fit', vp.loc['exp1', 'lev'], np.nan, np.nan, 's'),
            ('Simulated, Exps 2-3 fit', vp.loc['exp23', 'lev'], np.nan, np.nan, 's')]
    for k, (lab, v, lo, hi, mk) in enumerate(rows):
        y = len(rows) - 1 - k
        if not np.isnan(lo): ax.plot([lo, hi], [y, y], color=MCOL['Human'], lw=1)
        ax.plot(v, y, mk, color=MCOL['Human'], mfc='white' if mk == 's' else MCOL['Human'], ms=4)
        ax.text(-0.02, y, lab, ha='right', va='center', fontsize=6, transform=ax.get_yaxis_transform())
    ax.axvline(0, color='#bbbbbb', lw=0.6); ax.set_yticks([]); ax.set_xlim(-0.02, 0.6); ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlabel('Leverage (efficiency, Interp. - Equal)')
    ax.set_title('Large leverage in human groups', loc='left')
    ax.text(-0.62, 1.02, 'c', transform=ax.transAxes, fontsize=9, fontweight='bold', va='bottom')
    save(fig, 'fig1'); return fig

# ------------------------------------------------------------------ Figure 2
FT = [('mean', 'Mean\nre-investment'), ('dR', 'Scarcity\nresponse'), ('ds', 'Reward\nresponse'), ('dq', 'Conditional\ncooperation'), ('dy', 'Inertia')]
def fig2():
    fig = plt.figure(figsize=(7.2, 5.0))
    g0 = fig.add_gridspec(2, 1, height_ratios=[1, 1.05], hspace=0.55)
    ga = g0[0].subgridspec(1, 5, wspace=0.38); gb = g0[1].subgridspec(1, 3, wspace=0.45, width_ratios=[1.25, 1, 1])
    rng = np.random.default_rng(3); base = FE2[FE2.cond == 'base']
    for j, (f, lab) in enumerate(FT):
        ax = fig.add_subplot(ga[0, j])
        ax.plot([0], [hF[f]], 'D', color=MCOL['Human'], ms=4)
        for i, m in enumerate(MODELS):
            v = base[base.model == m][f].values; c = MCOL[SHORT[m]]
            ax.scatter(i + 1 + rng.uniform(-0.18, 0.18, len(v)), v, s=4, color=c, alpha=0.75, lw=0)
            ax.plot([i + 1 - 0.3, i + 1 + 0.3], [np.median(v)] * 2, color=c, lw=1.4)
        if f != 'mean': ax.axhline(0, color='#bbbbbb', lw=0.6, zorder=0)
        ax.set_xticks(range(7)); ax.set_xticklabels(['Human table'] + MORDER, rotation=90)
        ax.set_title(lab, loc='left'); ax.margins(x=0.06, y=0.08)
        if j == 0: ax.set_ylabel('Fraction or contrast'); panel_letter(ax, 'a')
    # b table vs direct contrasts (matched pairs)
    ax = fig.add_subplot(gb[0, 0]); mk = {'R': 'o', 's': 's', 'q': '^', 'y': 'D'}
    nm = {'R': 'Scarcity', 's': 'Reward', 'q': 'Cond. coop.', 'y': 'Inertia'}
    ax.plot([-0.6, 0.6], [-0.6, 0.6], color='#bbbbbb', lw=0.6, ls='--', zorder=0)
    ax.axhline(0, color='#dddddd', lw=0.5, zorder=0); ax.axvline(0, color='#dddddd', lw=0.5, zorder=0)
    for _, r in MPT[MPT.model != 'human'].iterrows():
        ax.plot(r.table, r.direct, mk[r.feat], color=MCOL[SHORT[r.model]], ms=4, mfc='none' if r.feat in 'Rq' else MCOL[SHORT[r.model]], mew=0.9)
    for f in 'Rsqy': ax.plot([], [], mk[f], color='#555555', mfc='none' if f in 'Rq' else '#555555', ms=4, label=nm[f])
    ax.legend(frameon=False, fontsize=5.5, loc='upper left', handletextpad=0.2, borderaxespad=0.1)
    ax.set_xlabel('Contrast in elicited table'); ax.set_ylabel('Contrast in direct play')
    ax.set_xlim(-0.55, 0.55); ax.set_ylim(-0.55, 0.55)
    ax.set_title('Tables and direct play disagree', loc='left'); panel_letter(ax, 'b')
    # c reward response both modes
    ax = fig.add_subplot(gb[0, 1]); x = np.arange(6)
    tv = [MPT[(MPT.model == m) & (MPT.feat == 's')].table.iloc[0] for m in MODELS]
    dv = [MPT[(MPT.model == m) & (MPT.feat == 's')].direct.iloc[0] for m in MODELS]
    for i, m in enumerate(MODELS):
        ax.plot([i - 0.12], [tv[i]], 'o', color=MCOL[SHORT[m]], ms=4); ax.plot([i + 0.12], [dv[i]], 'o', mfc='white', color=MCOL[SHORT[m]], ms=4)
    hs = MPT[(MPT.model == 'human') & (MPT.feat == 's')].table.iloc[0]
    ax.axhline(hs, color=MCOL['Human'], ls='--', lw=1); ax.text(5.4, hs + 0.012, 'human table', ha='right', fontsize=5.5)
    ax.axhline(0, color='#bbbbbb', lw=0.6, zorder=0)
    ax.plot([], [], 'o', color='#555555', ms=4, label='table'); ax.plot([], [], 'o', mfc='white', color='#555555', ms=4, label='direct play')
    ax.legend(frameon=False, fontsize=5.5, loc='lower right', handletextpad=0.2)
    ax.set_xticks(x); ax.set_xticklabels(MORDER, rotation=40, ha='right'); ax.set_ylabel('Reward response (s = 2 vs 0.5)')
    ax.set_ylim(-0.3, 0.2); ax.set_title('No model rewards large offers', loc='left'); panel_letter(ax, 'c')
    # d between-agent heterogeneity
    ax = fig.add_subplot(gb[0, 2])
    sds = [base[base.model == m]['mean'].std() for m in MODELS]
    for i, (m, s_) in enumerate(zip(MODELS, sds)):
        ax.plot([i, i], [0, s_], color=MCOL[SHORT[m]], lw=1.2); ax.plot(i, s_, 'o', color=MCOL[SHORT[m]], ms=4)
    ax.axhline(SD_U, color=MCOL['Human'], ls='--', lw=1); ax.text(1.5, SD_U + 0.004, 'human between-player', fontsize=5.5)
    ax.set_xticks(range(6)); ax.set_xticklabels(MORDER, rotation=40, ha='right'); ax.set_ylim(0, 0.14)
    ax.set_ylabel('s.d. of agent mean re-investment'); ax.set_title('Little variation between agents', loc='left'); panel_letter(ax, 'd')
    save(fig, 'fig2'); return fig, sds

# ------------------------------------------------------------------ Figure 3
def fig3():
    fig = plt.figure(figsize=(7.2, 3.2)); gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 1, 0.9], wspace=0.5)
    ax = fig.add_subplot(gs[0, 0]); x = np.arange(len(INST))
    for p in POPS:
        lab = plab(p); v = PT.loc[p, INST].values; ci = np.array([ci_of(p, i) for i in INST])
        ls = '--' if p.endswith('pooled') else '-'
        ax.plot(x, v, ls, marker='o', color=MCOL[lab], ms=3, lw=1.6 if p == 'human' else 1, zorder=5 if p == 'human' else 3)
    ax.plot(np.array([0, 1, 2, 3, 5, 6]) + 0.0, OBS.loc[['equal', 'mixed', 'prop', 'interp', 'm1', 'm2'], 'eff'], 'x', color=MCOL['Human'], ms=4, mew=1, zorder=6)
    ends = sorted([(PT.loc[p, 'm2'], plab(p)) for p in POPS]); ys = [e[0] for e in ends]
    for k in range(1, len(ys)):
        if ys[k] - ys[k - 1] < 0.045: ys[k] = ys[k - 1] + 0.045
    for (v, l), yy in zip(ends, ys): ax.text(6.25, yy, l if l != 'Human' else 'Human (sim.)', color=MCOL[l], fontsize=5.5, va='center')
    ax.text(0.0, 0.655, 'x  observed human groups', fontsize=5.5, color=MCOL['Human'])
    ax.set_xticks(x); ax.set_xticklabels([INSTL[i] for i in INST], rotation=30, ha='right'); ax.set_xlim(-0.3, 8.0)
    ax.set_ylabel('Efficiency'); ax.set_ylim(0.08, 0.68); ax.set_title('Efficiency by institution', loc='left')
    panel_letter(ax, 'a')
    ax = fig.add_subplot(gs[0, 1]); order = POPS; y = np.arange(len(order))[::-1]
    ax.axvspan(-0.05, 0.05, color='#eeeeee', zorder=0); ax.axvline(0, color='#bbbbbb', lw=0.6)
    for yy, p in zip(y, order):
        l = plab(p); r = LV.loc[p]
        ax.plot([r.lev_lo, r.lev_hi], [yy, yy], color=MCOL[l], lw=1.2); ax.plot(r.lev, yy, 'o', color=MCOL[l], ms=4)
    ax.set_yticks(y); ax.set_yticklabels(['Human (sim.)' if plab(p) == 'Human' else plab(p) for p in order])
    ax.set_xlabel('Leverage (Interp. - Equal)'); ax.set_xlim(-0.08, 0.45)
    ax.text(0.0, len(order) - 0.35, '$\\pm$0.05', ha='center', fontsize=5.5, color='#777777')
    ax.set_ylim(-0.6, len(order) - 0.1)
    ax.set_title('Leverage, 95% CI', loc='left'); panel_letter(ax, 'b')
    ax = fig.add_subplot(gs[0, 2])
    for yy, p in zip(y, order):
        l = plab(p); r = LV.loc[p]
        ax.plot([r.r7_lo, r.r7_hi], [yy, yy], color=MCOL[l], lw=1.2); ax.plot(r.r7, yy, 'o', color=MCOL[l], ms=4)
    ax.set_yticks(y); ax.set_yticklabels([]); ax.set_xlabel('Range across 7 institutions'); ax.set_xlim(0, 0.45)
    ax.set_ylim(-0.6, len(order) - 0.1)
    ax.set_title('Best minus worst rule', loc='left'); panel_letter(ax, 'c')
    save(fig, 'fig3'); return fig

# ------------------------------------------------------------------ Figure 4
CONDS = [('base', 'Default'), ('base_think', 'Extended reasoning'), ('persona', 'Personas'), ('neutral', 'Neutral wording'),
         ('neutral_mixed', 'Mixed wording (v1)'), ('expl_prop', 'Rule explained: Prop.'), ('expl_interp', 'Rule explained: Interp.'),
         ('expl_rl', 'Rule explained: RL M2')]
DM = ['claude-sonnet-4-6', 'claude-opus-5']
def fig4():
    fig = plt.figure(figsize=(7.2, 3.1)); gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.1, 0.8], wspace=0.25)
    y = np.arange(len(CONDS))[::-1]; off = {DM[0]: 0.15, DM[1]: -0.15}
    ax = fig.add_subplot(gs[0, 0]); ax.axvspan(-0.05, 0.05, color='#eeeeee', zorder=0); ax.axvline(0, color='#bbbbbb', lw=0.6)
    ax.axvline(LV.loc['human', 'lev'], color=MCOL['Human'], ls='--', lw=0.9); ax.text(LV.loc['human', 'lev'] + 0.005, -0.75, 'human (sim.)', fontsize=5.5)
    for m in DM:
        for yy, (c, _) in zip(y, CONDS):
            r = LV.loc[f'{c}|{m}']; col = MCOL[SHORT[m]]
            ax.plot([r.lev_lo, r.lev_hi], [yy + off[m]] * 2, color=col, lw=1.1); ax.plot(r.lev, yy + off[m], 'o', color=col, ms=3.5)
    ax.set_yticks(y); ax.set_yticklabels([l for _, l in CONDS]); ax.set_xlim(-0.06, 0.3); ax.set_ylim(-1, len(CONDS) - 0.4)
    ax.set_xlabel('Leverage (Interp. - Equal)'); ax.set_title('No prompt restores human leverage', loc='left')
    for m in DM: ax.plot([], [], 'o', color=MCOL[SHORT[m]], ms=3.5, label=SHORT[m])
    ax.legend(frameon=False, loc='center right', fontsize=5.5, handletextpad=0.2)
    panel_letter(ax, 'a', dx=-0.62)
    ax = fig.add_subplot(gs[0, 1])
    for m in DM:
        for yy, (c, _) in zip(y, CONDS):
            p = f'{c}|{m}'; col = MCOL[SHORT[m]]; e0, e1 = PT.loc[p, 'equal'], PT.loc[p, 'interp']
            ax.plot([e0, e1], [yy + off[m]] * 2, color=col, lw=0.9); ax.plot(e0, yy + off[m], 'o', mfc='white', color=col, ms=3.5); ax.plot(e1, yy + off[m], 'o', color=col, ms=3.5)
    ax.plot([], [], 'o', mfc='white', color='#555555', ms=3.5, label='Equal'); ax.plot([], [], 'o', color='#555555', ms=3.5, label='Interp.')
    ax.legend(frameon=False, loc='lower right', fontsize=5.5, handletextpad=0.2)
    ax.set_yticks(y); ax.set_yticklabels([]); ax.set_ylim(-1, len(CONDS) - 0.4); ax.set_xlim(0, 0.95)
    ax.set_xlabel('Efficiency'); ax.set_title('Prompts move the level, not the gap', loc='left'); panel_letter(ax, 'b')
    ax = fig.add_subplot(gs[0, 2])
    for m in DM:
        for yy, (c, _) in zip(y, CONDS):
            p = f'{c}|{m}'; col = MCOL[SHORT[m]]
            ax.plot(SUST.loc[p, 'interp'], yy + off[m], 'o', color=col, ms=3.5); ax.plot(SUST.loc[p, 'equal'], yy + off[m], 'o', mfc='white', color=col, ms=3.5)
    ax.set_yticks(y); ax.set_yticklabels([]); ax.set_ylim(-1, len(CONDS) - 0.4); ax.set_xlim(-0.05, 1.05)
    ax.set_xlabel('Share of games sustained'); ax.set_title('Pool survival', loc='left'); panel_letter(ax, 'c')
    save(fig, 'fig4'); return fig

# ------------------------------------------------------------------ Figure 5
VARS = [('', 'Elicited tables'), ('Ronly', 'Pool-only tables'), ('flat_s', 'Reward response removed'), ('hum_s', 'Human reward response'),
        ('hum_R', 'Human scarcity response'), ('hum_q', 'Human cond. cooperation'), ('hum_y', 'Human inertia'), ('mu', 'Human mean level'),
        ('humoff', 'Human between-player offsets'), ('sig15', 'Heterogeneity, s.d. 0.15'), ('sig15mm', 'Heterogeneity, mean-matched'),
        ('flat_s+sig15', 'Heterogeneity, no reward response'), ('hum_s+sig15', 'Heterogeneity + human reward resp.'),
        ('allmain', 'All human main effects'), ('allmain+first+humoff+noise', 'Full human substitution'),
        ('dpadj', 'Direct-play-adjusted tables'), ('dpadj+sig15', 'Direct-play-adjusted + heterogeneity')]
def fig5():
    fig = plt.figure(figsize=(7.2, 4.2)); gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1], wspace=0.32)
    ax = fig.add_subplot(gs[0, 0]); M = L5.loc[[v for v, _ in VARS]].values
    cmap = mpl.colormaps['PuOr_r']; norm = mpl.colors.TwoSlopeNorm(vmin=-0.5, vcenter=0, vmax=0.5)
    ax.imshow(M, cmap=cmap, norm=norm, aspect='auto')
    lo = T5[0.025].unstack(0).loc[[v for v, _ in VARS], MODELS].values
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f'{M[i, j]:.2f}' + ('' if lo[i, j] > 0 else '\u2009\u00b0'), ha='center', va='center', fontsize=5.5,
                    color='white' if abs(M[i, j]) > 0.33 else '#1a1a1a')
    ax.set_xticks(range(6)); ax.set_xticklabels(MORDER, rotation=35, ha='right'); ax.set_yticks(range(len(VARS))); ax.set_yticklabels([l for _, l in VARS])
    for yy in [0.5, 8.5, 12.5, 14.5]: ax.axhline(yy, color='white', lw=1.5)
    ax.tick_params(length=0); [s_.set_visible(False) for s_ in ax.spines.values()]
    ax.set_title('Leverage after replacing one ingredient', loc='left'); panel_letter(ax, 'a', dx=-0.75)
    ax = fig.add_subplot(gs[0, 1])
    keep = ['', 'Ronly', 'flat_s', 'hum_s', 'humoff', 'sig15', 'sig15mm', 'flat_s+sig15', 'hum_s+sig15', 'dpadj', 'dpadj+sig15']
    for m in MODELS:
        d = T5.loc[m].loc[keep]; ax.scatter(d.prop_sd, d.lev, s=12, color=MCOL[SHORT[m]], lw=0, alpha=0.9, label=SHORT[m])
    ax.axhline(LV.loc['human', 'lev'], color=MCOL['Human'], ls='--', lw=0.9); ax.text(0.003, LV.loc['human', 'lev'] + 0.01, 'human (sim.)', fontsize=5.5)
    ax.axhline(0, color='#bbbbbb', lw=0.6)
    ax.set_xlabel('Realized between-agent s.d. of re-investment'); ax.set_ylabel('Leverage (Interp. - Equal)')
    ax.legend(frameon=False, fontsize=5.5, loc='upper left', handletextpad=0.1, ncol=2, columnspacing=0.6)
    ax.set_ylim(-0.05, 0.42); ax.set_xlim(0, 0.19)
    ax.set_title('Heterogeneity predicts leverage', loc='left'); panel_letter(ax, 'b')
    save(fig, 'fig5'); return fig

# ------------------------------------------------------------------ Figure 6
def fig6():
    fig = plt.figure(figsize=(7.2, 3.1)); gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1, 1], wspace=0.42)
    ax = fig.add_subplot(gs[0, 0]); x = np.arange(len(INST))
    for p in POPS:
        l = plab(p); d = SEL[SEL['pop'] == p].set_index('inst').loc[INST]
        ax.plot(x, d.gain_eval, '--' if p.endswith('pooled') else '-', marker='o', ms=3, color=MCOL[l], lw=1.6 if p == 'human' else 1, zorder=5 if p == 'human' else 3, label='Human (sim.)' if l == 'Human' else l)
    ax.legend(frameon=False, fontsize=5.5, loc='upper left', ncol=2, handlelength=1.4, columnspacing=0.6, handletextpad=0.3, bbox_to_anchor=(0, 1.0))
    ax.set_ylim(-0.15, 3.6)
    ax.axhline(0, color='#bbbbbb', lw=0.6)
    ax.set_xticks(x); ax.set_xticklabels([INSTL[i] for i in INST], rotation=30, ha='right')
    ax.set_ylabel('Gain of best tested deviation\n(fair shares, fresh seeds)'); ax.set_title('Exploitability by rule', loc='left')
    ends = sorted([(SEL[(SEL['pop'] == p) & (SEL.inst == 'prop')].gain_eval.iloc[0], plab(p)) for p in POPS])
    panel_letter(ax, 'a')
    ax = fig.add_subplot(gs[0, 1]); order = POPS; yv = np.arange(len(order))[::-1]
    for yy, p in zip(yv, order):
        l = plab(p); d = SEL[SEL['pop'] == p].set_index('inst')
        ax.plot([0.25, d.loc['interp', 'dev_offer_share']], [yy, yy], color=MCOL[l], lw=0.8)
        ax.plot(d.loc['interp', 'dev_offer_share'], yy, 'o', color=MCOL[l], ms=4); ax.plot(d.loc['prop', 'dev_offer_share'], yy, 'D', mfc='white', color=MCOL[l], ms=3.5)
    ax.axvline(0.25, color='#bbbbbb', lw=0.6); ax.text(0.26, len(order) - 0.45, 'equal share', fontsize=5.5, color='#777777')
    ax.plot([], [], 'o', color='#555555', ms=4, label='Interp.'); ax.plot([], [], 'D', mfc='white', color='#555555', ms=3.5, label='Prop.')
    ax.legend(frameon=False, fontsize=5.5, loc='upper right', ncol=2, handletextpad=0.2, columnspacing=0.6)
    ax.set_yticks(yv); ax.set_yticklabels(['Human (sim.)' if plab(p) == 'Human' else plab(p) for p in order]); ax.set_xlim(0.2, 1.0); ax.set_ylim(-0.6, len(order) + 0.5)
    ax.set_xlabel("Deviator's share of all offers"); ax.set_title('Capture concentrates offers', loc='left'); panel_letter(ax, 'b', dx=-0.55)
    ax = fig.add_subplot(gs[0, 2]); insts = ['equal', 'prop', 'interp', 'm2']; rng = np.random.default_rng(5)
    for k, i in enumerate(insts):
        for m, mk in zip(DM, ['o', 's']):
            d = PG[(PG.inst == i) & (PG.model == m)]
            for _, r in d.iterrows():
                l = r.who; xx = k + (-0.15 if m == DM[0] else 0.15) + rng.uniform(-0.06, 0.06)
                ax.plot(xx, r.gain, mk, color=MCOL[l if l != 'Human sim.' else 'Human'], ms=3, alpha=0.9)
    ax.axhline(0, color='#bbbbbb', lw=0.6)
    for m, mk in zip(DM, ['o', 's']): ax.plot([], [], mk, color='#555555', ms=3, label=f'{SHORT[m]} prompted')
    ax.legend(frameon=False, fontsize=5.5, loc='lower left', handletextpad=0.2)
    ax.set_xticks(range(len(insts))); ax.set_xticklabels([INSTL[i] for i in insts]); ax.set_ylim(-0.6, 0.3)
    ax.set_ylabel('Gain of prompted exploiter (fair shares)'); ax.set_title('Prompted exploiters do not gain', loc='left'); panel_letter(ax, 'c')
    save(fig, 'fig6'); return fig

# ------------------------------------------------------------------ Extended Data
def ed1():
    fig = plt.figure(figsize=(7.2, 2.9)); gs = fig.add_gridspec(1, 2, width_ratios=[1.3, 1], wspace=0.35)
    ax = fig.add_subplot(gs[0, 0]); cols = [('frac_R_lo', 'Pool below grid (R < 40)'), ('frac_s_out', 'Offer ratio outside [0.5, 2]'), ('frac_w0', 'Interp. weight = 0 (= Prop.)')]
    pops = POPS; x = np.arange(len(pops)); wd = 0.26; hat = ['', '///', '...']
    for k, (c, lab) in enumerate(cols):
        v = [dg.loc[(p, 'interp'), c] for p in pops]
        ax.bar(x + (k - 1) * wd, v, wd, color=[MCOL[plab(p)] for p in pops], hatch=hat[k], edgecolor='white', lw=0.3, label=lab)
    ax.set_xticks(x); ax.set_xticklabels(['Human (sim.)' if plab(p) == 'Human' else plab(p) for p in pops], rotation=35, ha='right')
    ax.set_ylabel('Share of decisions or rounds (Interp.)'); ax.set_ylim(0, 1.25)
    h = [mpl.patches.Patch(facecolor='#999999', hatch=hat[k], edgecolor='white', label=l) for k, (_, l) in enumerate(cols)]
    ax.legend(handles=h, frameon=False, fontsize=5.5, loc='upper left', ncol=1)
    ax.set_title('Simulated play often leaves the elicited grid', loc='left'); panel_letter(ax, 'a')
    ax = fig.add_subplot(gs[0, 1]); vlab = {'none': 'Clamped (main)', 'extrap': 'Linear extrapolation', 'reset0': 'Reset to 0', 'continuous': 'Continuous coins'}
    mk = {'none': 'o', 'extrap': 's', 'reset0': '^', 'continuous': 'D'}
    for k, p in enumerate(pops):
        for j, v in enumerate(vlab):
            ax.plot(SENS.loc[p, v], len(pops) - 1 - k + (j - 1.5) * 0.15, mk[v], color=MCOL[plab(p)], ms=3, mfc='white' if v != 'none' else MCOL[plab(p)])
    for v in vlab: ax.plot([], [], mk[v], color='#555555', ms=3, mfc='white' if v != 'none' else '#555555', label=vlab[v])
    ax.legend(frameon=False, fontsize=5.5, loc='lower right')
    ax.set_yticks(range(len(pops))[::-1]); ax.set_yticklabels(['Human (sim.)' if plab(p) == 'Human' else plab(p) for p in pops])
    ax.axvline(0, color='#bbbbbb', lw=0.6); ax.set_xlabel('Leverage (Interp. - Equal)'); ax.set_xlim(-0.03, 0.42)
    ax.set_title('Boundary conventions change little', loc='left'); panel_letter(ax, 'b', dx=-0.4)
    save(fig, 'ed1'); return fig

def ed2():
    fig = plt.figure(figsize=(7.2, 2.7)); gs = fig.add_gridspec(1, 3, wspace=0.45)
    ax = fig.add_subplot(gs[0, 0]); fits = ['full', 'fold0', 'fold1', 'fold2', 'fold3', 'fold4', 'exp1', 'exp23']
    labs = ['All data', 'Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5', 'Exp 1 only', 'Exps 2-3 only']
    extra = [('No M1 groups', 0.261), ('Group-weighted', 0.218)]
    vals = [vp.loc[f, 'lev'] for f in fits] + [e[1] for e in extra]; L = labs + [e[0] for e in extra]
    y = np.arange(len(vals))[::-1]
    ax.plot(vals, y, 's', mfc='white', color=MCOL['Human'], ms=4); ax.axvline(0.404, color='#999999', ls=':', lw=0.9)
    ax.text(0.40, len(vals) - 0.4, 'observed', fontsize=5.5, color='#777777', ha='right')
    ax.set_yticks(y); ax.set_yticklabels(L); ax.set_xlim(0, 0.45); ax.set_xlabel('Simulated human leverage')
    ax.set_title('Stable across fits', loc='left'); panel_letter(ax, 'a', dx=-0.6)
    ax = fig.add_subplot(gs[0, 1]); tk = TUNE2[TUNE2['pop'] == 'human']
    for fam, ls, lab in [('k', '-', 'k, unrounded'), ('kr', '--', 'k, rounded weight')]:
        d = tk[tk.fam == fam].sort_values('v'); d = d[d.v > 0]; ax.plot(d.v, d.eff, ls, marker='o', ms=2.5, color=MCOL['Human'], lw=0.9, label=lab)
    ax.set_xscale('log'); ax.axvline(22, color='#999999', ls=':', lw=0.8); ax.text(23, 0.30, 'k = 22', fontsize=5.5, color='#777777')
    ax.set_xticks([1, 4, 16, 64, 256]); ax.set_xticklabels(['1', '4', '16', '64', '256'])
    ax.set_xlabel('Interp. exponent k'); ax.set_ylabel('Simulated human efficiency'); ax.legend(frameon=False, fontsize=5.5, loc='lower right')
    ax.set_title('Flat optimum near k = 22', loc='left'); panel_letter(ax, 'b')
    ax = fig.add_subplot(gs[0, 2]); strata = ['<40', '40-80', '80-120', '120-160', '160-200']; xx = np.arange(5)
    for k, (key, mkk) in enumerate([('m1', 'o'), ('m2', 's')]):
        ax.plot(xx + (k - 0.5) * 0.15, SURE[key].loc[strata, 'mae_share'], mkk, color=MCOL['Human'], mfc='white' if key == 'm2' else MCOL['Human'], ms=4, label=f'RL {key.upper()} offer shares')
    ax.plot(xx + 0.075, SURE['m2'].loc[strata, 'mae_alloc'], '^', color='#777777', ms=4, label='RL M2 allocated fraction')
    ax.set_xticks(xx); ax.set_xticklabels(strata, rotation=30, ha='right'); ax.set_xlabel('Pool size R'); ax.set_ylabel('Held-out mean absolute error')
    ax.legend(frameon=False, fontsize=5.5, loc='upper right'); ax.set_ylim(0, 0.08)
    ax.set_title('Surrogates are accurate', loc='left'); panel_letter(ax, 'c')
    save(fig, 'ed2'); return fig

def ed3():
    fig = plt.figure(figsize=(7.2, 2.8)); gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1], wspace=0.55)
    ax = fig.add_subplot(gs[0, 0]); pops = list(lr.index); pops = ['base|' + m for m in MODELS] + [f'{c}|{m}' for c, _ in CONDS[1:] for m in DM]
    pops = [p for p in pops if p in lr.index]; y = np.arange(len(pops))[::-1]
    for yy, p in zip(y, pops):
        m = p.split('|')[1]; c = MCOL[SHORT[m]]
        ax.plot([lr.loc[p, 'min'], lr.loc[p, 'max']], [yy, yy], color=c, lw=2); ax.plot(LV.loc[p, 'lev'], yy, '|', color='#1a1a1a', ms=5)
    cl = dict(CONDS)
    ax.set_yticks(y); ax.set_yticklabels([f"{SHORT[p.split('|')[1]]}, {cl[p.split('|')[0]].lower()}" if not p.startswith('base|') else SHORT[p.split('|')[1]] for p in pops], fontsize=5.5)
    ax.axvline(0, color='#bbbbbb', lw=0.6); ax.set_xlabel('Leverage, leave-one-agent-out range'); ax.set_xlim(-0.02, 0.35)
    ax.set_title('No single agent drives leverage', loc='left'); panel_letter(ax, 'a', dx=-0.75)
    ax = fig.add_subplot(gs[0, 1]); keys = list(VDV.keys()); y = np.arange(len(keys))[::-1]
    lefts = np.zeros(len(keys)); cols = ['#4a4a4a', '#e6ab02', '#cccccc']; names = ['Population', 'Institution', 'Interaction/residual']
    for c in range(3):
        v = np.array([VDV[k][c] for k in keys]); ax.barh(y, v, left=lefts, color=cols[c], height=0.6, label=names[c]); lefts += v
    ax.set_yticks(y); ax.set_yticklabels(keys, fontsize=5.5); ax.set_xlim(0, 1); ax.set_xlabel('Share of variance in efficiency')
    ax.legend(frameon=False, fontsize=5.5, loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=3)
    ax.set_title('Population dominates (descriptive)', loc='left'); panel_letter(ax, 'b', dx=-0.6)
    save(fig, 'ed3'); return fig

def ed4():
    fig = plt.figure(figsize=(7.2, 2.7)); gs = fig.add_gridspec(1, 3, wspace=0.45)
    ax = fig.add_subplot(gs[0, 0])
    for p in POPS:
        d = SEL[SEL['pop'] == p]; ax.plot(d.gain_sel, d.gain_eval, 'o', color=MCOL[plab(p)], ms=3, label='Human (sim.)' if plab(p) == 'Human' else plab(p))
    ax.legend(frameon=False, fontsize=5.5, loc='upper left', handletextpad=0.1, borderaxespad=0.1)
    ax.plot([0, 2.8], [0, 2.8], color='#bbbbbb', ls='--', lw=0.6); ax.set_xlabel('Gain, selection seeds'); ax.set_ylabel('Gain, fresh evaluation seeds')
    ax.set_title('Negligible selection bias', loc='left'); panel_letter(ax, 'a')
    ax = fig.add_subplot(gs[0, 1])
    for p in POPS:
        a = SEL[SEL['pop'] == p].set_index('inst').loc[INST]; b = SELn[SELn['pop'] == p].set_index('inst').loc[INST]
        ax.plot(a.gain_eval, b.gain_eval, 'o', color=MCOL[plab(p)], ms=3)
    ax.plot([0, 2.8], [0, 2.8], color='#bbbbbb', ls='--', lw=0.6); ax.set_xlabel('Gain, noiseless incumbents'); ax.set_ylabel('Gain, incumbents with human noise')
    ax.set_title('Noise does not protect', loc='left'); panel_letter(ax, 'b')
    ax = fig.add_subplot(gs[0, 2]); x = np.arange(len(INST))
    for p in POPS:
        d = SELpw[SELpw['pop'] == p].set_index('inst').loc[INST]
        ax.plot(x, d.ratio_pw, '--' if p.endswith('pooled') else '-', marker='o', ms=2.5, lw=0.9, color=MCOL[plab(p)])
    ax.axhline(1, color='#bbbbbb', lw=0.6); ax.set_yscale('log'); ax.set_yticks([1, 2, 5, 10]); ax.set_yticklabels(['1', '2', '5', '10'])
    ax.set_xticks(x); ax.set_xticklabels([INSTL[i] for i in INST], rotation=30, ha='right')
    ax.set_ylabel('Deviator / incumbent earnings\n(payment-weighted)'); ax.set_title('Same pattern for paid rounds', loc='left'); panel_letter(ax, 'c')
    save(fig, 'ed4'); return fig
