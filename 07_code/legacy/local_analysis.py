"""local_analysis.py - data preparation, human strategy table, M2 surrogate training data and LLM elicitation helpers.
Run order: (1) download sustainable_behavior.csv (see README); (2) python local_analysis.py prepares 08_derived_data inputs;
(3) run_sims.py and run_attrib.py (server) produce the simulation outputs; (4) figure code in figures/ uses the objects below.
The elicitation helpers call the Anthropic API through the host platform's `host.llm` wrapper used in this study;
replace `call_batch` with a direct API client to rerun elicitation."""
import os, re, json, time, numpy as np, pandas as pd, scipy.sparse as sp
from scipy.sparse.linalg import lsqr
import commons, elicit
def decisions(G, label_cols):
    recs=[]
    for key,g in G.groupby('launch_id'):
        g=g.sort_values('round_id')
        Og=g[[f'offer_{p}' for p in range(4)]].values; Cg=g[[f'player_action_{p}' for p in range(4)]].values; Rg=g['mechanism_observation.pool'].values
        lab=g.iloc[0][label_cols].to_dict()
        for t in range(len(g)):
            for i in range(4):
                o=Og[t,i]
                if o<1: continue
                if t>0:
                    oth=[j for j in range(4) if j!=i]
                    so=Og[t-1,oth].sum(); q=Cg[t-1,oth].sum()/so if so>=1 else np.nan
                    op=Og[t-1,i]; yprev=Cg[t-1,i]/op if op>=1 else np.nan
                else: q=np.nan; yprev=np.nan; op=np.nan
                recs.append(dict(game=key,t=t,player=i,R=Rg[t],o=o,s=o/(Rg[t]/4),q=q,yprev=yprev,oprev=op,y=Cg[t,i]/o,**lab))
    return pd.DataFrame(recs)


def game_outcomes(G):
    recs=[]
    for key,g in G.groupby('launch_id'):
        g=g.sort_values('round_id')
        Og=g[[f'offer_{p}' for p in range(4)]].values; Cg=g[[f'player_action_{p}' for p in range(4)]].values; Rg=g['mechanism_observation.pool'].values
        Sp=(Og-Cg).sum(0)
        Rlast=min(200,Rg[-1]-Og[-1].sum()+1.4*Cg[-1].sum())
        recs.append(dict(game=key,mech=g.mech_name.iloc[0],label=g.mech_name_by_player.iloc[0],surplus=Sp.sum(),
                         gini=commons.gini_rows(Sp[None])[0],sustained=float(Rlast>=1),active=(Og>=1).sum(1).mean(),
                         reward_check=g[[f'player_reward_{p}' for p in range(4)]].values.sum()))
    return pd.DataFrame(recs)


def smooth_pen():
    rows=[]
    for ax in range(4):
        for a in np.ndindex(*commons.SHAPE):
            if a[ax]==1:
                lo=list(a); lo[ax]=0; hi=list(a); hi[ax]=2
                r=np.zeros(81); r[np.ravel_multi_index(lo,commons.SHAPE)]=1; r[np.ravel_multi_index(a,commons.SHAPE)]=-2; r[np.ravel_multi_index(hi,commons.SHAPE)]=1
                rows.append(r)
    return np.array(rows)


def fit_table(df,lam=5.0):
    idx,w=commons.interp_weights(df.R.values,df.s.values,df.q.values,df.yprev.values)
    n=len(df); A=sp.csr_matrix((w.ravel(),(np.repeat(np.arange(n),16),idx.ravel())),shape=(n,81))
    A2=sp.vstack([A,sp.csr_matrix(PEN*np.sqrt(lam))]); b=np.r_[df.y.values,np.zeros(len(PEN))]
    T=lsqr(A2,b,atol=1e-10,btol=1e-10,iter_lim=20000)[0]
    return T, A


def m2_rows(M):
    recs=[]
    for gid,g in M.groupby('gid'):
        g=g.sort_values('round_id')
        Og=g[[f'offer_{p}' for p in range(4)]].values; Cg=g[[f'player_action_{p}' for p in range(4)]].values; Rg=g['mechanism_observation.pool'].values
        for t in range(1,len(g)):
            R=Rg[t]
            if R<1: continue
            cp=Cg[t-1]; op=Og[t-1]; sc=cp.sum(); so=op.sum()
            cpp=Cg[t-2] if t>=2 else np.zeros(4)
            alloc=Og[t].sum()/R
            base=dict(gid=gid,src=g.src.iloc[0],t=t,R=R/200,Rprev=Rg[t-1]/200,gret=sc/so if so>0 else 0,alloc=alloc)
            for i in range(4):
                recs.append(dict(**base,player=i,cshare=cp[i]/sc if sc>0 else .25,yprev=cp[i]/op[i] if op[i]>=1 else 0,
                                 oshare=op[i]/so if so>0 else .25,cshare2=cpp[i]/cpp.sum() if cpp.sum()>0 else .25,
                                 rank=(cp[i]>cp).sum()+0.5*((cp[i]==cp).sum()-1),
                                 share=Og[t,i]/Og[t].sum() if Og[t].sum()>0 else .25))
    return pd.DataFrame(recs)


def feat(T):
    G=np.asarray(T).reshape(3,3,3,3)
    return dict(mean=G.mean(), dR=(G[0]-G[2]).mean(), ds=(G[:,2]-G[:,0]).mean(), dq=(G[:,:,2]-G[:,:,0]).mean(), dy=(G[:,:,:,2]-G[:,:,:,0]).mean())


def human_pop(G,rng,noise=True):
    T=np.broadcast_to(T_h,(G,4,81)).copy()
    fv=np.clip(rng.choice(first_h.values,(G,4)),0,1)
    u=rng.choice(uemp,(G,4)) if noise else np.zeros((G,4))
    return T,fv,u


def summarize(o):
    return dict(surplus=o['surplus'].mean(),gini=o['gini'].mean(),sustained=o['sustained'].mean(),active=o['active'].mean())


def req(model,prompt,thinking=False,max_tokens=1500):
    r={"prompt":prompt,"system":elicit.SYSTEM,"model":model,"max_tokens":max_tokens}
    if thinking: r["thinking"]={"type":"enabled","budget_tokens":2048}; r["max_tokens"]=max_tokens+2048
    else: r["thinking"]={"type":"disabled"}; r["temperature"]=1.0
    return r


def call_batch(reqs, conc=8):
    t0=time.time()
    outs=host.llm(reqs, max_concurrency=conc)
    for r,o in zip(reqs,outs):
        u=o.get('usage',{}) if isinstance(o,dict) else {}
        tok=sum(u.get(k,0) or 0 for k in ['input_tokens','output_tokens','cache_creation_input_tokens','cache_read_input_tokens'])
        LEDGER.append(dict(model=r['model'],tok=tok,err=o.get('error') if isinstance(o,dict) else None))
    print('batch',len(reqs),'sec',round(time.time()-t0,1),'tokens',sum(l['tok'] for l in LEDGER[-len(reqs):]),'cum',sum(l['tok'] for l in LEDGER))
    return outs


def run_cond(cond, models, n, thinking=False, mech=None, seed=0, persona_fn=None):
    reqs=[];metas=[]
    rng=np.random.default_rng(seed)
    for m in models:
        for k in range(n):
            pers=elicit.persona(rng) if cond=='persona' else None
            prompt,_=elicit.build_prompt(cond,persona=pers,mech=mech)
            reqs.append(req(m,prompt,thinking=thinking)); metas.append(dict(cond=cond+('_think' if thinking else ''),mech=mech,model=m,rep=k,persona=pers))
    outs=call_batch(reqs,conc=16)
    ok=0
    for meta,o in zip(metas,outs):
        store(meta,o); ok+= elicit.parse(o.get('text') or '',ids) is not None
    print(cond, mech, 'parsed',ok,'/',len(outs))
    return metas,outs


def load_raw():
    rows=[json.loads(l) for l in open(RAW,encoding='utf-8')]
    out=[]
    for r in rows:
        pr=elicit.parse(r.get('text') or '',ids)
        r['ok']=pr is not None
        if pr: r['F'],r['T']=pr[0],pr[1]
        out.append(r)
    return out


def probe_prompt(row):
    g=Hs[Hs.launch_id==row.game].sort_values('round_id').reset_index(drop=True)
    t=int(row.t); i=int(row.player); order=[i]+[j for j in range(4) if j!=i]; names=['You','Player 2','Player 3','Player 4']
    prev=g.iloc[t-1]; cur=g.iloc[t]
    def fmt(v): return '%.0f'%v
    lines=[f"GAME IN PROGRESS\nYou are now playing round {t+1} of the game (you do not know how many rounds are left).",
           f"Overview of the previous round: the field held {fmt(prev['mechanism_observation.pool'])} flowers.",
           'Offers: '+'; '.join(f"{names[k]} {fmt(prev[f'offer_{j}'])}" for k,j in enumerate(order)),
           'Re-invested: '+'; '.join(f"{names[k]} {fmt(prev[f'player_action_{j}'])}" for k,j in enumerate(order)),
           f"After re-investment and growth the field now holds {fmt(cur['mechanism_observation.pool'])} flowers.",
           'This round the manager offers: '+'; '.join(f"{names[k]} {fmt(cur[f'offer_{j}'])}" for k,j in enumerate(order)),
           f"How many of your {fmt(cur[f'offer_{i}'])} flowers do you re-invest in the field (a whole number from 0 to {int(np.floor(cur[f'offer_{i}']))})? Reply with one JSON object of the form {{\"reinvest\": <integer>}}."]
    return elicit.INSTR['flowers']+'\n\n'+'\n'.join(lines)


def main(csv='sustainable_behavior.csv'):
    cols = None
    D = pd.read_csv(csv, low_memory=False)
    H = D[D.mech_name_by_player.str.contains('Exp [123]$', regex=True)].copy()
    H['exp'] = H.mech_name_by_player.str.extract(r'Exp (\d)$')[0].astype(int)
    HD = decisions(H, ['mech_name', 'exp']); HD['pid'] = HD.game.astype(str) + '_' + HD.player.astype(str)
    fitd = HD[(HD.t > 0) & HD.q.notna() & HD.yprev.notna()].copy()
    global PEN
    PEN = smooth_pen()
    T_h, A = fit_table(fitd)
    res = fitd.y.values - A @ T_h; fitd['res'] = res
    pm = fitd.groupby('pid').res.agg(['mean', 'count'])
    within = fitd.groupby('pid').res.transform(lambda x: x - x.mean()); s2w = within.var()
    s2u = max(pm['mean'].var() - s2w / pm['count'].mean(), 1e-6)
    pm['u'] = pm['mean'] * pm['count'] / (pm['count'] + s2w / s2u)
    first = HD[HD.t == 0].y.values
    os.makedirs('siminput', exist_ok=True)
    np.savez_compressed('siminput/human.npz', T_h=T_h, first=first, u=pm['u'].values, res=within.values.astype(np.float32))
    np.save('siminput/pen.npy', PEN)
    fitd[['game', 'R', 's', 'q', 'yprev', 'y']].to_csv('siminput/human_decisions.csv.gz', index=False, compression='gzip')
    M2 = D[D.mech_name == 'RL Agent (M2)'].copy()
    M2['src'] = np.where(M2.mech_name_by_player.str.contains('Exp'), 'human', 'bc')
    M2['gid'] = M2.src + '_' + M2.launch_id.astype(str) + '_' + M2.episode_id.astype(str)
    MR = m2_rows(M2); MR['tt'] = MR.t / 40
    MR.to_csv('siminput/m2_train.csv.gz', index=False, compression='gzip')
    game_outcomes(H).to_csv('human_outcomes.csv', index=False)

if __name__ == '__main__':
    main()
