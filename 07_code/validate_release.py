from pathlib import Path
import json,sys,hashlib,re,platform,importlib.metadata
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'07_code'))
import run_rev as run

def main():
    out=ROOT/'output/validation';out.mkdir(parents=True,exist_ok=True)
    read=lambda s:pd.read_excel(ROOT/'05_source_data/Source_Data.xlsx',sheet_name=s)
    main=pd.DataFrame(json.loads((ROOT/'08_derived_data/results_rev/main.json').read_text()))
    main=main[main.variant=='none']
    point=main.pivot(index='pop',columns='inst',values='eff')
    source=read('Fig3_eff').set_index('pop')
    diff=np.max(np.abs(point.loc[source.index,source.columns].values-source.values))
    assert diff<1e-12
    lev=read('Fig3_lev').set_index('index')
    err=np.max(np.abs((point.interp-point.equal).loc[lev.index].values-lev.lev.values))
    assert err<1e-12
    boot=pd.read_csv(ROOT/'08_derived_data/results_rev/boot.csv')
    boot['leverage']=boot.interp_eff-boot.equal_eff
    ci=boot.groupby('pop').leverage.quantile([.025,.975]).unstack()
    common=lev.index.intersection(ci.index)
    no_boot=lev.index.difference(ci.index)
    assert lev.loc[no_boot,['lev_lo','lev_hi']].isna().all().all()
    cierr=max(np.max(np.abs(ci.loc[common,.025]-lev.loc[common,'lev_lo'])),np.max(np.abs(ci.loc[common,.975]-lev.loc[common,'lev_hi'])))
    assert cierr<1e-12
    assert len(run.AG)==213 and run.HFIRST.game.nunique()==440 and len(run.HDF)==36156
    token={};original_rows=[]
    for name in ['elicitations.jsonl','revision_calls.jsonl']:
        rows=[json.loads(l) for l in (ROOT/'06_llm_raw'/name).read_text(encoding='utf-8').splitlines() if l.strip()]
        counts=[sum((r.get('usage') or {}).get(k,0) or 0 for k in ['input_tokens','output_tokens','cache_creation_input_tokens','cache_read_input_tokens']) for r in rows]
        fields={k:sum((r.get('usage') or {}).get(k,0) or 0 for r in rows) for k in ['input_tokens','output_tokens','cache_creation_input_tokens','cache_read_input_tokens']}
        token[name]={'calls':len(rows),'api_usage_total':sum(counts),'fields':fields,'noncache_input_plus_output':fields['input_tokens']+fields['output_tokens']}
        if name=='elicitations.jsonl':
            original_rows=rows
            token[name]['first_224_noncache_input_plus_output']=sum(sum((r.get('usage') or {}).get(k,0) or 0 for k in ['input_tokens','output_tokens']) for r in rows[:224])
    ledger=pd.read_csv(ROOT/'06_llm_raw/token_ledger.csv')
    full_usage=lambda r:sum((r.get('usage') or {}).get(k,0) or 0 for k in ['input_tokens','output_tokens','cache_creation_input_tokens','cache_read_input_tokens'])
    mismatches=sum(int(v)!=full_usage(r) for v,r in zip(ledger.tok,original_rows))
    token['legacy_ledger']={'calls':len(ledger),'recorded_tokens':int(ledger.tok.sum()),'row_token_mismatches':mismatches,'reconciled':len(ledger)==len(original_rows) and mismatches==0}
    # Six selected published cells, not the complete simulation experiment.
    run.HUMAN_FULL=run.human_model(run.HDF,run.HFIRST)
    replay=[]
    pops=['human','base|'+run.MODELS[0],'base|'+run.MODELS[1]]
    for pop in pops:
        for inst in ['equal','interp']:
            print('Replaying',pop,inst,'6000 games',flush=True)
            result=run.t_main((pop,inst,'none'))
            expected=float(point.loc[pop,inst])
            replay.append({'population':pop,'institution':inst,'games':6000,'published_efficiency':expected,'replayed_efficiency':result['eff'],'absolute_difference':abs(expected-result['eff'])})
    assert max(x['absolute_difference'] for x in replay)<1e-10
    # Exact homogeneity proposition for the five analytic full-allocation rules.
    T=np.broadcast_to(np.linspace(.1,.9,81),(24,4,81)).copy();F=np.full((24,4),.7)
    homo={i:run.simulate2(i,T,F,rng=np.random.default_rng(4),params={'theta':194})['eff'] for i in ['equal','mixed','prop','interp','step']}
    hdiff=max(float(np.max(np.abs(v-homo['equal']))) for v in homo.values())
    assert hdiff==0
    secret_patterns=[r'sk-ant-[A-Za-z0-9_-]{20,}',r'gh[pousr]_[A-Za-z0-9]{30,}',r'-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----']
    suspected=[]
    for f in ROOT.rglob('*'):
        if not f.is_file() or 'output' in f.relative_to(ROOT).parts or f.suffix not in ['.py','.json','.jsonl','.md','.txt','.csv']:continue
        s=f.read_text(encoding='utf-8',errors='replace')
        if any(re.search(p,s) for p in secret_patterns):suspected.append(f.relative_to(ROOT).as_posix())
    assert not suspected
    report={'source_data_point_max_abs_error':float(diff),'leverage_max_abs_error':float(err),'bootstrap_CI_max_abs_error':float(cierr),'groups_in_first_round_input':440,'groups_in_fitting_decisions':int(run.HDF.game.nunique()),'human_decisions':36156,'elicited_strategies':213,'tokens':token,'selected_cell_replays':replay,'homogeneity_max_abs_difference':hdiff,'secret_pattern_suspect_files':suspected,'python':platform.python_version(),'packages':{n:importlib.metadata.version(n) for n in ['numpy','pandas','scipy','scikit-learn','matplotlib','openpyxl']},'limitations':['No full bootstrap, planner training, full exploitability search or new API elicitation was rerun.','Source-data redraws reuse the released estimates; they do not independently reproduce every analysis step.','SI token accounting and original-raw-log count were corrected on 2026-10-31; the original ledger reconciles row by row when cache tokens are included.']}
    (out/'validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
