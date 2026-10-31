"""Load the released source-data tables and execute the original plotting functions.
Run from any directory: python 07_code/reproduce_figures.py [--output-dir PATH]
This redraws saved results; it does not rerun simulation experiments or API calls.
"""
from pathlib import Path
import os, sys, argparse, importlib.util, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output-dir',type=Path,default=ROOT/'output/figures')
args=parser.parse_args()
args.output_dir.mkdir(parents=True,exist_ok=True)
os.environ.setdefault('OMP_NUM_THREADS','1')
sys.path.insert(0,str(ROOT/'07_code'))
import run_rev

book=ROOT/'05_source_data/Source_Data.xlsx'
read=lambda name:pd.read_excel(book,sheet_name=name)
MODELS=run_rev.MODELS
names=['Haiku 4.5','Sonnet 4.5','Sonnet 4.6','Opus 4.6','Sonnet 5','Opus 5']
short=dict(zip(MODELS,names))
fe=read('Fig2a')
human=run_rev.human_model(run_rev.HDF,run_rev.HFIRST)
g=human['T'].reshape(3,3,3,3)
hf=dict(mean=g.mean(),dR=(g[0]-g[2]).mean(),ds=(g[:,2]-g[:,0]).mean(),dq=(g[:,:,2]-g[:,:,0]).mean(),dy=(g[:,:,:,2]-g[:,:,:,0]).mean())
mech=read('Fig5a');mech['var']=mech['var'].fillna('')
T5=mech.set_index(['model','var'])
obs=read('Fig1b').set_index('institution')
vp=read('ED2a').set_index('fit')
ed4=read('ED4')
diag=read('ED1')
diag=diag[diag.variant=='none'].set_index(['pop','inst'])
sur=read('ED2c')
state=dict(MODELS=MODELS,SHORT=short,MORDER=names,INST=run_rev.INST,
 PT=read('Fig3_eff').set_index('pop'), LV=read('Fig3_lev').set_index('index'),
 SUST=read('Fig3_sustained').set_index('pop'),OBS=obs,CV=obs,
 BOOT2=pd.read_csv(ROOT/'08_derived_data/results_rev/boot.csv'),
 FE2=fe,hF=hf,SD_U=read('Fig2d').set_index('population').loc['human offsets','sd_mean_reinvestment'],
 MPT=read('Fig2bc'),vp=vp,T5=T5,
 L5=mech.pivot(index='var',columns='model',values='lev').loc[:,MODELS],
 SEL=read('Fig6ab'),PG=read('Fig6c'),dg=diag,
 SENS=pd.read_csv(ROOT/'08_derived_data/derived/boundary_sensitivity_leverage.csv').set_index('pop'),
 TUNE2=read('ED2b'),SURE={m:sur[sur.level_0==m].set_index('str') for m in ['m1','m2']},
 lr=read('ED3a').set_index('pop'),
 VDV={r['index']:[r['population'],r['institution'],r['residual']] for r in read('ED3b').to_dict('records')},
 SELn=ed4.rename(columns={'gain_eval':'original_gain_eval','gain_eval_noise':'gain_eval'}),
 SELpw=ed4)
def panel_letter(ax,letter,dx=-0.15):
    ax.text(dx,1.02,letter,transform=ax.transAxes,fontsize=9,fontweight='bold',va='bottom')
state['panel_letter']=panel_letter
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7,'axes.titlesize':7,'axes.labelsize':7,'xtick.labelsize':6,'ytick.labelsize':6,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
# The original module creates a relative output directory during import.
previous=Path.cwd()
os.chdir(args.output_dir)
try:
    spec=importlib.util.spec_from_file_location('released_figures',ROOT/'07_code/fig_rev.py')
    module=importlib.util.module_from_spec(spec)
    module.__dict__.update(state)
    spec.loader.exec_module(module)
    def save(fig,name):
        for ext in ['pdf','png']:
            fig.savefig(args.output_dir/f'{name}.{ext}',dpi=220,bbox_inches='tight')
    module.save=save
    results=[]
    for name in ['fig1','fig2','fig3','fig4','fig5','fig6','ed1','ed2','ed3','ed4']:
        print('Rendering',name,flush=True)
        result=getattr(module,name)()
        fig=result[0] if isinstance(result,tuple) else result
        fig.canvas.draw()
        results.append({'figure':name,'axes':len(fig.axes),'pdf':name+'.pdf','png':name+'.png'})
        plt.close(fig)
    (args.output_dir/'render_manifest.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
finally:
    os.chdir(previous)
print('Rendered all 10 figures from saved source data; no API calls or full simulations.')
