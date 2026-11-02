"""Independent, reproducible checks for the slide-6 audit; no source changes."""
from pathlib import Path
import hashlib, json, sys
import numpy as np
import pandas as pd
from scipy import stats
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'src'))
from si_shap.linear_shapley import LinearShapleyStatistic
from si_shap.selection import LinearShapleySelector
from si_shap.inference import _run_conditional_mc
out={}
rng=np.random.default_rng(20261008)
# Independent coalition projections, including intercept, using least squares.
X=rng.normal(size=(30,4)); y=rng.normal(size=30)
stat=LinearShapleyStatistic(X,mode='exact')
A=np.zeros_like(stat.matrices)
from math import factorial
for j in range(4):
    for mask in range(1<<4):
        if mask & (1<<j): continue
        S=[h for h in range(4) if mask & (1<<h)]
        D=np.column_stack([np.ones(30),X[:,S]])
        E=np.column_stack([D,X[:,j]])
        w=factorial(len(S))*factorial(3-len(S))/factorial(4)
        A[j]+=w*(E@np.linalg.pinv(E)-D@np.linalg.pinv(D))
err=float(np.max(np.abs(stat.matrices-A)))
assert err<1e-12
out['independent_coalition_matrix_max_error']=err
# General mathematical identities and full-range ray.
X20=np.random.default_rng(3405814020).normal(size=(100,20))
s20=LinearShapleySelector(); a20=s20.statistic(X20)
H=a20.basis; P=H@H.T
out['p20_ranks']=[a20.support_basis(j).shape[1] for j in range(20)]
out['p20_smallest_reduced_eigenvalue']=float(np.linalg.eigvalsh(a20.reduced_matrices).min())
out['p20_efficiency_error']=float(np.max(np.abs(a20.matrices.sum(axis=0)-P)))
y20=rng.normal(size=100); j=0; Q=a20.support_basis(j); v=Q@(Q.T@y20); u=v/np.linalg.norm(v); z=y20-v
m=set(s20.select(X20,y20,2).selected_features)
out['p20_ray_selected_sets_unchanged']=all(set(s20.select(X20,z+r*u,2).selected_features)==m for r in [0.01,0.1,1.,3.,10.,100.])
# Correlated example and rank discontinuity.
B=rng.normal(size=(30,2)); B-=B.mean(axis=0); B=np.linalg.qr(B)[0]
x1,e=B.T
examples=[]
for rho in [0.,0.7,1e-5,1e-8]:
    xx=np.column_stack([x1,rho*x1+np.sqrt(1-rho*rho)*e])
    ss=LinearShapleyStatistic(xx,mode='exact')
    mu=6*x1
    examples.append({'rho':rho,'rank2':ss.support_basis(1).shape[1],
                     'score2_mean':float(ss.scores(mu)[1]),'expected':18*rho*rho,
                     'eigenvalues2':np.linalg.eigvalsh(ss.reduced_matrices[1]).tolist()})
out['correlated_examples']=examples
# Fixed finite permutations can change the null's range.
out['one_permutation_ranks']=[LinearShapleyStatistic(X,mode='monte_carlo',n_permutations=1).support_basis(j).shape[1] for j in range(4)]
# Orthogonal top-1 selection has a nontrivial truncation; reference integral.
r_obs=2.; threshold=1.
truth=stats.chi.sf(r_obs,1)/stats.chi.sf(threshold,1)
p_mc,diag=_run_conditional_mc(r_obs,1,lambda r:r>=threshold,np.random.default_rng(7),n_proposals=100000)
assert abs(p_mc-truth)<0.01
out['orthogonal_top1']={'exact_p':float(truth),'conditional_mc_p':p_mc,'retained':diag['selected_samples']}
# Constant decimal column should have zero credit with an intercept.
Xconstant=np.full((100,1),0.1); yconstant=np.ones(100)
sc=LinearShapleyStatistic(Xconstant)
out['constant_decimal_bug']={'column_value':0.1,'mean_computed':float(Xconstant.mean()),'score':float(sc.scores(yconstant)[0]),'rank':sc.support_basis(0).shape[1],'expected_score':0.0}
# Verify paired saved data and hashes without modifying original results.
base=ROOT/'outputs/selection_method_validation/exact_set'
ais_dir=base/'linear_shapley_selected_five_matched_300_20260928'
exact_dir=base/'linear_shapley_selected_five_matched_300_exact_20261002'
combo=base/'linear_shapley_selected_five_matched_300_ais_exact_20261002'
man=json.loads((combo/'source_manifest.json').read_text())
out['saved_data_hashes_match']={key:hashlib.sha256((ROOT/value['path']/'unified_tests.csv').read_bytes()).hexdigest()==value['data_sha256'] for key,value in man['sources'].items()}
df=pd.read_csv(exact_dir/'unified_tests.csv')
err=np.abs(df.p_value-stats.chi2.sf(df.radial_t_obs**2,df=20))
out['saved_exact_checks']={'n':len(df),'all_target_ranks20':bool(df.test_rank.eq(20).all()),'p_formula_error':float(err.max()),'changed_5pct':int(df.decision_changed.sum()),'max_ais_absolute_error':float(np.abs(df.p_value-df.source_ais_p_value).max())}
old=pd.read_csv(ais_dir/'unified_tests.csv')
out['saved_all_proposals_selected']=bool(old.selected_samples.eq(old.proposals).all())
null_settings=[json.loads((ais_dir/f'runs/linear_shapley/shard_{i}/linear_shapley/baseline/null_fresh/settings.json').read_text()) for i in range(32)]
out['saved_null_design_seeds']=sorted(set(s['design_seed'] for s in null_settings))
out['saved_null_fixed_within_shards']=all(s['fixed_design'] for s in null_settings)
source_hashes=json.loads((ais_dir/'source_hashes.json').read_text())
out['current_vs_saved_source_hashes']={p:hashlib.sha256((ROOT/p).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==source_hashes.get(p) for p in ['src/si_shap/linear_shapley.py','src/si_shap/selection.py','src/si_shap/api.py','src/si_shap/inference.py','src/si_shap/power.py','src/si_shap/null_calibration.py']}
Path(__file__).with_name('verification.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
