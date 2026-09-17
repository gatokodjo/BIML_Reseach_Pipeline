from pathlib import Path
import json, warnings, math, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings('ignore')

SEED = 42
N_REPEATS = 10
TEST_SIZE = 0.20
BOOTSTRAPS = 2000
TARGET='Incidence'; GROUP='Country'; YEAR='Year'

ROOT = Path(__file__).resolve().parent
DATA_PATH = Path(os.environ.get("BIML_DATA_PATH", ROOT/"data"/"SSA_Malaria_Environmental_Incidence_2015_2020.csv"))
OUT = Path(os.environ.get("BIML_OUTPUT_DIR", ROOT/"outputs"))
FIG = OUT/'figures'; TAB=OUT/'tables'; DAT=OUT/'data'; META=OUT/'metadata'
for p in [OUT,FIG,TAB,DAT,META]: p.mkdir(parents=True, exist_ok=True)

ENV_BASE = ['Temp_mean','Temp_std','Temp_min','Temp_max','NDVI_mean','NDVI_std',
            'NDWI_mean','NDWI_std','Rainfall_total','Rainfall_mean','Rainfall_std',
            'Rainfall_max','Wet_months','Population_density']
NONLIN = ['Rainfall_log','Population_log','Temp_sq','Rainfall_sq','Temp_Rain','NDVI_NDWI']
ENV_FULL = ENV_BASE + NONLIN
COMPACT_CANDIDATES = ['Temp_mean','Temp_std','NDVI_mean','NDVI_std','NDWI_mean',
                      'Rainfall_total','Rainfall_std','Wet_months','Population_log']
VC_FEATURES = ['VC_norm','VC_lag','VC_change']

REPRESENTATIONS = {
    'M1_history':'history',
    'M2_history_env_compact':'compact',
    'M3_history_vc':'vc',
    'M4_history_env_compact_vc':'compact_vc',
    'M5_history_env_full':'full',
    'M6_history_env_full_vc':'full_vc',
}

# ------------------------------------------------------------------
# 1. Annual aggregation and target-history features
# ------------------------------------------------------------------
def aggregate_annual():
    d = pd.read_csv(DATA_PATH).copy()
    # Incidence is repeated monthly within a country-year; missing target rows are excluded.
    d = d.dropna(subset=[TARGET]).copy()
    a = d.groupby([GROUP,YEAR],as_index=False).agg(
        Incidence=(TARGET,'first'),
        Temp_mean=('LST_C','mean'), Temp_std=('LST_C','std'),
        Temp_min=('LST_C','min'), Temp_max=('LST_C','max'),
        NDVI_mean=('NDVI','mean'), NDVI_std=('NDVI','std'),
        NDWI_mean=('NDWI','mean'), NDWI_std=('NDWI','std'),
        Rainfall_total=('Rainfall_mm','sum'), Rainfall_mean=('Rainfall_mm','mean'),
        Rainfall_std=('Rainfall_mm','std'), Rainfall_max=('Rainfall_mm','max'),
        Wet_months=('Rainfall_mm', lambda z: int((z>50).sum())),
        Population_density=('Population_density','mean')
    ).sort_values([GROUP,YEAR]).reset_index(drop=True)
    # Engineered environmental features (deterministic; no learned distribution parameters).
    a['Rainfall_log'] = np.log1p(a.Rainfall_total.clip(lower=0))
    a['Population_log'] = np.log1p(a.Population_density.clip(lower=0))
    a['Temp_sq'] = a.Temp_mean**2
    a['Rainfall_sq'] = a.Rainfall_total**2
    a['Temp_Rain'] = a.Temp_mean*a.Rainfall_total
    a['NDVI_NDWI'] = a.NDVI_mean*a.NDWI_mean
    g=a.groupby(GROUP,sort=False)
    a['lag1_incidence']=g[TARGET].shift(1)
    a['lag2_incidence']=g[TARGET].shift(2)
    a.to_csv(DAT/'annual_base.csv',index=False)
    return a

# ------------------------------------------------------------------
# 2. VC parameterization
# ------------------------------------------------------------------
# Fixed biological/operational thermal shape; distributional scaling is fitted ONLY on training data.
# s_T = exp(-(T-25)^2/(2*5^2))
# a(T)=0.05+0.03 s_T; p(T)=0.70+0.25 s_T; n(T)=20-8 s_T
# rainfall, vegetation, population terms are scaled from training data only.
def fit_vc_params(context_train):
    z=context_train.copy()
    rain_scale=max(float(z.Rainfall_total.median()),1.0)
    nd_min=float(z.NDVI_mean.min()); nd_max=float(z.NDVI_mean.max())
    lp=np.log1p(z.Population_density.clip(lower=0))
    pop_min=float(lp.min()); pop_max=float(lp.max())
    # Raw VC normalization limits must also come from training data only.
    tmp={'rain_scale':rain_scale,'nd_min':nd_min,'nd_max':nd_max,
         'pop_min':pop_min,'pop_max':pop_max,'t_opt':25.0,'t_sigma':5.0}
    raw=_vc_raw(z,tmp)
    tmp['vc_min']=float(np.nanmin(raw)); tmp['vc_max']=float(np.nanmax(raw))
    return tmp

def _vc_raw(x,p):
    s_t=np.exp(-((x.Temp_mean-p['t_opt'])**2)/(2*p['t_sigma']**2))
    s_r=1-np.exp(-x.Rainfall_total.clip(lower=0)/p['rain_scale'])
    s_v=((x.NDVI_mean-p['nd_min'])/(p['nd_max']-p['nd_min']+1e-12)).clip(0,1)
    lp=np.log1p(x.Population_density.clip(lower=0))
    s_p=((lp-p['pop_min'])/(p['pop_max']-p['pop_min']+1e-12)).clip(0,1)
    a=0.05+0.03*s_t
    surv=(0.70+0.25*s_t).clip(0.01,0.99)
    n=(20-8*s_t).clip(lower=5)
    m=0.25+0.75*s_r*s_v*(0.5+0.5*s_p)
    vc=m*a**2*surv**n/(-np.log(surv))
    return np.asarray(vc,float)

def apply_vc(annual, params):
    x=annual.copy().sort_values([GROUP,YEAR]).reset_index(drop=True)
    x['VC_raw']=_vc_raw(x,params)
    x['VC_norm']=((x.VC_raw-params['vc_min'])/(params['vc_max']-params['vc_min']+1e-12)).clip(0,1)
    g=x.groupby(GROUP,sort=False)
    x['VC_lag']=g['VC_norm'].shift(1)
    x['VC_change']=x['VC_norm']-x['VC_lag']
    return x

# ------------------------------------------------------------------
# 3. One-lag / two-lag eligible target datasets
# ------------------------------------------------------------------
def eligible_keys(annual, design):
    if design=='one_lag':
        e=annual.dropna(subset=['lag1_incidence']).copy()
    elif design=='two_lag':
        e=annual.dropna(subset=['lag1_incidence','lag2_incidence']).copy()
    else: raise ValueError(design)
    return e[[GROUP,YEAR,TARGET,'lag1_incidence','lag2_incidence']].copy()

def history_features(design):
    return ['lag1_incidence'] if design=='one_lag' else ['lag1_incidence','lag2_incidence']

# ------------------------------------------------------------------
# 4/5. Leakage-safe feature-set construction and correlation pruning
# ------------------------------------------------------------------
def corr_prune(train, candidates, threshold=0.90):
    kept=[]; dropped=[]
    for f in candidates:
        if not kept:
            kept.append(f); continue
        c=train[kept+[f]].corr(numeric_only=True)[f].drop(f).abs()
        if len(c) and np.nanmax(c.values)>=threshold:
            dropped.append((f, str(c.idxmax()), float(c.max())))
        else:
            kept.append(f)
    return kept,dropped

def feature_names(train_transformed, design, rep):
    hist=history_features(design)
    if rep=='history': return hist,[]
    if rep=='vc': return hist+VC_FEATURES,[]
    if rep in ('compact','compact_vc'):
        kept,dropped=corr_prune(train_transformed,COMPACT_CANDIDATES,0.90)
        f=hist+kept+(VC_FEATURES if rep=='compact_vc' else [])
        return f,dropped
    if rep=='full': return hist+ENV_FULL,[]
    if rep=='full_vc': return hist+ENV_FULL+VC_FEATURES,[]
    raise ValueError(rep)

# ------------------------------------------------------------------
# 6. Models and nested grouped alpha selection
# ------------------------------------------------------------------
def make_model(kind, seed=SEED):
    if kind=='rf':
        return RandomForestRegressor(n_estimators=500,max_depth=5,min_samples_leaf=3,
                                     max_features=0.8,random_state=seed,n_jobs=-1)
    if kind=='gb':
        return GradientBoostingRegressor(n_estimators=180,learning_rate=0.03,max_depth=2,
                                         min_samples_leaf=4,loss='huber',random_state=seed)
    raise ValueError(kind)

def metric_dict(y,p):
    return {'mae':float(mean_absolute_error(y,p)),
            'rmse':float(np.sqrt(mean_squared_error(y,p))),
            'r2':float(r2_score(y,p))}

def rows_by_keys(transformed, keys_df):
    cols=[GROUP,YEAR]
    return keys_df[cols].merge(transformed,on=cols,how='left',validate='one_to_one')

def get_context_train(annual, train_countries, temporal_cutoff=None):
    q=annual[annual[GROUP].isin(set(train_countries))].copy()
    if temporal_cutoff is not None:
        q=q[q[YEAR] < temporal_cutoff]
    return q

def nested_oof_raw(annual, train_keys, design, rep, kind, temporal_cutoff=None):
    # OOF predictions generated with VC scaling and compact-feature selection refit inside each inner fold.
    train_keys=train_keys.reset_index(drop=True)
    groups=train_keys[GROUP].values
    n=min(5,pd.Series(groups).nunique())
    splitter=GroupKFold(n_splits=n)
    out=np.full(len(train_keys),np.nan)
    records=[]
    for fold,(ti,vi) in enumerate(splitter.split(train_keys,groups=groups),1):
        ik=train_keys.iloc[ti].copy(); iv=train_keys.iloc[vi].copy()
        inner_train_countries=ik[GROUP].unique()
        context=get_context_train(annual,inner_train_countries,temporal_cutoff)
        params=fit_vc_params(context)
        tr=apply_vc(annual,params)
        ikt=rows_by_keys(tr,ik); ivt=rows_by_keys(tr,iv)
        feats,drops=feature_names(ikt,design,rep)
        m=make_model(kind,SEED+fold)
        m.fit(ikt[feats],ikt[TARGET])
        out[vi]=m.predict(ivt[feats])
        records.append({'fold':fold,'features':'|'.join(feats),'n_features':len(feats),
                        'dropped':'|'.join([d[0] for d in drops])})
    assert np.isfinite(out).all()
    return out,records

def choose_alpha(annual,train_keys,design,rep,kind,temporal_cutoff=None):
    raw,inner_records=nested_oof_raw(annual,train_keys,design,rep,kind,temporal_cutoff)
    y=train_keys[TARGET].to_numpy(float); base=train_keys['lag1_incidence'].to_numpy(float)
    candidates=np.linspace(0,1,21)
    rmse=[]
    for a in candidates:
        pred=np.maximum(0,base+a*(raw-base))
        rmse.append(np.sqrt(np.mean((y-pred)**2)))
    j=int(np.argmin(rmse))
    return float(candidates[j]),float(rmse[j]),inner_records

# ------------------------------------------------------------------
# Single outer evaluation
# ------------------------------------------------------------------
def evaluate_outer(annual,train_keys,test_keys,design,split,repeat=None,temporal_cutoff=None):
    out=[]; preds=[]; selection=[]
    y=test_keys[TARGET].to_numpy(float); base=test_keys['lag1_incidence'].to_numpy(float)
    met=metric_dict(y,base)
    out.append({'design':design,'split':split,'repeat':repeat,'model':'M0_persistence','learner':'baseline',
                'alpha':np.nan,'n_features':1,**met})
    for i,row in test_keys.reset_index(drop=True).iterrows():
        preds.append({'design':design,'split':split,'repeat':repeat,'model':'M0_persistence','learner':'baseline',
                      GROUP:row[GROUP],YEAR:int(row[YEAR]),'observed':row[TARGET],'baseline':row['lag1_incidence'],
                      'raw_prediction':row['lag1_incidence'],'prediction':row['lag1_incidence']})

    train_countries=train_keys[GROUP].unique()
    context=get_context_train(annual,train_countries,temporal_cutoff)
    params=fit_vc_params(context)
    transformed=apply_vc(annual,params)
    tr=rows_by_keys(transformed,train_keys); te=rows_by_keys(transformed,test_keys)

    for model_id,rep in REPRESENTATIONS.items():
        feats,drops=feature_names(tr,design,rep)
        for kind in ['rf','gb']:
            alpha,inner_rmse,inner_records=choose_alpha(annual,train_keys,design,rep,kind,temporal_cutoff)
            m=make_model(kind,SEED+(repeat or 0))
            m.fit(tr[feats],tr[TARGET])
            raw=m.predict(te[feats])
            pred=np.maximum(0,base+alpha*(raw-base))
            met=metric_dict(y,pred); rawmet=metric_dict(y,np.maximum(0,raw))
            name=f'{model_id}_{kind}'
            out.append({'design':design,'split':split,'repeat':repeat,'model':name,'learner':kind,
                        'representation':rep,'alpha':alpha,'inner_oof_rmse':inner_rmse,
                        'n_features':len(feats),**met,
                        'raw_mae':rawmet['mae'],'raw_rmse':rawmet['rmse'],'raw_r2':rawmet['r2']})
            selection.append({'design':design,'split':split,'repeat':repeat,'model':name,
                              'features':'|'.join(feats),'n_features':len(feats),
                              'dropped_corr':'|'.join([f'{a}>{b}:{c:.3f}' for a,b,c in drops]),
                              'vc_params':json.dumps(params,sort_keys=True)})
            for i,row in te.reset_index(drop=True).iterrows():
                preds.append({'design':design,'split':split,'repeat':repeat,'model':name,'learner':kind,
                              GROUP:row[GROUP],YEAR:int(row[YEAR]),'observed':row[TARGET],
                              'baseline':row['lag1_incidence'],'raw_prediction':float(raw[i]),
                              'prediction':float(pred[i]),'alpha':alpha})
    return out,preds,selection

# ------------------------------------------------------------------
# 7. Grouped validation / 8. temporal evaluation
# ------------------------------------------------------------------
def run_all(annual):
    all_metrics=[]; all_preds=[]; all_sel=[]
    for design in ['one_lag','two_lag']:
        keys=eligible_keys(annual,design).reset_index(drop=True)
        # grouped repeated country holdout
        for r in range(N_REPEATS):
            splitter=GroupShuffleSplit(n_splits=1,test_size=TEST_SIZE,random_state=SEED+r)
            ti,vi=next(splitter.split(keys,groups=keys[GROUP]))
            tr=keys.iloc[ti].copy(); te=keys.iloc[vi].copy()
            m,p,s=evaluate_outer(annual,tr,te,design,'grouped',r+1,None)
            all_metrics.extend(m); all_preds.extend(p); all_sel.extend(s)
        # temporal 2020; training target rows strictly pre-2020
        tr=keys[keys[YEAR]<2020].copy(); te=keys[keys[YEAR]==2020].copy()
        m,p,s=evaluate_outer(annual,tr,te,design,'temporal_2020',1,2020)
        all_metrics.extend(m); all_preds.extend(p); all_sel.extend(s)
    metrics=pd.DataFrame(all_metrics); preds=pd.DataFrame(all_preds); sel=pd.DataFrame(all_sel)
    metrics.to_csv(DAT/'all_outer_metrics.csv',index=False)
    preds.to_csv(DAT/'all_predictions.csv',index=False)
    sel.to_csv(DAT/'feature_selection_and_vc_params.csv',index=False)
    return metrics,preds,sel

# ------------------------------------------------------------------
# 9. Bootstrap paired uncertainty
# ------------------------------------------------------------------
def bootstrap_temporal(preds, n_boot=BOOTSTRAPS):
    rng=np.random.default_rng(SEED)
    rows=[]
    for design in ['one_lag','two_lag']:
        d=preds[(preds.design==design)&(preds.split=='temporal_2020')].copy()
        base=d[d.model=='M0_persistence'][[GROUP,'observed','prediction']].rename(columns={'prediction':'base_pred'})
        for model in sorted(d.model.unique()):
            if model=='M0_persistence': continue
            md=d[d.model==model][[GROUP,'prediction']].merge(base,on=GROUP,how='inner')
            n=len(md); vals=[]
            y=md.observed.to_numpy(); p=md.prediction.to_numpy(); b=md.base_pred.to_numpy()
            for _ in range(n_boot):
                idx=rng.integers(0,n,n)
                yy=y[idx]; pp=p[idx]; bb=b[idx]
                # positive = model improves over persistence
                vals.append((mean_absolute_error(yy,bb)-mean_absolute_error(yy,pp),
                             np.sqrt(mean_squared_error(yy,bb))-np.sqrt(mean_squared_error(yy,pp))))
            arr=np.asarray(vals)
            rows.append({'design':design,'model':model,'comparison':'vs_persistence',
                         'delta_mae':float(mean_absolute_error(y,b)-mean_absolute_error(y,p)),
                         'delta_mae_ci_low':float(np.quantile(arr[:,0],.025)),
                         'delta_mae_ci_high':float(np.quantile(arr[:,0],.975)),
                         'delta_rmse':float(np.sqrt(mean_squared_error(y,b))-np.sqrt(mean_squared_error(y,p))),
                         'delta_rmse_ci_low':float(np.quantile(arr[:,1],.025)),
                         'delta_rmse_ci_high':float(np.quantile(arr[:,1],.975)),
                         'p_improve_rmse':float(np.mean(arr[:,1]>0)),
                         'p_improve_mae':float(np.mean(arr[:,0]>0))})
        # Targeted incremental VC comparisons for the same learner.
        # M3 vs M1 is the primary history-only VC ablation:
        # it directly tests whether VC-derived information adds predictive
        # value beyond historical incidence alone.
        # M4 vs M2 and M6 vs M5 test the incremental contribution of VC
        # when compact or full environmental representations are present.
        for learner in ['rf','gb']:
            for with_vc,without_vc,label in [
                (f'M3_history_vc_{learner}',f'M1_history_{learner}','history +VC vs history'),
                (f'M4_history_env_compact_vc_{learner}',f'M2_history_env_compact_{learner}','compact +VC vs compact'),
                (f'M6_history_env_full_vc_{learner}',f'M5_history_env_full_{learner}','full +VC vs full')]:
                a=d[d.model==with_vc][[GROUP,'observed','prediction']].rename(columns={'prediction':'p1'})
                b2=d[d.model==without_vc][[GROUP,'prediction']].rename(columns={'prediction':'p0'})
                z=a.merge(b2,on=GROUP)
                y=z.observed.to_numpy(); p1=z.p1.to_numpy(); p0=z.p0.to_numpy(); n=len(z); vals=[]
                for _ in range(n_boot):
                    idx=rng.integers(0,n,n); yy=y[idx]; x1=p1[idx]; x0=p0[idx]
                    vals.append((mean_absolute_error(yy,x0)-mean_absolute_error(yy,x1),
                                 np.sqrt(mean_squared_error(yy,x0))-np.sqrt(mean_squared_error(yy,x1))))
                arr=np.asarray(vals)
                rows.append({'design':design,'model':with_vc,'comparison':label,
                             'delta_mae':float(mean_absolute_error(y,p0)-mean_absolute_error(y,p1)),
                             'delta_mae_ci_low':float(np.quantile(arr[:,0],.025)),
                             'delta_mae_ci_high':float(np.quantile(arr[:,0],.975)),
                             'delta_rmse':float(np.sqrt(mean_squared_error(y,p0))-np.sqrt(mean_squared_error(y,p1))),
                             'delta_rmse_ci_low':float(np.quantile(arr[:,1],.025)),
                             'delta_rmse_ci_high':float(np.quantile(arr[:,1],.975)),
                             'p_improve_rmse':float(np.mean(arr[:,1]>0)),
                             'p_improve_mae':float(np.mean(arr[:,0]>0))})
    out=pd.DataFrame(rows)
    out.to_csv(TAB/'bootstrap_paired_temporal.csv',index=False)
    return out

def grouped_uncertainty(metrics):
    g=metrics[metrics.split=='grouped'].copy()
    # descriptive across repeated overlapping splits; not treated as independent inferential replicates.
    summ=g.groupby(['design','model'],as_index=False).agg(
        mae_mean=('mae','mean'),mae_sd=('mae','std'),rmse_mean=('rmse','mean'),rmse_sd=('rmse','std'),
        r2_mean=('r2','mean'),r2_sd=('r2','std'),alpha_mean=('alpha','mean'))
    # paired deltas per repeat vs baseline
    base=g[g.model=='M0_persistence'][['design','repeat','mae','rmse','r2']].rename(columns={'mae':'b_mae','rmse':'b_rmse','r2':'b_r2'})
    d=g.merge(base,on=['design','repeat'],how='left')
    d['rmse_improvement']=d.b_rmse-d.rmse; d['mae_improvement']=d.b_mae-d.mae; d['r2_improvement']=d.r2-d.b_r2
    paired=d.groupby(['design','model'],as_index=False).agg(
        rmse_improvement_mean=('rmse_improvement','mean'),rmse_improvement_sd=('rmse_improvement','std'),
        mae_improvement_mean=('mae_improvement','mean'),r2_improvement_mean=('r2_improvement','mean'))
    summ=summ.merge(paired,on=['design','model'])
    summ.to_csv(TAB/'grouped_summary.csv',index=False)
    return summ

# ------------------------------------------------------------------
# 10. Compare compact/full and one/two lag
# ------------------------------------------------------------------
def temporal_summary(metrics):
    t=metrics[metrics.split=='temporal_2020'].copy()
    t=t.sort_values(['design','rmse','mae']).reset_index(drop=True)
    t.to_csv(TAB/'temporal_2020_summary.csv',index=False)
    return t

def ranking_table(grouped,temporal):
    g=grouped[['design','model','rmse_mean','mae_mean','r2_mean']]
    t=temporal[['design','model','rmse','mae','r2','alpha']]
    x=t.merge(g,on=['design','model'],how='left',suffixes=('_temporal','_grouped'))
    # primary rank temporal RMSE; secondary grouped RMSE
    x['rank_temporal_rmse']=x.groupby('design').rmse.rank(method='min')
    x['rank_grouped_rmse']=x.groupby('design').rmse_mean.rank(method='min')
    x['combined_rank']=x.rank_temporal_rmse+x.rank_grouped_rmse
    x=x.sort_values(['combined_rank','rmse','rmse_mean'])
    x.to_csv(TAB/'model_ranking.csv',index=False)
    return x

# ------------------------------------------------------------------
# 11. Figures and tables
# ------------------------------------------------------------------
def generate_figures(annual,metrics,preds,grouped,temporal,boot,ranking,sel):
    # F1 dataset availability
    cnt=annual.groupby(YEAR)[GROUP].nunique()
    fig,ax=plt.subplots(figsize=(7,4)); ax.bar(cnt.index.astype(str),cnt.values); ax.set_ylabel('Countries with incidence'); ax.set_xlabel('Year'); ax.set_title('Annual country availability'); fig.tight_layout(); fig.savefig(FIG/'Figure_1_dataset_availability.png',dpi=220); plt.close(fig)

    # F2 one vs two lag sample sizes
    vals=[len(eligible_keys(annual,'one_lag')),len(eligible_keys(annual,'two_lag'))]
    fig,ax=plt.subplots(figsize=(6,4)); ax.bar(['One-lag','Two-lag'],vals); ax.set_ylabel('Eligible country-year observations'); ax.set_title('Analytical sample size by lag design'); fig.tight_layout(); fig.savefig(FIG/'Figure_2_lag_sample_sizes.png',dpi=220); plt.close(fig)

    # F3 grouped RMSE top variants
    top=grouped[grouped.model!='M0_persistence'].copy().sort_values('rmse_mean').groupby('design').head(6)
    labels=[f"{r.design.replace('_',' ')}\n{r.model}" for r in top.itertuples()]
    fig,ax=plt.subplots(figsize=(12,6)); ax.barh(labels,top.rmse_mean,xerr=top.rmse_sd); ax.invert_yaxis(); ax.set_xlabel('Mean grouped RMSE ± SD'); ax.set_title('Best grouped-validation configurations'); fig.tight_layout(); fig.savefig(FIG/'Figure_3_grouped_rmse.png',dpi=220,bbox_inches='tight'); plt.close(fig)

    # F4 temporal RMSE all model ladder averages by learner/design
    tt=temporal[temporal.model!='M0_persistence'].copy().sort_values('rmse').head(16)
    labels=[f"{r.design.replace('_',' ')} | {r.model}" for r in tt.itertuples()]
    fig,ax=plt.subplots(figsize=(12,7)); ax.barh(labels,tt.rmse); ax.invert_yaxis(); ax.set_xlabel('RMSE'); ax.set_title('Temporal 2020 performance ranking'); fig.tight_layout(); fig.savefig(FIG/'Figure_4_temporal_rmse_ranking.png',dpi=220,bbox_inches='tight'); plt.close(fig)

    # F5 temporal observed vs predicted for overall best nonbaseline model
    best=ranking[ranking.model!='M0_persistence'].iloc[0]
    d=preds[(preds.design==best.design)&(preds.split=='temporal_2020')&(preds.model==best.model)]
    fig,ax=plt.subplots(figsize=(5,5)); ax.scatter(d.observed,d.prediction); lo=min(d.observed.min(),d.prediction.min()); hi=max(d.observed.max(),d.prediction.max()); ax.plot([lo,hi],[lo,hi],'--'); ax.set_xlabel('Observed incidence'); ax.set_ylabel('Predicted incidence'); ax.set_title(f'Best temporal model: {best.design} / {best.model}'); fig.tight_layout(); fig.savefig(FIG/'Figure_5_best_temporal_observed_vs_predicted.png',dpi=220); plt.close(fig)

    # F6 paired country improvement best model over persistence
    b=preds[(preds.design==best.design)&(preds.split=='temporal_2020')&(preds.model=='M0_persistence')][[GROUP,'observed','prediction']].rename(columns={'prediction':'base'})
    m=preds[(preds.design==best.design)&(preds.split=='temporal_2020')&(preds.model==best.model)][[GROUP,'prediction']].rename(columns={'prediction':'modelp'})
    z=b.merge(m,on=GROUP); z['improvement']=(z.observed-z.base).abs()-(z.observed-z.modelp).abs(); z=z.sort_values('improvement')
    fig,ax=plt.subplots(figsize=(8,8)); ax.barh(z[GROUP],z.improvement); ax.axvline(0,ls='--'); ax.set_xlabel('Absolute-error improvement over persistence'); ax.set_title('Country-level temporal improvement: best model'); fig.tight_layout(); fig.savefig(FIG/'Figure_6_country_improvement_best_model.png',dpi=220,bbox_inches='tight'); plt.close(fig)

    # F7 bootstrap RMSE improvement CIs for all temporal models vs persistence
    q=boot[boot.comparison=='vs_persistence'].copy().sort_values('delta_rmse',ascending=False).head(16)
    y=np.arange(len(q)); err=np.vstack([q.delta_rmse-q.delta_rmse_ci_low,q.delta_rmse_ci_high-q.delta_rmse])
    fig,ax=plt.subplots(figsize=(11,7)); ax.errorbar(q.delta_rmse,y,xerr=err,fmt='o'); ax.axvline(0,ls='--'); ax.set_yticks(y); ax.set_yticklabels([f'{a} | {b}' for a,b in zip(q.design,q.model)]); ax.invert_yaxis(); ax.set_xlabel('RMSE improvement over persistence (95% bootstrap CI)'); ax.set_title('Paired temporal uncertainty'); fig.tight_layout(); fig.savefig(FIG/'Figure_7_bootstrap_rmse_improvement.png',dpi=220,bbox_inches='tight'); plt.close(fig)

    # F8 VC incremental effect (M4 vs M2, M6 vs M5)
    q=boot[boot.comparison!='vs_persistence'].copy().sort_values('delta_rmse',ascending=False)
    y=np.arange(len(q)); err=np.vstack([q.delta_rmse-q.delta_rmse_ci_low,q.delta_rmse_ci_high-q.delta_rmse])
    fig,ax=plt.subplots(figsize=(10,6)); ax.errorbar(q.delta_rmse,y,xerr=err,fmt='o'); ax.axvline(0,ls='--'); ax.set_yticks(y); ax.set_yticklabels([f'{a} | {b} | {c}' for a,b,c in zip(q.design,q.model,q.comparison)]); ax.invert_yaxis(); ax.set_xlabel('Incremental RMSE improvement from adding VC'); ax.set_title('Ablation: incremental value of VC in temporal 2020 test'); fig.tight_layout(); fig.savefig(FIG/'Figure_8_incremental_vc_effect.png',dpi=220,bbox_inches='tight'); plt.close(fig)

    # F9 alpha distribution grouped
    gm=metrics[(metrics.split=='grouped')&(metrics.model!='M0_persistence')].copy()
    # top 8 by grouped RMSE
    ids=grouped[grouped.model!='M0_persistence'].sort_values('rmse_mean').head(8)[['design','model']]
    gm=gm.merge(ids,on=['design','model'])
    labs=[]; data=[]
    for (des,mod),dd in gm.groupby(['design','model']): labs.append(f'{des}|{mod}'); data.append(dd.alpha.dropna().values)
    fig,ax=plt.subplots(figsize=(12,6)); ax.boxplot(data,vert=False,labels=labs); ax.set_xlabel('Selected blending alpha'); ax.set_title('Nested grouped alpha selection'); fig.tight_layout(); fig.savefig(FIG/'Figure_9_alpha_selection.png',dpi=220,bbox_inches='tight'); plt.close(fig)

    # F10 compact feature selection frequency
    ss=sel[sel.model.str.contains('M2_|M4_',regex=True)].copy()
    counts={}
    for fs in ss.features:
        for f in fs.split('|'):
            if f in COMPACT_CANDIDATES: counts[f]=counts.get(f,0)+1
    q=pd.Series(counts).sort_values()
    fig,ax=plt.subplots(figsize=(8,5)); ax.barh(q.index,q.values); ax.set_xlabel('Number of outer fits retaining feature'); ax.set_title('Leakage-safe compact environmental feature retention'); fig.tight_layout(); fig.savefig(FIG/'Figure_10_compact_feature_retention.png',dpi=220,bbox_inches='tight'); plt.close(fig)

    pd.DataFrame({'design':['one_lag','two_lag'], 'eligible_observations':vals,
                  'eligible_years':['2016, 2017, 2018, 2020','2017, 2018, 2020']}).to_csv(TAB/'dataset_design_summary.csv',index=False)

# ------------------------------------------------------------------
# 12. Draft updated Methods/Results/Discussion/Conclusion after results
# ------------------------------------------------------------------
def write_manuscript_draft(annual,grouped,temporal,boot,ranking,sel):
    best=ranking[ranking.model!='M0_persistence'].iloc[0]
    bt=temporal[(temporal.design==best.design)&(temporal.model==best.model)].iloc[0]
    bg=grouped[(grouped.design==best.design)&(grouped.model==best.model)].iloc[0]
    base_t=temporal[(temporal.design==best.design)&(temporal.model=='M0_persistence')].iloc[0]
    # incremental VC best targeted comparison by delta RMSE
    inc=boot[boot.comparison!='vs_persistence'].sort_values('delta_rmse',ascending=False).iloc[0]
    text=f'''% BIML v2 manuscript replacement draft -- generated from leakage-controlled experiment\n\n\\subsection{{Leakage-controlled preprocessing and biological feature construction}}\nThe monthly environmental--epidemiological data were aggregated to country-year records. Historical incidence features were generated strictly chronologically within country. Two analytical designs were evaluated: a one-lag design, which required the last observed incidence value and retained 2016, 2017, 2018, and 2020 as eligible target years, and a two-lag design, which additionally required the second-most-recent incidence observation and therefore retained 2017, 2018, and 2020. Because 2019 was absent, the lag variables represent the most recent available observations rather than necessarily the immediately preceding calendar year.\n\nVectorial capacity was represented using the Ross--Macdonald structure, $C=ma^2p^n/[-\\ln(p)]$. Temperature suitability was defined as $s_T=\\exp[-(T-25)^2/(2\\times5^2)]$. The operational parameterization used $a=0.05+0.03s_T$, $p=0.70+0.25s_T$, $n=20-8s_T$, and $m=0.25+0.75s_Rs_V(0.5+0.5s_P)$. Rainfall, vegetation, population and final VC normalization parameters were estimated only from the corresponding training partition and then applied unchanged to held-out observations. Thus, no held-out-country or 2020 feature-distribution information contributed to VC scaling. VC lag and VC change were then formed chronologically within country.\n\n\\subsection{{Ablation design and feature reduction}}\nFor each lag design, a model ladder separated the contributions of incidence history, environmental information, and VC-derived information. The ladder comprised persistence, history only, history plus a compact environmental representation, history plus VC, history plus compact environment plus VC, history plus the full environmental representation, and history plus the full environmental representation plus VC. Compact environmental sets were selected independently within each training partition by correlation pruning at $|r|\\geq0.90$ using a fixed feature-priority order; held-out observations were never used for feature selection. Random forest and gradient boosting regressors used fixed hyperparameters to limit optimization degrees of freedom in the small sample.\n\n\\subsection{{Nested blending and validation}}\nFor every non-baseline model, the final prediction was a convex blend of the persistence prediction and the machine-learning prediction, $\\hat y=(1-\\alpha)y_{{lag1}}+\\alpha\\hat y_{{ML}}$. The blending coefficient $\\alpha\\in\\{{0,0.05,\\ldots,1\\}}$ was selected exclusively from grouped out-of-fold predictions generated inside the training partition. In each inner fold, VC scaling and compact-feature selection were re-estimated from the inner training countries before prediction of inner validation countries. Outer evaluation used ten repeated 80/20 country-grouped splits and a separate temporal holdout in which 2020 was never used for training or transformation fitting.\n\n\\subsection{{Uncertainty analysis}}\nTemporal comparisons used paired country-level bootstrap resampling (2,000 replicates) so that baseline and candidate predictions were resampled for the same countries. Bootstrap intervals were calculated for changes in MAE and RMSE. Repeated grouped splits were summarized descriptively because their test sets overlap and therefore should not be interpreted as independent inferential replicates.\n\n\\section{{Results -- v2 replacement draft}}\nThe one-lag design retained {len(eligible_keys(annual,'one_lag'))} eligible country-year observations, whereas the two-lag design retained {len(eligible_keys(annual,'two_lag'))}. Under the pre-specified combined ranking based on temporal and grouped RMSE, the best non-baseline configuration was \\texttt{{{best.model}}} under the \\texttt{{{best.design}}} design. Its 2020 temporal performance was MAE={bt.mae:.3f}, RMSE={bt.rmse:.3f}, and $R^2$={bt.r2:.3f}, compared with persistence MAE={base_t.mae:.3f}, RMSE={base_t.rmse:.3f}, and $R^2$={base_t.r2:.3f}. Its mean grouped performance was MAE={bg.mae_mean:.3f}, RMSE={bg.rmse_mean:.3f}, and $R^2$={bg.r2_mean:.3f}.\n\nThe ablation analysis directly tested the incremental contribution of VC-derived information rather than inferring that contribution from a full-model comparison alone. The strongest temporal VC increment in the tested ablations was {inc.comparison} under {inc.design} ({inc.model}), with RMSE improvement={inc.delta_rmse:.3f} and a 95\\% paired-bootstrap interval [{inc.delta_rmse_ci_low:.3f}, {inc.delta_rmse_ci_high:.3f}]. Positive values indicate lower RMSE after adding VC. The complete model ladder and all uncertainty intervals are reported in the generated v2 tables.\n\n\\section{{Discussion -- v2 replacement draft}}\nThe leakage-controlled experiment changes the interpretation from a simple comparison between persistence and a large hybrid model to a controlled decomposition of where predictive information arises. In particular, the ablation ladder distinguishes the value of VC-derived information from gains attributable to additional incidence history or environmental covariates. This is important because a persistence baseline and a full hybrid representation differ along several dimensions simultaneously. The v2 design therefore provides a more defensible test of biological information integration.\n\nThe one-lag/two-lag comparison also quantifies the trade-off between richer epidemiological history and sample size. Retaining a second incidence lag removes 2016 from the target set, while the one-lag design preserves it. The preferred design should therefore be selected from out-of-sample performance rather than from the assumption that either more observations or more lagged history is inherently superior.\n\n\\section{{Conclusion -- v2 replacement draft}}\nThis revised experiment evaluates biologically informed malaria-incidence prediction under leakage-controlled transformation fitting, nested grouped blend selection, explicit ablation, and paired temporal uncertainty. The principal conclusion should be based on the incremental VC comparisons and the consistency of temporal and grouped results, rather than on feature importance alone. The final manuscript should report the exact v2 tables and avoid claims of universal superiority when bootstrap intervals or grouped results do not support them.\n'''
    (OUT/'manuscript_v2_replacement_sections.tex').write_text(text)

def write_readme(annual,metrics,grouped,temporal,boot,ranking):
    best=ranking[ranking.model!='M0_persistence'].iloc[0]
    lines=[]
    lines.append('BIML CLEAN V2 EXPERIMENTAL PIPELINE\n')
    lines.append('Purpose: leakage-controlled one-lag/two-lag ablation study with nested grouped alpha selection.\n')
    lines.append(f"Raw annual country-year rows after missing-target removal: {len(annual)}")
    lines.append(f"One-lag eligible targets: {len(eligible_keys(annual,'one_lag'))}")
    lines.append(f"Two-lag eligible targets: {len(eligible_keys(annual,'two_lag'))}\n")
    lines.append('Primary ranking rule: temporal RMSE rank + grouped mean RMSE rank; lower combined rank is better.')
    lines.append(f"Best non-baseline by this rule: {best.design} / {best.model}")
    lines.append(f"Temporal RMSE: {best.rmse:.4f}; grouped mean RMSE: {best.rmse_mean:.4f}\n")
    lines.append('IMPORTANT INTERPRETATION NOTES:')
    lines.append('- VC distributional scaling uses training data only. Fixed thermal constants are not estimated from test data.')
    lines.append('- Compact environmental feature pruning is fit within training partitions and re-fit inside inner folds.')
    lines.append('- Alpha is selected from inner grouped OOF predictions only.')
    lines.append('- 2020 is a temporal prediction/holdout, not a prospective forecast, because 2020 environmental covariates are used.')
    lines.append('- Grouped holdout countries still use their own historical incidence as an input; this is not zero-history country prediction.')
    lines.append('- Grouped repeated-split SDs are descriptive because test sets overlap.')
    (OUT/'README_v2.txt').write_text('\n'.join(lines))

if __name__=='__legacy_main__':
    annual=aggregate_annual()
    metrics,preds,sel=run_all(annual)
    boot=bootstrap_temporal(preds)
    grouped=grouped_uncertainty(metrics)
    temporal=temporal_summary(metrics)
    ranking=ranking_table(grouped,temporal)
    generate_figures(annual,metrics,preds,grouped,temporal,boot,ranking,sel)
    write_manuscript_draft(annual,grouped,temporal,boot,ranking,sel)
    write_readme(annual,metrics,grouped,temporal,boot,ranking)
    print('\nTOP 20 RANKING')
    print(ranking[ranking.model!='M0_persistence'].head(20).to_string(index=False))
    print('\nTEMPORAL BEST BY DESIGN')
    print(temporal[temporal.model!='M0_persistence'].sort_values('rmse').groupby('design').head(5).to_string(index=False))
    print('\nBOOTSTRAP VC INCREMENT')
    print(boot[boot.comparison!='vs_persistence'].sort_values('delta_rmse',ascending=False).to_string(index=False))
