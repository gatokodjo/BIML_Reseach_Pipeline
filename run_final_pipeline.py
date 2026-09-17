"""
BIML PIPELINE
Biologically Informed Machine Learning for Malaria Incidence Prediction
Using Vectorial-Capacity-Derived Information

This runner is aligned to the manuscript:
- 32 countries; monthly raw data
- annual target-complete dataset = 159
- one-lag = 127; two-lag = 95
- M0--M6 ablation
- RF: 500 trees, depth 5, min leaf 3, max_features 0.8
- GB: 180 stages, learning rate 0.03, depth 2, min leaf 4, Huber
- repeated country-grouped validation (10 splits)
- 2020 temporal holdout
- training-only VC scaling / compact-feature selection
- nested grouped alpha selection
- paired country bootstrap for temporal comparisons
- manuscript figure filenames reproduced exactly
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import biml_core as core

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
FIG = OUT / "figures"
TAB = OUT / "tables"
DAT = OUT / "data"
META = OUT / "metadata"
for p in (FIG, TAB, DAT, META):
    p.mkdir(parents=True, exist_ok=True)

EXPECTED = {
    "raw_rows": 1920,
    "countries": 32,
    "annual_complete": 159,
    "one_lag": 127,
    "two_lag": 95,
    "temporal_2020_countries": 31,
    "M0_temporal_rf_reference": {"mae":32.2710, "rmse":54.6111, "r2":0.6571},
    "M1_temporal_rf": {"mae":31.5723, "rmse":52.7665, "r2":0.6798},
    "M3_temporal_rf": {"mae":30.5827, "rmse":50.4361, "r2":0.7075},
    "M3_vs_M1_rmse_reduction": 2.330454,
    "M3_vs_M1_mae_reduction": 0.989604,
}

def audit_raw():
    d = pd.read_csv(core.DATA_PATH)
    rows = {
        "raw_rows": len(d),
        "countries": d.Country.nunique(),
        "years": sorted(d.Year.unique().tolist()),
        "duplicates": int(d.duplicated().sum()),
        "missing_incidence_rows": int(d.Incidence.isna().sum()),
        "missing_incidence_country_years":
            d.loc[d.Incidence.isna(), ["Country","Year"]].drop_duplicates().to_dict("records"),
        "country_year_month_counts":
            d.groupby(["Country","Year"]).size().value_counts().sort_index().to_dict(),
    }
    print("RAW DATA AUDIT")
    print(json.dumps(rows, indent=2))
    assert rows["raw_rows"] == 1920
    assert rows["countries"] == 32
    assert rows["years"] == [2015, 2016, 2017, 2018, 2020]
    assert rows["duplicates"] == 0
    assert rows["missing_incidence_rows"] == 12
    return d

def manuscript_tables(metrics, grouped, temporal, boot):
    # One-lag RF table used in Results
    g = grouped[(grouped.design=="one_lag")].copy()
    rf_models = [
        "M0_persistence",
        "M1_history_rf","M2_history_env_compact_rf","M3_history_vc_rf",
        "M4_history_env_compact_vc_rf","M5_history_env_full_rf",
        "M6_history_env_full_vc_rf"
    ]
    g = g[g.model.isin(rf_models)].copy()
    order = {m:i for i,m in enumerate(rf_models)}
    g["order"] = g.model.map(order)
    g.sort_values("order").to_csv(TAB/"Table_grouped_one_lag_RF.csv", index=False)

    t = temporal[(temporal.design=="one_lag") & temporal.model.isin(rf_models)].copy()
    t["order"] = t.model.map(order)
    t.sort_values("order").to_csv(TAB/"Table_temporal_2020_one_lag_RF.csv", index=False)

    boot.to_csv(TAB/"Table_bootstrap_temporal_comparisons.csv", index=False)

def temporal_m3_predictions(preds, learner):
    model = f"M3_history_vc_{learner}"
    return preds[
        (preds.design=="one_lag") &
        (preds.split=="temporal_2020") &
        (preds.model==model)
    ].copy()

def fig2_observed_predicted(preds):
    base = preds[(preds.design=="one_lag")&(preds.split=="temporal_2020")&
                 (preds.model=="M0_persistence")][["Country","observed","prediction"]].copy()
    rf = temporal_m3_predictions(preds, "rf")[["Country","prediction"]].rename(columns={"prediction":"RF History+VC"})
    gb = temporal_m3_predictions(preds, "gb")[["Country","prediction"]].rename(columns={"prediction":"GB History+VC"})
    z = base.rename(columns={"prediction":"Persistence"}).merge(rf,on="Country").merge(gb,on="Country")

    fig, ax = plt.subplots(figsize=(7.2,6.2))
    ax.scatter(z.observed, z["Persistence"], marker="o", label="Persistence", alpha=.8)
    ax.scatter(z.observed, z["RF History+VC"], marker="s", label="History+VC RF", alpha=.8)
    ax.scatter(z.observed, z["GB History+VC"], marker="^", label="History+VC GB", alpha=.8)
    lo = min(z.observed.min(), z[["Persistence","RF History+VC","GB History+VC"]].min().min())
    hi = max(z.observed.max(), z[["Persistence","RF History+VC","GB History+VC"]].max().max())
    ax.plot([lo,hi],[lo,hi],"--",linewidth=1)
    ax.set_xlabel("Observed malaria incidence")
    ax.set_ylabel("Predicted malaria incidence")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG/"Figure_2_observed_vs_predicted_corrected.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    z.to_csv(DAT/"Figure_2_prediction_values.csv", index=False)

def diagnostic_frame(preds):
    frames=[]
    for model,label in [
        ("M0_persistence","Persistence"),
        ("M3_history_vc_rf","History+VC RF"),
        ("M3_history_vc_gb","History+VC GB")
    ]:
        q=preds[(preds.design=="one_lag")&(preds.split=="temporal_2020")&(preds.model==model)].copy()
        q["Model"]=label
        q["Absolute_error"]=(q.observed-q.prediction).abs()
        q["Signed_error"]=q.prediction-q.observed
        frames.append(q)
    return pd.concat(frames, ignore_index=True)

def fig3_error_cdf(preds):
    d=diagnostic_frame(preds)
    fig,ax=plt.subplots(figsize=(7.2,5.2))
    for label,q in d.groupby("Model",sort=False):
        x=np.sort(q.Absolute_error.to_numpy())
        y=np.arange(1,len(x)+1)/len(x)
        ax.step(x,y,where="post",label=label)
    ax.axvline(25,linestyle="--",linewidth=1)
    ax.set_xlabel("Absolute prediction error")
    ax.set_ylabel("Cumulative proportion of countries")
    ax.set_ylim(0,1.02)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG/"Figure_3_error_distributions_corrected.png",dpi=300,bbox_inches="tight")
    plt.close(fig)
    summary=(d.assign(within_25=d.Absolute_error<=25)
             .groupby("Model").agg(n=("Country","size"),
                                   proportion_within_25=("within_25","mean"),
                                   median_absolute_error=("Absolute_error","median"))
             .reset_index())
    summary.to_csv(TAB/"Figure_3_error_distribution_summary.csv",index=False)

def fig4_error_boxplot(preds):
    d=diagnostic_frame(preds)
    labels=["Persistence","History+VC RF","History+VC GB"]
    vals=[d.loc[d.Model==x,"Absolute_error"].to_numpy() for x in labels]
    fig,ax=plt.subplots(figsize=(7.2,5.2))
    ax.boxplot(vals,labels=labels,showfliers=True)
    ax.set_ylabel("Absolute prediction error")
    fig.tight_layout()
    fig.savefig(FIG/"Figure_4_error_boxplot_corrected.png",dpi=300,bbox_inches="tight")
    plt.close(fig)

def country_grouped_improvement(preds, learner):
    m=f"M3_history_vc_{learner}"
    base=preds[(preds.design=="one_lag")&(preds.split=="grouped")&
               (preds.model=="M0_persistence")][["repeat","Country","observed","prediction"]].copy()
    mod=preds[(preds.design=="one_lag")&(preds.split=="grouped")&
              (preds.model==m)][["repeat","Country","prediction"]].rename(columns={"prediction":"model_prediction"})
    z=base.rename(columns={"prediction":"persistence_prediction"}).merge(mod,on=["repeat","Country"],how="inner")
    z["persistence_abs_error"]=(z.observed-z.persistence_prediction).abs()
    z["model_abs_error"]=(z.observed-z.model_prediction).abs()
    q=z.groupby("Country",as_index=False).agg(
        persistence_mae=("persistence_abs_error","mean"),
        history_vc_mae=("model_abs_error","mean"),
        n_test_appearances=("repeat","nunique")
    )
    q["mae_improvement"]=q.persistence_mae-q.history_vc_mae
    return q.sort_values("mae_improvement")

def country_figures(preds):
    for learner,fn in [
        ("rf","Figure_7_country_improvement_hybrid_rf.png"),
        ("gb","Figure_7_country_improvement_hybrid_gb.png")
    ]:
        q=country_grouped_improvement(preds,learner)
        q.to_csv(TAB/f"Figure_7_{learner.upper()}_country_values.csv",index=False)
        fig,ax=plt.subplots(figsize=(7.2,8.5))
        ax.barh(q.Country,q.mae_improvement)
        ax.axvline(0,linestyle="--",linewidth=1)
        ax.set_xlabel("MAE improvement over persistence")
        ax.set_ylabel("")
        fig.tight_layout()
        fig.savefig(FIG/fn,dpi=300,bbox_inches="tight")
        plt.close(fig)
        print(f"{learner.upper()}: positive MAE improvement in {(q.mae_improvement>0).sum()}/{len(q)} countries")

def fit_temporal_m3_importance(annual, learner):
    design="one_lag"; rep="vc"
    keys=core.eligible_keys(annual,design).reset_index(drop=True)
    trk=keys[keys.Year<2020].copy()
    tek=keys[keys.Year==2020].copy()
    context=core.get_context_train(annual,trk.Country.unique(),2020)
    params=core.fit_vc_params(context)
    transformed=core.apply_vc(annual,params)
    tr=core.rows_by_keys(transformed,trk)
    te=core.rows_by_keys(transformed,tek)
    feats,_=core.feature_names(tr,design,rep)
    # Same seed used by the temporal outer fit in the established pipeline (repeat=1).
    model=core.make_model(learner,core.SEED+1)
    model.fit(tr[feats],tr[core.TARGET])
    imp=pd.Series(model.feature_importances_,index=feats)
    return imp,model,params

def fig9_importance(annual):
    rf,_,_=fit_temporal_m3_importance(annual,"rf")
    gb,_,_=fit_temporal_m3_importance(annual,"gb")
    features=["VC_norm","VC_lag","VC_change"]
    vals=pd.DataFrame({"Feature":features,
                       "Random Forest":[rf.get(x,np.nan) for x in features],
                       "Gradient Boosting":[gb.get(x,np.nan) for x in features]})
    vals.to_csv(TAB/"Figure_9_vc_feature_importance_values.csv",index=False)
    full=pd.DataFrame({
        "Feature":sorted(set(rf.index)|set(gb.index)),
    })
    full["Random Forest"]=full.Feature.map(rf)
    full["Gradient Boosting"]=full.Feature.map(gb)
    full.to_csv(TAB/"Feature_importance_full_M3_temporal.csv",index=False)

    x=np.arange(len(features)); w=.36
    fig,ax=plt.subplots(figsize=(7.2,5.2))
    ax.bar(x-w/2,vals["Random Forest"],width=w,label="Random Forest")
    ax.bar(x+w/2,vals["Gradient Boosting"],width=w,label="Gradient Boosting")
    ax.set_xticks(x); ax.set_xticklabels(features)
    ax.set_ylabel("Impurity-based feature importance")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG/"Figure_9_vc_feature_importance.png",dpi=300,bbox_inches="tight")
    plt.close(fig)
    print("M3 temporal feature importance:")
    print(full.to_string(index=False))

def verify_against_manuscript(annual, grouped, temporal):
    checks=[]
    def add(name, actual, expected, tol=5e-4):
        ok=abs(float(actual)-float(expected))<=tol
        checks.append({"check":name,"actual":float(actual),"expected":float(expected),
                       "tolerance":tol,"pass":ok})
    add("annual target-complete rows",len(annual),159,0)
    add("one-lag eligible rows",len(core.eligible_keys(annual,"one_lag")),127,0)
    add("two-lag eligible rows",len(core.eligible_keys(annual,"two_lag")),95,0)

    t=temporal[temporal.design=="one_lag"].set_index("model")
    for model,prefix,exp in [
        ("M0_persistence","M0",EXPECTED["M0_temporal_rf_reference"]),
        ("M1_history_rf","M1 RF",EXPECTED["M1_temporal_rf"]),
        ("M3_history_vc_rf","M3 RF",EXPECTED["M3_temporal_rf"])
    ]:
        for metric in ["mae","rmse","r2"]:
            add(f"{prefix} temporal {metric}",t.loc[model,metric],exp[metric])

    add("M3 vs M1 temporal RMSE reduction",
        t.loc["M1_history_rf","rmse"]-t.loc["M3_history_vc_rf","rmse"],
        EXPECTED["M3_vs_M1_rmse_reduction"],1e-3)
    add("M3 vs M1 temporal MAE reduction",
        t.loc["M1_history_rf","mae"]-t.loc["M3_history_vc_rf","mae"],
        EXPECTED["M3_vs_M1_mae_reduction"],1e-3)

    c=pd.DataFrame(checks)
    c.to_csv(TAB/"MANUSCRIPT_ALIGNMENT_CHECKS.csv",index=False)
    print("\nMANUSCRIPT ALIGNMENT CHECKS")
    print(c.to_string(index=False))
    if not c["pass"].all():
        print("\nWARNING: At least one manuscript value differs from the reproduced pipeline.")
        print("Do not silently overwrite manuscript values; inspect MANUSCRIPT_ALIGNMENT_CHECKS.csv.")
    return c

def main():
    audit_raw()
    annual=core.aggregate_annual()
    assert len(annual)==159
    assert len(core.eligible_keys(annual,"one_lag"))==127
    assert len(core.eligible_keys(annual,"two_lag"))==95

    print("\nRunning M0--M6 grouped and temporal experiments...")
    metrics,preds,sel=core.run_all(annual)
    boot=core.bootstrap_temporal(preds)
    grouped=core.grouped_uncertainty(metrics)
    temporal=core.temporal_summary(metrics)
    ranking=core.ranking_table(grouped,temporal)

    manuscript_tables(metrics,grouped,temporal,boot)
    fig2_observed_predicted(preds)
    fig3_error_cdf(preds)
    fig4_error_boxplot(preds)
    country_figures(preds)
    fig9_importance(annual)
    checks=verify_against_manuscript(annual,grouped,temporal)

    manifest={
        "dataset":str(core.DATA_PATH),
        "result_figures":[
            "Figure_2_observed_vs_predicted_corrected.png",
            "Figure_3_error_distributions_corrected.png",
            "Figure_4_error_boxplot_corrected.png",
            "Figure_7_country_improvement_hybrid_rf.png",
            "Figure_7_country_improvement_hybrid_gb.png",
            "Figure_9_vc_feature_importance.png",
        ],
        "note":"Malaria_Transmission_Modelling_Workflow.png is a conceptual Methods schematic, not an empirical model-output figure.",
        "all_alignment_checks_passed":bool(checks["pass"].all())
    }
    (META/"manifest.json").write_text(json.dumps(manifest,indent=2))
    print("\nDONE. Outputs:",OUT)

if __name__=="__main__":
    main()
