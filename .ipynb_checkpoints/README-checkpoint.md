# BIML Final Reproducible Pipeline

This package is aligned to the revised manuscript:
**Biologically Informed Machine Learning for Malaria Incidence Prediction Using Vectorial-Capacity-Derived Information**.

## Run
From JupyterLab/terminal in this folder:

```bash
python run_final_pipeline.py
```

The complete experiment is computationally intensive because alpha selection is nested inside repeated country-grouped validation.

## Manuscript-aligned analysis
- Raw monthly data: 1,920 rows, 32 countries.
- Years: 2015, 2016, 2017, 2018, 2020 (2019 absent).
- Mali 2020 incidence is missing (12 monthly rows).
- Annual target-complete dataset: 159 country-years.
- One-lag design: 127 target observations.
- Two-lag design: 95 target observations.
- M0–M6 controlled ablation.
- RF: 500 trees, depth 5, min leaf 3, max_features 0.8.
- GB: 180 stages, learning rate 0.03, depth 2, min leaf 4, Huber loss.
- 10 repeated country-grouped splits.
- Separate 2020 temporal holdout.
- Training-only VC scaling and compact-feature pruning.
- Nested grouped selection of blending alpha.
- Paired country bootstrap for temporal uncertainty.

## Exact manuscript figure filenames
The pipeline writes:
1. `Figure_2_observed_vs_predicted_corrected.png`
2. `Figure_3_error_distributions_corrected.png`
3. `Figure_4_error_boxplot_corrected.png`
4. `Figure_7_country_improvement_hybrid_rf.png`
5. `Figure_7_country_improvement_hybrid_gb.png`
6. `Figure_9_vc_feature_importance.png`

`Malaria_Transmission_Modelling_Workflow.png` is a conceptual Methods schematic and is not generated from model outputs.

## Critical reproducibility safeguard
`outputs/tables/MANUSCRIPT_ALIGNMENT_CHECKS.csv` compares reproduced central values with the manuscript values. If a check fails, inspect the discrepancy rather than silently changing the paper.

## Important alignment correction
The nonlinear environmental equations follow the revised manuscript exactly:
- `Rainfall_sq = Rainfall_total^2`
- `Temp_Rain = Temp_mean * Rainfall_total`

The VC representation is called **vectorial capacity (VC)** / **VC-derived information**. It is not labelled a proxy.
