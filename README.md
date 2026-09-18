# Biologically Informed Machine Learning for Malaria Incidence Prediction

## Overview

This repository contains the data-processing workflow, modelling code,
evaluation pipeline, and reproducibility outputs associated with the study:

**Biologically Informed Machine Learning Framework for Malaria Incidence Prediction Using Vectorial-Capacity-Derived Information**

The study investigates whether **vectorial-capacity-derived information**
provides incremental predictive information beyond historical malaria
incidence and conventional environmental predictors.

Rather than treating vectorial capacity (VC) as a direct entomological
measurement, the framework uses an **environmentally parameterized VC
representation** derived from temperature, rainfall, vegetation, and
population information. Its predictive contribution is evaluated through
controlled feature-ablation experiments using Random Forest and Gradient
Boosting models.

The repository is intended to support transparent and reproducible
evaluation of the analyses reported in the accompanying manuscript.

---

## Study Design

The analysis uses monthly environmental, demographic, and malaria-incidence
data for **32 sub-Saharan African countries** covering:

- 2015
- 2016
- 2017
- 2018
- 2020

The year 2019 is absent from the source analytical dataset.

The raw dataset contains **1,920 monthly observations**. Environmental
variables are aggregated to annual country-level representations before
model fitting.

After annual aggregation:

- 160 country-year records are represented before target-missingness removal;
- Mali has no malaria-incidence value for 2020;
- 159 target-complete country-year observations remain;
- 127 observations are eligible for the primary one-lag analysis; and
- 95 observations are eligible for the two-lag sensitivity analysis.

Because 2019 is unavailable, lagged variables are defined using the
**last available chronological observation**, rather than necessarily the
previous calendar year. For example, the last-observed incidence used for
the 2020 target corresponds to 2018.

---

## Data Sources

The analytical dataset integrates information from publicly available
sources:

| Variable | Source |
|---|---|
| Malaria incidence | World Health Organization Global Health Observatory |
| Land-surface temperature (LST) | Landsat 8 |
| NDVI | Landsat 8 |
| NDWI | Landsat 8 |
| Rainfall | CHIRPS |
| Population density | WorldPop |

Landsat-derived environmental information was processed using
**Google Earth Engine**.

The repository contains the processed analytical data required to reproduce
the reported experiments. Users wishing to reconstruct the dataset from the
original products should consult the corresponding data providers and their
applicable terms of use.

---

## Biologically Informed Vectorial-Capacity Representation

The framework uses the classical vectorial-capacity structure

\[
C = \frac{ma^2p^n}{-\ln(p)},
\]

where:

- \(m\) represents mosquito density relative to humans;
- \(a\) represents biting rate;
- \(p\) represents daily mosquito survival probability; and
- \(n\) represents the extrinsic incubation period.

Direct country-year entomological measurements of these quantities were not
available. They are therefore **operationally parameterized from
environmental and demographic information**.

The resulting VC-derived variables are:

- `VC_norm` — normalized contemporaneous VC representation;
- `VC_lag` — last-observed VC representation; and
- `VC_change` — change relative to the last available VC observation.

These variables should not be interpreted as direct country-specific
measurements of mosquito abundance, biting behaviour, survival, parasite
development, or biological transmission intensity.

---

## Feature-Ablation Experiment

The principal experiment separates three sources of predictive information:

- **H** — historical malaria incidence;
- **E** — environmental information; and
- **V** — VC-derived information.

Seven configurations are evaluated:

| Model | Representation |
|---|---|
| M0 | Incidence persistence |
| M1 | History |
| M2 | History + compact environmental features |
| M3 | History + VC |
| M4 | History + compact environmental features + VC |
| M5 | History + full environmental features |
| M6 | History + full environmental features + VC |

The principal direct VC-ablation comparisons are:

- M3 vs M1;
- M4 vs M2; and
- M6 vs M5.

These comparisons assess whether adding VC-derived information changes
predictive performance when the remaining information is held as comparable
as possible.

---

## Machine-Learning Models

Two tree-based regression algorithms are evaluated.

### Random Forest

- 500 trees
- maximum depth: 5
- minimum terminal-leaf size: 3
- maximum feature proportion: 0.8

### Gradient Boosting

- 180 boosting stages
- learning rate: 0.03
- maximum depth: 2
- minimum terminal-leaf size: 4
- Huber loss

Machine-learning predictions are blended with the incidence-persistence
prediction. The blending weight is selected exclusively from training data
using grouped inner validation.

---

## Evaluation Strategy

Two complementary evaluation settings are implemented.

### Country-grouped validation

The pipeline performs **10 repeated country-grouped train-test splits**.
Observations belonging to a country are assigned exclusively to either the
training or test partition within a split.

This evaluation examines generalization across countries.

### 2020 temporal holdout

Eligible observations before 2020 are used for model development, and the
models are evaluated on the **31 countries with observed malaria incidence
in 2020**.

Because contemporaneous 2020 environmental information is included, this
experiment should be interpreted as a **temporal holdout**, not as a
strictly prospective forecast.

Performance is evaluated using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Coefficient of Determination (\(R^2\))

Paired model comparisons in the temporal evaluation use
**country-level bootstrap resampling with 2,000 replicates**.

---

## Leakage Control

The pipeline is designed to prevent information from the evaluation data
from influencing model development.

In particular:

- VC scaling parameters are estimated from training data only;
- environmental feature pruning is performed from training data only;
- blending weights are selected within training data;
- preprocessing is re-estimated inside the relevant validation partitions;
- incidence and VC lag features are constructed chronologically; and
- held-out target values are never used for feature construction or model
  selection.

These safeguards are implemented directly in the reproducible modelling
pipeline.

---

## Repository Structure

The principal components of the repository are:

```text
BIML_Research_Pipeline/
│
├── README.md
├── requirements.txt
├── biml_core.py
├── run_final_pipeline.py
├── BIML_Final_Pipeline.ipynb
│
├── data/
│   └── [analytical dataset]
│
└── outputs/
    ├── data/
    ├── tables/
    ├── figures/
    └── metadata/
```

### `biml_core.py`

Contains the core analytical implementation, including:

- annual aggregation;
- chronological lag construction;
- VC parameterization;
- training-only scaling;
- environmental feature selection;
- model construction;
- nested blending-weight selection;
- grouped evaluation;
- temporal evaluation; and
- bootstrap comparison procedures.

### `run_final_pipeline.py`

Runs the complete manuscript-aligned reproducibility workflow and generates
the numerical outputs, tables, and empirical figures used in the study.

### `BIML_Final_Pipeline.ipynb`

Provides a notebook-based interface for executing and inspecting the
analysis interactively.

---

## Installation

A Python environment can be created using the dependencies listed in
`requirements.txt`.

For example:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows:

```bash
.venv\Scripts\activate
pip install -r requirements.txt
```

---

## Running the Analysis

From the repository root, execute:

```bash
python run_final_pipeline.py
```

The complete experiment may require substantial computation because model
blending is selected through nested country-grouped validation within the
repeated outer evaluation.

The notebook can alternatively be opened in JupyterLab:

```bash
jupyter lab BIML_Final_Pipeline.ipynb
```

---

## Reproduced Figures

The computational pipeline generates the following empirical manuscript
figures:

1. `Figure_2_observed_vs_predicted_corrected.png`
2. `Figure_3_error_distributions_corrected.png`
3. `Figure_4_error_boxplot_corrected.png`
4. `Figure_7_country_improvement_hybrid_rf.png`
5. `Figure_7_country_improvement_hybrid_gb.png`
6. `Figure_9_vc_feature_importance.png`

The manuscript workflow figure,

`Malaria_Transmission_Modelling_Workflow.png`

is a conceptual representation of the analytical framework and is not
generated from model predictions.

Numerical data underlying the empirical figures are also retained in the
pipeline outputs to facilitate independent verification.

---

## Reproducibility Checks

The pipeline includes manuscript-alignment checks comparing reproduced
central results against the values reported in the manuscript.

If an alignment check fails, the discrepancy should be investigated rather
than modifying the reproduced output or manuscript value without
verification.

The pipeline also preserves the manuscript definitions of the nonlinear
environmental variables, including:

```text
Rainfall_sq = Rainfall_total^2
Temp_Rain   = Temp_mean * Rainfall_total
```

---

## Interpretation

The repository implements a predictive modelling framework rather than a
mechanistic simulation of malaria transmission.

In particular:

- the environmentally parameterized VC representation is not a direct
  entomological measurement;
- predictive feature importance should not be interpreted as causal
  importance;
- controlled feature ablation provides the primary assessment of the
  incremental contribution of VC-derived information; and
- results should be interpreted in the context of the country-level annual
  resolution and limited temporal coverage of the analytical dataset.

---

## Reproducibility and Research Use

This repository is provided to support reproducibility, methodological
inspection, and further research on biologically informed machine learning
for malaria prediction.

When using the data or code, users should also acknowledge and comply with
the citation and licensing requirements of the original data providers,
including WHO Global Health Observatory, Landsat, CHIRPS, WorldPop, and
Google Earth Engine.

---

## Citation

If you use this repository, please cite the associated manuscript and the
archived repository release.

### Manuscript

> EWLI, K. G. D., Karanja, M., Agboka, K. M., Barreaux, A.,
> Oluwabukonla, F. S., and WUIGA, S. A.
> *Biologically Informed Machine Learning Framework for Malaria Incidence
> Prediction Using Vectorial-Capacity-Derived Information.*

**Publication details will be added following publication.**

### Software and data

Find the Zenodo DOI below.

```text
DOI: 10.5281/zenodo.22833571
```


---

## Authors

**Kodjo Gato Didier EWLI**  
Pan African University Institute for Basic Sciences, Technology and
Innovation (PAUSTI), Nairobi, Kenya

**Mwangi Karanja**  
Jomo Kenyatta University of Agriculture and Technology (JKUAT),
Nairobi, Kenya

**Komi Mensah Agboka**  
International Centre of Insect Physiology and Ecology (icipe),
Nairobi, Kenya

**Antoine Barreaux**  
International Centre of Insect Physiology and Ecology (icipe),
Nairobi, Kenya

**Folorunso Sakinat Oluwabukonla**  
Olabisi Onabanjo University, Ago-Iwoye, Nigeria

**Selina Akouwa WUIGA**  
Dibrugarh University, Assam, India

---

## Funding

This research received no specific grant from any funding agency in the
public, commercial, or not-for-profit sectors. The study was conducted as
part of the academic requirements for the Master's programme of the
corresponding author.

---

## Competing Interests

The authors declare that they have no competing interests.

---

## License


The original software and source code developed for this project are
released under the MIT License. See the `LICENSE` file for the full
license terms.

The processed analytical dataset included in this repository was derived
from publicly available third-party data sources, including the World
Health Organization Global Health Observatory, Landsat 8, CHIRPS, and
WorldPop. These source datasets remain subject to the terms, conditions,
and attribution requirements of their respective data providers.

The MIT License applies to the original software developed for this
repository and does not supersede or replace the licensing terms of the
underlying third-party data sources.
