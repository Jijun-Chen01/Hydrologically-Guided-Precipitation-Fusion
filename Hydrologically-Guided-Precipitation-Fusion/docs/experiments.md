# Experimental configurations and release notes

## Published-study design vs archived code

The submitted manuscript (HYDROL89413) reports the daily semi-distributed model in 82 subbasins, 21 gauging stations, 2007 warm-up, 2008–2016 training, 2017–2020 validation and five streamflow-held-out PUB stations. Three single-hidden-layer networks use 128/128/64 neurons for precipitation fusion, HBV parameter prediction, and river routing. Adam learning rate = 0.001; maximum 1000 epochs; early stopping.

The four archived Jupyter notebooks are:

| Original notebook | Model | Flow calibration |
| --- | --- | --- |
| `Sta_HBV.ipynb` | Single-product HBV + Gamma + RiverUH | All 21 gauges |
| `Sta_HBV_P(3).ipynb` | Four-product MLP fusion + HBV + Gamma + RiverUH | All 21 gauges |
| `PUB_Sta_HBV.ipynb` | Single-product HBV + Gamma + RiverUH | Excludes 5 PUB gauges |
| `PUB_Sta_HBV_P.ipynb` | Four-product MLP fusion + HBV + Gamma + RiverUH | Excludes 5 PUB gauges |

Instead of publishing four full notebooks (which contain private map and analysis outputs), this repository uses one shared implementation and YAML-configured model/experiment selection.

## Parameters and numerical settings retained from notebooks

- `seed=2026`, PyTorch CPU tensors (`float32`).
- One-hidden-layer MLPs, widths 128/128/64 with LayerNorm + SiLU, not plain ReLU.
- Fusion initialized to favor CHMPRE through a bias of 3.0 in the last linear layer; missing products are masked and weights renormalized; all-missing input falls back to CHMPRE (with NaNs replaced by zero).
- Twelve learned HBV + Gamma parameters; soil capacity and reservoir constants are bounded by a sigmoid mapping.
- Local Gamma UH 24 days; RiverUH 30 days. Unit hydrographs normalized by softmax.
- Warm-up state is the default of the archived HBV implementation; all days in the time axis are run forward.
- `Adam(lr=0.001)`, gradient clip norm 1, `epochs=1000`, `patience=64`, best checkpoint by **calibration-station validation objective**.
- Train interval `[2008-01-01, 2017-01-01)` and validation `[2017-01-01, 2021-01-01)`.

### Objective actually present in submitted notebooks

`L = 0.70 * L_flow + 0.20 * L_process + 0.10 * L_RCI`

`L_flow` combines normalized flow RMSE (0.75) and square-root flow RMSE (0.25). `L_process` averages basin losses for SWE (normalized RMSE), SM (standardized RMSE), and ET (normalized RMSE). `L_RCI` uses a KGE-style monthly runoff-centroid term. These functions were extracted without changing their underlying equations.

**Important discrepancy to review:** Section 2.3 of the PDF describes a streamflow-plus-process loss and does not explicitly identify the separate 10% RCI term used in the uploaded notebooks. Do not change the original loss retroactively; verify which configuration generated the submitted results and, if necessary, clarify in a revision.

### Parameter-range discrepancy

| Parameter | Submitted paper Table 1 | Notebook actually used |
| --- | --- | --- |
| `T1` (°C) | [-3, 3] | [-2, 2] |
| `DD` (mm/°C/day) | [0, 3] | [0, 2] |

The public `src/fusion.py` retains the **notebook bounds** deliberately. Do not substitute the paper ranges without re-running and documenting a changed experiment.

## PUB streamflow exclusions

- `zhouqu` (Topo 43)
- `lueyang` (Topo 54)
- `fujiangqiao` (Topo 57)
- `fengtan` (Topo 70)
- `jinxi` (Topo 77)

In the original PUB notebooks, only calibration-station discharge enters `station_loss` and the RCI loss. SWE, SM and ET constraints are applied across the full model domain. `src/rci.py` rebuilds the incremental station matrix for the calibration subset instead of wrongly slicing the all-gauge matrix. All PUB observations are used only in the held-out reporting branch after model fitting.

## What has and has not been verified

- Source code from all four notebooks has been inspected; principal TorchScript HBV, Gamma UH, RiverUH, MLP and loss functions are extracted or consolidated. An identical-seed comparison on synthetic inputs confirmed zero numerical differences in the refactored vs. original four-source model forward outputs (including hydrological states and routing outputs).
- Public synthetic smoke tests exercise both `Rain_HBV` and `Static_HBV` and gradient flow without exposing field data.
- Raw GIS preprocessing follows the provided notebook blocks but **cannot be integration-tested without their original private GIS/CSV data**.
- No claim is made that trained metrics or checkpoint files exactly match the paper after refactoring.
- The reference notebooks build a forcing fourth channel `T_monthly` which the HBV core does not access; it is retained in prepared arrays to match original input expectations.
- The original precipitation merge performs inner joins, potentially discarding dates before the masked fusion module; negative precipitation is treated as valid until clamped to zero. Those scientific input choices are kept, not silently modified.
- The original flow mask is `torch.isfinite`, so negative observed discharge is not excluded before being clipped inside `station_loss`. Audit original measurements before running real-data experiments.
- The source notebooks standardize static inputs using all available subbasins; this is retained in the private preprocessing pathway. Evaluate alternative normalization separately if strict geographic extrapolation is an objective.

## Packaging scope

This is an implementation release, with a demonstrative synthetic example. It is not a full public re-run of the proprietary-data paper. Raw data, processed observations, shapefiles, interactive maps, and trained weights have been intentionally omitted. Before release, obtain approval for the license and verify that no private content exists in Git history.
