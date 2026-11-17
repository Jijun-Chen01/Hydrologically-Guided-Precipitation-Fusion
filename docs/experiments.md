# Experimental configurations

## Study design

The submitted manuscript (HYDROL89413) reports the daily semi-distributed model in 82 subbasins, 21 gauging stations, 2007 warm-up, 2008–2016 training, 2017–2020 validation and five streamflow-held-out PUB stations. Three single-hidden-layer networks use 128/128/64 neurons for precipitation fusion, HBV parameter prediction, and river routing. Adam learning rate = 0.001; maximum 1000 epochs; early stopping.

The research experiments include:

| Experiment source | Model | Flow calibration |
| --- | --- | --- |
| `Sta_HBV.ipynb` | Single-product HBV + Gamma + RiverUH | All 21 gauges |
| `Sta_HBV_P(3).ipynb` | Four-product MLP fusion + HBV + Gamma + RiverUH | All 21 gauges |
| `PUB_Sta_HBV.ipynb` | Single-product HBV + Gamma + RiverUH | Excludes 5 PUB gauges |
| `PUB_Sta_HBV_P.ipynb` | Four-product MLP fusion + HBV + Gamma + RiverUH | Excludes 5 PUB gauges |

Shared model components are configured for single-product and four-product simulations, including station-held-out PUB experiments.

## Model parameters and numerical settings

- `seed=2026`, PyTorch CPU tensors (`float32`).
- One-hidden-layer MLPs, widths 128/128/64 with LayerNorm + SiLU, not plain ReLU.
- Fusion initialized to favor CHMPRE through a bias of 3.0 in the last linear layer; missing products are masked and weights renormalized; all-missing input falls back to CHMPRE (with NaNs replaced by zero).
- Twelve learned HBV + Gamma parameters; soil capacity and reservoir constants are bounded by a sigmoid mapping.
- Local Gamma UH 24 days; RiverUH 30 days. Unit hydrographs normalized by softmax.
- Warm-up state follows the HBV implementation; all days in the time axis are run forward.
- `Adam(lr=0.001)`, gradient clip norm 1, `epochs=1000`, `patience=64`, best checkpoint by **calibration-station validation objective**.
- Train interval `[2008-01-01, 2017-01-01)` and validation `[2017-01-01, 2021-01-01)`.

### Training objective

`L = 0.70 * L_flow + 0.20 * L_process + 0.10 * L_RCI`

`L_flow` combines normalized flow RMSE (0.75) and square-root flow RMSE (0.25). `L_process` averages basin losses for SWE (normalized RMSE), SM (standardized RMSE), and ET (normalized RMSE). `L_RCI` uses a KGE-style monthly runoff-centroid term. The corresponding loss components are implemented in `src/losses.py` and `src/rci.py`.

**Manuscript comparison:** Section 2.3 describes streamflow and process constraints; the released training configuration additionally includes an RCI term with weight 0.10. This setting should be considered when interpreting and comparing experiments.

### Parameter bounds

| Parameter | Manuscript Table 1 | Released code |
| --- | --- | --- |
| `T1` (°C) | [-3, 3] | [-2, 2] |
| `DD` (mm/°C/day) | [0, 3] | [0, 2] |

`src/fusion.py` uses the released-code bounds shown above. Changing them defines a different model configuration and requires recalibration.

## PUB streamflow exclusions

- `zhouqu` (Topo 43)
- `lueyang` (Topo 54)
- `fujiangqiao` (Topo 57)
- `fengtan` (Topo 70)
- `jinxi` (Topo 77)

In the PUB configuration, only calibration-station discharge enters `station_loss` and the RCI loss. SWE, SM and ET constraints are applied across the full model domain. `src/rci.py` constructs the incremental station matrix for the calibration subset. All PUB observations are used only in the held-out reporting branch after model fitting.

## Model checks and data handling

- The example tests exercise precipitation fusion, HBV runoff generation, local Gamma routing, river routing, and gradient propagation.
- The preprocessing workflow reads locally held CSV and GIS inputs and writes a model-ready NPZ archive. Input column names, geometry, and temporal alignment are described in `docs/data_schema.md`.
- `T_monthly` is present as the fourth forcing channel for compatibility with the input preparation workflow; the HBV calculation uses precipitation, PET, and temperature.
- The precipitation preparation uses inner joins on the original daily input tables. Products missing an entire date can therefore shorten the common data period.
- The streamflow observation mask uses finite-value checks. Check discharge quality-control flags and missing-value conventions before new experiments.
- Static predictor standardization uses the available subbasin attributes. The same transformation should be applied consistently to new spatial units.

## Packaging scope

This repository provides the model implementation and a simulated-input example. Study observations, processed field data, shapefiles, interactive maps, and fitted weights are not distributed. Access to restricted datasets remains subject to the original data providers.
