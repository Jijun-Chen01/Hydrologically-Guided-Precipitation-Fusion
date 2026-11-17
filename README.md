# Hydrologically Guided Differentiable Multisource Precipitation Fusion

Research code associated with the manuscript:

**Chen, J., Liu, J., Guo, J., Xiong, J., Liu, C., and Gu, J.**  
*Hydrologically Guided Differentiable Multisource Precipitation Fusion for Hydrological Modeling.*  
Submitted to **Journal of Hydrology** (manuscript HYDROL89413; submitted November 2026).

This repository contains a semi-distributed differentiable hydrological modeling framework developed for the Jialing River Basin, China. It jointly learns **dynamic multisource precipitation fusion weights**, **spatially regionalized HBV parameters**, and **river-reach unit hydrographs** with PyTorch.

> **Data availability:** Observed streamflow and restricted spatial datasets are not distributed. The included example uses simulated inputs to illustrate the model workflow; it is not a substitute for the Jialing River observations.

## 1. Model overview

1. **Dynamic precipitation fusion:** A single-hidden-layer MLP learns nonnegative, sum-to-one weights for daily CHMPRE, ERA5-Land, GSMaP, and IMERG precipitation from meteorological inputs and 24 standardized subbasin attributes.
2. **Runoff generation:** A regionalization MLP predicts 10 bounded HBV parameters and 2 Gamma-UH parameters with snow, soil, and linear-reservoir components. Local runoff is routed with a 24-day Gamma unit hydrograph.
3. **River routing:** A third MLP maps nine channel attributes to a 30-day softmax-normalized unit hydrograph. Reach flows are accumulated using the upstream–downstream topology.
4. **Joint optimization:** Multi-gauge flow losses and SWE, soil moisture and actual evapotranspiration process constraints are differentiated through the coupled model. The supplied notebooks additionally employ a monthly runoff-centroid index (RCI) regularizer.
5. **PUB experiments:** Selected gauging stations are removed from the streamflow calibration and model-selection losses. The same parameter networks and process constraints remain in use throughout the basin.

The study uses **82 subbasins**, **21 gauging stations**, and **five held-out streamflow stations** for the PUB evaluation. Daily forcings cover 2007–2020 (2007 warm-up, 2008–2016 training, 2017–2020 validation). See `docs/experiments.md` for implementation details and limitations.

## 2. Repository layout

```text
Hydrologically-Guided-Precipitation-Fusion/
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
├── src/
│   ├── hbv.py           # HBV physical states and runoff generation
│   ├── fusion.py        # Four-source fusion and static parameter MLPs
│   ├── routing.py       # Gamma UH and learnable RiverUH routing
│   ├── losses.py        # Flow, SWE, SM, ET and RCI losses
│   ├── data.py          # Validated model array loading
│   ├── rci.py           # Gauge-increment setup for RCI
│   └── train.py         # Gauged/PUB training entry point
├── configs/
│   ├── gauged.yaml
│   └── pub.yaml
├── scripts/
│   └── prepare_data.py # Private local preprocessing from original paths
├── examples/
│   ├── demo.ipynb
│   ├── make_synthetic_data.py
│   └── synthetic_data/tiny_demo.npz
├── docs/
│   ├── data_schema.md
│   └── experiments.md
└── tests/
    └── test_smoke.py
```

The framework is organized into reusable modules for model components, data preparation, calibration, evaluation, and PUB experiments. The Jialing River observations are required to reproduce the manuscript results.

## 3. Install

Create an isolated Python environment at the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate       # macOS/Linux
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` lists the packages used in this repository without pinning installation versions. For reproducible computing environments, record the installed versions after installation and testing. The default training device is CPU.

## 4. Run the example

From the repository root:

```bash
python examples/make_synthetic_data.py
python -m unittest discover -s tests -v
```

Open `examples/demo.ipynb` from the repository root using Jupyter or VS Code. It loads simulated example arrays, executes the coupled model, and checks that precipitation weights sum to one. The example does not contain observed streamflow.

## 5. Run with your own or authorized study data

1. Obtain data from the original providers and confirm that you are permitted to use them. Do not commit private files.
2. Arrange the raw CSV and shapefile inputs as shown in `docs/data_schema.md`, under local `Data/`.
3. From the repository root, create the prepared array file locally:

```bash
python scripts/prepare_data.py
```

This writes `Data/processed/model_inputs.npz`, which includes observed streamflow and must remain private. The preparation pipeline expects the fields and spatial reference information specified in `docs/data_schema.md`; verify the topology and date alignment when preparing a new dataset.

To fit the precipitation-fusion model with streamflow gauges:

```bash
python -m src.train --config configs/gauged.yaml
```

To fit with the five PUB gauges held out from streamflow losses:

```bash
python -m src.train --config configs/pub.yaml
```

To fit the single-precipitation baseline (for example CHMPRE):

```bash
python -m src.train --config configs/gauged.yaml --model single --source CHMPRE
```

Other baseline sources: `ERA5Land`, `GSMaP`, and `IMERG`. Choose a different `output_dir` for each run to avoid overwriting saved checkpoints. **Jialing River observations, fitted model checkpoints, and study outputs are not included.**

## 6. Gauged vs. PUB calibration

The PUB configuration holds out the streamflow observations at **zhouqu, lueyang, fujiangqiao, fengtan, and jinxi** when calculating training and validation objectives. Internal-process reference data (SWE/SM/ET) are still used over the study area, as in the original experiments. Thus, PUB denotes **ungauged with respect to local streamflow calibration**, not absence of all gridded reference information.

See `docs/experiments.md` for the objective function, exact periods and implementation notes.

## 7. Availability of data and code

Meteorological and gridded hydrological datasets are documented in the manuscript and `docs/data_schema.md`. Streamflow observations were derived from Chinese hydrological yearbooks and are subject to data-sharing restrictions. Their source files and derivatives are excluded from this public release. Use the documented schema to supply legally obtained data or alternative open data.

Releasing the method implementation does **not** imply that third-party proprietary observations, station GIS files or derived private model arrays may be redistributed. The `.gitignore` prevents ordinary accidental inclusion but does not sanitize already committed Git history; check the files and repository history before publishing.

## 8. Citation and license

Please cite the manuscript above when using the research framework. A DOI and final publication metadata will be added after publication, if available. The source code is distributed under the MIT license. External input products retain their own terms and licenses.

## 9. Implementation settings

Parameter bounds, loss terms, time periods, and PUB station exclusions are recorded in `docs/experiments.md`. The released model bounds for `T1` and `DD` are **[-2, 2] °C** and **[0, 2]**, respectively. The implemented objective also includes an RCI term. These settings should be considered when comparing code-based experiments with the manuscript.
