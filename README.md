# Hydrologically Guided Differentiable Multisource Precipitation Fusion

Research code associated with the manuscript:

**Chen, J., Liu, J., Guo, J., Xiong, J., Liu, C., and Gu, J.**  
*Hydrologically Guided Differentiable Multisource Precipitation Fusion for Hydrological Modeling.*  
Submitted to **Journal of Hydrology** (manuscript HYDROL89413; submitted November 2026).

This repository contains a semi-distributed differentiable hydrological modeling framework developed for the Jialing River Basin, China. It jointly learns **dynamic multisource precipitation fusion weights**, **spatially regionalized HBV parameters**, and **river-reach unit hydrographs** with PyTorch.

> **Data restrictions:** Observed streamflow and private GIS inputs are **not distributed**. The bundled artificial example is **not** a reconstruction of the Jialing River dataset and does not reproduce the manuscript's numerical results.

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

The original four exploratory notebooks have been consolidated into shared modules. The scientific kernels, network shapes, parameter bounds and loss definitions are based on the uploaded notebooks. Data preparation and training have been reorganized; **equivalence to all manuscript results has not been verified against the unavailable restricted inputs**.

## 3. Install

Python **3.10+** is recommended. Create an isolated environment at the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate       # macOS/Linux
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The dependency ranges in `requirements.txt` are **inferred from import statements**, not a record of the authors' original environment. For a frozen reproducible release, record an exact lockfile after testing on the intended platform. Model execution currently defaults to CPU, consistent with the supplied notebooks.

## 4. Run the included artificial example

From the repository root:

```bash
python examples/make_synthetic_data.py
python -m unittest discover -s tests -v
```

Open `examples/demo.ipynb` from the repository root using Jupyter or VS Code. It loads only the supplied artificial arrays, executes an end-to-end forward pass and checks that learned precipitation weights sum to one. **The example is methodological, not a scientific validation case.**

## 5. Run with your own or authorized study data

1. Obtain data from the original providers and confirm that you are permitted to use them. Do not commit private files.
2. Arrange the raw CSV and shapefile inputs as shown in `docs/data_schema.md`, under local `Data/`.
3. From the repository root, create the prepared array file locally:

```bash
python scripts/prepare_data.py
```

This writes `Data/processed/model_inputs.npz` (which includes observed flow, and **must remain private**). The preprocessing is adapted from the research notebooks and relies on their actual shapefile fields, CRS and station overrides. Check the resulting Topo/date alignment before fitting; this raw-data pathway has not been exercised with the restricted Jialing River dataset in this release.

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

Other baseline sources: `ERA5Land`, `GSMaP`, and `IMERG`. Choose a different `output_dir` for each run to avoid overwriting saved checkpoints. **No original Jialing River observations, model checkpoints or paper outputs are included.**

## 6. Gauged vs. PUB calibration

The PUB configuration holds out the streamflow observations at **zhouqu, lueyang, fujiangqiao, fengtan, and jinxi** when calculating training and validation objectives. Internal-process reference data (SWE/SM/ET) are still used over the study area, as in the original experiments. Thus, PUB denotes **ungauged with respect to local streamflow calibration**, not absence of all gridded reference information.

See `docs/experiments.md` for the objective function, exact periods and implementation notes.

## 7. Availability of data and code

Meteorological and gridded hydrological datasets are documented in the manuscript and `docs/data_schema.md`. Streamflow observations were derived from Chinese hydrological yearbooks and are subject to data-sharing restrictions. Their source files and derivatives are excluded from this public release. Use the documented schema to supply legally obtained data or alternative open data.

Releasing the method implementation does **not** imply that third-party proprietary observations, station GIS files or derived private model arrays may be redistributed. The `.gitignore` prevents ordinary accidental inclusion but does not sanitize already committed Git history; check the files and repository history before publishing.

## 8. Citation and license

Please cite the manuscript above when using the research framework. A DOI and final publication metadata will be added after publication, if available. The source code is distributed under the MIT license, **subject to approval by the rights holders and co-authors before public release**. External input products retain their own terms and licenses.

## 9. Known implementation distinctions

The archived notebooks set the model's `T1` parameter bounds to **[-2, 2] °C** and its `DD` bounds to **[0, 2]**, whereas Table 1 of the submitted manuscript reports **[-3, 3]** and **[0, 3]**. The code preserves the notebooks' actual values. The training notebooks also include the RCI objective, which should be checked against the manuscript's description of the training loss. See `docs/experiments.md` before making any scientific claims of identical reproduction.
