# Input data schema (private Jialing River dataset is not included)

## 1. Relative paths used by the original notebooks

Run `python scripts/prepare_data.py` from the repository root after placing **authorized copies** in the following paths. All `Data/` paths are Git-ignored. Do not publish raw or processed content from these directories without explicit permission.

| Path | Required columns or structure | Origin / interpretation |
| --- | --- | --- |
| `Data/Forcing/P/P_CHMPRE.csv` | `Topo`, `Date`, `P` | CHMPRE daily precipitation (mm/day) |
| `Data/Forcing/P/P_ERA5Land.csv` | `Topo`, `Date`, `P` | ERA5-Land daily precipitation (mm/day) |
| `Data/Forcing/P/P_GSMaP.csv` | `Topo`, `Date`, `P` | GSMaP daily precipitation (mm/day) |
| `Data/Forcing/P/P_IMERG.csv` | `Topo`, `Date`, `P` | IMERG daily precipitation (mm/day) |
| `Data/Forcing/PET.csv` | `Topo`, `Date`, `PET` | Daily potential ET (mm/day), GLEAM |
| `Data/Forcing/T/T.csv` | `Topo`, `Date`, `T` | Mean daily temperature (°C), ERA5-Land |
| `Data/Forcing/T/T_monthly.csv` | `Topo`, `Date`, `T` | Monthly temperature input duplicated across days (unused in HBV core) |
| `Data/Process/SWE.csv` | `Topo`, `Date`, `SWE` | Snow water equivalent (mm), ERA5-Land |
| `Data/Process/SM.csv` | `Topo`, `Date`, `SM` | Soil moisture index (SMCI1.0); do not treat as mm storage |
| `Data/Process/E.csv` | `Topo`, `Date`, `E` | Actual evapotranspiration (mm/day), GLEAM |
| `Data/Flow/{station}.csv` | `date`, `qobs` | **Restricted** daily discharge (m³/s); filenames use lowercased station names |
| `Data/JLR_Subbasins_attributes.csv` | `basin_id` and attribute columns below | 24 HydroATLAS subbasin features |
| `Data/JLR_Topo/Sub_Topo.shp` | `PolygonId`, `Area`, polygon geometry, plus source-dependent fields | Hydrological subbasins; source `Area` is divided by **100** in the notebook to produce km², implying source units of hectares |
| `Data/JLR_Topo/River/Streams.shp` | `LINKNO`, `DSLINKNO`, `Slope`, `strmOrder`, polyline geometry | River topology and channels |
| `Data/JLR_Topo/River/Channels.shp` | Same basic river identifiers and channel fields | More detailed river segments and RCI channel reach lengths |
| `Data/JLR_Topo/Stations/Stations.shp` | `station`, point geometry | **Restricted** observed station locations |

`Date` should parse as a calendar day. One row per (`Topo`, `Date`) is expected. `Topo` is the model's upstream-to-downstream index, not a universal public basin identifier. Station names must match the observation filenames after conversion to lowercase. Source IDs, geometry, time spans, data units and licenses must be verified against original metadata.

### Source datasets

Manuscript Table 3 cites CHMPRE V2, ERA5-Land, GSMaP V8, IMERG V07, GLEAM V4, SMCI1.0, HydroATLAS and a 90 m HydroSHEDS DEM. Access them via the original providers and cited papers; availability or redistribution rights differ by dataset. This repository does not redistribute their original or reprocessed files.

## 2. The 24 subbasin features used by the model

The following feature order comes from `Sta_HBV_P(3).ipynb`'s parameter preparation cell. Two columns are derived as max-minus-min differences:

```python
hbv_features = [
    'area', 'ele_mt_sav', 'elevation_range_m', 'slp_dg_sav',
    'snd_pc_sav', 'slt_pc_sav', 'cly_pc_sav', 'soc_th_sav',
    'swc_pc_s01', 'swc_pc_s04', 'swc_pc_s09', 'gwt_cm_sav',
    'kar_pc_sse', 'for_pc_sse', 'crp_pc_sse', 'urb_pc_sse',
    'wet_pc_sg1', 'pre_mm_syr', 'pet_mm_syr', 'tmp_dc_syr',
    'temperature_range_c', 'snw_pc_syr', 'ari_ix_sav', 'cmi_ix_syr',
]
# elevation_range_m = ele_mt_smx - ele_mt_smn
# temperature_range_c = tmp_dc_smx - tmp_dc_smn
```

## 3. Nine river-routing attributes

In order, these are `source_contrib_area`, `target_contrib_area`, `route_length_km`, `route_slope`, `route_sinuosity`, `source_order`, `target_order`, `source_slope`, `target_slope`. They are calculated from the private river shapefiles and topological relationships by `scripts/prepare_data.py`.

These 24 + 9 attributes are standardized using `sklearn.preprocessing.StandardScaler`. The original feature preparation fills missing numeric entries with median values, then fits the scaler over all available subbasins.

## 4. Model-ready local NPZ archive

The preparation script writes `Data/processed/model_inputs.npz` from local authorized data. It contains the following arrays. This file **contains derived restricted observations and must not be committed**.

| Array | Shape | Description |
| --- | --- | --- |
| `dates` | `[T]` | ISO dates `YYYY-MM-DD`, full daily axis |
| `topo_ids` | `[B]` | Subbasin Topo numbers, consistent across inputs |
| `station_names` | `[S]` | Lowercased station names |
| `station_topos` | `[S]` | Gauged subbasin Topo IDs |
| `forcing` | `[B,T,4]` | CHMPRE, PET, T, T_monthly (last is unused by model core) |
| `x_rain` | `[B,T,4]` | CHMPRE, ERA5Land, GSMaP, IMERG in this order |
| `x_hbv` | `[B,24]` | Standardized static subbasin features |
| `x_river` | `[B,9]` | Standardized static river features |
| `area` | `[B]` | Individual subbasin area (km²), for mm/day-to-m³/s conversion |
| `downstream_index` | `[B]` | Zero-based downstream row index or `-1` (outlet) |
| `routing_order` | `[B]` | Zero-based order from upstream to downstream |
| `station_indices` | `[S]` | Zero-based row index of the gauged subbasin |
| `qobs_station` | `[S,T]` | **Restricted** observed discharge, m³/s |
| `swe_obs`, `sm_obs`, `e_obs` | `[B,T]` | Process reference variables |
| `rci_sub_position` | `[B]` | Relative RCI location derived from projected private river geometry |
| `rci_sub_area` | `[B]` | Subbasin polygon area used for RCI location weighting |

`B`: number of subbasins; `S`: number of stations; `T`: number of days. The artificial `examples/synthetic_data/tiny_demo.npz` follows the API but has only four mock subbasins, two invented gauges and a short non-study time range. It does **not** contain the RCI geometry arrays, because the artificial demo uses no RCI objective.

## 5. Data-handling limitations of original scripts

- The original data preparation inner-merges four rainfall products plus PET and T; a missing entire date/Topo row can disappear before the masked MLP receives it. Keep this behavior in mind when comparing experiments and perform a separate audit before modifying it.
- The network masks `isfinite` rainfall but clamps negative finite values to zero. This behavior is reproduced, not silently changed.
- Streamflow observations have the mask `isfinite` in the original notebook; independently audit negative values before publication-quality reruns.
- Source `Area` is assumed to be in hectares when converted by `/100`. Check source metadata rather than guessing other area units.
- The notebook uses explicit station-to-Topo overrides for `pingwu`, `bazhong` and `luoduxi`. These are kept only in the private preprocessing script.
- The saved NPZ arrays are model inputs, not public open data. Do not use them as synthetic examples.
