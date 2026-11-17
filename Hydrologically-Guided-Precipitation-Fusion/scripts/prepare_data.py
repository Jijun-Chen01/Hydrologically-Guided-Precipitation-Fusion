# =====================================================
# Prepare private Jialing River arrays from original relative paths
# =====================================================
# Run locally only; DO NOT commit data, shapefiles or model_inputs.npz.
# The blocks below retain the original notebook preprocessing algorithms.
# =====================================================
# Read HBV forcing and multi-source precipitation
# =====================================================
from pathlib import Path

import numpy as np
import pandas as pd


# Paths
forcing_dir = Path("Data/Forcing")
p_dir = forcing_dir / "P"

p_files = {
    "P_CHMPRE": p_dir / "P_CHMPRE.csv",
    "P_ERA5Land": p_dir / "P_ERA5Land.csv",
    "P_GSMaP": p_dir / "P_GSMaP.csv",
    "P_IMERG": p_dir / "P_IMERG.csv",
}

pet_file = forcing_dir / "PET.csv"
t_file = forcing_dir / "T" / "T.csv"
t_monthly_file = forcing_dir / "T" / "T_monthly.csv"


# Precipitation
p_data = []

for name, file in p_files.items():
    df = pd.read_csv(file, parse_dates=["Date"])
    df = df.rename(columns={"P": name})
    p_data.append(df[["Topo", "Date", name]])

p_df = p_data[0]

for df in p_data[1:]:
    p_df = p_df.merge(df, on=["Topo", "Date"])


# Other forcing
pet_df = pd.read_csv(pet_file, parse_dates=["Date"])
t_df = pd.read_csv(t_file, parse_dates=["Date"])
t_monthly_df = pd.read_csv(t_monthly_file, parse_dates=["Date"])

t_monthly_df = t_monthly_df.rename(columns={"T": "T_monthly"})
t_monthly_df["Month"] = t_monthly_df["Date"].dt.to_period("M")


# Merge forcing
forcing_df = (
    p_df
    .merge(pet_df, on=["Topo", "Date"])
    .merge(t_df, on=["Topo", "Date"])
    .sort_values(["Topo", "Date"])
    .reset_index(drop=True)
)

forcing_df["Month"] = forcing_df["Date"].dt.to_period("M")

forcing_df = (
    forcing_df
    .merge(
        t_monthly_df[["Topo", "Month", "T_monthly"]],
        on=["Topo", "Month"],
    )
    .drop(columns="Month")
)


# Index
topo_ids = np.sort(forcing_df["Topo"].unique())
dates = np.sort(forcing_df["Date"].unique())


# Tensor function
def build_tensor(df, variables):
    return np.stack([
        df.pivot(index="Topo", columns="Date", values=variable)
        .reindex(index=topo_ids, columns=dates)
        .to_numpy(dtype=np.float32)
        for variable in variables
    ], axis=-1)


# HBV forcing
forcing_variables = ["P_CHMPRE", "PET", "T", "T_monthly"]
forcing = build_tensor(forcing_df, forcing_variables)


# Rainfall correction inputs
rain_variables = [
    "P_CHMPRE",
    "P_ERA5Land",
    "P_GSMaP",
    "P_IMERG",
]

x_rain = build_tensor(forcing_df, rain_variables)


# Information
print(
    f"Forcing: 0:P_CHMPRE | 1:PET | 2:T | 3:T_monthly\n"
    f"Rain: 0:CHMPRE | 1:ERA5Land | 2:GSMaP | 3:IMERG\n"
    f"Basins: {len(topo_ids)} | "
    f"Period: {str(dates[0])[:10]} ~ {str(dates[-1])[:10]}\n"
    f"Forcing tensor: {forcing.shape}\n"
    f"Rain tensor: {x_rain.shape}"
)
forcing_topo_ids = np.asarray(topo_ids).copy()
forcing_dates = np.asarray(dates).copy()


# =====================================================
# Load hydrological process variables
# =====================================================
from pathlib import Path

import numpy as np
import pandas as pd
import torch


# Settings
process_dir = Path("Data/Process")
topo_order = np.asarray(topo_ids, dtype=int)
model_dates = pd.DatetimeIndex(dates)

device = torch.device("cpu")
dtype = torch.float32


# Load data
swe_df = pd.read_csv(process_dir / "SWE.csv", parse_dates=["Date"])
sm_df = pd.read_csv(process_dir / "SM.csv", parse_dates=["Date"])
e_df = pd.read_csv(process_dir / "E.csv", parse_dates=["Date"])


# Process daily matrices
def build_matrix(df, variable):
    return (
        df.pivot(index="Topo", columns="Date", values=variable)
        .reindex(index=topo_order, columns=model_dates)
        .to_numpy(dtype=np.float32)
    )


swe_matrix = build_matrix(swe_df, "SWE")
sm_matrix = build_matrix(sm_df, "SM")
e_matrix = build_matrix(e_df, "E")


# Torch observations
swe_obs = torch.as_tensor(swe_matrix, dtype=dtype, device=device)
sm_obs = torch.as_tensor(sm_matrix, dtype=dtype, device=device)
e_obs = torch.as_tensor(e_matrix, dtype=dtype, device=device)


# Observation masks
swe_obs_mask = torch.isfinite(swe_obs)
sm_obs_mask = torch.isfinite(sm_obs)
e_obs_mask = torch.isfinite(e_obs)


# Information
print(
    f"SWE daily: {swe_obs.shape} | "
    f"{model_dates[0].date()} ~ {model_dates[-1].date()} | "
    f"Valid: {swe_obs_mask.sum().item()}\n"
    f"SM daily:  {sm_obs.shape} | "
    f"{model_dates[0].date()} ~ {model_dates[-1].date()} | "
    f"Valid: {sm_obs_mask.sum().item()}\n"
    f"E daily:   {e_obs.shape} | "
    f"{model_dates[0].date()} ~ {model_dates[-1].date()} | "
    f"Valid: {e_obs_mask.sum().item()}"
)

# =====================================================
# Prepare JLR topology and station routing
# =====================================================
import warnings
from collections import defaultdict
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import torch


warnings.filterwarnings(
    "ignore",
    message="urllib3 v2 only supports OpenSSL 1.1.1+.*",
)


# Files
data_dir = Path("Data/JLR_Topo")

station_file = data_dir / "Stations/Stations.shp"
subbasin_file = data_dir / "Sub_Topo.shp"
stream_file = data_dir / "River/Streams.shp"


# Settings
device = torch.device("cpu")
dtype = torch.float32

station_topo_override = {
    "pingwu": 38,
    "bazhong": 32,
    "luoduxi": 74,
}


# Read
sub = gpd.read_file(subbasin_file)
streams = gpd.read_file(stream_file).to_crs(sub.crs)
stations = gpd.read_file(station_file).to_crs(sub.crs)


# Match streams to subbasins
stream_points = streams[["LINKNO", "DSLINKNO", "geometry"]].copy()
stream_points["geometry"] = stream_points.geometry.interpolate(
    0.5,
    normalized=True,
)

stream_sub = gpd.sjoin(
    stream_points,
    sub[["PolygonId", "geometry"]],
    how="left",
    predicate="within",
).drop(columns="index_right")


# Match boundary streams
missing_mask = stream_sub["PolygonId"].isna()

if missing_mask.any():
    nearest = gpd.sjoin_nearest(
        stream_points.loc[missing_mask],
        sub[["PolygonId", "geometry"]],
        how="left",
    )

    nearest = nearest.loc[
        ~nearest.index.duplicated(keep="first")
    ]

    stream_sub.loc[
        missing_mask,
        "PolygonId",
    ] = nearest["PolygonId"]


stream_sub["LINKNO"] = stream_sub["LINKNO"].astype(int)
stream_sub["DSLINKNO"] = (
    stream_sub["DSLINKNO"]
    .fillna(-1)
    .astype(int)
)
stream_sub["PolygonId"] = stream_sub["PolygonId"].astype(int)


# ID mappings
link_to_polygon = dict(
    zip(
        stream_sub["LINKNO"],
        stream_sub["PolygonId"],
    )
)

polygon_to_link = dict(
    zip(
        stream_sub["PolygonId"],
        stream_sub["LINKNO"],
    )
)


# River-network edges
edges = set()

for row in stream_sub.itertuples():
    if row.DSLINKNO in link_to_polygon:
        upstream_polygon = row.PolygonId
        downstream_polygon = link_to_polygon[row.DSLINKNO]

        if upstream_polygon != downstream_polygon:
            edges.add(
                (
                    upstream_polygon,
                    downstream_polygon,
                )
            )


# Upstream-downstream dictionaries
nodes = sorted(
    sub["PolygonId"]
    .astype(int)
    .unique()
)

downstream_dict = defaultdict(list)
upstream_dict = defaultdict(list)
indegree = {node: 0 for node in nodes}

for upstream, downstream in edges:
    downstream_dict[upstream].append(downstream)
    upstream_dict[downstream].append(upstream)
    indegree[downstream] += 1

for node in nodes:
    downstream_dict[node] = sorted(
        set(downstream_dict[node])
    )

    upstream_dict[node] = sorted(
        set(upstream_dict[node])
    )


# Topological sorting
available = sorted(
    node for node in nodes
    if indegree[node] == 0
)

topological_order = []

while available:
    current = available.pop(0)
    topological_order.append(current)

    for downstream in downstream_dict[current]:
        indegree[downstream] -= 1

        if indegree[downstream] == 0:
            available.append(downstream)
            available.sort()


# Topo numbering
topo_mapping = {
    polygon_id: topo
    for topo, polygon_id in enumerate(
        topological_order,
        start=1,
    )
}


# Routing table
routing_records = []

for polygon_id in topological_order:
    downstream_polygons = downstream_dict[polygon_id]

    downstream_polygon = (
        downstream_polygons[0]
        if downstream_polygons
        else -1
    )

    downstream_topo = (
        topo_mapping[downstream_polygon]
        if downstream_polygon >= 0
        else -1
    )

    upstream_topos = sorted(
        topo_mapping[upstream_polygon]
        for upstream_polygon
        in upstream_dict[polygon_id]
    )

    routing_records.append({
        "Topo": topo_mapping[polygon_id],
        "PolygonId": polygon_id,
        "LINKNO": polygon_to_link[polygon_id],
        "DownTopo": downstream_topo,
        "DownPolygonId": downstream_polygon,
        "UpTopo": upstream_topos,
        "UpstreamCount": len(upstream_topos),
    })

routing_table = pd.DataFrame(routing_records)


# Add routing fields
routing_by_polygon = routing_table.set_index(
    "PolygonId"
)

sub_topo = sub.drop(
    columns=[
        "Topo",
        "DownTopo",
        "LINKNO",
        "DownPoly",
    ],
    errors="ignore",
).copy()

sub_topo["Topo"] = (
    sub_topo["PolygonId"]
    .map(routing_by_polygon["Topo"])
    .astype(int)
)

sub_topo["DownTopo"] = (
    sub_topo["PolygonId"]
    .map(routing_by_polygon["DownTopo"])
    .astype(int)
)

sub_topo["LINKNO"] = (
    sub_topo["PolygonId"]
    .map(routing_by_polygon["LINKNO"])
    .astype(int)
)

sub_topo["DownPoly"] = (
    sub_topo["PolygonId"]
    .map(routing_by_polygon["DownPolygonId"])
    .astype(int)
)

sub_topo["Area_km2"] = sub_topo["Area"] / 100.0

sub_topo = (
    sub_topo[
        [
            "PolygonId",
            "Topo",
            "DownTopo",
            "LINKNO",
            "DownPoly",
            "Area",
            "Area_km2",
            "geometry",
        ]
    ]
    .sort_values("Topo")
    .reset_index(drop=True)
)


# Match stations to subbasins
station_topo = gpd.sjoin(
    stations,
    sub_topo[
        [
            "PolygonId",
            "Topo",
            "DownTopo",
            "Area",
            "Area_km2",
            "geometry",
        ]
    ],
    how="left",
    predicate="within",
).drop(columns="index_right")


# Match boundary stations
missing_mask = station_topo["Topo"].isna()

if missing_mask.any():
    nearest = gpd.sjoin_nearest(
        stations.loc[missing_mask],
        sub_topo[
            [
                "PolygonId",
                "Topo",
                "DownTopo",
                "Area",
                "Area_km2",
                "geometry",
            ]
        ],
        how="left",
    )

    nearest = nearest.loc[
        ~nearest.index.duplicated(keep="first")
    ]

    for column in [
        "PolygonId",
        "Topo",
        "DownTopo",
        "Area",
        "Area_km2",
    ]:
        station_topo.loc[
            missing_mask,
            column,
        ] = nearest[column]


station_topo["station"] = (
    station_topo["station"]
    .astype(str)
    .str.strip()
    .str.lower()
)


# Correct confirmed station outlets
topo_lookup = sub_topo.set_index("Topo")

for station_name, station_topo_id in station_topo_override.items():
    mask = station_topo["station"] == station_name
    corrected = topo_lookup.loc[station_topo_id]

    station_topo.loc[mask, "Topo"] = station_topo_id
    station_topo.loc[mask, "PolygonId"] = corrected["PolygonId"]
    station_topo.loc[mask, "DownTopo"] = corrected["DownTopo"]
    station_topo.loc[mask, "Area"] = corrected["Area"]
    station_topo.loc[mask, "Area_km2"] = corrected["Area_km2"]


station_topo["PolygonId"] = station_topo["PolygonId"].astype(int)
station_topo["Topo"] = station_topo["Topo"].astype(int)
station_topo["DownTopo"] = station_topo["DownTopo"].astype(int)


# Station routing table
station_routing_table = (
    station_topo[
        [
            "station",
            "Topo",
            "PolygonId",
            "DownTopo",
            "Area_km2",
        ]
    ]
    .sort_values("Topo")
    .reset_index(drop=True)
)


# HBV routing arrays
topo_ids = sub_topo["Topo"].to_numpy(
    dtype=np.int64
)

downstream_topo = sub_topo["DownTopo"].to_numpy(
    dtype=np.int64
)

topo_to_index = {
    topo: i
    for i, topo in enumerate(topo_ids)
}

index_to_topo = {
    i: topo
    for i, topo in enumerate(topo_ids)
}

downstream_index = [
    topo_to_index[down_topo]
    if down_topo >= 0
    else -1
    for down_topo in downstream_topo
]

upstream_indices = [
    []
    for _ in topo_ids
]

for upstream_i, downstream_i in enumerate(
    downstream_index
):
    if downstream_i >= 0:
        upstream_indices[
            downstream_i
        ].append(upstream_i)

routing_order = list(
    range(len(topo_ids))
)


# Station indices
station_names = (
    station_routing_table["station"]
    .tolist()
)

station_topos = (
    station_routing_table["Topo"]
    .tolist()
)

station_indices = torch.tensor(
    [
        topo_to_index[topo]
        for topo in station_topos
    ],
    dtype=torch.long,
    device=device,
)


# HBV subbasin area
area_tensor = torch.tensor(
    sub_topo["Area_km2"].to_numpy(
        dtype=np.float32
    ),
    dtype=dtype,
    device=device,
)


# Information
print(
    f"Basins: {len(topo_ids)} | "
    f"Area: {area_tensor.min().item():.2f} ~ {area_tensor.max().item():.2f} km²\n"
    f"Stations: {len(station_names)} | "
    f"Station Topos: {station_topos}"
)

# =====================================================
# Prepare compact HBV-Gamma and RiverUH attributes
# =====================================================
from functools import lru_cache
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import torch

from shapely.geometry import LineString, MultiLineString
from shapely.ops import linemerge
from sklearn.preprocessing import StandardScaler


# Settings
attribute_file = Path("Data/JLR_Subbasins_attributes.csv")
channel_file = Path("Data/JLR_Topo/River/Channels.shp")
stream_file = Path("Data/JLR_Topo/River/Streams.shp")
sub_file = Path("Data/JLR_Topo/Sub_Topo.shp")

device = torch.device("cpu")
dtype = torch.float32


# Feature scaling
def scale_features(df):
    df = (
        df.apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(df.median())
    )
    scaler = StandardScaler()
    scaled = pd.DataFrame(
        scaler.fit_transform(df),
        index=df.index,
        columns=df.columns,
    )
    return df, scaled, scaler


# Channel sinuosity
def calculate_sinuosity(geometry):
    if geometry is None or geometry.is_empty:
        return 1.0

    if isinstance(geometry, MultiLineString):
        geometry = linemerge(geometry)

        if isinstance(geometry, MultiLineString):
            parts = list(geometry.geoms)
            lengths = np.asarray([part.length for part in parts])
            values = np.asarray([calculate_sinuosity(part) for part in parts])
            return float(np.average(values, weights=np.maximum(lengths, 1.0e-8)))

    if not isinstance(geometry, LineString):
        return 1.0

    coords = list(geometry.coords)
    if len(coords) < 2:
        return 1.0

    direct_length = np.hypot(
        coords[-1][0] - coords[0][0],
        coords[-1][1] - coords[0][1],
    )
    return max(float(geometry.length) / max(float(direct_length), 1.0e-8), 1.0)


# Merge river path
def merge_path_geometry(geometries):
    lines = []
    for geometry in geometries:
        if isinstance(geometry, LineString):
            lines.append(geometry)
        elif isinstance(geometry, MultiLineString):
            lines.extend(geometry.geoms)

    return lines[0] if len(lines) == 1 else linemerge(lines)


# Trace river path inside downstream Topo
def trace_downstream_basin(start_link, downstream_topo_i):
    path_links = []
    current_link = int(start_link)

    while current_link > 0 and link_to_topo.get(current_link, -1) == downstream_topo_i:
        path_links.append(current_link)
        current_link = int(downstream_link_map.get(current_link, -1))

    return path_links


# Topology
topo_order = np.asarray(topo_ids, dtype=np.int64)
downstream_topo_by_topo = dict(
    zip(topo_order.astype(int), np.asarray(downstream_topo, dtype=int))
)


# HBV attributes
attribute_df = pd.read_csv(attribute_file).rename(columns={"basin_id": "Topo"})
attribute_df["Topo"] = pd.to_numeric(attribute_df["Topo"], errors="coerce").astype(int)
attribute_df = (
    attribute_df
    .drop_duplicates("Topo")
    .set_index("Topo")
    .reindex(topo_order)
)

attribute_df["elevation_range_m"] = (
    attribute_df["ele_mt_smx"] - attribute_df["ele_mt_smn"]
)
attribute_df["temperature_range_c"] = (
    attribute_df["tmp_dc_smx"] - attribute_df["tmp_dc_smn"]
)

hbv_features = [
    "area", "ele_mt_sav", "elevation_range_m", "slp_dg_sav",
    "snd_pc_sav", "slt_pc_sav", "cly_pc_sav", "soc_th_sav",
    "swc_pc_s01", "swc_pc_s04", "swc_pc_s09", "gwt_cm_sav",
    "kar_pc_sse", "for_pc_sse", "crp_pc_sse", "urb_pc_sse",
    "wet_pc_sg1", "pre_mm_syr", "pet_mm_syr", "tmp_dc_syr",
    "temperature_range_c", "snw_pc_syr", "ari_ix_sav", "cmi_ix_syr",
]

hbv_df, hbv_scaled, hbv_scaler = scale_features(attribute_df[hbv_features])
x_hbv = torch.as_tensor(
    hbv_scaled.to_numpy(),
    dtype=dtype,
    device=device,
)


# River data
river_file = channel_file if channel_file.exists() else stream_file

rivers = gpd.read_file(river_file)
subbasins = gpd.read_file(sub_file).rename(columns={"basin_id": "Topo"})
subbasins["Topo"] = pd.to_numeric(subbasins["Topo"], errors="coerce").astype(int)

for column in ["LINKNO", "DSLINKNO", "Slope", "strmOrder"]:
    rivers[column] = pd.to_numeric(rivers[column], errors="coerce")

rivers["LINKNO"] = rivers["LINKNO"].astype(int)
rivers["DSLINKNO"] = rivers["DSLINKNO"].fillna(-1).astype(int)
rivers = (
    rivers[rivers.geometry.notna() & ~rivers.geometry.is_empty]
    .drop_duplicates("LINKNO")
    .copy()
)

if rivers.crs != subbasins.crs:
    rivers = rivers.to_crs(subbasins.crs)

if rivers.crs.is_geographic:
    metric_crs = subbasins.estimate_utm_crs()
    rivers = rivers.to_crs(metric_crs)
    subbasins = subbasins.to_crs(metric_crs)


# Match river reaches to Topo
midpoints = rivers[["LINKNO", "geometry"]].copy()
midpoints["geometry"] = midpoints.geometry.interpolate(0.5, normalized=True)

assignment = (
    gpd.sjoin(
        midpoints,
        subbasins[["Topo", "geometry"]],
        how="left",
        predicate="within",
    )
    .sort_values(["LINKNO", "Topo"], na_position="last")
    .drop_duplicates("LINKNO")[["LINKNO", "Topo"]]
)

missing = assignment["Topo"].isna()
if missing.any():
    missing_links = assignment.loc[missing, "LINKNO"]
    nearest = (
        gpd.sjoin_nearest(
            midpoints[midpoints["LINKNO"].isin(missing_links)],
            subbasins[["Topo", "geometry"]],
            how="left",
            distance_col="_distance",
        )
        .sort_values(["LINKNO", "_distance", "Topo"])
        .drop_duplicates("LINKNO")
    )
    nearest_topo = dict(zip(nearest["LINKNO"], nearest["Topo"]))
    assignment.loc[missing, "Topo"] = missing_links.map(nearest_topo)

assignment["Topo"] = assignment["Topo"].astype(int)
rivers = rivers.merge(assignment, on="LINKNO", how="left")


# Basic river attributes
rivers["Slope"] = (
    rivers["Slope"]
    .replace([np.inf, -np.inf], np.nan)
    .fillna(rivers["Slope"].median())
    .clip(lower=0.0)
)
rivers["strmOrder"] = (
    rivers["strmOrder"]
    .replace([np.inf, -np.inf], np.nan)
    .fillna(rivers["strmOrder"].median())
)

rivers["length_km"] = rivers.geometry.length / 1000.0
rivers["sinuosity"] = rivers.geometry.apply(calculate_sinuosity)

stream_by_link = rivers.set_index("LINKNO", drop=False)
downstream_link_map = rivers.set_index("LINKNO")["DSLINKNO"].astype(int).to_dict()
link_to_topo = rivers.set_index("LINKNO")["Topo"].astype(int).to_dict()


# Contributing area
area_by_topo = attribute_df["area"].astype(float).to_dict()
upstream_topos_by_topo = {int(topo): [] for topo in topo_order}

for topo, downstream in downstream_topo_by_topo.items():
    if downstream in upstream_topos_by_topo:
        upstream_topos_by_topo[downstream].append(topo)


@lru_cache(maxsize=None)
def calculate_contributing_area(topo):
    return float(
        area_by_topo[topo]
        + sum(calculate_contributing_area(upstream) for upstream in upstream_topos_by_topo[topo])
    )


contrib_area_by_topo = {
    int(topo): calculate_contributing_area(int(topo))
    for topo in topo_order
}


# Select outlet river reach for each Topo
def select_outlet_link(topo):
    downstream_topo_i = downstream_topo_by_topo[topo]
    reaches = rivers[rivers["Topo"] == topo].copy()
    reaches["next_topo"] = reaches["DSLINKNO"].map(link_to_topo).fillna(-1).astype(int)

    if downstream_topo_i > 0:
        candidates = reaches[reaches["next_topo"] == downstream_topo_i]
    else:
        candidates = reaches[
            (reaches["DSLINKNO"] <= 0)
            | (reaches["next_topo"] != topo)
        ]

    candidates = candidates.sort_values(
        ["strmOrder", "length_km", "LINKNO"],
        ascending=[False, False, True],
    )
    return int(candidates.iloc[0]["LINKNO"])


outlet_link_by_topo = {
    int(topo): select_outlet_link(int(topo))
    for topo in topo_order
}


# Outlet-to-outlet RiverUH attributes
river_records = []

for topo_value in topo_order:
    topo = int(topo_value)
    target_topo = int(downstream_topo_by_topo[topo])

    source_link = outlet_link_by_topo[topo]
    source_reach = stream_by_link.loc[source_link]

    source_area = float(contrib_area_by_topo[topo])
    source_slope = float(source_reach["Slope"])
    source_order = float(source_reach["strmOrder"])

    if target_topo > 0:
        target_link = outlet_link_by_topo[target_topo]
        target_reach = stream_by_link.loc[target_link]

        path_links = trace_downstream_basin(
            int(source_reach["DSLINKNO"]),
            target_topo,
        )

        target_area = float(contrib_area_by_topo[target_topo])
        target_slope = float(target_reach["Slope"])
        target_order = float(target_reach["strmOrder"])
    else:
        target_topo = topo
        target_link = source_link
        target_reach = source_reach
        path_links = [source_link]

        target_area = source_area
        target_slope = source_slope
        target_order = source_order

    path_reaches = stream_by_link.loc[path_links]
    path_lengths = path_reaches["length_km"].to_numpy(dtype=float)
    path_slopes = path_reaches["Slope"].to_numpy(dtype=float)

    route_length_km = float(path_lengths.sum())
    route_slope = float(
        np.average(
            path_slopes,
            weights=np.maximum(path_lengths, 1.0e-8),
        )
    )
    route_geometry = merge_path_geometry(path_reaches.geometry.tolist())
    route_sinuosity = float(calculate_sinuosity(route_geometry))

    river_records.append({
        "Topo": topo,
        "downstream_Topo": int(downstream_topo_by_topo[topo]),
        "source_LINKNO": source_link,
        "target_LINKNO": target_link,
        "source_contrib_area": source_area,
        "target_contrib_area": target_area,
        "route_length_km": route_length_km,
        "route_slope": route_slope,
        "route_sinuosity": route_sinuosity,
        "source_order": source_order,
        "target_order": target_order,
        "source_slope": source_slope,
        "target_slope": target_slope,
    })


# RiverUH feature tensor
river_full_df = (
    pd.DataFrame(river_records)
    .set_index("Topo")
    .reindex(topo_order)
)

river_features = [
    "source_contrib_area",
    "target_contrib_area",
    "route_length_km",
    "route_slope",
    "route_sinuosity",
    "source_order",
    "target_order",
    "source_slope",
    "target_slope",
]

river_df, river_scaled, river_scaler = scale_features(
    river_full_df[river_features]
)

x_river = torch.as_tensor(
    river_scaled.to_numpy(),
    dtype=dtype,
    device=device,
)


# Output
print(
    f"x_hbv:   {tuple(x_hbv.shape)} | "
    f"Features: {', '.join(hbv_features)}"
)

print(
    f"x_river: {tuple(x_river.shape)} | "
    f"Features: {', '.join(river_features)}"
)

# =====================================================
# Read JLR streamflow data
# =====================================================

from pathlib import Path

import numpy as np
import pandas as pd
import torch


flow_dir = Path("Data/Flow")

device = torch.device("cpu")
dtype = torch.float32


# Station information
station_routing_table = station_routing_table.sort_values("Topo").reset_index(drop=True)

station_names = station_routing_table["station"].astype(str).str.lower().tolist()
station_topos = station_routing_table["Topo"].astype(int).tolist()
station_indices_np = np.array([topo_to_index[topo] for topo in station_topos], dtype=np.int64)

date_index = pd.DatetimeIndex(dates)


# Read observations
qobs_station_list = []

for station in station_names:
    station_df = pd.read_csv(flow_dir / f"{station}.csv")
    station_df.columns = station_df.columns.str.strip().str.replace("\ufeff", "", regex=False).str.lower()
    station_df["date"] = pd.to_datetime(station_df["date"])

    qobs = (
        station_df.drop_duplicates("date")
        .set_index("date")["qobs"]
        .reindex(date_index)
        .to_numpy(dtype=np.float32)
    )

    qobs_station_list.append(qobs)

qobs_station_np = np.stack(qobs_station_list)


# Full Topo observations
qobs_np = np.full((len(topo_ids), len(dates)), np.nan, dtype=np.float32)
qobs_np[station_indices_np] = qobs_station_np


# Torch tensors
station_indices = torch.tensor(station_indices_np, dtype=torch.long, device=device)

qobs = torch.tensor(qobs_np, dtype=dtype, device=device)
qobs_station = torch.tensor(qobs_station_np, dtype=dtype, device=device)

qobs_mask = torch.isfinite(qobs)
qobs_station_mask = torch.isfinite(qobs_station)


# Information
print(
    f"Flow: {len(station_names)} stations | "
    f"Qobs: {tuple(qobs.shape)} | "
    f"Valid: {int(qobs_station_mask.sum())}"
)

# Verify alignment before saving (do not silently reorder time or Topo).
if not np.array_equal(forcing_topo_ids, np.asarray(topo_ids)):
    raise ValueError("Forcing Topo order differs from GIS Topo order")
if not np.array_equal(forcing_dates, np.asarray(dates)):
    raise ValueError("Forcing dates changed during preprocessing")

# Reproduce the notebook's RCI river coordinate from the private geodata.
projected_crs = 32648
channel_path = "Data/JLR_Topo/River/Channels.shp"
sub_proj = (sub_topo[["Topo", "geometry"]].to_crs(epsg=projected_crs)
            .sort_values("Topo").reset_index(drop=True))
channels = gpd.read_file(channel_path).to_crs(epsg=projected_crs)
channel_parts = gpd.overlay(channels[["geometry"]], sub_proj[["Topo", "geometry"]],
                            how="intersection", keep_geom_type=False)
channel_parts["Length"] = channel_parts.length
reach_length = (channel_parts.groupby("Topo")["Length"].sum()
                .reindex(sub_proj["Topo"], fill_value=0.0).to_numpy(dtype=np.float32))
sub_area = sub_proj.area.to_numpy(dtype=np.float32)
downstream_np = np.asarray(downstream_index, dtype=int)
distance_to_outlet = np.zeros(len(topo_ids), dtype=np.float32)
for basin_i in range(len(topo_ids)):
    current_i = basin_i
    distance = -0.5 * reach_length[basin_i]
    while current_i >= 0:
        distance += reach_length[current_i]
        current_i = downstream_np[current_i]
    distance_to_outlet[basin_i] = distance
sub_position = distance_to_outlet / (distance_to_outlet + 0.5 * reach_length).max()

# These processed arrays can contain restricted measurements: retain only locally.
output = Path("Data/processed/model_inputs.npz")
output.parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(
    output,
    dates=np.asarray(pd.DatetimeIndex(dates).strftime("%Y-%m-%d"), dtype="U10"),
    topo_ids=np.asarray(topo_ids, dtype=np.int64),
    station_names=np.asarray(station_names, dtype="U64"),
    station_topos=np.asarray(station_topos, dtype=np.int64),
    forcing=np.asarray(forcing, dtype=np.float32),
    x_rain=np.asarray(x_rain, dtype=np.float32),
    x_hbv=x_hbv.detach().cpu().numpy().astype(np.float32),
    x_river=x_river.detach().cpu().numpy().astype(np.float32),
    area=area_tensor.detach().cpu().numpy().astype(np.float32),
    downstream_index=np.asarray(downstream_index, dtype=np.int64),
    routing_order=np.asarray(routing_order, dtype=np.int64),
    station_indices=station_indices.detach().cpu().numpy().astype(np.int64),
    qobs_station=qobs_station.detach().cpu().numpy().astype(np.float32),
    swe_obs=swe_obs.detach().cpu().numpy().astype(np.float32),
    sm_obs=sm_obs.detach().cpu().numpy().astype(np.float32),
    e_obs=e_obs.detach().cpu().numpy().astype(np.float32),
    rci_sub_position=sub_position.astype(np.float32),
    rci_sub_area=sub_area.astype(np.float32),
)
print(f"Prepared {output} (private; do not publish)")
