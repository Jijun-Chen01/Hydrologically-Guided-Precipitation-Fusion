"""Load prepared (private or synthetic) tensors with explicit Topo/date alignment."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import torch


@dataclass
class ModelData:
    dates: pd.DatetimeIndex
    topo_ids: np.ndarray
    station_names: list[str]
    station_topos: np.ndarray
    forcing: torch.Tensor
    x_rain: torch.Tensor
    x_hbv: torch.Tensor
    x_river: torch.Tensor
    area: torch.Tensor
    downstream_index: torch.Tensor
    routing_order: torch.Tensor
    station_indices: torch.Tensor
    qobs_station: torch.Tensor
    swe_obs: torch.Tensor
    sm_obs: torch.Tensor
    e_obs: torch.Tensor
    rci_sub_position: np.ndarray | None
    rci_sub_area: np.ndarray | None


def load_model_data(path: str | Path, device: str = "cpu") -> ModelData:
    """Load preprocessed arrays; the repository does not ship restricted observations."""
    with np.load(path, allow_pickle=False) as z:
        required = ("dates", "topo_ids", "station_names", "station_topos", "forcing",
                    "x_rain", "x_hbv", "x_river", "area", "downstream_index",
                    "routing_order", "station_indices", "qobs_station", "swe_obs", "sm_obs", "e_obs")
        absent = sorted(set(required) - set(z.files))
        if absent:
            raise ValueError(f"Missing arrays in {path}: {absent}")
        a = {key: z[key] for key in required}
        positions = z["rci_sub_position"].copy() if "rci_sub_position" in z else None
        areas = z["rci_sub_area"].copy() if "rci_sub_area" in z else None
    b, t, s = len(a["topo_ids"]), len(a["dates"]), len(a["station_names"])
    for name, shape in {"forcing": (b,t,4), "x_rain": (b,t,4), "x_hbv": (b,24),
                        "x_river": (b,9), "area": (b,), "qobs_station": (s,t),
                        "swe_obs": (b,t), "sm_obs": (b,t), "e_obs": (b,t)}.items():
        if a[name].shape != shape:
            raise ValueError(f"{name}: expected {shape}, received {a[name].shape}")
    if not np.array_equal(a["topo_ids"][a["station_indices"]], a["station_topos"]):
        raise ValueError("Station Topo ordering and model rows do not match")
    order = a["routing_order"].astype(int)
    down = a["downstream_index"].astype(int)
    if sorted(order.tolist()) != list(range(b)) or np.any((down < -1) | (down >= b)):
        raise ValueError("Invalid routing indices")
    if any(d >= 0 and np.flatnonzero(order == d)[0] <= np.flatnonzero(order == i)[0]
           for i, d in enumerate(down)):
        raise ValueError("Routing order must run upstream to downstream")
    to_float = lambda name: torch.as_tensor(a[name], device=device, dtype=torch.float32)
    to_long = lambda name: torch.as_tensor(a[name], device=device, dtype=torch.long)
    return ModelData(
        dates=pd.DatetimeIndex(a["dates"]), topo_ids=a["topo_ids"],
        station_names=a["station_names"].astype(str).tolist(),
        station_topos=a["station_topos"].astype(int),
        forcing=to_float("forcing"), x_rain=to_float("x_rain"),
        x_hbv=to_float("x_hbv"), x_river=to_float("x_river"),
        area=to_float("area"), downstream_index=to_long("downstream_index"),
        routing_order=to_long("routing_order"), station_indices=to_long("station_indices"),
        qobs_station=to_float("qobs_station"), swe_obs=to_float("swe_obs"),
        sm_obs=to_float("sm_obs"), e_obs=to_float("e_obs"),
        rci_sub_position=positions, rci_sub_area=areas,
    )
