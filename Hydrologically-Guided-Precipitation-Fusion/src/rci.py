"""Monthly runoff-centroid index (RCI) geometry and station-increment setup."""
import numpy as np
import torch


def build_rci_context(data, station_positions):
    """Recompute gauge increments for calibration subset (important for PUB)."""
    down = data.downstream_index.cpu().numpy().astype(int)
    topos = data.topo_ids.astype(int)
    stations = data.station_topos[station_positions]
    lookup = {int(t): i for i, t in enumerate(stations)}
    basin_lookup = {int(t): i for i, t in enumerate(topos)}
    ns = len(stations)
    inc = np.eye(ns, dtype=np.float32)
    for i, topo in enumerate(stations):
        j = down[basin_lookup[int(topo)]]
        while j >= 0:
            k = lookup.get(int(topos[j]))
            if k is not None:
                inc[k, i] = -1.0
                break
            j = down[j]
    pos = data.rci_sub_position
    area = data.rci_sub_area
    if pos is None or area is None:
        raise ValueError("RCI needs rci_sub_position and rci_sub_area in input arrays")
    groups = [[] for _ in stations]
    for i in range(len(topos)):
        j = i
        while j >= 0:
            k = lookup.get(int(topos[j]))
            if k is not None:
                groups[k].append(i)
                break
            j = down[j]
    if any(not group for group in groups):
        raise ValueError("RCI station has an empty contributing region")
    location = np.array([np.average(pos[group], weights=area[group]) for group in groups],
                        dtype=np.float32)
    obs = data.qobs_station.index_select(0, torch.as_tensor(station_positions, dtype=torch.long))
    valid = torch.isfinite(obs)
    inc_mask = torch.stack([
        valid.index_select(0, torch.tensor(np.flatnonzero(np.abs(inc[i]) > 0), dtype=torch.long)).all(0)
        for i in range(ns)
    ])
    return (torch.as_tensor(inc), inc_mask, torch.as_tensor(location))
