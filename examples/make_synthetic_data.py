"""Generate simulated example data for model interface testing."""
from pathlib import Path
import numpy as np
import pandas as pd

rng = np.random.default_rng(2026)
folder = Path(__file__).resolve().parent / "synthetic_data"
folder.mkdir(parents=True, exist_ok=True)
nb = 4
nt = 96
topo = np.arange(1, nb + 1, dtype=np.int64)
dates = pd.date_range("2007-01-01", periods=nt, freq="D")
downstream = np.array([2, 2, 3, -1], dtype=np.int64)  # 1,2 -> 3 -> 4
rain = rng.gamma(0.8, 3, size=(nb, nt, 4)).astype("float32")
rain[rng.random((nb,nt,4)) < 0.55] = 0
rain[:, :, 1] = 0.7*rain[:, :, 0] + 0.3*rain[:, :, 1]
time = np.arange(nt)[None, :]
p = rain[:, :, 0]
pet = (2.5 + 0.3*np.cos(2*np.pi*time/nt) + np.zeros((nb,1))).astype("float32")
temp = (8 + 6*np.sin(2*np.pi*time/nt) + np.arange(nb)[:, None]).astype("float32")
tmonthly = np.broadcast_to(temp.mean(axis=1, keepdims=True), (nb, nt)).astype("float32")
forcing = np.stack([p, pet, temp, tmonthly], axis=-1)
attr = rng.normal(size=(nb, 24)).astype("float32")
river = rng.normal(size=(nb, 9)).astype("float32")
area = np.array([120, 150, 200, 300], dtype="float32")
q = np.stack([16+0.35*p[2]+0.12*np.arange(nt)/nt, 40+0.6*p[3]+0.2*np.arange(nt)/nt])
q = q.astype("float32")
swe = (25+10*np.sin(2*np.pi*time/nt)+np.arange(nb)[:,None]).astype("float32")
sm = (0.25+0.08*np.cos(2*np.pi*time/nt) +np.arange(nb)[:,None]*0.01).astype("float32")
et = (1.5+0.3*np.sin(2*np.pi*time/nt) + np.arange(nb)[:,None]*0.05).astype("float32")
file = folder / "tiny_demo.npz"
np.savez_compressed(file, dates=np.asarray(dates.strftime('%Y-%m-%d'),dtype='U10'),
    topo_ids=topo, station_names=np.array(['mock_mid','mock_outlet'],dtype='U16'),
    station_topos=topo[[2,3]], forcing=forcing, x_rain=rain,
    x_hbv=attr, x_river=river, area=area, downstream_index=downstream,
    routing_order=np.arange(nb,dtype=np.int64), station_indices=np.array([2,3],dtype=np.int64),
    qobs_station=q, swe_obs=swe, sm_obs=sm, e_obs=et)
print(f'Created synthetic example: {file}')
