"""Train the submitted baseline or fusion model with optional PUB held-out stations.

Usage: python -m src.train --config configs/gauged.yaml
Prepared real data are private and are not distributed with this repository.
"""
import argparse
import copy
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import yaml
from .data import load_model_data
from .fusion import Rain_HBV, Static_HBV
from .losses import station_loss, calculate_process_loss, temporal_rci_loss
from .rci import build_rci_context

RAIN_SOURCES = ("CHMPRE", "ERA5Land", "GSMaP", "IMERG")


def read_config(path):
    with open(path, encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("Configuration should be a YAML mapping")
    return config


def prepare_periods(dates, config, device):
    days = pd.DatetimeIndex(dates)
    periods = {
        name: torch.as_tensor((days >= start) & (days < end), device=device, dtype=torch.bool)
        for name, (start, end) in config["periods"].items()
    }
    if not periods["Train"].any() or not periods["Val"].any():
        raise ValueError("No samples for specified training or validation date bounds")
    months = days.to_period("M")
    unique_months = pd.PeriodIndex(months.unique())
    indices = torch.as_tensor(unique_months.get_indexer(months), device=device, dtype=torch.long)
    month_start = unique_months.to_timestamp()
    monthly_masks = {
        name: torch.as_tensor((month_start >= start) & (month_start < end), device=device, dtype=torch.bool)
        for name, (start, end) in config["periods"].items()
    }
    return periods, indices, len(unique_months), monthly_masks


def build_model(data, config):
    shared = dict(
        hbv_attr_dim=data.x_hbv.shape[1], river_attr_dim=data.x_river.shape[1],
        downstream_index=data.downstream_index, routing_order=data.routing_order,
        hbv_hidden=tuple(config.get("hbv_hidden", [128])),
        river_hidden=tuple(config.get("river_hidden", [64])),
        kernel_size=config.get("kernel_size", 30),
    )
    if config.get("model", "fusion") == "fusion":
        return Rain_HBV(rain_attr_dim=data.x_rain.shape[2],
                        rain_hidden=tuple(config.get("rain_hidden", [128])), **shared)
    if config["model"] == "single":
        return Static_HBV(**shared)
    raise ValueError("model must be 'fusion' or 'single'")


def train(config, save=True):
    # CPU is the default training device.
    device = config.get("device", "cpu")
    seed = int(config.get("seed", 2026))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    data = load_model_data(config["data_file"], device=device)
    model = build_model(data, config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config.get("learning_rate", 1e-3)))
    periods, month_index, n_months, month_masks = prepare_periods(data.dates, config, device)
    heldout = set(str(x).lower() for x in config.get("pub_stations", []))
    known = set(s.lower() for s in data.station_names)
    if heldout - known:
        raise ValueError(f"PUB stations missing from dataset: {heldout - known}")
    cal_pos = [i for i, name in enumerate(data.station_names) if name.lower() not in heldout]
    pub_pos = [i for i, name in enumerate(data.station_names) if name.lower() in heldout]
    if not cal_pos:
        raise ValueError("No calibration stations remain after PUB split")
    cal_pos_t = torch.as_tensor(cal_pos, dtype=torch.long, device=device)
    cal_indices = data.station_indices.index_select(0, cal_pos_t)
    cal_obs = data.qobs_station.index_select(0, cal_pos_t)
    cal_mask = torch.isfinite(cal_obs)  # Matches the original notebooks.
    use_process = bool(config.get("use_process_constraint", True))
    use_rci = bool(config.get("use_rci_constraint", True))
    flow_weight = float(config.get("flow_weight", 0.7))
    process_weight = float(config.get("process_weight", 0.2))
    rci_weight = float(config.get("rci_weight", 0.1))
    if not use_process and process_weight or not use_rci and rci_weight:
        raise ValueError("Disabled constraints must have weight zero; set explicit weights in YAML")
    if abs(flow_weight + process_weight + rci_weight - 1) > 1e-6:
        raise ValueError("Loss weights must sum to one")
    if use_rci:
        increment, increment_mask, position = build_rci_context(data, cal_pos)
        increment, increment_mask, position = increment.to(device), increment_mask.to(device), position.to(device)
    model_type = config.get("model", "fusion")
    forcing = data.forcing.clone()
    if model_type == "single":
        name = config.get("rain_source", "CHMPRE")
        if name not in RAIN_SOURCES:
            raise ValueError(f"Unsupported source {name}; choose from {RAIN_SOURCES}")
        forcing[:, :, 0] = data.x_rain[:, :, RAIN_SOURCES.index(name)]

    def forward():
        if model_type == "fusion":
            qsim, _, _, _, _, _, aet, comp = model(
                data.x_hbv, data.x_rain, data.x_river, forcing, data.area,
                return_comp=use_process)
        else:
            qsim, _, _, _, aet, comp = model(
                data.x_hbv, data.x_river, forcing, data.area,
                return_comp=use_process)
        return qsim, aet, comp

    def objective(period):
        qsim, aet, comp = forward()
        station_q = qsim.index_select(0, cal_indices)
        flow_loss, _, _, nse, kge = station_loss(station_q, cal_obs, cal_mask, periods[period])
        process_loss = flow_loss.new_zeros(())
        rci_loss = flow_loss.new_zeros(())
        if use_process:
            process_loss, _, _, _ = calculate_process_loss(comp, aet,
                data.swe_obs, data.sm_obs, data.e_obs, periods[period])
        if use_rci:
            rci_loss, _, _ = temporal_rci_loss(
                station_q, cal_obs, increment, increment_mask, position,
                month_index, n_months, month_masks[period])
        total = flow_weight * flow_loss + process_weight * process_loss + rci_weight * rci_loss
        return total, flow_loss, nse, kge

    epochs = int(config.get("epochs", 1000))
    patience = int(config.get("patience", 64))
    print_interval = int(config.get("print_interval", 5))
    best_loss, best_epoch, best_state, wait = float("inf"), 0, None, 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        tr_total, tr_flow, tr_nse, tr_kge = objective("Train")
        if not torch.isfinite(tr_total):
            raise RuntimeError("Non-finite training loss; inspect observations and RCI coverage")
        tr_total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        model.eval()
        with torch.no_grad():
            va_total, va_flow, va_nse, va_kge = objective("Val")
        val = float(va_total)
        if not np.isfinite(val):
            raise RuntimeError("Non-finite validation loss; inspect observations and RCI coverage")
        history.append(dict(epoch=epoch, train_total=float(tr_total.detach()), val_total=val,
                            train_nse=float(tr_nse.detach()), val_nse=float(va_nse)))
        if val < best_loss:
            best_loss, best_epoch, best_state, wait = val, epoch, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
        if epoch == 1 or epoch % print_interval == 0:
            print(f"{epoch:4d} | Train {float(tr_flow.detach()):.4f} ({float(tr_nse.detach()):.3f}/{float(tr_kge.detach()):.3f})"
                  f" | Val {float(va_flow):.4f} ({float(va_nse):.3f}/{float(va_kge):.3f})"
                  f" | Total {val:.4f}")
        if wait >= patience:
            break
    model.load_state_dict(best_state)
    if save:
        out = Path(config.get("output_dir", "results")); out.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state_dict": best_state, "configuration": config,
                    "best_epoch": best_epoch, "val_loss": best_loss}, out / "best_model.pt")
        pd.DataFrame(history).to_csv(out / "history.csv", index=False)
    if pub_pos:
        model.eval()
        with torch.no_grad():
            qsim, _, _ = forward()
            sel = torch.as_tensor(pub_pos, dtype=torch.long, device=device)
            qo = data.qobs_station.index_select(0, sel)
            qp = qsim.index_select(0, data.station_indices.index_select(0, sel))
            loss, _, _, nse, kge = station_loss(qp, qo, torch.isfinite(qo), periods["Val"])
        print(f"PUB held-out stations={len(pub_pos)} | Val NSE={float(nse):.3f}"
              f" | KGE={float(kge):.3f} | loss={float(loss):.4f}")
    print(f"Best epoch={best_epoch} | Val objective={best_loss:.6f}")
    return model, history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="YAML configuration")
    parser.add_argument("--data", help="override data_file")
    parser.add_argument("--model", choices=["fusion", "single"], help="override model type")
    parser.add_argument("--source", choices=RAIN_SOURCES, help="single-product rainfall source")
    args = parser.parse_args()
    config = read_config(args.config)
    if args.data: config["data_file"] = args.data
    if args.model: config["model"] = args.model
    if args.source: config["rain_source"] = args.source
    train(config)


if __name__ == "__main__":
    main()
