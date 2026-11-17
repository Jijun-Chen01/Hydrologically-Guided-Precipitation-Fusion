"""Streamflow, SWE, SM, ET and monthly RCI losses from the notebooks.

Scientific loss definitions are preserved; caller must supply valid observations.
"""
import torch

def _masked_moments(values, mask):
    count = mask.sum(1).to(values.dtype).clamp_min(1.0)
    values = torch.where(mask, values, 0.0)
    mean = values.sum(1) / count
    dev = torch.where(mask, values - mean[:, None], 0.0)
    square = dev.square().sum(1)
    return count, mean, dev, square


def station_loss(qsim, qobs, obs_mask, time_mask, raw_weight=0.75, eps=1.0e-6):
    # Valid streamflow
    valid = obs_mask & time_mask[None, :] & torch.isfinite(qsim) & torch.isfinite(qobs)
    sim, obs = qsim.clamp_min(0.0), qobs.clamp_min(0.0)

    count, sim_mean, sim_dev, sim_square = _masked_moments(sim, valid)
    _, obs_mean, obs_dev, obs_square = _masked_moments(obs, valid)

    # Raw-flow loss
    sse = torch.where(valid, sim - obs, 0.0).square().sum(1)
    sim_std = (sim_square / count).sqrt().clamp_min(eps)
    obs_std = (obs_square / count).sqrt().clamp_min(eps)
    raw_rmse = (sse / count + eps).sqrt() / obs_std

    # Square-root-flow loss
    sim_sqrt, obs_sqrt = (sim + eps).sqrt(), (obs + eps).sqrt()
    _, _, _, obs_sqrt_square = _masked_moments(obs_sqrt, valid)
    sqrt_sse = torch.where(valid, sim_sqrt - obs_sqrt, 0.0).square().sum(1)
    sqrt_rmse = (sqrt_sse / count + eps).sqrt() / (
        obs_sqrt_square / count
    ).sqrt().clamp_min(eps)

    # NSE and KGE
    correlation = (
        (sim_dev * obs_dev).sum(1)
        / (sim_square * obs_square).clamp_min(eps).sqrt()
    ).clamp(-1.0, 1.0)

    nse = 1.0 - sse / obs_square.clamp_min(eps)
    kge = 1.0 - (
        (correlation - 1.0).square()
        + (sim_std / obs_std - 1.0).square()
        + (sim_mean / obs_mean.clamp_min(eps) - 1.0).square()
    ).sqrt()

    valid_station = (count > 1) & (obs_square > eps)
    mean_raw = raw_rmse[valid_station].mean()
    mean_sqrt = sqrt_rmse[valid_station].mean()
    loss = raw_weight * mean_raw + (1.0 - raw_weight) * mean_sqrt

    return loss, mean_raw, mean_sqrt, nse[valid_station].mean(), kge[valid_station].mean()


def basin_nrmse(sim, obs, mask, eps=1.0e-8):
    # Basin-wise normalized RMSE
    mask = mask & torch.isfinite(sim) & torch.isfinite(obs)
    count, _, sim_dev, sim_square = _masked_moments(sim, mask)
    _, _, obs_dev, obs_square = _masked_moments(obs, mask)

    sse = torch.where(mask, sim - obs, 0.0).square().sum(1)
    loss = (sse / obs_square.clamp_min(eps)).sqrt()

    correlation = (
        (sim_dev * obs_dev).sum(1)
        / (sim_square * obs_square).clamp_min(eps).sqrt()
    ).clamp(-1.0, 1.0)

    valid = (count > 1) & (obs_square > eps)
    return loss[valid].mean(), correlation[valid].mean()


def basin_standardized_rmse(sim, obs, mask, eps=1.0e-8):
    # Basin-wise standardized RMSE
    mask = mask & torch.isfinite(sim) & torch.isfinite(obs)
    count, _, sim_dev, sim_square = _masked_moments(sim, mask)
    _, _, obs_dev, obs_square = _masked_moments(obs, mask)

    sim_std = (sim_square / count).sqrt().clamp_min(eps)
    obs_std = (obs_square / count).sqrt().clamp_min(eps)
    sim_z = sim_dev / sim_std[:, None]
    obs_z = obs_dev / obs_std[:, None]

    sse = torch.where(mask, sim_z - obs_z, 0.0).square().sum(1)
    loss = (sse / count).sqrt()

    correlation = (
        (sim_dev * obs_dev).sum(1)
        / (sim_square * obs_square).clamp_min(eps).sqrt()
    ).clamp(-1.0, 1.0)

    valid = (count > 1) & (sim_square > eps) & (obs_square > eps)
    return loss[valid].mean(), correlation[valid].mean()


def calculate_process_loss(comp, aet, swe_obs, sm_obs, e_obs, time_mask):
    # SWE, soil moisture, and ET
    time_mask = time_mask[None, :]

    swe_loss, swe_r = basin_nrmse(
        comp["snow"], swe_obs, torch.isfinite(swe_obs) & time_mask,
    )
    sm_loss, sm_r = basin_standardized_rmse(
        comp["soil"], sm_obs, torch.isfinite(sm_obs) & time_mask,
    )
    e_loss, e_r = basin_nrmse(
        aet, e_obs, torch.isfinite(e_obs) & time_mask,
    )

    return (swe_loss + sm_loss + e_loss) / 3.0, swe_r, sm_r, e_r


def _monthly_masked_mean(values, mask, month_index, n_months):
    # Daily to monthly mean
    index = month_index[None, :].expand(values.shape[0], -1)

    monthly_sum = values.new_zeros(values.shape[0], n_months).scatter_add(
        1, index, torch.where(mask, values, 0.0),
    )
    monthly_count = values.new_zeros(values.shape[0], n_months).scatter_add(
        1, index, mask.to(values.dtype),
    )

    return monthly_sum / monthly_count.clamp_min(1.0), monthly_count


def temporal_rci_loss(
    qsim, qobs, increment_matrix, increment_mask, position,
    month_index, n_months, month_mask, eps=1.0e-8,
):
    # Monthly incremental runoff
    qsim = torch.where(torch.isfinite(qsim), qsim, 0.0)
    qobs = torch.where(torch.isfinite(qobs), qobs, 0.0)

    sim_inc = increment_matrix @ qsim
    obs_inc = increment_matrix @ qobs

    sim_month, month_count = _monthly_masked_mean(
        sim_inc, increment_mask, month_index, n_months,
    )
    obs_month, _ = _monthly_masked_mean(
        obs_inc, increment_mask, month_index, n_months,
    )

    # Monthly relative centroid
    sim_total = sim_month.sum(0)
    obs_total = obs_month.sum(0)

    rci_sim = (
        sim_month * position[:, None]
    ).sum(0) / sim_total.clamp_min(eps)

    rci_obs = (
        obs_month * position[:, None]
    ).sum(0) / obs_total.clamp_min(eps)

    valid = (
        month_mask
        & (month_count > 0).all(0)
        & (sim_total > eps)
        & (obs_total > eps)
        & torch.isfinite(rci_sim)
        & torch.isfinite(rci_obs)
    )

    sim, obs = rci_sim[valid], rci_obs[valid]

    # Temporal RCI KGE
    sim_mean, obs_mean = sim.mean(), obs.mean()
    sim_dev, obs_dev = sim - sim_mean, obs - obs_mean
    sim_square = sim_dev.square().sum()
    obs_square = obs_dev.square().sum()
    count = sim.new_tensor(sim.numel())

    sim_std = (sim_square / count).sqrt()
    obs_std = (obs_square / count).sqrt().clamp_min(eps)

    correlation = (
        (sim_dev * obs_dev).sum()
        / (sim_square * obs_square).clamp_min(eps).sqrt()
    ).clamp(-1.0, 1.0)

    kge = 1.0 - (
        (correlation - 1.0).square()
        + (sim_std / obs_std - 1.0).square()
        + (sim_mean / obs_mean.clamp_min(eps) - 1.0).square()
    ).sqrt()

    return 1.0 - kge, kge, correlation
