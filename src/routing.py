"""Differentiable Gamma and river-network unit-hydrograph routing.

Core mathematical kernels extracted from the submitted research notebooks.
"""
import torch
import torch.nn.functional as F
from typing import List

@torch.jit.script
def gamma_weights(alpha: torch.Tensor, tau: torch.Tensor, kernel_size: int):
    # Local Gamma unit hydrograph
    lag = torch.arange(
        1, kernel_size + 1,
        dtype=alpha.dtype, device=alpha.device,
    )[None, :]

    log_weights = (
        (alpha[:, None] - 1.0) * torch.log(lag)
        - lag / tau[:, None]
    )
    return torch.softmax(log_weights, dim=1)


@torch.jit.script
def _parallel_uh_routing(q: torch.Tensor, weights: torch.Tensor):
    # Parallel unit hydrograph routing
    kernel_size = weights.shape[1]
    kernel = torch.flip(weights, dims=[1]).unsqueeze(1)

    q_pad = F.pad(q[:, None, :], (kernel_size - 1, 0))
    q_pad = q_pad.reshape(1, q.shape[0], q.shape[1] + kernel_size - 1)

    return F.conv1d(q_pad, kernel, groups=q.shape[0]).squeeze(0)


@torch.jit.script
def _river_network_routing(
    local_q: torch.Tensor,
    river_uh: torch.Tensor,
    downstream_index: torch.Tensor,
    routing_order: torch.Tensor,
    return_comp: bool,
):
    nb, _ = local_q.shape
    kernel_size = river_uh.shape[1]

    # Network flows
    network_flow = torch.jit.annotate(List[torch.Tensor], [])
    outlet_flow = torch.jit.annotate(List[torch.Tensor], [])
    routed_flow = torch.jit.annotate(List[torch.Tensor], [])

    for basin_i in range(nb):
        network_flow.append(local_q[basin_i].clone())
        outlet_flow.append(torch.empty_like(local_q[basin_i]))
        routed_flow.append(torch.empty_like(local_q[basin_i]))

    # Outlet-to-outlet routing
    for order_i in range(routing_order.numel()):
        basin_i = int(routing_order[order_i])
        basin_outlet = network_flow[basin_i]
        outlet_flow[basin_i] = basin_outlet

        downstream = int(downstream_index[basin_i])

        if downstream >= 0:
            basin_pad = F.pad(
                basin_outlet[None, None, :],
                (kernel_size - 1, 0),
            )
            kernel = torch.flip(
                river_uh[basin_i], dims=[0],
            )[None, None, :]

            basin_routed = F.conv1d(basin_pad, kernel)[0, 0]
            network_flow[downstream] = (
                network_flow[downstream] + basin_routed
            )
        else:
            basin_routed = basin_outlet

        routed_flow[basin_i] = basin_routed

    qsim = torch.stack(outlet_flow, dim=0)

    # Optional routing outputs
    if return_comp:
        river_inflow = qsim
        river_outflow = torch.stack(routed_flow, dim=0)
    else:
        river_inflow = torch.empty(0, dtype=local_q.dtype, device=local_q.device)
        river_outflow = torch.empty(0, dtype=local_q.dtype, device=local_q.device)

    return qsim, river_inflow, river_outflow


@torch.jit.script
def uh_moments(weights: torch.Tensor):
    # Unit hydrograph moments
    lag = torch.arange(
        weights.shape[1],
        dtype=weights.dtype, device=weights.device,
    )[None, :]

    center = (weights * lag).sum(dim=1)
    variance = (
        weights * (lag - center[:, None]).square()
    ).sum(dim=1)

    return center, torch.sqrt(variance.clamp_min(1.0e-8))
