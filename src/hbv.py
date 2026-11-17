"""Differentiable HBV snow, soil and runoff generation.

Hydrological fluxes and parameterization for daily simulations.
"""
from typing import List, Tuple
import torch
import torch.nn.functional as F
from .routing import gamma_weights, _parallel_uh_routing, _river_network_routing, uh_moments

@torch.jit.script
def _hbv_runoff(
    prec: torch.Tensor, pet: torch.Tensor, temp: torch.Tensor,
    T1: torch.Tensor, DD: torch.Tensor, FC: torch.Tensor,
    beta: torch.Tensor, eta: torch.Tensor,
    Ksuf: torch.Tensor, Kinter: torch.Tensor, Kbase: torch.Tensor,
    kperc: torch.Tensor, l_ratio: torch.Tensor,
    init_state: torch.Tensor, return_comp: bool,
):
    nb, nt = prec.shape
    zero = torch.zeros(nb, dtype=prec.dtype, device=prec.device)
    one = torch.ones_like(zero)

    # Initial states
    if init_state.numel() == 0:
        snow, soil = zero.clone(), 0.5 * FC
        upper, lower = zero.clone(), zero.clone()
    else:
        snow, soil, upper, lower = init_state.unbind(1)

    # Output lists
    runoff_list = torch.jit.annotate(List[torch.Tensor], [])
    aet_list = torch.jit.annotate(List[torch.Tensor], [])
    snow_list = torch.jit.annotate(List[torch.Tensor], [])
    soil_list = torch.jit.annotate(List[torch.Tensor], [])
    upper_list = torch.jit.annotate(List[torch.Tensor], [])
    lower_list = torch.jit.annotate(List[torch.Tensor], [])
    peff_list = torch.jit.annotate(List[torch.Tensor], [])
    qsuf_list = torch.jit.annotate(List[torch.Tensor], [])
    qinter_list = torch.jit.annotate(List[torch.Tensor], [])
    qperc_list = torch.jit.annotate(List[torch.Tensor], [])
    qbase_list = torch.jit.annotate(List[torch.Tensor], [])

    threshold = l_ratio * FC

    # HBV simulation
    for t in range(nt):
        p, e, temperature = prec[:, t], pet[:, t], temp[:, t]

        # Snow
        snow_fraction = torch.sigmoid((T1 - temperature) / 1.0)
        snowfall = p * snow_fraction
        rainfall = p * (1.0 - snow_fraction)

        snow_available = snow + snowfall

        melt = torch.minimum(
            DD * F.softplus(temperature - T1),
            snow_available,
        )

        snow = (snow_available - melt).clamp_min(0.0)
        water_input = rainfall + melt

        # Soil
        relative_soil = (soil / FC).clamp(1.0e-4, 1.0)
        peff = water_input * relative_soil.pow(beta)

        aet = e * (soil / (eta * FC)).clamp(0.0, 1.0)
        available_water = (soil + water_input - peff).clamp_min(0.0)
        aet = torch.minimum(aet, available_water)

        soil_raw = soil + water_input - peff - aet
        overflow = (soil_raw - FC).clamp_min(0.0)
        soil = torch.minimum(soil_raw.clamp_min(0.0), FC)
        peff = peff + overflow

        # Upper reservoir
        upper_available = (upper + peff).clamp_min(0.0)
        qsuf = Ksuf * (upper_available - threshold).clamp_min(0.0)
        qinter = Kinter * upper_available
        qperc = kperc * upper_available

        total_out = qsuf + qinter + qperc
        scale = torch.minimum(
            one,
            upper_available / total_out.clamp_min(1.0e-6),
        )
        qsuf, qinter, qperc = qsuf * scale, qinter * scale, qperc * scale
        upper = (upper_available - qsuf - qinter - qperc).clamp_min(0.0)

        # Lower reservoir
        lower_available = (lower + qperc).clamp_min(0.0)
        qbase = Kbase * lower_available
        lower = (lower_available - qbase).clamp_min(0.0)

        runoff_list.append(qsuf + qinter + qbase)
        aet_list.append(aet)

        if return_comp:
            snow_list.append(snow)
            soil_list.append(soil)
            upper_list.append(upper)
            lower_list.append(lower)
            peff_list.append(peff)
            qsuf_list.append(qsuf)
            qinter_list.append(qinter)
            qperc_list.append(qperc)
            qbase_list.append(qbase)

    # Outputs
    runoff = torch.stack(runoff_list, dim=1)
    aet = torch.stack(aet_list, dim=1)
    final_state = torch.stack([snow, soil, upper, lower], dim=1)

    empty = torch.empty(0, dtype=prec.dtype, device=prec.device)
    comp = torch.jit.annotate(
        Tuple[
            torch.Tensor, torch.Tensor, torch.Tensor,
            torch.Tensor, torch.Tensor, torch.Tensor,
            torch.Tensor, torch.Tensor, torch.Tensor,
        ],
        (empty, empty, empty, empty, empty, empty, empty, empty, empty),
    )

    if return_comp:
        comp = (
            torch.stack(snow_list, dim=1),
            torch.stack(soil_list, dim=1),
            torch.stack(upper_list, dim=1),
            torch.stack(lower_list, dim=1),
            torch.stack(peff_list, dim=1),
            torch.stack(qsuf_list, dim=1),
            torch.stack(qinter_list, dim=1),
            torch.stack(qperc_list, dim=1),
            torch.stack(qbase_list, dim=1),
        )

    return runoff, final_state, aet, comp


def Torch_rain_hbv_riveruh(
    forcing, corrected_prec, params, river_uh, area,
    downstream_index, routing_order,
    init_state=None, return_comp=False,
):
    # Inputs
    prec = corrected_prec.clamp_min(0.0)
    pet = forcing[:, :, 1].clamp_min(0.0)
    temp = forcing[:, :, 2]

    (
        T1, DD, FC, beta, eta,
        Ksuf, Kinter, Kbase, kperc, l_ratio,
        alpha, tau,
    ) = params.unbind(1)

    empty = torch.empty(0, dtype=forcing.dtype, device=forcing.device)
    hbv_state = empty if init_state is None else init_state

    # HBV runoff
    runoff, final_state, aet, hbv_comp = _hbv_runoff(
        prec, pet, temp,
        T1, DD, FC, beta, eta,
        Ksuf, Kinter, Kbase, kperc, l_ratio,
        hbv_state, return_comp,
    )

    # Local Gamma routing
    gamma_uh = gamma_weights(alpha, tau, 24)
    gamma_runoff = _parallel_uh_routing(runoff, gamma_uh)
    local_q = gamma_runoff * area[:, None] / 86.4

    # River network routing
    qsim, river_inflow, river_outflow = _river_network_routing(
        local_q, river_uh,
        downstream_index, routing_order,
        return_comp,
    )

    if not return_comp:
        return qsim, final_state, aet, None

    river_center, river_dispersion = uh_moments(river_uh)

    comp = {
        "params": params,
        "corrected_prec": corrected_prec,
        "gamma_uh": gamma_uh,
        "river_uh": river_uh,
        "river_center": river_center,
        "river_dispersion": river_dispersion,
        "snow": hbv_comp[0],
        "soil": hbv_comp[1],
        "upper": hbv_comp[2],
        "lower": hbv_comp[3],
        "peff": hbv_comp[4],
        "aet": aet,
        "qsuf": hbv_comp[5],
        "qinter": hbv_comp[6],
        "qperc": hbv_comp[7],
        "qbase": hbv_comp[8],
        "runoff": runoff,
        "gamma_runoff": gamma_runoff,
        "local_q": local_q,
        "river_inflow": river_inflow,
        "river_outflow": river_outflow,
        "qsim": qsim,
        "final_state": final_state,
    }

    return qsim, final_state, aet, comp


def Torch_hbv_riveruh(forcing, params, river_uh, area, downstream_index,
                      routing_order, init_state=None, return_comp=False):
    """Single-product mode using forcing[:, :, 0] as rainfall."""
    qsim, final_state, aet, comp = Torch_rain_hbv_riveruh(
        forcing, forcing[:, :, 0], params, river_uh, area,
        downstream_index, routing_order, init_state, return_comp,
    )
    if comp is not None:
        comp.pop("corrected_prec", None)
    return qsim, final_state, aet, comp
