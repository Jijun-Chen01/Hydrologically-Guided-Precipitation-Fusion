"""Static parameter MLP, multisource rainfall fusion and RiverUH learning.

Class definitions and model bounds follow the original notebooks, not the paper table.
"""
import torch
from torch import nn
from .hbv import Torch_rain_hbv_riveruh, Torch_hbv_riveruh

def make_mlp(input_dim, hidden_dims, output_dim):
    if isinstance(hidden_dims, int):
        hidden_dims = (hidden_dims,)

    dims = [input_dim, *hidden_dims]
    layers = []

    for in_dim, out_dim in zip(dims[:-1], dims[1:]):
        layers.extend([
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.SiLU(),
        ])

    layers.append(nn.Linear(dims[-1], output_dim))
    return nn.Sequential(*layers)


class Rain_HBV(nn.Module):

    def __init__(
        self, hbv_attr_dim, rain_attr_dim, river_attr_dim,
        downstream_index, routing_order,
        hbv_hidden=(128,), rain_hidden=(128,), river_hidden=(64,),
        kernel_size=30, baseline_rain_index=0, baseline_bias=3.0,
    ):
        super().__init__()

        self.kernel_size = int(kernel_size)
        self.rain_attr_dim = int(rain_attr_dim)
        self.baseline_rain_index = int(baseline_rain_index)

        # Topology
        self.register_buffer(
            "downstream_index",
            torch.as_tensor(downstream_index, dtype=torch.long),
        )
        self.register_buffer(
            "routing_order",
            torch.as_tensor(routing_order, dtype=torch.long),
        )

        # Networks
        self.hbv_net = make_mlp(
            hbv_attr_dim, hbv_hidden, 12,
        )
        self.rain_net = make_mlp(
            rain_attr_dim + hbv_attr_dim,
            rain_hidden, rain_attr_dim,
        )
        self.river_net = make_mlp(
            river_attr_dim, river_hidden, self.kernel_size,
        )

        # Rainfall fusion initialization
        nn.init.zeros_(self.rain_net[-1].weight)
        nn.init.zeros_(self.rain_net[-1].bias)

        with torch.no_grad():
            self.rain_net[-1].bias[self.baseline_rain_index] = baseline_bias

        # Parameter bounds
        bounds = torch.tensor([
            [-2.0, 2.0],       # T1
            [0.0, 2.0],        # DD
            [20.0, 2000.0],    # FC
            [1.0, 6.0],        # beta
            [0.1, 1.0],        # eta
            [0.05, 1.0],       # Ksuf
            [0.005, 0.50],     # Kinter
            [0.0005, 0.10],    # Kbase
            [0.001, 0.10],     # kperc
            [0.01, 0.50],      # l_ratio
            [0.1, 3.0],        # alpha
            [0.1, 8.0],        # tau
        ], dtype=torch.float32)

        self.register_buffer("param_lo", bounds[:, 0])
        self.register_buffer("param_range", bounds[:, 1] - bounds[:, 0])

    def forward(
        self, x_hbv, x_rain, x_river, forcing, area,
        init_state=None, return_comp=False,
    ):
        # Static basin parameters
        params = (
            self.param_lo
            + torch.sigmoid(self.hbv_net(x_hbv)) * self.param_range
        )

        # Rainfall input
        rain_valid = torch.isfinite(x_rain)
        rain_input = torch.where(
            rain_valid, x_rain, torch.zeros_like(x_rain),
        ).clamp_min(0.0)

        # Rainfall features
        nt = rain_input.shape[1]
        rain_attr = x_hbv[:, None, :].expand(-1, nt, -1)

        rain_features = torch.cat([
            torch.log1p(rain_input),
            rain_attr,
        ], dim=-1)

        # Rainfall fusion weights
        rain_logits = self.rain_net(rain_features)
        rain_logits = rain_logits.masked_fill(
            ~rain_valid, -1.0e9,
        )

        rain_weights = torch.softmax(
            rain_logits, dim=-1,
        )

        rain_weights = torch.where(
            rain_valid,
            rain_weights,
            torch.zeros_like(rain_weights),
        )

        rain_weights = rain_weights / rain_weights.sum(
            dim=-1, keepdim=True,
        ).clamp_min(1.0e-8)

        # Fused precipitation
        corrected_prec = (
            rain_weights * rain_input
        ).sum(dim=-1)

        # Missing rainfall fallback
        all_missing = ~rain_valid.any(dim=-1)

        original_prec = torch.nan_to_num(
            forcing[:, :, 0],
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).clamp_min(0.0)

        corrected_prec = torch.where(
            all_missing,
            original_prec,
            corrected_prec,
        )

        # RiverUH
        river_uh = torch.softmax(
            self.river_net(x_river), dim=1,
        )

        # Simulation
        qsim, final_state, aet, comp = Torch_rain_hbv_riveruh(
            forcing, corrected_prec, params, river_uh, area,
            self.downstream_index, self.routing_order,
            init_state, return_comp,
        )

        if return_comp:
            comp["rain_weights"] = rain_weights
            comp["rain_valid"] = rain_valid
            comp["corrected_prec"] = corrected_prec

        return (
            qsim, final_state, params, river_uh,
            rain_weights, corrected_prec, aet, comp,
        )


class Static_HBV(nn.Module):

    def __init__(
        self,
        hbv_attr_dim,
        river_attr_dim,
        downstream_index,
        routing_order,
        hbv_hidden=(128,),
        river_hidden=(64,),
        kernel_size=None,
    ):
        super().__init__()
        self.kernel_size = int(kernel_size)

        # Topology
        self.register_buffer(
            "downstream_index",
            torch.as_tensor(downstream_index, dtype=torch.long),
        )
        self.register_buffer(
            "routing_order",
            torch.as_tensor(routing_order, dtype=torch.long),
        )

        # Networks
        self.hbv_net = make_mlp(hbv_attr_dim, hbv_hidden, 12)
        self.river_net = make_mlp(
            river_attr_dim,
            river_hidden,
            self.kernel_size,
        )

        # Parameter bounds
        bounds = torch.tensor([
            [-2.0, 2.0],       # T1
            [0.0, 2.0],        # DD
            [20.0, 2000.0],    # FC
            [1.0, 6.0],        # beta
            [0.1, 1.0],        # eta
            [0.05, 1.0],       # Ksuf
            [0.005, 0.50],     # Kinter
            [0.0005, 0.10],    # Kbase
            [0.001, 0.10],     # kperc
            [0.01, 0.50],      # l_ratio
            [0.1, 3.0],        # alpha
            [0.1, 8.0],        # tau
        ], dtype=torch.float32)

        self.register_buffer("param_lo", bounds[:, 0])
        self.register_buffer("param_range", bounds[:, 1] - bounds[:, 0])

    def forward(
        self,
        x_hbv,
        x_river,
        forcing,
        area,
        init_state=None,
        return_comp=False,
    ):
        # Static basin parameters
        params = (
            self.param_lo
            + torch.sigmoid(self.hbv_net(x_hbv)) * self.param_range
        )

        # RiverUH
        river_uh = torch.softmax(
            self.river_net(x_river),
            dim=1,
        )

        # Simulation
        qsim, final_state, aet, comp = Torch_hbv_riveruh(
            forcing, params, river_uh, area,
            self.downstream_index, self.routing_order,
            init_state, return_comp,
        )

        return qsim, final_state, params, river_uh, aet, comp
