from typing import List

import torch
import torch.nn.functional as F
from torch import nn


class GateMixerSigLip(nn.Module):
    def __init__(
            self,
            hidden_size,  # 5120
            input_v_dims: List = [1152, 5760],
            num_register_tokens=0,
    ):
        super().__init__()
        self.mid_dim = hidden_size
        self.num_register_tokens = num_register_tokens
        self.proj_dn1 = nn.Linear(input_v_dims[0], self.mid_dim)
        self.proj_dn2 = nn.Linear(input_v_dims[1], self.mid_dim)

        self.gate = nn.Linear(2 * self.mid_dim, self.mid_dim)
        self.sigmoid = nn.Sigmoid()

        self.proj_up = nn.Linear(self.mid_dim, hidden_size)

        self.register_token_embedding = None
        if self.num_register_tokens > 0:
            self.register_token_embedding = nn.Parameter(torch.zeros(
                1, self.num_register_tokens, self.mid_dim))
            nn.init.trunc_normal_(self.register_token_embedding, mean=0.0, std=0.02)

    def forward(self, xv: List):
        hv1 = self.proj_dn1(xv[0])
        hv2 = self.proj_dn2(xv[1])

        gate = self.sigmoid(self.gate(torch.cat([hv1, hv2], dim=-1)))
        hv = (1 - gate) * hv1 + gate * hv2

        if self.register_token_embedding is not None:
            learn_embed = self.register_token_embedding.expand(hv.shape[0], -1, -1)
            hv = torch.cat([learn_embed, hv], dim=1)

        return self.proj_up(F.silu(hv))


class GateMixer(nn.Module):
    def __init__(
            self,
            hidden_size,  # 5120
            input_v_dims: List = [1024, 1536],
            num_register_tokens=24,
    ):
        super().__init__()
        self.mid_dim = hidden_size
        self.num_register_tokens = num_register_tokens

        self.proj_dn1 = nn.Linear(input_v_dims[0], self.mid_dim)
        self.proj_dn2 = nn.Linear(input_v_dims[1], self.mid_dim)

        self.gate = nn.Linear(2 * self.mid_dim, self.mid_dim)
        self.sigmoid = nn.Sigmoid()

        self.proj_up = nn.Linear(self.mid_dim, hidden_size)

        self.register_token_embedding = None
        if self.num_register_tokens > 0:
            self.register_token_embedding = nn.Parameter(torch.zeros(
                1, self.num_register_tokens, self.mid_dim))
            nn.init.trunc_normal_(
                self.register_token_embedding, mean=0.0, std=0.02)

    def forward(self, xv: List):
        hv1 = self.proj_dn1(xv[0])
        hv2 = self.proj_dn2(xv[1])

        gate = self.sigmoid(self.gate(torch.cat([hv1, hv2], dim=-1)))
        hv = (1 - gate) * hv1 + gate * hv2

        if self.register_token_embedding is not None:
            learn_embed = self.register_token_embedding.expand(
                hv.shape[0], -1, -1)
            hv = torch.cat([learn_embed, hv], dim=1)

        return self.proj_up(F.silu(hv))


def build_mlp(depth, hidden_size, output_hidden_size):
    layers = [nn.Linear(hidden_size, output_hidden_size)]
    for _ in range(1, depth):
        layers.append(nn.SiLU())
        layers.append(nn.Linear(output_hidden_size, output_hidden_size))
    return nn.Sequential(*layers)
