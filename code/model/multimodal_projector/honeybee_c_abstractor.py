import copy
import os
from typing import Union

import torch
import torch.nn as nn
from einops import rearrange
from functools import partial

from timm.models.regnet import RegStage
from timm.models.layers import LayerNorm, LayerNorm2d
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class HoneybeeVisualProjectorConfig(PretrainedConfig):
    model_type = "mllm_visual_projector"

    def __init__(
            self,
            projector_type: str = "resampler",
            hidden_size: int = 1024,  #
            num_hidden_layers: int = 6,  #
            num_attention_heads: int = 16,  #
            intermediate_size: int = 4096,  #
            attention_probs_dropout_prob: float = 0.1,  #
            initializer_range: float = 0.02,
            layer_norm_eps: float = 1e-6,  #
            encoder_hidden_size: int = 1024,
            # This will be overwritten by vision_model's hidden_size
            pos_emb=False,
            feature_layer_index=-1,
            # vision feature layer index; -1: last layer
            num_eos_tokens=1,
            use_cls=True,
            prenorm=False,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.projector_type = projector_type
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.encoder_hidden_size = encoder_hidden_size

        self.pos_emb = pos_emb
        self.feature_layer_index = feature_layer_index
        self.num_eos_tokens = num_eos_tokens
        self.use_cls = use_cls
        self.prenorm = prenorm

    @classmethod
    def from_pretrained(
            cls, pretrained_model_name_or_path: Union[str, os.PathLike],
            **kwargs
    ) -> "PretrainedConfig":
        config_dict, kwargs = cls.get_config_dict(
            pretrained_model_name_or_path, **kwargs)

        # get the visual_projector config dict if we are loading from HoneybeeConfig
        if config_dict.get("model_type") == "mllm":
            config_dict = config_dict["projector_config"]

        if (
                "model_type" in config_dict
                and hasattr(cls, "model_type")
                and config_dict["model_type"] != cls.model_type
        ):
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} "
                f"to instantiate a model of type {cls.model_type}. "
                f"This is not supported for all configurations of models and "
                f"can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)


class CAbstractor(nn.Module):
    """C-Abstractor"""
    def __init__(
            self,
            config: HoneybeeVisualProjectorConfig,
            num_input_tokens: int,
            output_hidden_size: int,
    ):
        super().__init__()
        self.config = config
        self.num_input_tokens = num_input_tokens
        self.output_hidden_size = output_hidden_size

        # think tokens
        self.eos_tokens = build_eos_tokens(config, output_hidden_size)

        # pos emb
        self.pos_emb = build_pos_embeds(
            config, num_input_tokens, config.encoder_hidden_size)

        self.prenorm = build_prenorm(config)

        self.build_net()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, encoder_hidden_size) tensor from the visual backbone
            (CLIP visual encoder), including cls token.
        """
        if self.prenorm is not None:
            x = self.prenorm(x)

        if self.pos_emb is not None:
            x += self.pos_emb

        x = self._forward(x)  # (B, L, output_hidden_size)

        B = x.size(0)
        if self.eos_tokens is not None:
            x = torch.cat([x, self.eos_tokens.expand(B, -1, -1)], dim=1)
        return x

    def _forward(self, x):
        # x: [B, L, encoder_hidden_size]
        x = x[:, 1:]  # drop cls token and 2d forward
        hw = int(x.size(1) ** 0.5)
        x = rearrange(x, "b (h w) d -> b d h w", h=hw, w=hw)
        x = self.net(x)
        x = rearrange(x, "b d h w -> b (h w) d")
        x = self.readout(x)

        return x

    def build_net(self):
        encoder_hidden_size = self.config.encoder_hidden_size
        hidden_size = self.config.hidden_size
        output_hidden_size = self.output_hidden_size
        depth = self.config.depth
        mlp_depth = self.config.mlp_depth

        n_queries = self.config.num_queries
        assert (n_queries ** 0.5).is_integer(), "n_queries must be square number"
        hw = int(n_queries ** 0.5)

        # RegBlock = ResBlock + SE
        RegBlock = partial(
            RegStage,
            stride=1,
            dilation=1,
            act_layer=nn.SiLU,
            norm_layer=LayerNorm2d,
        )

        s1 = RegBlock(
            depth,
            encoder_hidden_size,
            hidden_size,
        )
        sampler = nn.AdaptiveAvgPool2d((hw, hw))
        s2 = RegBlock(
            depth,
            hidden_size,
            hidden_size,
        )

        self.net = nn.Sequential(s1, sampler, s2)

        self.readout = build_mlp(mlp_depth, hidden_size, output_hidden_size)


def build_mlp(depth, hidden_size, output_hidden_size):
    layers = [nn.Linear(hidden_size, output_hidden_size)]
    for _ in range(1, depth):
        layers.append(nn.SiLU())
        layers.append(nn.Linear(output_hidden_size, output_hidden_size))
    return nn.Sequential(*layers)


def build_prenorm(config: HoneybeeVisualProjectorConfig):
    if config.prenorm:
        prenorm = LayerNorm(config.encoder_hidden_size)
    else:
        prenorm = None
    return prenorm


def build_pos_embeds(
        config: HoneybeeVisualProjectorConfig,
        num_input_tokens: int,
        vision_hidden_size: int
):
    # pos emb
    if config.pos_emb:
        pos_emb = torch.nn.Parameter(
            torch.zeros(1, num_input_tokens, vision_hidden_size))
        nn.init.trunc_normal_(pos_emb, mean=0.0, std=0.02)
    else:
        pos_emb = None

    return pos_emb


def build_eos_tokens(config: HoneybeeVisualProjectorConfig,
                     output_hidden_size: int):
    # think tokens
    num_eos_tokens = config.num_eos_tokens
    if num_eos_tokens:
        eos_tokens = torch.nn.Parameter(
            torch.randn(1, num_eos_tokens, output_hidden_size))
        nn.init.trunc_normal_(
            eos_tokens, mean=0.0, std=config.initializer_range)
    else:
        eos_tokens = None

    return eos_tokens
