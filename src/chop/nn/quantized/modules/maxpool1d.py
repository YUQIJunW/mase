from functools import partial

import torch
from torch import Tensor
from torch.nn import functional as F

from ..utils import get_stats, quantiser_passthrough

from chop.nn.quantizers import (
    block_fp_quantizer,
    block_log_quantizer,
    block_minifloat_quantizer,
    integer_quantizer,
    log_quantizer,
    minifloat_denorm_quantizer,
    minifloat_ieee_quantizer,
    binary_quantizer,
    ternary_quantizer,
)


class _MaxPool1dBase(torch.nn.MaxPool1d):
    def __init__(self, inplace: bool = False):
        super().__init__(inplace)
        self.bypass = False
        self.x_quantizer = None

    def forward(self, x: Tensor) -> Tensor:
        if self.bypass:
            return F.relu(x)
        else:
            x = self.x_quantizer(x)
            return F.relu(x, self.inplace)

    def get_quantized_output(self, x: Tensor) -> Tensor:
        x = self.x_quantizer(x)
        return {"x": x}


class MaxPool1dInteger(_MaxPool1dBase):
    bypass = None

    def __init__(self, inplace: bool = False, config: dict = None):
        super().__init__(inplace)
        assert config is not None, "config is None!"
        self.config = config
        self.bypass = config.get("bypass", False)
        if self.bypass:
            return
        # establish quantizers
        x_width, x_frac_width = config["data_in_width"], config["data_in_frac_width"]
        self.x_quantizer = partial(
            integer_quantizer, width=x_width, frac_width=x_frac_width, is_signed=False
        )
        self.config = config
        self.x_width = x_width
        self.x_frac_width = x_frac_width

    # def get_output_bitwidth(self) -> dict:
    #     return {
    #         "data_out_width": self.config["data_in_width"],
    #         "data_out_frac_width": self.config["data_in_frac_width"],
    #     }


    bypass = None

    def __init__(self, inplace: bool = False, config: dict = None):
        super().__init__(inplace)
        assert config is not None, "config is None!"
        self.config = config
        self.bypass = config.get("bypass", False)
        if self.bypass:
            return
        # establish quantisers
        x_scaling_factor = config["data_in_scaling_factor"]
        x_mean = get_stats(config, "data_in_mean")
        x_median = get_stats(config, "data_in_median")
        x_max = get_stats(config, "data_in_max")
        self.x_quantizer = quantiser_passthrough
        # self.x_quantizer = partial(
        #     ternary_quantizer,
        #     scaling_factor=x_scaling_factor,
        #     median=x_median,
        #     maximum=x_max,
        #     mean=x_mean,
        # )
        self.config = config
