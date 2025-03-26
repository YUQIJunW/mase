import torch

from torch import nn
from torch.autograd.function import InplaceFunction

import cocotb

from cocotb.triggers import Timer

import pytest
import cocotb
import numpy as np

from mase_cocotb.runner import mase_runner


# snippets
class MyClamp(InplaceFunction):
    @staticmethod
    def forward(ctx, input, min, max):
        return input.clamp(min=min, max=max)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input


class MyRound(InplaceFunction):
    @staticmethod
    def forward(ctx, input):
        ctx.input = input
        return input.round()

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input


# wrap through module to make it a function
my_clamp = MyClamp.apply
my_round = MyRound.apply


# fixed-point quantization with a bias
def quantize(x, bits, bias):  # bits = 32
    """Do linear quantization to input according to a scale and number of bits"""
    thresh = 2 ** (bits - 1)
    scale = 2**bias
    return my_clamp(my_round(x.mul(scale)), -thresh, thresh - 1).div(scale)


class VerificationCase:
    bitwidth = 32
    bias = 4
    num = 8
    out_num = 3
    stride = int(num/out_num)
    kernal_size = int(num-stride*(out_num-1))

    def __init__(self, samples=2):
        self.m = nn.AvgPool1d(kernel_size=self.kernal_size, stride=self.stride)
        self.inputs, self.outputs = [], []
        for _ in range(samples):
            i, o = self.single_run()
            self.inputs.append(i)
            self.outputs.append(o)
        self.samples = samples

    def single_run(self):
        xs = torch.rand(1,self.num)
        r1, r2 = 4, -4
        xs = (r1 - r2) * xs + r2
        # 8-bit, (5, 3)
        xs = quantize(xs, self.bitwidth, self.bias)
        xs = torch.abs(xs)
        return xs[0], self.m(xs)

    def get_dut_parameters(self):
        return {
            "DATA_IN_0_PARALLELISM_DIM_0": self.num,
            "DATA_OUT_0_PARALLELISM_DIM_0": self.out_num,
            "DATA_IN_0_PRECISION_0": self.bitwidth,
            "DATA_OUT_0_PRECISION_0": self.bitwidth,
            "KERNEL_SIZE": self.kernal_size,
            "STRIDE": self.stride,
        }

    def get_dut_input(self, i):
        inputs = self.inputs[i]
        shifted_integers = (inputs * (2**self.bias)).int()
        return shifted_integers.numpy().tolist()

    def get_dut_output(self, i):
        outputs = self.outputs[i]
        shifted_integers = self.convert_to_fixed(outputs)
        return shifted_integers

    def convert_to_fixed(self, x):
        return (x * (2**self.bias)).int().numpy().tolist()

def twos_complement_to_signed(val, bits=32):
    if val & (1 << (bits - 1)):
        val -= (1 << bits)
    return val


@cocotb.test()
async def cocotb_test_adaptiveavgpool1d(dut):
    test_case = VerificationCase(samples=100)

    # set inputs outputs
    for i in range(test_case.samples):
        x = test_case.get_dut_input(i)
        y = test_case.get_dut_output(i)

        dut.data_in_0.value = x
        await Timer(2, units="ns")
        dut_out = dut.data_out_0.value
        dut_out = [x.integer for x in dut.data_out_0.value]
        dut_out = np.array(dut_out)
        for i in range(len(dut_out)):
            dut_out[i] = twos_complement_to_signed(dut_out[i], 32)
        out = dut_out.tolist()
        out = np.array(out)
        print(out)
        print(y[0])
        assert np.allclose(out, y[0]), f"output q was incorrect on the {i}th cycle"
        # assert dut == y[0], f"output q was incorrect on the {i}th cycle"


import pytest


def test_adaptiveavgpool1d():
    tb = VerificationCase()
    mase_runner(module_param_list=[tb.get_dut_parameters()])


if __name__ == "__main__":
    test_adaptiveavgpool1d()
