`timescale 1ns / 1ps

module lp_pool1d #(
    /* verilator lint_off UNUSEDPARAM */
    parameter DATA_IN_0_PRECISION_0 = 8,
    parameter DATA_IN_0_PRECISION_1 = 3,
    parameter DATA_IN_0_TENSOR_SIZE_DIM_0 = 8,
    parameter DATA_IN_0_TENSOR_SIZE_DIM_1 = 1,
    parameter DATA_IN_0_PARALLELISM_DIM_0 = 1,
    parameter DATA_IN_0_PARALLELISM_DIM_1 = 1,

    parameter PADDING = 0,
    parameter KERNEL_SIZE = 2,
    parameter STRIDE = 2,
    parameter NORM_TYPE = 2,

    parameter DATA_OUT_0_PRECISION_0 = 8,
    parameter DATA_OUT_0_PRECISION_1 = 3,
    parameter DATA_OUT_0_TENSOR_SIZE_DIM_0 = 8,
    parameter DATA_OUT_0_TENSOR_SIZE_DIM_1 = 1,
    parameter DATA_OUT_0_PARALLELISM_DIM_0 = 1,
    parameter DATA_OUT_0_PARALLELISM_DIM_1 = 1,

    parameter INPLACE = 0
) (
    /* verilator lint_off UNUSEDSIGNAL */
    input rst,
    input clk,
    input logic [DATA_IN_0_PRECISION_0-1:0] data_in_0[DATA_IN_0_PARALLELISM_DIM_0*DATA_IN_0_PARALLELISM_DIM_1-1:0],
    output logic [DATA_OUT_0_PRECISION_0-1:0] data_out_0[DATA_OUT_0_PARALLELISM_DIM_0*DATA_OUT_0_PARALLELISM_DIM_1-1:0],

    input  logic data_in_0_valid,
    output logic data_in_0_ready,
    output logic data_out_0_valid,
    input  logic data_out_0_ready
);

  initial begin
    assert (DATA_IN_0_PRECISION_0 == DATA_OUT_0_PRECISION_0)
    else $error("Lppool1d: DATA_IN_0_PRECISION_0 must be equal to DATA_OUT_0_PRECISION_0");
    assert (DATA_IN_0_PRECISION_1 == DATA_OUT_0_PRECISION_1)
    else $error("Lppool1d: DATA_IN_0_PRECISION_1 must be equal to DATA_OUT_0_PRECISION_1");
  end


  always_comb begin
  for (int i = 0; i < DATA_OUT_0_PARALLELISM_DIM_0; i++) begin
    int base_idx = i * STRIDE;
    shortreal result=0;
    for (int j = 0; j < KERNEL_SIZE; j++) begin
        result = result + $pow(data_in_0[base_idx + j], NORM_TYPE);
    end
    data_out_0[i] = $rtoi($pow(result,1.0/NORM_TYPE));
  end
end

  assign data_out_0_valid = data_in_0_valid;
  assign data_in_0_ready  = data_out_0_ready;

endmodule
