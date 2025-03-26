`timescale 1ns / 1ps

module adaptive_max_pool2d #(
    /* verilator lint_off UNUSEDPARAM */
    parameter DATA_IN_0_PRECISION_0 = 8,
    parameter DATA_IN_0_PRECISION_1 = 3,
    parameter DATA_IN_0_TENSOR_SIZE_DIM_0 = 8,
    parameter DATA_IN_0_TENSOR_SIZE_DIM_1 = 1,
    parameter DATA_IN_0_TENSOR_SIZE_DIM_2 = 1,
    parameter DATA_IN_0_PARALLELISM_DIM_0 = 1,
    parameter DATA_IN_0_PARALLELISM_DIM_1 = 1,
    parameter DATA_IN_0_PARALLELISM_DIM_2 = 1,

    parameter DATA_IN_0_WIDTH = 4,
    parameter DATA_IN_0_HEIGHT = 4,
    parameter DATA_OUT_0_WIDTH = 2,
    parameter DATA_OUT_0_HEIGHT = 2,
    parameter PADDING = 0,
    parameter KERNEL_WIDTH = 2,
    parameter KERNEL_HEIGHT = 2,
    parameter STRIDE = 2,

    parameter DATA_OUT_0_PRECISION_0 = 8,
    parameter DATA_OUT_0_PRECISION_1 = 3,
    parameter DATA_OUT_0_TENSOR_SIZE_DIM_0 = 8,
    parameter DATA_OUT_0_TENSOR_SIZE_DIM_1 = 1,
    parameter DATA_OUT_0_TENSOR_SIZE_DIM_2 = 1,
    parameter DATA_OUT_0_PARALLELISM_DIM_0 = 1,
    parameter DATA_OUT_0_PARALLELISM_DIM_1 = 1,
    parameter DATA_OUT_0_PARALLELISM_DIM_2 = 1,

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
    else $error("MaxPool1d: DATA_IN_0_PRECISION_0 must be equal to DATA_OUT_0_PRECISION_0");
    assert (DATA_IN_0_PRECISION_1 == DATA_OUT_0_PRECISION_1)
    else $error("MaxPool1d: DATA_IN_0_PRECISION_1 must be equal to DATA_OUT_0_PRECISION_1");
  end

  always_comb begin
  for (int i = 0; i < DATA_OUT_0_HEIGHT;i++) begin
    for (int j = 0; j < DATA_OUT_0_WIDTH; j++) begin
      int max_val = 0;

      for (int m = 0; m < KERNEL_HEIGHT; m++) begin
        for (int n = 0; n < KERNEL_WIDTH; n++) begin
          int row = i*STRIDE + m;
          int col = j*STRIDE + n;

          int index = row * DATA_IN_0_WIDTH + col;

          if (data_in_0[index] > max_val) begin
            max_val = data_in_0[index];
          end
        end
      end
      data_out_0[i * DATA_OUT_0_WIDTH + j] = max_val;
    end
  end
end
      


  assign data_out_0_valid = data_in_0_valid;
  assign data_in_0_ready  = data_out_0_ready;

endmodule
