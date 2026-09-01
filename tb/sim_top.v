// sim_top -- DUT + memory-model wrapper shared by the Verilator harness
// (sim_main.cpp) and the Icarus testbench, so both flows test the same thing.
`default_nettype none

`ifndef NDIM
 `define NDIM 16
`endif
`ifndef MEMINIT
 `define MEMINIT "build/mem_init.hex"
`endif

module sim_top #(
    parameter integer ADDR_W = 20
) (
    input  wire              clk,
    input  wire              rst,
    input  wire              multiply_en,
    output wire              ready,
    input  wire [ADDR_W-1:0] dbg_addr,
    output wire [63:0]       dbg_data,
    output wire [31:0]       n_reads,
    output wire [31:0]       n_writes,
    output wire [31:0]       n_errors
);

    localparam integer N    = `NDIM;
    localparam integer WPR  = N / 4;
    localparam integer NW   = N * WPR;
    localparam integer MEMW = 3 * NW;

    wire              read_en, write_en;
    wire [ADDR_W-1:0] read_addr, write_addr;
    wire [63:0]       read_data, write_data;

    mem_model #(
        .ADDR_W(ADDR_W), .NW(NW), .MEMW(MEMW), .INIT_FILE(`MEMINIT)
    ) u_mem (
        .clk        (clk),
        .read_en    (read_en),
        .read_addr  (read_addr),
        .read_data  (read_data),
        .write_en   (write_en),
        .write_addr (write_addr),
        .write_data (write_data),
        .dbg_addr   (dbg_addr),
        .dbg_data   (dbg_data),
        .n_reads    (n_reads),
        .n_writes   (n_writes),
        .n_errors   (n_errors)
    );

`ifdef USE_4MUL
    matmul_4mul #(.N(N), .ADDR_W(ADDR_W)) dut (
`else
    matmul_fast #(.N(N), .ADDR_W(ADDR_W)) dut (
`endif
        .clk            (clk),
        .rst            (rst),
        .multiply_en    (multiply_en),
        .mem_read_en    (read_en),
        .mem_read_addr  (read_addr),
        .mem_read_data  (read_data),
        .mem_write_en   (write_en),
        .mem_write_addr (write_addr),
        .mem_write_data (write_data),
        .ready          (ready)
    );

endmodule

`default_nettype wire
