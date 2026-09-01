// mem_model -- external memory: read data valid 3 cycles after read_en,
// fully pipelined, no backpressure. Polices regions (reads in A/B, writes in
// C) and counts writes. dbg_* is a pure readout for the Verilator harness.
`default_nettype none

module mem_model #(
    parameter integer ADDR_W = 20,
    parameter integer DATA_W = 64,
    parameter integer NW     = 65536,      // 64-bit words per matrix
    parameter integer MEMW   = 196608,     // total words modelled (3 * NW)
    parameter         INIT_FILE = "build/mem_init.hex"
) (
    input  wire                clk,
    input  wire                read_en,
    input  wire [ADDR_W-1:0]   read_addr,
    output wire [DATA_W-1:0]   read_data,
    input  wire                write_en,
    input  wire [ADDR_W-1:0]   write_addr,
    input  wire [DATA_W-1:0]   write_data,

    input  wire [ADDR_W-1:0]   dbg_addr,
    output wire [DATA_W-1:0]   dbg_data,
    output wire [31:0]         n_reads,
    output wire [31:0]         n_writes,
    output wire [31:0]         n_errors
);

    function integer clog2;
        input integer v;
        integer i;
        begin
            clog2 = 0;
            for (i = v - 1; i > 0; i = i >> 1) clog2 = clog2 + 1;
        end
    endfunction
    localparam integer MEMAW = clog2(MEMW);

    reg [DATA_W-1:0] mem [0:MEMW-1];

    initial $readmemh(INIT_FILE, mem);

    reg [DATA_W-1:0] pipe0, pipe1, pipe2;
    assign read_data = pipe2;
    assign dbg_data  = mem[dbg_addr[MEMAW-1:0]];

    // Zero-extend before the region compares so the widths are explicit.
    wire [31:0] raddr32 = {{(32-ADDR_W){1'b0}}, read_addr};
    wire [31:0] waddr32 = {{(32-ADDR_W){1'b0}}, write_addr};

    reg [31:0] rd_cnt = 32'd0, wr_cnt = 32'd0, err_cnt = 32'd0;
    assign n_reads  = rd_cnt;
    assign n_writes = wr_cnt;
    assign n_errors = err_cnt;

    always @(posedge clk) begin
        pipe0 <= read_en ? mem[read_addr[MEMAW-1:0]] : {DATA_W{1'bx}};
        pipe1 <= pipe0;
        pipe2 <= pipe1;

        if (read_en) begin
            rd_cnt <= rd_cnt + 32'd1;
            if (raddr32 >= 2*NW) begin
                err_cnt <= err_cnt + 32'd1;
                if (err_cnt < 10)
                    $display("MEM ERROR: read from address %0d is outside A/B",
                             read_addr);
            end
        end

        if (write_en) begin
            wr_cnt <= wr_cnt + 32'd1;
            if (waddr32 < 2*NW || waddr32 >= 3*NW) begin
                err_cnt <= err_cnt + 32'd1;
                if (err_cnt < 10)
                    $display("MEM ERROR: write to address %0d is outside C",
                             write_addr);
            end else begin
                mem[write_addr[MEMAW-1:0]] <= write_data;
            end
        end
    end

endmodule

`default_nettype wire
