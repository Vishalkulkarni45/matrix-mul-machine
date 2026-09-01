// tb_cycles -- schedule check of matmul_fast at the real N with LANE_STUB
// (no control signal depends on a lane): cycles vs N^2/2 + N/4 + 28, every
// read address, every write. Data values are tb_matmul's job at N <= 256.
`default_nettype none
`timescale 1ns/1ps

`ifndef NDIM
 `define NDIM 1024
`endif

module tb_cycles;

    localparam integer N   = `NDIM;
    localparam integer WPR = N / 4;
    localparam integer NW  = N * WPR;          // 64-bit words per matrix
    localparam integer EXP_CYCLES = N*N/2 + N/4 + 28;
    localparam integer TIMEOUT = EXP_CYCLES + 1000;

    reg  clk = 1'b0, rst = 1'b1, multiply_en = 1'b0;
    wire rd_en, wr_en, ready;
    wire [19:0] rd_addr, wr_addr;

    // No memory pipe modelled: the DUT's return timing comes entirely from
    // its own rv_sr shift register, and the stub consumes no read data.

    matmul_fast #(.N(N)) dut (
        .clk            (clk),
        .rst            (rst),
        .multiply_en    (multiply_en),
        .mem_read_en    (rd_en),
        .mem_read_addr  (rd_addr),
        .mem_read_data  (64'd0),
        .mem_write_en   (wr_en),
        .mem_write_addr (wr_addr),
        .mem_write_data (),
        .ready          (ready)
    );

    integer reads = 0, writes = 0, errs = 0, cycles = 0;
    reg     wrote [0:NW-1];                    // C coverage bitmap
    integer i;

    always #1 clk = ~clk;

    task fail(input [1023:0] why);
        begin errs = errs + 1; if (errs <= 5) $display("TB ERROR: %0s", why); end
    endtask

    always @(posedge clk) begin
        if (rd_en) begin
            reads = reads + 1;
            // Phase 1: B[0..NW) at B_BASE = NW; phase 2: A[0..NW) at 0.
            // Reported through fail() so the five-message cap applies.
            if (reads <= NW) begin
                if (rd_addr !== NW + (reads - 1)) fail("B read address wrong");
            end else if (reads <= 2*NW) begin
                if (rd_addr !== (reads - 1 - NW)) fail("A read address wrong");
            end else
                fail("more reads issued than 2*NW");
        end

        if (wr_en) begin
            writes = writes + 1;
            if (ready === 1'b1)                        fail("write issued while ready is high");
            if (wr_addr < 2*NW || wr_addr >= 3*NW)     fail("write outside the C region");
            else if (wrote[wr_addr - 2*NW] === 1'b1)   fail("C word written twice");
            else wrote[wr_addr - 2*NW] = 1'b1;
        end
    end

    initial begin
        for (i = 0; i < NW; i = i + 1) wrote[i] = 1'b0;

        repeat (4) @(posedge clk);
        rst = 1'b0;
        @(negedge clk);
        if (ready !== 1'b1) fail("ready is not asserted after reset");

        // One-cycle pulse straddling exactly one rising edge, asserted from a
        // falling edge so there is no race with the DUT's own posedge block.
        multiply_en = 1'b1;
        @(posedge clk);
        @(negedge clk);
        multiply_en = 1'b0;
        cycles = 1;
        if (ready !== 1'b0) fail("ready did not deassert after multiply_en");

        // `ready` is a continuous assign off a register: sample a delta after
        // the edge or it returns the pre-edge value and over-counts by one.
        while ((ready !== 1'b1) && (cycles <= TIMEOUT)) begin
            @(posedge clk);
            #1;
            cycles = cycles + 1;
        end
        if (cycles > TIMEOUT) fail("timed out waiting for ready");

        for (i = 0; i < NW; i = i + 1)
            if (wrote[i] !== 1'b1) begin
                errs = errs + 1;
                if (errs <= 5) $display("TB ERROR: C word %0d never written", i);
            end

        if (cycles !== EXP_CYCLES) fail("cycle count does not match N^2/2 + N/4 + 28");
        if (reads  !== 2*NW)       fail("wrong number of reads");
        if (writes !== NW)         fail("wrong number of writes");

        $display("CYCLES N=%0d cycles=%0d (formula %0d) reads=%0d writes=%0d errors=%0d",
                 N, cycles, EXP_CYCLES, reads, writes, errs);
        $display("CYCLE HARNESS VERDICT: %0s", (errs == 0) ? "PASS" : "FAIL");
        $finish;
    end

endmodule

`default_nettype wire
