// tb_matmul -- Icarus testbench for both machines (-DUSE_4MUL, -DNDIM=<n>).
// Asserts ready behaviour, zero region violations, exactly N^2/4 words
// written, none while ready is high; prints one verdict line run.sh greps.
`timescale 1ns/1ps
`default_nettype none

`ifndef NDIM
 `define NDIM 16
`endif
`ifndef COUT
 `define COUT "build/c_out.hex"
`endif

module tb_matmul;

    localparam integer N      = `NDIM;
    localparam integer ADDR_W = 20;
    localparam integer WPR    = N / 4;
    localparam integer NW     = N * WPR;

    reg clk = 1'b0;
    reg rst = 1'b1;
    reg multiply_en = 1'b0;

    reg  [ADDR_W-1:0] dbg_addr = {ADDR_W{1'b0}};
    wire [63:0]       dbg_data;
    wire [31:0]       n_reads, n_writes, n_errors;
    wire              ready;

    always #5 clk = ~clk;

    sim_top #(.ADDR_W(ADDR_W)) u_sys (
        .clk         (clk),
        .rst         (rst),
        .multiply_en (multiply_en),
        .ready       (ready),
        .dbg_addr    (dbg_addr),
        .dbg_data    (dbg_data),
        .n_reads     (n_reads),
        .n_writes    (n_writes),
        .n_errors    (n_errors)
    );

    integer errors = 0;
    integer cycles = 0;
    integer fd, i;

    // Protocol monitor: a write must never be in flight once the machine has
    // declared itself done.
    integer late_writes = 0;
    always @(posedge clk)
        if (!rst && ready && u_sys.write_en) late_writes = late_writes + 1;

    // 8x the analytic worst case. Computed in 64 bits: at N=1024 the 32-bit
    // form overflows to a negative number and the guard fires immediately.
    localparam [63:0] TIMEOUT_CYCLES = 64'd8 * N * N * N / 64'd4 + 64'd32768;
    reg [63:0] timeout;

    task fail(input [64*8:1] why);
        begin
            errors = errors + 1;
            $display("  TB ERROR: %0s", why);
        end
    endtask

    initial begin
        repeat (4) @(negedge clk);
        rst = 1'b0;
        @(negedge clk);

        if (ready !== 1'b1) fail("ready is not asserted after reset");

        // One-cycle multiply_en pulse straddling exactly one rising edge.
        multiply_en = 1'b1;
        @(posedge clk);
        @(negedge clk);
        multiply_en = 1'b0;
        cycles = 1;

        if (ready !== 1'b0) fail("ready did not deassert after multiply_en");

        // Sample `ready` a delta after the edge: it is a continuous assign off
        // a register, so reading it in the same instant as @(posedge clk)
        // returns the pre-edge value and over-counts by one.
        timeout = 64'd0;
        while ((ready !== 1'b1) && (timeout <= TIMEOUT_CYCLES)) begin
            @(posedge clk);
            #1;
            cycles  = cycles + 1;
            timeout = timeout + 64'd1;
        end
        if (timeout > TIMEOUT_CYCLES) fail("timed out waiting for ready");

        if (n_errors !== 32'd0)  fail("memory model reported region violations");
        if (n_writes !== NW[31:0]) fail("wrong number of writes to the C region");

        // The monitor increments on the posedge AFTER the offending write,
        // so give it a window before checking.
        repeat (16) @(posedge clk);
        #1;
        if (late_writes != 0)    fail("a write was issued while ready was high");

        $display("RESULT N=%0d cycles=%0d reads=%0d writes=%0d mem_errors=%0d late_writes=%0d",
                 N, cycles, n_reads, n_writes, n_errors, late_writes);

        // Dump C before announcing the verdict, so a verdict line always
        // corresponds to a file that this run actually produced.
        fd = $fopen(`COUT, "w");
        if (fd == 0) fail("could not open the C dump file");
        else begin
            for (i = 0; i < NW; i = i + 1) begin
                dbg_addr = 2*NW + i;
                #1 $fwrite(fd, "%016h\n", dbg_data);
            end
            $fclose(fd);
        end

        if (errors == 0) $display("TB VERDICT: PASS");
        else             $display("TB VERDICT: FAIL (%0d checks failed)", errors);
        $finish;
    end

endmodule

`default_nettype wire
