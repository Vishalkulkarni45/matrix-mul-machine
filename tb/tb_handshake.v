// tb_handshake -- command-interface tests the one-shot regression cannot
// see: back-to-back run without reset, reset mid-operation, and multiply_en
// held high across completion (level-sensitive restart).
`timescale 1ns/1ps
`default_nettype none

`ifndef NDIM
 `define NDIM 16
`endif
module tb_handshake;

    localparam integer N      = `NDIM;
    localparam integer ADDR_W = 20;
    localparam integer WPR    = N / 4;
    localparam integer NW     = N * WPR;
    localparam [63:0]  POISON = 64'hDEAD_BEEF_DEAD_BEEF;

    reg clk = 1'b0;
    reg rst = 1'b1;
    reg multiply_en = 1'b0;

    always #5 clk = ~clk;

    wire [31:0] n_writes, n_errors;
    wire        ready;

    sim_top #(.ADDR_W(ADDR_W)) u_sys (
        .clk(clk), .rst(rst), .multiply_en(multiply_en), .ready(ready),
        .dbg_addr({ADDR_W{1'b0}}), .dbg_data(),
        .n_reads(), .n_writes(n_writes), .n_errors(n_errors)
    );

    reg [63:0] snapshot [0:NW-1];
    integer errors = 0;
    integer cyc1, cyc2, cyc3, cyc4;

    task poison_c;
        integer j;
        begin
            for (j = 0; j < NW; j = j + 1) u_sys.u_mem.mem[2*NW + j] = POISON;
        end
    endtask

    task snap_c;                       // record C into `snapshot`
        integer j;
        begin
            for (j = 0; j < NW; j = j + 1) snapshot[j] = u_sys.u_mem.mem[2*NW + j];
        end
    endtask

    task check_c(input [80*8:1] who);  // compare C against `snapshot`
        integer j, bad, poisoned;
        begin
            bad = 0; poisoned = 0;
            for (j = 0; j < NW; j = j + 1) begin
                if (u_sys.u_mem.mem[2*NW + j] === POISON) poisoned = poisoned + 1;
                else if (u_sys.u_mem.mem[2*NW + j] !== snapshot[j]) bad = bad + 1;
            end
            if (poisoned != 0 || bad != 0) begin
                errors = errors + 1;
                $display("  FAIL %0s: %0d words never written, %0d differ",
                         who, poisoned, bad);
            end else begin
                $display("  ok   %0s: all %0d words rewritten, identical",
                         who, NW);
            end
        end
    endtask

    // Pulse multiply_en for exactly one rising edge and count until ready.
    task run_op(output integer cycles);
        integer guard;
        begin
            @(negedge clk);
            multiply_en = 1'b1;
            @(posedge clk);
            @(negedge clk);
            multiply_en = 1'b0;
            cycles = 1;
            guard  = 0;
            while (ready !== 1'b1) begin
                @(posedge clk);
                #1;
                cycles = cycles + 1;
                guard  = guard + 1;
                if (guard > 8*(N*N*N/4 + 4096)) begin
                    $display("  FAIL: timeout, ready never returned");
                    errors = errors + 1;
                    disable run_op;
                end
            end
        end
    endtask

    initial begin
        $display("=== handshake tests, N=%0d ===", N);

        repeat (4) @(negedge clk);
        rst = 1'b0;
        @(negedge clk);
        if (ready !== 1'b1) begin
            $display("  FAIL: ready not asserted after reset");
            errors = errors + 1;
        end else $display("  ok   ready asserted after reset");

        // ---------------- RUN 1: baseline ----------------
        run_op(cyc1);
        $display("  RUN1 cycles=%0d writes=%0d mem_errors=%0d",
                 cyc1, n_writes, n_errors);
        snap_c;

        // ---------------- RUN 2: back-to-back ----------------
        // No reset in between. Every result word must be produced again from
        // scratch and match the baseline exactly.
        poison_c;
        run_op(cyc2);
        $display("  RUN2 cycles=%0d (back-to-back, no reset)", cyc2);
        if (cyc2 !== cyc1) begin
            $display("  FAIL: RUN2 took %0d cycles, RUN1 took %0d", cyc2, cyc1);
            errors = errors + 1;
        end else $display("  ok   RUN2 cycle count identical to RUN1");
        check_c("RUN2 back-to-back");

        // ---------------- RUN 3: reset mid-operation ----------------
        poison_c;
        @(negedge clk);
        multiply_en = 1'b1;
        @(posedge clk);
        @(negedge clk);
        multiply_en = 1'b0;
        repeat (N*N/8 + 37) @(negedge clk);   // abort partway through
        rst = 1'b1;
        repeat (3) @(negedge clk);
        rst = 1'b0;
        @(negedge clk);
        if (ready !== 1'b1) begin
            $display("  FAIL: ready not asserted after mid-operation reset");
            errors = errors + 1;
        end else $display("  ok   ready reasserted after mid-operation reset");
        poison_c;
        run_op(cyc3);
        $display("  RUN3 cycles=%0d (after aborted run)", cyc3);
        if (cyc3 !== cyc1) begin
            $display("  FAIL: RUN3 took %0d cycles, RUN1 took %0d", cyc3, cyc1);
            errors = errors + 1;
        end else $display("  ok   RUN3 cycle count identical to RUN1");
        check_c("RUN3 after aborted run");

        // ---------------- RUN 4: multiply_en held high ----------------
        // Level-sensitive semantics: the machine must restart while
        // multiply_en stays high.
        poison_c;
        @(negedge clk);
        multiply_en = 1'b1;
        while (ready !== 1'b0) begin @(posedge clk); #1; end
        // Same convention as run_op(): the acceptance edge counts as cycle 1;
        // the loop above already consumed it.
        cyc4 = 1;
        while (ready !== 1'b1) begin @(posedge clk); #1; cyc4 = cyc4 + 1; end
        $display("  RUN4 cycles=%0d (multiply_en held high)", cyc4);
        if (cyc4 != cyc1) begin
            $display("  FAIL: RUN4 took %0d cycles, RUN1 took %0d", cyc4, cyc1);
            errors = errors + 1;
        end else $display("  ok   RUN4 cycle count identical to RUN1");

        // multiply_en is still high, so the machine must restart on the next
        // edge; releasing it before this check would prove nothing.
        @(posedge clk); #1;
        if (ready !== 1'b0) begin
            $display("  FAIL: multiply_en held high did not restart the machine");
            errors = errors + 1;
        end else $display("  ok   RUN4 restarted while multiply_en was held");

        @(negedge clk);
        multiply_en = 1'b0;             // release during the restarted run
        // Let the restarted operation finish so C is complete again.
        while (ready !== 1'b1) begin @(posedge clk); #1; end
        repeat (4) @(negedge clk);
        check_c("RUN4 multiply_en held high");

        if (n_errors != 0) begin
            $display("  FAIL: memory model reported %0d region violations",
                     n_errors);
            errors = errors + 1;
        end

        if (errors == 0) $display("HANDSHAKE VERDICT: PASS");
        else             $display("HANDSHAKE VERDICT: FAIL (%0d)", errors);
        $finish;
    end

endmodule

`default_nettype wire
