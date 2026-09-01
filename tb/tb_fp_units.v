// tb_fp_units -- bit-exactness regression for the three FP primitives:
// drive vectors from file, compare against numpy's answer, no tolerance.
`timescale 1ns/1ps
`default_nettype none
`include "fp_counts.vh"

module tb_fp_units;

    integer errors = 0;
    integer i;

    // ---------------- fp16_mul_to_fp32 ----------------
    reg  [63:0] mul_vec [0:`NMUL-1];
    reg  [15:0] m_a, m_b;
    wire [31:0] m_y;
    reg  [31:0] m_exp;
    fp16_mul_to_fp32 u_mul (.a(m_a), .b(m_b), .y(m_y));

    // ---------------- fp32_add, STAGES = 0 ------------
    reg  [95:0] add_vec [0:`NADD-1];
    reg  [31:0] a_a, a_b;
    wire [31:0] a_y;
    reg  [31:0] a_exp;
    fp32_add #(.STAGES(0)) u_add (.clk(clk), .a(a_a), .b(a_b), .y(a_y));

    // ---------------- fp32_add, STAGES = 3 ------------
    // Must return identical bits for the same vectors; a mismatch means a
    // stage boundary was placed wrongly.
    reg         clk = 1'b0;
    reg  [31:0] p3_a, p3_b, p3_exp;
    reg         p3_val = 1'b0;
    wire [31:0] p3_y;
    fp32_add #(.STAGES(3)) u_add3 (.clk(clk), .a(p3_a), .b(p3_b), .y(p3_y));

    // Expected-value shadow, three deep -- exactly the DUT's three registers,
    // so `sh3` and `p3_y` always describe the same input vector.
    reg [31:0] sh1, sh2, sh3;
    reg        sv1 = 1'b0, sv2 = 1'b0, sv3 = 1'b0;
    always @(posedge clk) begin
        sh1 <= p3_exp; sv1 <= p3_val;
        sh2 <= sh1;    sv2 <= sv1;
        sh3 <= sh2;    sv3 <= sv2;
    end

    integer p3_errors = 0;
    always @(negedge clk) if (sv3) begin
        if (p3_y !== sh3) begin
            errors = errors + 1;
            p3_errors = p3_errors + 1;
            if (p3_errors <= 20)
                $display("  FAIL add3 got=%h exp=%h", p3_y, sh3);
        end
    end

    always #1 clk = ~clk;

    // ---------------- fp32_to_fp16 --------------------
    reg  [47:0] cvt_vec [0:`NCVT-1];
    reg  [31:0] c_a;
    wire [15:0] c_y;
    reg  [15:0] c_exp;
    fp32_to_fp16 u_cvt (.a(c_a), .y(c_y));

    // A missing or short vector file leaves entries at X, the DUT outputs X,
    // and `X !== X` is FALSE -- a silent PASS. This guards against that.
    task check_loaded(input [64*8:1] who, input loaded_ok);
        begin
            if (!loaded_ok) begin
                errors = errors + 1;
                $display("  FAIL: %0s vector file missing, short or unreadable"
                         , who);
            end
        end
    endtask

    initial begin
        $readmemh("vectors/fp16mul.hex", mul_vec);
        $readmemh("vectors/fp32add.hex", add_vec);
        $readmemh("vectors/fp32cvt.hex", cvt_vec);

        check_loaded("fp16mul.hex", (^mul_vec[`NMUL-1] !== 1'bx));
        check_loaded("fp32add.hex", (^add_vec[`NADD-1] !== 1'bx));
        check_loaded("fp32cvt.hex", (^cvt_vec[`NCVT-1] !== 1'bx));

        $display("== fp16_mul_to_fp32: %0d vectors ==", `NMUL);
        for (i = 0; i < `NMUL; i = i + 1) begin
            {m_a, m_b, m_exp} = mul_vec[i];
            #1;
            if (m_y !== m_exp) begin
                errors = errors + 1;
                if (errors <= 20)
                    $display("  FAIL mul[%0d] a=%h b=%h got=%h exp=%h",
                             i, m_a, m_b, m_y, m_exp);
            end
        end

        $display("== fp32_add: %0d vectors ==", `NADD);
        for (i = 0; i < `NADD; i = i + 1) begin
            {a_a, a_b, a_exp} = add_vec[i];
            #1;
            if (a_y !== a_exp) begin
                errors = errors + 1;
                if (errors <= 20)
                    $display("  FAIL add[%0d] a=%h b=%h got=%h exp=%h",
                             i, a_a, a_b, a_y, a_exp);
            end
        end

        $display("== fp32_to_fp16: %0d vectors ==", `NCVT);
        for (i = 0; i < `NCVT; i = i + 1) begin
            {c_a, c_exp} = cvt_vec[i];
            #1;
            if (c_y !== c_exp) begin
                errors = errors + 1;
                if (errors <= 20)
                    $display("  FAIL cvt[%0d] a=%h got=%h exp=%h",
                             i, c_a, c_y, c_exp);
            end
        end

        // Same vectors again, now through the three-stage pipeline.
        $display("== fp32_add STAGES=3: %0d vectors ==", `NADD);
        for (i = 0; i < `NADD + 4; i = i + 1) begin
            @(negedge clk);
            if (i < `NADD) begin
                {p3_a, p3_b, p3_exp} = add_vec[i];
                p3_val = 1'b1;
            end else begin
                p3_val = 1'b0;                 // flush the last three results
            end
        end
        @(negedge clk);
        if (p3_errors == 0)
            $display("   pipelined adder bit-identical to combinational");

        if (errors == 0) $display("FP UNIT TESTS: PASS (%0d vectors)",
                                  `NMUL + 2*`NADD + `NCVT);
        else             $display("FP UNIT TESTS: FAIL (%0d mismatches)", errors);
        $finish;
    end

endmodule

`default_nettype wire
