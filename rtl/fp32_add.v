// fp32_add -- IEEE-754 binary32 addition, round-to-nearest-even. STAGES = 0
// (combinational) or 3 (pipelined, one result/cycle); same arithmetic either
// way, bit-identical. Subnormals flush to zero (DAZ/FTZ) -- unreachable here.
`default_nettype none

module fp32_add #(
    parameter integer STAGES = 0        // 0 = combinational, 3 = pipelined
) (
    input  wire        clk,             // unused when STAGES == 0
    input  wire [31:0] a,
    input  wire [31:0] b,
    output wire [31:0] y
);

    initial begin
        if ((STAGES != 0) && (STAGES != 3)) begin
            $display("FATAL fp32_add: STAGES must be 0 or 3 (got %0d)", STAGES);
            $finish;
        end
    end

    // =======================================================================
    // STAGE 1 -- classify, align, add
    // =======================================================================
    wire        sa = a[31], sb = b[31];
    wire [7:0]  ea = a[30:23], eb = b[30:23];
    wire [22:0] fa = a[22:0],  fb = b[22:0];

    // DAZ: exponent field 0 means zero or subnormal; both are treated as zero.
    wire a_zero = (ea == 8'd0);
    wire b_zero = (eb == 8'd0);
    wire a_inf  = (ea == 8'hFF) && (fa == 23'd0);
    wire b_inf  = (eb == 8'hFF) && (fb == 23'd0);
    wire a_nan  = (ea == 8'hFF) && (fa != 23'd0);
    wire b_nan  = (eb == 8'hFF) && (fb != 23'd0);

    // IEEE-754 magnitude ordering coincides with unsigned integer ordering of
    // the sign-cleared bit pattern, so one 31-bit compare does the job.
    wire a_is_larger = (a[30:0] >= b[30:0]);

    wire        sl = a_is_larger ? sa : sb;
    wire        ss = a_is_larger ? sb : sa;
    wire [7:0]  el = a_is_larger ? ea : eb;
    wire [7:0]  es = a_is_larger ? eb : ea;
    wire [22:0] fl = a_is_larger ? fa : fb;
    wire [22:0] fs = a_is_larger ? fb : fa;

    // No zero-mux on sig_s: a zero (or DAZ'd subnormal) operand returns via
    // the special-case chain below, so the mux would be dead logic.
    wire [23:0] sig_l = {1'b1, fl};
    wire [23:0] sig_s = {1'b1, fs};

    // Shifts > 33 clamp to 33: the small operand is then entirely inside the
    // sticky window, so the clamp is bit-exact.
    wire [8:0]  shift_raw = {1'b0, el} - {1'b0, es};
    wire [5:0]  shift     = (shift_raw > 9'd33) ? 6'd33 : shift_raw[5:0];

    //   bit 56 : carry room
    //   bits 55:32 : 24-bit significand
    //   bit 31 : guard,  bit 30 : round,  bits 29:0 : sticky window
    wire [56:0] op_l    = {1'b0, sig_l, 32'd0};
    wire [56:0] op_s    = {1'b0, sig_s, 32'd0};
    wire [56:0] aligned = op_s >> shift;

    wire eff_sub = sl ^ ss;
    wire [56:0] sum_w = eff_sub ? (op_l - aligned) : (op_l + aligned);

    // Every case that does not depend on `sum` is resolved here and carried
    // forward as a {valid, value} pair.
    reg         spec_v_w;
    reg  [31:0] spec_y_w;
    always @* begin
        spec_v_w = 1'b1;
        if (a_nan | b_nan | (a_inf & b_inf & (sa ^ sb)))
                                        spec_y_w = 32'h7FC0_0000;
        else if (a_inf)                 spec_y_w = {sa, 8'hFF, 23'd0};
        else if (b_inf)                 spec_y_w = {sb, 8'hFF, 23'd0};
        else if (a_zero & b_zero)       spec_y_w = {sa & sb, 31'd0};  // (+0)+(-0) = +0
        else if (a_zero)                spec_y_w = b;
        else if (b_zero)                spec_y_w = a;
        else begin spec_v_w = 1'b0;     spec_y_w = 32'd0; end
    end

    // stage-1 payload: sum | sl | el | spec_v | spec_y
    localparam integer W1 = 57 + 1 + 8 + 1 + 32;
    wire [W1-1:0] s1_w = {sum_w, sl, el, spec_v_w, spec_y_w};
    wire [W1-1:0] s1_q;

    // =======================================================================
    // STAGE 2 -- normalise
    // =======================================================================
    wire [56:0] sum2   = s1_q[W1-1 -: 57];
    wire        sl2    = s1_q[41];
    wire [7:0]  el2    = s1_q[40:33];
    wire        spec_v2 = s1_q[32];
    wire [31:0] spec_y2 = s1_q[31:0];

    // Leading-one position of `sum2` (0..56); last write wins.
    reg  [5:0] p;
    integer    i;
    always @* begin
        p = 6'd0;
        for (i = 0; i < 57; i = i + 1)
            if (sum2[i]) p = i[5:0];
    end

    wire sum_zero_w = (sum2 == 57'd0);

    // Renormalise so the leading one sits at bit 55. Only p == 56 (carry out
    // of an effective addition) needs a right shift, by one, and the
    // shifted-out bit is provably always zero, so no sticky capture.
    wire [56:0] norm = (p >= 6'd55) ? (sum2 >> (p - 6'd55)) : (sum2 << (6'd55 - p));
    wire signed [10:0] exp_adj_w = $signed({5'b0, p}) - 11'sd55;

    wire [23:0] sig_n_w = norm[55:32];
    wire        g_bit_w = norm[31];
    wire        r_bit_w = norm[30];
    wire        s_bit_w = |norm[29:0];

    // stage-2 payload: sig_n | g | r | s | exp_adj | sl | el | spec_v | spec_y | sum_zero
    localparam integer W2 = 24 + 1 + 1 + 1 + 11 + 1 + 8 + 1 + 32 + 1;
    wire [W2-1:0] s2_w = {sig_n_w, g_bit_w, r_bit_w, s_bit_w, exp_adj_w,
                          sl2, el2, spec_v2, spec_y2, sum_zero_w};
    wire [W2-1:0] s2_q;

    // =======================================================================
    // STAGE 3 -- round and encode
    // =======================================================================
    wire [23:0] sig_n = s2_q[W2-1 -: 24];
    wire        g_bit = s2_q[56];
    wire        r_bit = s2_q[55];
    wire        s_bit = s2_q[54];
    wire signed [10:0] exp_adj = $signed(s2_q[53:43]);
    wire        sl3   = s2_q[42];
    wire [7:0]  el3   = s2_q[41:34];
    wire        spec_v = s2_q[33];
    wire [31:0] spec_y = s2_q[32:1];
    wire        sum_zero = s2_q[0];

    wire round_up = g_bit & (r_bit | s_bit | sig_n[0]);
    wire [24:0] sig_r = {1'b0, sig_n} + {24'b0, round_up};
    wire        rcarry = sig_r[24];
    wire [23:0] sig_f  = rcarry ? sig_r[24:1] : sig_r[23:0];

    wire signed [11:0] exp_f = $signed({4'b0, el3}) + $signed({exp_adj[10], exp_adj})
                             + $signed({11'b0, rcarry});

    wire ovf = (exp_f >= 12'sd255);
    wire unf = (exp_f <= 12'sd0);        // FTZ, see header

    reg [31:0] y_w;
    always @* begin
        if (spec_v)                     y_w = spec_y;
        else if (sum_zero)              y_w = 32'd0;              // exact cancellation
        else if (ovf)                   y_w = {sl3, 8'hFF, 23'd0};
        else if (unf)                   y_w = {sl3, 31'd0};
        else                            y_w = {sl3, exp_f[7:0], sig_f[22:0]};
    end

    // =======================================================================
    // The stage boundaries: registers when STAGES == 3, wires when 0.
    // =======================================================================
    generate
        if (STAGES == 3) begin : g_pipe
            reg [W1-1:0] r1;
            reg [W2-1:0] r2;
            reg [31:0]   r3;
            always @(posedge clk) begin
                r1 <= s1_w;
                r2 <= s2_w;
                r3 <= y_w;
            end
            assign s1_q = r1;
            assign s2_q = r2;
            assign y    = r3;
        end else begin : g_comb
            assign s1_q = s1_w;
            assign s2_q = s2_w;
            assign y    = y_w;
            /* verilator lint_off UNUSED */
            wire _unused_clk = clk;
            /* verilator lint_on UNUSED */
        end
    endgenerate

endmodule

`default_nettype wire
