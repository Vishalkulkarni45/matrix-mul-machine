// fp32_to_fp16 -- binary32 -> binary16, round-to-nearest-even, combinational.
// One shift (rs = 13 normal grid, -1-e subnormal grid) plus the encoding
// ((e+14)<<10) + q makes every rounding carry correct by construction.
`default_nettype none

module fp32_to_fp16 (
    input  wire [31:0] a,
    output wire [15:0] y
);

    wire        s   = a[31];
    wire [7:0]  e8  = a[30:23];
    wire [22:0] f23 = a[22:0];

    wire a_zero = (e8 == 8'd0);                       // zero or subnormal -> +-0
    wire a_inf  = (e8 == 8'hFF) && (f23 == 23'd0);
    wire a_nan  = (e8 == 8'hFF) && (f23 != 23'd0);

    wire signed [9:0] e = $signed({2'b0, e8}) - 10'sd127;

    wire [23:0] sig = {1'b1, f23};
    wire [31:0] sigx = {8'd0, sig};

    wire signed [10:0] rs_raw = (e >= -10'sd14) ? 11'sd13
                                                : (-11'sd1 - $signed({e[9], e}));
    // Any rs >= 25 makes both q and the guard bit zero, so clamping at 31 is
    // bit-exact and keeps the shifter narrow.
    wire [5:0] rs = (rs_raw > 11'sd31) ? 6'd31 : rs_raw[5:0];

    wire [31:0] shifted = sigx >> rs;
    wire [11:0] q       = shifted[11:0];

    // rs is always in [13, 31], so rs-1 indexes a 32-bit vector in 5 bits.
    wire [4:0]  gpos     = rs[4:0] - 5'd1;
    wire [31:0] low_mask = (32'd1 << gpos) - 32'd1;
    wire g_bit  = sigx[gpos];
    wire s_bit  = |(sigx & low_mask);

    wire round_up = g_bit & (s_bit | q[0]);
    wire [12:0] qf = {1'b0, q} + {12'b0, round_up};

    wire signed [15:0] e_plus14 = $signed({{6{e[9]}}, e}) + 16'sd14;
    wire [15:0] res15 = (e >= -10'sd14) ? ((e_plus14 << 10) + {3'b0, qf})
                                        : {3'b0, qf};

    // Only e >= 16 needs an explicit clamp: on the normal grid res15 tops
    // out at 31744, which is already the binary16 infinity encoding.
    wire overflow = (e >= 10'sd16);

    assign y = a_nan   ? 16'h7E00                  // canonical quiet NaN
             : a_inf   ? {s, 5'h1F, 10'd0}
             : a_zero  ? {s, 15'd0}
             : overflow? {s, 5'h1F, 10'd0}
             :           {s, res15[14:0]};

endmodule

`default_nettype wire
