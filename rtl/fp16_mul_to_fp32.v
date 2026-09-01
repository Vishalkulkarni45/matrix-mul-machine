// fp16_mul_to_fp32 -- binary16 x binary16 -> binary32, combinational.
// A finite fp16 product is always exact in binary32 (22-bit significand
// product, exponent stays normal), so there is no rounding logic here.
`default_nettype none

module fp16_mul_to_fp32 (
    input  wire [15:0] a,
    input  wire [15:0] b,
    output wire [31:0] y
);

    wire        sa = a[15];
    wire        sb = b[15];
    wire [4:0]  ea = a[14:10];
    wire [4:0]  eb = b[14:10];
    wire [9:0]  ma = a[9:0];
    wire [9:0]  mb = b[9:0];

    wire a_zero = (ea == 5'd0)  && (ma == 10'd0);
    wire b_zero = (eb == 5'd0)  && (mb == 10'd0);
    wire a_inf  = (ea == 5'd31) && (ma == 10'd0);
    wire b_inf  = (eb == 5'd31) && (mb == 10'd0);
    wire a_nan  = (ea == 5'd31) && (ma != 10'd0);
    wire b_nan  = (eb == 5'd31) && (mb != 10'd0);

    wire sy = sa ^ sb;

    // Uniform "integer significand x 2^ie" form; the leading-one detector
    // below absorbs subnormals' leading zeros, only `ie` differs:
    //   normal    : {1,ma} * 2^(ea-25)
    //   subnormal : {0,ma} * 2^(-24)
    wire [10:0] siga = (ea == 5'd0) ? {1'b0, ma} : {1'b1, ma};
    wire [10:0] sigb = (eb == 5'd0) ? {1'b0, mb} : {1'b1, mb};

    wire signed [8:0] iea = (ea == 5'd0) ? -9'sd24 : ($signed({4'b0, ea}) - 9'sd25);
    wire signed [8:0] ieb = (eb == 5'd0) ? -9'sd24 : ($signed({4'b0, eb}) - 9'sd25);
    wire signed [8:0] iesum = iea + ieb;

    wire [21:0] prod = siga * sigb;

    // MSB position of `prod` (0..21); last assignment wins, synthesises to a
    // priority encoder.
    reg  [4:0] msb;
    integer    i;
    always @* begin
        msb = 5'd0;
        for (i = 0; i < 22; i = i + 1)
            if (prod[i]) msb = i[4:0];
    end

    // Left-align so the leading one sits at bit 21; the 21 bits below it are
    // the fraction. binary32 wants 23 fraction bits, so pad two zeros. Exact.
    wire [21:0] aligned = prod << (5'd21 - msb);
    wire [22:0] frac32  = {aligned[20:0], 2'b00};

    wire signed [10:0] exp_unb = $signed({{2{iesum[8]}}, iesum}) + $signed({6'b0, msb});
    wire [7:0]         exp32   = exp_unb[7:0] + 8'd127;

    wire prod_zero = a_zero | b_zero;
    wire nan_out   = a_nan | b_nan | (a_inf & b_zero) | (b_inf & a_zero);
    wire inf_out   = (a_inf | b_inf) & ~nan_out;

    assign y = nan_out   ? 32'h7FC0_0000                 // canonical quiet NaN
             : inf_out   ? {sy, 8'hFF, 23'd0}
             : prod_zero ? {sy, 31'd0}
             :             {sy, exp32, frac32};

endmodule

`default_nettype wire
