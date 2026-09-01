// matmul_fast -- machine #1: "as fast as possible, area is not a concern".
// Loads B into per-column banks (N^2/4 cycles), then streams A once at 4*N
// MACs/word, draining C rows concurrently. Cycles: N^2/2 + N/4 + 28.
`default_nettype none

module matmul_fast #(
    parameter integer N      = 1024,   // matrix dimension, power of two, >= 16
    parameter integer ADDR_W = 20,
    parameter integer DATA_W = 64
) (
    input  wire                clk,
    input  wire                rst,          // synchronous, active high
    input  wire                multiply_en,

    output wire                mem_read_en,
    output wire [ADDR_W-1:0]   mem_read_addr,
    input  wire [DATA_W-1:0]   mem_read_data, // valid 3 cycles after read_en

    output reg                 mem_write_en,
    output reg  [ADDR_W-1:0]   mem_write_addr,
    output reg  [DATA_W-1:0]   mem_write_data,

    output wire                ready
);

    function integer clog2;
        input integer v;
        integer i;
        begin
            clog2 = 0;
            for (i = v - 1; i > 0; i = i >> 1) clog2 = clog2 + 1;
        end
    endfunction

    // -----------------------------------------------------------------------
    // ADD_LAT   fp32_add's pipeline depth
    // NPART     interleaved partial accumulators per lane. A partial is
    //           touched once every NPART cycles, so NPART > ADD_LAT is what
    //           makes a pipelined adder legal in the loop-carried path.
    // ACC_RD    delay from "A word present" to the accumulate adder's inputs
    // ACC_WR    ... to its result being written back  (= ACC_RD + ADD_LAT)
    // CAP_LAT   ... to the row total being ready: one cycle past ACC_WR all
    //           NPART partials hold this row's finals, plus 2*ADD_LAT for
    //           the combine tree
    // -----------------------------------------------------------------------
    localparam integer ADD_LAT = 3;
    localparam integer NPART   = 4;
    localparam integer ACC_RD  = 8;
    localparam integer ACC_WR  = ACC_RD + ADD_LAT;              // 11
    localparam integer CAP_LAT = ACC_WR + 1 + 2 * ADD_LAT;      // 18

    localparam integer EPW     = 4;              // binary16 elements per word
    localparam integer WPR     = N / EPW;        // 64-bit words per matrix row
    localparam integer NW      = N * WPR;        // 64-bit words per matrix
    localparam integer A_BASE  = 0;
    localparam integer B_BASE  = NW;
    localparam integer C_BASE  = 2 * NW;
    localparam integer LOG_N   = clog2(N);
    localparam integer LOG_WPR = clog2(WPR);     // == LOG_N - 2
    localparam integer LOG_NW  = clog2(NW);

    // N >= 16 (not 8): a row must contain at least NPART k-groups
    // (WPR >= NPART), or some partial would never be initialised for the row
    // and would carry the previous row's value into the result.
    initial begin
        if ((N < 16) || ((N & (N - 1)) != 0)) begin
            $display("FATAL matmul_fast: N must be a power of two >= 16 (got %0d)", N);
            $finish;
        end
    end

    localparam [2:0] S_IDLE   = 3'd0,
                     S_LOADB  = 3'd1,
                     S_GAP    = 3'd2,
                     S_STREAM = 3'd3,
                     S_FINISH = 3'd4;

    reg [2:0]        state;
    reg [LOG_NW:0]   rd_issue;   // read index within the current phase
    reg [LOG_NW:0]   rd_ret;     // return index within the current phase
    reg [2:0]        gap_cnt;
    reg [2:0]        rv_sr;      // models the 3-cycle read latency

    assign ready = (state == S_IDLE);

    // -----------------------------------------------------------------------
    // Read issue. One request every cycle in both streaming phases: the read
    // port is the bottleneck, so it never idles. Address arithmetic at full
    // width, truncated once, so it is correct for every legal N.
    // -----------------------------------------------------------------------
    wire [31:0] rd_addr_full = ((state == S_LOADB) ? B_BASE : A_BASE)
                             + {{(31-LOG_NW){1'b0}}, rd_issue};

    assign mem_read_en   = (state == S_LOADB) || (state == S_STREAM);
    assign mem_read_addr = rd_addr_full[ADDR_W-1:0];

    wire rvalid = rv_sr[2];
    wire ret_b  = (state == S_LOADB)  || (state == S_GAP);
    wire ret_a  = (state == S_STREAM) || (state == S_FINISH);

    // -----------------------------------------------------------------------
    // Phase 1 decode: returning word index w -> B[k][4*jg .. 4*jg+3]
    // (row-major B, four consecutive columns per word).
    // -----------------------------------------------------------------------
    wire [LOG_N-1:0]   b_k    = rd_ret[LOG_NW-1:LOG_WPR];
    wire [LOG_WPR-1:0] b_jg   = rd_ret[LOG_WPR-1:0];
    wire [LOG_WPR-1:0] b_wadr = b_k[LOG_N-1:2];   // four k values share a word
    wire [1:0]         b_lane = b_k[1:0];
    wire               b_we   = rvalid & ret_b;

    // -----------------------------------------------------------------------
    // Phase 2 decode: returning word index w -> A[i][4*kg .. 4*kg+3].
    // `kg` doubles as the B-bank read address: all N banks are addressed
    // identically every cycle, i.e. they act as one wide memory.
    // -----------------------------------------------------------------------
    wire [LOG_WPR-1:0] a_kg    = rd_ret[LOG_WPR-1:0];
    wire               a_v     = rvalid & ret_a;
    wire               a_last  = (a_kg == WPR[LOG_WPR-1:0] - 1'b1);

    // Partial `j` is initialised (not accumulated into) by the row's k-group
    // j, so the first NPART groups of every row load. Full-width compare:
    // LOG_WPR is 2 at N = 16, where [LOG_WPR-1:2] would be a reverse range.
    wire [31:0]        a_kg32   = {{(32-LOG_WPR){1'b0}}, a_kg};
    wire               a_pfirst = (a_kg32 < NPART);

    wire [LOG_WPR-1:0] bank_raddr = a_kg;

    wire [31:0] b_jg32 = {{(32-LOG_WPR){1'b0}}, b_jg};

    // -----------------------------------------------------------------------
    // Control pipeline: one shared shift register carries valid / last /
    // load-this-partial / which-partial alongside the data.
    //   stage ACC_RD  the accumulate adder samples part[] and s_q
    //   stage ACC_WR  its result is written back into part[]
    //   stage CAP_LAT the row total leaves the combine tree
    // -----------------------------------------------------------------------
    reg [DATA_W-1:0]     a_q;
    reg [CAP_LAT:1]      v_sr, l_sr, pf_sr;
    reg [2*ACC_WR-1:0]   px_sr;      // two bits of partial index per stage

    always @(posedge clk) begin
        a_q   <= mem_read_data;
        v_sr  <= {v_sr [CAP_LAT-1:1], a_v};
        l_sr  <= {l_sr [CAP_LAT-1:1], a_last};
        pf_sr <= {pf_sr[CAP_LAT-1:1], a_pfirst};
        px_sr <= {px_sr[2*ACC_WR-3:0], a_kg[1:0]};
        if (rst) begin
            v_sr  <= {CAP_LAT{1'b0}};
            l_sr  <= {CAP_LAT{1'b0}};
            pf_sr <= {CAP_LAT{1'b0}};
            px_sr <= {(2*ACC_WR){1'b0}};
        end
    end

    // No valid gate on the read side: the adder computes every cycle and the
    // write port discards the junk.
    wire [1:0] rd_px   = px_sr[2*ACC_RD-1 -: 2];
    wire       wr_v    = v_sr[ACC_WR];
    wire       wr_pf   = pf_sr[ACC_WR];
    wire [1:0] wr_px   = px_sr[2*ACC_WR-1 -: 2];
    wire       cap     = v_sr[CAP_LAT] & l_sr[CAP_LAT];

    // -----------------------------------------------------------------------
    // N compute lanes. Lane j owns column j of B and accumulator C[i][j].
    // -----------------------------------------------------------------------
    wire [16*N-1:0] row_fp16;

    // `LANE_STUB` removes the lanes and nothing else. No control signal is
    // derived from a lane output, so a stubbed build reproduces the real
    // cycle count, read order and write schedule at the real N without
    // elaborating 12,288 arithmetic instances (see tb/tb_cycles.v). Data
    // values are tb/tb_matmul.v's job at N <= 256.
`ifdef LANE_STUB
    assign row_fp16 = {N{16'hBEEF}};
`else
    genvar g, c;
    generate
        for (g = 0; g < N; g = g + 1) begin : lane
            // ---- B bank for column j = g -------------------------------
            // Entry `kw` holds B[4kw+0..3][g]. A memory word carries one k
            // for four adjacent columns, so a load writes a single 16-bit
            // lane in each of four banks.
            localparam integer MY_JG = g >> 2;   // column group of lane g
            reg [DATA_W-1:0] bmem [0:WPR-1];
            reg [DATA_W-1:0] brd;

            always @(posedge clk) begin
                if (b_we && (b_jg32 == MY_JG))
                    bmem[b_wadr][b_lane*16 +: 16] <= mem_read_data[(g % 4)*16 +: 16];
                brd <= bmem[bank_raddr];
            end

            // ---- 4 exact binary16 multiplies ----------------------------
            wire [127:0] pw;
            for (c = 0; c < 4; c = c + 1) begin : mul
                fp16_mul_to_fp32 u_mul (
                    .a (a_q[16*c +: 16]),
                    .b (brd[16*c +: 16]),
                    .y (pw[32*c +: 32])
                );
            end
            reg [127:0] p_q;
            always @(posedge clk) p_q <= pw;

            // ---- balanced 4-input adder tree ----------------------------
            wire [31:0] s01_q, s23_q, s_q;
            fp32_add #(.STAGES(ADD_LAT)) u_a01
                (.clk(clk), .a(p_q[31:0]),  .b(p_q[63:32]),  .y(s01_q));
            fp32_add #(.STAGES(ADD_LAT)) u_a23
                (.clk(clk), .a(p_q[95:64]), .b(p_q[127:96]), .y(s23_q));
            fp32_add #(.STAGES(ADD_LAT)) u_as
                (.clk(clk), .a(s01_q), .b(s23_q), .y(s_q));

            // ---- NPART interleaved partial accumulators -----------------
            // A single accumulator would be a one-cycle loop-carried path
            // through the pipelined adder. Rotating over NPART = 4 partials
            // touches each once every four cycles, so the ADD_LAT = 3 adder
            // retires with a cycle to spare.
            //
            // `s_q` is delayed to match the adder because a partial's first
            // term LOADS it. Adding to zero instead is NOT bit-identical:
            // 0 + (-0.0) = +0.0 under round-to-nearest-even.
            reg [31:0] s_d [0:ADD_LAT-1];
            integer sd;
            always @(posedge clk) begin
                s_d[0] <= s_q;
                for (sd = 1; sd < ADD_LAT; sd = sd + 1) s_d[sd] <= s_d[sd-1];
            end

            reg  [31:0] part [0:NPART-1];
            wire [31:0] acc_add;
            fp32_add #(.STAGES(ADD_LAT)) u_acc
                (.clk(clk), .a(part[rd_px]), .b(s_q), .y(acc_add));
            always @(posedge clk)
                if (wr_v) part[wr_px] <= wr_pf ? s_d[ADD_LAT-1] : acc_add;

            // ---- combine the partials -----------------------------------
            // The tree is pipelined and fed continuously; `cap` samples it on
            // the one cycle its output is this row's four finals.
            wire [31:0] c01, c23, row_sum;
            fp32_add #(.STAGES(ADD_LAT)) u_c01
                (.clk(clk), .a(part[0]), .b(part[1]), .y(c01));
            fp32_add #(.STAGES(ADD_LAT)) u_c23
                (.clk(clk), .a(part[2]), .b(part[3]), .y(c23));
            fp32_add #(.STAGES(ADD_LAT)) u_cf
                (.clk(clk), .a(c01), .b(c23), .y(row_sum));

            // ---- round to binary16 for writeback ------------------------
            fp32_to_fp16 u_cvt (.a(row_sum), .y(row_fp16[16*g +: 16]));
        end
    endgenerate
`endif

    // -----------------------------------------------------------------------
    // Writeback. A row of C is captured whole into one of two buffers, then
    // drained WPR words at a time. Captures land exactly every WPR cycles and
    // a drain takes WPR cycles, so they never collide.
    // -----------------------------------------------------------------------
    reg [16*N-1:0]  rowbuf [0:1];
    reg             wsel, drain_buf, drain_active;
    reg [LOG_WPR:0] drain_cnt;
    reg [LOG_N-1:0] drain_row;
    reg [LOG_N:0]   rows_done;   // also supplies the next row index

    wire [31:0] wr_addr_full = C_BASE + drain_row * WPR
                             + {{(31-LOG_WPR){1'b0}}, drain_cnt};

    always @* begin
        mem_write_en   = drain_active;
        mem_write_addr = wr_addr_full[ADDR_W-1:0];
        mem_write_data = rowbuf[drain_buf][drain_cnt*64 +: 64];
    end

    always @(posedge clk) begin
        if (rst) begin
            wsel <= 1'b0; drain_active <= 1'b0; drain_cnt <= 0;
            drain_row <= 0; rows_done <= 0; drain_buf <= 1'b0;
        end else if (state == S_IDLE) begin
            rows_done <= 0; wsel <= 1'b0; drain_active <= 1'b0;
        end else if (cap) begin
            rowbuf[wsel] <= row_fp16;
            wsel         <= ~wsel;
            drain_buf    <= wsel;
            drain_row    <= rows_done[LOG_N-1:0];
            drain_cnt    <= 0;
            drain_active <= 1'b1;
            rows_done    <= rows_done + 1'b1;
        end else if (drain_active) begin
            // Hold at WPR-1: incrementing past it would take the rowbuf
            // part-select out of range and drive X onto mem_write_data.
            if (drain_cnt == WPR[LOG_WPR:0] - 1'b1) drain_active <= 1'b0;
            else                                    drain_cnt <= drain_cnt + 1'b1;
        end
    end

    // -----------------------------------------------------------------------
    // Sequencer
    // -----------------------------------------------------------------------
    always @(posedge clk) begin
        if (rst) begin
            state <= S_IDLE; rd_issue <= 0; rd_ret <= 0;
            gap_cnt <= 0; rv_sr <= 3'd0;
        end else begin
            rv_sr <= {rv_sr[1:0], mem_read_en};
            if (rvalid) rd_ret <= rd_ret + 1'b1;

            case (state)
                S_IDLE: if (multiply_en) begin
                    state <= S_LOADB; rd_issue <= 0; rd_ret <= 0;
                end

                S_LOADB: begin
                    rd_issue <= rd_issue + 1'b1;
                    if (rd_issue == NW[LOG_NW:0] - 1'b1) begin
                        state <= S_GAP; gap_cnt <= 0;
                    end
                end

                // Required, not slack: the return-side phase tag and rd_ret
                // switch when this state ends, so every in-flight B read must
                // have returned first. Read latency is 3; 5 cycles is the
                // minimum plus two of margin.
                S_GAP: begin
                    gap_cnt <= gap_cnt + 1'b1;
                    if (gap_cnt == 3'd4) begin
                        state <= S_STREAM; rd_issue <= 0; rd_ret <= 0;
                    end
                end

                S_STREAM: begin
                    rd_issue <= rd_issue + 1'b1;
                    if (rd_issue == NW[LOG_NW:0] - 1'b1) state <= S_FINISH;
                end

                S_FINISH: if ((rows_done == N[LOG_N:0]) && !drain_active)
                    state <= S_IDLE;

                default: state <= S_IDLE;
            endcase
        end
    end

`ifdef ASSERT_ON
    always @(posedge clk) if (!rst) begin
        if (cap && drain_active && (drain_cnt != WPR[LOG_WPR:0] - 1'b1))
            $fatal(1, "matmul_fast: row capture collided with an unfinished drain");
    end
`endif

endmodule

`default_nettype wire
