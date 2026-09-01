// matmul_4mul -- machine #2: "only four multipliers", same interface as
// matmul_fast. 4x4-output-stationary; one microstep = one tile x four k
// values (8 reads, 64 MACs, 16 cycles), prefetched. Cycles: N^3/4 + 32.
`default_nettype none

module matmul_4mul #(
    parameter integer N      = 1024,   // matrix dimension, power of two, >= 8
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

    // fp32_add's pipeline depth; absorbed by the four-cycle accumulator
    // rotation (see header).
    localparam integer ADD_LAT = 3;

    localparam integer EPW     = 4;
    localparam integer WPR     = N / EPW;    // words per matrix row
    localparam integer NW      = N * WPR;    // words per matrix
    localparam integer TB      = N / 4;      // tiles per dimension (== WPR)
    localparam integer A_BASE  = 0;
    localparam integer B_BASE  = NW;
    localparam integer C_BASE  = 2 * NW;
    localparam integer LOG_N   = clog2(N);
    localparam integer LOG_TB  = clog2(TB);

    initial begin
        if ((N < 8) || ((N & (N - 1)) != 0)) begin
            $display("FATAL matmul_4mul: N must be a power of two >= 8 (got %0d)", N);
            $finish;
        end
    end

    localparam [1:0] S_IDLE  = 2'd0,
                     S_PRE   = 2'd1,   // prologue: prefetch microstep 0
                     S_RUN   = 2'd2,
                     S_FLUSH = 2'd3;

    reg [1:0] state;
    reg [3:0] t;                        // cycle within the 16-cycle microstep

    assign ready = (state == S_IDLE);

    wire compute_en = (state == S_RUN);
    wire [1:0] r = t[1:0];              // tile row
    wire [1:0] s = t[3:2];              // k offset within the microstep

    // -----------------------------------------------------------------------
    // Tile iteration. `cur_*` is the microstep being computed, `pf_*` the one
    // being prefetched. kb innermost: it is the reduction axis, so the
    // accumulators stay put across it.
    // -----------------------------------------------------------------------
    reg [LOG_TB-1:0] cur_ib, cur_jb, cur_kb;
    reg [LOG_TB-1:0] pf_ib,  pf_jb,  pf_kb;
    reg              pf_valid;          // false once the last tile is prefetched

    wire kb_wrap = (pf_kb == TB[LOG_TB-1:0] - 1'b1);
    wire jb_wrap = (pf_jb == TB[LOG_TB-1:0] - 1'b1);
    wire ib_wrap = (pf_ib == TB[LOG_TB-1:0] - 1'b1);
    wire pf_is_last = kb_wrap & jb_wrap & ib_wrap;

    wire cur_kb_last = (cur_kb == TB[LOG_TB-1:0] - 1'b1);
    wire cur_is_last = cur_kb_last
                    && (cur_jb == TB[LOG_TB-1:0] - 1'b1)
                    && (cur_ib == TB[LOG_TB-1:0] - 1'b1);

    // -----------------------------------------------------------------------
    // Prefetch: 8 reads issued on cycles 0..7, returning on cycles 3..10,
    // landing in shadow registers that swap into the active set at t == 15.
    // Cycles 0..3 fetch B rows 4*kb+s; cycles 4..7 fetch A rows 4*ib+r.
    // -----------------------------------------------------------------------
    wire issue = ((state == S_PRE) || (state == S_RUN && pf_valid)) && (t < 4'd8);

    wire [LOG_N-1:0] pf_b_row = {pf_kb, t[1:0]};
    wire [LOG_N-1:0] pf_a_row = {pf_ib, t[1:0]};

    localparam integer PAD_TB = 32 - LOG_TB;
    wire [31:0] b_addr_full = B_BASE + pf_b_row * WPR + {{PAD_TB{1'b0}}, pf_jb};
    wire [31:0] a_addr_full = A_BASE + pf_a_row * WPR + {{PAD_TB{1'b0}}, pf_kb};

    assign mem_read_en   = issue;
    assign mem_read_addr = t[2] ? a_addr_full[ADDR_W-1:0]
                                : b_addr_full[ADDR_W-1:0];

    reg [2:0] rv_sr;
    reg [3:0] ret_cnt;
    wire      rvalid = rv_sr[2];

    reg [DATA_W-1:0] a_buf  [0:3];      // active  A[4ib+r][4kb..4kb+3]
    reg [DATA_W-1:0] b_buf  [0:3];      // active  B[4kb+s][4jb..4jb+3]
    reg [DATA_W-1:0] a_next [0:3];
    reg [DATA_W-1:0] b_next [0:3];

    integer q;
    always @(posedge clk) begin
        if (rst) begin
            rv_sr <= 3'd0; ret_cnt <= 4'd0;
        end else begin
            rv_sr <= {rv_sr[1:0], issue};
            if (t == 4'd15) ret_cnt <= 4'd0;
            else if (rvalid) ret_cnt <= ret_cnt + 1'b1;

            if (rvalid) begin
                if (ret_cnt[2]) a_next[ret_cnt[1:0]] <= mem_read_data;
                else            b_next[ret_cnt[1:0]] <= mem_read_data;
            end

            if (t == 4'd15)
                for (q = 0; q < 4; q = q + 1) begin
                    a_buf[q] <= a_next[q];
                    b_buf[q] <= b_next[q];
                end
        end
    end

    // -----------------------------------------------------------------------
    // The four multipliers. One binary16 from A broadcast against one 64-bit
    // word of B. Products are exact in binary32 (see fp16_mul_to_fp32).
    // -----------------------------------------------------------------------
    wire [15:0]  a_el = a_buf[r][16*s +: 16];
    wire [DATA_W-1:0] b_word = b_buf[s];

    wire [127:0] pw;
    genvar c;
    generate
        for (c = 0; c < 4; c = c + 1) begin : mul
            fp16_mul_to_fp32 u_mul (
                .a (a_el),
                .b (b_word[16*c +: 16]),
                .y (pw[32*c +: 32])
            );
        end
    endgenerate

    reg [127:0] p_q;
    reg [1:0]   r_q;
    reg         v_q, first_q;

    // A tile's very first product initialises its accumulators rather than
    // adding to a cleared register, so tiles run back to back with no bubble.
    wire tile_first = (cur_kb == {LOG_TB{1'b0}}) && (s == 2'd0);

    always @(posedge clk) begin
        p_q     <= pw;
        r_q     <= r;
        first_q <= tile_first;
        v_q     <= compute_en;
        if (rst) v_q <= 1'b0;
    end

    // -----------------------------------------------------------------------
    // Write-side replay of the accumulate control, delayed by ADD_LAT: the
    // pipelined adder's `sum` belongs to operands presented ADD_LAT cycles
    // ago, so the write index, valid, first flag and the product to
    // initialise with must arrive with it.
    //
    // p_q is carried forward because loading x is NOT bit-identical to 0 + x:
    // for x = -0.0 the adder returns +0.0 under round-to-nearest-even.
    // -----------------------------------------------------------------------
    reg [127:0] p_d1, p_d2, p_d3;
    reg [1:0]   r_d1, r_d2, r_d3;
    reg         v_d1, v_d2, v_d3;
    reg         f_d1, f_d2, f_d3;

    always @(posedge clk) begin
        p_d1 <= p_q;   p_d2 <= p_d1;   p_d3 <= p_d2;
        r_d1 <= r_q;   r_d2 <= r_d1;   r_d3 <= r_d2;
        f_d1 <= first_q; f_d2 <= f_d1; f_d3 <= f_d2;
        v_d1 <= v_q;   v_d2 <= v_d1;   v_d3 <= v_d2;
        if (rst) begin v_d1 <= 1'b0; v_d2 <= 1'b0; v_d3 <= 1'b0; end
    end

    // -----------------------------------------------------------------------
    // 16 binary32 accumulators, four adders. r_q selects which group of four
    // is live this cycle; the group was last touched four cycles ago.
    // -----------------------------------------------------------------------
    reg [31:0] acc [0:15];

    generate
        for (c = 0; c < 4; c = c + 1) begin : accu
            wire [31:0] sum;
            // Read side uses the live indices, write side the delayed ones.
            // A group is read every four cycles and written ADD_LAT = 3
            // later, so the next read always sees the previous update.
            fp32_add #(.STAGES(ADD_LAT)) u_add (
                .clk (clk),
                .a (acc[r_q*4 + c]),
                .b (p_q[32*c +: 32]),
                .y (sum)
            );
            always @(posedge clk)
                if (v_d3)
                    acc[r_d3*4 + c] <= f_d3 ? p_d3[32*c +: 32] : sum;
        end
    endgenerate

    // -----------------------------------------------------------------------
    // Writeback. When a tile's last microstep retires, snapshot all 16
    // accumulators (the live set is already being overwritten by the next
    // tile) and drain four C words -- 4 writes per 4096 compute cycles.
    // -----------------------------------------------------------------------
    reg        wb_arm, wb_cap, wb_active;
    reg [ADD_LAT-1:0] wb_lat;          // adder-latency delay on the snapshot
    reg [1:0]  wb_cnt;
    reg [31:0] wb_acc [0:15];
    // ret_ib/ret_jb are latched at t==15 of the retiring tile and stay
    // stable for the whole 9-cycle writeback window: the next tile retires
    // at least 16*TB >= 32 cycles later.
    reg [LOG_TB-1:0] ret_ib, ret_jb; // tile that retired at t==15
    reg [3:0]  flush_cnt;

    wire [DATA_W-1:0] wb_word;
    generate
        for (c = 0; c < 4; c = c + 1) begin : cvt
            fp32_to_fp16 u_cvt (
                .a (wb_acc[wb_cnt*4 + c]),
                .y (wb_word[16*c +: 16])
            );
        end
    endgenerate

    wire [LOG_N-1:0] wb_row = {ret_ib, wb_cnt};
    wire [31:0] wb_addr_full = C_BASE + wb_row * WPR + {{PAD_TB{1'b0}}, ret_jb};

    always @* begin
        mem_write_en   = wb_active;
        mem_write_addr = wb_addr_full[ADDR_W-1:0];
        mem_write_data = wb_word;
    end

    integer w;
    always @(posedge clk) begin
        if (rst) begin
            wb_arm <= 1'b0; wb_cap <= 1'b0; wb_active <= 1'b0; wb_cnt <= 2'd0;
            wb_lat <= {ADD_LAT{1'b0}};
        end else begin
            // Latch the retiring tile's indices at t==15: one cycle later,
            // when wb_arm is seen, cur_* has already advanced to the next tile.
            if ((t == 4'd15) && compute_en && cur_kb_last) begin
                ret_ib <= cur_ib;
                ret_jb <= cur_jb;
            end
            // The snapshot waits ADD_LAT extra cycles for the pipelined
            // adder to retire the tile's last accumulate.
            wb_arm <= (t == 4'd15) && compute_en && cur_kb_last;
            wb_lat <= {wb_lat[ADD_LAT-2:0], wb_arm};
            wb_cap <= wb_lat[ADD_LAT-1];

            if (wb_cap) begin
                for (w = 0; w < 16; w = w + 1) wb_acc[w] <= acc[w];
                wb_active <= 1'b1;
                wb_cnt    <= 2'd0;
            end else if (wb_active) begin
                if (wb_cnt == 2'd3) wb_active <= 1'b0;
                wb_cnt <= wb_cnt + 1'b1;
            end
        end
    end

    // -----------------------------------------------------------------------
    // Sequencer
    // -----------------------------------------------------------------------
    always @(posedge clk) begin
        if (rst) begin
            state <= S_IDLE; t <= 4'd0; pf_valid <= 1'b0;
            cur_ib <= 0; cur_jb <= 0; cur_kb <= 0;
            pf_ib  <= 0; pf_jb  <= 0; pf_kb  <= 0;
            flush_cnt <= 4'd0;
        end else begin
            case (state)
                S_IDLE: if (multiply_en) begin
                    state <= S_PRE; t <= 4'd0; pf_valid <= 1'b1;
                    pf_ib <= 0; pf_jb <= 0; pf_kb <= 0;
                end

                S_PRE: begin
                    t <= t + 1'b1;
                    if (t == 4'd15) begin
                        state  <= S_RUN;
                        cur_ib <= pf_ib; cur_jb <= pf_jb; cur_kb <= pf_kb;
                        pf_kb  <= pf_kb + 1'b1;   // TB >= 2, so no wrap here
                        t      <= 4'd0;
                    end
                end

                S_RUN: begin
                    t <= t + 1'b1;
                    if (t == 4'd15) begin
                        t <= 4'd0;
                        if (cur_is_last) begin
                            state <= S_FLUSH; flush_cnt <= 4'd0;
                        end else begin
                            cur_ib <= pf_ib; cur_jb <= pf_jb; cur_kb <= pf_kb;
                            if (pf_is_last) begin
                                pf_valid <= 1'b0;
                            end else if (kb_wrap) begin
                                pf_kb <= 0;
                                if (jb_wrap) begin
                                    pf_jb <= 0;
                                    pf_ib <= pf_ib + 1'b1;
                                end else begin
                                    pf_jb <= pf_jb + 1'b1;
                                end
                            end else begin
                                pf_kb <= pf_kb + 1'b1;
                            end
                        end
                    end
                end

                // Retire the final MACs through the adder pipeline,
                // snapshot, and drain the last four writes.
                S_FLUSH: begin
                    flush_cnt <= flush_cnt + 1'b1;
                    if (flush_cnt == 4'd11 + ADD_LAT[3:0]) state <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule

`default_nettype wire
