#!/usr/bin/env bash
# Mutation testing: break the RTL in one place at a time and require the
# verification flow to report FAIL. A surviving mutant is a testbench hole.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# Not `set -e`: try_mutant relies on non-zero exits it handles itself, so the
# setup steps are checked explicitly instead.
[ -f .toolchain/env.sh ] || { echo "FATAL: .toolchain/env.sh missing -- run ./scripts/setup_toolchain.sh"; exit 2; }
# shellcheck disable=SC1091
source .toolchain/env.sh
: "${IVL_B:=}" "${IVL_M:=}"
command -v iverilog >/dev/null || { echo "FATAL: iverilog not on PATH after sourcing the toolchain"; exit 2; }

WORK="build/mutate"
rm -rf "$WORK"; mkdir -p "$WORK"
FP="rtl/fp16_mul_to_fp32.v rtl/fp32_add.v rtl/fp32_to_fp16.v"
# N=32, not 16: at N=16 WPR = NPART, so every k-group is a partial LOAD and
# the accumulate path is never exercised (two index mutants survived there).
N=32

python3 scripts/golden.py gen $N normal "$WORK" >/dev/null \
    || { echo "FATAL: stimulus generation failed -- nothing below would mean anything"; exit 2; }
[ -s "$WORK/mem_init.hex" ] \
    || { echo "FATAL: no stimulus produced"; exit 2; }

killed=0; survived=0; declare -a SURVIVORS=()

# The baseline must compile and pass, or "killed (compile error)" would be
# indistinguishable from a broken toolchain.
baseline_check() {
    local design="$1" core defs=""
    if [ "$design" = "b" ]; then core="rtl/matmul_4mul.v"; defs="-DUSE_4MUL";
    else                        core="rtl/matmul_fast.v"; fi
    rm -f "$WORK/c_out.hex"
    iverilog $IVL_B -g2005 -DNDIM=$N $defs -DASSERT_ON \
        -DMEMINIT="\"$ROOT/$WORK/mem_init.hex\"" -DCOUT="\"$ROOT/$WORK/c_out.hex\"" \
        -o "$WORK/base.vvp" tb/tb_matmul.v tb/sim_top.v tb/mem_model.v \
        $core $FP >/dev/null 2>&1 \
        || { echo "FATAL: unmutated design $design does not compile"; exit 2; }
    timeout 300 vvp $IVL_M "$WORK/base.vvp" 2>&1 | grep -q "TB VERDICT: PASS" \
        || { echo "FATAL: unmutated design $design does not pass"; exit 2; }
    python3 scripts/golden.py check $N "$design" "$WORK" >/dev/null 2>&1 \
        || { echo "FATAL: unmutated design $design is not bit-exact"; exit 2; }
    echo "  baseline design $design compiles and passes"
}
echo "=== Baseline ==="
baseline_check a
baseline_check b

# Shared outcomes for all three harnesses: a sed that changed nothing, or a
# mutation that broke syntax, tests nothing -- both count as SURVIVORS.
not_applied() {
    echo "  !! MUTATION DID NOT APPLY: $1 -- fix the sed expression"
    SURVIVORS+=("$1 (not applied)"); survived=$((survived+1))
}
weak_kill() {
    echo "  WEAK KILL (compile error, tests nothing)  $1"
    SURVIVORS+=("$1 (compile error -- rewrite so it compiles)")
    survived=$((survived+1))
}

# $1 label  $2 design(a|b)  $3 target file  $4 sed expression
try_mutant() {
    local label="$1" design="$2" target="$3" expr="$4"
    local core defs=""
    if [ "$design" = "b" ]; then core="rtl/matmul_4mul.v"; defs="-DUSE_4MUL";
    else                        core="rtl/matmul_fast.v"; fi

    local mdir="$WORK/m"; rm -rf "$mdir"; mkdir -p "$mdir"
    cp rtl/*.v "$mdir/"
    sed -i "$expr" "$mdir/$(basename "$target")"
    if diff -q "rtl/$(basename "$target")" "$mdir/$(basename "$target")" >/dev/null; then
        not_applied "$label"; return
    fi

    local srcs=""
    for f in $(basename "$core") $(for x in $FP; do basename "$x"; done); do
        srcs="$srcs $mdir/$f"
    done

    rm -f "$WORK/c_out.hex"
    if ! iverilog $IVL_B -g2005 -DNDIM=$N $defs -DASSERT_ON \
            -DMEMINIT="\"$ROOT/$WORK/mem_init.hex\"" -DCOUT="\"$ROOT/$WORK/c_out.hex\"" \
            -o "$mdir/t.vvp" tb/tb_matmul.v tb/sim_top.v tb/mem_model.v $srcs >/dev/null 2>&1; then
        weak_kill "$label"; return
    fi
    local out; out=$(timeout 300 vvp $IVL_M "$mdir/t.vvp" 2>&1)
    if ! echo "$out" | grep -q "TB VERDICT: PASS"; then
        echo "  killed (testbench self-check)  $label"; killed=$((killed+1)); return
    fi

    if [ ! -f "$WORK/c_out.hex" ]; then
        echo "  killed (no output produced)  $label"; killed=$((killed+1)); return
    fi
    if python3 scripts/golden.py check $N "$design" "$WORK" >/dev/null 2>&1; then
        echo "  SURVIVED                 $label"
        SURVIVORS+=("$label"); survived=$((survived+1))
    else
        echo "  killed (bit mismatch)    $label"; killed=$((killed+1))
    fi
}

# Schedule faults for the cycle harness (tb/tb_cycles.v): they change the
# cycle count, read order or write addresses. Data-only faults are invisible
# to it by construction and covered by the campaign above.
# $1 label  $2 sed expression
try_cycle_mutant() {
    local label="$1" expr="$2"
    local mdir="$WORK/mc"; rm -rf "$mdir"; mkdir -p "$mdir"
    cp rtl/*.v "$mdir/"
    sed -i "$expr" "$mdir/matmul_fast.v"
    if diff -q rtl/matmul_fast.v "$mdir/matmul_fast.v" >/dev/null; then
        not_applied "$label"; return
    fi
    if ! iverilog $IVL_B -g2005 -DNDIM=64 -DLANE_STUB -o "$mdir/t.vvp" \
            tb/tb_cycles.v "$mdir/matmul_fast.v" \
            "$mdir/fp16_mul_to_fp32.v" "$mdir/fp32_add.v" "$mdir/fp32_to_fp16.v" \
            >/dev/null 2>&1; then
        weak_kill "$label"; return
    fi
    local out; out=$(timeout 300 vvp $IVL_M "$mdir/t.vvp" 2>&1)
    if echo "$out" | grep -q "CYCLE HARNESS VERDICT: PASS"; then
        echo "  SURVIVED                 $label"
        SURVIVORS+=("$label"); survived=$((survived+1))
    else
        echo "  killed (schedule check)  $label"; killed=$((killed+1))
    fi
}

# Mutants for the arithmetic primitives, run against tb/tb_fp_units.v.
# $1 label  $2 file  $3 sed expression
try_fp_mutant() {
    local label="$1" target="$2" expr="$3"
    local mdir="$WORK/mf"; rm -rf "$mdir"; mkdir -p "$mdir"
    cp rtl/*.v "$mdir/"
    sed -i "$expr" "$mdir/$target"
    if diff -q "rtl/$target" "$mdir/$target" >/dev/null; then
        not_applied "$label"; return
    fi
    if ! iverilog $IVL_B -g2005 -I vectors -o "$mdir/fp.vvp" \
            tb/tb_fp_units.v "$mdir/fp16_mul_to_fp32.v" "$mdir/fp32_add.v" \
            "$mdir/fp32_to_fp16.v" >/dev/null 2>&1; then
        weak_kill "$label"; return
    fi
    local out; out=$(timeout 300 vvp $IVL_M "$mdir/fp.vvp" 2>&1)
    if echo "$out" | grep -q "FP UNIT TESTS: PASS"; then
        echo "  SURVIVED                 $label"
        SURVIVORS+=("$label"); survived=$((survived+1))
    else
        echo "  killed (vector mismatch) $label"; killed=$((killed+1))
    fi
}

echo "=== Design A mutants ==="
try_mutant "A: adder tree reassociated (p0+p2)+(p1+p3)" a rtl/matmul_fast.v \
  's|\.a(p_q\[31:0\]),  \.b(p_q\[63:32\])|.a(p_q[31:0]),  .b(p_q[95:64])|'
try_mutant "A: a_last off by one" a rtl/matmul_fast.v \
  "s|a_kg == WPR\[LOG_WPR-1:0\] - 1'b1|a_kg == WPR[LOG_WPR-1:0] - 2'd2|"
try_mutant "A: B bank lane index inverted" a rtl/matmul_fast.v \
  's|wire \[1:0\]         b_lane = b_k\[1:0\];|wire [1:0]         b_lane = ~b_k[1:0];|'
try_mutant "A: bank write column group off by one" a rtl/matmul_fast.v \
  's|localparam integer MY_JG = g >> 2;|localparam integer MY_JG = (g >> 2) ^ 1;|'
try_mutant "A: writeback address off by one" a rtl/matmul_fast.v \
  "s|wire \[31:0\] wr_addr_full = C_BASE + drain_row \* WPR|wire [31:0] wr_addr_full = C_BASE + 1 + drain_row * WPR|"
try_mutant "A: partial never re-initialised (wr_pf ignored)" a rtl/matmul_fast.v \
  "s|if (wr_v) part\[wr_px\] <= wr_pf ? s_d\[ADD_LAT-1\] : acc_add;|if (wr_v) part[wr_px] <= acc_add;|"
try_mutant "A: partial write index off by one" a rtl/matmul_fast.v \
  "s|wire \[1:0\] wr_px   = px_sr\[2\*ACC_WR-1 -: 2\];|wire [1:0] wr_px   = px_sr[2*ACC_WR-1 -: 2] + 2'd1;|"
try_mutant "A: partial combine reassociated to (p0+p2)+(p1+p3)" a rtl/matmul_fast.v \
  "s|.a(part\[0\]), .b(part\[1\]), .y(c01)|.a(part[0]), .b(part[2]), .y(c01)|"
try_mutant "A: accumulate read index not offset from the write index" a rtl/matmul_fast.v \
  "s|wire \[1:0\] rd_px   = px_sr\[2\*ACC_RD-1 -: 2\];|wire [1:0] rd_px   = px_sr[2*ACC_WR-1 -: 2];|"
try_mutant "A: row buffer single-buffered" a rtl/matmul_fast.v \
  's|drain_buf    <= wsel;|drain_buf    <= ~wsel;|'
try_mutant "A: S_GAP shortened below the 3-cycle minimum" a rtl/matmul_fast.v \
  "s|if (gap_cnt == 3'd4) begin|if (gap_cnt == 3'd0) begin|"

echo "=== Design B mutants ==="
try_mutant "B: wrong B word selected (b_buf[r] not b_buf[s])" b rtl/matmul_4mul.v \
  's|wire \[DATA_W-1:0\] b_word = b_buf\[s\];|wire [DATA_W-1:0] b_word = b_buf[r];|'
try_mutant "B: tile_first dropped (accumulators never reinitialised)" b rtl/matmul_4mul.v \
  's|acc\[r_d3\*4 + c\] <= f_d3 ? p_d3\[32\*c +: 32\] : sum;|acc[r_d3*4 + c] <= sum;|'
try_mutant "B: write port not delayed to match the adder pipeline" b rtl/matmul_4mul.v \
  's|acc\[r_d3\*4 + c\] <= f_d3 ?|acc[r_q*4 + c] <= f_d3 ?|'
try_mutant "B: writeback uses live cur_ib instead of latched ret_ib" b rtl/matmul_4mul.v \
  's|wire \[LOG_N-1:0\] wb_row = {ret_ib, wb_cnt};|wire [LOG_N-1:0] wb_row = {cur_ib, wb_cnt};|'
try_mutant "B: prefetch A/B selector inverted" b rtl/matmul_4mul.v \
  's|assign mem_read_addr = t\[2\] ? a_addr_full\[ADDR_W-1:0\]|assign mem_read_addr = !t[2] ? a_addr_full[ADDR_W-1:0]|'
try_mutant "B: writeback snapshot one cycle early" b rtl/matmul_4mul.v \
  's|if (wb_cap) begin|if (wb_arm) begin|'
try_mutant "B: flush shortened, last writes lost" b rtl/matmul_4mul.v \
  "s|if (flush_cnt == 4'd11 + ADD_LAT\[3:0\]) state <= S_IDLE;|if (flush_cnt == 4'd1) state <= S_IDLE;|"
try_mutant "B: writeback snapshot not delayed for the adder pipeline" b rtl/matmul_4mul.v \
  "s|wb_cap <= wb_lat\[ADD_LAT-1\];|wb_cap <= wb_arm;|"

# Three mutants were tried here and removed because they are provably
# EQUIVALENT, not missed by the vectors: a_is_larger >= -> > (operands
# interchangeable at |a| == |b|), alignment clamp 33 -> 32 (leading one
# already inside the sticky window either way), fp16 boundary -14 -> -13
# (both branches identical at e = -14). Recorded so nobody re-adds them.
echo "=== Arithmetic-primitive mutants (tb/tb_fp_units.v) ==="
# The first two turn round-to-nearest-EVEN into round-half-up, which only an
# exact-midpoint vector can reveal.
try_fp_mutant "fp32_add: ties-to-even dropped (becomes round-half-up)" fp32_add.v \
  "s|wire round_up = g_bit & (r_bit \| s_bit \| sig_n\[0\]);|wire round_up = g_bit;|"
try_fp_mutant "fp32_to_fp16: ties-to-even dropped" fp32_to_fp16.v \
  "s|wire round_up = g_bit & (s_bit \| q\[0\]);|wire round_up = g_bit;|"
# NOTE the @ delimiter: these patterns contain '|', and with '|' as the sed
# delimiter the expression mis-parses into broken Verilog -- a compile-error
# "kill" that tests nothing.
try_fp_mutant "fp32_add: sticky bit ignored" fp32_add.v \
  "s@wire        s_bit_w = |norm\[29:0\];@wire        s_bit_w = 1'b0;@"
try_fp_mutant "fp32_add: guard bit taken one position low" fp32_add.v \
  "s|wire        g_bit_w = norm\[31\];|wire        g_bit_w = norm[30];|"
try_fp_mutant "fp32_add: alignment clamp far too low (loses guard)" fp32_add.v \
  "s|(shift_raw > 9'd33) ? 6'd33|(shift_raw > 9'd24) ? 6'd24|"
try_fp_mutant "fp32_to_fp16: sticky bit ignored" fp32_to_fp16.v \
  "s@wire s_bit  = |(sigx & low_mask);@wire s_bit  = 1'b0;@"
try_fp_mutant "fp16_mul: exponent bias off by one" fp16_mul_to_fp32.v \
  "s|\$signed({4'b0, ea}) - 9'sd25|\$signed({4'b0, ea}) - 9'sd24|"
try_fp_mutant "fp16_mul: subnormal inputs treated as normal" fp16_mul_to_fp32.v \
  "s|wire \[10:0\] siga = (ea == 5'd0) ? {1'b0, ma} : {1'b1, ma};|wire [10:0] siga = {1'b1, ma};|"
try_fp_mutant "fp32_to_fp16: subnormal boundary shifted into the normals" fp32_to_fp16.v \
  "s|(e >= -10'sd14) ? 11'sd13|(e >= -10'sd16) ? 11'sd13|"

echo
echo "=== Cycle-harness mutants (schedule faults, tb/tb_cycles.v) ==="
try_cycle_mutant "S_GAP shortened below the 3-cycle minimum" \
  "s|if (gap_cnt == 3'd4) begin|if (gap_cnt == 3'd0) begin|"
try_cycle_mutant "writeback address off by one" \
  "s|wire \\[31:0\\] wr_addr_full = C_BASE + drain_row \\* WPR|wire [31:0] wr_addr_full = C_BASE + 1 + drain_row * WPR|"
try_cycle_mutant "a_last off by one (rows retire at the wrong rate)" \
  "s|a_kg == WPR\\[LOG_WPR-1:0\\] - 1'b1|a_kg == WPR[LOG_WPR-1:0] - 2'd2|"
try_cycle_mutant "read phases swapped (A streamed before B is resident)" \
  "s|((state == S_LOADB) ? B_BASE : A_BASE)|((state == S_LOADB) ? A_BASE : B_BASE)|"

echo
echo "=============================================="
echo "mutants killed:   $killed"
echo "mutants survived: $survived"
if [ "$survived" -ne 0 ]; then
    printf '  SURVIVOR: %s\n' "${SURVIVORS[@]}"
    echo "A surviving mutant is a hole in the testbench. Fix the testbench."
    exit 1
fi
echo "ALL MUTANTS KILLED -- the flow is capable of failing."
