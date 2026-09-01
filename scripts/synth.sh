#!/usr/bin/env bash
# Real synthesis against Nangate45 (free 45nm): delay and area, not gate
# levels; only the before/after ratio transfers. Tools install rootless via
# scripts/setup_toolchain.sh. Usage: synth.sh [adder|machines|before]
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# yosys/berkeley-abc land in opt/ via setup_toolchain.sh (rootless).
export PATH="$ROOT/.toolchain/opt/usr/bin:$PATH"
export LD_LIBRARY_PATH="$ROOT/.toolchain/opt/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"

LIB="$ROOT/.toolchain/lib/Nangate45_typ.lib"
FP="rtl/fp16_mul_to_fp32.v rtl/fp32_add.v rtl/fp32_to_fp16.v"
OUT="build/synth"
mkdir -p "$OUT"

if ! command -v yosys >/dev/null; then
    echo "yosys not found -- run ./scripts/setup_toolchain.sh" >&2; exit 1
fi
if ! command -v berkeley-abc >/dev/null; then
    echo "berkeley-abc not found -- run ./scripts/setup_toolchain.sh" >&2; exit 1
fi
if ! command -v bc >/dev/null; then
    echo "bc not found -- needed to format results; install it (e.g. apt bc)" >&2; exit 1
fi
if [ ! -s "$LIB" ]; then
    echo "cell library missing -- run ./scripts/setup_toolchain.sh (it fetches" >&2
    echo "Nangate45; the URL is in this script's header if you need it by hand)" >&2
    exit 1
fi

# The ABC script must come from a yosys SCRIPT FILE, not -p: its ';'
# collides with yosys's separator, and within one ABC command a space is
# written as a comma. buffer/upsize/dnsize are NOT optional: without them ABC
# reports the delay of an unbuffered netlist and any high-fanout net
# dominates the result.
ABC='abc -liberty '"$LIB"' -script +strash;ifraig;scorr;dc2;dretime;strash;&get,-n;&dch,-f;&nf;&put;buffer;upsize;dnsize;stime'

FAILS=0

# $1 label  $2 top  $3 extra yosys lines  $4 sources
run() {
    local label="$1" top="$2" extra="$3" srcs="$4"
    local tag; tag=$(echo "$label" | tr -cd '[:alnum:]')
    {
        echo "read_verilog $srcs"
        [ -n "$extra" ] && echo "$extra"
        echo "hierarchy -top $top"
        echo "synth -top $top -flatten"
        echo "dfflibmap -liberty $LIB"
        echo "$ABC"
        echo "opt_clean"
        echo "stat -liberty $LIB"
    } > "$OUT/$tag.ys"
    timeout "${SYNTH_TIMEOUT:-5400}" yosys -s "$OUT/$tag.ys" > "$OUT/$tag.log" 2>&1
    local ps area
    ps=$(grep -oE 'Delay = *[0-9.]+ ps' "$OUT/$tag.log" | tail -1 | grep -oE '[0-9.]+')
    area=$(grep -oE "Chip area for module '..[^']*': [0-9.]+" "$OUT/$tag.log" | tail -1 | grep -oE '[0-9.]+$')
    if [ -z "$ps" ]; then
        printf '  %-34s %s\n' "$label" "FAILED -- see $OUT/$tag.log"
        FAILS=$((FAILS+1))
        return 1
    fi
    printf '  %-34s %8.2f ns   %6.0f MHz   %10.0f um2\n' \
        "$label" "$(echo "$ps/1000" | bc -l)" "$(echo "1000000/$ps" | bc -l)" "$area"
}

what="${1:-all}"
case "$what" in
    all|adder|machines|before) ;;
    *) echo "usage: $0 [adder|machines|before]" >&2; exit 1 ;;
esac

echo "======================================================================"
echo "Synthesis against Nangate45 (free 45nm) -- real delay and area"
echo "======================================================================"
echo
printf '  %-34s %11s %12s %14s\n' "" "delay" "Fmax" "area"

if [ "$what" = all ] || [ "$what" = adder ]; then
    echo
    echo "The block that set the clock of both machines:"
    for st in 0 3; do
        cat > "$OUT/w$st.v" <<EOF
module top(input clk, input [31:0] a, input [31:0] b, output [31:0] y);
  fp32_add #(.STAGES($st)) u (.clk(clk), .a(a), .b(b), .y(y));
endmodule
EOF
        run "fp32_add STAGES=$st" top "" "$OUT/w$st.v rtl/fp32_add.v"
    done
    echo
    echo "  fp16_mul_to_fp32 and fp32_to_fp16, for comparison:"
    run "fp16_mul_to_fp32" fp16_mul_to_fp32 "" "rtl/fp16_mul_to_fp32.v"
    run "fp32_to_fp16"     fp32_to_fp16     "" "rtl/fp32_to_fp16.v"
fi

if [ "$what" = all ] || [ "$what" = machines ]; then
    echo
    echo "Whole machines at small N. N-independent for matmul_4mul (its muxes"
    echo "are tile-local); NOT for matmul_fast, whose (N/4):1 drain mux is 4:1"
    echo "here but 256:1 at N=1024 -- its clock will likely fall at full size:"
    run "matmul_4mul N=8"  matmul_4mul  "chparam -set N 8 matmul_4mul"   "rtl/matmul_4mul.v $FP"
    run "matmul_fast N=16" matmul_fast  "chparam -set N 16 matmul_fast"  "rtl/matmul_fast.v $FP"
fi

# "What would the clock be without the adder pipeline": take this RTL and
# make the adder instances combinational. Like-for-like for matmul_4mul; an
# UPPER BOUND only for matmul_fast (the NPART partial mux stays). This is a
# critical-path experiment, not a working machine -- with .STAGES(0) the
# schedule no longer matches the adder latency. Never simulate these.
if [ "$what" = all ] || [ "$what" = before ]; then
    echo
    echo "Same machines with the adder pipeline removed (critical path only,"
    echo "these do not compute correct results -- see the comment in this script):"
    mkdir -p "$OUT/before"
    for m in matmul_4mul matmul_fast; do
        sed 's/\.STAGES(ADD_LAT)/.STAGES(0)/g' "rtl/$m.v" > "$OUT/before/$m.v"
        cmp -s "rtl/$m.v" "$OUT/before/$m.v" && {
            echo "  $m: sed matched nothing -- instantiation changed, fix this script" >&2
            exit 1
        }
    done
    run "matmul_4mul N=8 comb-add"  matmul_4mul \
        "chparam -set N 8 matmul_4mul"  "$OUT/before/matmul_4mul.v $FP"
    # Upper bound only and very slow, so the default run skips it.
    if [ "$what" = before ]; then
        run "matmul_fast N=16 comb-add" matmul_fast \
            "chparam -set N 16 matmul_fast" "$OUT/before/matmul_fast.v $FP"
    else
        printf '  %-34s %s\n' "matmul_fast N=16 comb-add" \
            "skipped (upper bound only, very slow -- './scripts/synth.sh before')"
    fi
fi

echo
echo "Logs and scripts in $OUT/"
if [ "$FAILS" -gt 0 ]; then
    echo "$FAILS run(s) FAILED" >&2
    exit 1
fi
