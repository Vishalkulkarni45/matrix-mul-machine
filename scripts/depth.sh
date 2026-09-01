#!/usr/bin/env bash
# Logic depth per block via yosys ltp: generic-gate LEVELS, not an Fmax --
# good only for ratios and before/after. Real delay/area: scripts/synth.sh.
# Needs yowasp-yosys (python3 -m pip install --user yowasp-yosys).
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PATH="$HOME/.local/bin:$PATH"
mkdir -p build/depth

if ! command -v yowasp-yosys >/dev/null 2>&1; then
    echo "yowasp-yosys not found. Install it with:"
    echo "    python3 -m pip install --user yowasp-yosys"
    exit 1
fi

FP="rtl/fp16_mul_to_fp32.v rtl/fp32_add.v rtl/fp32_to_fp16.v"

# $1 label  $2 top  $3 extra chparam/read args
measure() {
    local label="$1" top="$2" extra="${3:-}" src="${4:-$FP}"
    local log="build/depth/${top}$(echo "$extra" | tr -cd '[:alnum:]').log"
    yowasp-yosys -p "
        read_verilog $src
        ${extra}
        hierarchy -top $top
        proc; flatten; opt -full; memory; techmap; opt -full; simplemap; opt -full
        stat
        ltp -noff
    " > "$log" 2>&1
    local depth cells
    depth=$(grep -oP 'Longest topological path in \S+ \(length=\K[0-9]+' "$log" | tail -1)
    cells=$(grep -oE '^[[:space:]]*[0-9]+ cells$' "$log" | tail -1 | grep -oE '[0-9]+')
    printf '  %-46s %6s levels   %8s cells\n' "$label" "${depth:-?}" "${cells:-?}"
}

echo "======================================================================"
echo "Logic depth -- generic gates, register to register, NOT an Fmax"
echo "======================================================================"
echo
echo "Arithmetic primitives:"
measure "fp16_mul_to_fp32"                    fp16_mul_to_fp32
measure "fp32_to_fp16"                        fp32_to_fp16
measure "fp32_add  STAGES=0  (combinational)" fp32_add "chparam -set STAGES 0 fp32_add"
measure "fp32_add  STAGES=3  (pipelined)"     fp32_add "chparam -set STAGES 3 fp32_add"
echo
# Smallest legal N each. N-independent for matmul_4mul (tile-local muxes);
# NOT for matmul_fast, whose (N/4):1 drain mux grows with N.
echo "Whole designs (smallest legal N; N-independent for matmul_4mul only):"
measure "matmul_4mul (N=8)"  matmul_4mul "chparam -set N 8 matmul_4mul"   "rtl/matmul_4mul.v $FP"
measure "matmul_fast (N=16)" matmul_fast "chparam -set N 16 matmul_fast" "rtl/matmul_fast.v $FP"
echo
echo "Logs in build/depth/"
