#!/usr/bin/env bash
# Single entry point for every simulation in this project.
#   ./scripts/run.sh fp                       FP primitive bit-exactness
#   ./scripts/run.sh sim <a|b> <N> [mode] [icarus|verilator]
#   ./scripts/run.sh handshake <a|b> [N]      command-interface tests
#   ./scripts/run.sh cycles [N]               matmul_fast schedule at real N
#   ./scripts/run.sh mutate                   mutation testing of the flow
#   ./scripts/run.sh regress                  the standard regression matrix
#   ./scripts/run.sh full <a|b> [mode]        N=1024, the real problem size
#   <a|b> a = matmul_fast, b = matmul_4mul; mode normal|stress|identity|extreme
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
[ -f .toolchain/env.sh ] || ./scripts/setup_toolchain.sh >/dev/null
# shellcheck disable=SC1091
source .toolchain/env.sh
: "${IVL_B:=}" "${IVL_M:=}"

# Prefer a locally built Verilator 5 over the packaged 4.038: it roughly
# halves elaboration memory for matmul_fast at N=1024. Build it with
# .toolchain/bin/v5env.sh.
if [ -x "$ROOT/.toolchain/v5/bin/verilator" ]; then
    export PATH="$ROOT/.toolchain/v5/bin:$PATH"
    export VERILATOR_ROOT="$ROOT/.toolchain/v5/share/verilator"
fi

RTL_COMMON="rtl/fp16_mul_to_fp32.v rtl/fp32_add.v rtl/fp32_to_fp16.v"
JOBS="$(nproc)"
mkdir -p build

green() { printf '\033[32m%s\033[0m\n' "$*"; }
red()   { printf '\033[31m%s\033[0m\n' "$*"; }

run_fp() {
    python3 scripts/gen_fp_vectors.py
    iverilog $IVL_B -g2005 -I vectors -o build/tb_fp.vvp \
        tb/tb_fp_units.v $RTL_COMMON
    # vvp exits 0 even when the testbench prints FAIL, so the verdict has to
    # be read out of the transcript.
    local out; out=$(vvp $IVL_M build/tb_fp.vvp 2>&1)
    echo "$out"
    echo "$out" | grep -q "FP UNIT TESTS: PASS" || { red "FP primitives FAILED"; return 1; }
}

# $1 design (a|b)  $2 N  $3 mode  $4 simulator  $5 outdir
run_sim() {
    local d="$1" n="$2" mode="${3:-normal}" sim="${4:-icarus}" out="${5:-build}"
    local core defs=""
    if [ "$d" = "b" ]; then core="rtl/matmul_4mul.v"; defs="-DUSE_4MUL";
    else                    core="rtl/matmul_fast.v"; fi

    mkdir -p "$out"
    # Remove the previous dump AND stimulus: if generation or simulation
    # dies, nothing stale must survive to be graded.
    rm -f "$out/c_out.hex" "$out/mem_init.hex" "$out/golden_$d.hex"
    python3 scripts/golden.py gen "$n" "$mode" "$out" "$d" >/dev/null \
        || { red "stimulus generation failed for $d N=$n $mode"; return 1; }
    [ -s "$out/mem_init.hex" ] \
        || { red "no stimulus produced for $d N=$n $mode"; return 1; }
    local mem_md5; mem_md5="$(md5sum "$out/mem_init.hex" | cut -d' ' -f1)"

    if [ "$sim" = "verilator" ]; then
        local od="$out/vobj_${d}_${n}"
        rm -rf "$od"
        verilator -Wno-fatal --cc tb/sim_top.v $core $RTL_COMMON \
            -Itb -Irtl --top-module sim_top \
            -DNDIM="$n" $defs -DMEMINIT="\"$ROOT/$out/mem_init.hex\"" \
            --exe "$ROOT/tb/sim_main.cpp" \
            -CFLAGS "-O2 -DNDIM=$n -DCOUT_PATH='\"$ROOT/$out/c_out.hex\"'" \
            -O3 --x-assign fast --x-initial fast \
            --Mdir "$od" -o vsim >/dev/null
        make -s -C "$od" -f Vsim_top.mk -j"$JOBS" >/dev/null
        # The harness asserts these itself; grep the transcript too.
        local vout; vout=$("$od/vsim") || { red "verilator harness FAILED"; return 1; }
        echo "$vout"
        echo "$vout" | grep -q "mem_errors=0" \
            || { red "memory region violations reported"; return 1; }
    else
        # ASSERT_ON compiles in matmul_fast's row-buffer collision assertion.
        iverilog $IVL_B -g2005 -DNDIM="$n" $defs -DASSERT_ON \
            -DMEMINIT="\"$ROOT/$out/mem_init.hex\"" \
            -DCOUT="\"$ROOT/$out/c_out.hex\"" \
            -o "$out/tb_${d}_${n}.vvp" \
            tb/tb_matmul.v tb/sim_top.v tb/mem_model.v $core $RTL_COMMON
        local tbout; tbout=$(vvp $IVL_M "$out/tb_${d}_${n}.vvp" 2>&1)
        echo "$tbout" | grep -E 'RESULT|TB ERROR|MEM ERROR|VERDICT' || true
        echo "$tbout" | grep -q "TB VERDICT: PASS" \
            || { red "testbench self-checks FAILED"; return 1; }
    fi

    # A missing dump means the simulation died; never grade a stale file.
    [ -f "$out/c_out.hex" ] || { red "no c_out.hex produced -- simulation failed"; return 1; }
    # The DUT must have run against the same stimulus the golden model used.
    [ "$(md5sum "$out/mem_init.hex" | cut -d' ' -f1)" = "$mem_md5" ] \
        || { red "mem_init.hex changed during the run -- concurrent use of $out?"; return 1; }
    python3 scripts/golden.py check "$n" "$d" "$out"
}

# Command-interface tests: back-to-back operations, reset mid-operation, and
# multiply_en held high.
run_handshake() {
    local d="$1" n="${2:-16}" core defs=""
    if [ "$d" = "b" ]; then core="rtl/matmul_4mul.v"; defs="-DUSE_4MUL";
    else                    core="rtl/matmul_fast.v"; fi
    local out="build/hs_${d}_${n}"; mkdir -p "$out"
    python3 scripts/golden.py gen "$n" normal "$out" "$d" >/dev/null
    iverilog $IVL_B -g2005 -DNDIM="$n" $defs -DASSERT_ON \
        -DMEMINIT="\"$ROOT/$out/mem_init.hex\"" -o "$out/hs.vvp" \
        tb/tb_handshake.v tb/sim_top.v tb/mem_model.v $core $RTL_COMMON
    local o; o=$(vvp $IVL_M "$out/hs.vvp" 2>&1)
    echo "$o"
    echo "$o" | grep -q "HANDSHAKE VERDICT: PASS" \
        || { red "handshake tests FAILED"; return 1; }
}

# Cycle count, read order and write schedule of matmul_fast at the real N,
# lanes stubbed (see tb/tb_cycles.v). Data is checked by run_sim at N <= 256.
run_cycles() {
    local n="${1:-1024}"
    iverilog $IVL_B -g2005 -DNDIM="$n" -DLANE_STUB -o "build/tbcyc_${n}.vvp" \
        tb/tb_cycles.v rtl/matmul_fast.v $RTL_COMMON
    local o; o=$(vvp $IVL_M "build/tbcyc_${n}.vvp" 2>&1)
    echo "$o" | grep -E 'CYCLES|TB ERROR|VERDICT' || true
    echo "$o" | grep -q "CYCLE HARNESS VERDICT: PASS" \
        || { red "cycle harness FAILED at N=$n"; return 1; }
}

run_regress() {
    local fails=0
    echo "=============== golden-model selftest ==============="
    # golden.py needs a check that does not go through golden.py: small
    # integers make everything exact, so the answer is plain integer matmul.
    if ! python3 scripts/golden.py selftest; then
        red "golden model disagrees with integer arithmetic"; fails=$((fails+1))
    fi

    echo "=============== FP primitives ==============="
    # Wrapped so a failure still reaches the final banner instead of
    # aborting the regression.
    if ! run_fp; then fails=$((fails+1)); fi
    # Icarus for the small sizes: four-state, so uninitialised registers
    # show up as X.
    for cfg in "a 16 normal icarus" "a 16 stress icarus" \
               "a 16 identity icarus" "a 16 extreme icarus" "a 32 normal icarus" \
               "b 8 normal icarus" "b 16 normal icarus" "b 16 stress icarus" \
               "b 16 identity icarus" "b 16 extreme icarus" \
               "a 64 normal verilator" "a 64 stress verilator" \
               "a 64 identity verilator" "a 64 extreme verilator" \
               "a 128 normal verilator" \
               "b 32 normal verilator" "b 64 normal verilator" \
               "b 64 stress verilator" "b 64 identity verilator" \
               "b 64 extreme verilator" \
               "b 128 normal verilator"; do
        # shellcheck disable=SC2086
        set -- $cfg
        echo "=============== design=$1 N=$2 mode=$3 sim=$4 ==============="
        if ! run_sim "$1" "$2" "$3" "$4"; then fails=$((fails+1)); fi
    done
    # matmul_fast needs N >= 16, so N = 8 runs on matmul_4mul only.
    for d in a b; do
        for n in 16 32; do
            echo "=============== handshake design=$d N=$n ==============="
            if ! run_handshake "$d" "$n" >/dev/null; then
                red "handshake FAILED design=$d N=$n"; fails=$((fails+1))
            else
                green "handshake PASS design=$d N=$n"
            fi
        done
    done

    for n in 16 32 64 128 256 512 1024; do
        echo "=============== cycle schedule N=$n ==============="
        if ! run_cycles "$n"; then fails=$((fails+1)); fi
    done

    echo "=============== mutation testing ==============="
    if ! ./scripts/mutate.sh | tail -4; then fails=$((fails+1)); fi

    echo
    if [ "$fails" -eq 0 ]; then green "REGRESSION PASSED"
    else red "REGRESSION FAILED ($fails config(s))"; exit 1; fi
}

case "${1:-}" in
    fp)        run_fp ;;
    handshake) run_handshake "$2" "${3:-16}" ;;
    cycles)    run_cycles "${2:-1024}" ;;
    mutate)    ./scripts/mutate.sh ;;
    sim)     run_sim "$2" "$3" "${4:-normal}" "${5:-icarus}" "${6:-build}" ;;
    regress) run_regress ;;
    full)    if [ "$2" = "a" ]; then
                 red "NOTE: 'full a' needs >4 GiB for Verilator model generation; Ctrl-C if small."
                 red "      './scripts/run.sh cycles 1024' checks the schedule in 5 s."
             fi
             run_sim "$2" 1024 "${3:-normal}" verilator "build/full_$2" ;;
    *)       sed -n '2,10p' "$0"; exit 1 ;;
esac
