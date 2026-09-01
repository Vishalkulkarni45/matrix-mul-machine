# Matrix multiplier machine: MoonMath challenge 01

Two hardware designs that compute **C = A × B** for 1024 × 1024 IEEE-754
binary16 matrices through a single 64-bit external-memory interface with a
3-cycle pipelined read latency.

Challenge: <https://zro.moonmath.ai/challenges/matrix-multiplier-machine>

```bash
./scripts/setup_toolchain.sh     # rootless: fetches iverilog + verilator
./scripts/run.sh regress         # everything: FP, 21 configs, N=1024 schedule,
                                 # handshake, mutation
```

---

## Results

| | **Design A**, "as fast as possible" | **Design B**, "only four multipliers" |
|---|---|---|
| | [`rtl/matmul_fast.v`](rtl/matmul_fast.v) | [`rtl/matmul_4mul.v`](rtl/matmul_4mul.v) |
| **cycles at N = 1024** | **524,572** | **268,435,488** |
| theoretical floor | 524,288, memory-bound | 268,435,456, compute-bound |
| **efficiency** | **99.946%** | **99.99999%** |
| **critical path** | **75 levels** (was 157) | **82 levels** (was 156) |
| binary16 multipliers | 4,096 | 4 (exactly the budget) |
| binary32 adders | 7,168 | 4 |
| on-chip storage | ~2.02 MiB | ~3.3 kbit |
| read-port utilisation | 100% | 50% |
| multiplier utilisation | 50% | 100% |
| correctness | bit-exact, N = 16…256 | **bit-exact at the full N = 1024** |
| cycles at N=1024 established by | **direct measurement** (schedule) | direct measurement |

Both designs are verified **bit-exact**, not within a tolerance, against an
IEEE-754 reference model that replays each machine's own reduction order.
Design B was run at the actual problem size, all 268 million cycles.

Design A is verified in two halves, because its 1024-lane array is what makes a
full-size simulation expensive and the lane array is not what the schedule
depends on:

- **Data**, bit-exact against the reference model at N = 16…256, full datapath.
- **Schedule**, measured at the real N = 1024: 524,572 cycles, 524,288 reads
  in exactly the two ascending streams, and 262,144 writes each landing on a
  distinct word of C with none omitted. This works by building the same RTL with its
  lanes stubbed out ([`tb/tb_cycles.v`](tb/tb_cycles.v), `./scripts/run.sh
  cycles 1024`). No control signal in the design is derived from a lane output,
  so removing them leaves every timing property intact; it costs 12 MB and five
  seconds instead of the >4 GiB Verilator needs to elaborate 12,288 arithmetic
  instances. It agrees with the full-datapath testbench at every one of the five
  sizes where both flows can run (N = 16…256), and is mutation-tested in its own
  right.

`matmul_4mul` also elaborates cleanly at N = 1024 under
`verilator --lint-only -Wall`; Verilator's linter runs out of memory on
`matmul_fast`'s 4,096-instance array at full size.

The verification is itself verified: `./scripts/run.sh mutate` injects 32
single-line faults, 19 against the data flow and 4 schedule faults against the
cycle harness, and requires the flow to catch every one. Two initially
survived and both were testbench holes, now closed.

## Why each design is the shape it is

**Design A is bandwidth-bound.** Every element of A and B must cross the read
port at least once: 2N² elements at 4 per cycle = N²/2 = 524,288 cycles, and
no amount of arithmetic can beat that. Hitting it requires reading each matrix
exactly once *and* overlapping all 262,144 writes underneath the reads. So:
load all of B into 1024 column-banks (2 MiB), then stream A once. Each A word
produces 4,096 MACs against a row-slice of B. The decisive property is that
one row of C finishes exactly every 256 cycles and drains in exactly 256
cycles, so writeback is free.

**Design B is compute-bound**, by 512×. Memory is irrelevant; the only
question is whether four multipliers ever idle. The feed requirement is 1 MAC
per element read, which rules out the tempting "one A element × one B word"
dataflow (0.8 MAC/element, so it *starves*) and selects a 4×4 output-stationary
tile (2.0 MAC/element). Then iterating the tile row on the inner loop index
leaves the accumulate adder three cycles of loop-carried slack at zero cost,
which is what lets its adder be pipelined without touching the schedule, and
without changing a single result bit.

## Cycles are not speed

The challenge asks machine 1 to be *"as fast as possible"*, and speed is
`cycles ÷ Fmax`. Both machines were already within 0.06% of their **cycle**
floor, so everything left was in the clock, and that is measurable without a
synthesis licence. `./scripts/depth.sh` reports longest register-to-register
paths in generic gates (a **proxy**; the measured numbers are below):

| block | before | after | gain |
|---|---|---|---|
| `fp16_mul_to_fp32` | 50 | 50 | |
| `fp32_to_fp16` | 76 | 76 | |
| `fp32_add` alone | 156 | 71 | 2.2× |
| **`matmul_fast` whole design** | **157** | **75** | **2.1×** |
| **`matmul_4mul` whole design** | **156** | **82** | **1.9×** |

The whole-design figures are the ones that set the clock, and they are *worse*
than the adder's 71, most likely because the accumulator read mux now rides on
top of the adder's first stage, Design B's 16:1 `acc[r_q*4 + c]` being wider
than Design A's 4:1 `part[rd_px]`. Real synthesis confirms the machines are
deeper than the adder, though no path report was captured, so which gate is
responsible is inference.

Before pipelining, the whole-design numbers equalled the adder's: one module
set the clock of both machines while 4,096 multipliers waited on it with 3×
slack. Cutting it into three balanced stages (`fp32_add #(.STAGES(3))`, the
same arithmetic with registers inserted, verified bit-identical on all 26,289
adder vectors) takes the adder from 156 levels to **71** and the whole designs
to 75 and 82.

**This made the cycle counts slightly worse, on purpose.** Stated plainly so it
is not mistaken for a regression:

| | before | after |
|---|---|---|
| `matmul_fast` cycles | 524,559 | 524,572, **+13** |
| `matmul_4mul` cycles | 268,435,485 | 268,435,488, **+3** |
| critical path | 157 / 156 levels | **75 / 82** |

`time = cycles ÷ Fmax`. Giving up **0.0025%** of the cycles bought a large
clock gain. Both machines were already at 99.95% of their cycle floor; all the
remaining speed was in the clock, and nobody had measured it.

Design B needed no restructuring, since the four-cycle accumulator rotation was
already there for exactly this, and **its results are bit-identical**. Design
A did: a single accumulator is a one-cycle loop-carried path through the adder,
which cannot be pipelined at all, so it now interleaves four partial
accumulators per lane and combines them with a tree once per row, inside the
writeback drain where it is free. That **changes the reduction order and
therefore some result bits**; the reference model replays the new order and the
comparison stays bit-exact with no tolerance. It is also slightly more
accurate, and it costs one constraint: `matmul_fast` now needs N ≥ 16.

### Then it was measured, and the proxy was wrong

`./scripts/synth.sh` runs yosys + berkeley-abc against **Nangate45**, a free 45nm
library. No licence, no account.

| | delay | Fmax | area | at |
|---|---:|---:|---:|---|
| `fp32_add` combinational | 3.56 ns | 281 MHz | 2,356 µm² | |
| `fp32_add` 3-stage | **2.15 ns** | **466 MHz** | 3,573 µm² | |
| `matmul_4mul` comb-adder | 3.92 ns | 255 MHz | 32,483 µm² | N = 8 |
| `matmul_4mul` pipelined | **2.50 ns** | **401 MHz** | 37,629 µm² | N = 8 |
| `matmul_fast` pipelined | **2.57 ns** | **389 MHz** | 539,275 µm² | N = 16 |

| | proxy claimed | measured | area cost |
|---|---:|---:|---:|
| `fp32_add` block | 2.2× | **1.66×** | +52% |
| `matmul_4mul` machine | 1.9× | **1.57×** | +15.8% |
| `matmul_fast` machine | 2.1× | *not measured* | n/a |

The proxy was optimistic by 20–33% every time, and its area figure of "3.4%
more cells" was badly wrong, because counting generic gates scores a flip-flop
the same as a NAND. The decision it drove was still correct; only the magnitude was.

**What is not claimed:** the absolute MHz is specific to an old free library and
only the ratio transfers; the areas are for the small N synthesised, at
*different* N for the two machines, so they are not comparable to each other;
Design A's clock will likely fall at N = 1024 (a 256:1 drain mux against a much
narrower one at N = 16); and there is no place-and-route, no timing closure and
no power figure. `matmul_fast`'s own before/after was never measured.

With those caveats, ≈**1.35 ms** and ≈**0.67 s** at N = 1024.

Seven alternative schedules and six candidate tile shapes were costed and
rejected before these two.

## Layout

```
rtl/     matmul_fast.v  matmul_4mul.v          the two deliverables
         fp16_mul_to_fp32.v  fp32_to_fp16.v
         fp32_add.v                             STAGES = 0 or 3, one datapath
tb/      sim_top.v                              device + memory subsystem,
                                                shared by both flows
         tb_cycles.v                            schedule of matmul_fast at the
                                                real N, lanes stubbed
         mem_model.v                            memory, 3-cycle read latency,
                                                region policing
         tb_matmul.v                            Icarus testbench
         sim_main.cpp                           Verilator harness
         tb_fp_units.v                          arithmetic regression
         tb_handshake.v                         command-interface tests
scripts/ setup_toolchain.sh                     rootless simulator bootstrap
         run.sh                                 one entry point for everything
         depth.sh                               logic depth, before/after
         golden.py                              stimulus, golden models, checker
         gen_fp_vectors.py                      FP primitive vectors
         mutate.sh                              32-fault mutation campaign
```

Verilog-2001 throughout, so it runs unmodified on Icarus, Verilator and any
commercial tool. Both designs take `N` as a parameter (default 1024, power of
two ≥ 8) purely so the *identical* RTL can be verified at simulable sizes.

## Two things worth knowing before reading the RTL

**binary16 × binary16 is exactly representable in binary32.** The 22-bit
product significand fits in binary32's 24, and the exponent range
[−48, +31] sits well inside binary32's normals. So the multipliers contain no
rounding logic at all, and binary32 accumulation is free at the multiply
stage.

**No value in either accumulator can ever be a binary32 subnormal.** Every
binary16 is a multiple of 2⁻²⁴, so every product is a multiple of 2⁻⁴⁸, and
that invariant survives rounding, so a nonzero accumulator is always
≥ 2⁻⁴⁸, which is 78 binary orders above the subnormal ceiling of 2⁻¹²⁶. The
adder therefore flushes subnormals, which is dead-code removal rather than a
corner cut. The claim is proved *and* asserted on every value in the reference
model across all regression modes.
