# Submission: MoonMath challenge 01, Matrix multiplier machine

**TL;DR.** Machine 1 (max speed): **524,572 cycles** at N = 1024, 284 over
the 524,288-cycle memory-bandwidth floor. Machine 2 (four multipliers):
**268,435,488 cycles**, 32 over the N³/4 compute floor, with exactly 4
multipliers. Both bit-exact against a reference model at
every verified size, both reproducible from a clean checkout. Each machine is
built onto the floor that binds it and measured against that floor rather than
against itself.

| | **Machine 1**, max speed<br>`rtl/matmul_fast.v` | **Machine 2**, four multipliers<br>`rtl/matmul_4mul.v` |
|---|---|---|
| **cycles at N = 1024** | **524,572** | **268,435,488** |
| floor, and what binds it | 524,288, read bandwidth | 268,435,456, multipliers |
| **efficiency** | **99.946%** | **99.99999%** |
| **time** | **≈ 1.35 ms** (524,572 ÷ 389 MHz) | **≈ 0.67 s** (268,435,488 ÷ 401 MHz) |
| Fmax, and where it was measured | 389 MHz synthesised at N = 16. **Not measured at N = 1024, where it will fall** | 401 MHz synthesised at N = 8. The state does not grow with N |
| binary16 multipliers | 4,096 | **4** |
| binary32 adders / converters | 7,168 / 1,024 | 4 / 4 |
| on-chip storage | ~2.02 MiB | ~3.3 kbit of registers, no SRAM |
| **verified** | schedule at N = 1024; data bit-exact, N = 16…256 (why that split holds: [Verification](#verification)) | full 1024×1024 run, bit-exact |

Times are cycles ÷ a measured small-N Fmax against a free 45nm library.
They are floors, not signoff numbers; machine 1's is the softer
([Clock](#clock)).

One fast check, after `./scripts/setup_toolchain.sh`, if you want a result
before reading further: `./scripts/run.sh cycles 1024` (5 s, reproduces the
524,572).

---

## Approach

**The memory floor.** Every element of A and B affects the result, so each
must cross the 64-bit read port at least once: 2N² elements ÷ 4 per cycle =
**N²/2 = 524,288 cycles**. No amount of arithmetic beats this.

**The compute floor.** N³ = 1,073,741,824 MACs at 4 per cycle =
**268,435,456 cycles** for the four-multiplier machine, 512× the memory
floor, so that machine is overwhelmingly compute-bound and memory is
effectively free.

The two designs are therefore near-opposites, and each is optimal for its own
constraint. Both landed within 0.06% of their cycle floor, which meant all the
remaining speed was in the clock. That work is in [Clock](#clock), below the
machines it applies to.

## Machine 1, maximum speed: 524,572 cycles (99.946% of the floor, schedule measured at N = 1024)

`matmul_fast.v`. Load all of B into 1024 on-chip column-banks (2 MiB), then
stream A once. Each A word yields four elements, each of which contributes to
all 1024 columns of one row of C, so 4,096 MACs per cycle.

```
mem_read ──► [load B]  ───► 1024 column-banks: physically ONE
  64b/cyc │                  65,536-bit × 256 memory, one shared rd addr
          └─► [stream A] ──► broadcast: 4 elements to all 1024 lanes
                                         │
              ┌─ lane (× 1024) ──────────▼─────────────────────┐
              │ 4 binary16 muls → tree → 4 rotating binary32   │
              │ partials (3-stage adder, 4-cycle dependency)   │
              └──────────────────────────┬─────────────────────┘
                 one row of C done       │
                 every 256 cycles        ▼
              2 row buffers: combine the 4 partials, round to binary16
                                         │
mem_write ◄── 4 elements/cycle: drains in exactly 256 cycles,
  64b/cyc     hidden under the next row's reads
```

Hitting the floor needs three things simultaneously: read each matrix exactly
once, never idle the read port, and hide all 262,144 writes under the reads.
The third is what eliminates most natural schedules, and it is why B (not A)
is the resident operand: **because B is resident, one row of C (1024 elements,
256 write words) completes exactly every 256 cycles, one row of A read, and
drains in exactly 256 cycles**, so writeback runs at 100% write-port occupancy
with only two row buffers and never collides. With A resident instead, no
element of C is finished until the last word of B arrives, so writeback cannot
be hidden at all.

Of the 284 cycles over the floor, 256 are that final row of C draining after
the last A word. That much is irreducible, since the last element of C cannot
exist before the last word of A is read. Another 23 are pipeline fill and
command handshake (13 of them added when the adder was pipelined, buying the
shorter clock), and 5 are the gap between the load-B and stream-A phases, 3 of
which are the read latency's hard minimum. **Only 2 of the 284, the slack in
that phase gap, are genuinely free.**

A design that keeps C on chip instead reads just as efficiently but cannot
write anything until the last k step, which turns writeback into a third phase
and costs 50% of the cycle count.

Key structural trick: the B banks all share one read address, because the four
k values needed each cycle are the same for every column. So 1024 banks is
physically one 65,536-bit-wide × 256-deep memory, not 1024 independent ones.

Cost: 4,096 binary16 multipliers at 50% duty, 7,168 binary32 adders and 1,024
converters. That is 12,288 arithmetic instances, with seven adders per lane the
price of the four partials and the combining tree below, plus ~2.02 MiB of
on-chip storage. All of it is what "area is not a concern" sanctions. The 50% duty is not
slack to reclaim: every MAC falls in the stream-A phase, where one A word must
hit all 1024 columns in a single cycle, so halving the array doubles that phase.

**Four partial accumulators per lane, not one.** A single accumulator makes the
adder a one-cycle loop-carried dependency, where `acc` feeds the adder which
feeds `acc`, and that is exactly the structure that cannot be pipelined. It is why
the machine asked to be *fastest* had a worse critical path than the
four-multiplier one. Rotating over four partials makes it a four-cycle
dependency, so a three-stage adder retires with a cycle to spare, and a
balanced tree combines the four once per row inside the 256-cycle writeback
drain, where it is free.

This changes the order of the floating-point reduction and therefore some
result bits, since addition is not associative. The reference model replays the
new order and the comparison is still bit-exact with no tolerance. It is also
slightly more accurate, since four chains of 64 terms combined by a tree
accumulate less error than one chain of 256. It costs one constraint: N ≥ 16,
since a row must hold at least four k-groups.

## Machine 2, four multipliers: 268,435,488 cycles (99.99999% of the floor)

`matmul_4mul.v`. The only question is whether the four multipliers ever idle,
which reduces to an arithmetic-intensity requirement of **1 MAC per element
read**. An R×C output tile gives R·C/(R+C) MACs per element read:

- 1×4 ("one A element × one B word", the tempting shape, since it maps
  perfectly onto four multipliers and the 64-bit word) → **0.80, starves**
- 2×2 → 1.00 nominal, but only half of each B word is used, so 0.67 effective
- **4×4 → 2.00, chosen**

The memory word width picks 4×4 for us: such a tile needs exactly one whole
64-bit word of B per k, and one whole word of A per tile row per four k
values. Nothing read is wasted, and the resulting 2× bandwidth margin lets a
plain double buffer hide all prefetch.

One microstep = a 4×4 tile advanced by 4 k values: 8 reads, 64 MACs, 16
cycles. Within it, cycle t broadcasts one A element against one B word.
Iterating the tile row on the *inner* index means each accumulator group is
touched once every four cycles rather than on four consecutive cycles. Both
orders give the identical bit pattern, so this is headroom rather than
correctness: it leaves the loop-carried accumulate path three spare cycles, so
the implemented three-stage binary32 adder can retire without changing the
compute schedule. The other order would have made the accumulate path a true
one-cycle loop-carried dependency that no amount of pipelining could fix
without restructuring. Concretely, within one microstep (row = t mod 4,
k offset = t div 4):

```
t           0     1     2     3     4     5     6     7    …   15
A element   a[0]  a[1]  a[2]  a[3]  a[0]  a[1]  a[2]  a[3]  …   one per cycle
B word      ─── word for k ──────── ─── word for k+1 ──────     held 4 cycles
acc group   0     1     2     3     0     1     2     3     …   4 cycles apart
```

4 × 268,435,456 = N³ exactly: every multiplier does a useful MAC on every
cycle. Total overhead is 32 cycles, the prefetch prologue and the final drain,
of which 3 come from the adder pipelining.

Because that headroom was already there, this machine's adder is pipelined
three deep with **no change to a single result bit**: only re-timing was
needed, moving the write port (which accumulator, valid, initialise, and the
product to initialise *with*) and the writeback snapshot three cycles later.

Cost: 4 binary16 multipliers, 4 binary32 adders, 4 output converters and
~3.3 kbit of registers. No SRAM; only address widths scale with N.

## Clock

"As fast as possible" means time, and time is cycles ÷ Fmax. Measuring logic
depth with yosys put the whole four-multiplier machine and the binary32 adder
alone at the same 156 gate levels. The adder was the entire critical path: one
module was setting the clock of both machines while 4,096 multipliers waited on
it with 3× slack. Cutting that adder into three balanced stages takes the adder
to **71 levels**, the fast machine to **75** and the four-multiplier machine
to **82**, for 13 extra cycles (524,559 → 524,572) and 3 out of 268 million.
`./scripts/depth.sh` reproduces the pipelined figures and the adder both ways;
the un-pipelined whole-machine figures are recorded measurements only, since
the un-pipelined RTL predates the repository.

Gate levels are a proxy, so it was then measured properly with yosys and
berkeley-abc against Nangate45, a free 45nm library, via `./scripts/synth.sh`:

| | delay | Fmax | area | at |
|---|---:|---:|---:|---|
| binary32 adder, combinational | 3.56 ns | 281 MHz | 2,356 um2 | |
| binary32 adder, 3-stage | **2.15 ns** | **466 MHz** | 3,573 um2 | |
| four-multiplier machine, comb-adder | 3.92 ns | 255 MHz | 32,483 um2 | N=8 |
| four-multiplier machine, pipelined | **2.50 ns** | **401 MHz** | 37,629 um2 | N=8 |
| fast machine, pipelined | **2.57 ns** | **389 MHz** | 539,275 um2 | N=16 |

The gate-level proxy that drove the decision proved optimistic by 20-33% on
speed and far worse on area (it scored a flip-flop the same as a NAND); only
the measured figures above are claimed.

Not claimed: the absolute MHz belongs to an old free library and only the
ratio transfers; the fast machine's own before/after was never measured; there
is no place-and-route, no timing closure and no power figure. The areas are at
the small N synthesised, which bites the fast machine only, since its
arithmetic scales with N. The four-multiplier machine's state is fixed-size
(the operand double-buffers, 16 accumulators plus their writeback snapshot, the
replay registers) with N entering only counter widths, so 37,629 um2 should sit
close to the real machine. The two areas are still not comparable to *each
other*, being at different N. The fast
machine's clock will also likely fall at N = 1024, where its drain mux is 256:1
rather than the 4:1 synthesised at N = 16.

After the change the machines are *deeper* than the adder, 2.50 ns
(four-multiplier) and 2.57 ns (fast) against 2.15, most likely because the
accumulator read mux now sits on top of its first stage. No gate-level path
report was captured, so that attribution is inference; cutting the mux is the
plausible next step, not a proven one.

## Numerics

Products are computed in binary16 and accumulated in binary32, rounded once to
binary16 on writeback, which is what tensor cores and TPUs do. Here it is
nearly free because of a useful fact:

**binary16 × binary16 is always exactly representable in binary32.** The
product significand needs 22 bits (fits in 24) and the exponent lands in
[−48, +31] (well inside binary32 normals). So the multipliers contain no
rounding logic at all, and every rounding error in either machine comes from
the accumulator or the output conversion.

A second consequence: every binary16 is a multiple of 2⁻²⁴, so every product
is a multiple of 2⁻⁴⁸, and that invariant survives rounding. A nonzero
accumulator value is therefore always ≥ 2⁻⁴⁸, which is 78 binary orders above
the binary32 subnormal ceiling of 2⁻¹²⁶. **binary32 subnormals are unreachable**,
so the adder flushes them, which is dead-code removal rather than a corner
cut. binary16 subnormals *are* reachable and are fully implemented on both
input and output.

Measured accuracy against a float64 reference: normwise relative error
2.08 × 10⁻⁴, essentially the binary16 rounding unit (4.88 × 10⁻⁴), and **flat
from N = 64 to N = 1024**. The output format, not the arithmetic, is the
accuracy limit.

## Verification

Both machines are verified **bit-exact**, with no tolerances anywhere, against a
reference model that replays each design's own reduction order in numpy
float32. The two designs associate their reductions differently (balanced tree
vs sequential chain) and so legitimately differ in the last bit; a shared
tolerance-based reference would have been weaker.

- 48,206 directed and random vectors on the three arithmetic primitives,
  74,495 applications in total, including 5,000 constructed exact midpoints, the only
  vectors that can distinguish round-to-nearest-even from round-half-up. The
  adder's 26,289 are run twice, against both `STAGES = 0` and `STAGES = 3`,
  which must return identical bits: the pipelined version is the combinational
  one with registers cut into it, so anything else means a stage boundary is
  in the wrong place.
- 21-configuration end-to-end regression: the fast machine at N = 16…128
  (its N ≥ 16 constraint is why 8 is skipped), the four-multiplier machine at
  N = 8…128, four stimulus modes (normal / subnormal-and-overflow stress /
  B = identity / infinities-NaNs-negative-zeros), two simulators (Icarus
  Verilog 11.0 and Verilator 4.038); the fast machine's data check extends to
  N = 256 in a separately invoked configuration.
- Command-interface tests: back-to-back operations, reset asserted
  mid-operation, and `multiply_en` held high across completion.
- The memory model polices region boundaries and write counts on every cycle
  and the testbench **asserts** them; the C region is prefilled with a poison
  pattern so any word never written is caught.
- **The four-multiplier machine was run at the full 1024×1024 problem**: all
  268,435,488 cycles, all 262,144 result words bit-exact.
- **The max-speed machine is verified in two halves at N = 1024**, because the
  expensive part of it is not the part the schedule depends on. Its 1024-lane
  array is a pure consumer, since no control signal is derived from a lane
  output, so a build with the lanes stubbed out has a bit-identical schedule, costs
  12 MB instead of the >4 GiB Verilator needs for 12,288 arithmetic instances,
  and returns the identical cycle count wherever both harnesses can run
  (N = 16…256). So: **data** bit-exact at N = 16…256 with the full datapath,
  and **schedule** at the real N = 1024: 524,572 cycles, all 524,288 read
  addresses checked against the two ascending streams, all 262,144 C words
  written exactly once and none after `ready`. What is not simulated at full
  size is the fast machine's arithmetic; a lane's datapath contains no N-dependent
  logic, since only its B-bank depth scales, so N = 1024 asks the arithmetic
  nothing N = 256 did not.

**The verification is itself verified.** `scripts/mutate.sh` injects 32
single-line faults (reassociated adder trees, off-by-one indices, inverted
lane decodes, a shortened phase gap, a truncated flush, swapped read phases,
ties-to-even silently downgraded to round-half-up) and requires the flow to
catch every one. Nineteen go against the data flow, nine against the three
arithmetic primitives, and four against the cycle harness that carries the
N = 1024 claim, since that harness has to be shown capable of failing on its
own terms too. Two initially survived. One exposed a real testbench hole:
nothing checked that all writes complete *before* `ready` rises, now a
monitored assertion. The other was a genuinely equivalent mutation, replaced
with one that is not. Separate fault injection found a second hole: nothing
checked that the vector files had actually loaded (an unread array is X, and
`X !== X` is false, so those vectors passed vacuously). Both holes are now
closed. A regression that has never been shown capable of failing is not
evidence, and this one now has been.

## Assumptions

The interface fixes less than it looks like. The assumptions that matter
most:

- **Word addressing** is forced, not chosen: only a 64-bit-word address puts B
  at exactly 256K, and only word addressing fits A, B and C into a 20-bit
  address at all.
- **The read port is pipelined** (a request every cycle, each returning 3
  cycles later). This is the load-bearing assumption; it follows from "it can
  read four elements and write four elements every clock cycle", which is
  false under a blocking read. If it is wrong, machine 1 is over-provisioned
  by 4×, the wrong machine, and not a free change either: the 3-cycle return
  decode is a shift register clocked off `mem_read_en`, with no return-valid
  input, so it would need a real valid signal rather than a re-parameterisation.
- **Row-major storage.** Convention, not stated. Machine 1 is indifferent to
  B's order (a column-major B is actually easier for it), though a
  column-major A would need transposing on the fly; machine 2 needs two
  address expressions swapped.
- **binary32 accumulation is permitted.** Measured, binary16 accumulation over
  K = 1024 costs 4.5 bits of the eleven available (22.6x the error), enough to
  make the arithmetic rather than the output format the accuracy limit.
- **"Four multipliers" counts binary16 multipliers.** The design uses exactly
  four, plus four adders and four output converters, so it also complies with
  a "four MACs" reading.

## Repository

```
rtl/     matmul_fast.v, matmul_4mul.v, and three IEEE-754 primitives
tb/      memory model with protocol checking, Icarus and Verilator flows
scripts/ rootless toolchain bootstrap, golden model, one-command regression
```

Verilog-2001 throughout, so it runs unmodified on any tool.
`./scripts/setup_toolchain.sh && ./scripts/run.sh regress` reproduces the
verification results from a clean checkout with no root privileges, about 35
minutes on four cores, everything except two long runs invoked separately:
`./scripts/run.sh full b` (the four-multiplier machine at the full 1024×1024,
a few minutes) and `./scripts/run.sh sim a 256 normal verilator build/a256`
(the fast machine's largest data check). `full a` is not the way to check the
fast machine at full size, because Verilator needs >4 GiB to elaborate its
12,288 arithmetic instances. `cycles 1024` is that measurement. The timing table is
`./scripts/depth.sh` plus `./scripts/synth.sh`.
