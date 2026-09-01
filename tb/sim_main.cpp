// Verilator harness: same stimulus as the Icarus testbench, cycle-driven so
// N=1024 runs in reasonable time. NDIM must match -DNDIM; COUT = C dump path.
#include "Vsim_top.h"
#include "verilated.h"

#include <cstdio>
#include <ctime>

#ifndef NDIM
#define NDIM 16
#endif
#ifndef COUT_PATH
#define COUT_PATH "build/c_out.hex"
#endif

static const long long N   = NDIM;
static const long long WPR = N / 4;
static const long long NW  = N * WPR;

static Vsim_top *top;

static inline void tick() {
    top->clk = 0;
    top->eval();
    top->clk = 1;
    top->eval();
}

int main(int argc, char **argv) {
    Verilated::commandArgs(argc, argv);
    top = new Vsim_top;

    top->rst = 1;
    top->multiply_en = 0;
    top->dbg_addr = 0;
    for (int i = 0; i < 8; i++) tick();
    top->rst = 0;
    for (int i = 0; i < 2; i++) tick();

    if (!top->ready) {
        fprintf(stderr, "TB ERROR: ready is not asserted after reset\n");
        return 1;
    }

    // multiply_en held across exactly one rising edge.
    top->multiply_en = 1;
    tick();
    top->multiply_en = 0;
    long long cycles = 1;

    if (top->ready) {
        // Not a warning: the wait loop below would exit immediately and the
        // harness would print a normal-looking RESULT line with cycles=1.
        fprintf(stderr, "TB ERROR: ready did not deassert after multiply_en\n");
        return 1;
    }

    // 8x the analytic worst case (the four-multiplier machine).
    const long long limit = 8LL * (N * N * N / 4 + 4096);
    const long long report_every = 20000000LL;
    long long next_report = report_every;
    clock_t t0 = clock();

    while (!top->ready) {
        tick();
        if (++cycles > limit) {
            fprintf(stderr, "TB ERROR: timeout after %lld cycles\n", cycles);
            return 1;
        }
        if (cycles >= next_report) {
            double el = double(clock() - t0) / CLOCKS_PER_SEC;
            fprintf(stderr, "  ... %lld cycles (%.0f cyc/s)\n",
                    cycles, cycles / (el > 0 ? el : 1));
            next_report += report_every;
        }
    }

    printf("RESULT N=%lld cycles=%lld reads=%u writes=%u mem_errors=%u\n",
           N, cycles, top->n_reads, top->n_writes, top->n_errors);

    // The Icarus flow asserts both of these (tb_matmul.v); this one used to
    // only print them, so a region violation or a short write count passed
    // silently at every size run under Verilator -- which is every size above
    // 32, including the full N=1024 runs.
    if (top->n_errors != 0) {
        fprintf(stderr, "TB ERROR: memory model reported %u region violation(s)\n",
                top->n_errors);
        return 1;
    }
    if (top->n_writes != (unsigned)NW) {
        fprintf(stderr, "TB ERROR: wrote %u words to C, expected %lld\n",
                top->n_writes, (long long)NW);
        return 1;
    }

    FILE *f = fopen(COUT_PATH, "w");
    if (!f) { perror("fopen"); return 1; }
    for (long long i = 0; i < NW; i++) {
        top->dbg_addr = (unsigned)(2 * NW + i);
        top->eval();
        fprintf(f, "%016llx\n", (unsigned long long)top->dbg_data);
    }
    fclose(f);

    delete top;
    return 0;
}
