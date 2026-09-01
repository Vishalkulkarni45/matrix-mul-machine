#!/usr/bin/env python3
"""Golden model, stimulus generator and checker: replays each machine's
exact summation order in numpy float32 and compares bit for bit (design A:
balanced 4-wide trees over NPART rotating partials; design B: sequential).

Usage:
    golden.py gen   <N> <mode> <outdir> [a|b|both]
    golden.py check <N> <design> <outdir>
    golden.py selftest
"""
import numpy as np
import os
import sys

POISON = "deadbeefdeadbeef"
MIN_NORM32 = np.float32(2.0 ** -126)


def assert_no_subnormals(x, where):
    """Empirical check of the no-subnormal claim behind FTZ/DAZ."""
    mag = np.abs(x)
    bad = (mag > 0) & (mag < MIN_NORM32)   # inf/NaN fail both tests already
    if bad.any():
        raise AssertionError(
            "binary32 subnormal reached %s -- the FTZ/DAZ no-subnormal "
            "claim is wrong for this data" % where)


# ---------------------------------------------------------------------------
# Stimulus
# ---------------------------------------------------------------------------
def make_matrices(n, mode, seed=0x5EED):
    rng = np.random.default_rng(seed)
    if mode == "normal":
        # Moderate dynamic range: nothing saturates, so mismatches are
        # attributable to logic.
        a = (rng.standard_normal((n, n)) * 0.5).astype(np.float16)
        b = (rng.standard_normal((n, n)) * 0.5).astype(np.float16)
    elif mode == "stress":
        # Exact zeros, fp16 subnormals, values near the fp16 maximum, and
        # mixed signs to force cancellation.
        def mk():
            m = (rng.standard_normal((n, n)) * 0.5).astype(np.float16)
            flat = m.reshape(-1)
            k = flat.size
            idx = rng.permutation(k)
            flat[idx[: k // 10]] = np.float16(0.0)
            sub = rng.integers(1, 0x0400, k // 10).astype(np.uint16)
            sign = rng.integers(0, 2, k // 10).astype(np.uint16) << 15
            flat[idx[k // 10: 2 * (k // 10)]] = (sub | sign).view(np.float16)
            big = rng.uniform(100.0, 60000.0, max(1, k // 50)).astype(np.float16)
            flat[idx[2 * (k // 10): 2 * (k // 10) + big.size]] = big
            return flat.reshape(n, n)
        a, b = mk(), mk()
    elif mode == "extreme":
        # Sparse infinities, NaNs and negative zeros: the only mode that
        # exercises non-finite propagation end to end. Sparse so most of C
        # stays finite and the test still discriminates.
        def mk():
            m = (rng.standard_normal((n, n)) * 0.5).astype(np.float16)
            flat = m.reshape(-1)
            idx = rng.permutation(flat.size)
            take = max(1, flat.size // 512)
            flat[idx[0:take]]          = np.float16(np.inf)
            flat[idx[take:2*take]]     = np.float16(-np.inf)
            flat[idx[2*take:3*take]]   = np.float16(np.nan)
            flat[idx[3*take:4*take]]   = np.float16(-0.0)
            return flat.reshape(n, n)
        a, b = mk(), mk()
    elif mode == "identity":
        # C must come back exactly equal to A, so address-generation and
        # transposition errors show up as structure, not noise.
        a = (rng.standard_normal((n, n)) * 0.5).astype(np.float16)
        b = np.eye(n, dtype=np.float16)
    else:
        raise SystemExit("unknown mode %r" % mode)
    return a, b


# ---------------------------------------------------------------------------
# Exact per-design reference reductions
# ---------------------------------------------------------------------------
NPART = 4          # interleaved partial accumulators per lane in matmul_fast


MIN_N_FAST = 4 * NPART      # matmul_fast needs WPR = N/4 >= NPART


def golden_fast(a, b):
    """Design A: balanced 4-wide tree per k-group; groups round-robined over
    NPART partial accumulators, combined by a balanced tree at the end.

    The round-robin must be replayed exactly: FP addition is not associative
    and the comparison in cmd_check() is bit-exact with no tolerance."""
    n = a.shape[0]
    assert n >= MIN_N_FAST and (n // 4) % NPART == 0, (
        f"matmul_fast needs N >= {MIN_N_FAST} with N/4 a multiple of {NPART}; got {n}")
    a32 = a.astype(np.float32)
    b32 = b.astype(np.float32)
    part = [None] * NPART
    for g in range(n // 4):
        k0 = 4 * g
        p = [a32[:, k0 + c:k0 + c + 1] * b32[k0 + c:k0 + c + 1, :]
             for c in range(4)]
        for x in p:
            assert_no_subnormals(x, "a product")
        s01 = p[0] + p[1]
        s23 = p[2] + p[3]
        s = s01 + s23
        assert_no_subnormals(s01, "tree level 1")
        assert_no_subnormals(s23, "tree level 1")
        assert_no_subnormals(s, "tree level 2")
        j = g % NPART
        part[j] = s if part[j] is None else part[j] + s
        assert_no_subnormals(part[j], "a partial accumulator")
    c01 = part[0] + part[1]
    c23 = part[2] + part[3]
    acc = c01 + c23
    assert_no_subnormals(c01, "partial combine level 1")
    assert_no_subnormals(c23, "partial combine level 1")
    assert_no_subnormals(acc, "partial combine level 2")
    return acc.astype(np.float16)


def golden_4mul(a, b):
    """Design B: strict sequential reduction over k."""
    n = a.shape[0]
    a32 = a.astype(np.float32)
    b32 = b.astype(np.float32)
    acc = None
    for k in range(n):
        p = a32[:, k:k + 1] * b32[k:k + 1, :]
        assert_no_subnormals(p, "a product")
        acc = p if acc is None else acc + p
        assert_no_subnormals(acc, "the accumulator")
    return acc.astype(np.float16)


# ---------------------------------------------------------------------------
# Memory image packing: word addressing, row-major, four binary16 elements
# per 64-bit word, element c at bits [16c+15 : 16c].
# ---------------------------------------------------------------------------
def canon_nan16(m):
    """Canonicalise NaN to the quiet NaN (0x7E00) the RTL emits: numpy
    propagates the operand's sign instead, and IEEE-754 only specifies
    "some NaN"."""
    u = m.view(np.uint16).copy()
    u[np.isnan(m)] = np.uint16(0x7E00)
    return u.view(np.float16)


def pack_words(m):
    n = m.shape[0]
    u = m.view(np.uint16).reshape(n, n // 4, 4)
    return [("%04x%04x%04x%04x" % (w[3], w[2], w[1], w[0]))
            for row in u for w in row]


def unpack_words(lines, n):
    out = np.zeros(n * n, dtype=np.uint16)
    for w, line in enumerate(lines):
        v = int(line, 16)
        for c in range(4):
            out[w * 4 + c] = (v >> (16 * c)) & 0xFFFF
    return out.view(np.float16).reshape(n, n)


def cmd_gen(n, mode, outdir, which="both"):
    os.makedirs(outdir, exist_ok=True)
    a, b = make_matrices(n, mode)
    nw = n * n // 4

    words = pack_words(a) + pack_words(b) + [POISON] * nw
    assert len(words) == 3 * nw
    with open(os.path.join(outdir, "mem_init.hex"), "w") as fh:
        fh.write("\n".join(words) + "\n")

    if which not in ("a", "b", "both"):
        raise SystemExit("design must be 'a', 'b' or 'both', got %r" % which)
    wanted = ("a", "b") if which == "both" else (which,)
    # matmul_fast does not exist below MIN_N_FAST, so "both" at a smaller N
    # means "both designs that exist at this N".
    if "a" in wanted and n < MIN_N_FAST:
        if which == "a":
            raise SystemExit(
                "design a (matmul_fast) requires N >= %d; got %d" % (MIN_N_FAST, n))
        print("note: skipping design a golden, N=%d < %d" % (n, MIN_N_FAST))
        wanted = tuple(x for x in wanted if x != "a")
    for name, fn in (("a", golden_fast), ("b", golden_4mul)):
        if name not in wanted:
            continue
        c = canon_nan16(fn(a, b))
        with open(os.path.join(outdir, "golden_%s.hex" % name), "w") as fh:
            fh.write("\n".join(pack_words(c)) + "\n")

    # Reference in float64 for a quality figure that is independent of the
    # hardware's association order.
    ref = a.astype(np.float64) @ b.astype(np.float64)
    np.save(os.path.join(outdir, "ref64.npy"), ref)
    print("generated N=%d mode=%s  (%d words per matrix)" % (n, mode, nw))


def cmd_selftest(n=16):
    """Check the golden models against arithmetic that cannot round.

    Everything else is checked against golden.py, so this check must not go
    through golden.py. Small non-negative integer inputs keep every product
    and partial sum exactly representable, so nothing rounds, the reduction
    order stops mattering, and the answer is plain integer matrix
    multiplication computable with Python ints. This verifies the indexing
    and reduction structure, not the rounding behaviour (which the FP vector
    suite covers)."""
    assert n % 16 == 0, "use a multiple of 16 so matmul_fast's NPART divides WPR"
    rng = np.random.default_rng(0xC0FFEE)
    ai = rng.integers(0, 8, size=(n, n), dtype=np.int64)
    bi = rng.integers(0, 8, size=(n, n), dtype=np.int64)

    worst = int((ai.max() * bi.max()) * n)
    assert worst <= 2048, f"result {worst} exceeds the exactly-representable range"

    ref = ai @ bi                                   # plain Python/NumPy integers
    a16 = ai.astype(np.float16)
    b16 = bi.astype(np.float16)
    assert (a16.astype(np.int64) == ai).all(), "stimulus not exact in binary16"

    bad = 0
    for name, fn in (("golden_fast", golden_fast), ("golden_4mul", golden_4mul)):
        # A broken model may raise or return the wrong shape; both must
        # report as failures, not escape as a broken checker.
        try:
            out = fn(a16, b16)
        except Exception as exc:                      # noqa: BLE001
            print("  FAIL  %-12s raised %s: %s" % (name, type(exc).__name__, exc))
            bad += 1
            continue
        if out.shape != ref.shape:
            print("  FAIL  %-12s returned shape %s, expected %s"
                  % (name, out.shape, ref.shape))
            bad += 1
            continue
        got = out.astype(np.int64)
        if not (got == ref).all():
            where = np.argwhere(got != ref)
            print("  FAIL  %-12s disagrees on %d of %d elements"
                  % (name, len(where), n * n))
            for i, j in where[:3]:
                print("          C[%d][%d]: model %d, integer arithmetic %d"
                      % (i, j, got[i, j], ref[i, j]))
            bad += 1
        else:
            print("  ok    %-12s agrees with integer arithmetic on %d elements"
                  % (name, n * n))

    if bad:
        print("GOLDEN SELFTEST: FAIL (%d model(s) disagree)" % bad)
        raise SystemExit(1)
    print("GOLDEN SELFTEST: PASS (N=%d, values 0..7, nothing rounds)" % n)


def cmd_check(n, design, outdir):
    gpath = os.path.join(outdir, "golden_%s.hex" % design)
    with open(gpath) as fh:
        gold_lines = [l.strip() for l in fh if l.strip()]
    with open(os.path.join(outdir, "c_out.hex")) as fh:
        got_lines = [l.strip() for l in fh if l.strip()]

    if len(got_lines) != len(gold_lines):
        print("FAIL: got %d words, expected %d"
              % (len(got_lines), len(gold_lines)))
        return 1

    unwritten = sum(1 for l in got_lines if l == POISON)
    if unwritten:
        print("FAIL: %d of %d C words were never written"
              % (unwritten, len(got_lines)))
        return 1

    bad = [i for i, (g, x) in enumerate(zip(gold_lines, got_lines)) if g != x]
    if bad:
        print("FAIL: %d of %d words differ from the golden model"
              % (len(bad), len(gold_lines)))
        for i in bad[:8]:
            print("   word %6d  got %s  expected %s"
                  % (i, got_lines[i], gold_lines[i]))
        return 1

    # Bit-exact against the hardware's own order. Now report how good that
    # order actually is, against an independent float64 reference.
    got = unpack_words(got_lines, n)
    ref = np.load(os.path.join(outdir, "ref64.npy"))
    with np.errstate(all="ignore"):
        finite = np.isfinite(got) & np.isfinite(ref)
        # Elements the hardware turned non-finite while the reference stayed
        # finite are counted and reported rather than dropped from the
        # average.
        nonfinite = int((~finite).sum())
        # 'top 3 decades' is measured against the largest reference magnitude
        # over the WHOLE matrix, not over the surviving subset.
        ref_scale = np.abs(ref[np.isfinite(ref)]).max() if np.isfinite(ref).any() else 0.0
        if finite.any() and ref_scale > 0:
            g = got[finite].astype(np.float64)
            r = ref[finite]
            err = np.abs(g - r)
            # Normwise relative error: the elementwise version is dominated
            # by near-cancellations, which say nothing about the hardware.
            fro = np.sqrt((err ** 2).sum()) / np.sqrt((r ** 2).sum())
            big = np.abs(r) > ref_scale * 1e-3
            worst = (err[big] / np.abs(r[big])).max() if big.any() else 0.0
            qual = ("normwise rel err = %.3e, worst elementwise (top 3 decades)"
                    " = %.3e, non-finite elements = %d"
                    % (fro, worst, nonfinite))
        else:
            qual = ("no finite reference entries to compare (non-finite = %d)"
                    % nonfinite)
    print("PASS: %d/%d words bit-exact; %s" % (len(gold_lines), len(gold_lines), qual))
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    cmd = sys.argv[1]
    if cmd == "selftest":
        cmd_selftest()
    elif cmd == "gen":
        cmd_gen(int(sys.argv[2]), sys.argv[3], sys.argv[4],
                sys.argv[5] if len(sys.argv) > 5 else "both")
    elif cmd == "check":
        if sys.argv[3] not in ("a", "b"):
            raise SystemExit("design must be 'a' or 'b', got %r" % sys.argv[3])
        sys.exit(cmd_check(int(sys.argv[2]), sys.argv[3], sys.argv[4]))
    else:
        raise SystemExit(__doc__)
