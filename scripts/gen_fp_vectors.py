#!/usr/bin/env python3
"""Bit-exact stimulus/expected vectors for the three FP primitives. numpy
IEEE-754 RNE is the reference; NaNs are canonicalised and binary32 subnormals
flushed (DAZ/FTZ) to match the RTL. Writes vectors/*.hex + fp_counts.vh."""
import numpy as np
import os

OUT = os.path.join(os.path.dirname(__file__), "..", "vectors")
os.makedirs(OUT, exist_ok=True)

rng = np.random.default_rng(0xC0FFEE)


def f16_bits(x):
    return int(np.float16(x).view(np.uint16))


def f32_bits(x):
    return int(np.float32(x).view(np.uint32))


def canon32(x):
    x = np.float32(x)
    return 0x7FC00000 if np.isnan(x) else f32_bits(x)


def canon16(x):
    x = np.float16(x)
    return 0x7E00 if np.isnan(x) else f16_bits(x)


MIN_NORM32 = np.float32(2.0 ** -126)


def flush32(x):
    """FTZ/DAZ: a binary32 subnormal becomes a zero of the same sign."""
    x = np.float32(x)
    if x != 0 and np.isfinite(x) and np.abs(x) < MIN_NORM32:
        return np.float32(-0.0) if np.signbit(x) else np.float32(0.0)
    return x


# binary16 encodings chosen to hit every branch: zeros, infinities, NaNs,
# subnormal extremes, smallest/largest normals, exact powers of two.
SPECIAL16 = [
    0x0000, 0x8000,          # +-0
    0x0001, 0x8001,          # +-smallest subnormal (2^-24)
    0x0200, 0x03FF, 0x83FF,  # mid / largest subnormal
    0x0400, 0x8400,          # +-smallest normal (2^-14)
    0x3C00, 0xBC00,          # +-1.0
    0x3C01,                  # 1 + 2^-10 (mantissa LSB set)
    0x4000, 0x4900,          # 2.0, 10.0
    0x7BFF, 0xFBFF,          # +-largest normal (65504)
    0x7C00, 0xFC00,          # +-inf
    0x7E00, 0x7C01,          # quiet NaN, signalling NaN
]


def gen_fp16_mul():
    """64-bit records: {a[15:0], b[15:0], y[31:0]}."""
    rows = []
    for a in SPECIAL16:                       # full special x special cross
        for b in SPECIAL16:
            rows.append((a, b))
    # Signed finite values (normals and subnormals, both signs).
    def rand_fin(k, hi):
        return [(int(m) | (int(sg) << 15)) for m, sg in
                zip(rng.integers(0, hi, k), rng.integers(0, 2, k))]
    rows += list(zip(rand_fin(4000, 0x7C00), rand_fin(4000, 0x7C00)))
    # Signed subnormals only: the leading-one detector at its widest.
    rows += list(zip(rand_fin(1500, 0x0400), rand_fin(1500, 0x0400)))
    rows += [(int(v), int(w)) for v, w in
             zip(rng.integers(0, 0x10000, 4000), rng.integers(0, 0x10000, 4000))]
    lines = []
    for a, b in rows:
        fa = np.uint16(a).view(np.float16)
        fb = np.uint16(b).view(np.float16)
        with np.errstate(all="ignore"):
            y = np.float32(fa) * np.float32(fb)
        lines.append("%04x%04x%08x" % (a, b, canon32(y)))
    return lines


def gen_fp32_add():
    """96-bit records: {a[31:0], b[31:0], y[31:0]}."""
    special32 = [
        0x00000000, 0x80000000,              # +-0
        0x3F800000, 0xBF800000,              # +-1.0
        0x00800000, 0x80800000,              # +-smallest normal
        0x7F7FFFFF, 0xFF7FFFFF,              # +-largest normal
        0x7F800000, 0xFF800000, 0x7FC00000,  # +-inf, NaN
        0x3F800001, 0x3F7FFFFF,              # 1+ulp, 1-ulp
        0x00000001, 0x80000001,              # +-smallest subnormal -> DAZ
        0x007FFFFF, 0x807FFFFF,              # +-largest subnormal  -> DAZ
    ]
    rows = [(a, b) for a in special32 for b in special32]
    # Subnormal against arbitrary normals, to cover DAZ in the swap network
    # (both operand orders) rather than only in the zero+zero shortcut.
    for _ in range(1500):
        sub = (int(rng.integers(0, 2)) << 31) | int(rng.integers(1, 1 << 23))
        nor = (int(rng.integers(0, 2)) << 31) | (int(rng.integers(1, 255)) << 23) \
              | int(rng.integers(0, 1 << 23))
        rows.append((sub, nor))
        rows.append((nor, sub))

    # Exact fp16xfp16 products -- what the accumulator actually sees.
    # Signed, so the effective-subtract path is exercised too.
    def rand_fin16(k):
        mag = rng.integers(0, 0x7BFF, k).astype(np.uint16)
        sgn = (rng.integers(0, 2, k).astype(np.uint16) << 15)
        return (mag | sgn).view(np.float16)

    a16 = rand_fin16(6000)
    b16 = rand_fin16(6000)
    with np.errstate(all="ignore"):
        prods = (a16.astype(np.float32) * b16.astype(np.float32))
    rows += [(f32_bits(prods[i]), f32_bits(prods[i + 1]))
             for i in range(0, 6000, 2)]
    # Wildly mismatched exponents drive the alignment-shift clamp and the
    # sticky path; near-equal magnitudes drive massive cancellation and the
    # leading-zero normaliser.
    for _ in range(3000):
        e1 = int(rng.integers(1, 254))
        e2 = int(rng.integers(1, 254))
        x = (int(rng.integers(0, 2)) << 31) | (e1 << 23) | int(rng.integers(0, 1 << 23))
        yv = (int(rng.integers(0, 2)) << 31) | (e2 << 23) | int(rng.integers(0, 1 << 23))
        rows.append((x, yv))
    for _ in range(3000):
        e1 = int(rng.integers(1, 254))
        m = int(rng.integers(0, 1 << 23))
        x = (e1 << 23) | m
        d = int(rng.integers(0, 8))
        yv = 0x80000000 | (e1 << 23) | max(0, m - d)
        rows.append((x, yv))

    # EXACT TIES: the only vectors that tell round-to-nearest-EVEN apart
    # from round-half-up; random sampling never lands on a midpoint.
    # Construction: b = exactly half an ulp of a, so a + b is a midpoint and
    # RNE must round to the even neighbour.
    for _ in range(4000):
        e = int(rng.integers(30, 250))          # leaves room for e-24 >= 1
        m = int(rng.integers(0, 1 << 23))
        sg = int(rng.integers(0, 2)) << 31
        a_tie = sg | (e << 23) | m
        b_tie = sg | ((e - 24) << 23)           # +half ulp, same sign
        rows.append((a_tie, b_tie))
        rows.append((b_tie, a_tie))             # and with the operands swapped
        rows.append((a_tie, b_tie ^ 0x80000000))  # -half ulp: tie on subtract
    # Ties where both operands share an exponent (the shift-0 path).
    for _ in range(1000):
        e = int(rng.integers(2, 253))
        a_tie = (e << 23) | int(rng.integers(0, 1 << 23))
        rows.append((a_tie, a_tie))             # exact doubling, never a tie
        rows.append((a_tie, a_tie ^ 0x00000001))

    lines = []
    for a, b in rows:
        fa = flush32(np.uint32(a).view(np.float32))    # DAZ
        fb = flush32(np.uint32(b).view(np.float32))
        with np.errstate(all="ignore"):
            y = flush32(fa + fb)                       # FTZ
        lines.append("%08x%08x%08x" % (a, b, canon32(y)))
    return lines


def gen_fp32_cvt():
    """48-bit records: {a[31:0], y[15:0]}."""
    rows = [
        0x00000000, 0x80000000, 0x3F800000, 0xBF800000,
        0x7F800000, 0xFF800000, 0x7FC00000,
        0x477FE000,   # 65504.0  -> largest fp16 normal
        0x477FF000,   # 65520.0  -> exact tie at the overflow threshold -> inf
        0x477FEFFF,   # just below the tie -> 65504
        0x38800000,   # 2^-14    -> smallest fp16 normal
        0x33800000,   # 2^-24    -> smallest fp16 subnormal
        0x33000000,   # 2^-25    -> exact tie to even -> +0
        0x33000001,   # just above 2^-25 -> rounds up to 2^-24
        0x387FE000,   # largest fp16 subnormal
        0x00000001, 0x007FFFFF,   # binary32 subnormals (DAZ -> +0)
    ]
    rows += [int(v) for v in rng.integers(0, 1 << 32, 6000)]
    # Concentrate on the exponent band where fp16 rounding actually happens.
    for _ in range(6000):
        e = int(rng.integers(90, 160))
        rows.append((int(rng.integers(0, 2)) << 31) | (e << 23)
                    | int(rng.integers(0, 1 << 23)))
    lines = []
    for a in rows:
        fa = np.uint32(a).view(np.float32)
        if (a >> 23) & 0xFF == 0:
            fa = np.float32(-0.0 if a >> 31 else 0.0)     # DAZ, see docstring
        with np.errstate(all="ignore"):
            y = np.float16(fa)
        lines.append("%08x%04x" % (a, canon16(y)))
    return lines


def write(name, lines):
    with open(os.path.join(OUT, name), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return len(lines)


if __name__ == "__main__":
    n_mul = write("fp16mul.hex", gen_fp16_mul())
    n_add = write("fp32add.hex", gen_fp32_add())
    n_cvt = write("fp32cvt.hex", gen_fp32_cvt())
    with open(os.path.join(OUT, "fp_counts.vh"), "w") as fh:
        fh.write("`define NMUL %d\n`define NADD %d\n`define NCVT %d\n"
                 % (n_mul, n_add, n_cvt))
    print("fp16_mul_to_fp32 : %6d vectors" % n_mul)
    print("fp32_add         : %6d vectors" % n_add)
    print("fp32_to_fp16     : %6d vectors" % n_cvt)
