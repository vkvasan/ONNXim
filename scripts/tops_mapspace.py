#!/usr/bin/env python3
"""TOPS mapspace search: fixed hardware + fixed workload -> best mapping.

    hardware (cores, array dim, scratchpad, accumulator)
    workload (GEMM shape)
        ->  search Tiling x Ordering x Parallelism x Stationarity

This is the Timeloop-style question, and it is different from choosing hardware.
Hardware is fixed at fabrication; the mapping is a compiler decision made per
GEMM on that fixed machine.

TRAFFIC MODEL
    For C[I,J] = A[I,K] x B[K,J] tiled (tI, tJ, tK), each operand is re-read
    once per iteration of the loops outside it:

        A traffic = I*K * ceil(J/tJ)      re-read for every J tile
        B traffic = K*J * ceil(I/tI)      re-read for every I tile
        C traffic = I*J * ceil(K/tK) * 2  read+write per K tile (accumulation)

    Which operand escapes re-reading is exactly STATIONARITY:
        weight-stationary  keeps B resident -> J outermost, B read once
        output-stationary  keeps C resident -> K innermost, C written once
        input-stationary   keeps A resident

CAPACITY CONSTRAINT (from Mapping.cc:55)
        tI*tK + tK*tJ <= spad_bytes / precision      operands
        tI*tJ         <= acc_bytes / 4               accumulator (fp32)

WHAT IS ALREADY SETTLED BY MEASUREMENT (so it is asserted, not searched)
    parallelism   head-parallel beat request- and seq-parallel by 3.3x. The
                  loss was load imbalance (one core got 611,200 rows, others
                  6,528), not memory behaviour.
    ordering      within a tile ONNXim emits address-sorted, and sorting vs
                  logical order measured 0.08% -- immaterial.
    stationarity  provably null at batch 1: with no reuse every weight crosses
                  the array boundary exactly once, so WS == OS. Searched here
                  because it does matter at batch > 1.

Usage
    python3 scripts/tops_mapspace.py --gemm qkv --batch 128
    python3 scripts/tops_mapspace.py --gemm ffn --batch 8 --spad 4096
"""
import argparse

GEMMS = {                      # (I=tokens, J=out, K=in) for LLaMA-2 7B
    "qkv":  ("batch", 12288, 4096),
    "proj": ("batch", 4096, 4096),
    "ffn":  ("batch", 11008, 4096),
    "down": ("batch", 4096, 11008),
}


def traffic(I, J, K, tI, tJ, tK, prec):
    """DRAM elements moved, including re-reads forced by the tiling."""
    a = I * K * -(-J // tJ)
    b = K * J * -(-I // tI)
    c = I * J * -(-K // tK) * 2          # accumulate: read + write
    return (a + b + c) * prec


def feasible(tI, tJ, tK, spad_b, acc_b, prec):
    return (tI * tK + tK * tJ) * prec <= spad_b and tI * tJ * 4 <= acc_b


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gemm", choices=GEMMS, default="qkv")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--dim", type=int, default=32, help="systolic array dim")
    ap.add_argument("--spad", type=int, default=4096, help="scratchpad KB")
    ap.add_argument("--acc", type=int, default=2048, help="accumulator KB")
    ap.add_argument("--tile-depth", type=int, default=3)
    ap.add_argument("--prec", type=int, default=2)
    ap.add_argument("--top", type=int, default=10)
    a = ap.parse_args()

    _, J, K = GEMMS[a.gemm]
    I = a.batch
    d = a.dim
    spad_b = a.spad * 1024 // a.tile_depth      # one bank of the N-deep pipeline
    acc_b = a.acc * 1024 // a.tile_depth
    print(f"  GEMM {a.gemm}: C[{I}, {J}] = A[{I}, {K}] x B[{K}, {J}]   prec {a.prec} B")
    print(f"  hardware: {d}x{d} array, spad {a.spad} KB / tile_depth {a.tile_depth} "
          f"= {spad_b//1024} KB usable, acc {acc_b//1024} KB\n")

    mults = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    cand = []
    for mi in mults:
        tI = min(I, d * mi)
        for mj in mults:
            tJ = min(J, d * mj)
            for mk in mults:
                tK = min(K, d * mk)
                if not feasible(tI, tJ, tK, spad_b, acc_b, a.prec):
                    continue
                t = traffic(I, J, K, tI, tJ, tK, a.prec)
                # which operand is stationary = which is NOT re-read
                stat = ("output" if tK >= K else
                        "weight" if tI >= I else
                        "input" if tJ >= J else "none")
                cand.append((t, tI, tJ, tK, stat))
    if not cand:
        print("  no feasible tiling -- scratchpad too small"); return
    cand.sort()
    best = cand[0][0]
    print(f"  {'rank':<5}{'tI':>6}{'tJ':>7}{'tK':>7}{'traffic MB':>13}{'vs best':>9}"
          f"{'stationary':>12}{'spad KB':>10}")
    print("  " + "-" * 72)
    seen = set()
    shown = 0
    for t, tI, tJ, tK, stat in cand:
        if (tI, tJ, tK) in seen:
            continue
        seen.add((tI, tJ, tK))
        shown += 1
        if shown > a.top:
            break
        print(f"  {shown:<5}{tI:>6}{tJ:>7}{tK:>7}{t/1e6:>13.1f}{t/best:>8.2f}x"
              f"{stat:>12}{(tI*tK+tK*tJ)*a.prec//1024:>10}")
    t, tI, tJ, tK, stat = cand[0]
    minimum = (I * K + K * J + I * J) * a.prec
    print(f"\n  BEST  tI={tI} tJ={tJ} tK={tK}, {stat}-stationary")
    print(f"    traffic {t/1e6:.1f} MB vs {minimum/1e6:.1f} MB if every operand were "
          f"read exactly once ({t/minimum:.2f}x)")
    print(f"    T tiling      {tI} x {tJ} x {tK}")
    print(f"    O ordering    {stat} operand outermost; sorted within a tile "
          f"(measured immaterial, 0.08%)")
    print(f"    P parallelism head-parallel (measured 3.3x better than request/seq)")
    print(f"    S stationarity {stat}")


if __name__ == "__main__":
    main()
