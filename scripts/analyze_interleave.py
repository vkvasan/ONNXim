#!/usr/bin/env python3
"""Why does row-buffer hit rate fall when core count rises?

The hypothesis is mechanical: with N cores, each bank receives requests from
several cores interleaved, so a row opened for core A is closed to serve core B
before core A returns to it. This script tests that directly, by walking each
bank's request sequence and asking, at every row change, whether the requester
also changed.

    row change WITH a core switch    -> caused by interleaving  (N-core specific)
    row change WITHOUT a core switch -> the stream's own walk    (present at N=1)

CAVEAT (open loop): the trace records ARRIVAL order at the controller, not
service order. FR-FCFS reorders, so the served hit rate is higher than what is
computed here. The point is the comparison between core counts on the same
measure, not the absolute value -- both are distorted identically.

Usage:
    python3 scripts/analyze_interleave.py out/trace/c1.csv out/trace/c4.csv
"""
import argparse
import collections
import sys


def analyze(path, max_rows):
    """Walk each bank's arrival sequence; attribute row changes to interleaving."""
    # bank -> (last_row, last_core)
    last = {}
    hits = changes = change_with_switch = 0
    per_bank_cores = collections.defaultdict(set)
    n = 0
    with open(path) as f:
        header = next(f)
        for line in f:
            if n >= max_rows:
                break
            p = line.split(',')
            try:
                ch, pch, bg, ba = int(p[1]), int(p[2]), int(p[3]), int(p[4])
                row, core = int(p[5]), int(p[9])
            except (ValueError, IndexError):
                continue
            n += 1
            bank = (ch, pch, bg, ba)
            per_bank_cores[bank].add(core)
            if bank in last:
                prow, pcore = last[bank]
                if row == prow:
                    hits += 1
                else:
                    changes += 1
                    if core != pcore:
                        change_with_switch += 1
            last[bank] = (row, core)
    return {
        "n": n, "hits": hits, "changes": changes, "switch": change_with_switch,
        "banks": len(per_bank_cores),
        "mean_cores_per_bank": (sum(len(v) for v in per_bank_cores.values())
                                / max(1, len(per_bank_cores))),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("traces", nargs="+")
    ap.add_argument("--max-rows", type=int, default=3_000_000)
    args = ap.parse_args()

    print(f"{'trace':<22}{'reqs':>10}{'banks':>7}{'cores/bank':>12}"
          f"{'arr.hit':>9}{'rowchg':>9}{'w/ core switch':>16}")
    print("-" * 86)
    for t in args.traces:
        try:
            r = analyze(t, args.max_rows)
        except OSError as e:
            print(f"{t:<22}  (unreadable: {e})")
            continue
        tot = r["hits"] + r["changes"]
        if tot == 0:
            print(f"{t:<22}  (no usable rows)")
            continue
        name = t.split('/')[-1]
        print(f"{name:<22}{r['n']:>10,}{r['banks']:>7}{r['mean_cores_per_bank']:>12.2f}"
              f"{r['hits']/tot*100:>8.1f}%{r['changes']:>9,}"
              f"{r['switch']/max(1,r['changes'])*100:>15.1f}%")
    print("\n  cores/bank      = distinct requesters seen by an average bank")
    print("  arr.hit         = consecutive same-row arrivals to a bank (open loop, pessimistic)")
    print("  w/ core switch  = share of row changes where the requester also changed")
    print("                    -> this is the fraction attributable to interleaving")


if __name__ == "__main__":
    main()
