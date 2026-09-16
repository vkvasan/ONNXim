#!/usr/bin/env python3
"""Summarise the model x context decode sweep.

The point of this sweep is that attention shape (MHA vs GQA) is INACTIVE at short
context and dominant at long context, because KV traffic grows linearly with the
cached sequence while weight traffic is fixed. So the interesting quantity is not
any single row -- it is how the columns move as context grows.

Reads the ONNXim logs directly; no trace files needed.

Usage:
    python3 scripts/analyze_sweep.py out/sweep
"""
import argparse
import os
import re
import sys


def parse(path):
    """Pull the controller/DRAM counters out of one ONNXim log."""
    try:
        t = open(path).read()
    except OSError:
        return None
    m = re.search(r"Finished at (\d+) cycle", t)
    if not m:
        return {"error": "did not finish"}
    cyc = int(m.group(1))

    rd = re.findall(r"CTRL cmd mix: RD (\d+) WR (\d+) ACT (\d+) PRE (\d+)", t)
    ct = re.findall(r"CTRL ticks (\d+) \| issued (\d+) \([\d.]+%\) \| "
                    r"BLOCKED\(work queued, none ready\) (\d+)", t)
    hits = sum(int(x) for x in re.findall(r"Row hits: (\d+)", t))
    miss = sum(int(x) for x in re.findall(r"Row misses: (\d+)", t))
    conf = sum(int(x) for x in re.findall(r"Row conflicts: (\d+)", t))

    # ROWSPLIT separates the two streams whose locality differs sharply.
    w = re.findall(r"ROWSPLIT weights\s+acc (\d+) hit ([\d.]+)% miss ([\d.]+)% confl ([\d.]+)%", t)
    k = re.findall(r"ROWSPLIT kv\+act\s+acc (\d+) hit ([\d.]+)% miss ([\d.]+)% confl ([\d.]+)%", t)

    out = {"cycles": cyc, "hits": hits, "miss": miss, "conf": conf}
    if rd:
        RD, WR, ACT, PRE = (int(x) for x in rd[-1])
        out.update(RD=RD, WR=WR, ACT=ACT, PRE=PRE, bytes=RD * 32 * 16)
    if ct:
        ticks, iss, blk = (int(x) for x in ct[-1])
        out.update(ticks=ticks, busy=iss + blk)
    if w:
        out["w"] = (int(w[-1][0]), *(float(x) for x in w[-1][1:]))
    if k:
        out["k"] = (int(k[-1][0]), *(float(x) for x in k[-1][1:]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir", nargs="?", default="out/sweep")
    ap.add_argument("--models", nargs="*", default=["llama2-7b", "llama3-8b"])
    ap.add_argument("--ctx", nargs="*", default=["1k", "4k", "16k", "32k"])
    args = ap.parse_args()

    PEAK = 819.2  # B per core-cycle at 1 GHz
    hdr = (f"{'model':<12}{'ctx':>5}{'cycles':>11}{'traffic':>10}{'KVshare':>9}"
           f"{'avgBW':>7}{'busy':>7}{'w.busy':>8}{'wgtHit':>8}{'kvHit':>7}{'kvShare':>9}")
    print(hdr)
    print("-" * len(hdr))
    for m in args.models:
        for c in args.ctx:
            r = parse(os.path.join(args.dir, f"{m}_{c}.log"))
            if r is None:
                print(f"{m:<12}{c:>5}  (missing)")
                continue
            if "error" in r:
                print(f"{m:<12}{c:>5}  ({r['error']})")
                continue
            B = r.get("bytes", 0)
            busy = r.get("busy", 0)
            wacc = r["w"][0] if "w" in r else 0
            kacc = r["k"][0] if "k" in r else 0
            tot = wacc + kacc
            print(f"{m:<12}{c:>5}{r['cycles']:>11,}{B/1e6:>9.0f}M"
                  f"{(kacc/tot*100 if tot else 0):>8.1f}%"
                  f"{B/r['cycles']/PEAK*100:>6.0f}%"
                  f"{(busy/r['ticks']*100 if busy else 0):>6.1f}%"
                  f"{(B/16/(busy*16)*100 if busy else 0):>7.1f}%"
                  f"{(r['w'][1] if 'w' in r else 0):>7.1f}%"
                  f"{(r['k'][1] if 'k' in r else 0):>6.1f}%"
                  f"{(kacc/tot*100 if tot else 0):>8.1f}%")
    print("\n  KVshare = kv+act accesses / all accesses (the mix knob)")
    print("  wgtHit / kvHit are hit/ALL (includes misses) -- the honest metric;")
    print("  hits/(hits+conflicts) stays 95-96% everywhere and hides the effect.")


if __name__ == "__main__":
    main()
