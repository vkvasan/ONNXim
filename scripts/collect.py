#!/usr/bin/env python3
"""Parse a sweep directory into the table the report quotes.

    python3 scripts/collect.py out/sweep
"""
import re, sys, os, glob

def parse(path):
    t = open(path, errors="ignore").read()
    if "Simulation time" not in t:
        return None
    cyc = [int(x) for x in re.findall(r"Total cycle: (\d+)", t)]
    hits = [int(x) for x in re.findall(r"Row hits: (\d+)", t)]
    miss = [int(x) for x in re.findall(r"Row misses: (\d+)", t)]
    conf = [int(x) for x in re.findall(r"Row conflicts: (\d+)", t)]
    bw   = [int(x) for x in re.findall(r"avg BW utilization (\d+)%", t)]
    rd   = [int(x) for x in re.findall(r"\((\d+) reads", t)]
    wr   = [int(x) for x in re.findall(r"(\d+) writes\)", t)]
    if not (cyc and hits):
        return None
    tot = sum(hits) + sum(miss) + sum(conf)
    return dict(cycles=max(cyc),
                rowhit=100.0 * sum(hits) / tot if tot else 0,
                conflict=100.0 * sum(conf) / tot if tot else 0,
                avgbw=sum(bw) / len(bw) if bw else 0,
                reads=sum(rd), writes=sum(wr))

d = sys.argv[1]
rows = {}
for f in sorted(glob.glob(os.path.join(d, "*.log"))):
    tag = os.path.basename(f)[:-4]
    if len(tag.split("_")) != 3:      # ignore logs that are not sweep points
        continue
    r = parse(f)
    if r: rows[tag] = r

hdr = f"{'kv':<6}{'parallel':<9}{'swizzle':<8}{'cycles':>12}{'row hit':>9}{'conflict':>10}{'avg BW':>8}"
print(hdr); print("-" * len(hdr))
base = {}
for tag in sorted(rows, key=lambda t: (t.split('_')[0] != 'head', t)):
    kv, par, w = tag.split("_")
    r = rows[tag]
    print(f"{kv:<6}{par:<9}{w[1]:<8}{r['cycles']:>12,}{r['rowhit']:>8.1f}%"
          f"{r['conflict']:>9.1f}%{r['avgbw']:>7.0f}%")
    base[(kv, par, w)] = r["cycles"]

print()
for par in ("head", "request", "seq"):
    for w in ("w1", "w0"):
        h, b = base.get(("head", par, w)), base.get(("block", par, w))
        if h and b:
            print(f"  {par:<8} swizzle={w[1]}   head-major is {b/h:.2f}x faster than block-major")
for kv in ("head", "block"):
    a, b = base.get((kv, "head", "w1")), base.get((kv, "head", "w0"))
    if a and b: print(f"  {kv:<8} swizzle on vs off: {b/a:.2f}x")
h, r_, s = (base.get((kv, p, "w1")) for kv, p in (("head","head"),("head","request"),("head","seq")))
if h and r_ and s:
    print(f"  head-parallel vs request-parallel {r_/h:.2f}x, vs seq-parallel {s/h:.2f}x")
