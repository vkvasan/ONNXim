#!/usr/bin/env python3
"""Split an ONNXim DRAM trace from a speculative-decoding run into phases and
streams, and (optionally) compare the per-phase counts with the open-loop
generator's numbers.

    python3 scripts/specdec_analyze.py out/sd.csv out/sd.log
    python3 scripts/specdec_analyze.py out/sd.csv out/sd.log --ref sd.json

Phases come from the scheduler's own log lines
    [SPECDEC] launch <phase> ... at core cycle N
    [SPECDEC] finish <phase> at core cycle N
converted to the DRAM clock with dram_freq/core_freq from --config. Streams
come from the address ranges the simulator logs at start-up
    [MEM] weights <model> allocated at [lo,hi)
    [SPECDEC] request r ... target K base+span V base+span draft K ... V ...
so every trace row is (phase, stream, R/W). A KV write to an address already
written before is a write-after-write (a stale lookahead row being replaced).

Row-hit here is the locality of the ARRIVAL stream (per bank, in issue order),
as in analyze_dram_trace.py; Ramulator2's own counters in the log say what the
controller achieved.
"""
import argparse
import bisect
import json
import os
import re
import sys
from collections import defaultdict


def parse_log(path):
    weights, reqs, phases = {}, {}, []
    rw = re.compile(r"\[MEM\] weights (\S+) allocated at \[(0x[0-9a-f]+),(0x[0-9a-f]+)\)")
    rr = re.compile(r"\[SPECDEC\] request (\d+) cached \d+ kv (target|draft) layer (\d+) "
                    r"K (0x[0-9a-f]+)\+(0x[0-9a-f]+) V (0x[0-9a-f]+)\+(0x[0-9a-f]+)")
    rt = re.compile(r"\[MEM\] tensor (\S+) (\S+) (0x[0-9a-f]+)\+(0x[0-9a-f]+)")
    tensors = []
    rl = re.compile(r"\[SPECDEC\] launch (\w+) : (\d+) requests, (\d+) query tokens, model (\S+) at core cycle (\d+)")
    rf = re.compile(r"\[SPECDEC\] finish (\w+) at core cycle (\d+)")
    ra = re.compile(r"\[SPECDEC\] verify step \d+ done in \d+ cycles, accepted \(req:a\) (.*)")
    with open(path) as f:
        for line in f:
            m = rw.search(line)
            if m:
                weights[m.group(1)] = (int(m.group(2), 16), int(m.group(3), 16)); continue
            m = rt.search(line)
            if m:
                tensors.append((m.group(1), m.group(2), int(m.group(3), 16), int(m.group(4), 16))); continue
            m = rr.search(line)
            if m:
                g = m.groups()
                d = reqs.setdefault(int(g[0]), {})
                kind = "t" if g[1] == "target" else "d"
                d[f"{kind}K{g[2]}"] = (int(g[3], 16), int(g[3], 16) + int(g[4], 16))
                d[f"{kind}V{g[2]}"] = (int(g[5], 16), int(g[5], 16) + int(g[6], 16))
                continue
            m = rl.search(line)
            if m:
                phases.append(dict(name=m.group(1), nreq=int(m.group(2)), ntok=int(m.group(3)),
                                   model=m.group(4), launch=int(m.group(5)), finish=None)); continue
            m = rf.search(line)
            if m and phases and phases[-1]["finish"] is None:
                phases[-1]["finish"] = int(m.group(2)); continue
            m = ra.search(line)
            if m and phases:
                phases[-1]["accepted"] = m.group(1).strip()
    # number the phases: spec step = drafts followed by a verify
    step, di = 0, 0
    for p in phases:
        if p["name"] == "draft":
            p["label"] = f"draft{di}"; p["step"] = step; di += 1
        elif p["name"] in ("verify", "plain"):
            p["label"] = p["name"]; p["step"] = step; step += 1; di = 0
        else:
            p["label"] = p["name"]; p["step"] = None
    return weights, reqs, phases, tensors


class Classifier:
    def __init__(self, weights, reqs, tensors):
        iv = []
        for name, (lo, hi) in weights.items():
            iv.append((lo, hi, ("weight", name)))
        # small parameters (bias, layernorm) are not part of the tile-major matrix
        # walk; the generator does not model them, so keep them apart. The lm head
        # stays "weight": it is never used by the graph, but the padded tail of the
        # swizzled fc2 walk (make_address_tiled pads K to whole tiles) lands in it.
        for model, tname, base, size in tensors:
            if "bias" in tname or ".ln." in tname or "ln_" in tname or tname.endswith("ln"):
                iv.append((base, base + size, ("wparam", model)))
        for r, d in reqs.items():
            for k, (lo, hi) in d.items():
                iv.append((lo, hi, ("kv" if k[0] == "t" else "dkv", r, k[1])))
        iv.sort()
        self.lo = [x[0] for x in iv]
        self.iv = iv

    def __call__(self, addr):
        i = bisect.bisect_right(self.lo, addr) - 1
        # nested: a wparam tensor lies inside its model's weight range, so take
        # the innermost match
        while i >= 0:
            lo, hi, key = self.iv[i]
            if addr < hi:
                if key[0] == "wparam":
                    return key
                # a wparam might start after this weight range's lo but before addr
                j = i + 1
                while j < len(self.iv) and self.iv[j][0] <= addr:
                    if addr < self.iv[j][1] and self.iv[j][2][0] == "wparam":
                        return self.iv[j][2]
                    j += 1
                return key
            i -= 1
        return ("other",)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace")
    ap.add_argument("log")
    ap.add_argument("--config", default=None, help="ONNXim config JSON (for dram_freq/core_freq); "
                    "default configs/_multi_16x32.json")
    ap.add_argument("--ref", default=None, help="JSON from specdec_trace.py --json to compare with")
    ap.add_argument("--per-request", action="store_true", help="also print KV counts per request")
    a = ap.parse_args()

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg = a.config or os.path.join(here, "configs", "_multi_16x32.json")
    with open(cfg) as f:
        j = json.load(f)
    ratio = j["dram_freq"] / j["core_freq"]

    weights, reqs, phases, tensors = parse_log(a.log)
    if not phases:
        sys.exit("no [SPECDEC] phases in the log -- was this a specdec run?")
    cls = Classifier(weights, reqs, tensors)
    starts = [int(p["launch"] * ratio) for p in phases]

    # counts[(phase_idx, stream, rw)] ; rowhit[(phase_idx, stream)] = [hits, total]
    counts = defaultdict(int)
    rowhit = defaultdict(lambda: [0, 0])
    per_req = defaultdict(int)
    written = {}                           # KV address -> phase idx of last write
    open_row = {}
    n = 0
    with open(a.trace) as f:
        next(f)
        for line in f:
            n += 1
            c = line.split(",")
            cyc = int(c[0])
            pi = bisect.bisect_right(starts, cyc) - 1
            if pi < 0:
                pi = 0
            key = cls(int(c[7], 16))
            st = key[0]
            rw = c[8].strip()
            counts[(pi, st, rw)] += 1
            if st in ("kv", "dkv"):
                if a.per_request:
                    per_req[(pi, st, key[1], rw)] += 1
                if rw == "W":
                    addr = int(c[7], 16)
                    if addr in written:
                        counts[(pi, st, "WAW")] += 1
                    written[addr] = pi
            bank = (c[1], c[2], c[3], c[4])
            row = c[5]
            rh = rowhit[(pi, st)]
            rh[1] += 1
            if open_row.get(bank) == row:
                rh[0] += 1
            open_row[bank] = row
    print(f"  {n:,} trace rows, {len(phases)} phases, dram/core clock ratio {ratio:g}")

    streams = ["weight", "kv", "dweight", "dkv", "other"]
    # weight stream split by model
    def stream_of(key):
        return key
    print()
    tot_by_stream = defaultdict(int)
    for pi, p in enumerate(phases):
        cyc = (p["finish"] - p["launch"]) if p["finish"] else 0
        parts = []
        for st in ("weight", "wparam", "kv", "dkv", "other"):
            r = counts.get((pi, st, "R"), 0); w = counts.get((pi, st, "W"), 0)
            if not (r or w):
                continue
            s = f"{st} R {r:,}"
            if w:
                s += f" W {w:,}"
            waw = counts.get((pi, st, "WAW"), 0)
            if waw:
                s += f" (waw {waw:,})"
            rh = rowhit[(pi, st)]
            if rh[1]:
                s += f" hit {rh[0]/rh[1]*100:.1f}%"
            parts.append(s)
            tot_by_stream[st] += r + w
        tag = f"step {p['step']} " if p["step"] is not None else ""
        acc = f"  accepted {p['accepted']}" if "accepted" in p else ""
        print(f"  {tag}{p['label']:15s} {p['model']:14s} {p['nreq']} req {p['ntok']:3d} tok "
              f"{cyc:>10,} core cyc{acc}")
        for s in parts:
            print(f"      {s}")
    tot = sum(tot_by_stream.values())
    if tot:
        print("\n  share: " + "  ".join(f"{st} {v/tot*100:.1f}%" for st, v in tot_by_stream.items()))
    if a.per_request:
        print("\n  per request (phase, stream, req, rw):")
        for k in sorted(per_req):
            print(f"    {k}: {per_req[k]:,}")

    if a.ref:
        with open(a.ref) as f:
            ref = json.load(f)
        print("\n  comparison with open-loop generator (ONNXim / reference):")
        # ONNXim's weight stream is one model per phase; the reference names it
        # weight (target) or dweight (draft)
        bad = 0
        for pi, p in enumerate(phases):
            if p["step"] is None or p["step"] >= len(ref):
                continue
            rp = ref[p["step"]].get(p["label"], {})
            for st_ref, st_sim in (("weight", "weight"), ("dweight", "weight"), ("kv", "kv"), ("dkv", "dkv")):
                for rw in ("R", "W", "WAW"):
                    rv = rp.get(st_ref, {}).get(rw, 0)
                    sv = counts.get((pi, st_sim, rw), 0)
                    if st_ref == "weight" and p["model"] != phases[0]["model"] and "draft" in p["label"]:
                        continue
                    if st_ref == "dweight" and "draft" not in p["label"]:
                        continue
                    if st_ref == "weight" and "draft" in p["label"]:
                        continue
                    if rv == 0 and sv == 0:
                        continue
                    flag = "" if rv == sv else f"   <-- {sv - rv:+,} ({(sv/rv-1)*100:+.2f}%)" if rv else "   <-- ref 0"
                    if flag:
                        bad += 1
                    print(f"    step {p['step']} {p['label']:8s} {st_ref:8s} {rw:3s} {sv:>12,} / {rv:<12,}{flag}")
        print(f"  {'all counts match' if bad == 0 else f'{bad} mismatching counts'}")
        # totals over all speculative phases: insensitive to requests that queue
        # in the per-channel buffer and reach DRAM after the phase's finish cycle
        print("\n  totals over speculative steps (ONNXim / reference):")
        nsteps = max((p["step"] for p in phases if p["step"] is not None), default=-1) + 1
        nsteps = min(nsteps, len(ref))
        agg_sim, agg_ref = defaultdict(int), defaultdict(int)
        for pi, p in enumerate(phases):
            if p["step"] is None or p["step"] >= nsteps:
                continue
            drafty = "draft" in p["label"]
            for rw in ("R", "W", "WAW"):
                agg_sim[("dweight" if drafty else "weight", rw)] += counts.get((pi, "weight", rw), 0)
                agg_sim[("kv", rw)] += counts.get((pi, "kv", rw), 0)
                agg_sim[("dkv", rw)] += counts.get((pi, "dkv", rw), 0)
        for st_i in range(nsteps):
            for ph, d in ref[st_i].items():
                for stq, m in d.items():
                    for rw, v in m.items():
                        agg_ref[(stq, rw)] += v
        for key in sorted(set(agg_sim) | set(agg_ref)):
            sv, rv = agg_sim.get(key, 0), agg_ref.get(key, 0)
            if not (sv or rv):
                continue
            d = f"{(sv/rv-1)*100:+.3f}%" if rv else "ref 0"
            print(f"    {key[0]:8s} {key[1]:3s} {sv:>14,} / {rv:<14,} {d}")


if __name__ == "__main__":
    main()
