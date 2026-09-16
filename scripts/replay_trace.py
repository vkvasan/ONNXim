#!/usr/bin/env python3
"""Reconstruct any window of a compact whole-generation trace.

    tops_visible.py --emit-compact DIR   ->  DIR/{weights.u64,blocks.jsonl,manifest.json}
    replay_trace.py DIR --step N --layer L  ->  the addresses for that window

WHY THE TRACE IS STORED THIS WAY
    A flat trace of a whole generation is ~11-16 TB, and 99.999% of it is the
    same weight address list repeated once per (layer, step). Storing the
    STRUCTURE instead of the samples is ~100,000x smaller and loses nothing:

        weights   step- and layer-INVARIANT. One canonical list; layer L is the
                  same list plus L * layer_stride.
        KV        a pure function of (block table, ctx, layout, block size).
                  The block tables come from the vLLM serving sim and are the
                  only thing that changes from step to step -- which is exactly
                  why the serving sim has to run the whole generation rather
                  than a snapshot.

    So any window is regenerated on demand, at full fidelity, without the
    generator or the serving sim in the loop.

Usage
    python3 scripts/replay_trace.py DIR --info
    python3 scripts/replay_trace.py DIR --step 40 --layer 0 --out window.trace
    python3 scripts/replay_trace.py DIR --step 40 --layer 0 --kv-only --request 3
"""
import argparse
import array
import json
import os


def load(d):
    man = json.load(open(os.path.join(d, "manifest.json")))
    w = array.array("Q")
    with open(os.path.join(d, "weights.u64"), "rb") as f:
        w.frombytes(f.read())
    return man, w


def steps(d):
    with open(os.path.join(d, "blocks.jsonl")) as f:
        for line in f:
            yield json.loads(line)


# --- timing model, MEASURED from a multi-step ONNXim run -------------------
# out/ms5_ts.csv: 4 requests x 5 generated tokens, 40M requests, real cycles.
#
#   within a burst   7.91 req/cycle, 87% of gaps are 0 and the rest 1 --
#                    an almost perfectly regular stream
#   stalls           234 gaps > 10 cycles totalling 4.3% of runtime, evenly
#                    spaced at ~9/27/44/62/79/97% through the trace. Those are
#                    OPERATION boundaries (the core finishing one GEMM and
#                    starting the next), not memory backpressure.
#
# The rate is stable: 7.29-7.79 req/cycle across the six sixths of that run,
# and 7.90 on a separate single-step run. So it is a property of the core, and
# one measured run parameterises many.
#
# CAVEAT: this holds only where memory KEEPS UP. In the KV-heavy, low-row-hit
# regime the core genuinely stalls on DRAM -- block-major measured 11.06 req/cyc
# demand against head-major's 22.11 on identical work, with array utilisation
# 47.7% vs 76.6%. Re-measure the rate for such a configuration rather than
# reusing these constants.
RATE_REQ_PER_CYCLE = 7.91
STALL_FRACTION = 0.043


def stamp(addrs, n_boundaries=1, rate=RATE_REQ_PER_CYCLE,
          stall_frac=STALL_FRACTION):
    """Attach arrival cycles to a demand-order address list.

    Injects at the measured steady rate, then inserts the measured stall budget
    split evenly across `n_boundaries` operation boundaries.
    """
    n = len(addrs)
    busy = n / rate
    total = busy / (1.0 - stall_frac)
    stall_each = (total - busy) / max(n_boundaries, 1)
    out, c, per = [], 0.0, max(n // max(n_boundaries, 1), 1)
    for i, ad in enumerate(addrs):
        if i and i % per == 0:
            c += stall_each
        c += 1.0 / rate
        out.append((int(c), ad))
    return out


def align(a, req=32):
    return a - (a % req)


def kv_for(req_id, ctx, blocks, man):
    """Regenerate one request's KV addresses. Mirrors kv_addrs_parallel."""
    dk, nkvh, prec = man["dk"], man["nkvh"], man["precision"]
    block, layout = man["block"], man["layout"]
    base = man["kv_base"] + req_id * (1 << 28)
    step_e = 32 // prec
    nb = -(-ctx // block)
    out = []
    for h in range(nkvh):
        for s in range(ctx):
            if layout == "contiguous":
                o = h * ctx * dk + s * dk
            else:
                phys, tok = blocks[s // block], s % block
                if layout == "head":
                    o = h * nb * block * dk + phys * block * dk + tok * dk
                else:
                    o = phys * nkvh * block * dk + h * block * dk + tok * dk
            for e in range(0, dk, step_e):
                out.append(align(base + (o + e) * prec))
    return out


def kv_writes_for(req_id, ctx, blocks, man):
    """Addresses WRITTEN this step: the newly appended token's K and V.

    Decode appends exactly one token per request per step, so the write set is
    the last slot of the cache -- seq index ctx-1 -- for every KV head. Same
    address arithmetic as the read path, one token instead of all of them.
    Weights are read-only; output-projection writes are not modelled.
    """
    dk, nkvh, prec = man["dk"], man["nkvh"], man["precision"]
    block, layout = man["block"], man["layout"]
    base = man["kv_base"] + req_id * (1 << 28)
    step_e = 32 // prec
    nb = -(-ctx // block)
    s_idx = ctx - 1
    out = []
    for h in range(nkvh):
        if layout == "contiguous":
            o = h * ctx * dk + s_idx * dk
        else:
            phys, tok = blocks[s_idx // block], s_idx % block
            if layout == "head":
                o = h * nb * block * dk + phys * block * dk + tok * dk
            else:
                o = phys * nkvh * block * dk + h * block * dk + tok * dk
        for e in range(0, dk, step_e):
            out.append(align(base + (o + e) * prec))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir")
    ap.add_argument("--info", action="store_true")
    ap.add_argument("--step", type=int, default=None)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--request", type=int, default=None, help="one request's KV only")
    ap.add_argument("--kv-only", action="store_true")
    ap.add_argument("--csv", action="store_true",
                    help="write 'address,rw' CSV instead of LD/ST lines "
                         "(still un-timestamped; use --stamp for cycles)")
    ap.add_argument("--with-writes", action="store_true",
                    help="also emit the KV append writes for this step (ST / W). "
                         "Decode is ~99.5%% reads, so this adds few lines but gives "
                         "the model real read/write turnaround")
    ap.add_argument("--weights-only", action="store_true")
    ap.add_argument("--out", default=None, help="write LD 0x.. lines instead of a summary")
    ap.add_argument("--stamp", action="store_true",
                    help="attach arrival cycles using the measured timing model "
                         "(writes 'cycle,address,rw' instead of LD lines)")
    ap.add_argument("--boundaries", type=int, default=8,
                    help="operation boundaries per window to spread stalls across")
    a = ap.parse_args()

    man, w = load(a.dir)
    if a.info or a.step is None:
        sz = sum(os.path.getsize(os.path.join(a.dir, f)) for f in os.listdir(a.dir))
        print(f"  model {man['model']}  layout {man['layout']}  intra {man['intra']}  "
              f"{man['precision']} B  block {man['block']}")
        print(f"  {man['layers']} layers x {man['steps']:,} steps  ->  "
              f"{man['total_addresses']/1e9:,.1f}G addresses")
        print(f"  weights {man['weight_addrs']:,} per layer per step (invariant), "
              f"layer stride {man['layer_stride']:,} B")
        print(f"  stored {sz/2**20:.1f} MB  ({man['total_addresses']*22/2**40:.1f} TB flat)")
        print(f"  steps available: 0 .. {man['steps']-1}")
        return

    st = None
    for i, s in enumerate(steps(a.dir)):
        if i == a.step:
            st = s
            break
    if st is None:
        raise SystemExit(f"  step {a.step} not in trace (have {man['steps']:,})")

    out, wr = [], []
    if not a.kv_only:
        off = a.layer * man["layer_stride"]
        out += [x + off for x in w]
    if not a.weights_only:
        for rid, ctx, blocks in st["reqs"]:
            if a.request is not None and rid != a.request:
                continue
            out += kv_for(rid, ctx, blocks, man)
            if a.with_writes:
                wr += kv_writes_for(rid, ctx, blocks, man)
    # writes retire at the end of the step, after the reads that fed them
    ops = [(ad, "R") for ad in out] + [(ad, "W") for ad in wr]

    if a.out:
        with open(a.out, "w") as fh:
            if a.stamp:
                fh.write("cycle,address,rw\n")
                ts = stamp([ad for ad, _ in ops], a.boundaries)
                for (c, ad), (_, op) in zip(ts, ops):
                    fh.write(f"{c},0x{ad:x},{op}\n")
                span = ts[-1][0] - ts[0][0]
                print(f"  wrote {a.out}: {len(ops):,} accesses over {span:,} cycles "
                      f"({len(ops)/max(span,1):.2f} req/cyc), step {a.step} layer {a.layer}")
            elif a.csv:
                fh.write("address,rw\n")
                for ad, op in ops:
                    fh.write(f"0x{ad:x},{op}\n")
            else:
                for ad, op in ops:
                    fh.write(f"{'LD' if op == 'R' else 'ST'} 0x{ad:x}\n")
                print(f"  wrote {a.out}: {len(ops):,} accesses "
                      f"({len(out):,} R + {len(wr):,} W), step {a.step} layer {a.layer}")
    else:
        live = [(r[0], r[1]) for r in st["reqs"]]
        print(f"  step {st['step']}, layer {a.layer}: {len(out):,} reads"
              + (f" + {len(wr):,} writes" if wr else ""))
        print(f"  live requests (id, ctx): {live}")
        print(f"  first 4: {[hex(x) for x in out[:4]]}")


if __name__ == "__main__":
    main()
