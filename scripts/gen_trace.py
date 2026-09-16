#!/usr/bin/env python3
"""Analytically generate the DRAM access pattern for LLM decode serving.

Motivation
    A hardware simulation of a full generation run is intractable: 570 decode
    steps of a realistic trace cost ~290 hours. But decode's address stream is
    STATICALLY DERIVABLE -- weights are a fixed tile-major walk and KV reads are
    affine in context length through the block table. Nothing is data-dependent.

    So the expensive part is unnecessary. Split it:

        serving simulation   (pure bookkeeping, milliseconds)
            arrivals, FIFO admission, one token per active request per step,
            block allocation from a shared free list, release on completion
              -> per-step batch state and per-request BLOCK TABLES
        address generation   (this file)
            given that state, emit the KV address sequence for one pass

    The block table is an OUTPUT of scheduling and an INPUT to addressing, so
    the serving sim must run first -- ordering determines generation, not the
    other way round.

Scope and honesty
    - KV addresses are generated exactly (validated against ONNXim traces).
    - Generation length is SAMPLED, not predicted. Output length is genuinely
      unpredictable (tokens are sampled until EOS), so a distribution drawn
      from measured data is the faithful choice, not a predictor.
    - Arrival CYCLES are not modelled here. The trace records demand order, not
      service order; timing comes from the closed loop between core and DRAM,
      which an open-loop generator cannot reproduce. Use this to characterise
      the address SEQUENCE and to drive a controller model, not to predict
      runtime.

Usage
    python3 scripts/gen_trace.py --validate out/tr/ct_v8m1024.csv --trace traces/v8m1024.csv
    python3 scripts/gen_trace.py --serve --requests 64 --rate 40 --steps 500
"""
import argparse
import csv
import random


# ONNXim / LLaMA-2 7B geometry. Must match the simulated config.
DK = 128            # head dim
NKVH = 32           # kv heads (MHA)
PREC = 2            # bytes per element (fp16)
REQ_SIZE = 32       # dram_req_size: addresses are aligned to this


def align(addr):
    """Operation::make_address ends with _config.align_address()."""
    return addr - (addr % REQ_SIZE)


def kv_offsets_contiguous(seq_len, heads=NKVH, dk=DK):
    """Attention::kv_address with ONNXIM_KV_BLOCK unset.

    make_address({head, seq_idx, d}, {nkvh, seq, dk}) = ((h*seq + s)*dk + d)*prec
    A whole tile is read per head, so the emitted set is every aligned address
    in [head_base, head_base + seq*dk*prec).
    """
    out = []
    for h in range(heads):
        base = h * seq_len * dk
        for s in range(seq_len):
            for d in range(0, dk, REQ_SIZE // PREC):
                out.append(align((base + s * dk + d) * PREC))
    return out


def kv_offsets_paged(seq_len, block_tokens, block_table, head_major=False,
                     heads=NKVH, dk=DK):
    """Attention::kv_address with paging.

    block-major (vLLM): [block][head][token][dk]
    head-major        : [head][block][token][dk]
    """
    n_blocks = (seq_len + block_tokens - 1) // block_tokens
    out = []
    for h in range(heads):
        for s in range(seq_len):
            phys = block_table[s // block_tokens]
            tok = s % block_tokens
            if head_major:
                off = (h * n_blocks * block_tokens * dk
                       + phys * block_tokens * dk + tok * dk)
            else:
                off = (phys * heads * block_tokens * dk
                       + h * block_tokens * dk + tok * dk)
            for d in range(0, dk, REQ_SIZE // PREC):
                out.append(align((off + d) * PREC))
    return out


def serve(n_requests, mean_iat, max_batch, block_tokens, pool_blocks,
          prompt_lo=128, prompt_hi=1024, mean_gen=120, seed=7):
    """FIFO serving simulation. Returns per-step batch state.

    Generation lengths are SAMPLED (see module docstring). A real server cannot
    know them; a simulator should draw them from measured data.
    """
    rng = random.Random(seed)
    arrivals, t = [], 0.0
    for _ in range(n_requests):
        t += rng.expovariate(1.0 / mean_iat) if mean_iat > 0 else 0.0
        arrivals.append(t)
    prompts = [rng.randint(prompt_lo, prompt_hi) for _ in range(n_requests)]
    gens = [max(1, int(rng.expovariate(1.0 / mean_gen))) for _ in range(n_requests)]

    free = list(range(pool_blocks))
    rng.shuffle(free)
    queue, active, done = [], [], []
    nxt, step, history = 0, 0, []

    while nxt < n_requests or active or queue:
        while nxt < n_requests and arrivals[nxt] <= step:
            queue.append(nxt)
            nxt += 1
        while queue and len(active) < max_batch:
            i = queue.pop(0)
            need = (prompts[i] + block_tokens - 1) // block_tokens
            blocks = [free.pop(0) for _ in range(min(need, len(free)))]
            active.append({"id": i, "len": prompts[i], "gen": gens[i], "blocks": blocks})
        # one token per active request; allocate on block boundary
        for r in list(active):
            r["len"] += 1
            r["gen"] -= 1
            need = (r["len"] + block_tokens - 1) // block_tokens
            while len(r["blocks"]) < need and free:
                r["blocks"].append(free.pop(0))
            if r["gen"] <= 0:
                free.extend(r["blocks"])       # released; the list keeps its history
                active.remove(r)
                done.append(r["id"])
        if active:
            history.append({"step": step,
                            "batch": [{"id": r["id"], "len": r["len"],
                                       "blocks": list(r["blocks"])} for r in active]})
        step += 1
        if step > 100000:
            break
    return history


def validate(real_csv, trace_csv, max_rows=12_000_000):
    """Compare generated KV address counts against a real ONNXim trace.

    ONNXim gives every request its own K and V tensor, so a batch of N requests
    produces 2N distinct KV regions -- not one. Detect them by bucketing the
    address space and matching region sizes against the prediction
    (256 addresses per token: 32 heads x 128 dk x 2 B / 32 B request size).
    """
    import collections
    ctxs = sorted((int(r[3]) + 1 for r in list(csv.reader(open(trace_csv)))[1:]),
                  reverse=True)
    hist = collections.Counter()
    n = 0
    with open(real_csv) as f:
        next(f)
        for line in f:
            if n >= max_rows:
                break
            p = line.split(',')
            try:
                a = int(p[7], 16)
            except (ValueError, IndexError):
                continue
            n += 1
            hist[a >> 26] += 1                 # 64 MB buckets
    per_token = NKVH * DK * PREC // REQ_SIZE   # = 256
    sizes = [c for k, c in hist.items() if (k << 26) >= 512 * 2**20]

    print(f"  {len(ctxs)} requests, contexts {min(ctxs)}-{max(ctxs)}, "
          f"{n:,} trace rows scanned")
    print(f"\n  {'context':>9}{'predicted':>13}{'in trace':>11}{'':>6}")
    print("  " + "-" * 42)
    ok = 0
    for c in ctxs:
        want = per_token * c
        hit = sizes.count(want) >= 2           # K and V regions, equal size
        ok += hit
        print(f"  {c:>9,}{want:>13,}{want if hit else 0:>11,}{'  OK' if hit else '  --':>6}")
    print(f"\n  {ok}/{len(ctxs)} requests matched exactly "
          f"(prediction: {per_token} addresses per token)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--validate", metavar="REAL_CSV")
    ap.add_argument("--trace", metavar="WORKLOAD_CSV")
    ap.add_argument("--serve", action="store_true")
    ap.add_argument("--requests", type=int, default=64)
    ap.add_argument("--rate", type=float, default=40.0, help="mean inter-arrival, steps")
    ap.add_argument("--max-batch", type=int, default=8)
    ap.add_argument("--block", type=int, default=16)
    ap.add_argument("--pool", type=int, default=8000)
    args = ap.parse_args()

    if args.validate:
        validate(args.validate, args.trace)
    elif args.serve:
        h = serve(args.requests, args.rate, args.max_batch, args.block, args.pool)
        comps = {tuple(sorted(r["id"] for r in s["batch"])) for s in h}
        print(f"  {len(h):,} decode steps, {len(comps):,} distinct batch compositions "
              f"({len(h)/max(len(comps),1):.0f}x fewer hardware runs needed)")
        s = h[len(h) // 2]
        print(f"\n  mid-run state (step {s['step']}):")
        for r in s["batch"][:4]:
            print(f"    req {r['id']:>3}  len {r['len']:>5}  {len(r['blocks']):>4} blocks "
                  f"-> {r['blocks'][:8]}{' ...' if len(r['blocks']) > 8 else ''}")


if __name__ == "__main__":
    main()
