#!/usr/bin/env python3
"""Generate DRAM memory traces for a vLLM-style serving scenario.

    objective + workload  ->  policy  ->  serving simulation  ->  addresses

Why this exists
    Simulating a full generation run in hardware is intractable (~290 h for 570
    decode steps). But decode's address stream is statically derivable: weights
    are a fixed tile-major walk and KV reads are affine in context length via
    the block table. Only the block table needs the serving simulation, and that
    is pure bookkeeping.

    Validated against ONNXim traces: the KV address count per request is exactly
    256 x context (32 kv-heads x 128 dk x 2 B / 32 B request), 8/8 requests
    matching to the digit on out/tr/ct_v8m1024.csv.

The three-way split this tool preserves
    1. WHAT addresses are generated -- layout, block size, precision, tiling
    2. HOW they are ordered         -- batch size, admission order, parallelism
    3. HOW they are served          -- NOT modelled here, deliberately.
       The tool emits raw addresses so your controller model applies its own
       address mapping, scheduling and timing. That is the whole point: the
       coupling between (1),(2) and (3) is what you are trying to study, so (3)
       must stay a free variable.

Dominant vs traded knobs (measured over 203 ONNXim runs)
    DOMINANT  head-major KV layout, tile-major weights -- no objective prefers
              the alternative, so they are defaults. Override to study coupling.
    TRADED    batch size, admission order, precision -- the objective picks these.

Limitations, stated plainly
    - Emits DEMAND order, not service order. Arrival cycles come from the closed
      core<->DRAM loop; an open-loop generator cannot reproduce them. Use this to
      characterise patterns and drive a controller model, not to predict runtime.
    - Generation length is SAMPLED, not predicted. Output length is genuinely
      unknowable (tokens are sampled until EOS); drawing from measured data is
      the faithful choice.
    - No preemption. Real vLLM evicts under memory pressure; this only throttles
      admission.

Usage
    python3 scripts/vllm_trace.py --objective throughput --requests 64 --out t.csv
    python3 scripts/vllm_trace.py --objective latency --layout block --steps 20
"""
import argparse
import json
import math
import random

REQ = 32                      # dram_req_size; addresses align to this

def load_model(name):
    """Read geometry from models/language_models/<name>.json.

    ffn_type 'llama' = SwiGLU (gate, up, down = 3 matrices);
    anything else    = standard MLP (up, down = 2).
    """
    import os
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(here, "models", "language_models", name + ".json")) as f:
        m = json.load(f)
    h  = m["hidden_size"]
    nh = m.get("num_attention_heads", m.get("num_heads"))
    return dict(name=name, H=h, NH=nh, NKVH=m.get("num_kv_heads", nh),
                DK=h // nh, INTER=m["intermediate_size"],
                LAYERS=m.get("num_hidden_layers", 32),
                FFN=m.get("ffn_type", "default"))

# Objective -> the knobs it actually decides. Layout is NOT here: head-major
# won 9/9 grid points on every metric, so it is a default, not a trade.
OBJECTIVES = {
    "latency":    dict(max_batch=8,   order="shortest_prompt", precision=2),
    "throughput": dict(max_batch=128, order="fifo",            precision=2),
    "memory":     dict(max_batch=128, order="shortest_prompt", precision=1),
}


def align(a):
    return a - (a % REQ)


def derive_tiling(dim_I, dim_J, dim_K, hw):
    """Derive the tile shape the way ONNXim does (MappingTable::gemm_mapping,
    Mapping.cc:52-110). TILING IS NOT A FREE CHOICE -- it falls out of the
    hardware: array dim, scratchpad size, accumulator size, precision and
    tile_depth. Verified to reproduce ONNXim's own mapping exactly for the
    QKV GEMM: Outer N:1 C:6 M:48, Inner N:32 C:800 M:256."""
    dim, prec, td, nc = hw["dim"], hw["prec"], hw["tile_depth"], hw["cores"]
    max_spad_rows = (hw["spad_kb"] * 1024) // (dim * prec * td)
    max_acc_rows  = (hw["acc_kb"] * 1024) // (dim * 4 * td)
    db_mats_part = (max_spad_rows // 2) // dim
    db_max_ij = int(math.sqrt(max(max_acc_rows // dim, 1)))
    db_max_k  = db_mats_part // max(db_max_ij, 1)
    pad = lambda d: ((d // dim) + (1 if d % dim else 0)) * dim
    cd  = lambda a, b: -(-a // max(b, 1))
    tI = min(pad(dim_I)//dim, cd(dim_I, db_max_ij*dim))
    tJ = min(pad(dim_J)//dim, cd(dim_J, db_max_ij*dim))
    tK = min(pad(dim_K)//dim, cd(dim_K, db_max_k*dim))
    n = tI * tJ
    if n < nc:
        inc = cd(nc, n)
        if dim_J > dim_I and dim_J > nc: tJ *= inc
        elif dim_I > dim_J and dim_I > nc: tI *= inc
        n = tI * tJ
    if n % nc:
        inc = n % nc
        if dim_J > dim_I and dim_J > nc: tJ += inc
        elif dim_I > dim_J and dim_I > nc: tI += inc
    iI, iK, iJ = cd(pad(dim_I), tI), cd(pad(dim_K), tK), cd(pad(dim_J), tJ)
    iI -= iI & (dim-1); iK -= iK & (dim-1); iJ -= iJ & (dim-1)
    return max(iI, dim), max(iK, dim), max(iJ, dim)


def weight_addrs(base, precision, mdl, hw):
    """Tile-major walk over one block's weight matrices (ONNXIM_WEIGHT_SWIZZLE).

    Emitted once per decode pass and identical every pass -- the weights do not
    depend on context, batch or schedule.
    """
    H, INTER = mdl["H"], mdl["INTER"]
    qkv_out = H + 2 * (mdl["DK"] * mdl["NKVH"])          # Q full + K,V at kv-head count
    mats = [(H, qkv_out), (H, H)]
    mats += ([(H, INTER)] * 2 + [(INTER, H)]) if mdl["FFN"] == "llama" else [(H, INTER), (INTER, H)]
    out, off = [], base
    step = REQ // precision
    for rows, cols in mats:
        _, iK, iJ = derive_tiling(1, cols, rows, hw)   # tile shape from HARDWARE
        for tr in range(0, rows, iK):
            for tc in range(0, cols, iJ):
                n = min(iK, rows-tr) * min(iJ, cols-tc)
                for e in range(0, n, step):
                    out.append(align(off + e * precision))
                off += n * precision
    return out


def kv_addrs(base, ctx, precision, layout, block, table, mdl):
    """KV reads for one request. Mirrors Attention::kv_address exactly."""
    NKVH, DK = mdl["NKVH"], mdl["DK"]
    step = REQ // precision
    out = []
    if layout == "contiguous":
        for h in range(NKVH):
            b = h * ctx * DK
            for s in range(ctx):
                for d in range(0, DK, step):
                    out.append(align(base + (b + s * DK + d) * precision))
        return out
    nb = (ctx + block - 1) // block
    for h in range(NKVH):
        for s in range(ctx):
            phys, tok = table[s // block], s % block
            if layout == "head":
                o = h * nb * block * DK + phys * block * DK + tok * DK
            else:                                     # block-major (vLLM)
                o = phys * NKVH * block * DK + h * block * DK + tok * DK
            for d in range(0, DK, step):
                out.append(align(base + (o + d) * precision))
    return out


def serve(workload, max_batch, order, block, pool, kv_budget):
    """Admission + block allocation. Yields per-step resident batch state.

    The block table is an OUTPUT of scheduling and an INPUT to addressing --
    ordering determines generation, so this must run first.
    """
    free = list(range(pool))
    rng = random.Random(1234)
    rng.shuffle(free)                                  # pool starts fragmented
    queue, active, nxt, step = [], [], 0, 0
    keyf = {"fifo": lambda r: r["id"],
            "shortest_prompt": lambda r: r["prompt"],
            "longest_prompt": lambda r: -r["prompt"]}[order]
    while nxt < len(workload) or active or queue:
        while nxt < len(workload) and workload[nxt]["arrive"] <= step:
            queue.append(workload[nxt]); nxt += 1
        queue.sort(key=keyf)
        while queue and len(active) < max_batch:
            if sum(a["ctx"] for a in active) + queue[0]["prompt"] > kv_budget and active:
                break
            r = queue.pop(0)
            need = (r["prompt"] + block - 1) // block
            r = dict(id=r["id"], ctx=r["prompt"], gen=r["gen"],
                     blocks=[free.pop(0) for _ in range(min(need, len(free)))])
            active.append(r)
        if active:
            yield step, [dict(id=a["id"], ctx=a["ctx"], blocks=list(a["blocks"]))
                         for a in active]
        for a in list(active):
            a["ctx"] += 1; a["gen"] -= 1
            need = (a["ctx"] + block - 1) // block
            while len(a["blocks"]) < need and free:
                a["blocks"].append(free.pop(0))
            if a["gen"] <= 0:
                free.extend(a["blocks"])               # released; list keeps history
                active.remove(a)
        step += 1
        if step > 200000:
            break


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="llama2-7b",
                    help="name in models/language_models/ (llama2-7b, llama3-8b, opt-125m, opt-66b)")
    ap.add_argument("--array-dim", type=int, default=32, help="systolic array height")
    ap.add_argument("--spad", type=int, default=4096, help="scratchpad KB")
    ap.add_argument("--acc", type=int, default=2048, help="accumulator KB")
    ap.add_argument("--tile-depth", type=int, default=3)
    ap.add_argument("--cores", type=int, default=16,
                    help="interleave the emission across N cores, head-parallel "
                         "(measured optimal: request/seq-parallel are 3.3x slower)")
    ap.add_argument("--objective", choices=OBJECTIVES, default="throughput")
    ap.add_argument("--requests", type=int, default=32)
    ap.add_argument("--arrival", choices=["poisson", "burst"], default="poisson",
                    help="burst = all arrive at once")
    ap.add_argument("--rate", type=float, default=40.0, help="mean inter-arrival, steps")
    ap.add_argument("--layout", choices=["head", "block", "contiguous"], default="head",
                    help="head-major is DOMINANT; override to study coupling")
    ap.add_argument("--block", type=int, default=16)
    ap.add_argument("--kv-budget", type=int, default=67584, help="tokens (46 GB device)")
    ap.add_argument("--pool", type=int, default=20000)
    ap.add_argument("--steps", type=int, default=5, help="decode steps to emit")
    ap.add_argument("--count-only", action="store_true",
                    help="report the pattern without writing ~1.5 GB/step")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    mdl = load_model(a.model)
    hw = dict(dim=a.array_dim, spad_kb=a.spad, acc_kb=a.acc,
              tile_depth=a.tile_depth, cores=a.cores, prec=OBJECTIVES[a.objective]["precision"])
    cfg = dict(OBJECTIVES[a.objective])
    if a.layout != "head":
        cfg["layout"] = a.layout
    rng = random.Random(a.seed)

    # workload: given, not chosen. lengths sampled (see module docstring).
    wl, t = [], 0.0
    for i in range(a.requests):
        if a.arrival == "poisson":
            t += rng.expovariate(1.0 / a.rate)
        wl.append(dict(id=i, arrive=t,
                       prompt=int(round(128 * (4096 / 128) ** rng.random())),
                       gen=max(1, int(rng.paretovariate(1.1) * 40))))

    prec = cfg["precision"]
    wbase, kvbase = 0, 1 << 33
    print(f"  model {mdl['name']}: hidden {mdl['H']}, heads {mdl['NH']}, kv_heads {mdl['NKVH']}, "
          f"inter {mdl['INTER']}, ffn {mdl['FFN']}, layers {mdl['LAYERS']}")
    print(f"  objective {a.objective}: max_batch {cfg['max_batch']}, order {cfg['order']}, "
          f"precision {prec} B, layout {a.layout}, block {a.block}")
    ti = derive_tiling(1, 3*mdl["H"], mdl["H"], hw)
    print(f"  hardware {a.cores} cores x {a.array_dim}x{a.array_dim}, spad {a.spad} KB, "
          f"tile_depth {a.tile_depth}  -> DERIVED tile K={ti[1]} M={ti[2]}")

    fh = open(a.out, "w") if (a.out and not a.count_only) else None
    if fh:
        fh.write("step,stream,request,address,rw\n")
    emitted = wsum = ksum = 0
    for n, (step, batch) in enumerate(serve(wl, cfg["max_batch"], cfg["order"],
                                            a.block, a.pool, a.kv_budget)):
        if n >= a.steps:
            break
        w = weight_addrs(wbase, prec, mdl, hw)
        wsum += len(w)
        if fh:
            for ad in w:
                fh.write(f"{step},weight,-1,0x{ad:x},R\n")
        for r in batch:
            k = kv_addrs(kvbase + r["id"] * (1 << 28), r["ctx"], prec,
                         a.layout, a.block, r["blocks"], mdl)
            ksum += len(k)
            if fh:
                for ad in k:
                    fh.write(f"{step},kv,{r['id']},0x{ad:x},R\n")
        emitted += 1
        if n == 0:
            print(f"  step {step}: {len(batch)} resident, contexts "
                  f"{min(r['ctx'] for r in batch)}-{max(r['ctx'] for r in batch)}")
    if fh:
        fh.close()
    tot = wsum + ksum
    print(f"  emitted {emitted} steps: {tot:,} addresses "
          f"({wsum/max(tot,1)*100:.2f}% weights, {ksum/max(tot,1)*100:.2f}% KV)")
    print(f"  per step: {wsum//max(emitted,1):,} weight + {ksum//max(emitted,1):,} KV"
          f"   ~{tot*23/1e9:.2f} GB if written")
    if a.out:
        print(f"  -> {a.out}    (addresses only; apply your own mapping/timing)")


if __name__ == "__main__":
    main()
