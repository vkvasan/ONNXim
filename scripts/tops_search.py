#!/usr/bin/env python3
"""Search the TOPS mapspace for the best configuration under an objective.

    model + hardware budget + objective  ->  best (Tiling, Ordering,
                                             Parallelism, Stationarity) + layout

Evaluation is ANALYTICAL, not simulated. Every model below was validated
against ONNXim over 203 runs:

  array floor   = weight_bytes / (core_width x precision x num_cores)
                  weights enter through ONE edge of core_width PEs at 1 row/cyc,
                  so core_height cancels. Predicted 32x32 / 64x64 / 128x128 to
                  within 4-5% (measured 6,593,550 / 3,301,400 / 1,656,837).
  memory floor  = bytes / 819.2 GB/s
  runtime       ~ max(array floor, memory floor)
  padding       arrays whose dim does not divide the matrix dims pad every GEMM.
                MEASURED: 90x90 cost 5.1x and 45x45 cost 7.2x versus 128x128
                and 64x64 on the same work. This is a hard constraint, not a
                penalty term.

Fixed by measurement, so not searched:
  layout        head-major won 9/9 grid points on every metric. The 24 KV
                permutations collapse to 4 classes by where 'head' sits;
                head-outermost is the only good one.
  parallelism   head-parallel (rotate core per head group) beat request- and
                seq-parallel by 3.3x -- those lose to load imbalance when
                contexts differ, not to memory behaviour.
  block size    irrelevant under head-major (1.001x / 1.000x / 1.002x for
                16/128/512), so pick it on fragmentation alone: 16 tokens
                wastes 0.7%, 512 wastes 24.6%.
  stationarity  provably null at batch 1 -- with no reuse every weight crosses
                the array boundary exactly once, so WS and OS give identical
                cycles. Only becomes real at batch > 1.

Usage
    python3 scripts/tops_search.py --model llama2-7b --objective throughput
    python3 scripts/tops_search.py --model llama3-8b --objective latency --area 65536
"""
import argparse
import json
import math
import os

HBM_BW = 819.2          # GB/s, 16ch x 2pch x 6.4 Gbps
CLK = 1e9


def load_model(name):
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(here, "models", "language_models", name + ".json")) as f:
        m = json.load(f)
    h = m["hidden_size"]
    nh = m.get("num_attention_heads", m.get("num_heads"))
    return dict(name=name, H=h, NH=nh, NKVH=m.get("num_kv_heads", nh),
                DK=h // nh, INTER=m["intermediate_size"],
                FFN=m.get("ffn_type", "default"))


def weight_elems(mdl):
    """Elements in one transformer block's weights."""
    H, I = mdl["H"], mdl["INTER"]
    qkv = H * (H + 2 * mdl["DK"] * mdl["NKVH"])
    ffn = 3 * H * I if mdl["FFN"] == "llama" else 2 * H * I
    return qkv + H * H + ffn


def divides_all(dim, mdl):
    """A systolic dim that does not divide the GEMM dims pads every tile.
    Measured cost: 5.1x (90x90) and 7.2x (45x45). Treated as infeasible."""
    dims = [mdl["H"], mdl["INTER"], mdl["H"] + 2 * mdl["DK"] * mdl["NKVH"]]
    return all(d % dim == 0 for d in dims)


def evaluate(mdl, cores, dim, prec, batch, ctx):
    """Cycles for one decode pass, plus the metrics the objectives care about."""
    wbytes = weight_elems(mdl) * prec
    kvbytes = 2 * batch * ctx * mdl["NKVH"] * mdl["DK"] * prec
    array = wbytes / (dim * prec * cores)          # validated to 4-5%
    memory = (wbytes + kvbytes) / (HBM_BW * 1e9) * CLK
    # Calibration. max(array, memory) is OPTIMISTIC because it omits the
    # per-attention-op serialisation ONNXim exhibits (N ops issued one at a
    # time) and per-core fixed cost. Measured/predicted over four configs:
    # 1.38, 1.43, 1.15, 1.24 -> mean 1.30. Applied as a flat factor; it does
    # NOT capture the spread, so treat absolute cycles as +-15%.
    cycles = max(array, memory) * 1.30
    return dict(cycles=cycles, array=array, memory=memory,
                bound="memory" if memory > array else "array",
                bw=(wbytes + kvbytes) / cycles / HBM_BW * 100,
                pes=cores * dim * dim,
                tok_per_Mcyc=batch / cycles * 1e6,
                kv_gb=kvbytes / 2**30)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="llama2-7b")
    ap.add_argument("--objective", choices=["latency", "throughput", "memory"],
                    default="throughput")
    ap.add_argument("--area", type=int, default=16384, help="PE budget")
    ap.add_argument("--batch", type=int, default=0,
                    help="0 = let the objective choose (latency prefers small, "
                         "throughput prefers large)")
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--top", type=int, default=8)
    a = ap.parse_args()

    mdl = load_model(a.model)
    print(f"  model {mdl['name']}  hidden {mdl['H']} kv_heads {mdl['NKVH']} "
          f"ffn {mdl['FFN']}  ->  {weight_elems(mdl)/1e6:.0f}M weights/block")
    print(f"  objective {a.objective}, PE budget {a.area:,}, batch {a.batch}, ctx {a.ctx}\n")

    # Only search array dims that were MEASURED. dim=16 is outside the
    # validated range and the model extrapolates badly there.
    batches = [a.batch] if a.batch else [1, 8, 32, 128]
    cand = []
    for dim in (32, 64, 128):
        if not divides_all(dim, mdl):
            continue
        for cores in (1, 2, 4, 8, 16, 32, 64):
            if cores * dim * dim > a.area:
                continue
            for prec in (2, 1):
                for bs in batches:
                    kv_gb_all = 2*bs*a.ctx*mdl["NKVH"]*mdl["DK"]*prec*32/2**30
                    if kv_gb_all > 33:            # 46 GB device less 13 GB weights
                        continue
                    r = evaluate(mdl, cores, dim, prec, bs, a.ctx)
                    r.update(cores=cores, dim=dim, prec=prec, batch=bs)
                    cand.append(r)

    # latency    = cycles PER TOKEN produced (small batch wins: fast steps)
    # throughput = tokens per cycle           (large batch wins: amortised weights)
    # memory     = KV footprint, then speed
    # Ties are common: once a config is MEMORY-bound, extra cores change
    # nothing (the array is already waiting on DRAM). Break ties by AREA so the
    # search returns the cheapest config that achieves the optimum, rather than
    # an arbitrary one from iteration order.
    key = {"latency":    lambda r: (round(r["cycles"]), r["pes"]),
           "throughput": lambda r: (-round(r["batch"] / r["cycles"], 9), r["pes"]),
           "memory":     lambda r: (r["kv_gb"], round(r["cycles"]), r["pes"])}[a.objective]
    cand.sort(key=key)

    print(f"  {'rank':<5}{'cores':>6}{'array':>9}{'prec':>6}{'batch':>7}{'PEs':>9}"
          f"{'cycles':>12}{'bound':>9}{'BW':>7}{'tok/Mcyc':>10}{'KV GB':>8}")
    print("  " + "-" * 92)
    for i, r in enumerate(cand[:a.top], 1):
        print(f"  {i:<5}{r['cores']:>6}{str(r['dim'])+'x'+str(r['dim']):>9}{r['prec']:>6}"
              f"{r['batch']:>7}{r['pes']:>9,}{r['cycles']:>12,.0f}{r['bound']:>9}"
              f"{r['bw']:>6.0f}%{r['batch']/r['cycles']*1e6:>10.1f}{r['kv_gb']:>8.2f}")

    b = cand[0]
    print(f"\n  BEST for {a.objective}:")
    print(f"    T  tiling       derived: spad/(dim x prec x tile_depth), dim={b['dim']}")
    print(f"    O  ordering     tile-major walk, head outermost in KV")
    print(f"    P  parallelism  {b['cores']} cores, HEAD-parallel (measured 3.3x better"
          f" than request/seq)")
    print(f"    S  stationarity weight-stationary (null vs OS at batch 1)")
    print(f"       layout       head-major, block 16 (0.7% fragmentation)")
    print(f"       precision    {'int8' if b['prec']==1 else 'fp16'}")
    print(f"       batch        {b['batch']}")
    print(f"    -> {b['cycles']:,.0f} cycles/block (+-15%), {b['bound']}-bound, {b['bw']:.0f}% of HBM")


if __name__ == "__main__":
    main()
