#!/usr/bin/env python3
"""DRAM address traces for vLLM-style SPECULATIVE DECODING (open loop).

    workload + policy  ->  serving sim (admission, block tables, acceptance)
                       ->  per-step address streams: draft x k, verify, accept

Builds on vllm_trace.py (same weight walk, same hardware-derived tiling, same
head-major KV layout) and adds what speculative decoding changes:

    draft steps     k passes of a small draft model, 1 query token each
                    (2 right after a fully-accepted step: the bonus token has
                    not been seen by the draft). Own weights, own KV pool.
    verify          ONE target pass with k+1 query tokens per request.
                    Weights are read once per pass instead of once per token;
                    the KV cache is read once (--scorer mqa, vLLM V1) or k+1
                    times (--scorer expand, vLLM V0 batch expansion).
    lookahead       k+1 KV rows are written per verify; a rejected tail is
                    stale and is OVERWRITTEN by the next step. Those
                    write-after-write hits are counted (column `waw`).
    acceptance      a ~ min(Geometric(alpha), k) per request per step, or a
                    replayed file of accepted counts (--accept-file).

What is fixed, from measurement (FINDINGS.md): head-parallel attention,
head-major KV, tile-major weights with sequential intra-tile order. Tiling is
derived from the hardware exactly as ONNXim's Mapping.cc does; with k+1
tokens per request the GEMM token dimension is B*(k+1), and once that exceeds
one token tile the weights are re-read per token tile (GemmWS loops N
outermost) -- the tool reproduces that.

Not modelled, deliberately: service order / timing (run ONNXim with the
"specdec" scheduler for that -- see --emit-workload), prefill, preemption.

Usage
    python3 scripts/specdec_trace.py --count-only --requests 8 --k 4 --steps 5
    python3 scripts/specdec_trace.py --scorer expand --count-only
    python3 scripts/specdec_trace.py --k 4 --steps 2 --out sd.csv
    python3 scripts/specdec_trace.py --emit-workload sd_v4m512 --requests 4 --gen-cap 6
"""
import argparse
import json
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vllm_trace import load_model, derive_tiling, OBJECTIVES, REQ, align   # noqa: E402


def ceil_div(a, b):
    return -(-a // b)


# --------------------------------------------------------------------------
# address generators
# --------------------------------------------------------------------------
def weight_walk(base, precision, mdl, hw, tokens):
    """Tile-major walk over one transformer block, repeated once per token tile.

    ONNXim's GemmWS iterates N (tokens) outermost, so when B*(k+1) tokens no
    longer fit one token tile every weight tile is fetched again for the next
    token tile. derive_tiling() gives the inner token-tile height iI.
    """
    H, INTER = mdl["H"], mdl["INTER"]
    qkv_out = H + 2 * (mdl["DK"] * mdl["NKVH"])
    mats = [(H, qkv_out), (H, H)]
    mats += ([(H, INTER)] * 2 + [(INTER, H)]) if mdl["FFN"] == "llama" else [(H, INTER), (INTER, H)]
    out, off = [], base
    step = REQ // precision
    for rows, cols in mats:
        iI, iK, iJ = derive_tiling(tokens, cols, rows, hw)
        n_tok_tiles = max(1, ceil_div(tokens, iI))
        mat = []
        for tr in range(0, rows, iK):
            for tc in range(0, cols, iJ):
                n = min(iK, rows - tr) * min(iJ, cols - tc)
                for e in range(0, n, step):
                    mat.append(align(off + e * precision))
                off += n * precision
        out.extend(mat * n_tok_tiles)
    return out


def weight_bytes(mdl, precision):
    H, INTER = mdl["H"], mdl["INTER"]
    qkv_out = H + 2 * (mdl["DK"] * mdl["NKVH"])
    n = H * qkv_out + H * H
    n += (2 * H * INTER + INTER * H) if mdl["FFN"] == "llama" else (2 * H * INTER)
    return n * precision


def weight_tok_tiles(mdl, hw, tokens):
    """How many times the weight stream is walked for this many tokens."""
    H = mdl["H"]
    iI, _, _ = derive_tiling(tokens, 3 * H, H, hw)
    return max(1, ceil_div(tokens, iI))


class KVPool:
    """One global paged KV pool per model per layer: vLLM's cache tensor.

    head-major  [kv_head][block][token][dk]   each head owns a contiguous
                                              slice of the WHOLE pool
    block-major [block][kv_head][token][dk]   vLLM's stock layout
    """

    def __init__(self, base, mdl, precision, layout, block, pool, seed):
        self.base, self.prec, self.layout = base, precision, layout
        self.block, self.pool = block, pool
        self.NKVH, self.DK = mdl["NKVH"], mdl["DK"]
        self.free = list(range(pool))
        random.Random(seed).shuffle(self.free)             # pool starts fragmented
        self.row_reqs = self.DK * precision // REQ          # DRAM requests per (head,row)
        self.bytes = pool * block * self.NKVH * self.DK * precision

    def alloc(self, n):
        got = [self.free.pop(0) for _ in range(min(n, len(self.free)))]
        if len(got) < n:
            sys.exit("  KV pool exhausted -- raise --pool")
        return got

    def release(self, blocks):
        self.free.extend(blocks)

    def blocks_for(self, rows):
        return ceil_div(rows, self.block)

    def row_addrs(self, table, head, row, kbase):
        """DRAM requests of one (head,row) of the K (kbase=0) or V region."""
        phys, tok = table[row // self.block], row % self.block
        if self.layout == "head":
            o = head * self.pool * self.block * self.DK + phys * self.block * self.DK + tok * self.DK
        else:
            o = phys * self.NKVH * self.block * self.DK + head * self.block * self.DK + tok * self.DK
        b = self.base + kbase + o * self.prec
        return [align(b + d * self.prec) for d in range(0, self.DK, REQ // self.prec)]

    def vbase(self):
        return self.bytes                                   # V region follows K


# --------------------------------------------------------------------------
# emission
# --------------------------------------------------------------------------
class Emitter:
    """Writes rows (if a file is open) and keeps per-step counts either way.

    Row format:  step,phase,stream,request,address,rw,waw
        phase   draft<i> | verify | plain
        stream  weight | kv | dweight | dkv      (d = draft model)
        rw      R | W
        waw     1 when a KV write lands on a stale (rejected) row
    """

    def __init__(self, fh):
        self.fh = fh
        self.counts = {}

    def add(self, key, n):
        self.counts[key] = self.counts.get(key, 0) + n

    def weights(self, step, phase, stream, addrs):
        self.add((phase, stream, "R"), len(addrs))
        if self.fh:
            for a in addrs:
                self.fh.write(f"{step},{phase},{stream},-1,0x{a:x},R,0\n")

    def kv_read(self, step, phase, stream, req, pool, table, rows):
        """All heads, K then V per head, rows [0, rows): Attention's order."""
        n = rows * pool.NKVH * pool.row_reqs * 2
        self.add((phase, stream, "R"), n)
        if self.fh:
            for kb in (0, pool.vbase()):
                for h in range(pool.NKVH):
                    for r in range(rows):
                        for a in pool.row_addrs(table, h, r, kb):
                            self.fh.write(f"{step},{phase},{stream},{req},0x{a:x},R,0\n")

    def kv_write(self, step, phase, stream, req, pool, table, rows, stale, consume=True):
        """Write K and V of the given rows; count stale-row overwrites.

        Every layer writes its own copy of the row, so each layer's write to a
        stale row is a write-after-write; the row leaves the stale set only
        after the last layer (consume=True)."""
        per_row = pool.NKVH * pool.row_reqs * 2
        n_waw = 0
        for r in rows:
            w = 1 if r in stale else 0
            if w:
                n_waw += per_row
                if consume:
                    stale.discard(r)
            if self.fh:
                for kb in (0, pool.vbase()):
                    for h in range(pool.NKVH):
                        for a in pool.row_addrs(table, h, r, kb):
                            self.fh.write(f"{step},{phase},{stream},{req},0x{a:x},W,{w}\n")
        self.add((phase, stream, "W"), len(rows) * per_row)
        self.add((phase, stream, "WAW"), n_waw)


# --------------------------------------------------------------------------
# serving + speculative loop
# --------------------------------------------------------------------------
class SpecServer:
    def __init__(self, a, target, draft, hw, prec):
        self.a, self.t, self.d, self.hw, self.prec = a, target, draft, hw, prec
        cfg = OBJECTIVES[a.objective]
        self.max_batch, self.order = cfg["max_batch"], cfg["order"]
        self.k = a.k
        self.tpool = KVPool(1 << 33, target, prec, a.layout, a.block, a.pool, 1234)
        self.dpool = KVPool(1 << 36, draft, prec, a.layout, a.block, a.pool, 4321) if draft else None
        self.rng = random.Random(a.seed + 1)
        self.accept_trace, self.accept_pos = [], 0
        if a.accept_file:
            with open(a.accept_file) as f:
                self.accept_trace = [min(int(x), self.k) for x in f.read().split() if x.strip().isdigit()]
        self.queue, self.active, self.nxt, self.step = [], [], 0, 0
        self.workload = self.build_workload()
        self.tl = a.target_layers
        self.dl = a.draft_layers or (draft["LAYERS"] if draft else 0)
        self.wt_bytes = weight_bytes(target, prec)
        self.wd_bytes = weight_bytes(draft, prec) if draft else 0
        self.dweight_base = 1 << 30          # draft weights live in their own region
        self.stats = dict(verify=0, draft=0, plain=0, tokens=0, accepted=0, verify_reqs=0, plain_ref=0.0)

    def build_workload(self):
        a, rng, wl, t = self.a, random.Random(self.a.seed), [], 0.0
        for i in range(a.requests):
            if a.arrival == "poisson":
                t += rng.expovariate(1.0 / a.rate)
            wl.append(dict(id=i, arrive=t,
                           prompt=int(round(128 * (4096 / 128) ** rng.random())),
                           gen=max(1, int(rng.paretovariate(1.1) * 40))))
        for r in wl:
            if a.gen_cap:
                r["gen"] = min(r["gen"], a.gen_cap)
            if a.max_prompt:
                r["prompt"] = min(r["prompt"], a.max_prompt)
        return wl

    def sample_accepted(self):
        if self.accept_trace:
            v = self.accept_trace[self.accept_pos % len(self.accept_trace)]
            self.accept_pos += 1
            return v
        n = 0
        while n < self.k and self.rng.random() < self.a.alpha:
            n += 1
        return n

    def admit(self):
        a = self.a
        while self.nxt < len(self.workload) and self.workload[self.nxt]["arrive"] <= self.step:
            self.queue.append(self.workload[self.nxt]); self.nxt += 1
        keyf = {"fifo": lambda r: r["id"], "shortest_prompt": lambda r: r["prompt"],
                "longest_prompt": lambda r: -r["prompt"]}[self.order]
        self.queue.sort(key=keyf)
        while self.queue and len(self.active) < self.max_batch:
            if sum(x["T"] for x in self.active) + self.queue[0]["prompt"] > a.kv_budget and self.active:
                break
            r = self.queue.pop(0)
            # rows resident after prefill = prompt + 1: the prompt plus the row of
            # the first generated token (ONNXim's convention, LanguageScheduler.cc).
            # That first token also counts toward `gen`, hence done=1.
            T = r["prompt"] + 1
            req = dict(id=r["id"], T=T, D=T, gen=r["gen"], done=1, steps=0, acc=0,
                       tblocks=self.tpool.alloc(self.tpool.blocks_for(T)),
                       dblocks=self.dpool.alloc(self.dpool.blocks_for(T)) if self.dpool else [],
                       stale_t=set(), stale_d=set())
            self.active.append(req)

    def ensure_blocks(self, r):
        k = self.k
        need = self.tpool.blocks_for(r["T"] + k + 1)
        if len(r["tblocks"]) < need:
            r["tblocks"] += self.tpool.alloc(need - len(r["tblocks"]))
        if self.dpool:
            need = self.dpool.blocks_for(r["D"] + k + 2)
            if len(r["dblocks"]) < need:
                r["dblocks"] += self.dpool.alloc(need - len(r["dblocks"]))

    def run_step(self, em):
        """One speculative step for the resident batch. Returns False when idle."""
        a, k, step = self.a, self.k, self.step
        self.admit()
        if not self.active:
            return bool(self.queue or self.nxt < len(self.workload))
        batch = self.active[:self.max_batch]
        B = len(batch)
        for r in batch:
            self.ensure_blocks(r)
        spec = k > 0 and (a.disable_batch == 0 or B <= a.disable_batch)
        tokens_at_start = self.stats["tokens"]

        if not spec:
            # plain autoregressive step
            for _ in range(self.tl):
                em.weights(step, "plain", "weight", weight_walk(0, self.prec, self.t, self.hw, B))
            for r in batch:
                em.kv_write(step, "plain", "kv", r["id"], self.tpool, r["tblocks"], [r["T"]], r["stale_t"])  # 1 layer
                em.kv_read(step, "plain", "kv", r["id"], self.tpool, r["tblocks"], r["T"] + 1)
                r["T"] += 1; r["done"] += 1
            self.stats["plain"] += 1; self.stats["tokens"] += B
        else:
            # ---- draft: k passes ------------------------------------------------
            if self.dpool:
                for i in range(k):
                    ph = f"draft{i}"
                    ntok = sum(max(1, r["T"] + i + 1 - r["D"]) for r in batch)
                    for l in range(self.dl):
                        em.weights(step, ph, "dweight",
                                   weight_walk(self.dweight_base + l * self.wd_bytes, self.prec,
                                               self.d, self.hw, ntok))
                    for r in batch:
                        want = r["T"] + i + 1
                        new_rows = list(range(r["D"], max(want, r["D"] + 1)))
                        # every draft layer reads/writes the same rows; emit per layer
                        for l in range(self.dl):
                            em.kv_write(step, ph, "dkv", r["id"], self.dpool, r["dblocks"], new_rows,
                                        r["stale_d"], consume=(l == self.dl - 1))
                            em.kv_read(step, ph, "dkv", r["id"], self.dpool, r["dblocks"], new_rows[-1] + 1)
                        r["D"] = new_rows[-1] + 1
                    self.stats["draft"] += 1
            # ---- verify: one target pass, k+1 query tokens ------------------------
            for _ in range(self.tl):
                em.weights(step, "verify", "weight", weight_walk(0, self.prec, self.t, self.hw, B * (k + 1)))
            for r in batch:
                T = r["T"]
                for l in range(self.tl):
                    last = (l == self.tl - 1)
                    if a.scorer == "mqa":
                        em.kv_write(step, "verify", "kv", r["id"], self.tpool, r["tblocks"],
                                    list(range(T, T + k + 1)), r["stale_t"], consume=last)
                        em.kv_read(step, "verify", "kv", r["id"], self.tpool, r["tblocks"], T + k + 1)
                    else:   # batch expansion: k+1 single-token sequences over the same cache
                        for i in range(k + 1):
                            em.kv_write(step, "verify", "kv", r["id"], self.tpool, r["tblocks"],
                                        [T + i], r["stale_t"], consume=last)
                            em.kv_read(step, "verify", "kv", r["id"], self.tpool, r["tblocks"], T + i + 1)
                # ---- accept ---------------------------------------------------------
                acc = self.sample_accepted()
                newT = T + acc + 1
                r["stale_t"] |= set(range(newT, T + k + 1))       # rejected tail: stale rows
                if self.dpool and r["D"] > newT:
                    r["stale_d"] |= set(range(newT, r["D"]))
                    r["D"] = newT
                r["T"] = newT
                r["done"] += acc + 1; r["steps"] += 1; r["acc"] += acc
                self.stats["accepted"] += acc; self.stats["tokens"] += acc + 1
                self.stats["verify_reqs"] += 1
            self.stats["verify"] += 1
        tokens_before_retire = self.stats["tokens"] - tokens_at_start
        # plain-decode reference: generating the same tokens one per step at this
        # batch costs (weight walk + sum of KV reads) / B per token
        W = self.tl * len(weight_walk(0, self.prec, self.t, self.hw, B))
        KV = sum(self.tl * (r["T"] + 1) * self.tpool.NKVH * self.tpool.row_reqs * 2 for r in batch)
        self.stats["plain_ref"] += tokens_before_retire * (W + KV) / B
        # ---- retire ---------------------------------------------------------------
        for r in list(self.active):
            if r["done"] >= r["gen"]:
                self.tpool.release(r["tblocks"])
                if self.dpool:
                    self.dpool.release(r["dblocks"])
                self.active.remove(r)
        self.step += 1
        return True


# --------------------------------------------------------------------------
def fmt_counts(counts, prefix=""):
    lines = []
    phases = sorted({k[0] for k in counts}, key=lambda p: (p == "verify", p))
    for ph in phases:
        parts = []
        for st in ("weight", "kv", "dweight", "dkv"):
            r = counts.get((ph, st, "R"), 0); w = counts.get((ph, st, "W"), 0)
            if r or w:
                s = f"{st} R {r:,}"
                if w:
                    s += f" W {w:,}"
                waw = counts.get((ph, st, "WAW"), 0)
                if waw:
                    s += f" (waw {waw:,})"
                parts.append(s)
        lines.append(f"{prefix}{ph:8s} " + " | ".join(parts))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="llama2-7b")
    ap.add_argument("--draft", default="llama-68m",
                    help="draft model name, or 'none' to leave draft traffic out")
    ap.add_argument("--k", type=int, default=4, help="draft tokens per step (0 = plain decode)")
    ap.add_argument("--alpha", type=float, default=0.8, help="per-token acceptance probability")
    ap.add_argument("--accept-file", default=None,
                    help="replay accepted counts (whitespace-separated ints), one per (step, request)")
    ap.add_argument("--scorer", choices=["mqa", "expand"], default="mqa",
                    help="mqa: KV read once per verify (vLLM V1); expand: k+1 times (V0 batch expansion)")
    ap.add_argument("--disable-batch", type=int, default=0,
                    help="fall back to plain decode above this batch size (0 = never)")
    ap.add_argument("--array-dim", type=int, default=32)
    ap.add_argument("--spad", type=int, default=4096)
    ap.add_argument("--acc", type=int, default=2048)
    ap.add_argument("--tile-depth", type=int, default=3)
    ap.add_argument("--cores", type=int, default=16)
    ap.add_argument("--objective", choices=OBJECTIVES, default="throughput")
    ap.add_argument("--requests", type=int, default=8)
    ap.add_argument("--arrival", choices=["poisson", "burst"], default="burst")
    ap.add_argument("--rate", type=float, default=40.0)
    ap.add_argument("--layout", choices=["head", "block"], default="head")
    ap.add_argument("--block", type=int, default=16)
    ap.add_argument("--kv-budget", type=int, default=67584)
    ap.add_argument("--pool", type=int, default=20000)
    ap.add_argument("--target-layers", type=int, default=1,
                    help="target layers to emit per step (ONNXim runs one; x32 for the model)")
    ap.add_argument("--draft-layers", type=int, default=0, help="0 = all draft layers")
    ap.add_argument("--steps", type=int, default=5, help="speculative steps to emit")
    ap.add_argument("--gen-cap", type=int, default=0, help="clamp sampled generation length")
    ap.add_argument("--max-prompt", type=int, default=0, help="clamp sampled prompt (context) length")
    ap.add_argument("--count-only", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--json", default=None,
                    help="dump per-step counts as JSON (specdec_analyze.py --ref compares against it)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--emit-workload", metavar="NAME", default=None,
                    help="write traces/NAME.csv + example/NAME.json for ONNXim's specdec scheduler")
    a = ap.parse_args()

    target = load_model(a.model)
    draft = load_model(a.draft) if a.draft != "none" else None
    prec = OBJECTIVES[a.objective]["precision"]
    hw = dict(dim=a.array_dim, spad_kb=a.spad, acc_kb=a.acc, tile_depth=a.tile_depth,
              cores=a.cores, prec=prec)
    srv = SpecServer(a, target, draft, hw, prec)

    print(f"  target {target['name']}: hidden {target['H']}, kv_heads {target['NKVH']}, "
          f"weights/block {weight_bytes(target, prec)/1e6:.1f} MB")
    if draft:
        print(f"  draft  {draft['name']}: hidden {draft['H']}, kv_heads {draft['NKVH']}, "
              f"layers {srv.dl}, weights/layer {weight_bytes(draft, prec)/1e6:.1f} MB")
    print(f"  spec: k {a.k}, alpha {a.alpha}, scorer {a.scorer}, "
          f"max_batch {srv.max_batch}, layout {a.layout}, block {a.block}, precision {prec} B")
    B = min(a.requests, srv.max_batch)
    print(f"  weight walks per target pass: {weight_tok_tiles(target, hw, B)} (plain, {B} tok) "
          f"vs {weight_tok_tiles(target, hw, B*(a.k+1))} (verify, {B*(a.k+1)} tok)")

    if a.emit_workload:
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        tp = os.path.join(here, "traces", a.emit_workload + ".csv")
        mp = os.path.join(here, "example", a.emit_workload + ".json")
        with open(tp, "w") as fh:
            fh.write("time, prompt_length, target_length, cached_length\n")
            # ONNXim rows after prefill = cached + prompt_length + 1, so cached =
            # prompt-1 with a 1-token prefill reproduces this tool's prompt+1 rows.
            # ONNXim retires a request once rows >= cached + prompt_length + target,
            # i.e. after target-1 tokens beyond the prefill's; this tool generates
            # gen-1 beyond it (done starts at 1), so target = gen.
            for r in srv.workload:
                fh.write(f"0, 1, {r['gen']}, {r['prompt'] - 1}\n")
        sc = {"max_batch_size": srv.max_batch, "spec_k": a.k, "spec_alpha": a.alpha,
              "spec_seed": a.seed, "scorer": a.scorer, "spec_disable_batch": a.disable_batch}
        if draft:
            sc["draft_model"] = draft["name"]
        # both sides replay the same accepted counts, so per-step traffic is comparable
        acc_rel = os.path.join("traces", a.emit_workload + ".accept")
        if a.accept_file:
            with open(a.accept_file) as f:
                vals = [min(int(x), a.k) for x in f.read().split() if x.strip().isdigit()]
        else:
            vals = [srv.sample_accepted() for _ in range(4096)]
        with open(os.path.join(here, acc_rel), "w") as fh:
            fh.write(" ".join(str(v) for v in vals) + "\n")
        sc["accept_file"] = acc_rel          # relative to ONNXIM_HOME
        print(f"  wrote {os.path.join(here, acc_rel)} ({len(vals)} accepted counts)")
        print(f"  reproduce the open-loop counts with:  --accept-file {acc_rel} "
              f"--requests {a.requests} --k {a.k} --seed {a.seed}"
              + (f" --gen-cap {a.gen_cap}" if a.gen_cap else "")
              + (f" --max-prompt {a.max_prompt}" if a.max_prompt else ""))
        with open(mp, "w") as fh:
            json.dump({"models": [{"name": a.model + "-1L" if a.model == "llama2-7b" else a.model,
                                   "trace_file": a.emit_workload + ".csv",
                                   "scheduler": "specdec", "scheduler_config": sc}]}, fh, indent=2)
        print(f"  wrote {tp}\n        {mp}")
        print(f"""
  Run it (closed loop, timestamped DRAM trace):

    docker exec -e ONNXIM_DRAM_TRACE=/workspace/ONNXim/out/{a.emit_workload}.csv \\
      -e ONNXIM_KV_BLOCK={a.block} -e ONNXIM_KV_LAYOUT=head -e ONNXIM_KV_POOL=1 \\
      -e ONNXIM_WEIGHT_SWIZZLE=1 -e RAMULATOR_WEIGHT_LIMIT=800000000 \\
      <container> bash -c "cd /workspace/ONNXim/build && ./bin/Simulator \\
        --config ../configs/_multi_16x32.json --mode language \\
        --models_list ../example/{a.emit_workload}.json --trace_file {a.emit_workload}.csv"

  then  python3 scripts/specdec_analyze.py out/{a.emit_workload}.csv out/{a.emit_workload}.log""")
        return

    fh = open(a.out, "w") if (a.out and not a.count_only) else None
    if fh:
        fh.write("step,phase,stream,request,address,rw,waw\n")
    total = {}
    emitted = 0
    per_step = []
    while emitted < a.steps:
        em = Emitter(fh)
        if not srv.run_step(em):
            break
        if em.counts:
            batch_desc = ""
            if srv.active:
                ctxs = [r["T"] for r in srv.active]
                batch_desc = f"{len(srv.active)} resident, rows {min(ctxs)}-{max(ctxs)}"
            print(f"  step {srv.step-1}: {batch_desc}")
            print(fmt_counts(em.counts, "    "))
            for kk, v in em.counts.items():
                total[kk] = total.get(kk, 0) + v
            js = {}
            for (ph, stq, rw), v in em.counts.items():
                js.setdefault(ph, {}).setdefault(stq, {})[rw] = v
            per_step.append(js)
            emitted += 1
    if a.json:
        with open(a.json, "w") as jf:
            json.dump(per_step, jf, indent=1)
    if fh:
        fh.close()

    st = srv.stats
    reads = sum(v for k, v in total.items() if k[2] == "R")
    writes = sum(v for k, v in total.items() if k[2] == "W")
    waw = sum(v for k, v in total.items() if k[2] == "WAW")
    tw = sum(v for k, v in total.items() if k[1] == "weight")
    tk = sum(v for k, v in total.items() if k[1] == "kv" and k[2] != "WAW")
    dw = sum(v for k, v in total.items() if k[1] == "dweight")
    dk = sum(v for k, v in total.items() if k[1] == "dkv" and k[2] != "WAW")
    tot = reads + writes
    print(f"\n  {emitted} steps: {st['verify']} verify, {st['draft']} draft passes, {st['plain']} plain; "
          f"{st['tokens']} tokens generated, {st['accepted']} draft tokens accepted"
          + (f" ({st['accepted']/st['verify_reqs']:.2f} accepted and "
             f"{(st['accepted']+st['verify_reqs'])/st['verify_reqs']:.2f} tokens per request-verify)"
             if st['verify_reqs'] else ""))
    print(f"  DRAM requests: {tot:,} = {tot*REQ/1e9:.3f} GB   reads {reads:,}  writes {writes:,}"
          f"  (write-after-write on stale rows: {waw:,})")
    if tot:
        print(f"  share: target weights {tw/tot*100:.1f}%  target KV {tk/tot*100:.1f}%  "
              f"draft weights {dw/tot*100:.1f}%  draft KV {dk/tot*100:.1f}%")
    if st["tokens"]:
        print(f"  per generated token: {tot/st['tokens']:,.0f} requests = {tot*REQ/st['tokens']/1e6:.2f} MB")
        ref = st["plain_ref"]
        if ref:
            print(f"  plain-decode reference for the same tokens and batch: {ref:,.0f} requests "
                  f"-> speculative / plain = {tot/ref:.3f}  (target-only: {(tw+tk)/ref:.3f})")
    if a.out and fh is None:
        print("  --count-only: no file written")
    elif a.out:
        print(f"  -> {a.out}")


if __name__ == "__main__":
    main()
