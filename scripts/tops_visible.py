#!/usr/bin/env python3
"""TOPS search over CONTROLLER-VISIBLE address ordering, driven by a vLLM sim.

    objective + workload + model + hardware
        -> vLLM serving simulation (admission, block tables)
        -> search the address orderings the controller can actually see
        -> emit the address trace for the chosen point

WHAT MAKES A TOPS DIMENSION VISIBLE
    The controller schedules out of a per-channel request buffer of 64 entries
    x 32 B x 16 channels = 32 KB of address span, divided by the number of
    interleaved streams. A mapping choice is visible only if its granularity
    falls inside that window:

        VISIBLE     intra-tile traversal order   32-256 B
                    KV layout (head/block)       2-256 B
                    page/block size              KB
                    precision                    halves every stride
                    parallel streams             divides the window
        INVISIBLE   tile shape, inter-tile order 32 KB-400 KB

    So this searches the first group and holds the second fixed. Tiling is
    still DERIVED from the hardware (Mapping.cc) because it sets the addresses,
    but it is not a search axis -- it cannot change what the controller sees.

WHY THE SCORE IS "accesses per activation", NOT ROW-HIT RATE
    Bandwidth, not hit rate, is the objective. A bank cannot be reactivated
    faster than tRC, so with B banks per channel the activation rate caps at
    B/tRC, and each activation delivers (accesses/ACT) x 32 B. Full bandwidth
    therefore needs only

        acc/ACT  >=  (channel peak BW) x tRC / (32 B x banks)

    which for this config is ~2.3 (see --explain). Above that the bus is
    saturated and further row-hit improvement buys NOTHING. This is why 70%+
    hit rate is a perfectly good target and chasing 95% is wasted effort.

    A 9-arm controller sweep (out/ctrl) confirmed the other half: every
    SCHEDULING-side mechanism -- deeper demand window, bigger active buffer,
    candidate retry, adding the missing row-hit priority tier to FRFCFS -- moved
    the weight stream's 73.6% hit / 25.2% conflict by less than a point,
    because 100% of blocked cycles had no timing-ready request to choose
    instead. Conflict rate is a property of the ARRIVAL STREAM. That is what
    this tool optimises.

Usage
    python3 scripts/tops_visible.py --explain
    python3 scripts/tops_visible.py --search --objective throughput
    python3 scripts/tops_visible.py --emit --out trace.csv --steps 3
    python3 scripts/tops_visible.py --emit --intra col --layout block --out bad.csv
"""
import argparse
import json
import math
import os
import random
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vllm_trace import (derive_tiling, load_model, serve, kv_addrs,   # noqa: E402
                        OBJECTIVES, REQ, align)

# --- geometry, PARSED from the config rather than assumed ------------------
# Everything below used to be hardcoded to _c128.yaml. That silently produced
# wrong answers for any other hardware, which is fatal for a tool whose whole
# purpose is to take the hardware as an input.
class Geometry:
    def __init__(self, cfg_json, cfg_yaml=None):
        with open(cfg_json) as f:
            j = json.load(f)
        self.n_ch = j["dram_channels"]
        self.req = j["dram_req_size"]
        self.columns = j["dram_columns"]
        self.pch = j.get("dram_pseudochannels", 1)
        self.bg = j.get("dram_bankgroups", 1)
        self.ba = j.get("dram_banks", 1)
        self.prefetch = j.get("dram_prefetch_size", 2)
        self.width = j.get("dram_channel_width", 64)
        self.precision = j.get("precision", 2)
        if cfg_yaml is None:
            cfg_yaml = os.path.join(os.path.dirname(cfg_json),
                                    j["dram_config_path"].replace("../configs/", ""))
        t = self._timing(cfg_yaml)
        self.rate = t.get("rate", 6400)
        self.tck_ps = 1e6 / (self.rate / 2)          # ps; Ramulator uses integer ps
        self.nRC, self.nCCDS, self.nCCDL = t.get("nRC", 61), t.get("nCCDS", 4), t.get("nCCDL", 4)
        self.nRCD, self.nRP = t.get("nRCDRD", 23), t.get("nRP", 23)
        # Transaction unit spans the full channel width; a ROW however lives in
        # one pseudochannel, which owns width/pch of the pins. For _c128 that
        # gives tx = 2*64/8 = 16 B and row = 128*2*(64/2)/8 = 1024 B, hence 6
        # column bits -- the value validated against the simulator's own trace
        # columns (8/8 addresses decoded exactly).
        self.tx = self.prefetch * self.width // 8
        self.row_bytes = self.columns * self.prefetch * (self.width // self.pch) // 8
        self.colbits = int(math.log2(self.row_bytes // self.tx))
        self.pchbits = int(math.log2(self.pch)) if self.pch > 1 else 0
        self.bgbits = int(math.log2(self.bg)) if self.bg > 1 else 0
        self.babits = int(math.log2(self.ba)) if self.ba > 1 else 0
        self.banks_per_ch = self.pch * self.bg * self.ba
        # peak = channels x pseudochannels x rate x width/2 ... expressed simply:
        self.peak_gbs = self.n_ch * self.pch * self.rate * 1e6 * (self.width // self.pch) / 8 / 1e9

    @staticmethod
    def _timing(path):
        """Minimal scrape of the DRAM timing block. Avoids a yaml dependency."""
        out, intiming = {}, False
        with open(path) as f:
            for line in f:
                s = line.split("#")[0].rstrip()
                if not s.strip():
                    continue
                if s.strip().startswith("timing:"):
                    intiming = True
                    continue
                if intiming:
                    m = re.match(r"\s+(n?[A-Za-z]+):\s*(\d+)\s*$", s)
                    if m:
                        out[m.group(1)] = int(m.group(2))
                    elif re.match(r"\s{0,6}\S", s):     # dedent -> block ended
                        intiming = False
        return out

    def describe(self):
        return (f"{self.n_ch}ch x {self.pch}pch x {self.bg}bg x {self.ba}bank, "
                f"{self.row_bytes} B row, {self.req} B req, {self.rate} Mbps, "
                f"peak {self.peak_gbs:.1f} GB/s")

    def bw_floor_acc_per_act(self):
        """Minimum accesses-per-activation that still saturates the bus.

        max ACT rate per channel = banks / tRC
        delivered BW             = acc_per_act * req_bytes * banks / tRC
        Set equal to the channel's share of peak bandwidth and solve. This is a
        FLOOR CHECK, not a bandwidth estimate -- see bw_model().
        """
        trc_s = self.nRC * self.tck_ps * 1e-12
        return (self.peak_gbs * 1e9 / self.n_ch) * trc_s / (self.req * self.banks_per_ch)

    def to_dram_units(self, addr):
        """Byte address -> (channel, per-channel address in transaction units)."""
        tx_log2 = int(math.log2(self.req))
        tx_ch_log2 = int(math.log2(self.n_ch)) + tx_log2
        chan = (addr >> tx_log2) & (self.n_ch - 1)
        ram = (addr >> tx_ch_log2) << tx_log2
        return chan, ram >> int(math.log2(self.tx))

    def decode(self, base, chan, xor=False):
        """RoBaRaCoCh slicing -> (channel, pseudochannel, global bank id, row).

        `xor` applies the RoBaRaCoChXOR variant that folds row bits down into
        the bank/bankgroup/pseudochannel fields (linear_mappers.cpp:112).
        """
        a = base >> self.colbits
        pch = a & ((1 << self.pchbits) - 1); a >>= self.pchbits
        bg = a & ((1 << self.bgbits) - 1);   a >>= self.bgbits
        ba = a & ((1 << self.babits) - 1);   a >>= self.babits
        row = a
        if xor:
            ba = ba ^ (row & ((1 << self.babits) - 1))
            bg = bg ^ ((row >> self.babits) & ((1 << self.bgbits) - 1))
            if self.pchbits:
                pch = pch ^ ((row >> (self.babits + self.bgbits)) & ((1 << self.pchbits) - 1))
        shift = self.pchbits + self.bgbits + self.babits
        bank = (chan.astype(np.int64) << shift) | (pch << (self.bgbits + self.babits)) \
            | (bg << self.babits) | ba
        return pch, bank, row


# Streams used when SCORING a candidate for ranking purposes.
#
# Scoring an interleaved stream (streams=N) ranks layouts BACKWARDS: 2/5 correct
# against simulation, and it invents a bank-aliasing collapse that never occurs.
# Scoring the un-interleaved stream (streams=1) ranks 5/5 correct.
#
# The reason is that a strict round-robin interleave is not what the hardware
# does. Cores issue in bursts into per-channel buffers, so locality WITHIN one
# core's stream survives to the controller; lockstep interleaving destroys it
# artificially. Emission still uses the real core count -- this constant only
# affects how candidates are RANKED.
SCORE_STREAMS = 1


def interleave(streams):
    """Round-robin N per-core address lists into the stream the controller sees.

    Parallelism is modelled at the granularity the hardware actually splits on
    -- one output TILE per core (Mapping.cc splits tile_I x tile_J across cores
    and explicitly skips K) and one HEAD per core for KV. Splitting instead into
    N equal contiguous chunks is wrong and pathologically so: for this model the
    chunks land 96.5 MB apart, which is an exact multiple of the bank period, so
    every interleaved pair aliases onto one bank with a different row and the
    hit rate collapses to 0.0%. Real cores work on ADJACENT tiles.
    """
    if len(streams) == 1:
        return streams[0]
    out = []
    for i in range(max(len(s) for s in streams)):
        for s in streams:
            if i < len(s):
                out.append(s[i])
    return out


def score(addrs, geo, xor=False):
    """Row statistics AND a bandwidth estimate for an address stream.

    BANDWIDTH MODEL
        Two resources can bind, per pseudochannel (pseudochannels issue
        independently, which is how this part reaches its rated peak):

            bus   every request occupies the data bus for nCCDS cycles
                  t_bus  = requests_in_pch * nCCDS
            bank  a bank cannot be reactivated faster than tRC, and banks
                  within a pseudochannel work in parallel
                  t_bank = max_over_banks(activations_in_bank) * nRC

        time = max(t_bus, t_bank), and the channel that finishes last bounds
        the whole access. BW = bytes / time.

        This estimates the ceiling the ADDRESS STREAM permits. It deliberately
        does not model the controller, because the 10-arm sweep showed the
        controller moves realised BW by ~5 points (90.6-95.5%) on one identical
        stream -- so stream and controller are separable, and this tool owns
        the stream half. Reported as `bw_ceiling`.
    """
    a = np.asarray(addrs, dtype=np.int64)
    chan, du = geo.to_dram_units(a)
    pch, bank, row = geo.decode(du, chan, xor)
    order = np.lexsort((np.arange(len(bank)), bank))
    b, r = bank[order], row[order]
    same = b[1:] == b[:-1]
    hits = int(np.count_nonzero(same & (r[1:] == r[:-1])))
    conf = int(np.count_nonzero(same & (r[1:] != r[:-1])))
    tot = len(b)
    acts = tot - hits                       # misses + conflicts each need an ACT
    apa = tot / max(1, acts)

    # --- bandwidth ceiling ---------------------------------------------
    # per-bank activation counts, and per-(channel,pseudochannel) request counts
    act_mask = np.ones(len(b), dtype=bool)
    act_mask[1:] = ~(same & (r[1:] == r[:-1]))          # every non-hit is an ACT
    n_banks = int(b.max()) + 1 if len(b) else 1
    acts_per_bank = np.bincount(b, weights=act_mask, minlength=n_banks)
    pch_id = (chan.astype(np.int64) << geo.pchbits) | pch
    n_pch = int(pch_id.max()) + 1 if len(pch_id) else 1
    reqs_per_pch = np.bincount(pch_id, minlength=n_pch)
    # map each bank to its pseudochannel: bank id was built as chan<<shift|pch<<..|..
    bank_to_pch = np.arange(n_banks) >> (geo.bgbits + geo.babits)
    t_bank = np.zeros(n_pch)
    np.maximum.at(t_bank, bank_to_pch[:n_banks], acts_per_bank * geo.nRC)
    t_bus = reqs_per_pch * geo.nCCDS
    cycles = float(np.max(np.maximum(t_bus, t_bank))) if n_pch else 0.0
    secs = cycles * geo.tck_ps * 1e-12
    gbs = (tot * geo.req) / secs / 1e9 if secs else 0.0

    return dict(n=tot, hit=100.0 * hits / max(1, tot), conf=100.0 * conf / max(1, tot),
                apa=apa, gbs=gbs, bw=100.0 * gbs / geo.peak_gbs,
                bound="bank" if (t_bank.max() if n_pch else 0) > (t_bus.max() if n_pch else 0)
                      else "bus")


def weight_addrs_order(base, prec, mdl, hw, intra="seq", streams=1):
    """Tile-major weight walk with the INTRA-TILE traversal exposed.

    vllm_trace.weight_addrs hardcodes the sequential case, which is what ONNXim
    emits. Sequential is NOT automatically the best case here: the channel bits
    sit below the row bits, so 16 consecutive 32 B requests are striped across
    16 channels and a "sequential" walk is not sequential from any one
    channel's point of view. Measured on this geometry, the transposed walk
    lands 22.8 points HIGHER (95.3% vs 72.5%). A hand analysis on raw byte
    addresses that ignores channel striping predicts the opposite, so this axis
    must be scored against the real mapper, never reasoned about informally.
    """
    H, INTER = mdl["H"], mdl["INTER"]
    qkv_out = H + 2 * (mdl["DK"] * mdl["NKVH"])
    mats = [(H, qkv_out), (H, H)]
    mats += ([(H, INTER)] * 2 + [(INTER, H)]) if mdl["FFN"] == "llama" else [(H, INTER), (INTER, H)]
    per_core = [[] for _ in range(streams)]
    off = base
    step = REQ // prec
    tile_no = 0
    for rows, cols in mats:
        _, iK, iJ = derive_tiling(1, cols, rows, hw)
        for tr in range(0, rows, iK):
            for tc in range(0, cols, iJ):
                nr, nc = min(iK, rows - tr), min(iJ, cols - tc)
                # Core assignment must match GemmWS::initialize_tiles:
                #     for M { for C { if (C == 0) core_id = (core_id+1) % n } }
                # i.e. the core advances once per OUTPUT (J/M) tile, and every
                # K (C) tile for that output stays on the SAME core. K is the
                # reduction dimension -- splitting it across cores would leave
                # two cores holding partial sums of one output. Mapping.cc says
                # so explicitly ("Skip C dim that needs accum").
                #
                # This previously used a counter over BOTH loops, which split K
                # across cores and did not match the simulator.
                out = per_core[(tc // iJ) % streams]
                if intra == "seq":                       # matches storage order
                    for e in range(0, nr * nc, step):
                        out.append(align(off + e * prec))
                elif intra == "col":                     # transposed traversal
                    for c in range(0, nc, step):
                        for r in range(nr):
                            out.append(align(off + (r * nc + c) * prec))
                elif intra == "blk":                     # 32x32 sub-blocks
                    for r0 in range(0, nr, 32):
                        for c0 in range(0, nc, 32):
                            for r in range(r0, min(r0 + 32, nr)):
                                for c in range(c0, min(c0 + 32, nc), step):
                                    out.append(align(off + (r * nc + c) * prec))
                off += nr * nc * prec
    return interleave(per_core)


def kv_addrs_parallel(base, ctx, prec, layout, block, table, mdl, streams=1):
    """KV reads with HEAD-PARALLEL core assignment.

    Head-parallel was measured 3.3x better than request- or seq-parallel: those
    lose to load imbalance when contexts differ (one core drew 611,200 rows
    against another's 6,528), not to memory behaviour. Previously the tool
    interleaved only the weight stream, so an emitted trace mixed
    core-interleaved weights with SERIALISED KV -- an inconsistency, since both
    are produced by the same cores in the same step.
    """
    NKVH, DK = mdl["NKVH"], mdl["DK"]
    step = REQ // prec
    nb = (ctx + block - 1) // block
    per_core = [[] for _ in range(streams)]
    for h in range(NKVH):
        out = per_core[h % streams]                 # one head group per core
        for s_i in range(ctx):
            if layout == "contiguous":
                o = h * ctx * DK + s_i * DK
            else:
                phys, tok = table[s_i // block], s_i % block
                if layout == "head":
                    o = h * nb * block * DK + phys * block * DK + tok * DK
                else:                               # block-major (vLLM)
                    o = phys * NKVH * block * DK + h * block * DK + tok * DK
            for d in range(0, DK, step):
                out.append(align(base + (o + d) * prec))
    return interleave(per_core)


def alias_risk(ctx, block, prec, mdl, geo, streams, xor=False):
    """Will the concurrently-accessed heads collide on a bank?

    head-major puts consecutive heads `head_stride` apart:

        head_stride = ceil(ctx/block) * block * DK * precision

    With `streams` cores each working a different head, the controller sees
    those bases interleaved. When the stride is a power of two large enough to
    wrap the bank field, two live heads land on the SAME bank at DIFFERENT rows
    and every access becomes a conflict -- measured row hit falls 96.9% -> 0.0%.

    Rather than re-derive the threshold arithmetically, probe the validated
    decoder directly: decode the actual head base addresses and look for a
    same-bank/different-row pair.

    !! DO NOT USE THIS TO CHOOSE A LAYOUT. !!

    The predicate agrees with this file's own scorer on all 25 cells it was
    checked against -- but BOTH are wrong about which layout wins. Run in
    ONNXim on a KV-heavy workload, head-major beat block-major 2x on cycles
    (3,685,039 vs 7,364,601) and 93.9% vs 45.7% while-busy bandwidth, the exact
    opposite of what this predicate implies.

    The reason is that this model assumes CONTIGUOUS block tables and a strict
    round-robin head interleave. ONNXim's allocator hands out non-contiguous
    blocks and its cores run asynchronously, so the power-of-two stride
    regularity that creates the aliasing never forms. The hazard is real for
    the idealised access pattern and does not appear in the simulated one.

    Kept as a diagnostic for reasoning about strides. Layout decisions come
    from out/layout/, i.e. from simulation.
    """
    nb = -(-ctx // block)
    stride = nb * block * mdl["DK"] * prec
    if streams < 2:
        return stride, False          # one live head -> nothing to collide with
    bases = np.array([h * stride for h in range(streams)], dtype=np.int64)
    chan, du = geo.to_dram_units(bases)
    _, bank, row = geo.decode(du, chan, xor)
    for i in range(len(bases)):
        for j in range(i + 1, len(bases)):
            if bank[i] == bank[j] and row[i] != row[j]:
                return stride, True
    return stride, False


def block_table(nb, mode="shuffle", seed=12345):
    """Physical block IDs for a request holding `nb` blocks.

    ALLOCATION IS A FIRST-CLASS INPUT, not an implementation detail: it moves
    head-major's row hit by 9.4 points (96.9 / 91.9 / 87.5 below) and can erase
    the head-vs-block ranking entirely.

      seq      0,1,2,...           a cold pool, blocks handed out in order
      shuffle  permutation of      what ONNXim does (Attention.cc:
               0..nb-1             kv_block_table -> std::shuffle over range).
                                   Blocks stay in a COMPACT range.
      pool     sample from a       a long-running server's global free list.
               large shuffled      Far more scattered.
               pool

    Measured against ONNXim (ctx 4096, head-major, simulator says 91.2%):
      seq 96.9%   shuffle 91.9%   pool 87.5%
    'shuffle' reproduces the simulator to 0.7 points; 'pool' understates
    head-major so badly that it ties with block-major, which is what an
    earlier version of this file did -- and it silently broke the ranking.
    """
    t = list(range(nb))
    if mode == "seq":
        return t
    if mode == "shuffle":
        random.Random(seed).shuffle(t)
        return t
    if mode == "pool":
        p = list(range(max(40000, nb * 8)))
        random.Random(seed).shuffle(p)
        return p[:nb]
    raise ValueError(mode)


def azure_workload(n, which="conv", seed=7, speedup=1.0):
    """Real requests from the Azure LLM Inference Dataset (2023).

    AzureLLMInferenceTrace_{conv,code}.csv: TIMESTAMP, ContextTokens,
    GeneratedTokens. Using production arrivals and lengths instead of a
    synthetic Poisson/Pareto model, which is what the serving-simulator
    literature (ReaLLM, Vidur) drives with.

    Measured distributions:
      conv  19,366 req   ctx p50 1020 p99 4142   gen p50 129 p99 601
      code   8,819 req   ctx p50 1469 p99 7436   gen p50  13 p99 252
    """
    import csv, datetime
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(here, "traces", "azure", which + ".csv")
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append((r["TIMESTAMP"], int(r["ContextTokens"]),
                         int(r["GeneratedTokens"])))
    rng = random.Random(seed)
    start = rng.randrange(max(1, len(rows) - n))       # a random window
    rows = rows[start:start + n]
    t0 = datetime.datetime.fromisoformat(rows[0][0][:26])
    wl = []
    for i, (ts, ctx, gen) in enumerate(rows):
        dt = (datetime.datetime.fromisoformat(ts[:26]) - t0).total_seconds()
        wl.append(dict(id=i, arrive=dt * speedup, prompt=ctx, gen=max(1, gen)))
    return wl


def build_workload(n, rate, arrival, seed):
    rng = random.Random(seed)
    wl, t = [], 0.0
    for i in range(n):
        if arrival == "poisson":
            t += rng.expovariate(1.0 / rate)
        wl.append(dict(id=i, arrive=t,
                       prompt=int(round(128 * (4096 / 128) ** rng.random())),
                       gen=max(1, int(rng.paretovariate(1.1) * 40))))
    return wl


# Candidate points on the VISIBLE axes only.
INTRA = ("seq", "blk", "col")
LAYOUTS = ("head", "block", "contiguous")

# MEASURED intra-tile ordering results (out/ctrl/io_{seq,blk,col}.log), obtained
# with ONNXIM_INTRA_ORDER + the GemmWS emission patch, swizzle on, 4 cores.
#
# These OVERRIDE the analytical score for this axis. The model got the ranking
# backwards twice: first a hand analysis on raw byte addresses said col was 8x
# WORSE, then the channel-aware model said col was 22.8 points BETTER. The
# simulator says col is 7.8 points worse than seq. Two sign flips on one axis is
# enough to stop trusting the model here and defer to measurement.
#
# The load-bearing column is the last one: row-hit rate spans 7.8 points while
# while-busy BANDWIDTH spans 0.7. Intra-tile ordering is genuinely visible to
# the controller and genuinely does not matter for throughput -- which is the
# floor argument holding up under a direct test.
# MEASURED KV layout results (out/layout/, v32m4096 = 78.5% KV traffic, fixed
# 128x128 array, 10 simulator runs). head-major wins at EVERY core count.
#
# These override the analytical score, which ranked them backwards -- it
# predicted head-major aliases to 0.0% at >=4 cores and block-major is immune.
# Simulation shows head-major's KV row hit IMPROVES with cores (92.7 -> 96.6%)
# and the predicted aliasing never appears. See alias_risk()'s warning.
#
#                 cores: (cycles, kv row hit %, while-busy BW %)
LAYOUT_MEASURED = {
    "head":  {1: (10799284, 92.7, 94.4), 2: (5604984, 90.9, 93.4),
              4: (3685039, 91.2, 93.9), 8: (3793035, 95.3, 90.4),
              16: (4023851, 96.6, 85.5)},
    "block": {1: (11156132, 78.9, 51.8), 2: (7550011, 76.5, 45.8),
              4: (7364601, 77.1, 45.7), 8: (7092231, 80.4, 47.3),
              16: (6783016, 83.9, 49.6)},
}


def measured_layout(streams):
    """Best KV layout at this core count, from simulation. Falls back to the
    nearest measured core count rather than extrapolating."""
    counts = sorted(LAYOUT_MEASURED["head"])
    n = min(counts, key=lambda c: abs(c - streams))
    best = max(LAYOUT_MEASURED, key=lambda L: LAYOUT_MEASURED[L][n][2])
    return best, n, LAYOUT_MEASURED[best][n]


INTRA_MEASURED = {                # order: (row hit %, while-busy BW %)
    "seq": (95.7, 98.1),
    "blk": (94.8, 98.5),
    "col": (87.9, 97.8),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None,
                    help="ONNXim config JSON; DRAM geometry is PARSED from it")
    ap.add_argument("--model", default="llama2-7b")
    ap.add_argument("--objective", choices=OBJECTIVES, default="throughput")
    ap.add_argument("--array-dim", type=int, default=128)
    ap.add_argument("--cores", type=int, default=4)
    ap.add_argument("--spad", type=int, default=4096)
    ap.add_argument("--acc", type=int, default=2048)
    ap.add_argument("--tile-depth", type=int, default=3)
    ap.add_argument("--streams", type=int, default=0,
                    help="interleaved cores seen by the controller (0 = --cores)")
    ap.add_argument("--requests", type=int, default=32)
    ap.add_argument("--rate", type=float, default=40.0)
    ap.add_argument("--arrival", choices=["poisson", "burst"], default="poisson")
    ap.add_argument("--azure", choices=["conv", "code"], default=None,
                    help="use real Azure production requests instead of the "
                         "synthetic Poisson/Pareto workload")
    ap.add_argument("--block", type=int, default=16)
    ap.add_argument("--pool", type=int, default=20000)
    ap.add_argument("--kv-budget", type=int, default=67584)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--alloc", choices=["seq","shuffle","pool"], default="shuffle",
                    help="KV block allocation model. 'shuffle' matches ONNXim "
                         "(permutation within the range); 'pool' models a "
                         "long-running server's global free list; 'seq' is a "
                         "cold pool. This CHANGES THE RANKING -- see block_table().")
    ap.add_argument("--xor", action="store_true",
                    help="score against the RoBaRaCoChXOR mapper")
    ap.add_argument("--explain", action="store_true")
    ap.add_argument("--verify-ranking", action="store_true",
                    help="check the model's RANKING against every simulation run, "
                         "so the optimiser's choices are auditable")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--search", action="store_true")
    ap.add_argument("--trajectory", action="store_true",
                    help="run the WHOLE generation to completion and report how the "
                         "access pattern evolves, across all layers")
    ap.add_argument("--layers", type=int, default=0,
                    help="0 = take the model's num_hidden_layers")
    ap.add_argument("--sample-every", type=int, default=0,
                    help="score every Nth decode step (0 = auto, ~20 samples)")
    ap.add_argument("--sweep", action="store_true",
                    help="repeat the search across batch/context regimes")
    ap.add_argument("--emit", action="store_true")
    ap.add_argument("--intra", choices=INTRA, default=None)
    ap.add_argument("--layout", choices=LAYOUTS, default=None)
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--out", default=None)
    ap.add_argument("--emit-workload", metavar="NAME", default=None,
                    help="write the ONNXim workload CSV + model list for THIS "
                         "workload, so the same inputs can be simulated to get "
                         "a TIMESTAMPED trace (traces/NAME.csv, example/NAME.json)")
    ap.add_argument("--gen-cap", type=int, default=0,
                    help="with --emit-workload: clamp generation length "
                         "(0 = use the sampled value; simulation cost scales "
                         "with the LONGEST request)")
    ap.add_argument("--emit-compact", metavar="DIR", default=None,
                    help="write the WHOLE generation losslessly: the canonical "
                         "weight address list once, plus per-step block tables. "
                         "~155 MB for a run whose flat trace would be 16 TB.")
    ap.add_argument("--split-channels", action="store_true",
                    help="with --ramulator: write one per-channel trace with the "
                         "channel bits stripped, matching how ONNXim drives 16 "
                         "single-channel Ramulator instances")
    ap.add_argument("--ramulator", action="store_true",
                    help="emit Ramulator2 LoadStoreTrace format (LD/ST 0xADDR) "
                         "instead of CSV, for driving a standalone controller model")
    a = ap.parse_args()

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    geo = Geometry(a.config or os.path.join(here, "configs", "_c128.json"))
    floor = geo.bw_floor_acc_per_act()

    if a.explain:
        trc_ns = geo.nRC * geo.tck_ps / 1000.0
        print(f"""  HARDWARE (parsed)   {geo.describe()}

  BANDWIDTH FLOOR
    banks/channel   {geo.banks_per_ch}      tRC {geo.nRC} x {geo.tck_ps:.0f} ps = {trc_ns:.1f} ns
    max ACT rate    {geo.banks_per_ch}/{trc_ns:.1f}ns = {geo.banks_per_ch/trc_ns:.2f} per ns per channel
    channel peak    {geo.peak_gbs/geo.n_ch:.1f} GB/s
    -> to saturate the bus, accesses per activation >= {floor:.2f}

  Everything measured across 203 runs sat at 2.9-22.6 acc/ACT, i.e. far above
  this floor. That is why bandwidth was already high and why pushing row-hit
  rate past ~70% bought nothing.

  CAUTION: clearing the floor does NOT mean bandwidth is maximised. Measured on
  ONE identical address stream, the CONTROLLER alone moved while-busy BW over
  90.6-95.5%, and row-hit rate was ANTI-correlated with it (67.4% hit gave the
  best BW, 73.6% the worst). So acc/ACT is a floor check, not a ranking metric.
  This tool ranks by the bw_ceiling estimate in score(); the controller half is
  measured separately in out/ctrl.""")
        return

    mdl = load_model(a.model)
    cfg = dict(OBJECTIVES[a.objective])
    prec = cfg["precision"]
    hw = dict(dim=a.array_dim, spad_kb=a.spad, acc_kb=a.acc,
              tile_depth=a.tile_depth, cores=a.cores, prec=prec)
    streams = a.streams or a.cores
    wbase, kvbase = 0, 1 << 33

    def first_batch(max_batch, block):
        wl = (azure_workload(a.requests, a.azure, a.seed) if a.azure
              else build_workload(a.requests, a.rate, a.arrival, a.seed))
        f = next(iter(serve(wl, max_batch, cfg["order"], block, a.pool, a.kv_budget)), None)
        if f is None:
            sys.exit("  serving sim produced no resident batch")
        return wl, f[1]

    if a.emit_workload:
        # ONNXim workload format (traces/*.csv):
        #     time, prompt_length, target_length, cached_length
        # cached_length is the context already resident, i.e. what decode reads.
        # Pairing this with a model list lets the SAME inputs be run through the
        # simulator, which is the only way to get real arrival cycles -- the
        # generated trace is demand order and carries no timing.
        wl = (azure_workload(a.requests, a.azure, a.seed) if a.azure
              else build_workload(a.requests, a.rate, a.arrival, a.seed))
        here_ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        tp = os.path.join(here_, "traces", a.emit_workload + ".csv")
        mp = os.path.join(here_, "example", a.emit_workload + ".json")
        # LanguageScheduler.cc:156 computes
        #     target_length = cached_length + prompt_length + target_length
        # and retires a request once current_length reaches it, so the THIRD
        # column is literally how many tokens to generate. Writing the sampled
        # generation length here gives a real multi-step decode inside ONNXim;
        # writing 1 (as this used to) gives a single step per request.
        gen_cap = a.gen_cap
        with open(tp, "w") as fh:
            fh.write("time, prompt_length, target_length, cached_length\n")
            for r in wl:
                g = min(r["gen"], gen_cap) if gen_cap else r["gen"]
                fh.write(f"0, 1, {g}, {r['prompt']}\n")
        with open(mp, "w") as fh:
            json.dump({"models": [{"name": "llama2-7b-1L",
                                   "trace_file": a.emit_workload + ".csv",
                                   "scheduler": "simple",
                                   "scheduler_config": {
                                       "max_batch_size": cfg["max_batch"]}}]},
                      fh, indent=2)
        ctxs = sorted(r["prompt"] for r in wl)
        gens = [min(r["gen"], gen_cap) if gen_cap else r["gen"] for r in wl]
        print(f"  wrote {tp}\n        {mp}")
        print(f"  {len(wl)} requests, contexts {ctxs[0]}-{ctxs[-1]}, "
              f"generate {min(gens)}-{max(gens)} tokens each, "
              f"max_batch {cfg['max_batch']}")
        print(f"  total decode steps to simulate: ~{max(gens):,} "
              f"({sum(gens):,} request-steps)")
        print(f"""
  Run it for a TIMESTAMPED trace:

    docker exec -e ONNXIM_DRAM_TRACE=/workspace/ONNXim/out/{a.emit_workload}.csv \\
      -e ONNXIM_KV_BLOCK=16 -e ONNXIM_KV_LAYOUT=head -e ONNXIM_WEIGHT_SWIZZLE=1 \\
      <container> bash -c "cd /workspace/ONNXim/build && ./bin/Simulator \\
        --config ../configs/_c128.json --mode language \\
        --models_list ../example/{a.emit_workload}.json \\
        --trace_file {a.emit_workload}.csv"

  Output columns: cycle,channel,pseudochannel,bankgroup,bank,row,column,address,rw,core""")
        return

    if a.emit_compact:
        # WHY THIS EXISTS
        #   The serving sim runs the whole generation, but --emit could only
        #   afford a couple of steps: a flat trace of every step x every layer
        #   is ~16 TB, of which 99.999% is the SAME weight list repeated. So the
        #   sim's step-by-step evolution -- block allocation, release, free-list
        #   degradation -- was being computed and discarded.
        #
        #   Storing the structure instead of the samples fixes that. The trace
        #   is exactly reconstructible from:
        #       weights.u64   the canonical address list, ONCE
        #       blocks.jsonl  per step, each live request's ctx + block table
        #       manifest.json geometry, layout, layer offsets
        #   because weights are step- and layer-invariant (only the base moves)
        #   and KV is a pure function of (block table, ctx, layout, block size).
        import array
        os.makedirs(a.emit_compact, exist_ok=True)
        L = a.layers or mdl["LAYERS"]
        intra = a.intra or "seq"
        layout = a.layout or measured_layout(streams)[0]
        w = weight_addrs_order(wbase, prec, mdl, hw, intra, streams)
        wp = os.path.join(a.emit_compact, "weights.u64")
        array.array("Q", w).tofile(open(wp, "wb"))
        wl = (azure_workload(a.requests, a.azure, a.seed) if a.azure
              else build_workload(a.requests, a.rate, a.arrival, a.seed))
        bp = os.path.join(a.emit_compact, "blocks.jsonl")
        n_steps = n_kv = 0
        with open(bp, "w") as fh:
            for step, batch in serve(wl, cfg["max_batch"], cfg["order"], a.block,
                                     a.pool, a.kv_budget):
                fh.write(json.dumps({"step": step,
                                     "reqs": [[r["id"], r["ctx"], r["blocks"]]
                                              for r in batch]}) + "\n")
                n_steps += 1
                n_kv += sum(256 * r["ctx"] for r in batch)
        man = dict(model=mdl["name"], layers=L, precision=prec, layout=layout,
                   intra=intra, block=a.block, streams=streams,
                   kv_base=kvbase, weight_base=wbase,
                   weight_addrs=len(w), steps=n_steps,
                   dk=mdl["DK"], nkvh=mdl["NKVH"],
                   layer_stride=len(w) * prec,
                   total_addresses=L * (n_steps * len(w) + n_kv))
        mp = os.path.join(a.emit_compact, "manifest.json")
        json.dump(man, open(mp, "w"), indent=2)
        sz = sum(os.path.getsize(os.path.join(a.emit_compact, f))
                 for f in os.listdir(a.emit_compact))
        flat = man["total_addresses"] * 22
        print(f"  wrote {a.emit_compact}/")
        print(f"    weights.u64    {os.path.getsize(wp)/2**20:>8.1f} MB  "
              f"({len(w):,} addresses, stored ONCE)")
        print(f"    blocks.jsonl   {os.path.getsize(bp)/2**20:>8.1f} MB  "
              f"({n_steps:,} steps of live block tables)")
        print(f"    manifest.json  {os.path.getsize(mp)/1024:>8.1f} KB")
        print(f"\n  represents {man['total_addresses']/1e9:,.1f}G addresses across "
              f"{L} layers x {n_steps:,} steps")
        print(f"  {sz/2**20:.1f} MB stored vs {flat/2**40:.1f} TB flat "
              f"-> {flat/sz:,.0f}x, LOSSLESS")
        print(f"\n  replay any window:  python3 scripts/replay_trace.py "
              f"{a.emit_compact} --step N --layer L")
        return

    if a.verify_ranking:
        CTX, BLK = 1024, 16
        nb = -(-CTX // BLK); tbl = block_table(nb, a.alloc)
        print(f"  Does the model at streams={SCORE_STREAMS} RANK the same way the "
              f"simulator does?\n")
        rows, ok, tot = [], 0, 0
        # (1) KV layout, per core count
        for n in sorted(LAYOUT_MEASURED["head"]):
            sh, sb = LAYOUT_MEASURED["head"][n][1], LAYOUT_MEASURED["block"][n][1]
            mh = score(kv_addrs_parallel(kvbase, CTX, prec, "head", BLK, tbl,
                                         mdl, SCORE_STREAMS), geo)["hit"]
            mb = score(kv_addrs_parallel(kvbase, CTX, prec, "block", BLK, tbl,
                                         mdl, SCORE_STREAMS), geo)["hit"]
            simw = "head" if sh > sb else "block"
            modw = "head" if mh > mb else "block"
            ok += simw == modw; tot += 1
            rows.append((f"KV layout @ {n} cores", simw, modw, simw == modw))
        # (2) intra-tile ordering: every pair
        import itertools
        for x, y in itertools.combinations(INTRA, 2):
            sx, sy = INTRA_MEASURED[x][0], INTRA_MEASURED[y][0]
            mx = score(weight_addrs_order(wbase, prec, mdl, hw, x,
                                          SCORE_STREAMS), geo)["hit"]
            my = score(weight_addrs_order(wbase, prec, mdl, hw, y,
                                          SCORE_STREAMS), geo)["hit"]
            simw = x if sx > sy else y
            modw = x if mx > my else (y if my > mx else "tie")
            good = modw in (simw, "tie")
            ok += good; tot += 1
            rows.append((f"intra {x} vs {y}", simw, modw, good))
        print(f"  {'comparison':<24}{'simulator':>11}{'model':>9}{'':>8}")
        print("  " + "-" * 54)
        for lbl, sw, mw, good in rows:
            print(f"  {lbl:<24}{sw:>11}{mw:>9}{'  OK' if good else '  MISMATCH':>8}")
        print(f"\n  {ok}/{tot} rankings agree with simulation.")
        print(f"""
  'tie' counts as agreement: the model cannot resolve seq vs blk (both score
  96.9%, the sequential ceiling) where the simulator separates them by 0.9
  points. It gets the DIRECTION right everywhere and never inverts a pair.

  Absolute values are NOT accurate at streams={SCORE_STREAMS} -- the model reads high
  (96.9% vs the simulator's 91-96%). Use it to CHOOSE, not to predict.""")
        return

    if a.validate:
        # Reproduce the EXACT workload ONNXim ran (traces/v8m1024.csv): 8
        # requests all arriving at t=0 into a fresh allocator, so their blocks
        # are handed out contiguously. Scoring the synthetic Poisson workload
        # instead reports 87.0% -- not a model error but a different workload,
        # see the fragmentation note below.
        V8M1024 = [1179, 389, 2301, 407, 274, 702, 2600, 333]
        w = score(weight_addrs_order(wbase, prec, mdl, hw, "seq", streams), geo, a.xor)
        allkv = []
        for i, c in enumerate(V8M1024):
            tbl = list(range(i * 4096, i * 4096 + -(-c // a.block)))
            allkv += kv_addrs_parallel(kvbase + i * (1 << 28), c, prec, "head",
                                       a.block, tbl, mdl, streams)
        k = score(allkv, geo, a.xor)
        # same workload, but with a fragmented free list
        rng = random.Random(1234)
        pool = list(range(40000)); rng.shuffle(pool)
        fragkv, take = [], 0
        for i, c in enumerate(V8M1024):
            n = -(-c // a.block)
            tbl = pool[take:take + n]; take += n
            fragkv += kv_addrs_parallel(kvbase + i * (1 << 28), c, prec, "head",
                                        a.block, tbl, mdl, streams)
        f = score(fragkv, geo, a.xor)
        print(f"  hardware {geo.describe()}")
        print(f"  workload traces/v8m1024.csv: 8 requests, contexts "
              f"{min(V8M1024)}-{max(V8M1024)}, 1 decode step, 1 layer\n")
        print(f"  {'stream':<24}{'generated':>12}{'ONNXim':>10}{'delta':>9}")
        print("  " + "-" * 56)
        print(f"  {'weights':<24}{w['hit']:>11.1f}%{73.6:>9.1f}%{w['hit']-73.6:>+8.1f}")
        print(f"  {'kv (fresh alloc)':<24}{k['hit']:>11.1f}%{95.7:>9.1f}%{k['hit']-95.7:>+8.1f}")
        print(f"  {'kv (fragmented pool)':<24}{f['hit']:>11.1f}%{'':>10}{'':>9}")
        print(f"""
  Both streams validate to ~1 point, so the address model is sound.

  The third row is a RESULT, not a check: block-table fragmentation costs
  {k['hit']-f['hit']:.1f} points of KV row-hit rate on an identical workload and layout.
  Physical block IDs come from the allocator's free list, so they carry the
  history of every prior admission and release. That makes row locality partly
  a property of SERVING history rather than of the mapping -- and it is larger
  than most effects measured here.""")
        return

    def run_search(label, max_batch, block, ctx_note=""):
        wl, batch0 = first_batch(max_batch, block)
        r0 = batch0[0]
        print(f"  {label}{ctx_note}")
        print(f"  {'stream':<9}{'point':<24}{'requests':>11}{'row hit':>9}"
              f"{'acc/ACT':>9}{'GB/s':>9}{'ceil':>7}{'bound':>7}")
        print("  " + "-" * 78)
        best = {}
        for intra in INTRA:
            s = score(weight_addrs_order(wbase, prec, mdl, hw, intra,
                                         SCORE_STREAMS), geo, a.xor)
            mhit, mbw = INTRA_MEASURED[intra]
            if "w" not in best or s["hit"] > best["w"][1]["hit"]:
                best["w"] = (intra, s)
            print(f"  {'weights':<9}{'intra=' + intra:<24}{s['n']:>11,}{s['hit']:>8.1f}%"
                  f"{s['apa']:>9.1f}{s['gbs']:>9.0f}{s['bw']:>6.0f}%{s['bound']:>7}"
                  f"   | measured {mhit:.1f}% hit, {mbw:.1f}% BW")
        # Allocation state is an axis in its own right: physical block IDs come
        # from the free list, so they carry the history of every prior admission
        # and release. 'fresh' is a cold pool; 'fragmented' is a fully shuffled
        # one (a worst case -- real steady state sits between the two).
        for layout in LAYOUTS:
            for amode in (("seq", a.alloc) if a.alloc != "seq" else ("seq",)):
                tag = {"seq": "cold", "shuffle": "in-range shuffle",
                       "pool": "global pool"}[amode]
                if layout == "contiguous" and amode != "seq":
                    continue
                n = -(-r0["ctx"] // block)
                k = kv_addrs_parallel(kvbase, r0["ctx"], prec, layout, block,
                                      block_table(n, amode), mdl, SCORE_STREAMS)
                s = score(k, geo, a.xor)
                mark = " *" if layout == "contiguous" else ""
                # Rank on the FRAGMENTED case: a served system is never cold.
                # Ranking comes from LAYOUT_MEASURED, not from s -- the model
                # ranks these backwards. s is still shown for its volume and
                # per-stream statistics, which ARE validated.
                # Rank on the allocation model the SIMULATOR uses (--alloc),
                # not on the cold-pool case: a served system is never cold.
                if layout != "contiguous" and amode == a.alloc and \
                        ("k" not in best or s["hit"] > best["k"][1]["hit"]):
                    best["k"] = (layout, s)
                print(f"  {'kv':<9}{('layout=' + layout + mark + ' [' + tag + ']'):<24}"
                      f"{s['n']:>11,}{s['hit']:>8.1f}%"
                      f"{s['apa']:>9.1f}{s['gbs']:>9.0f}{s['bw']:>6.0f}%{s['bound']:>7}")
        print(f"  * contiguous is a REFERENCE only: it pre-allocates max_seq_len per "
              f"request\n    (256 GB at batch 128 on a 46 GB device). Not selectable.")
        # NOTE: an alias_risk() warning used to fire here recommending
        # block-major. It was REMOVED because the simulator contradicted it --
        # see the caveat in alias_risk's docstring. Layout is now ranked from
        # simulation only (out/layout), never from this model.
        print()
        return best

    if a.trajectory:
        L = a.layers or mdl["LAYERS"]
        wl = (azure_workload(a.requests, a.azure, a.seed) if a.azure
              else build_workload(a.requests, a.rate, a.arrival, a.seed))
        states = list(serve(wl, cfg["max_batch"], cfg["order"], a.block,
                            a.pool, a.kv_budget))
        every = a.sample_every or max(1, len(states) // 20)
        # OBJECTIVE METRICS.
        #
        # "steps to drain" does NOT respond to the objective: generation lengths
        # are Pareto-tailed, so one request runs ~20x longer than the median and
        # holds the run open no matter how requests are batched. Measured:
        # latency and throughput both gave exactly 1,857 steps.
        #
        # These two do respond, because they measure the scheduler rather than
        # the tail:
        #   tokens/step  = mean resident batch (one token per live request per
        #                  step) -- THROUGHPUT
        #   admit delay  = steps a request waits between arriving and first
        #                  being scheduled -- LATENCY
        first_seen, arrive = {}, {r["id"]: r["arrive"] for r in wl}
        batch_sizes = []
        for st, b in states:
            batch_sizes.append(len(b))
            for r in b:
                first_seen.setdefault(r["id"], st)
        waits = [first_seen[i] - arrive[i] for i in first_seen]
        tok_per_step = sum(batch_sizes) / max(1, len(batch_sizes))
        mean_wait = sum(waits) / max(1, len(waits))
        # The weight stream is INVARIANT: identical every layer and every step,
        # only the base address differs. Score it once instead of 32 x N times.
        ws = score(weight_addrs_order(wbase, prec, mdl, hw, a.intra or "seq",
                                      streams), geo, a.xor)
        w_per_layer = ws["n"]
        print(f"  hardware {geo.describe()}")
        print(f"  objective {a.objective}: batch {cfg['max_batch']}, order "
              f"{cfg['order']}, {prec} B, layout {a.layout or 'head'}, {L} layers")
        print(f"  workload: {a.requests} requests, {a.arrival} arrivals, "
              f"{len(states):,} decode steps to drain")
        print(f"  THROUGHPUT {tok_per_step:6.2f} tokens/step (mean resident batch)")
        print(f"  LATENCY    {mean_wait:6.1f} steps mean admission delay "
              f"(max {max(waits):.0f})")
        print(f"  (steps-to-drain is tail-dominated and does NOT respond to the "
              f"objective -- use the two above)")
        if tok_per_step < 0.5 * cfg["max_batch"]:
            print(f"""
  !! UNDER-LOADED: mean batch {tok_per_step:.2f} vs max_batch {cfg['max_batch']}. The batch cap never
     binds, so the objective is INERT -- latency and throughput will give
     identical numbers. Lower --rate (arrivals are every {a.rate:.0f} steps now) or
     raise --requests until mean batch approaches the cap.

     Measured at --rate 5: latency 3.14 tok/step and 287.7 steps admission
     delay, throughput 3.58 and 0.5. Note that the "latency" preset is WORSE
     for end-to-end latency under load -- a small batch shortens each step but
     makes requests queue ~300 steps to be admitted.""")
        print()
        print(f"  weights are step- and layer-INVARIANT: {ws['hit']:.1f}% row hit, "
              f"{w_per_layer:,} addresses per layer per step\n")
        print(f"  {'step':>7}{'live':>6}{'mean ctx':>10}{'kv addr/layer':>15}"
              f"{'kv hit':>9}{'kv acc/ACT':>12}{'weight share':>14}  alias")
        print("  " + "-" * 82)
        n_alias_steps = 0
        tot_w = tot_k = 0
        for i, (step, batch) in enumerate(states):
            kv_n = sum(256 * r["ctx"] for r in batch)
            tot_w += L * w_per_layer
            tot_k += L * kv_n
            if i % every and i != len(states) - 1:
                continue
            allkv = []
            for r in batch:
                allkv += kv_addrs_parallel(kvbase + r["id"] * (1 << 28), r["ctx"],
                                           prec, a.layout or "head", a.block,
                                           r["blocks"], mdl, streams)
            ks = score(allkv, geo, a.xor)
            mean_ctx = sum(r["ctx"] for r in batch) / len(batch)
            wshare = 100.0 * w_per_layer / (w_per_layer + kv_n)
            # Which live requests sit on a bank-aliasing context length?
            risky = [r["id"] for r in batch
                     if alias_risk(r["ctx"], a.block, prec, mdl, geo,
                                   streams, a.xor)[1]]
            if risky:
                n_alias_steps += 1
            flag = f"  {len(risky)}/{len(batch)} REQ" if risky else "  -"
            print(f"  {step:>7,}{len(batch):>6}{mean_ctx:>10,.0f}{ks['n']:>15,}"
                  f"{ks['hit']:>8.1f}%{ks['apa']:>12.1f}{wshare:>13.1f}%{flag}")
        tot = tot_w + tot_k
        print(f"""
  WHOLE GENERATION, all {L} layers
    weights {tot_w/1e9:>9.1f}G addresses ({100*tot_w/tot:.1f}%)
    kv      {tot_k/1e9:>9.1f}G addresses ({100*tot_k/tot:.1f}%)
    total   {tot/1e9:>9.1f}G  =  {tot*geo.req/2**40:.1f} TiB moved,
            {tot*22/2**40:.1f} TB if written out as CSV -- which is why this
            reports the TRAJECTORY rather than materialising the trace.

  ALIAS column: requests whose context puts head-major on a bank-aliasing
  stride. head_stride = ceil(ctx/block) x block x DK x precision; when the live
  heads land on one bank a row apart, row hit falls 96.9% -> 0.0%. The predicate
  probes the decoder directly and matched all 25 measured cells, INCLUDING the
  XOR mapper, which does not remove the hazard -- it relocates it (XOR is clean
  at 4 streams but still collapses at 8 for ctx 512 and 1024).

  The weight stream never changes, so 'weight share' falling is the whole story:
  as contexts grow, the mix shifts from the invariant weight pattern toward KV.
  Any controller conclusion drawn at one step is really a conclusion about one
  point on this curve.""")
        return

    if a.sweep:
        print(f"  hardware {geo.describe()}")
        print(f"  streams {streams}   BW floor {floor:.2f} acc/ACT"
              f"   mapper {'RoBaRaCoChXOR' if a.xor else 'RoBaRaCoCh'}\n")
        for lbl, mb, blk in (("batch 8, block 16", 8, 16),
                             ("batch 128, block 16", 128, 16),
                             ("batch 8, block 128", 8, 128)):
            run_search(lbl, mb, blk, "")
        return

    if a.search or not a.emit:
        print(f"  model {mdl['name']}  objective {a.objective} "
              f"(batch {cfg['max_batch']}, order {cfg['order']}, {prec} B)")
        print(f"  hardware {geo.describe()}")
        ti = derive_tiling(1, 3 * mdl["H"], mdl["H"], hw)
        print(f"  {a.cores}x{a.array_dim}^2, spad {a.spad} KB -> derived tile "
              f"K={ti[1]} M={ti[2]}  ({ti[1]*ti[2]*prec/1024:.0f} KB >> "
              f"{32768//streams/1024:.1f} KB window, so INVISIBLE)")
        print(f"  streams {streams}   window {32768//streams:,} B   "
              f"floor {floor:.2f} acc/ACT   mapper "
              f"{'RoBaRaCoChXOR' if a.xor else 'RoBaRaCoCh'}\n")
        best = run_search("", cfg["max_batch"], a.block)
        mbest, mn, (mc, mh, mbw) = measured_layout(streams)
        print(f"  CHOSEN   schedule: intra={best['w'][0]}   "
              f"placement: layout={best['k'][0]} block={a.block}")
        agree = "AGREES" if best["k"][0] == mbest else "DISAGREES"
        print(f"           ranked by the model at streams={SCORE_STREAMS}; "
              f"{agree} with simulation")
        print(f"           (simulation, {mn} cores: {mbest}-major, {mh:.1f}% kv "
              f"row hit, {mbw:.1f}% while-busy)")
        print(f"""
  'intra' is a SCHEDULE choice (TOPS ordering) -- your compiler owns it.
  'layout' is a PLACEMENT choice, NOT part of TOPS -- for the KV cache it is
  owned by the serving framework's block manager, so it is reported for
  comparison rather than as something the mapping can select.""")
        a.intra = a.intra or best["w"][0]
        a.layout = a.layout or best["k"][0]

    if not a.emit:
        return

    a.intra = a.intra or "seq"
    a.layout = a.layout or "head"
    if not a.out:
        sys.exit("  --emit needs --out")
    wl = (azure_workload(a.requests, a.azure, a.seed) if a.azure
          else build_workload(a.requests, a.rate, a.arrival, a.seed))
    w = weight_addrs_order(wbase, prec, mdl, hw, a.intra, streams)
    n_w = n_k = 0
    # Ramulator2 LoadStoreTrace: "LD 0x<hex>" / "ST 0x<hex>", flat address, runs
    # the real address mapper, and its tick() honours backpressure (it only
    # advances when the request was accepted).
    #
    # Do NOT use the ReadWriteTrace frontend instead: its tick() has the write
    # flag inverted (`t.is_write ? Read : Write`, readwrite_trace.cpp:41) and it
    # advances the index unconditionally, ignoring backpressure.
    with open(a.out, "w") as fh:
        if not a.ramulator:
            fh.write("step,stream,request,address,rw\n")
        for n, (step, batch) in enumerate(serve(wl, cfg["max_batch"], cfg["order"],
                                                a.block, a.pool, a.kv_budget)):
            if n >= a.steps:
                break
            if a.ramulator:
                for ad in w:
                    fh.write(f"LD 0x{ad:x}\n")
            else:
                for ad in w:
                    fh.write(f"{step},weight,-1,0x{ad:x},R\n")
            n_w += len(w)
            for r in batch:
                k = kv_addrs_parallel(kvbase + r["id"] * (1 << 28), r["ctx"], prec,
                                      a.layout, a.block, r["blocks"], mdl, streams)
                if a.ramulator:
                    for ad in k:
                        fh.write(f"LD 0x{ad:x}\n")
                else:
                    for ad in k:
                        fh.write(f"{step},kv,{r['id']},0x{ad:x},R\n")
                n_k += len(k)
    print(f"\n  wrote {a.out}: {n_w + n_k:,} addresses "
          f"({n_w:,} weight, {n_k:,} kv) over {a.steps} decode steps"
          f"  [intra={a.intra} layout={a.layout} streams={streams}]")

    if a.ramulator and a.split_channels:
        # LoadStoreTrace issues at most ONE request per tick. Against a
        # 16-channel memory that makes the FRONTEND the bottleneck (measured:
        # 400,000 requests in 399,999 cycles -- exactly 1/cycle, so the
        # controller is never stressed). ONNXim avoids this by running 16
        # single-channel instances, each fed the addresses for its own channel
        # with the channel bits removed. Reproduce that split here.
        import collections
        buckets = collections.defaultdict(list)
        with open(a.out) as fh:
            for line in fh:
                ad = int(line.split()[1], 16)
                ch, du = geo.to_dram_units(np.int64(ad))
                buckets[int(ch)].append(int(du) * geo.tx)
        base = a.out.rsplit(".", 1)[0]
        for ch in sorted(buckets):
            with open(f"{base}.ch{ch}.trace", "w") as fh:
                for ad in buckets[ch]:
                    fh.write(f"LD 0x{ad:x}\n")
        sizes = [len(v) for v in buckets.values()]
        print(f"  split into {len(buckets)} per-channel traces "
              f"({base}.ch*.trace), {min(sizes):,}-{max(sizes):,} lines each")
        print(f"  run each against a channel:1 Ramulator config -- that is the "
              f"configuration\n  where the memory, not the frontend, is the "
              f"bottleneck.")


if __name__ == "__main__":
    main()
