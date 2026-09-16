#!/usr/bin/env python3
"""Locality analysis and offline mapping sweeps for ONNXim DRAM traces.

Two kinds of question, answered without re-running the simulator:

  1. LOCALITY -- is there reuse to exploit, and how far apart is it?
     Depends only on the address sequence, so it is a property of the
     workload, not of the memory system.

  2. WHAT-IF -- how would a different address mapping or a bigger controller
     window change row-buffer behaviour? The trace carries the raw untranslated
     `address`, so the mapping can be re-applied offline.

OPEN-LOOP CAVEAT (important):
    Arrival cycles in the trace were produced by the simulated machine *with its
    memory system*. Change the mapping and the core would stall differently, so
    arrivals would shift. Therefore:
      valid offline  -- anything derived from the address SEQUENCE: row hit rate
                        in arrival order, reuse interval, bank spread, strides.
      NOT valid      -- end-to-end runtime or latency predictions.
    Use this to shortlist candidates cheaply, then confirm the top few in ONNXim.

Calibration: offline hit rates run ~10 points below the simulator's, because
FR-FCFS reorders to capture reuse that strict arrival order misses. Treat the
offline number as a pessimistic bound and compare configurations, not absolutes.

Usage:
    python3 scripts/analyze_locality.py out/llama_dec.csv
    python3 scripts/analyze_locality.py out/llama_dec.csv --max-rows 5000000
    python3 scripts/analyze_locality.py out/decode.csv out/prefill.csv --label decode prefill
"""
import argparse
import os
import sys

import numpy as np

# ONNXim address translation, from src/Dram.cc DramRamulator2::push():
#   ram_addr = (dram_address >> tx_ch_log2) << tx_log2
# with tx_log2 = log2(dram_req_size) and tx_ch_log2 = log2(n_ch) + tx_log2.
# The DRAM then decodes ram_addr >> tx_offset in prefetch-size units.
DEFAULT_CHANNELS = 16
DEFAULT_REQ_SIZE = 32   # bytes
DEFAULT_TX_BYTES = 16   # prefetch_size * channel_width / 8


def load(path, max_rows):
    """Return (channel, address) arrays. Only the two columns we need."""
    chans, addrs = [], []
    with open(path) as f:
        next(f)
        for i, line in enumerate(f):
            if i >= max_rows:
                break
            p = line.split(',')
            chans.append(int(p[1]))
            addrs.append(int(p[7], 16))
    if not chans:
        sys.exit(f"{path}: no rows")
    return np.array(chans, dtype=np.int16), np.array(addrs, dtype=np.int64)


def to_dram_units(addr, n_ch=DEFAULT_CHANNELS, req=DEFAULT_REQ_SIZE, tx=DEFAULT_TX_BYTES):
    """Untranslated address -> per-channel address in DRAM transaction units."""
    tx_ch_log2 = int(np.log2(n_ch)) + int(np.log2(req))
    tx_log2 = int(np.log2(req))
    ram = (addr >> tx_ch_log2) << tx_log2
    return ram >> int(np.log2(tx))


def decode(base, chan, colbits, pchbits, bgbits, babits):
    """Apply a RoBaRaCoCh-style slicing. Returns (bank_id, row)."""
    a = base.copy()
    a >>= colbits
    pch = a & ((1 << pchbits) - 1); a >>= pchbits
    bg = a & ((1 << bgbits) - 1);   a >>= bgbits
    ba = a & ((1 << babits) - 1);   a >>= babits
    row = a
    shift = pchbits + bgbits + babits
    bank = (chan.astype(np.int64) << shift) | (pch << (bgbits + babits)) | (bg << babits) | ba
    return bank, row


def row_hit_rate(bank, row):
    """Hit rate in arrival order, per bank (no scheduler reordering)."""
    order = np.lexsort((np.arange(len(bank)), bank))
    b, r = bank[order], row[order]
    same = b[1:] == b[:-1]
    hits = np.count_nonzero(same & (r[1:] == r[:-1]))
    conf = np.count_nonzero(same & (r[1:] != r[:-1]))
    return hits / max(1, hits + conf)


def bank_spread(bank, window=64):
    """Mean distinct banks among `window` consecutive requests to one channel."""
    n = min(len(bank), 2_000_000)
    counts = [len(np.unique(bank[s:s + window])) for s in range(0, n - window, window)]
    return float(np.mean(counts)) if counts else 0.0


def reuse_intervals(bank, row, chan):
    """Requests between successive accesses to the same (bank,row).

    Measured WITHIN EACH CHANNEL, because Ramulator2's read request buffer
    (ReqBuffer::max_size, base/request.h:45, default 64) is per channel. A
    global-stream interval would overstate the required depth by ~n_channels,
    since requests are spread across channels by the ipoly hash.

    So W here is directly comparable to RAMULATOR_REQBUF / ReqBuffer::max_size.
    """
    out = []
    for c in np.unique(chan):
        m = chan == c
        b, r = bank[m], row[m]
        key = (b.astype(np.int64) << 32) | (r & 0xFFFFFFFF)
        order = np.argsort(key, kind='stable')
        k, pos = key[order], order          # pos = index within THIS channel
        same = k[1:] == k[:-1]
        out.append(np.abs(np.diff(pos))[same])
    return np.concatenate(out) if out else np.array([], dtype=np.int64)


def analyze(path, label, max_rows, out_dir):
    chan, addr = load(path, max_rows)
    base = to_dram_units(addr)
    print(f"\n{'=' * 74}\n{label}   ({len(addr):,} requests from {path})\n{'=' * 74}")

    # --- Group 1: locality, mapping-independent ------------------------------
    bank, row = decode(base, chan, 3, 1, 2, 2)     # the shipped tuned mapping
    ivals = reuse_intervals(bank, row, chan)
    uniq_rows = len(np.unique((bank.astype(np.int64) << 32) | (row & 0xFFFFFFFF)))
    print(f"\nLOCALITY")
    print(f"  distinct (bank,row) touched : {uniq_rows:,}")
    print(f"  accesses per row            : {len(addr)/max(1,uniq_rows):.1f}")
    if len(ivals):
        print(f"  reuse events                : {len(ivals):,}")
        print(f"  reuse interval  median      : {int(np.median(ivals)):,} requests")
        print(f"                  p90         : {int(np.percentile(ivals,90)):,}")
        print(f"\n  fraction of row reuse captured by a per-channel ReqBuffer of W entries:")
        for W in (16, 32, 64, 128, 256, 1024, 4096):
            print(f"    W = {W:>6} : {np.count_nonzero(ivals <= W)/len(ivals)*100:5.1f}%")

    # --- strides -------------------------------------------------------------
    d = np.diff(base[:2_000_000])
    pos = d[(d > 0) & (d < 4096)]
    if len(pos):
        vals, cnts = np.unique(pos, return_counts=True)
        top = np.argsort(cnts)[::-1][:5]
        print(f"\n  dominant address strides (DRAM transaction units):")
        for i in top:
            print(f"    {int(vals[i]):>6} : {cnts[i]/len(d)*100:5.1f}%")

    # --- Group 3: mapping sweep ---------------------------------------------
    print(f"\nMAPPING SWEEP (offline; hit rate is a pessimistic bound, see header)")
    print(f"  {'col bits':>9}{'B per bank':>12}{'row hit':>10}{'banks/64':>11}")
    print("  " + "-" * 42)
    sweep = []
    for cb in range(0, 8):
        bk, rw = decode(base, chan, cb, 1, 2, 2)
        hr = row_hit_rate(bk, rw)
        bs = bank_spread(bk)
        sweep.append((cb, (1 << cb) * DEFAULT_TX_BYTES, hr * 100, bs))
        print(f"  {cb:>9}{(1<<cb)*DEFAULT_TX_BYTES:>12}{hr*100:>9.1f}%{bs:>11.1f}")

    plot(label, ivals, sweep, out_dir)
    return sweep


def plot(label, ivals, sweep, out_dir):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib not available - skipping plots)")
        return

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    fig.suptitle(f"DRAM locality - {label}", fontsize=13)

    # 1. reuse interval CDF -> controller window sizing
    if len(ivals):
        s = np.sort(ivals)
        y = np.arange(1, len(s) + 1) / len(s) * 100
        ax[0].semilogx(s, y)
        ax[0].set_xlabel("requests between reuses (same channel)")
        ax[0].set_ylabel("% of row reuse captured")
        ax[0].set_title("Required ReqBuffer depth (per channel)")
        ax[0].grid(alpha=.3)
        for W in (64, 1024):  # stock default, and the value we swept
            ax[0].axvline(W, ls='--', lw=.8, color='crimson')
            ax[0].text(W, 5, f" W={W}", fontsize=8, color='crimson')

    cb = [s[0] for s in sweep]; hr = [s[2] for s in sweep]; bs = [s[3] for s in sweep]
    # 2. the two effects vs mapping
    ax[1].plot(cb, hr, 'o-', label="row hit %")
    ax[1].set_xlabel("column bits (more = larger run per bank)")
    ax[1].set_ylabel("row hit %")
    a2 = ax[1].twinx(); a2.plot(cb, bs, 's--', color='seagreen', label="banks / 64 reqs")
    a2.set_ylabel("banks per 64 requests")
    ax[1].set_title("Row locality vs bank parallelism")
    ax[1].grid(alpha=.3)

    # 3. the Pareto view
    ax[2].plot(bs, hr, 'o-')
    for c, h, b in zip(cb, hr, bs):
        ax[2].annotate(f"col{c}", (b, h), fontsize=8,
                       textcoords="offset points", xytext=(4, 4))
    ax[2].set_xlabel("banks per 64 requests")
    ax[2].set_ylabel("row hit %")
    ax[2].set_title("Pareto: which mapping to pick")
    ax[2].grid(alpha=.3)

    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, f"locality_{label}.png")
    fig.savefig(p, dpi=120)
    print(f"\n  plot -> {p}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("traces", nargs="+")
    ap.add_argument("--label", nargs="*", default=None)
    ap.add_argument("--max-rows", type=int, default=3_000_000,
                    help="cap rows read per trace (default 3M)")
    ap.add_argument("--out-dir", default="out/plots")
    args = ap.parse_args()

    labels = args.label or [os.path.basename(t).replace('.csv', '') for t in args.traces]
    for t, l in zip(args.traces, labels):
        analyze(t, l, args.max_rows, args.out_dir)


if __name__ == "__main__":
    main()
