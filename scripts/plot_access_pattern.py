#!/usr/bin/env python3
"""Visualise the shape of a DRAM access stream from an ONNXim trace.

Four views, each answering a different question about the pattern:

  1. address vs time      -- what does the stream actually look like? Weight
                             streaming shows as diagonal sweeps, one per GEMM;
                             KV-cache reads sit in a separate band.
  2. per-channel strides  -- spatial regularity. MUST be per channel: the ipoly
                             hash sends consecutive addresses to different
                             channels, so a global stride histogram is noise.
  3. bank x time          -- is bank usage spread or concentrated, and does the
                             concentration move over time?
  4. row-visit histogram  -- how many times a row is touched before it is done.
                             Sets the ceiling on any row-buffer policy.

Usage:
    python3 scripts/plot_access_pattern.py out/llama_dec.csv --label llama7b_decode
"""
import argparse
import os

import numpy as np

N_CH, REQ, TX = 16, 32, 16
COLBITS, PCHBITS, BGBITS, BABITS = 3, 1, 2, 2     # the shipped tuned mapping


def load(path, max_rows):
    cyc, ch, ad = [], [], []
    with open(path) as f:
        next(f)
        for i, line in enumerate(f):
            if i >= max_rows:
                break
            p = line.split(',')
            cyc.append(int(p[0])); ch.append(int(p[1])); ad.append(int(p[7], 16))
    return (np.array(cyc, dtype=np.int64), np.array(ch, dtype=np.int16),
            np.array(ad, dtype=np.int64))


def decode(addr, chan):
    tx_ch = int(np.log2(N_CH)) + int(np.log2(REQ))
    base = ((addr >> tx_ch) << int(np.log2(REQ))) >> int(np.log2(TX))
    a = base >> COLBITS
    pch = a & ((1 << PCHBITS) - 1); a >>= PCHBITS
    bg = a & ((1 << BGBITS) - 1);   a >>= BGBITS
    ba = a & ((1 << BABITS) - 1);   a >>= BABITS
    shift = PCHBITS + BGBITS + BABITS
    bank = (chan.astype(np.int64) << shift) | (pch << (BGBITS + BABITS)) | (bg << BABITS) | ba
    return base, bank, a          # a is now the row index


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace")
    ap.add_argument("--label", default=None)
    ap.add_argument("--max-rows", type=int, default=2_000_000)
    ap.add_argument("--out-dir", default="out/plots")
    args = ap.parse_args()
    label = args.label or os.path.basename(args.trace).replace('.csv', '')

    cyc, chan, addr = load(args.trace, args.max_rows)
    base, bank, row = decode(addr, chan)
    print(f"{label}: {len(addr):,} requests, "
          f"address span {addr.min():#x}..{addr.max():#x} "
          f"({(addr.max()-addr.min())/1e6:.0f} MB)")

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(15, 9))
    fig.suptitle(f"DRAM access pattern - {label}", fontsize=14)

    # --- 1. address vs time ------------------------------------------------
    step = max(1, len(addr) // 120_000)
    ax[0][0].scatter(cyc[::step], addr[::step] / 1e6, s=.4, alpha=.35, lw=0)
    ax[0][0].set_xlabel("DRAM cycle"); ax[0][0].set_ylabel("address (MB)")
    ax[0][0].set_title("Address vs time - the stream's fingerprint")
    ax[0][0].grid(alpha=.3)

    # --- 2. per-channel stride ---------------------------------------------
    strides = []
    for c in range(N_CH):
        b = base[chan == c]
        if len(b) > 1:
            strides.append(np.diff(b))
    d = np.concatenate(strides) if strides else np.array([0])
    pos = d[(d > 0) & (d <= 512)]
    vals, cnts = np.unique(pos, return_counts=True)
    keep = np.argsort(cnts)[::-1][:12]
    v, c_ = vals[keep], cnts[keep] / len(d) * 100
    o = np.argsort(v)
    ax[0][1].bar([str(int(x)) for x in v[o]], c_[o], color='steelblue')
    ax[0][1].set_xlabel("stride (DRAM transaction units, within one channel)")
    ax[0][1].set_ylabel("% of consecutive pairs")
    ax[0][1].set_title(f"Per-channel strides  (top 12; +ve strides = "
                       f"{len(pos)/len(d)*100:.0f}% of all)")
    ax[0][1].grid(alpha=.3, axis='y')

    # --- 3. bank x time ----------------------------------------------------
    nb, nt = 64, 150
    bmax = bank.max() + 1
    H, _, _ = np.histogram2d(bank, cyc, bins=[np.linspace(0, bmax, nb + 1),
                                              np.linspace(cyc.min(), cyc.max(), nt + 1)])
    im = ax[1][0].imshow(np.log1p(H), aspect='auto', origin='lower', cmap='magma',
                         extent=[cyc.min(), cyc.max(), 0, bmax])
    ax[1][0].set_xlabel("DRAM cycle"); ax[1][0].set_ylabel("bank id (channel x pch x bg x ba)")
    ax[1][0].set_title("Bank activity over time  (log scale)")
    fig.colorbar(im, ax=ax[1][0], label="log(1+accesses)")

    # --- 4. row-visit histogram --------------------------------------------
    key = (bank.astype(np.int64) << 32) | (row & 0xFFFFFFFF)
    _, counts = np.unique(key, return_counts=True)
    ax[1][1].hist(counts, bins=np.logspace(0, np.log10(max(2, counts.max())), 40),
                  color='seagreen')
    ax[1][1].set_xscale('log'); ax[1][1].set_yscale('log')
    ax[1][1].set_xlabel("accesses to a given (bank,row)")
    ax[1][1].set_ylabel("number of rows")
    ax[1][1].set_title(f"Row-visit distribution  (mean {counts.mean():.1f}, "
                       f"max {counts.max()})")
    ax[1][1].grid(alpha=.3)

    fig.tight_layout()
    os.makedirs(args.out_dir, exist_ok=True)
    p = os.path.join(args.out_dir, f"pattern_{label}.png")
    fig.savefig(p, dpi=110)
    print(f"  -> {p}")

    # text summary
    print(f"\n  strides (within a channel): top 5")
    for i in np.argsort(cnts)[::-1][:5]:
        print(f"    {int(vals[i]):>5} units ({int(vals[i])*TX:>6} B) : {cnts[i]/len(d)*100:5.1f}%")
    print(f"  rows touched : {len(counts):,}   mean visits {counts.mean():.1f}   "
          f"median {int(np.median(counts))}   max {counts.max()}")


if __name__ == "__main__":
    main()
