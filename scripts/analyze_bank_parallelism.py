#!/usr/bin/env python3
"""Is PagedAttention serialising onto banks?

PagedAttention gives the KV cache a power-of-two block stride. With the shipped
RoBaRaCoCh mapping the bank field ends at bit 15, so any stride that is a
multiple of 2^16 lands on the SAME bank and only changes the row. If a block is
small, each per-head run is short and the stream keeps returning to one bank with
a different row every time -- bank serialisation.

This measures it directly, per channel:

  same-bank %   share of consecutive requests that hit the same bank.
                Low = the stream spreads; high = it is stuck on one bank.
  banks/32      distinct banks among 32 consecutive requests. The channel has 32
                banks (2 pch x 4 bg x 4 ba), so 32 = perfect spread, 1 = fully
                serialised.
  same-bank-diff-row %
                consecutive same-bank pairs that also change row. THIS is the
                serialising case: the bank must precharge and re-activate, and
                nothing else can use it meanwhile.

Weights and KV are separated by address, matching RAMULATOR_WEIGHT_LIMIT.

Usage:
    python3 scripts/analyze_bank_parallelism.py out/pagedtrace/contig.csv out/pagedtrace/b16.csv
"""
import argparse
import collections


def analyze(path, max_rows, weight_limit):
    """Per channel, walk consecutive requests and characterise bank behaviour."""
    prev = {}                                    # channel -> (bank, row, is_kv)
    stat = {k: collections.Counter() for k in ("w", "k")}
    win = {k: collections.defaultdict(list) for k in ("w", "k")}
    cur = {k: collections.defaultdict(list) for k in ("w", "k")}
    n = 0
    with open(path) as f:
        next(f)
        for line in f:
            if n >= max_rows:
                break
            p = line.split(',')
            try:
                ch = int(p[1]); pch = int(p[2]); bg = int(p[3]); ba = int(p[4])
                row = int(p[5]); addr = int(p[7], 16)
            except (ValueError, IndexError):
                continue
            n += 1
            kind = "w" if addr < weight_limit else "k"
            bank = (pch << 4) | (bg << 2) | ba          # 32 banks within a channel
            key = (kind, ch)
            if key in prev:
                pb, pr = prev[key]
                s = stat[kind]
                s["pairs"] += 1
                if bank == pb:
                    s["same_bank"] += 1
                    if row != pr:
                        s["same_bank_diff_row"] += 1
            prev[key] = (bank, row)
            c = cur[kind][ch]
            c.append(bank)
            if len(c) == 32:
                win[kind][ch].append(len(set(c)))
                c.clear()
    out = {}
    for kind in ("w", "k"):
        s = stat[kind]
        spreads = [v for ch in win[kind].values() for v in ch]
        out[kind] = {
            "pairs": s["pairs"],
            "same_bank": s["same_bank"] / max(1, s["pairs"]) * 100,
            "sb_diff_row": s["same_bank_diff_row"] / max(1, s["pairs"]) * 100,
            "spread": sum(spreads) / max(1, len(spreads)),
        }
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("traces", nargs="+")
    ap.add_argument("--max-rows", type=int, default=3_000_000)
    ap.add_argument("--weight-limit", type=int, default=500_000_000,
                    help="addresses below this are weights (match RAMULATOR_WEIGHT_LIMIT)")
    args = ap.parse_args()

    print(f"{'trace':<14}{'stream':>8}{'pairs':>12}{'same-bank':>11}"
          f"{'same-bank+diff-row':>20}{'banks/32':>10}")
    print("-" * 78)
    for t in args.traces:
        try:
            r = analyze(t, args.max_rows, args.weight_limit)
        except OSError as e:
            print(f"{t:<14}  (unreadable: {e})")
            continue
        name = t.split('/')[-1].replace('.csv', '')
        for kind, label in (("w", "weights"), ("k", "kv+act")):
            d = r[kind]
            if d["pairs"] == 0:
                continue
            print(f"{name:<14}{label:>8}{d['pairs']:>12,}{d['same_bank']:>10.1f}%"
                  f"{d['sb_diff_row']:>19.1f}%{d['spread']:>10.1f}")
    print("\n  banks/32: 32 = perfect spread across the channel's banks, 1 = serialised")
    print("  same-bank+diff-row is the costly case: forced precharge + activate,")
    print("  with the bank unavailable to anyone else meanwhile.")


if __name__ == "__main__":
    main()
