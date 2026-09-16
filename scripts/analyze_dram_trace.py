#!/usr/bin/env python3
"""Summarize and compare DRAM access traces emitted by ONNXIM_DRAM_TRACE.

The trace is written by DramRamulator2::log_access() in src/Dram.cc, one row
per request handed to Ramulator2:

    cycle,channel,pseudochannel,bankgroup,bank,row,column,address,rw

Usage:
    python3 scripts/analyze_dram_trace.py out/prefill.csv
    python3 scripts/analyze_dram_trace.py out/prefill.csv out/decode.csv

With two or more traces the metrics are printed side by side, which is the
prefill-vs-decode comparison.

Note on the row-hit metric: this is the locality *present in the request
stream*, measured in arrival order per bank. Ramulator2's own "Row hits/misses/
conflicts" counters differ because the FR-FCFS scheduler reorders requests. Use
this number to characterize the access pattern, and Ramulator2's counters to
characterize what the controller achieved.
"""
import argparse
import csv
import sys
from collections import Counter, defaultdict


def load(path):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append(
                (
                    int(r["cycle"]),
                    int(r["channel"]),
                    int(r["pseudochannel"]),
                    int(r["bankgroup"]),
                    int(r["bank"]),
                    int(r["row"]),
                    int(r["column"]),
                    int(r["address"], 16),
                    r["rw"],
                )
            )
    if not rows:
        sys.exit(f"{path}: no rows")
    return rows


def analyze(path):
    rows = load(path)
    n = len(rows)
    reads = sum(1 for r in rows if r[8] == "R")
    writes = n - reads

    first_cycle, last_cycle = rows[0][0], rows[-1][0]
    span = max(1, last_cycle - first_cycle)

    # Row-buffer locality in arrival order, tracked per physical bank.
    open_row = {}
    hits = conflicts = 0
    for _, ch, pch, bg, ba, row, _, _, _ in rows:
        bank = (ch, pch, bg, ba)
        prev = open_row.get(bank)
        if prev is None:
            pass                 # compulsory: bank not yet touched
        elif prev == row:
            hits += 1
        else:
            conflicts += 1
        open_row[bank] = row

    banks_touched = len(open_row)
    rows_touched = len({(ch, pch, bg, ba, row) for _, ch, pch, bg, ba, row, _, _, _ in rows})
    per_channel = Counter(r[1] for r in rows)
    accesses_per_row = Counter((r[1], r[2], r[3], r[4], r[5]) for r in rows)

    # How concentrated is the traffic? Share taken by the busiest channel.
    busiest = max(per_channel.values())
    ideal = n / max(1, len(per_channel))

    # Streaming share: consecutive accesses within a bank+row that advance the
    # column by a constant stride. The stride is detected rather than assumed --
    # ONNXim issues dram_req_size (32 B) requests onto a 16 B DRAM transaction
    # granularity, so a pure stream steps 2 columns at a time, not 1.
    last_col = {}
    deltas = Counter()
    for _, ch, pch, bg, ba, row, col, _, _ in rows:
        bank = (ch, pch, bg, ba)
        prev = last_col.get(bank)
        if prev is not None and prev[0] == row:
            d = col - prev[1]
            if d > 0:
                deltas[d] += 1
        last_col[bank] = (row, col)

    if deltas:
        stride, stride_n = deltas.most_common(1)[0]
        seq_frac = stride_n / sum(deltas.values())
    else:
        stride, seq_frac = 0, 0.0

    return {
        "path": path,
        "requests": n,
        "reads": reads,
        "writes": writes,
        "read_frac": reads / n,
        "cycle_span": span,
        "req_per_kcycle": n / span * 1000,
        "row_hit_frac": hits / max(1, hits + conflicts),
        "banks_touched": banks_touched,
        "rows_touched": rows_touched,
        "accesses_per_row": n / max(1, rows_touched),
        "channels_used": len(per_channel),
        "channel_imbalance": busiest / max(1.0, ideal),
        "seq_frac": seq_frac,
        "stride": stride,
        "row_reuse_p50": sorted(accesses_per_row.values())[len(accesses_per_row) // 2],
        "row_reuse_max": max(accesses_per_row.values()),
    }


FIELDS = [
    ("requests", "DRAM requests", "{:,.0f}"),
    ("reads", "  reads", "{:,.0f}"),
    ("writes", "  writes", "{:,.0f}"),
    ("read_frac", "  read fraction", "{:.3f}"),
    ("cycle_span", "DRAM cycles spanned", "{:,.0f}"),
    ("req_per_kcycle", "requests / 1k cycles", "{:,.1f}"),
    ("row_hit_frac", "row hit rate (arrival order)", "{:.3f}"),
    ("stride", "dominant column stride", "{:,.0f}"),
    ("seq_frac", "  share at that stride", "{:.3f}"),
    ("banks_touched", "distinct banks touched", "{:,.0f}"),
    ("rows_touched", "distinct rows touched", "{:,.0f}"),
    ("accesses_per_row", "accesses per row", "{:,.1f}"),
    ("row_reuse_p50", "row reuse (median)", "{:,.0f}"),
    ("row_reuse_max", "row reuse (max)", "{:,.0f}"),
    ("channels_used", "channels used", "{:,.0f}"),
    ("channel_imbalance", "busiest/mean channel", "{:.2f}"),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("traces", nargs="+", help="trace CSV(s) to summarize")
    args = ap.parse_args()

    stats = [analyze(p) for p in args.traces]

    labels = [s["path"].rsplit("/", 1)[-1].replace(".csv", "") for s in stats]
    width = max(30, max(len(l) for l in labels) + 2)

    print()
    print("metric".ljust(32) + "".join(l.rjust(width) for l in labels))
    print("-" * (32 + width * len(labels)))
    for key, label, fmt in FIELDS:
        line = label.ljust(32)
        for s in stats:
            line += fmt.format(s[key]).rjust(width)
        print(line)
    print()


if __name__ == "__main__":
    main()
