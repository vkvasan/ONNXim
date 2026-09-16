#!/usr/bin/env python3
"""DRAM energy from ONNXim/Ramulator2 command counters (post-processing).

Ramulator2 has no power model. This applies a JEDEC-IDD-style split to the
per-channel command mix the patched controller prints at the end of a run:

    CTRL cmd mix: RD <n> WR <n> ACT <n> PRE <n> ...
    [DRAM] channel-busy (...) ... dram cycles <n>

    E = ACT * E_act                       (row open + close, per activation)
      + bytes * E_core_bit                (array/column path, per bit)
      + bytes * E_io_bit                  (interface, per bit)   <- where HBM and LPDDR differ most
      + refresh: ceil(cycles/nREFI) * banks * E_ref_bank
      + P_bg_active * t_busy + P_bg_idle * t_idle   (background)

Defaults are literature-class values (HBM: E_act 909 pJ, 3.9 pJ/bit total split
core/IO 1.4/2.5; LPDDR5X-class: 2.0 pJ/bit core+IO ... ) and are the weakest
part of any absolute number. The RATIOS between two runs with identical traffic
depend only on the measured ACT/RD/WR counts and on E_io, so the script also
prints the break-even E_io at which the LPDDR6 run's energy equals the HBM3 run's.

Usage:
  dram_energy.py LOG --mem hbm3|lpddr6 [--tck-ps N] [overrides ...]
  dram_energy.py --compare HBM3_LOG LPDDR6_LOG [--tck-ps-a N --tck-ps-b N]
"""
import argparse, math, re, sys

PRESETS = {
    # pJ per activation, pJ/bit core, pJ/bit IO, pJ per bank refresh, mW active bg, mW idle bg, nREFI cycles
    "hbm3":   dict(e_act=909.0, e_core_bit=1.4, e_io_bit=2.5, e_ref_bank=8000.0, p_bg_act=180.0, p_bg_idle=60.0, nrefi=12500),
    "lpddr6": dict(e_act=700.0, e_core_bit=1.2, e_io_bit=4.0, e_ref_bank=6000.0, p_bg_act=90.0,  p_bg_idle=25.0, nrefi=20833),
}

def parse(path):
    txt = open(path).read()
    mixes = re.findall(r"CTRL cmd mix: RD (\d+) WR (\d+) ACT (\d+) PRE (\d+)", txt)
    cyc = re.findall(r"dram cycles (\d+)", txt)
    busy = re.findall(r"outstanding\) ([0-9.]+)% of DRAM cycles", txt)
    if not mixes or not cyc:
        sys.exit(f"{path}: no CTRL cmd mix / dram cycles lines (run must finish)")
    # controller prints once per channel; sum them
    rd = sum(int(m[0]) for m in mixes); wr = sum(int(m[1]) for m in mixes); act = sum(int(m[2]) for m in mixes)
    nch = len(mixes)
    return dict(rd=rd, wr=wr, act=act, nch=nch, cycles=int(cyc[-1]), busy=float(busy[-1]) / 100 if busy else 1.0)

def energy(s, p, tck_ps, req_bytes=32, banks_per_ch=32):
    bits = (s["rd"] + s["wr"]) * req_bytes * 8
    t_s = s["cycles"] * tck_ps * 1e-12
    n_ref = math.ceil(s["cycles"] / p["nrefi"]) * s["nch"] * banks_per_ch
    e_act = s["act"] * p["e_act"]
    e_core = bits * p["e_core_bit"]
    e_io = bits * p["e_io_bit"]
    e_ref = n_ref * p["e_ref_bank"]
    e_bg = (p["p_bg_act"] * s["busy"] + p["p_bg_idle"] * (1 - s["busy"])) * 1e-3 * t_s * 1e12 * s["nch"] / 16
    tot = e_act + e_core + e_io + e_ref + e_bg
    return dict(total_mJ=tot / 1e9, act_mJ=e_act / 1e9, core_mJ=e_core / 1e9, io_mJ=e_io / 1e9, ref_mJ=e_ref / 1e9,
                bg_mJ=e_bg / 1e9, pj_per_bit=tot / bits, bits=bits, time_ms=t_s * 1e3)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--mem", default="hbm3"); ap.add_argument("--tck-ps", type=float, default=312.0)
    ap.add_argument("--compare", action="store_true", help="logs = HBM3_LOG LPDDR6_LOG")
    ap.add_argument("--tck-ps-a", type=float, default=312.0); ap.add_argument("--tck-ps-b", type=float, default=187.5)
    for k in PRESETS["hbm3"]: ap.add_argument(f"--{k.replace('_','-')}", type=float)
    a = ap.parse_args()
    if a.compare:
        sa, sb = parse(a.logs[0]), parse(a.logs[1])
        pa, pb = dict(PRESETS["hbm3"]), dict(PRESETS["lpddr6"])
        ea, eb = energy(sa, pa, a.tck_ps_a), energy(sb, pb, a.tck_ps_b)
        for name, s, e in (("HBM3", sa, ea), ("LPDDR6", sb, eb)):
            print(f"{name:7s} ACT {s['act']:,}  bits {e['bits']/8/1e6:,.0f} MB  time {e['time_ms']:.2f} ms  "
                  f"E {e['total_mJ']:.1f} mJ = act {e['act_mJ']:.1f} + core {e['core_mJ']:.1f} + IO {e['io_mJ']:.1f} + ref {e['ref_mJ']:.1f} + bg {e['bg_mJ']:.1f}   "
                  f"{e['pj_per_bit']:.2f} pJ/bit")
        # break-even LPDDR6 IO energy: solve eb_total(e_io) == ea_total
        fixed_b = eb["total_mJ"] - eb["io_mJ"]
        be = (ea["total_mJ"] - fixed_b) * 1e9 / eb["bits"] if eb["bits"] else float("nan")
        print(f"LPDDR6 wins on energy iff its IO energy < {be:.2f} pJ/bit (assumed {pb['e_io_bit']:.2f}); "
              f"ratio LPDDR6/HBM3 = {eb['total_mJ']/ea['total_mJ']:.2f}x at the assumed constants")
    else:
        p = dict(PRESETS[a.mem])
        for k in p:
            v = getattr(a, k)
            if v is not None: p[k] = v
        for log in a.logs:
            s = parse(log); e = energy(s, p, a.tck_ps)
            print(f"{log}: ACT {s['act']:,}  {e['bits']/8/1e6:,.0f} MB  {e['time_ms']:.2f} ms  E {e['total_mJ']:.1f} mJ "
                  f"(act {e['act_mJ']:.1f}, core {e['core_mJ']:.1f}, IO {e['io_mJ']:.1f}, ref {e['ref_mJ']:.1f}, bg {e['bg_mJ']:.1f})  {e['pj_per_bit']:.2f} pJ/bit")

if __name__ == "__main__":
    main()
