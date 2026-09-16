# DRAM access patterns for LLM decode on an NPU — what is measured, what is not

Scope: LLaMA-2 7B, one transformer block, ONNXim + Ramulator2, HBM3 16ch/2pch/4bg/4bank,
1 KB row, 32 B request, 819.2 GB/s peak. Configs `configs/_c128.json` + `_c128.yaml`.

This file separates **measured** results from **inferred** ones, because several
confident inferences in this work turned out to be wrong when simulated. The
inference track record is bad enough that it is itself a finding: see
"Where analysis failed" at the end.

---

## 1. Measured, and load-bearing

### Traffic is dominated by weights, but the ratio is workload-dependent

| workload | KV share of DRAM traffic |
|---|---|
| `v8m1024` (8 req, ~1k ctx) | 23.2% |
| whole generation, all 32 layers, 12 req | 3.9% |
| `wl11`/`wl22`/`wl33` (8 req, 154-3851 ctx) | 20-27% |
| `v32m4096` (32 req, 4k ctx) | **78.5%** |

The weight stream is *invariant*: identical every layer and every decode step,
only the base address changes. Only KV grows. Any single-step result is one
point on this curve, and the curve spans 4%–79%.

### The weight layout swizzle is the largest lever on the weight stream

Tile-major storage (`ONNXIM_WEIGHT_SWIZZLE=1`, `make_address_tiled`):

    row hit  73.6% -> 95.7%      while-busy BW  90.6% -> 98.1%

Replicated across three further independent workloads:

| workload | off: hit / confl | on: hit / confl | BW gain | speedup |
|---|---|---|---|---|
| wl11 | 72.8% / 25.9% | 95.8% / 4.0% | +7.9 | 1.08x |
| wl22 | 72.4% / 26.3% | 95.6% / 4.1% | +6.7 | 1.07x |
| wl33 | 73.5% / 25.2% | 95.8% / 3.9% | +8.3 | 1.09x |

**This is a local patch, not stock ONNXim.** Stock uses plain row-major
`make_address`. NeuPIMs, which embeds ONNXim as its NPU half, inherits the
unswizzled behaviour.

**Both streams now sit within ~1.2 points of the physical ceiling.** A 1 KB row
holds 32 x 32 B requests, so perfectly sequential access gives 31/32 = 96.9%.
Weights row-major sat 23.3 points below it; KV was already at it, which is why
there is no "KV swizzle" to apply. No headroom remains on either stream.

**The simulator is deterministic** -- identical config, two runs, bit-identical
(777,168 cycles, 95.80%, 98.39%). There is no run-to-run noise, so every delta
reported here is real; the only question is whether it is LARGE.

### KV layout: head-major beats block-major on every workload measured

Six workloads, 4 cores, swizzle on, block 16, RoBaRaCoCh, ONNXim's allocator.
The margin tracks KV's share of DRAM traffic and saturates near 2x.

| workload | reqs | KV share | head hit | head BW | block hit | block BW | ratio |
|---|---|---|---|---|---|---|---|
| `wl33` | 8 | 15.4% | 95.9% | 98.1% | 78.4% | 80.6% | 1.22x |
| `v8m1024` | 8 | 23.2% | 96.9% | 98.0% | 76.1% | 72.1% | 1.34x |
| `wl11` | 8 | 28.8% | 94.7% | 98.4% | 77.4% | 69.1% | 1.41x |
| `wl22` | 8 | 34.6% | 96.5% | 97.5% | 77.8% | 66.5% | 1.45x |
| `v32m4096` | 32 | 78.2% | 91.3% | 95.4% | 77.3% | 45.8% | **2.03x** |
| **`az128` (REAL Azure)** | 128 | 89.5% | 94.8% | 94.8% | 76.4% | 45.8% | **1.98x** |

`az128` is 128 real requests from the Azure LLM Inference Dataset at
`max_batch 128` -- production context distribution, realistic batch.

Block-major's row hit is pinned at **76-78% on every workload** -- 8 requests
or 128, synthetic or real, 1k or 4k contexts. Its 4 KB per-head fragment is a
quarter-row per channel by construction, independent of everything else.

### Layout dominates the controller by an order of magnitude (KV-heavy)

`v32m4096`, 4 cores, identical work (81,460,378 vs 81,460,399 accesses):

| arm | w hit | kv hit | kv conflict | cycles | while-busy BW |
|---|---|---|---|---|---|
| `base_head` | 79.1% | 91.2% | 7.5% | 3,685,039 | 93.9% |
| `rb256rh_head` | 70.0% | 89.1% | — | 3,595,235 | 96.1% |
| `base_block` | 75.8% | 77.1% | 20.3% | 7,364,601 | 45.7% |
| `rb256rh_block` | 70.1% | 77.3% | — | 6,759,074 | 49.8% |

Layout is worth **48 points** of bandwidth; the controller is worth 2–4.

### Controller mechanisms are worth 2–4 points, not zero

`RAMULATOR_REQBUF=256` + `RAMULATOR_SCHED_ROWHIT=1` gives +2.2 (head) and +4.1
(block) points of while-busy BW, consistently, in both traffic regimes. An
earlier "every controller mechanism is a no-op" claim was drawn from the
weight-heavy workload alone and overstated.

Genuinely null (bit-identical to baseline): `RAMULATOR_ACTBUF`, `RAMULATOR_RETRY`.
Both were predicted null by instrumentation first — `m_active_buffer` sits at
2.0/64 occupancy and the 2.3 row-conflict path fires 7 times in 848,580 stalls.

### The bandwidth floor is very low

    acc/ACT >= (channel peak) x tRC / (req_bytes x banks) = 0.95

Everything measured sits at 2.9–22.6. Row locality is not what feeds the bus,
which is why 70%+ row hit is a perfectly adequate target.

### Intra-tile ordering is visible but does not affect bandwidth

Simulated via `ONNXIM_INTRA_ORDER` (`GemmWS.cc`, needs both a loop permutation
and a bypass of the `std::set` that re-sorts addresses):

| order | row hit | while-busy BW |
|---|---|---|
| seq | 95.7% | 98.1% |
| blk | 94.8% | 98.5% |
| col | 87.9% | 97.8% |

7.8 points of row hit, 0.7 points of bandwidth. Sequential — what ONNXim
already emits — is best.

---

## 2. Measured, smaller effects

- **Block-table fragmentation** costs 9.4 points of KV row hit (96.8% -> 87.5%)
  on an identical workload and layout. Physical block IDs carry the serving
  history, so row locality is partly a property of the allocator.
- **Paging is free in prefill** (1.001x) and costs 1.07–1.67x in decode.
- **Parallelism strategy**: head-parallel beat request- and seq-parallel by 3.3x,
  from load imbalance (611,200 rows on one core vs 6,528), not memory behaviour.
- **Non-power-of-two array dims** cost 5.1x (90x90) and 7.2x (45x45). Treat as
  infeasible, not as a penalty term.

---

## 3. Not measured — do not rely on

- **Stationarity at batch > 1.** `SystolicOS` is `assert(0)`; only weight-stationary
  exists. Provably null at batch 1; unknown above it.
- **Anything at int8 or block != 16** for the stride-sensitive results.
- **Multi-step generation inside ONNXim.** `--emit-workload` now writes the
  sampled generation length into `target_length` (LanguageScheduler.cc:156 makes
  that the token count), but only the single-step case has been run to
  completion.
- **Tiling above batch 128.** See §7: tiling is inert only while the token
  dimension is padded up to the array height. Beyond batch 128 re-reads appear
  and tiling becomes a real axis. Untested there.
- **Timeloop-searched mappings.** ONNXim derives tiling from a closed-form
  heuristic (`Mapping.cc:52`) with no search — one `sqrt` and a capacity
  division. `MappingTable::parse_mapping_file` can load an external `.mapping`,
  but none was used.


---

## 4. Where analysis failed -- and the fix

**RESOLVED.** The model mis-ranked layouts and orderings four times. The root
cause was a single wrong assumption, now corrected.

`scripts/tops_visible.py` predicts an address stream analytically and is
**validated to ±1.1 points on row-hit rate** for a given stream (weights 72.5%
vs 73.6%; KV 96.8% vs 95.7%). That validation is real.

It was nonetheless **wrong four times out of four at RANKING** layouts or
orderings:

1. **Intra-tile ordering, first attempt.** Hand analysis on raw byte addresses
   said column-major was 8x worse. It ignored that channel bits sit *below* row
   bits, striping 16 consecutive requests across 16 channels.
2. **Intra-tile ordering, second attempt.** The channel-aware model said col was
   22.8 points *better*. Simulation: 7.8 points worse.
3. **KV layout.** The model predicted head-major aliases catastrophically
   (96.9% -> 0.0%) at >=4 cores while block-major is immune, and a bank-collision
   predicate agreed on all 25 cells checked. Simulation says the opposite:
   head-major is **2x faster** at every core count.
4. **The alias predicate itself**, which was briefly wired into `--search` as a
   guard recommending block-major. Removed.

### The root cause: scoring an interleaved stream

All four shared one assumption -- that the controller sees the N core streams
**round-robin interleaved**. Scoring that way:

| scored at | KV layout ranking vs simulation |
|---|---|
| `streams = N` (round-robin) | **2/5 correct** |
| `streams = 1` (un-interleaved) | **5/5 correct** |

It also flips intra-tile ordering the right way round (`col` worst, matching
simulation, instead of `col` best) and removes the phantom aliasing cliff.

The physical reason: lockstep interleaving is not what the hardware does. Cores
issue in **bursts** into per-channel buffers, so locality *within* one core's
stream survives to the controller. A strict round-robin destroys it artificially.

`SCORE_STREAMS = 1` encodes this. Emission still uses the real core count -- the
constant only affects how candidates are RANKED.

### Current status: 8/8

`--verify-ranking` re-checks the model against every simulation run:

    KV layout @ 1/2/4/8/16 cores    head    head    OK  (x5)
    intra seq vs blk                 seq     tie    OK
    intra seq vs col                 seq     seq    OK
    intra blk vs col                 blk     blk    OK
    -> 8/8 rankings agree

The tool now DERIVES its choices rather than looking them up. `LAYOUT_MEASURED`
and `INTRA_MEASURED` are the **test set**, not the answer key.

**Remaining limits.** Ranking only -- absolute values read high (96.9% vs the
simulator's 91-96%), so use simulation for magnitudes. Sub-1-point differences
are unresolvable (`seq` vs `blk` ties at the 96.9% ceiling). And 8 comparisons
from one model and one workload family is a hypothesis with 8/8 support, not a
proven property: re-run `--verify-ranking` whenever new simulation data arrives.

The general lesson stands: every failure came from reasoning about addresses
*before* applying the full mapping -- channel striping, real allocator output,
and asynchronous cores each broke an assumption that looked safe.

---

## 5. Tools

| script | what it does | validated? |
|---|---|---|
| `tops_visible.py` | vLLM sim -> search -> trajectory -> trace | ±1.1 pts on row hit; **8/8 on ranking** |
| `vllm_trace.py` | objective+model+hardware -> addresses | KV 8/8 exact, weights 1.0002 |
| `gen_trace.py` | serving sim + address generation | 8/8 |
| `tops_search.py` | hardware search under an objective | analytical, ±15% |
| `tops_mapspace.py` | fixed HW+workload -> best tiling **by traffic** | traffic only; controller-blind |

`tops_visible.py --trajectory` covers a whole generation (544.6G addresses,
15.9 TiB) without materialising it, by exploiting weight-stream invariance —
a literal trace would be ~10.9 TB of CSV.

## 6. Simulator modifications

All default to stock behaviour when the env var is unset.

| knob | file | effect |
|---|---|---|
| `ONNXIM_WEIGHT_SWIZZLE` | `Operation.cc` | tile-major weight storage |
| `ONNXIM_INTRA_ORDER` | `GemmWS.cc` | intra-tile traversal seq/col/blk |
| `ONNXIM_KV_BLOCK`/`_LAYOUT`/`_POOL`/`_ALLOC`/`_UNSORTED` | `Attention.cc` | paging, layout, pooling |
| `ONNXIM_PAR_STRATEGY` | `Attention.cc` | head / request / seq parallel |
| `RAMULATOR_REQBUF`/`ACTBUF`/`RETRY` | `generic_dram_controller.cpp` | buffer depths, candidate retry |
| `RAMULATOR_SCHED_ROWHIT`/`STARVE_CAP` | `generic_scheduler.cpp` | the row-hit tier FRFCFS lacks |
| `ONNXIM_DRAM_TRACE`/`_LIMIT`/`_OCC` | `Dram.cc` | address tracing |

Note: stock `FRFCFS::compare` ranks ready-over-unready then falls back to
arrival order, so a row hit and a row miss that are both timing-ready tie on
age. It is FCFS-among-ready, not first-ready-first-row-hit.

Also fixed: `KVCacheConcat.cc` derived `value_out_address` from
`key_cache_tensor`, so V writes landed on K's address.


---

## 7. Why tiling is inert — the CORRECT reason

Earlier drafts of this file said "decode is GEMV, so there is no reuse." That is
wrong. At batch 32 there genuinely is 32x reuse available. The real mechanism:

    Mapping.cc pads the token dimension UP to the array height (128) before
    tiling, so inner_I >= 128 for any batch <= 128. Weight re-reads =
    ceil(batch / inner_I) = 1, and no tiling choice can change it.

| accum KB | batch | inner_I | re-reads | W traffic |
|---|---|---|---|---|
| 2048 | 32 | 128 | 1 | 96 MB |
| 2048 | 128 | 128 | 1 | 96 MB |
| 2048 | 256 | 256 | 1 | 96 MB |
| 2048 | **512** | 256 | **2** | 192 MB |
| 512 | **256** | 128 | **2** | 192 MB |

So the scope is **batch <= array height**, and above that the accumulator size
decides. This is arithmetic on `derive_tiling`, which reproduces `Mapping.cc`
exactly; it is not a memory-behaviour prediction and does not need simulating.

It also reconciles with FlashDecoding++ and FlashGEMM, which DO find tiling and
stationarity to matter for batched decode on GPUs -- where the accumulator is
registers/SMEM, far smaller relative to batch.

## 8. KV layout across attention type and address mapper

`v32m4096`, 4 cores, swizzle on. Ratio = block-major cycles / head-major cycles.

| model | RoBaRaCoCh | XOR | XOR erodes |
|---|---|---|---|
| MHA (32 kv heads) | **2.03x** | **1.24x** | 76% |
| GQA-4 (8 kv heads) | **1.48x** | **1.27x** | 44% |
| MQA (1 kv head) | 1.00x | 1.00x | — |

MQA is **bit-identical** (1,079,742 cycles both layouts) because with one KV
head the two index orders are the same addresses -- a sanity check that the
mechanism is understood rather than curve-fitted.

**The defensible claim is the floor: 1.24x**, holding across MHA and GQA, with
and without bank hashing. Not the headline 2.03x, which is the best case.

## 9. Allocation model — resolved against vLLM's real allocator

vLLM v1 `block_pool.py` uses a deque: `popleft` to allocate, freed non-cached
blocks **prepended** (LIFO "for better GPU locality"). Not random. Modelled
faithfully, a table after 2,000 requests of churn is still only 10 contiguous
runs, and head-major holds 94.4%.

| allocation model | head-major | verdict |
|---|---|---|
| `seq` cold pool | 96.9% | optimistic |
| **vLLM LIFO (real)** | **94.4%** | the right model |
| ONNXim in-range shuffle | 91.9% | conservative |
| uniform-random pool | 87.5% | **unrealistic — discard** |

An earlier version of the tool ranked under the uniform-random model, which
made the layouts tie and silently broke the ranking (3/8 vs 8/8).

## 10. DRAM energy, from measured activation counts

`E_ACT = 909 pJ`, `E_BIT = 3.9 pJ/bit` (HBM-class literature values; the
*ratios* are solid because ACT counts are measured, the absolute mJ inherit
the constants' error).

| configuration | ACTs | E_act | E_bit | total |
|---|---|---|---|---|
| weights row-major | 3,676,464 | 3.34 mJ | 17.25 mJ | 20.59 mJ |
| weights tile-major | 674,944 | 0.61 mJ | 17.25 mJ | **17.86 mJ (-13%)** |
| KV head-major | 9,308,192 | 8.46 mJ | 81.33 mJ | **89.79 mJ** |
| KV block-major | 18,900,528 | 17.18 mJ | 81.33 mJ | 98.51 mJ (+10%) |

Identical bytes moved; only the activation count differs. Layout controls
activation energy, which is ~16% of DRAM energy on the weight stream.

## 11. Silicon check of the mechanism

A microbenchmark on this machine's DDR5 (AMD EPYC 9374F), 384 MB working set,
identical bytes and access count, only the ORDER differing:

    row-major storage, tile-order traversal    15.8 GB/s
    tile-major storage, tile-order traversal   46.2 GB/s   2.92x

This validates the MECHANISM on real hardware, independent of ONNXim. It does
NOT validate the NPU magnitudes -- that needs an NPU. CPU prefetchers and a
256 MB LLC likely make 2.92x an understatement of what a bare accelerator sees.

## 12. Related work — where this sits

| | addresses | serving dynamics | layout comparison | validated |
|---|---|---|---|---|
| Duplex / LLMSimulator (MICRO'24) | yes | yes | **no** | cycle-accurate |
| ATLAS (3D-DRAM) | yes | no (fixed shapes) | **no** | **silicon** |
| LLMServingSim / ReaLLM / Vidur | no | yes | no | <14.7% vs GPU |
| RH+ | yes | — | no (scheduling, PIM) | — |
| this work | yes | yes | **yes** | ONNXim only |

ATLAS explicitly assumes the mismatched configuration -- *"matrices are laid
out row-wise, while tiles are accessed column-wise"* -- and reports no layout
comparison or row-hit improvement. Duplex has all the machinery (Ramulator2,
continuous batching, GQA/MQA/MLA, energy) and does not explore layout.

**CAUTION:** ATLAS and RH+ were read via HTML/metadata summarisers, never the
PDFs. Both are load-bearing for this table. Read them before citing.


## 13. Head-major's cost ledger -- every objection tested

Each of these was raised as a reason head-major could not work, and each was
checked rather than argued.

| objection | verdict |
|---|---|
| unrealizable under paging (head stride needs `nb`) | **wrong** -- ONNXim uses a POOL-relative stride, a startup constant |
| needs 8x the block-table metadata | **wrong** -- one entry per block, same as vLLM |
| over-allocates memory per head | **wrong** -- 32 regions tile the pool exactly (32 x 1.25 GB = 40 GB) |
| extra internal fragmentation | **wrong** -- regions are filled in lockstep, identical granularity |
| wider address span hurts TLB | **wrong** -- a fragmented block-major pool spans 39.89 GB anyway |
| breaks prefix sharing | **wrong** -- sharing is a block-TABLE operation, layout-blind |
| copy-on-write becomes 32 memcpys | **true, but 1.20x** (measured on DDR5: 63.7 vs 53.3 GB/s) |
| KV duplicated across cores | **wrong** -- traffic ratio 1.00x under MQA; the cache is SPLIT |

The mechanism, stated once: a DRAM row holds 4 heads' worth of a block-major
slice, so reading one head consumes 8 of its 32 request-slots. Head-major's
region contains only that head, so all 32 are useful.

## 14. Novelty -- checked, and narrower than it first appeared

**The idea is not new.** KVNAND (arXiv 2512.03608) states the principle
directly -- KV entries mixing heads within a page hurts locality, so group by
head -- but targets **NAND flash** at flash-page granularity, does not compare
against vLLM's layout, and reports no speedup from the layout change itself.
Head-wise organisation is an established theme (HeadInfer offloads head-wise;
AdaKV and HeadKV compress head-adaptively).

**No prior work found doing the DRAM version.** Kelle works on eDRAM bank
assignment; RH+ does row-hit *scheduling* on PIM; ATLAS fixes one layout and
does not compare; Duplex has all the machinery and does not explore layout.

So the defensible claim is a MEASUREMENT contribution:

> Head-grouped KV layout is known to help locality in flash. We show it holds
> for DRAM row buffers on a systolic NPU and quantify it: 1.22-2.03x across
> six workloads including real Azure traces, surviving GQA and XOR hashing.

Two things no prior work appears to do: the **mapper ablation** (XOR halves
the gap, 2.03x -> 1.24x) and the **negative controller result** (nine
scheduling mechanisms worth 2-4 points against layout's 48).

**CAUTION:** ATLAS, RH+ and KVNAND were all assessed through HTML/metadata
summarisers, never the PDFs. All three are load-bearing here. Read them.

## 15. Parallelism -- what `head` actually means

`ONNXIM_PAR_STRATEGY=head` (the default, used in every run) rotates the core
once per **N tile** and keeps all of that N's M tiles together:

    head      split N (head groups), keep M (sequence) whole on one core
    seq       split M (sequence), so one head's tokens spread across cores
    request   the whole op for one request pinned to one core

It is not "one head per core" -- `tile_out_loop.N` exceeds the head count, so
MQA (1 KV head) still filled all four cores. The name is loose.

Head-parallel wins by **3.3x** because head count is the only quantity that is
constant across requests. Measured on `v8m1024` (contexts 274-2600):

| strategy | per-core work | imbalance |
|---|---|---|
| head | [64, 64, 64, 64] | **1.0x** |
| request | [1453, 1091, 4901, 740] | 6.6x |
| seq | [92, 69, 307, 47] | 6.5x |

It is not unconditional: it needs `num_heads >= num_cores` (starves past 32
cores here), and it is coupled to the layout -- head-major's benefit assumes
sequence-preserving assignment. `seq`-parallel + head-major was never measured.

---

## 9. Speculative decoding, and three stock-ONNXim address bugs (2026-09-02)

Speculative decoding (draft x k, verify with k+1 query tokens, acceptance,
KV rollback) is now a closed loop in ONNXim (`scheduler: "specdec"`) and an
open-loop generator (`scripts/specdec_trace.py`). See SPECDEC.md.

Building it exposed three **upstream** ONNXim address bugs that every earlier
run in this file inherited (the local edits never touched these lines):

| bug | effect on earlier runs | fixed in |
|---|---|---|
| `kv_head_idx = head_idx / _nkvh` | every attention head read KV head 0: the KV footprint was 1/32 of the real one, all cores hammering the same 256 KB per request instead of 32 separate regions | `Attention.cc` |
| tile-local `seq_idx` in the tiled address | contexts > 2730 tokens (4 MB spad) re-read rows [0, chunk) per tile: `v8m4096`, `v32m4096`, `az128`, `wl33` never touched their upper context | `Attention.cc` |
| KV-concat tiles are `skip` | the KV cache was never written; the prefill "write traffic" was activations only. Executed with `ONNXIM_KV_WRITES=1` (writes also mis-placed and the tile loop unbounded: fixed) | `KVCacheConcat.cc` |

Address COUNTS were unaffected, which is exactly what the earlier validation
checked (256 x context per request) -- a count validation cannot see a
wrong head index. Row-hit and bandwidth numbers for the KV stream in sections
1-2 were measured on a 32x smaller working set and should be re-run before
being cited; the weight-stream results are untouched.

`out/specdec/base_hm16_v8m1024.log` re-runs `hm16_v8m1024` with the fixes
(single-token decode, contexts <= 2730, so only the head-index fix applies).

### The head-index fix, measured: KV row hit 96.1% -> 66.7%

`hm16_v8m1024` (8 req, ~1k ctx, head-major, block 16, 16 cores), identical
config, before and after the fixes:

| | old (all heads read KV head 0) | fixed |
|---|---|---|
| weights: hit / miss / confl | 71.0% / 18.2% / 10.7% | 71.0% / 18.4% / 10.6% |
| kv+act: hit / miss / confl | **96.1%** / 2.5% / 1.4% | **66.7%** / 20.2% / 13.1% |
| DRAM cycles | 2,626,483 | 2,623,736 |
| request cycles | 819,384 | 818,528 |

The weight stream and the runtime are untouched (this workload is
weight-bound), but the "KV is already at the physical ceiling, no headroom"
conclusion of section 1 was an artefact: 16 cores were re-reading the SAME
256 KB per request, so the controller saw one hot region. With each core
walking its own head's region the KV stream has the same 30-point headroom
the weight stream had before the swizzle. The head-major vs block-major
ranking must be re-measured; the mechanism that made block-major bad
(4 KB fragments 128 KB apart) is unchanged, so the direction is probably the
same, but the margins in section 1 are not to be cited.

## 16. Post-fix layout study, bank-isolated KV, and two measurement pitfalls (2026-09-09/10)

Everything in this section is measured on the fixed binary (section 9 fixes plus the
fifth bug below), `_c128.json` (4 x 128x128, HBM3 16ch 819.2 GB/s) unless noted, with
`ONNXIM_ICNT_PORT_BY_CHANNEL=1` where marked (pbc). Logs: `out/blocksweep2/`.

### A fifth upstream bug: scratchpad bank rotation races in-flight tiles

`Core::issue()` chose the accumulate-scratchpad bank by comparing against the LAST
tile issued (`_current_layer_id/_current_fused_op_id`). When another op's tile
interleaved between two accumulating tiles of one op, the second rotated to a fresh
bank and flushed it while its partner's data sat in the old one; the tile retired
against an empty bank and `Sram::fill()` threw robin_hood "key not found" (the guards
are `assert`s, compiled out in release). Only reachable when memory stalls reorder
tile issue -- so it fired under block-major KV and at 1 core, and block-major could
not complete on any workload until it was fixed. Fix (`Core.cc`): reuse the bank of
any LIVE tile from the same fused op; never flush a bank a live tile references.
Behaviour-neutral: head-major is bit-identical before/after. `Sram.cc` now reports
the missing address, bank occupancy and which bank actually holds it instead of
throwing.

### The section-1 layout claim, re-measured

`az128`, 128 real Azure requests, one decode step, KV = 90% of DRAM traffic (pbc,
operand-split counters):

| layout | weights hit | KV hit | KV confl | ACT | dram cycles | avg BW | while-busy |
|---|---|---|---|---|---|---|---|
| head-major | 74.1% | 70.0% | 7.9% | 2.31M | 17.68M | 86.3% | 98.5% |
| block-major b16 (vLLM) | 74.0% | 79.4% | 12.0% | 1.70M | 24.13M | 63.3% | 68.7% |
| headbank K=1 | 74.0% | 91.9% | 4.9% | 0.90M | 23.21M | 65.8% | 71.5% |
| headbank K=2 | 73.9% | 83.2% | 4.9% | 1.46M | 18.88M | 80.9% | 90.2% |

Head-major is 1.36x faster than vLLM's block-major at the default block size -- but
with the LOWEST KV row hit in the table and MORE activations. It wins on conflicts and
while-busy bandwidth, not locality. The section-1 sentences "head-major has better row
locality" and "fewer activations" are wrong and are superseded by this table. The
pre-fix 1.98x and every KV row-hit number in sections 1, 8 and 10 were measured with
32 heads reading KV head 0 and are not to be cited.

### One axis explains the whole layout space

Block-major with block size B keeps B x 256 B of one head contiguous; head-major keeps
the whole context. Row hit peaks where that slice is exactly one DRAM row per channel
(16 KB, B=64); runtime improves monotonically toward head-major:

| per-head slice | layout | KV hit | ACT | dram cycles | while-busy |
|---|---|---|---|---|---|
| 4 KB | block-major b16 | 74% | 1.98M | 25.3M | 66% |
| 16 KB | b64 == headbank K=1 | 86-88% | 1.0-1.2M | 23.1M | 72% |
| 32 KB | b128 == headbank K=2 | 81% | 1.5M | 18.9-19.1M | 90% |
| 64 KB | b256 | 72% | 2.1M | 17.8M | 97% |
| whole context | head-major | 68% | 2.4M | 17.9M | 98% |

(stock interconnect; address for address, block-64 IS the head-to-bank layout below.)
vLLM's default sits at the worst point. Block size is a runtime flag, so the practical
form of the finding is "use 128-256-token blocks", not "adopt a new layout". For
head-major itself block size is inert (b16..b256: +2.5 pts, `hm_b*`).

### Bank-isolated KV: `ONNXIM_KV_LAYOUT=headbank`

Under RoBaRaCoCh with this geometry the bank field (pch/bg/bank) is global address
bits [14:19): a 16 KB chunk is one row of one bank in every channel, 32 chunks span
the 32 banks. Headbank interleaves each head's stream at 16 KB so head h owns bank
slot (h*K + c%K + tile*stride) mod 32 -- `Attention.cc::kv_headbank_offset`. Knobs:
`ONNXIM_KV_BANKS_PER_HEAD` (K), `_V_BANK_OFFSET`, `_TILE_BANK_STRIDE` (keyed on
tile->M), `_REQ_ROTATE`, `_BANKCHUNK`/`_NBANKS` (re-derive for other geometries),
`ONNXIM_ACT_LANE`. The last four measured neutral and are kept only as negatives.

K is a one-dimensional dial between isolation and streaming (8 req, 4 cores, pbc):

| | KV hit | KV confl | ACT | cycles vs head-major | while-busy |
|---|---|---|---|---|---|
| head-major | 70.4% | 7.3% | 312k | -- | 99.4% |
| K=1 | 91.7% | 4.4% | 249k | +5.3% | 93.2% |
| K=2 | 82.6% | 4.4% | 276k | +1.5% | 96.8% |
| K=4 | ~ head-major | | | +0.3% | |

Why K=1 costs bandwidth: one row buffer per bank, so a bank-confined stream pays
tRCD + 64 data + tRTP + tRP (~117 cycles per 64-cycle row, ~55% duty) and nothing but
another bank streaming can hide it. Proven, not inferred: a scheduler tier that issues
PRE/ACT ahead of row hits (`RAMULATOR_SCHED_BANKPREP=1`) cut the measured PRE->ACT
gap 62 -> 28 cycles at 1 core with a 127-cycle runtime change out of 7.4M, and was
+3% WORSE at 4 cores. At 1 core the two layouts are equivalent (K=1 +2.4% at 92% hit).

### The interconnect was shuffling requests

Stock `Simulator.cc` fans a core's single request FIFO across all 16 injection ports
(request i -> port i mod 16) and `SimpleInterconnect` drains a channel's inputs
round-robin over ports, so a core's requests reach one channel locally reordered by up
to 16 x port backlog. Harmless for a bank-hopping stream; a bank-confined stream gets
rows r and r+1 interleaved and the controller ping-pongs them.
`ONNXIM_ICNT_PORT_BY_CHANNEL=1` injects on the port that serves the request's channel
(one FIFO per (core, channel), what a crossbar gives a single flow anyway; the response
path already worked this way). Conflicts fell 35-45% on every layout; block-major
+3-5%, K=1 +2.7%, head-major unchanged. Every number in this section marked pbc uses it.

### Two measurement pitfalls that understated KV locality

1. The controller's "kv+act" class was an address threshold (`RAMULATOR_WEIGHT_LIMIT`),
   so activation traffic (Q, GEMM inputs, outputs; ~37% row hit) was averaged into
   "KV". Requests now carry `Instruction::operand_id` (MemoryAccess -> mem_fetch ->
   Ramulator `Request::source_id`, unused before) and the controller prints three
   ROWSPLIT lines: weights / KV (operand 101, 102) / act. K=1's KV hit is 91.7%, not
   the 87.5% the mixed class reported.
2. Trace scripts must use the FULL `RAMULATOR_WEIGHT_LIMIT` (500,000,000): the CSV's
   `address` column is the global address; only the controller sees the /16 compacted
   form. Applying the compacted threshold to the CSV classes 97% of the weight stream
   as KV -- several hours of "KV stream" trace analysis on 2026-09-10 were the weight
   stream, and three layout knobs were built against it. The trace now carries an
   `operand` column; use it.

### Negative results, all measured (do not re-run)

RoCoBaCh mapper (bank bits below column: tFAW-bound, 2x slower on both layouts);
block-major b16 is itself tFAW-bound (3.8 reads/ACT x 4 ACT per 49 cycles = 63% of
the bus, measured 66%); `tile_depth` 6 (+-1%); request-parallel (KV hit +10 pts,
busy 59%, 1.6x slower); V-bank shift, activation lane, request rotation, M-tile
stride (all neutral once ordering was fixed); bank-prep scheduling (above); weight
tile-bank isolation (`ONNXIM_WEIGHT_TILEBANK=4`: weights 73.9 -> 73.1%, conflicts
4.6 -> 3.5%, cycles -0.2%). The weight stream's 4-core "loss" (96% at 1 core, 74% at
4) is 21.5% MISSES with only 4.6% conflicts, i.e. requests arriving during their own
stream's PRE->ACT window, not cross-stream eviction; head-major already serves a full
queue at 99.4%, so there is no bandwidth there to recover.

### Also fixed

`dram_nbl` is now 2 in every HBM3 config and is passed to the Ramulator wrapper
(was hard-coded 1): ONNXim's own printed BW utilization was 2x low (capped at ~50%).
The FINDINGS numbers were never affected (computed against 819.2 GB/s directly).

## 17. LPDDR6 in Ramulator2, and the HBM3 comparison harness (2026-09-10)

**Device model** `extern/ramulator2/src/dram/impl/LPDDR6.cpp`, derived from the LPDDR5X model.
One Ramulator channel = one JESD209-6 sub-channel: 12 DQ, 4 bank groups x 4 banks, own CA
(JEDEC "LPDDR6 Key Architecture" deck). Burst BL24 x 12 DQ = 288 bits = 32 B data + 4 B
S-ECC/meta, so a 32 B ONNXim request is one burst; the template applies a 2-bursts-per-4xnBL32
interleave window, so `nBL32: 6` gives 12 CK per RD. Presets `LPDDR6_6Gb_x12` / `_12Gb_x12`.
**Core timings are not public**: `configs/ramulator2_configs/_lpddr6_{10667,14400}.yaml` carry
LPDDR5X-class values in ns (tRCD 18, tRP 18/21, tRAS 42, tRC 60, tCL 24, tCWL 12, tWR 34,
tRTP 7.5, tRRD 3.75, tFAW 15) re-expressed in cycles at tCK = 1E6/(rate/2) ps -- the
HBM3_6.4Gbps.yaml method; fine for relative comparison, not citable as absolute LPDDR6 latency.
ONNXim configs `_lp6_48ch_{rate}.json`: 48 sub-channels (24 x 24-bit channels = 576-bit, the
Apple M4-Max-class bus), `dram_channel_width 12`, `dram_prefetch_size 24`, `dram_nbl 12`,
nominal payload peak 683 / 922 GB/s (10667 / 14400), bracketing one HBM3 stack (819).

**Three ONNXim/Ramulator bugs hit on the way** (all fixed): `ipoly_hash_function` only
implements 16/32/64 channel sets and `exit(1)`s silently for 48 -- every LPDDR6 run died at
`QKVgen` with a clean log (channel id is now a plain block interleave for other counts); the
trace decoder truncated `log2(2048)-log2(24)` as one float (now floored per term, as Ramulator
does); the `ReadWriteTrace` front-end mapped 'W' to Read, looped its trace forever and its
address vectors were overwritten by the mapper (use `LoadStoreTrace` with flat addresses). The
controller's command counters matched "RD"/"ACT" exactly and reported 0 for LPDDR's RD32/ACT-2
(prefix-matched now; LPDDR6 runs before 2026-09-10 18:00 have ACT = misses + conflicts instead).

**Validation gates** (standalone `bin/ramulator2`, `out/lp6val/`): unloaded read latency
HBM3 51 cycles = 15.9 ns (= tRCD + tCL + nBL + 1 exactly), LPDDR6 45.9 / 45.0 ns at 10667 /
14400 (same ns, as intended: 2.9x HBM3). Sequential streaming: HBM3 2.00 cycles/RD = 51.2 GB/s
per channel (nominal), LPDDR6 13.1 cycles/RD = 13.0 / 17.6 GB/s per sub-channel (92% of nominal
-- the WCK CAS-sync command). Effective 48-sub-channel peaks: 625 and 845 GB/s.

**Microbenchmark ladder, `llama_dec1` (1 request, 1023 ctx, weight-bound), head-major, pbc:**

| | HBM3 819 GB/s | LPDDR6-10667 | LPDDR6-14400 |
|---|---|---|---|
| 1 core, core cycles | 1,655,575 | 1,657,147 | 1,656,580 |
| 1 core, BW (avg / while-busy) | 31.2% / 94.6% | 35.4% / 82.8% | 26.1% / 80.3% |
| 1 core, mean read latency | 178 ns | 323 ns | 228 ns |
| 4 cores, core cycles | 532,739 (96.9% BW) | pending | pending |

At 1 core the three memories are identical to 0.1% -- the array shift-in floor binds
(section 15 / [[llama-decode-two-floors]]), and neither bandwidth nor a 2.9x latency difference
reaches the runtime. That is the floor model transferring across a DRAM technology, as
predicted. The 4-core cells (memory-floor-bound: predicted +20% at 682 GB/s, -11% at 922) and
`az128` on both rates (head-major and K=2) are running; `scripts/dram_energy.py` turns the
finished logs' command mix into a core/IO/refresh energy split with the LPDDR6 IO break-even.

### §16 addendum (2026-09-10 evening): XOR hashing equalises the layouts; 16 cores break K=1

`az128`, 4 x 128x128, pbc, operand-split. XOR = `RoBaRaCoChXOR` (row bits folded into the bank field):

| layout | mapper | KV hit | KV confl | ACT/ch | cycles | while-busy |
|---|---|---|---|---|---|---|
| head-major | RoBaRaCoCh | 70.0% | 7.9% | 2.31M | 17.68M | 98.5% |
| head-major | XOR | 70.1% | 5.9% | 2.30M | 17.66M | 98.4% |
| block-major b16 | RoBaRaCoCh | 79.4% | 12.0% | 1.70M | 24.13M | 68.7% |
| block-major b16 | XOR | 67.8% | 11.6% | 2.46M | **17.93M** | 95.6% |
| headbank K=2 | RoBaRaCoCh | 83.2% | 4.9% | 1.46M | 18.88M | 90.2% |
| headbank K=2 | XOR | 69.9% | 5.5% | 2.32M | 17.67M | 97.4% |

**Under XOR bank hashing all three layouts converge to ~17.7M cycles and ~70% KV hit.** Head-major's
1.36x over vLLM's block-major exists only when the bank is chosen by low address bits; a hashed
controller gives block-major the same bank parallelism (68.7% -> 95.6% while-busy) at the price of
its locality (79.4% -> 67.8%). XOR also undoes headbank: folding row bits into the bank field
scrambles the head->bank assignment, so K=2 becomes head-major-like. The defensible runtime claim is
therefore "1.36x under RoBaRaCoCh, ~1.01x under XOR hashing"; the claim that survives any mapper is
the locality/energy one -- headbank K=1 at 92% KV hit and 61% fewer activations is unreachable by
hashing (which tops out near 70%). The pre-fix section-8 figures (2.03x / 1.24x) are superseded.

Consistency re-runs (pbc): K1 23.21M -> K1+V16 21.95M (89.0% hit); K2+V16 18.33M; **K2+V16+b64
18.12M at 82.2% KV hit, 3.2% conflicts, 94.4% while-busy** -- the best "both goals" point: 2.5%
slower than head-major with +12 points of hit and 34% fewer activations.

16 cores x 32x32 (32 GB config): head 24.62M / block 23.94M / K1 24.45M / K2 25.29M cycles -- all
1.35-1.43x slower than 4 x 128x128 (day-1 result confirmed) and layout-indifferent. K=1's KV hit
drops to 66.3% there: 16 cores x 3 tiles = 48 concurrent head streams over 32 banks, so heads must
share banks and the isolation premise fails. Bank isolation needs concurrent streams <= banks.

Speculative decoding on the fixed binary (`sd_v4m512`, 16x32, k=4, 2 verify steps, 15/20 draft tokens
accepted, 21 tokens): head-major 2,694,329 cycles vs K=2 2,711,547 (+0.6%); KV conflicts 9.6% -> 3.7%.
KV is 7% of this workload's traffic, so layout is a second-order effect for it.

### §17 addendum: LPDDR6 4-core ladder and energy

| `llama_dec1`, 4 cores | HBM3 819 | LPDDR6-10667 (682 nom / 625 eff) | LPDDR6-14400 (922 / 845) |
|---|---|---|---|
| core cycles | 532,739 | 757,698 (+42%) | 565,583 (+6%) |
| delivered BW | 794 GB/s = 96.9% nom | 528 GB/s = 77.4% nom / 85% eff | 705 GB/s = 76.4% / 83% |
| while-busy | 99.5% | 85.8% | 85.4% |
| weights hit / ACT per ch | 74.9% / 198k | 92.3% / 20k | 94.2% / 15k |
| mean read latency (loaded) | 1.03 us | 1.56 us | 1.14 us |

The floor model predicted +20% / -11% from nominal bandwidth; measured +42% / +6% because the
LPDDR5X-derived timing model serves a full queue at only ~86% (its 2-burst interleave window and the
WCK CAS-sync command), against HBM3's 99.5%. Report LPDDR6 numbers against effective peak; the
10-15% model pessimism is a property of the template, not of JESD209-6, and should be revisited
against a vendor timing set. Row locality is much better on LPDDR6 (48 sub-channels x 3 KB rows,
92-94% hit, 10x fewer activations per channel).

Energy (literature-class constants, ratios use measured ACT/bytes/time; `scripts/dram_energy.py`):
HBM3 16.9 mJ (5.0 pJ/bit) = act 3.1 + core 4.7 + IO 8.5 + ref 0.6; LPDDR6-10667 19.8 mJ (5.8 pJ/bit)
= act 1.1 + core 4.1 + IO 13.5 + ref 0.9 + bg 0.2. LPDDR6 does 2.2-3x fewer activations, so the whole
comparison rides on the IO constant: **LPDDR6 wins on energy iff its interface energy is below
3.2 pJ/bit** (HBM3 assumed 2.5, LPDDR6 assumed 4.0). That inequality is the citable result; the 1.17x
at the assumed constants is not.
