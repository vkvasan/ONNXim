# HBM3 + DRAM access tracing on ONNXim

Local additions for comparing prefill vs decode DRAM access patterns.

## What was added

| Path | Purpose |
|---|---|
| `configs/ramulator2_configs/HBM3.yaml` | HBM3 using Ramulator2's stock `HBM3_2Gbps` preset |
| `configs/ramulator2_configs/HBM3_6.4Gbps.yaml` | HBM3 at a realistic 6.4 Gbps data rate |
| `configs/systolic_ws_128x128_c4_hbm3_spad{8,16,24}mb.json` | tpuv4 topology, HBM3 6.4 Gbps, scratchpad swept 8/16/24 MB per core |
| `src/Dram.cc`, `src/Dram.h` | the access-trace hook in `DramRamulator2` |
| `src/SimulationConfig.h`, `src/Common.cc` | DRAM organization fields used to decode the trace |
| `scripts/devshell.sh`, `scripts/build_in_docker.sh` | build/run this checkout inside the `onnxim` image |
| `scripts/analyze_dram_trace.py` | summarize and compare traces |
| `traces/prefill_1k.csv`, `traces/decode_1k.csv` | request traces that isolate each phase |
| `example/prefill_1k.json`, `example/decode_1k.json` | matching model lists |

Ramulator2 itself is **unmodified** — it ships as the `extern/ramulator2` submodule
and needs no separate install.

## Build

The upstream Dockerfile targets Ubuntu 20.04 / gcc-10 / conan 1.57, none of which
match this host (Ubuntu 24.04, gcc 13.3, Python 3.12), so everything runs in the
container. `scripts/devshell.sh` bind-mounts this checkout over
`/workspace/ONNXim`, so host edits are what gets compiled.

```bash
docker build . -t onnxim              # once
./scripts/devshell.sh ./scripts/build_in_docker.sh
```

The build directory must be named `build/` — `CMakeLists.txt:6` hardcodes
`include("${CMAKE_SOURCE_DIR}/build/conanbuildinfo.cmake")`.

Container processes run as root, so build outputs are root-owned on the host.

## Running with tracing

Tracing is off unless `ONNXIM_DRAM_TRACE` names an output file.

```bash
./scripts/devshell.sh bash -c '
  cd /workspace/ONNXim/build && \
  ONNXIM_DRAM_TRACE=/workspace/ONNXim/out/prefill.csv \
  ONNXIM_DRAM_TRACE_LIMIT=20000000 \
  ./bin/Simulator \
    --config ../configs/systolic_ws_128x128_c4_hbm3_spad16mb.json \
    --models_list ../example/prefill_1k.json \
    --mode language \
    --trace_file prefill_1k.csv'
```

`--trace_file` is **required**. `main.cc:100-102` overwrites the `trace_file`
field from the models-list JSON with the command-line value (default
`input.csv`), so the field documented in the upstream README is ignored. Two runs
that differ only in that JSON field will silently produce identical results.

`ONNXIM_DRAM_TRACE_LIMIT` caps the row count (0 or unset = unlimited). It matters:
a 256-step decode run of opt-125m emitted 142M rows / 4.8 GB and took ~20 minutes.

## Trace format

```
cycle,channel,pseudochannel,bankgroup,bank,row,column,address,rw
10,0,1,1,2,44,2,0x164c560,R
```

- `cycle` — DRAM-domain cycle at request issue. `DramRamulator2` did not
  previously maintain the base-class `_cycles` counter; the hook initializes and
  advances it.
- `channel` — the ONNXim-side channel. ONNXim instantiates one single-channel
  Ramulator2 per channel, so Ramulator2's own channel index is always 0 and
  carries no information.
- `pseudochannel,bankgroup,bank,row,column` — decoded with the same bit slicing
  Ramulator2's `RoBaRaCoCh` mapper applies internally.
- `address` — the original untranslated `MemoryAccess::dram_address`.
- `rw` — `R` or `W`.

### Decode correctness

The decode was validated against ground truth, not just reasoned about:
`RoBaRaCoCh::apply()` was temporarily instrumented to dump its own `addr_vec`,
and both outputs were compared over a full ResNet-18 run — 13,267 unique
addresses and all 294,114 emitted rows matched exactly. The instrumentation has
been reverted.

That check caught a real bug. Ramulator2's transaction offset uses
`org.channel_width`, which **defaults to 64**, not `org.dq` (128). The transaction
size is therefore 16 B, not 32 B, and an assumed 128 shifted every decoded field.
Hence `dram_channel_width: 64` in the configs.

If you change the DRAM organization or the `AddrMapper`, the decode fields in the
JSON config must be updated to match — nothing cross-checks them at runtime.
Only `RoBaRaCoCh` (the mapper all shipped configs use) is replicated.

## HBM3 caveat — read before citing numbers

Ramulator2 ships exactly one HBM3 timing preset, `HBM3_2Gbps`, and its own source
comment says `TODO: Find more sources on HBM3 timings...`. At 2.0 Gbps/pin it is
**slower** than the `HBM2_2.5Gbps` preset the stock tpuv4 config uses, and its
`nBL` is 4 versus HBM2's 2. Switching to `HBM3.yaml` alone therefore *reduces*
memory bandwidth — the opposite of the intended effect.

`HBM3_6.4Gbps.yaml` addresses this. Ramulator2 rejects a `rate` override combined
with a timing preset, so it specifies the full timing set explicitly: the
nanosecond values of the bundled preset held constant and re-expressed in cycles
at the 6.4 Gbps clock (same DRAM core, faster bus). `nBL` and `nCCDS` are held as
bus-cycle constants rather than scaled. Every value and its derivation is in the
file header.

These are frequency-scaled figures, not vendor-validated JEDEC HBM3 timings. They
are self-consistent and fine for relative comparisons; do not cite them as
absolute HBM3 latencies.

`dram_freq` in the JSON config must track the DRAM clock: **3200** for
`HBM3_6.4Gbps.yaml`, **1000** for `HBM3.yaml`, 1200 for the stock HBM2 config.

## Analysis

```bash
python3 scripts/analyze_dram_trace.py out/prefill.csv out/decode.csv
```

The reported row-hit rate is the locality present in the *request stream*, in
arrival order per bank. It will not equal Ramulator2's own "Row hits/misses/
conflicts" counters, because FR-FCFS reorders requests. Use this to characterize
the access pattern and Ramulator2's counters to characterize what the controller
achieved.

### Measured baseline

opt-125m, 16 MB scratchpad, HBM3 6.4 Gbps, 16 channels:

| metric | prefill (1024 tok) | decode (256 steps) |
|---|---:|---:|
| DRAM requests | 3,491,424 | 142,491,552 |
| read fraction | 0.761 | 0.999 |
| requests / 1k cycles | 3,148.9 | 3,139.3 |
| row hit rate (arrival order) | 0.779 | 0.855 |
| distinct rows touched | 83,600 | 44,384 |
| accesses per row | 41.8 | 3,210.4 |
| row reuse (median) | 32 | 4,096 |
| busiest/mean channel | 1.00 | 1.00 |

The split is what the phases predict: prefill carries real write traffic (24%,
KV-cache fills and activation spills) and streams broadly across rows, while
decode is essentially pure reads (99.9%) hammering ~half as many rows two orders
of magnitude harder. The `ipoly` channel hash keeps both phases perfectly
balanced across all 16 channels, so channel skew is not a variable here.

The dominant column stride is 2 in both phases, not 1 — a consequence of 32 B
requests on a 16 B transaction granularity, not of the access pattern.

### Why decode sits at 39% bandwidth

Decode *is* memory-bound; the average hides it. Binning the trace by DRAM cycle
(batch=1, 32 steps, 16 MB spad):

| | |
|---|---:|
| cycles with zero requests arriving | 78.5% |
| cycles with >= 8 requests arriving | 20.6% |
| mean requests/cycle **when active** | 14.56 |
| DRAM sustainable requests/cycle | 8 (32 buses / nBL=4) |

During its bursts the workload demands **1.8x more than the DRAM can deliver**,
but it can only present that demand 21.5% of the time. 1.82 x 0.215 = 39.1%,
exactly the measured figure. The memory system is saturated whenever it has work
and idle the rest of the time — a duty-cycle problem, not a capacity one.

The gaps are systolic-array preload. At batch=1 every GEMM is a `GEMM_PRELOAD`
(5082 of 5082), and `SystolicWS.cc:50` charges each one
`MAX(compute_size, core_height)` = at least 128 cycles to shift a 128x128 weight
tile in, for 0.19% PE utilization. That is 650,496 cycles, 37% of runtime, during
which no new tile fetch is issued.

Probes that rule out the alternatives, all on the same 32-step decode:

| change | result |
|---|---|
| spad 8 -> 24 MB | request counts byte-identical (1,090,440 R / 1,541 W) |
| DRAM latency collapsed (tCL/tRCD/tRP -> ~1, same nBL) | 1.6% faster; BW still 19% |
| 2x channels (2x peak BW) | 1.41x faster, BW/channel falls to 13% |
| batch 1 -> 8 | 2.86x better per sequence; BW *falls* to 16% |

Neither scratchpad capacity, DRAM latency, nor DRAM bandwidth is the binding
constraint. What would raise utilization is overlapping the next tile's fetch
with the current tile's preload — deeper prefetch, not a faster memory.

The run-ahead depth is fixed in the model. `Core::can_issue()` is
`return _tiles.size() < 2;` (`Core.cc:33`), and `Core::issue()` assigns each tile
a scratchpad bank with `spad_id = (spad_id + 1) % 2` followed by
`_spad.flush(spad_id)` (`Core.cc:52-53`), which clears that bank's contents. So
the scratchpad is modeled as exactly **two flushable banks** — one computing, one
loading — regardless of `spad_size`. A third tile cannot be prefetched because
there is no third buffer, and nothing survives across tiles.

That single fact explains both anomalies: it is why decode re-streams the full
weight set every step (the bank is flushed, so no weight is ever reused across
tiles) and why `spad_size` changes nothing (capacity bounds how big *one* tile
may be, not how many may be in flight).

### Cycle accounting (added instrumentation)

`Core::print_stats()` now emits an exact partition of every core cycle plus a
memory round-trip latency histogram, because the pre-existing counters
(systolic active, memory idle, core idle) overlap and leave most cycles unnamed.
Decode, batch=1, 16 MB spad, core 0:

```
CYCLES inst-issued 7,682 (0.4%) | stall-no-issuable-inst 1,564,099 (89.8%)
       | no-tile-resident 170,531 (9.8%)   [sums exactly to 1,742,312]
tile slot free (<2 resident) 1,114,611 (64.0%)
MEM round-trip latency avg 4,106.8 max 11,673 over 4,267,880 accesses
MEM latency hist <256:2.0% <512:4.3% <1024:8.4% <2048:17.0% >=2048:67.1%
```

The core spends **89.8% of cycles unable to issue any instruction**, waiting for
tile data, and issues on only 0.4% of cycles.

The mechanism is self-inflicted queueing. `Core::cycle()` moves at most one
instruction per cycle into `_ld_inst_queue`, but `handle_ld_inst_queue()`
(`Core.cc:343-356`) then pushes **every** `src_addr` of that MOVIN into the
request queue in a single cycle — averaging **556 requests per issue cycle**. The
DRAM absorbs 8 per DRAM cycle (2.5 per core cycle), so one MOVIN alone takes
~220 core cycles to drain. Queues build to thousands of entries: average round
trip is **4,107 core cycles against a ~14-cycle device latency, so 99.6% of the
observed latency is queueing**, and 67% of accesses exceed 2048 cycles.

`can_issue_compute()` (`Core.cc:199-208`) requires *every* source address to be
resident, so compute waits on the slowest request in the burst. The result is a
convoy: dump a burst, saturate DRAM at 1.82x while the queue drains, stall on the
tail, go idle. Average DRAM utilization 39%.

Note also `tile slot free 64.0%` — over half the run the core holds only one
tile, so the two-deep double buffer is not even full and no load/compute overlap
occurs. That is an upstream tile-supply problem, separate from the queueing.

An earlier hypothesis blamed the 128-cycle `GEMM_PRELOAD` floor. Removing that
floor entirely (probe, reverted) cut array occupancy from 43.3% to 20.7% but
bought only 14% runtime and moved bandwidth 19% -> 22%. The array is not the
constraint.

### Where the 19% actually comes from

`GenericDRAMController::tick()` is now instrumented (see "Submodule changes"
below). It issues **at most one command per cycle per channel**, and every cycle
falls into exactly one bucket:

| controller cycle | share |
|---|---:|
| empty — core had nothing queued | 33.3% |
| blocked — work queued, nothing timing-eligible | 42.6% |
| issued ACT / PRE / WR (command overhead) | 4.6% |
| **issued RD — this is the bandwidth** | **19.5%** |
| total | 100.0% |

1,088,351 RD commands over 5,584,331 ticks is 19.5%, which *is* the "19% BW
utilization" ONNXim prints. The metric is simply the RD-command rate, and the
table above accounts for every remaining cycle.

Most of that 42.6% "blocked" is **not** DRAM timing. Fixing the feed path --
`icnt_freq` 1000 -> 3200 *and* `RAMULATOR_REQBUF` 64 -> 1024 together, which had
only ever been tested separately -- collapses it:

| controller cycle | stock | feed fixed |
|---|---:|---:|
| empty (no work queued) | 33.3% | **46.2%** |
| blocked (nothing eligible) | 42.6% | **15.0%** |
| issued | 24.1% | 38.7% |
| runtime (core cycles) | 1,742,312 | 1,347,193 |
| BW utilization | 19% | 25% |

23% faster, and blocked falls by two thirds. The DRAM was never timing-limited;
it was being fed too slowly and too thinly for the scheduler to find eligible
work. With the feed fixed, the dominant bucket is **empty** -- the DRAM has
nothing to do 46% of cycles because the core has not issued anything.

The time is lost in the core, not the memory. Neither fix helps much alone
because each exposes the other.

### Root cause: there is nothing to pipeline

ONNXim *does* double-buffer (`Core::can_issue()` allows 2 tiles), so load and
compute should overlap and no core/memory handoff should be visible. Measured:

```
[SCHED] core had a free tile slot on 4,207,146 core-cycles;
        scheduler had NO tile to give on 4,205,866 of them (100.0%)
[SCHED] barrier-blocked tile-issue attempts 0 | barriers crossed 0
```

**100%.** The second buffer sits empty not because the core refuses tiles, but
because the scheduler has none. Barriers (`Tile::Status::BAR`) are not involved
at all -- zero were crossed on this workload.

The whole 32-step decode run produces only **1,280 tiles** -- 40 per step across
4 cores, 10 per core per step, each ~417 KB and ~5,445 cycles. Batch-1 decode is
a chain of GEMVs with M=1, so each GEMM decomposes into very few tiles, and
consecutive operations (QKV -> attention -> O-proj -> MLP) are strictly
dependent. At any instant there is usually exactly one tile that *can* run.

So the handoff is not a flaw in the pipelining logic. Pipelining needs two
independent pieces of work, and batch-1 decode does not supply them.

### Tile granularity is the lever -- and it is `accum_spad_size`

Tile count is set by the accumulator, not the scratchpad (`Mapping.cc:75-79`):

```cpp
db_mats_in_acc  = max_acc_rows / dim;          // max_acc_rows from accum_spad_size
db_max_tile_i_j = sqrt(db_mats_in_acc);
tile_J = min(dim_J_padded/dim, ceil_div(dim_J, db_max_tile_i_j*dim));
```

`spad_size` only feeds `tile_K`, which clamps to 1 for these shapes -- which is
why sweeping `spad_size` 8/16/24 MB changed nothing at all. It was the wrong
parameter.

Measured (8-step decode, batch 1):

| accum_spad | tiles | DRAM empty | free tile slot | BW | cycles |
|---:|---:|---:|---:|---:|---:|
| 4096 KB (stock) | 320 | 33.0% | 1,050,421 | 19% | 433,025 |
| 8192 KB | 416 | 29.6% | 681,213 | 21% | 391,626 |
| 16384 KB | 432 | 29.8% | 774,550 | 21% | 393,802 |

More tiles -> less tile starvation -> higher bandwidth. Gains saturate by 16 MB.

Note the direction: a *larger* accumulator produced *more* tiles here, because the
attention operator's own tiling (`Attention.cc:685-722`) dominates the tile count
for decode and moves opposite to the GEMM heuristic.

Reducing `accum_spad_size` below 4096 KB crashes with a floating-point exception
in the attention tiling path. One cause is fixed here -- `q_len` could reach 0 and
divide by zero at `Attention.cc:701`, now guarded with `std::max(q_len, 1u)` --
but at least one more remains, so the finer-tile direction is still blocked.

### Full stack, including the bank-spread fix

`column: 16` / `row: 131072` + `accum_spad_size: 8192` + `icnt_freq: 3200` +
`RAMULATOR_REQBUF=1024`, on the 32-step decode:

Shipped as `configs/systolic_ws_128x128_c4_hbm3_tuned.json` +
`configs/ramulator2_configs/HBM3_6.4Gbps_bankspread.yaml`. Still needs
`RAMULATOR_REQBUF=1024` in the environment (there is no config field for it).

| | stock | tuned |
|---|---:|---:|
| Total cycle | 1,742,312 | **1,366,206** (-21.6%) |
| BW utilization | 19% | **24%** |
| blocked (bank serialization) | 42.8% | **18.0%** |
| empty (core starvation) | 33.3% | 38.0% |
| issued | 24.1% | 44.0% |
| array occupancy | 44.75% | 56.42% |
| DRAM reads | 1,090,368 | 1,090,976 |

Blocking more than halves. Traffic is unchanged (+0.06%), so this is purely
better overlap, not less work. Bank parallelism stops being the constraint and
`empty` -- the core failing to supply requests -- becomes the dominant loss at
38%. Past this point only the structural changes (tile depth > 2, partial-tile
compute start) matter.

### Best validated configuration

Stacking the levers that survived testing -- `accum_spad_size` 8192,
`icnt_freq` 3200, `RAMULATOR_REQBUF=1024`:

| | stock | stacked |
|---|---:|---:|
| Total cycle | 433,025 | **313,379** (-27.6%) |
| BW utilization | 19% | **27%** |
| free tile slot (starvation) | 1,050,421 | **581,639** (-45%) |
| array occupancy | 44.75% | 60.33% |

For memory-system studies specifically, also drop to 4 channels: that makes DRAM
the actual critical path (`empty` falls to 9.3%) so its timing and access
patterns are what you are measuring.

The core still cannot be saturated by configuration --
`stall-no-issuable-inst` stays ~90% across every batch size (1-64) and channel
count (4/8/16) tested. That needs the code changes listed above (tile depth > 2,
partial-tile compute start).

Individual interventions:

| intervention | gain |
|---|---|
| 2x channels | 41% |
| remove `GEMM_PRELOAD` floor | 14% |
| `nCCDL` 8 -> 4 | 7% |
| controller read buffer 64 -> 256 | 7% |
| `icnt_freq` 1000 -> 3200 | 5.5% |
| DRAM device latency -> ~0 | 1.6% |
| refresh disabled (`nREFI` x10) | 0% |
| scratchpad 8 -> 24 MB | 0% |

Read buffer beyond 256 and `nCCDL` below 4 both give nothing further.

### Per-layer breakdown: the time is lost in pipeline drains

One decode step (opt-125m, batch 1, 52,741 core cycles), correlating the access
trace against the `Layer ... finish at` log:

| layer | cycles | % step | MB moved | ideal cyc | efficiency | banks/64 |
|---|---:|---:|---:|---:|---:|---:|
| QKVgen | 9,501 | 18.0% | 3.56 | 4,340 | 46% | 7.0 |
| **Attention** | **12,851** | **24.4%** | 3.15 | 3,843 | **30%** | **3.3** |
| proj | 3,610 | 6.8% | 1.19 | 1,455 | 40% | 9.1 |
| fc1 | 12,191 | 23.1% | 4.74 | 5,784 | 47% | 3.6 |
| fc2 | 14,428 | 27.4% | 4.78 | 5,830 | 40% | 4.5 |
| **step** | **52,741** | 100% | 17.42 | 21,261 | **40%** | |

("ideal cyc" = bytes / 819.2 GB/s at the 1 GHz core clock.)

Attention is simultaneously the largest slice and the least efficient. The reason
is visible in the DRAM idle structure:

| layer | idle in gaps >= 200 cyc | largest single gap | efficiency |
|---|---:|---:|---:|
| fc1 | 11.8% | 2,261 | 47% |
| QKVgen | 24.6% | 2,858 | 46% |
| fc2 | 25.8% | 9,494 | 40% |
| proj | 34.1% | 2,320 | 40% |
| **Attention** | **51.8%** | **14,960** | **30%** |

Efficiency tracks gap fraction almost perfectly. Attention spends **over half its
duration with the DRAM completely idle**, including one gap of 14,960 DRAM cycles
(4,675 core cycles, 36% of the layer). fc1 -- a single uninterrupted GEMM -- has
the fewest gaps and the best efficiency.

The gaps are **not** barrier drains, and not softmax. Per-tile issue times
(`--log_level debug`, "Get Tile at") show the real mechanism:

| layer | tiles | issue waves (cycle:count) | duration |
|---|---:|---|---:|
| QKVgen | 5 | 0:5 | 9,501 |
| **Attention** | **12** | **9503:8  15593:4** | 12,851 |
| proj | 6 | 22356:6 | 3,610 |
| fc1 | 6 | 26009:6 | 12,191 |
| fc2 | 6 | 38265:6 | 14,428 |

The machine has **8 tile slots** (4 cores x 2-deep). Every layer except attention
has <= 6 tiles, so all of them are issued in a single wave, dump their loads, and
the DRAM works through one continuous burst.

Attention has **12 tiles** -- one per head -- which overflows the 8 slots. Eight
issue at cycle 9503, and the remaining four cannot start until slots free at
15593. Between those two waves the resident tiles have already fetched
everything they need, so the DRAM has nothing to do: a 6,089-cycle hole in the
middle of the layer. That is the 14,960 DRAM-cycle gap.

Softmax is not the cause: the vector unit is active for ~30 cycles in the entire
step (52,753 cycles), 156x smaller than the gap it sits inside.

Barriers are not the cause either. `Attention.cc:608/624/641` do insert
`Tile::Status::BAR`, but only inside `initialize_non_fused_tiles`, which
`initialize_tiles` calls only when `!use_fused && onnx`. The language path uses
the fused attention, which emits no barriers at all.

### Deeper tile pipeline: `tile_depth`

`tile_depth` (JSON field, or `ONNXIM_TILE_DEPTH`) sets the number of tile slots
and scratchpad banks. 2 = stock double buffering. Threaded through
`SimulationConfig`, `Sram` (banks are now a vector, each `spad_size/tile_depth`),
`Core::can_issue`/`issue`, `Mapping.cc`'s capacity heuristic, and the operators
that assumed a half-scratchpad tile.

Note each bank is `spad_size / tile_depth`, so deeper means *smaller* tiles --
the capacity is split, not added.

| depth | empty | blocked | BW | cycles | PE util |
|---:|---:|---:|---:|---:|---:|
| 2 (stock) | 33.0% | 42.9% | 19% | 433,025 | 0.21% |
| **3** | **22.3%** | 45.6% | **23%** | **354,369** | 0.24% |
| 4 | 22.3% | 45.6% | 23% | 354,369 | 0.24% |
| 6 | 23.7% | 42.6% | 23% | 360,652 | 0.25% |

Depth 3 cuts idle cycles by a third and runs 18.2% faster; depth 4 is identical
(saturated) and 6 is slightly worse. Crucially **PE utilisation is unchanged**
(0.21% -> 0.24%) -- a deeper pipeline gives the DRAM more independent work to
fetch, it does not make the array compute anything more. It addresses the
symptom of level 1, not its cause.

Two implementation notes, both bugs found while building it:
* `Core::issue()` derived the next bank from `_tiles[0]` only when exactly one
  tile was resident. That is correct for depth 2 but collides with a live tile's
  bank at any greater depth. The rotation is now an explicit member.
* Depth 2 reproduces stock **bit-identically** (403,625 cycles / 21% BW with
  `RAMULATOR_REQBUF=1024`), which is the regression check for the refactor.

### The two levers compose

| config | empty | blocked | BW | cycles |
|---|---:|---:|---:|---:|
| stock | 33.0% | 42.9% | 19% | 433,025 |
| tuned (bank spread + feed path) | 37.7% | **18.1%** | 25% | 339,962 |
| `tile_depth: 3` alone | **22.3%** | 45.6% | 23% | 354,369 |
| **tuned + `tile_depth: 3`** | 29.4% | 20.3% | **28%** | **299,728** |

**30.8% faster than stock, BW 19% -> 28%** -- which against the 50% bus ceiling
is 38% -> **56% of physical peak**. They compose because they attack different
levels: `tile_depth` reduces `empty` (level 1), bank spread reduces `blocked`
(level 2). Each alone converts one loss into the other; together both fall.

`configs/systolic_ws_128x128_c4_hbm3_tuned.json` now carries `tile_depth: 3`.

### SRAM sizing: where bandwidth saturates

20-point sweep on the tuned baseline, 8-step decode. Cycles (best = 296,824):

| spad \ accum | 2 MB | 4 MB | 8 MB | 16 MB |
|---|---:|---:|---:|---:|
| 2 MB | 325,335 | 380,099 | 407,569 | 414,230 |
| **4 MB** | **298,279** | 324,696 | 323,709 | 385,792 |
| 8 MB | 300,561 | 298,344 | **296,824** | 328,394 |
| 16 MB | 302,357 | 298,761 | 299,728 | 299,475 |
| 32 MB | 302,357 | 300,663 | 301,009 | 299,371 |

BW saturates at 28% across the flat region. **DRAM traffic is identical at all 20
points** (272,100 reads/channel, +-0.04%), so every difference is scheduling, not
capacity -- decode reads each weight exactly once regardless of SRAM size.

**Minimum that saturates: spad 4 MB + accum 2 MB.** Note a *larger* accumulator
is actively harmful (at spad 2 MB, accum 2 -> 16 MB costs 30%): `accum_spad_size`
drives the tiling heuristic, and a bigger accumulator yields fewer, larger tiles,
so there is less to pipeline.

### Per-layer bandwidth (tuned config: spad 4 MB / accum 2 MB, tile_depth 3)

DECODE, one step, 1023 cached:

| layer | cycles | % of block | avg of bus | DRAM busy | while busy |
|---|---:|---:|---:|---:|---:|
| attn.QKVgen | 7,301 | 19.9% | 60% | 72.3% | 82% |
| **attn.Attention** | 7,747 | 21.1% | 50% | **52.0%** | **95%** |
| attn.proj | 3,173 | 8.6% | 46% | 55.0% | 83% |
| ffn.fc1 | 7,792 | 21.2% | **74%** | 90.0% | 82% |
| ffn.fc2 | 10,600 | 28.9% | 55% | 91.0% | **60%** |

PREFILL, 1024 tokens:

| layer | cycles | % of block | avg of bus | DRAM busy | while busy |
|---|---:|---:|---:|---:|---:|
| attn.QKVgen | 49,780 | 15.7% | 81% | 97.5% | 83% |
| **attn.Attention** | 69,092 | 21.8% | **29%** | **36.2%** | 80% |
| attn.proj | 20,341 | 6.4% | 84% | 96.9% | 87% |
| attn.ln | 5,829 | 1.8% | 85% | 94.7% | 90% |
| ffn.fc1 | 59,433 | 18.8% | 79% | 95.8% | 83% |
| ffn.act | 13,095 | 4.1% | 89% | 97.6% | 91% |
| ffn.fc2 | 93,640 | 29.6% | 72% | 98.8% | 73% |
| ffn.ln | 5,556 | 1.8% | 87% | 91.7% | 95% |

Readings:
* **Attention is idle-limited, not efficiency-limited** -- its while-busy is the
  *highest* of any decode layer, but the DRAM only has work 52% (decode) / 36%
  (prefill) of the time. The array's serial SV chain sets the pace.
* **fc2 is the opposite** -- busy 91%/98.8% but the worst while-busy (60%/73%).
  Most traffic of any layer, and its K-dim accumulation drains poorly.
* **Prefill is near-saturated outside attention** -- six of eight layers at
  95-99% busy. The 60-66% whole-phase figure is dragged down almost entirely by
  attention.

Measurement caveats -- these are indicative, not publication-grade:
* Occupancy is sampled every 500 DRAM cycles, so a layer's busy% rests on 20-67
  samples (+-2% for the big layers, +-5% for `proj`). Read the 95% as "high,
  plausibly 85-95%".
* Requests are attributed by *arrival* cycle but busy is measured over the same
  window, so requests arriving late in a layer are served after the boundary and
  inflate short layers. Attribution is complete (99.9% of requests land in a
  layer) but boundary-shifted.
* Tightening both is easy: sample every 50 cycles and attribute by service.

### ...but prefill wants the opposite

| | RD | busy | avg of bus | while busy | cycles | PE util | reads/ch |
|---|---:|---:|---:|---:|---:|---:|---:|
| decode, spad 4/2 | 28% | 73.3% | 56% | 76% | 298,279 | 0.24% | 272,171 |
| prefill, spad 4/2 | 33% | 83.8% | **66%** | 79% | 317,909 | 45.0% | **292,048** |
| prefill, spad 16/8 | 30% | 76.7% | 60% | 78% | **215,626** | **62.7%** | **158,348** |

Prefill at 4 MB has *higher* bandwidth and is *47% slower*, because it fetches
**1.84x more data**. Prefill has real tile reuse, so scratchpad capacity genuinely
avoids DRAM traffic; decode has none, so capacity buys nothing.

| | decode | prefill |
|---|---|---|
| does spad change traffic? | **no** (constant) | **yes** (1.84x) |
| best spad / accum | 4 MB / 2 MB | >= 16 MB / 8 MB |
| PE utilisation | 0.24% | 62.7% |

There is no single optimum. The shipped config uses **4 MB / 2 MB** (decode-
optimal); prefill studies should raise it. The prefill knee was not swept -- only
the 16/8 point is known, and it may not be the top.

### Load-ahead gate: implemented, and it does NOT work

`ONNXIM_LOAD_AHEAD_TILES=D` (default 0 = off) holds a tile's `MOVIN`s until
fewer than D **older** resident tiles are still loading. Unlike the earlier
`ONNXIM_MAX_OUTSTANDING` throttle it does not restrict a tile's own loads, so
intra-tile memory-level parallelism is untouched. Implementation: a `load_epoch`
on `Tile`/`Instruction`/`MemoryAccess` plus a per-epoch outstanding count in
`Core`.

The intent was the classic stagger -- tile N+1 fetches while tile N preloads:

|  | stock (D=0) | gate (D=1) | predicted |
|---|---:|---:|---:|
| total cycles | 52,753 | 52,385 | ~46,000 |
| attention layer | 12,851 | 12,857 | ~8,200 |
| attention DRAM busy | 49.4% | 51.2% | 75% |
| BW utilization | 20% | 20% | ~30% |

The gate demonstrably fires -- it blocks a `MOVIN` on **19.1%** of cycles -- but
the attention tile-issue pattern is unchanged (8 tiles then 4, with a ~6,085
cycle gap either way), and the DRAM backlog profile is visually identical.

So the model behind it is wrong. It assumed a tile retires roughly at
`load + preload` (~4,250 cycles), which would free a slot early and pull the
second wave forward. Measured, a tile takes ~6,085 cycles to retire, and the
extra ~1,800 cycles are unaccounted for. Whatever gates tile retirement is not
the load/preload sequence.

The gate is left in place, defaulted off, since it is correct and cheap; it
simply is not the bottleneck.

### Per-tile phases: attention is compute-bound, everything else is load-bound

`ONNXIM_TILE_PHASES=<path>` (one CSV per core) stamps each tile with issue,
first-MOVIN, last-load-returned, first-compute-issued, instructions-emptied and
retire. One decode step:

| layer | tiles | lifetime | load | compute | overlap | compute/load |
|---|---:|---:|---:|---:|---:|---:|
| QKVgen | 5 | 8,300 | 6,954 | 5,195 | 3,852 | 0.75 |
| **Attention** | 12 | 6,701 | **2,827** | **4,813** | 942 | **1.70** |
| proj | 6 | 3,096 | 1,921 | 1,171 | 0 | 0.61 |
| fc1 | 6 | 10,122 | 8,944 | 5,763 | 4,588 | 0.64 |
| fc2 | 6 | 12,368 | 8,890 | 3,474 | 0 | 0.39 |

(`load` = first MOVIN issued -> last load returned; `compute` = first compute
instruction issued -> instruction list emptied; `overlap` = how far compute
started *before* loads finished.)

**Attention is the only layer where compute exceeds load** -- 1.70x, against
0.39-0.75 everywhere else. Its tiles are limited by systolic-array preload, not
by memory. It also overlaps worst (942 cycles versus fc1's 4,588).

This explains the whole attention anomaly and retires several wrong theories:

* The DRAM idles at 49.4% during attention because the **array** is the critical
  path there, not because loads are badly scheduled.
* The load-ahead gate did nothing because staggering loads cannot help a
  compute-bound tile.
* Extra bandwidth, bank spread and feed-path fixes do nothing for attention --
  though they are exactly right for fc1/fc2/QKVgen, which are load-bound.

An earlier model here assumed load ~= preload (2,036 vs 2,048). Measured, the
compute phase is 4,813 -- roughly 37 preloads per tile, not the 16 assumed.

Practical split: **attention needs array-side work**; **the GEMM layers need
memory-side work** (the bank-spread and feed-path levers documented above).

### What attention's compute phase is NOT

Per-tile instruction counts (`n_gemm`/`n_vector` columns) show an attention tile
carries **16 GEMM + 9 vector** instructions for a 4,915-cycle compute phase --
about 300 cycles per compute instruction. Three candidate explanations were
tested and all are dead:

| hypothesis | test | result |
|---|---|---|
| softmax vector ops serialize (`!_vector_pipeline.empty()`) | `ONNXIM_VECTOR_DEPTH` 1 / 4 / 16 | **bit-identical**, zero effect |
| preloads take the cold 255-cycle path | warm/cold counters | only **16%** cold (24 of 150) |
| systolic array traversal (`height+width-2` = 254/GEMM) | core 128 / 64 / 32 | 64x64 barely moved (12,585 vs 12,851); 32x32 worse. Confounded -- `core_height` also drives the tiling heuristic, so tile count changes with array size |

`ONNXIM_VECTOR_DEPTH` is left in, defaulting to 1 (stock). The vector-unit cost
model confirms why it cannot matter: vector ops are `vec_op_iter x latency`,
i.e. single-digit cycles, against 258 for a GEMM.

What remains unexplained is why 16 GEMMs cost ~4,900 cycles rather than
pipelining. `SystolicWS::can_issue_compute` permits `core_height` (128) GEMMs in
flight, so if they were independent they would issue 128 cycles apart and finish
in ~2,200. They do not, which implies they are dependency-serialized through the
FlashAttention rescaling chain -- but that has not been measured. Settling it
needs per-instruction start/finish timestamps within a tile, one level below the
per-tile phases already instrumented.

Summing all gaps >= 200 cycles across the step: 49,234 DRAM cycles = 15,385 core
cycles = **29% of the step** spent with an idle DRAM at a drain. That is the
headroom available from softening barriers and overlapping layer boundaries --
independent of any DRAM-side tuning.

### Why 42.6% blocked: bank serialization, not channel under-use

Measured from the trace (2M requests, decode, stock config):

| | |
|---|---|
| channels active per 100-cycle window | **15.4 of 16** (93.5% of windows use all 16) |
| 64-request windows using both data buses | **100%** |
| **distinct banks among 64 consecutive requests to a channel** | **4.9 of 32** |

Channel and pseudochannel parallelism are essentially perfect -- ONNXim's `ipoly`
hash spreads addresses across all 16 channels, and both pseudochannel data buses
are always in use. The concentration is at the **bank** level: a channel's
64-entry scheduling window only ever spans ~5 of its 32 banks, and a third of
windows span just 3.

That follows directly from the address mapping. `RoBaRaCoCh` puts column bits
lowest, so 32 column slots x 16 B = **512 B of consecutive addresses land in one
bank** before the mapping advances; 64 requests x 32 B = 2 KB = 4 banks. With so
few banks in play, a row conflict (tRP+tRCD = 46 cycles of that bank being dead)
has almost nothing to overlap against, so the scheduler finds nothing eligible.

Two independent sweeps confirm bank spread *causes* the blocking:

| address mapper | blocked | row hits | BW | cycles |
|---|---:|---:|---:|---:|
| ChRaBaRoCo (worst spread) | 80.3% | 91.4% | 7% | 4,752,358 |
| RoBaRaCoCh (stock) | 42.8% | 88.1% | 19% | 1,742,312 |
| MOP4CLXOR (best spread) | 26.9% | 56.9% | 20% | 1,635,299 |

| columns/row (org) | B per bank | blocked | row hits | BW | cycles |
|---:|---:|---:|---:|---:|---:|
| 64 (stock) | 512 | 42.8% | 88.1% | 19% | 1,742,312 |
| **16** | **128** | **33.8%** | 68.5% | **21%** | **1,606,442** |
| 8 | 64 | 28.9% | 54.2% | 20% | 1,638,070 |

Blocking tracks bank spread monotonically. But there is a real tension: spreading
banks costs row locality, so the net gain peaks around 128 B per bank (~8%) and
then reverses. Changing `column`/`row` keeps density constant
(2 x 4 x 4 x row x column x 128 must equal 8 Gb) and is config-only.

**If you change the mapper or the org, update `dram_columns`/`dram_rows` in the
JSON too** -- the trace decoder replicates `RoBaRaCoCh` with those values, and
will emit wrong bank/row fields otherwise. It does not replicate MOP4CLXOR at all.

### Submodule changes (extern/ramulator2)

Unlike everything else here, these edit the submodule:

- `resources/ndp_wrappers/ramulator2.hh` — added `pending_requests()` /
  `pending_returns()` accessors so unserved backlog can be told apart from
  served-but-uncollected responses.
- `src/dram_controller/impl/generic_dram_controller.cpp` — the cycle-bucket and
  command-mix counters above, plus a `RAMULATOR_REQBUF` env override for the
  read/write buffer size.

Both are additive and off by default. `git -C extern/ramulator2 checkout .`
reverts them.

## Phase isolation

Prefill and decode are separated by request trace rather than by tagging rows,
which keeps the hook simple. Columns are `time, prompt_length, target_length,
cached_length`; the scheduler computes the target as
`cached_length + prompt_length + target_length`.

- `traces/prefill_1k.csv` — `0, 1024, 1, 0`: a 1024-token prefill, one decode step.
- `traces/decode_1k.csv` — `0, 1, 256, 1023`: starts with 1023 tokens already
  cached, so the run is 256 decode steps against a warm KV cache.

Both stay under `opt-125m`'s 2048-token `max_seq_length`. Raise the lengths when
moving to `llama3-8b` (8192).

---

# LLaMA-2 7B batch-1 decode: the performance model

Everything below is for one transformer block of LLaMA-2 7B (`models/language_models/llama2-7b.json`,
`run_single_layer: true`), batch 1, decode, on 128x128 weight-stationary cores with
16-channel HBM3 at 6.4 Gbps (819.2 GB/s). Runs used `ONNXIM_WEIGHT_SWIZZLE=1` and
`RAMULATOR_REQBUF=1024` unless stated.

## Two floors, and which one binds

Batch-1 decode is bounded by two independent floors. Optimisation advice is only
correct once you know which one is active.

    memory floor = weight_bytes / 819.2 GB/s          scales with precision
    array floor  = weight_tiles x core_height / cores  NO precision term

    weight_tiles = 202,375,168 / core_height^2   (one LLaMA-2 7B block)

| config      |    cycles | mem floor | array floor | binding | array util |
|-------------|----------:|----------:|------------:|---------|-----------:|
| fp16 4-core |   536,187 |   494,080 |     395,264 | memory  |        80% |
| fp16 8-core |   537,506 |   494,080 |     197,632 | memory  |        43% |
| int8 4-core |   426,372 |   247,040 |     395,264 | array   |        97% |
| int8 8-core |   287,356 |   247,040 |     197,632 | memory  |        75% |

Consequences that are easy to get wrong:

- **int8 gives 1.26x, not 2x**, at 4 cores. Halving the bytes drops the memory floor
  *below* the array floor, so the array becomes binding. **int4 at 4 cores buys nothing.**
- **Core scaling only pays after the precision cut.** Cores lower only the array floor.
  At fp16 that floor is already slack, so 4->8 cores = 0.998x (measured). At int8 it is
  1.48x. Quantise first, then scale cores.
- Combined fp16 4-core -> int8 8-core: **1.87x** (17.2 -> 9.2 ms/token, 58.3 -> 108.8 tok/s).

## The ingestion formula

One expression predicted every geometry result in this document:

    array floor cycles = weight_bytes / (core_width x precision x num_cores)
    avg BW utilisation = core_width x precision x num_cores / 819.2

Weights enter the array through **one edge of `core_width` PEs at 1 value/cycle**
(domino shift-in), so ingestion = `core_width x precision` bytes/cycle. `core_height`
**cancels**: a taller tile is more bytes *and* proportionally more cycles.

Validated at 1 core, fp16, 16ch, to within 4-5% (the residual is the cold-preload path,
7.2% of preloads at `2*height-1` = 255 cyc instead of 128):

| array   | predicted |  measured | ratio | avg BW | row hit | PE util |
|---------|----------:|----------:|------:|-------:|--------:|--------:|
| 32x32   | 6,324,224 | 6,593,550 |  1.04 |     7% |     96% |   3.12% |
| 64x64   | 3,162,112 | 3,301,400 |  1.04 |    15% |     96% |   1.56% |
| 128x128 | 1,581,056 | 1,656,837 |  1.05 |    30% |     96% |   0.78% |

**A smaller array is worse on both latency and bandwidth.** Halving `d` quarters the tile
*area* (4x the tiles) but only halves the load *time* -> net 2x slower. PE utilisation is
the only metric that improves, and it is `batch_size / core_width`.

## Things that cannot help at batch 1 (with proofs)

**Dataflow / stationarity.** At batch 1 every weight crosses the array boundary exactly
once, so the traffic is dataflow-independent; and the boundary is a property of the array,
not of how data moves inside it. WS and OS give *identical* cycles. For the 4096x4096 GEMM:

    weight-stationary : 1,024 tiles x 128 cyc            = 131,072 cyc
    output-stationary : 4,096 outputs x 4,096 K = 33.6 MB
                        streamed at 256 B/cyc            = 131,072 cyc

OS is additionally worse on occupancy (M=4,096 outputs fills 25% of a 128x128 array).
`SystolicOS` is unimplemented (`assert(0)`) and `LanguageModel.cc` hardcodes `GemmWS`;
implementing it is worthwhile only to study batch >1, where the reuse term stops being 1.

**Deeper pipelining.** Already modelled. `SystolicWS.cc:64-88` ("Preload can be hided")
overlaps shift-in with the previous tile's compute: initiation interval
`MAX(compute_size, core_height)` = 128 while full latency is 258. Measured 92.8% warm /
7.2% cold. The 128 cycles is a **throughput** limit (edge injection rate), not a latency
that deeper pipelining could hide. At batch 1 `compute_size` = 1, so there is nothing to
hide the fill behind.

**More memory buffering.** Already sufficient. Core cycle accounting at 1 core:
`inst-issued 1.03% | stall-no-issuable-inst 98.96% | no-tile-resident 0.00%` — the core
waited on memory for **41 cycles out of 1,656,837**. `tile_depth: 3` holds 1.69 MB
(3 mapping tiles) resident; DRAM supplies that 2x faster than the array drains it, hence
the measured 34.3% channel-busy. Burst-filling the whole 4 MB would raise *while-busy*
utilisation toward 100% but leaves the average at 31.25% exactly
(`fill 5,120 cyc / drain 16,384 cyc`), because buffering cannot change an average.

**Non-square arrays.** `Mapping.cc:54,58` sets `dim = core_height` and asserts
`height == width`. `core_width` never affects tile count. Measured: 256x64 = 857,804
(2.0x slower), 512x32 = 1,953,878 (4.6x slower); 64x256 and 256x256 die with SIGFPE
(exit 136) downstream of SkipLayerNorm at seq=1.

## The 1-core result

The user's constraint was 1 core, 128x128, `preload_div = 1`, using the **full** 16-channel
bandwidth (reducing channels was explicitly disallowed). Both original targets are met:

| 1 core 128x128, nCCDL=4 |    cycles | traffic | avg BW | busy% | while-busy | wgt row hit |
|-------------------------|----------:|--------:|-------:|------:|-----------:|------------:|
| fp16                    | 1,655,501 |  423 MB |  31.2% | 32.6% |    **95.5%** |    **96.4%** |
| int8                    | 1,650,905 |  211 MB |  15.6% | 16.5% |      94.8% |       95.5% |

- **Average BW of 31.2% is structural, not a shortfall.** A 128x128 array at 1 row/cycle
  is a **256 GB/s consumer**; HBM3 supplies 819.2. The bus is not inefficient when it runs
  (95.5%) — it is only asked to run a third of the time. Stated positively: *HBM3 is 3.2x
  over-provisioned for one core.*
- **int8 at 1 core is the wrong direction for a bandwidth goal**: identical runtime (the
  array floor has no precision term), half the traffic, so average BW halves. while-busy
  and row hit are precision-insensitive.
- Reducing to 4 channels reaches 95% *average* BW and 96% row hit at 1 core
  (2,088,468 cyc, 1.26x slower) — architecturally the correct rebalance, but it does not
  satisfy "use the available bandwidth".

## nCCDL: the one free win

`nCCDL` is the column-to-column spacing **within a bank group** — exactly the constraint a
high row-hit stream hits, because a row hit means same row -> same bank -> same bank group.
With 8 bank groups per channel (2 pch x 4 bg), `nCCDL=8` allows `8 x 1/8 = 1` access/cycle,
i.e. peak is reachable only with a perfect 8-way interleave and zero slack for refresh or
conflicts. `nCCDL=4` (the legal floor, `>= nBL`) gives 2x headroom.

Measured at 1 core: **while-busy BW 87% -> 95.5%**, runtime -0.08%. Now the default in
`configs/ramulator2_configs/_c128.yaml`. Runs recorded before this change used `nCCDL: 8`.

**Compute while-busy BW from TOTAL traffic (423 MB), not weights-only (404.8 MB)** — the
latter understates it (gives 90% instead of 95.5%).

## Where the residual 4.5% goes

Per channel, from the controller's own counters:

    ticks 5,306,092 | issued 900,230 (17.0%) | blocked 830,437 (15.7%) | empty 3,575,425 (67.4%)
    busy = issued + blocked = 1,730,667 (32.6%)
    served 25.2 MB of 26.4 MB bus capacity -> 95.5%

- **refresh ~35% of the gap** (`nRFCSB 513` / `nREFI 12500`) — unavoidable in real DRAM.
- **un-hidden tRCD/tRP ~65%** — 36,744 ACTs + 36,712 PREs per channel, driven by the 3.5%
  row conflicts; the controller hides most but not all behind other banks.

~95-96% is the practical while-busy ceiling. This is not a scheduling failure.

## Row buffer: two metrics, and they disagree

**Report both. `hits/(hits+conflicts)` hides the effect that matters.**

- `hits/(hits+conflicts)` is **95-96% in every configuration measured** — array
  32x32..512x512, cores 1..8, channels 4..16, fp16 and int8. It excludes *misses*
  (no row open), so it only measures conflict behaviour, which the tile-major swizzle
  (`ONNXIM_WEIGHT_SWIZZLE=1`, `Operation::make_address_tiled`) already fixed: conflicts
  fell 25% -> 3.5% and stay there.
- **weights `hit/all` is NOT invariant** and is the honest number:

| config  |    cycles | avg BW | h/(h+c) | weights hit/all |         miss |
|---------|----------:|-------:|--------:|----------------:|-------------:|
| 1 core  | 1,655,501 |  31.2% |   96.4% |       **96.4%** |     **0.1%** |
| 4 cores |   532,386 |  96.9% |   95.1% |       **74.8%** |    **21.2%** |

## THE ONE PLACE A CONTROLLER HAS HEADROOM

At 4 cores the bus reaches 96.9% but weights row hit drops to 74.8%, and the loss is
**misses (21.2%), not conflicts (4.0%)**. At 1 core misses are 0.1%.

It is tempting to call this a fundamental trade-off (bandwidth needs concurrency, locality
needs sequentiality). **That is wrong, and the trace disproves it.** Measured with
`scripts/analyze_interleave.py` on traces captured via `ONNXIM_DRAM_TRACE` (the hook logs
`core_id` per request):

|                          | 1 core | 4 cores |                              |
|--------------------------|-------:|--------:|------------------------------|
| arrival-order row hit    |  96.4% |   95.9% | locality PRESENT in demand   |
| simulator hit (weights)  |  96.4% |   74.8% | locality REALISED            |
| shortfall                |  0 pts | 21.1 pts|                              |

At 1 core the two agree **exactly**, which validates the method. At 4 cores they diverge by
21 points. **The 4-core address stream still has the locality; FR-FCFS discards it.** This
is a SERVICE-ORDER problem, not an address-pattern problem.

The obvious mechanical story — cores evicting each other's rows — was tested and is NOT the
cause. Banks do go from 1.00 to 4.00 distinct requesters, and 87.4% of row changes coincide
with a core switch (0.0% at 1 core), but row changes only rise from 3.6% to 4.1% of
accesses. Cores switch at row-run boundaries, not mid-row, so interleaving costs 0.5 pts,
not 21.

**This is the only evidence in this document that a controller microarchitecture has
headroom.** Every other result points the other way: conflicts already at 3.5%, every
scheduling experiment flat or negative, only address-layout changes ever working. Here
there are 21 points of locality in the demand stream that the current controller drops.

Untested hypothesis for the cause (MEASURE before believing): at 4 cores the bus is 97%
busy with ~1024 buffered requests spread over 512 banks — about 2 candidates per bank — so
FR-FCFS has almost nothing to choose from when looking for a row hit; at 1 core the bus is
33% busy and rows sit undisturbed. This would also explain why growing `RAMULATOR_REQBUF`
saturated at +3.5 points: the problem is how requests are GROUPED per bank, not how many
are buffered.

By stream (1 core, fp16): weights 96.4% hit / 0.1% miss / 3.5% conflict;
kv+activations 76.2% / 17.8% / 6.0%. The kv+act stream is much worse but is only 4.4% of
accesses, so it barely moves the aggregate — that is where any remaining headroom is.

## What worked, and what did not

Worked (all measured): `column: 16 -> 128` (41% faster on LLaMA); tile-major weight layout
(conflicts 25% -> 4%, BW 88% -> 96%, 8.5% faster); `tile_depth: 3`; `RAMULATOR_REQBUF` 64 ->
1024; `nCCDL: 8 -> 4`.

Failed (all measured): bank XOR hashing (78.3 -> 78.9%), stream isolation (negative),
scheduling-window growth (saturates at +3.5 pts), precharge-owner priority (73% -> 68%),
`RoCoBaCh` split-column mapper (2.04x slower), vector pipeline depth (bit-identical),
smaller array (toll x count invariant, then worse), load-ahead gating (zero gain).

**The pattern: only changes to *what addresses the stream generates* worked. Every attempt
to reorganise an existing stream inside the memory controller failed.**

## KV-cache writes (`ONNXIM_KV_WRITES=1`)

Stock ONNXim marks the KV-concat tiles `skip`, so `Core::push_tile()` retires
them without executing anything: the KV cache is never written, and the write
traffic in the prefill row of the table above is activations only.
`ONNXIM_KV_WRITES=1` executes those tiles (read of the QKV output, write of the
query and of the new K/V rows). Needed for speculative decoding (SPECDEC.md),
where the overwrite of rejected lookahead rows is part of what is measured.
Default stays off so earlier numbers are reproducible.
