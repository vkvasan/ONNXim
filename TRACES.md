# Generating DRAM traces

How to build the simulator, run a serving workload, and get a cycle-stamped, per-stream
DRAM address trace out of it.

Everything below is decode-phase LLaMA-2 7B on HBM3 unless stated otherwise. See
`KV_PLACEMENT.md` for what the traces were used to measure.

## 1. Build

The build runs inside a container; the build directory must be named `build` because
`CMakeLists.txt` hardcodes that path.

```bash
./scripts/devshell.sh ./scripts/build_in_docker.sh
```

Or keep a long-lived container and exec into it, which is what the sweeps do:

```bash
docker run -d --name onnxim_work \
  -v $PWD:/workspace/ONNXim \
  -v $PWD/.conan-cache:/root/.conan/data \
  -w /workspace/ONNXim -e ONNXIM_HOME=/workspace/ONNXim \
  onnxim sleep infinity

docker exec onnxim_work bash -c 'cd /workspace/ONNXim/build && cmake --build . -j 16'
```

## 2. Define a workload

Two files. A **request trace** (`traces/<name>.csv`), one row per request:

```
time, prompt_length, target_length, cached_length
0, 1, 1, 237
0, 1, 1, 4089
```

| column | meaning |
|---|---|
| `time` | arrival cycle; requests are admitted when `time <= cycle` |
| `prompt_length` | prefill tokens (`1` for a decode-only step) |
| `target_length` | retire when `current_length` reaches this (`1` = one step) |
| `cached_length` | KV already resident, so a run can start mid-conversation |

And a **model list** (`example/<name>.json`) naming the model and scheduler:

```json
{ "models": [{
    "name": "llama2-7b-1L",
    "trace_file": "az128.csv",
    "scheduler": "simple",
    "scheduler_config": { "max_batch_size": 128 }
}]}
```

`scheduler` is `simple` (static batching: admits only when the batch is empty),
`iter_level` (continuous batching), or `specdec` (see §6). The `-1L` model runs a single
transformer layer and scales the reported footprint by `num_layers`; this is what makes a
cycle-level step take hours instead of days.

## 3. Run and emit a trace

```bash
docker exec onnxim_work bash -c 'cd /workspace/ONNXim/build && \
  env ONNXIM_WEIGHT_SWIZZLE=1 RAMULATOR_REQBUF=1024 \
      RAMULATOR_WEIGHT_LIMIT=500000000 ONNXIM_KV_WRITES=1 \
      ONNXIM_ICNT_PORT_BY_CHANNEL=1 ONNXIM_DRAM_OCC=1 \
      ONNXIM_KV_LAYOUT=headbank ONNXIM_KV_BLOCK=64 ONNXIM_KV_BANKS_PER_HEAD=1 \
      ONNXIM_DRAM_TRACE=/workspace/ONNXim/out/mytrace.csv \
      ONNXIM_DRAM_TRACE_LIMIT=40000000 \
      ./bin/Simulator --config ../configs/_c128.json \
        --models_list ../example/az128.json --mode language \
        --trace_file az128.csv > /workspace/ONNXim/out/mytrace.log 2>&1'
```

`ONNXIM_DRAM_TRACE` is the only variable required to get a trace; `ONNXIM_DRAM_TRACE_LIMIT`
caps the row count (a full `az128` run is ~122 M rows, ~5.5 GB raw, ~0.77 GB gzipped).

**Set `ONNXIM_ICNT_PORT_BY_CHANNEL=1`.** Stock ONNXim fans one core's request FIFO across
16 injection ports, so a core's requests reach a channel locally reordered. One FIFO per
(core, channel) cuts row conflicts 35–45% on every layout.

### Layout selection

| variable | values | effect |
|---|---|---|
| `ONNXIM_KV_LAYOUT` | `block` (default), `head`, `headbank` | placement family |
| `ONNXIM_KV_BLOCK` | tokens per page, e.g. `16`, `64` | allocator granularity |
| `ONNXIM_KV_BANKS_PER_HEAD` | `1`, `2`, … | banks per head under `headbank` |
| `ONNXIM_KV_V_BANK_OFFSET` | e.g. `16` | move V to a different bank than K |
| `ONNXIM_KV_ALLOC` | `shuffle` (default), `seq` | scattered vs identity block table |

The three layouts compared in `KV_PLACEMENT.md`:

```bash
ONNXIM_KV_LAYOUT=block    ONNXIM_KV_BLOCK=16                              # paged baseline
ONNXIM_KV_LAYOUT=head     ONNXIM_KV_BLOCK=16                              # head-major
ONNXIM_KV_LAYOUT=headbank ONNXIM_KV_BLOCK=64 ONNXIM_KV_BANKS_PER_HEAD=1   # head->bank
```

## 4. Trace format

One line per 32 B DRAM request, in arrival order at the controller:

```
cycle,channel,pseudochannel,bankgroup,bank,row,column,address,rw,core,operand
7,0,1,0,0,192,12,0x6004c00,R,0,102
```

| field | notes |
|---|---|
| `cycle` | DRAM cycles at 3.2 GHz; × 0.3125 = ns |
| `bank` slot | `pseudochannel + 2·bankgroup + 8·bank`, 0–31 |
| `column` | always even: Ramulator counts 16 B columns, requests are 32 B |
| `address` | global byte address (not the ÷16 compacted form the controller sees) |
| `operand` | **100** = Q, **101** = K, **102** = V, **≥200** = outputs, **0** = KV writes |

**Use the `operand` column to separate streams, never an address range.** Activation
tensors share the KV pool's address region and run at ~37% row hit, so an address-threshold
split folds them into KV and understates its locality by several points.

## 5. Read the results

The run log carries per-stream row-buffer statistics without any post-processing:

```
ROWSPLIT weights  acc … hit …% miss …% confl …%
ROWSPLIT kv+act   acc … hit …% miss …% confl …%     <- KV only (operand 101/102)
ROWSPLIT act      acc … hit …% miss …% confl …%
CTRL cmd mix: RD … WR … ACT … PRE … other …
[DRAM] channel-busy (>=1 request outstanding) …% | dram cycles …
```

Derived quantities:

```
KV reads per row visit = 1 / (miss% + conflict%)
KV activations         = KV_accesses × (miss% + conflict%)
opens per row          = 32 × (miss% + conflict%)
avg BW                 = (reads + writes) × 32 B × n_channels / (cycles/3.2e9) / 819.2e9
while-busy BW          = avg BW / channel-busy fraction
```

Do not quote `reads/ACT` from the `CTRL cmd mix` line as a KV figure; it is an all-stream
aggregate diluted by the weight stream.

For offline analysis of an emitted trace:

```bash
python3 scripts/analyze_dram_trace.py out/mytrace.csv
```

## 6. Speculative decoding traces

Cycle-level closed loop, which also emits a DRAM trace:

```bash
env ONNXIM_KV_WRITES=1 ./bin/Simulator --config ../configs/_multi_16x32.json \
    --models_list ../example/sd_v4m512.json --mode language --trace_file sd_v4m512.csv
```

Open loop, seconds instead of ~40 minutes per verify step, for sweeps:

```bash
python3 scripts/specdec_trace.py --k 4 --alpha 0.8 --scorer mqa --emit-workload
```

`--scorer mqa` is the single-call scorer (cache read once); `--scorer expand` is batch
expansion (cache read k+1 times). They differ by an order of magnitude in KV traffic.

To diff a closed-loop trace against the generator:

```bash
python3 scripts/specdec_analyze.py out/specdec/sd_v4m512.csv --ref sd_v4m512.ref.json
```

## 7. Reproducing the published numbers

```bash
E='ONNXIM_WEIGHT_SWIZZLE=1 RAMULATOR_REQBUF=1024 RAMULATOR_WEIGHT_LIMIT=500000000
   ONNXIM_KV_WRITES=1 ONNXIM_ICNT_PORT_BY_CHANNEL=1 ONNXIM_DRAM_OCC=1'

# paged baseline            79.3% KV row hit, 24.19M DRAM cycles
env $E ONNXIM_KV_LAYOUT=block ONNXIM_KV_BLOCK=16 ./bin/Simulator \
  --config ../configs/_c128.json --models_list ../example/az128.json \
  --mode language --trace_file az128.csv

# page size 64 alone        89.5%, 20.84M
env $E ONNXIM_KV_LAYOUT=block ONNXIM_KV_BLOCK=64 ...

# head -> bank placement    93.5%, 21.46M, 3.2x fewer KV activations
env $E ONNXIM_KV_LAYOUT=headbank ONNXIM_KV_BLOCK=64 ONNXIM_KV_BANKS_PER_HEAD=1 ...
```

One `az128` run is ~3 hours at one layer. Runs are deterministic: identical configurations
give bit-identical cycle counts, so two runs differing only in layout are directly
comparable.
