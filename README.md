# LLaMA-2 7B DRAM traces on a simulated NPU

Simulate LLaMA-2 7B decode on a cycle-level NPU + HBM3 model and get the DRAM address
trace: every memory request, with its cycle, bank, row, and which tensor it belongs to.

## Requirements

Docker, ~10 GB of disk, and a CPU core per run. Nothing else — the toolchain
(gcc-10, CMake 3.22, conan) lives inside the image.

## Run it

```bash
git clone --recursive https://github.com/vkvasan/ONNXim_vLLM.git
cd ONNXim_vLLM
./run.sh
```

`--recursive` matters: Ramulator2 is a submodule and the simulator will not build
without it. If you forget, `git submodule update --init --recursive`.

**What happens.** `run.sh` does each step only if it has not been done, so the first
invocation is slow and later ones start immediately:

| step | when | takes |
|---|---|---|
| build the Docker image | image `onnxim` absent | ~20 min |
| start a container | not already running | seconds |
| build the simulator | `build/bin/Simulator` absent | ~10 min |
| **simulate** | every time | **~3 hours** |

The simulation itself is one decode step over **128 real serving requests** (contexts
169–5,305 tokens, from the Azure LLM Inference Dataset) against one transformer layer
of LLaMA-2 7B, on a 4-core 128×128 NPU with HBM3.

**What you get**, in `out/`:

```
az128_headbank_pg64_k1.csv     the DRAM trace, one line per 32 B request
az128_headbank_pg64_k1.log     row-buffer statistics, bandwidth, command mix
```

The filename encodes the configuration, so runs with different layouts do not
overwrite each other.

```
cycle,channel,pseudochannel,bankgroup,bank,row,column,address,rw,core,operand
7,0,1,0,0,192,12,0x6004c00,R,0,102
```

`cycle` is DRAM cycles at 3.2 GHz (× 0.3125 = ns). `operand` identifies the stream:
**100** = Q, **101** = K, **102** = V, **≥200** = outputs, **0** = KV writes — use this
rather than address ranges, since activations share the KV address region.

When it finishes it prints the headline numbers and where the files are:

```
==> done
    ROWSPLIT kv+act   acc 6410112 hit 93.5% miss 3.4% confl 3.1%
    dram cycles 21459674
    trace  out/az128_headbank_pg64_k1.csv  (40000000 rows, 1.8G)
```

## Options

```bash
./run.sh --layout block        # paged block-major, the vLLM default
./run.sh --layout head         # head-major
./run.sh --page 16             # KV page size in tokens
./run.sh --workload v8m1024    # a different request trace from traces/
./run.sh --limit 5000000       # cap trace rows (default 40M; ~1.8 GB)
./run.sh --no-trace            # statistics only, no trace file
./run.sh --help
```

A full trace is ~5.5 GB uncompressed; `--limit` caps the rows written but not the
runtime, which is always a complete simulation.

## Results

Measured on 128 Azure serving requests, one transformer layer of LLaMA-2 7B,
4 x 128x128 NPU, HBM3 at 6.4 Gbps. Decode phase.

### KV cache placement

| KV layout | row-buffer hit | row activations | cycles |
|---|---|---|---|
| block-major, 16-token pages *(the vLLM default)* | 79.3% | 1,326,887 | 24.19M |
| block-major, 64-token pages † | 89.5% | 673,062 | 20.84M |
| **head-to-bank, 64-token pages** | **93.5%** | **416,657** | 21.46M |
| head-major | 69.9% | 1,935,917 | **17.73M** |

**Putting the head index on the DRAM bank field cuts row activations 3.2x.** Each head
then owns a bank no other head can address, so nothing evicts its rows.

† At 64 tokens the block-major address expression reduces to `bank = head` — it *is* the
head-to-bank mapping, without the 512 KB-aligned base that makes the assignment exact.
Supplying that alignment is worth the remaining 4 points and a further 1.6x on activations.

```bash
./run.sh --layout block     # the vLLM default
./run.sh --layout head      # head-major
./run.sh                    # head-to-bank (default)
```

### Locality and bandwidth pull against each other

Head-major is the *fastest* layout and has the *worst* locality. Isolation confines a head
to one bank, so it has nowhere to read while that bank switches rows; head-major hops a
bank every 16 KB and hides its switches behind 28 live banks instead of 2.8.

| | banks live | reads per row visit | bus busy while work outstanding |
|---|---|---|---|
| head-to-bank, K=1 | 2.8 | 15.4 | 77.9% |
| head-to-bank, K=2 | ~5.6 | 6.7 | 92.3% |
| head-major | 28.4 | 3.3 | 98.3% |

### The memory controller cannot recover what placement did not create

Every controller mechanism tried is worth **≤1.1 points of row hit on every layout**: the
row-hit tier that stock `FRFCFS` is missing, bank preparation, and refresh. A scheduler can
reorder requests but cannot change which addresses share a row.

## Documentation

- [`KV_PLACEMENT.md`](KV_PLACEMENT.md) — the writeup: what was measured and why
- [`TRACES.md`](TRACES.md) — generating DRAM traces in detail
- [`SETUP.md`](SETUP.md) — build, and publishing a fork
- [`SPECDEC.md`](SPECDEC.md) — speculative decoding
