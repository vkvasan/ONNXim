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

### Comparing KV cache layouts

```bash
./run.sh --layout block     # paged block-major, the vLLM default
./run.sh --layout head      # head-major
./run.sh                    # head-to-bank placement (default)
```

Measured on 128 Azure requests, 4 x 128x128 NPU, HBM3:

| KV layout | row-buffer hit | row activations |
|---|---|---|
| block-major, 16-token pages *(the vLLM default)* | 79.3% | 1,326,887 |
| block-major, 64-token pages † | 89.5% | 673,062 |
| **head-to-bank, 64-token pages** | **93.5%** | **416,657** |

**Putting the head index on the DRAM bank field cuts row activations 3.2x.** Each head
then owns a bank no other head can address, so nothing evicts its rows.

† The middle row is the same mechanism reached by accident. At 64 tokens the block-major
address expression reduces to `bank = head` — it *is* the head-to-bank mapping, without the
512 KB-aligned base that makes the assignment exact. Supplying that alignment is worth the
remaining 4 points and a further 1.6x on activations.

A run takes ~3 hours. `./run.sh --help` lists the options.

## Documentation

- [`KV_PLACEMENT.md`](KV_PLACEMENT.md) — the writeup: what was measured and why
- [`TRACES.md`](TRACES.md) — generating DRAM traces in detail
- [`SETUP.md`](SETUP.md) — build, and publishing a fork
- [`SPECDEC.md`](SPECDEC.md) — speculative decoding
