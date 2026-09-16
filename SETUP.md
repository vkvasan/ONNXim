# Setup

Publishing this work, and getting the simulator running from a clean machine.

## 1. What has to be published: two repositories

The tree is ONNXim with **Ramulator2 as a git submodule**, and both are modified.
Publishing only ONNXim yields a simulator that builds but reproduces none of the reported
numbers, because the submodule would resolve to upstream Ramulator2.

| repo | upstream | what changed here |
|---|---|---|
| **ONNXim** | `PSAL-POSTECH/ONNXim` | KV placement (`Attention.cc`), specdec scheduler, per-core issue, DRAM tracing, workloads |
| **Ramulator2** | `PSAL-POSTECH/ramulator2` | ~1,000 lines: operand-tagged row-buffer stats, close-cause audit, row-hit scheduler tier, refresh fix, LPDDR6 |

Both are MIT-licensed; keep the `LICENSE` and copyright notices in place.

### Publish Ramulator2 first

```bash
cd extern/ramulator2
git checkout -b kv-instrumentation
git add -A
git commit -m "Row-buffer instrumentation, row-hit scheduler tier, all-bank refresh fix"
git remote add mine https://github.com/<you>/ramulator2.git
git push -u mine kv-instrumentation
```

### Repoint the submodule, then publish ONNXim

```bash
cd ../..                                   # back to the ONNXim root
git config -f .gitmodules submodule.extern/ramulator2.url \
    https://github.com/<you>/ramulator2.git
git config -f .gitmodules submodule.extern/ramulator2.branch kv-instrumentation
git submodule sync

git checkout -b kv-placement
git add -A
git commit -m "KV cache placement study: head-to-bank layout, specdec scheduler, DRAM tracing"
git remote add mine https://github.com/<you>/ONNXim_vLLM.git
git push -u mine kv-placement
```

`out/` (82 GB of simulation output) is already in `.gitignore`. Nothing else in the tree
exceeds 5 MB.

If you have the GitHub CLI, `gh repo create <name> --private --source=. --remote=mine`
replaces the manual repo creation in both cases.

## 2. Running it: Docker

Docker is the supported path. The build needs gcc-10, CMake 3.22 and conan 1.57, which the
image pins; host builds against newer toolchains are not tested.

```bash
git clone --recursive https://github.com/<you>/ONNXim_vLLM.git
cd ONNXim
docker build -t onnxim .          # ~20 min: builds CMake from source, installs torch
```

`--recursive` matters: five submodules (`ramulator2`, `booksim`, `onnx`, `protobuf`,
`torch2timeloop`). If you forget it, `git submodule update --init --recursive`.

### Build the simulator

The image already contains a build, but the working flow bind-mounts your checkout over
`/workspace/ONNXim`, which shadows it. **Build once inside the container after mounting:**

```bash
./scripts/devshell.sh ./scripts/build_in_docker.sh
```

which runs `conan install .. --build=missing && cmake .. && make -j$(nproc)` in `build/`
and produces `build/bin/Simulator`. The build directory must be named `build`:
`CMakeLists.txt` hardcodes `include("${CMAKE_SOURCE_DIR}/build/conanbuildinfo.cmake")`.

### A long-lived container for sweeps

Starting a container per run wastes time on image start-up. For repeated runs:

```bash
docker run -d --name onnxim_work \
  -v $PWD:/workspace/ONNXim \
  -v $PWD/.conan-cache:/root/.conan/data \
  -w /workspace/ONNXim -e ONNXIM_HOME=/workspace/ONNXim \
  onnxim sleep infinity

docker exec onnxim_work bash -c 'cd /workspace/ONNXim/build && cmake --build . -j 16'
```

The conan cache is bind-mounted to the host so rebuilds do not re-download dependencies.

## 3. Verify

```bash
docker exec onnxim_work bash -c 'cd /workspace/ONNXim/build && \
  env ONNXIM_WEIGHT_SWIZZLE=1 RAMULATOR_REQBUF=1024 RAMULATOR_WEIGHT_LIMIT=500000000 \
      ONNXIM_KV_WRITES=1 ONNXIM_ICNT_PORT_BY_CHANNEL=1 ONNXIM_DRAM_OCC=1 \
      ONNXIM_KV_LAYOUT=headbank ONNXIM_KV_BLOCK=64 ONNXIM_KV_BANKS_PER_HEAD=1 \
      ./bin/Simulator --config ../configs/_c128.json \
        --models_list ../example/az128.json --mode language --trace_file az128.csv' \
  | grep -E "ROWSPLIT|REFRESH|dram cycles"
```

Expected on a correct build (~3 hours; `az128` is 128 requests over one transformer layer):

```
[REFRESH] AllBank enabled: nREFI=12500 nRFC=1122 ...
ROWSPLIT kv+act   acc 6410112 hit 93.5% miss 3.4% confl 3.1%
dram cycles 21459674
```

Two checks that the instrumented Ramulator2 is actually in the build:

- the `[REFRESH] AllBank enabled:` line appears at start-up — upstream silently issues no
  refresh at all on HBM3, and prints nothing;
- `ROWSPLIT` lines appear — upstream has no per-operand row-buffer statistics.

If either is missing, the submodule is pointing at upstream Ramulator2.

Runs are deterministic: identical configurations give bit-identical cycle counts, so any
difference from the numbers above means a real difference in configuration or build.

See `TRACES.md` for generating DRAM traces, and `KV_PLACEMENT.md` for what was measured.
