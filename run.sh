#!/usr/bin/env bash
# Simulate LLaMA-2 7B decode on an NPU + HBM3 and write a DRAM address trace.
#
#   ./run.sh                          headbank layout, page 64   (best locality)
#   ./run.sh --layout block           paged block-major          (vLLM default)
#   ./run.sh --layout head            head-major
#   ./run.sh --workload v8m1024       a different request trace
#   ./run.sh --limit 5000000          cap the trace at N requests (default 40M)
#   ./run.sh --no-trace               statistics only, no trace file
#
# Everything runs in Docker. First invocation builds the image (~20 min) and the
# simulator (~10 min); later invocations reuse both.
set -euo pipefail

LAYOUT=headbank; PAGE=64; K=1; WORKLOAD=az128; LIMIT=40000000; TRACE=1
while [ $# -gt 0 ]; do
  case "$1" in
    --layout)   LAYOUT=$2; shift 2 ;;
    --page)     PAGE=$2;   shift 2 ;;
    --banks)    K=$2;      shift 2 ;;
    --workload) WORKLOAD=$2; shift 2 ;;
    --limit)    LIMIT=$2;  shift 2 ;;
    --no-trace) TRACE=0;   shift ;;
    -h|--help)  sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 1 ;;
  esac
done
[ "$LAYOUT" = block ] && [ "$PAGE" = 64 ] && PAGE=16   # block-major default is page 16

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TAG="${WORKLOAD}_${LAYOUT}_pg${PAGE}$([ "$LAYOUT" = headbank ] && echo "_k${K}")"
command -v docker >/dev/null || { echo "docker is required" >&2; exit 1; }

if ! docker image inspect onnxim >/dev/null 2>&1; then
  echo "==> building the image (once, ~20 min)"
  docker build -t onnxim "$ROOT"
fi
if ! docker ps --format '{{.Names}}' | grep -qx onnxim_work; then
  docker rm -f onnxim_work >/dev/null 2>&1 || true
  mkdir -p "$ROOT/.conan-cache"
  docker run -d --name onnxim_work \
    -v "$ROOT:/workspace/ONNXim" -v "$ROOT/.conan-cache:/root/.conan/data" \
    -w /workspace/ONNXim -e ONNXIM_HOME=/workspace/ONNXim onnxim sleep infinity >/dev/null
fi
if [ ! -x "$ROOT/build/bin/Simulator" ]; then
  echo "==> building the simulator (once, ~10 min)"
  docker exec onnxim_work bash -c 'cd /workspace/ONNXim && mkdir -p build && cd build &&
    conan install .. --build=missing >/dev/null && cmake .. >/dev/null && make -j"$(nproc)"' \
    || { echo "build failed" >&2; exit 1; }
fi

mkdir -p "$ROOT/out"
TRACE_ENV=""
[ "$TRACE" = 1 ] && TRACE_ENV="ONNXIM_DRAM_TRACE=/workspace/ONNXim/out/${TAG}.csv ONNXIM_DRAM_TRACE_LIMIT=${LIMIT}"

cat <<INFO
==> simulating LLaMA-2 7B decode
    workload  $WORKLOAD          $(sed -n '2,$p' "$ROOT/traces/${WORKLOAD}.csv" 2>/dev/null | wc -l) requests
    layout    $LAYOUT  page $PAGE$([ "$LAYOUT" = headbank ] && echo "  banks/head $K")
    memory    HBM3 16ch x 2pch x 4bg x 4bank, 6.4 Gbps
    output    out/${TAG}.log$([ "$TRACE" = 1 ] && echo " + out/${TAG}.csv")
    expect    ~3 hours
INFO

docker exec onnxim_work bash -c "cd /workspace/ONNXim/build && \
  env ONNXIM_WEIGHT_SWIZZLE=1 RAMULATOR_REQBUF=1024 RAMULATOR_WEIGHT_LIMIT=500000000 \
      ONNXIM_KV_WRITES=1 ONNXIM_ICNT_PORT_BY_CHANNEL=1 ONNXIM_DRAM_OCC=1 \
      ONNXIM_KV_LAYOUT=${LAYOUT} ONNXIM_KV_BLOCK=${PAGE} ONNXIM_KV_BANKS_PER_HEAD=${K} \
      ${TRACE_ENV} \
      ./bin/Simulator --config ../configs/_c128.json \
        --models_list ../example/${WORKLOAD}.json --mode language \
        --trace_file ${WORKLOAD}.csv > /workspace/ONNXim/out/${TAG}.log 2>&1"

echo
echo "==> done"
grep -E "ROWSPLIT kv|dram cycles" "$ROOT/out/${TAG}.log" | tail -2 | sed 's/^/    /'
[ "$TRACE" = 1 ] && [ -f "$ROOT/out/${TAG}.csv" ] && \
  echo "    trace  out/${TAG}.csv  ($(wc -l < "$ROOT/out/${TAG}.csv") rows, $(du -h "$ROOT/out/${TAG}.csv" | cut -f1))"
echo "    log    out/${TAG}.log"
