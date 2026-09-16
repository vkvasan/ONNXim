#!/usr/bin/env bash
# Full axis sweep: KV layout x parallelism x weight layout.
#   ./scripts/run_sweep.sh <workload> <outdir> [trace]
# "trace" as 3rd arg also emits the DRAM address trace per run (large).
set -u
WL=${1:-v8m1024}; OUT=${2:-sweep}; TRACE=${3:-}
cd /workspace/ONNXim/build
mkdir -p /workspace/ONNXim/out/$OUT

run_one() {
  local kv=$1 par=$2 swz=$3 tag="${1}_${2}_w${3}"
  local extra=""
  [ -n "${TRACE:-}" ] && extra="ONNXIM_DRAM_TRACE=/workspace/ONNXim/out/$OUT/$tag.csv ONNXIM_DRAM_TRACE_LIMIT=30000000"
  env ONNXIM_KV_LAYOUT=$kv ONNXIM_PAR_STRATEGY=$par ONNXIM_WEIGHT_SWIZZLE=$swz \
      ONNXIM_KV_BLOCK=16 ONNXIM_DRAM_OCC=1 $extra \
      ./bin/Simulator --config ../configs/_c128.json \
        --models_list ../example/$WL.json --mode language --trace_file $WL.csv \
        > /workspace/ONNXim/out/$OUT/$tag.log 2>&1
  echo "done $tag"
}
export -f run_one; export WL OUT TRACE

for kv in head block; do for par in head request seq; do for swz in 1 0; do
  echo "$kv $par $swz"
done; done; done | xargs -P 6 -n 3 bash -c 'run_one "$0" "$1" "$2"'

chown -R $(stat -c "%u:%g" /workspace/ONNXim/README.md) /workspace/ONNXim/out/$OUT
echo "SWEEP DONE"
