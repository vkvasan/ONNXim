#!/usr/bin/env bash
cd /workspace/ONNXim/build
mkdir -p /workspace/ONNXim/out/varhead
run_one() {
  env ONNXIM_WEIGHT_SWIZZLE=1 RAMULATOR_REQBUF=1024 RAMULATOR_WEIGHT_LIMIT=500000000 \
      ONNXIM_DRAM_OCC=1 ONNXIM_KV_BLOCK="$1" ONNXIM_KV_LAYOUT="$2" \
      ./bin/Simulator --config ../configs/_multi_16x32.json \
        --models_list ../example/${3}.json --mode language --trace_file ${3}.csv \
        > /workspace/ONNXim/out/varhead/${4}.log 2>&1
}
export -f run_one
xargs -P 10 -n 4 bash -c 'run_one "$0" "$1" "$2" "$3"' < /workspace/ONNXim/jobs_head.txt
chown -R $(stat -c "%u:%g" /workspace/ONNXim/README.md) /workspace/ONNXim/out/varhead
