#!/usr/bin/env bash
cd /workspace/ONNXim/build
mkdir -p /workspace/ONNXim/out/varpaged
run_one() {
  local kv="$1" model="$2" out="$3"
  env ONNXIM_WEIGHT_SWIZZLE=1 RAMULATOR_REQBUF=1024 RAMULATOR_WEIGHT_LIMIT=500000000 \
      ONNXIM_DRAM_OCC=1 $kv \
      ./bin/Simulator --config ../configs/_multi_16x32.json \
        --models_list ../example/${model}.json --mode language --trace_file ${model}.csv \
        > /workspace/ONNXim/out/varpaged/${out}.log 2>&1
}
export -f run_one
xargs -P 14 -n 3 bash -c 'run_one "$0" "$1" "$2"' < /workspace/ONNXim/jobs_paged.txt
chown -R $(stat -c "%u:%g" /workspace/ONNXim/README.md) /workspace/ONNXim/out/varpaged
