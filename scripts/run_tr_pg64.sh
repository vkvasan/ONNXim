#!/usr/bin/env bash
set -u
cd /workspace/ONNXim/build; mkdir -p /workspace/ONNXim/out/tr_pg64
while pgrep -f "bin/Simulator --config" >/dev/null; do sleep 60; done
env ONNXIM_WEIGHT_SWIZZLE=1 RAMULATOR_REQBUF=1024 RAMULATOR_WEIGHT_LIMIT=500000000 \
    ONNXIM_KV_WRITES=1 ONNXIM_ICNT_PORT_BY_CHANNEL=1 ONNXIM_DRAM_OCC=1 \
    ONNXIM_KV_LAYOUT=headbank ONNXIM_KV_BANKS_PER_HEAD=1 ONNXIM_KV_BLOCK=64 \
    ONNXIM_DRAM_TRACE=/workspace/ONNXim/out/tr_pg64/k1_pg64.csv \
    ONNXIM_DRAM_TRACE_LIMIT=40000000 \
    ./bin/Simulator --config ../configs/_c128.json \
      --models_list ../example/az128.json --mode language --trace_file az128.csv \
      > /workspace/ONNXim/out/tr_pg64/k1_pg64.log 2>&1
chown -R $(stat -c "%u:%g" /workspace/ONNXim/README.md) /workspace/ONNXim/out/tr_pg64
echo TR_PG64_DONE
