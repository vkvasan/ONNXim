#!/usr/bin/env bash
cd /workspace/ONNXim/build
env ONNXIM_WEIGHT_SWIZZLE=1 ONNXIM_KV_BLOCK=16 ONNXIM_KV_LAYOUT=block \
    ONNXIM_DRAM_TRACE=/workspace/ONNXim/out/smdip/vbase_block.csv \
    ONNXIM_DRAM_TRACE_LIMIT=30000000 \
    ./bin/Simulator --config ../configs/_1c128_vbase.json \
      --models_list ../example/v8m1024.json --mode language --trace_file v8m1024.csv \
      > /workspace/ONNXim/out/smdip/vbase_block.log 2>&1
chown -R $(stat -c "%u:%g" /workspace/ONNXim/README.md) /workspace/ONNXim/out/smdip
echo "BLOCK TRACE DONE"
