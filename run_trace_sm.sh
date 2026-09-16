#!/usr/bin/env bash
cd /workspace/ONNXim/build
mkdir -p /workspace/ONNXim/out/smdip
for v in vbase vreal; do
  env ONNXIM_WEIGHT_SWIZZLE=1 ONNXIM_KV_BLOCK=16 ONNXIM_KV_LAYOUT=head \
      ONNXIM_DRAM_TRACE=/workspace/ONNXim/out/smdip/${v}.csv \
      ONNXIM_DRAM_TRACE_LIMIT=30000000 \
      ./bin/Simulator --config ../configs/_1c128_${v}.json \
        --models_list ../example/v8m1024.json --mode language --trace_file v8m1024.csv \
        > /workspace/ONNXim/out/smdip/${v}.log 2>&1
  echo "done $v"
done
chown -R $(stat -c "%u:%g" /workspace/ONNXim/README.md) /workspace/ONNXim/out/smdip
echo "SMDIP DONE"
