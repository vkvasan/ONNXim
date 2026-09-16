#!/usr/bin/env bash
cd /workspace/ONNXim/build
for t in v8m256 v32m256 v8m4096; do
  ( env ONNXIM_WEIGHT_SWIZZLE=1 RAMULATOR_REQBUF=1024 RAMULATOR_WEIGHT_LIMIT=500000000 ONNXIM_DRAM_OCC=1 \
      ./bin/Simulator --config ../configs/_multi_16x32.json --models_list ../example/$t.json \
        --mode language --trace_file $t.csv > /workspace/ONNXim/out/varsweep/$t.log 2>&1 ) &
done
wait
chown -R $(stat -c "%u:%g" /workspace/ONNXim/README.md) /workspace/ONNXim/out/varsweep
