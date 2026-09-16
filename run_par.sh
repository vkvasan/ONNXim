#!/usr/bin/env bash
cd /workspace/ONNXim/build
mkdir -p /workspace/ONNXim/out/par
for n in 1 2 4 8 16; do
  ( env ONNXIM_WEIGHT_SWIZZLE=1 RAMULATOR_REQBUF=1024 RAMULATOR_WEIGHT_LIMIT=500000000 \
      ONNXIM_DRAM_OCC=1 ONNXIM_DRAM_TRACE=/workspace/ONNXim/out/par/p${n}c.csv \
      ONNXIM_DRAM_TRACE_LIMIT=15000000 \
      ./bin/Simulator --config ../configs/_par_${n}c.json --models_list ../example/v8m1024.json \
        --mode language --trace_file v8m1024.csv > /workspace/ONNXim/out/par/p${n}c.log 2>&1 ) &
done
wait
chown -R $(stat -c "%u:%g" /workspace/ONNXim/README.md) /workspace/ONNXim/out/par
