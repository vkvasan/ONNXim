#!/usr/bin/env bash
cd /workspace/ONNXim/build
run(){ env ONNXIM_WEIGHT_SWIZZLE=1 RAMULATOR_REQBUF=1024 RAMULATOR_WEIGHT_LIMIT=500000000 \
       ONNXIM_DRAM_OCC=1 ONNXIM_PAR_STRATEGY=$2 $3 ./bin/Simulator \
       --config ../configs/_multi_16x32.json --models_list ../example/v8m1024.json \
       --mode language --trace_file v8m1024.csv > /workspace/ONNXim/out/parstrat/$1.log 2>&1; }
for st in head request seq; do
  run ct_$st  $st ""                                        &
  run hm_$st  $st "ONNXIM_KV_BLOCK=16 ONNXIM_KV_LAYOUT=head" &
  run bm_$st  $st "ONNXIM_KV_BLOCK=16"                       &
done
wait
chown -R $(stat -c "%u:%g" /workspace/ONNXim/README.md) /workspace/ONNXim/out/parstrat
