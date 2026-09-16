#!/usr/bin/env bash
cd /workspace/ONNXim/build
run() { env ONNXIM_WEIGHT_SWIZZLE=1 RAMULATOR_REQBUF=1024 RAMULATOR_WEIGHT_LIMIT=500000000 \
        ONNXIM_DRAM_OCC=1 $2 ./bin/Simulator --config ../configs/_multi_16x32.json \
        --models_list ../example/v8m1024.json --mode language --trace_file v8m1024.csv \
        > /workspace/ONNXim/out/sortchk/$1.log 2>&1; }
run ct_sorted   ""                                                       &
run ct_unsorted "ONNXIM_KV_UNSORTED=1"                                   &
run bm_sorted   "ONNXIM_KV_BLOCK=16"                                     &
run bm_unsorted "ONNXIM_KV_BLOCK=16 ONNXIM_KV_UNSORTED=1"                &
run bmseq_sorted   "ONNXIM_KV_BLOCK=16 ONNXIM_KV_ALLOC=seq"              &
run bmseq_unsorted "ONNXIM_KV_BLOCK=16 ONNXIM_KV_ALLOC=seq ONNXIM_KV_UNSORTED=1" &
wait
chown -R $(stat -c "%u:%g" /workspace/ONNXim/README.md) /workspace/ONNXim/out/sortchk
