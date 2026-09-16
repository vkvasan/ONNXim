#!/usr/bin/env bash
set -u
cd /workspace/ONNXim/build; mkdir -p /workspace/ONNXim/out/rowhit
one(){ tag=$1; shift
  env ONNXIM_WEIGHT_SWIZZLE=1 RAMULATOR_REQBUF=1024 RAMULATOR_WEIGHT_LIMIT=500000000 \
      ONNXIM_KV_WRITES=1 ONNXIM_ICNT_PORT_BY_CHANNEL=1 ONNXIM_DRAM_OCC=1 \
      ONNXIM_KV_LAYOUT=block "$@" \
      ./bin/Simulator --config ../configs/_c128.json \
        --models_list ../example/az128.json --mode language --trace_file az128.csv \
        > /workspace/ONNXim/out/rowhit/$tag.log 2>&1; echo "done $tag"; }
one rh_block_pg16 RAMULATOR_SCHED_ROWHIT=1 ONNXIM_KV_BLOCK=16 &
one rh_block_pg64 RAMULATOR_SCHED_ROWHIT=1 ONNXIM_KV_BLOCK=64 &
wait
chown -R $(stat -c "%u:%g" /workspace/ONNXim/README.md) /workspace/ONNXim/out/rowhit
echo RH_BLOCK_DONE
