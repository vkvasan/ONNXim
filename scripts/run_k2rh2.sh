#!/usr/bin/env bash
set -u
cd /workspace/ONNXim/build; mkdir -p /workspace/ONNXim/out/audit
one(){ tag=$1; shift
  env ONNXIM_WEIGHT_SWIZZLE=1 RAMULATOR_REQBUF=1024 RAMULATOR_WEIGHT_LIMIT=500000000 \
      ONNXIM_KV_WRITES=1 ONNXIM_ICNT_PORT_BY_CHANNEL=1 ONNXIM_DRAM_OCC=1 \
      ONNXIM_KV_LAYOUT=headbank ONNXIM_KV_BLOCK=64 ONNXIM_KV_BANKS_PER_HEAD=2 "$@" \
      ./bin/Simulator --config ../configs/_c128.json \
        --models_list ../example/az128.json --mode language --trace_file az128.csv \
        > /workspace/ONNXim/out/audit/$tag.log 2>&1; echo "done $tag"; }
one k2_rowhit      RAMULATOR_SCHED_ROWHIT=1 &
one k2_rowhit_cap  RAMULATOR_SCHED_ROWHIT=1 RAMULATOR_STARVE_CAP=2000 &
# K=1 with the row-hit tier at pg64 + audit, as the matched control
one k1_rowhit      RAMULATOR_SCHED_ROWHIT=1 ONNXIM_KV_BANKS_PER_HEAD=1 &
wait
chown -R $(stat -c "%u:%g" /workspace/ONNXim/README.md) /workspace/ONNXim/out/audit
echo K2RH_DONE
