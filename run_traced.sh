#!/usr/bin/env bash
cd /workspace/ONNXim/build
mkdir -p /workspace/ONNXim/out/tr
run_one() {
  local blk="$1" lay="$2" model="$3" out="$4" extra=""
  [ "$blk" != "0" ] && extra="ONNXIM_KV_BLOCK=$blk"
  [ "$lay" = "head" ] && extra="$extra ONNXIM_KV_LAYOUT=head"
  env ONNXIM_WEIGHT_SWIZZLE=1 RAMULATOR_REQBUF=1024 RAMULATOR_WEIGHT_LIMIT=500000000 \
      ONNXIM_DRAM_OCC=1 ONNXIM_DRAM_TRACE=/workspace/ONNXim/out/tr/${out}.csv $extra \
      ./bin/Simulator --config ../configs/_multi_16x32.json \
        --models_list ../example/${model}.json --mode language --trace_file ${model}.csv \
        > /workspace/ONNXim/out/tr/${out}.log 2>&1
}
export -f run_one
date "+WAVE1 start %F %T" >> /workspace/ONNXim/out/tr/progress.txt
xargs -P 12 -n 4 bash -c 'run_one "$0" "$1" "$2" "$3"' < /workspace/ONNXim/jobs_tr1.txt
date "+WAVE1 done  %F %T" >> /workspace/ONNXim/out/tr/progress.txt
xargs -P 8 -n 4 bash -c 'run_one "$0" "$1" "$2" "$3"' < /workspace/ONNXim/jobs_tr2.txt
date "+WAVE2 done  %F %T" >> /workspace/ONNXim/out/tr/progress.txt
chown -R $(stat -c "%u:%g" /workspace/ONNXim/README.md) /workspace/ONNXim/out/tr
