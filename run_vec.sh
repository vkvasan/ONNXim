#!/usr/bin/env bash
cd /workspace/ONNXim/build
mkdir -p /workspace/ONNXim/out/vecsens
run_one() {
  env ONNXIM_WEIGHT_SWIZZLE=1 ONNXIM_KV_BLOCK=16 ONNXIM_KV_LAYOUT="$2" \
      ./bin/Simulator --config ../configs/${1}.json \
        --models_list ../example/${3}.json --mode language --trace_file ${3}.csv \
        > /workspace/ONNXim/out/vecsens/${4}.log 2>&1
  echo "done $4"
}
export -f run_one
xargs -P 6 -n 4 bash -c 'run_one "$0" "$1" "$2" "$3"' < /workspace/ONNXim/jobs_vec.txt
chown -R $(stat -c "%u:%g" /workspace/ONNXim/README.md) /workspace/ONNXim/out/vecsens
echo "ALL DONE"
