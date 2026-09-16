#!/usr/bin/env python3
"""Build an ONNXim request trace from the Azure LLM Inference Dataset.

The dataset is not redistributed here. Download it from Microsoft:

    https://github.com/Azure/AzurePublicDataset
    AzureLLMInferenceTrace_conv.csv     conversation workload
    AzureLLMInferenceTrace_code.csv     code workload

Then:

    python3 scripts/make_workload.py AzureLLMInferenceTrace_conv.csv \
        --n 128 --out traces/az128.csv

which samples N requests and writes one row each:

    time, prompt_length, target_length, cached_length

`cached_length` is the request's ContextTokens, so the run starts mid-conversation
with that much KV already resident; `target_length` is how many tokens to generate
(1 = a single decode step). Also writes example/<name>.json, the model list that
`./run.sh --workload <name>` expects.
"""
import argparse, csv, json, os, random, sys

ap = argparse.ArgumentParser()
ap.add_argument("dataset", help="AzureLLMInferenceTrace_*.csv")
ap.add_argument("--n", type=int, default=128, help="number of requests")
ap.add_argument("--steps", type=int, default=1, help="tokens to generate per request")
ap.add_argument("--min-context", type=int, default=128)
ap.add_argument("--max-context", type=int, default=8192)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--model", default="llama2-7b-1L")
ap.add_argument("--out", required=True, help="traces/<name>.csv")
a = ap.parse_args()

rows = []
with open(a.dataset, newline="") as f:
    for r in csv.DictReader(f):
        try: c = int(r["ContextTokens"])
        except (KeyError, ValueError): continue
        if a.min_context <= c <= a.max_context: rows.append(c)
if len(rows) < a.n:
    sys.exit(f"only {len(rows)} requests in range, need {a.n}")

random.seed(a.seed)
picked = random.sample(rows, a.n)

os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
with open(a.out, "w") as f:
    f.write("time, prompt_length, target_length, cached_length\n")
    for c in picked:
        f.write(f"0, 1, {a.steps}, {c}\n")

name = os.path.splitext(os.path.basename(a.out))[0]
listing = {"models": [{"name": a.model, "trace_file": f"{name}.csv",
                       "scheduler": "simple",
                       "scheduler_config": {"max_batch_size": a.n}}]}
with open(f"example/{name}.json", "w") as f:
    json.dump(listing, f, indent=2)

print(f"{a.out}: {a.n} requests, contexts {min(picked)}-{max(picked)}, "
      f"{sum(picked):,} cached tokens")
print(f"example/{name}.json written  ->  ./run.sh --workload {name}")
