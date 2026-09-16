# Speculative decoding on ONNXim — cycle-level closed loop + open-loop trace generator

Scope: vLLM-style speculative decoding (draft model + target verify) for
LLaMA-2 7B, HBM3 16ch, the `_multi_16x32` NPU (16 cores x 32x32), head-parallel
attention, head-major paged KV (block 16), tile-major weights, sequential
intra-tile order. Everything else from FINDINGS.md is held fixed.

Two tools, one model of the step:

| tool | what it gives | cost |
|---|---|---|
| `scheduler: "specdec"` in ONNXim (`src/scheduler/SpecDecScheduler.*`) | cycle-level closed loop: draft x k, verify, acceptance, KV rollback, real arrival cycles, Ramulator2 row hit / BW | ~35-45 min per verify step of 4-8 requests, 1 layer |
| `scripts/specdec_trace.py` | open-loop DRAM address stream of the same step sequence, per-phase counts, write-after-write, plain-decode reference | seconds |
| `scripts/specdec_analyze.py` | splits an ONNXim DRAM trace into (phase, stream, R/W), row hit per stream, WAW, and diffs it against the generator | minutes |

## The step

```
draft0 .. draft(k-1)   draft model, 1 query token per request (2 after a step in
                       which all k draft tokens were accepted: the bonus token)
verify                 target model, k+1 query tokens per request, context L
accept                 a ~ min(Geometric(alpha), k) per request; +a+1 tokens
```

KV bookkeeping follows vLLM: verify writes k+1 lookahead rows L..L+k; the
rows past L+a are stale and get overwritten by the next step. The draft KV is
rolled back the same way. Those overwrites are the `waw` column.

Two scorers, because they differ by an order of magnitude in KV traffic:

- `mqa` (default): one attention call per request with q_len = k+1, the KV
  cache read once. This is vLLM V1 and V0's MQA scorer.
- `expand`: batch expansion, k+1 single-token sequences with contexts
  L..L+k, the cache read k+1 times. vLLM V0's default scorer.

Acceptance can be sampled (alpha) or replayed from a file so both tools see
the same accepted counts. `--emit-workload` writes the file plus the ONNXim
workload for the same requests.

## ONNXim: `scheduler: "specdec"`

```json
{ "models": [ { "name": "llama2-7b-1L", "trace_file": "sd_v4m512.csv",
    "scheduler": "specdec",
    "scheduler_config": {
      "max_batch_size": 128,
      "draft_model": "llama-68m",       // models/language_models/llama-68m.json; omit = verify only
      "spec_k": 4, "spec_alpha": 0.8, "spec_seed": 7,
      "scorer": "mqa",                  // or "expand"
      "spec_disable_batch": 0,          // vLLM speculative_disable_by_batch_size
      "accept_file": "traces/sd_v4m512.accept"   // optional, relative to ONNXIM_HOME
    } } ] }
```

The draft model is registered next to the target (`Simulator::register_language_model`),
its weights allocated right after the target's, so the address-threshold
weight/KV split still works: use `RAMULATOR_WEIGHT_LIMIT=800000000` (target
1-layer block + lm head + draft = 754 MB) instead of the 500 MB used before.

The log carries everything the analyzer needs:

```
[MEM] weights llama2-7b-1L allocated at [0x0,0x27c1d800) (13 tensors)
[SPECDEC] request 0 cached 392 target K 0x2cf0be00+0x14000000 V ... draft K ... V ...
[SPECDEC] launch draft : 4 requests, 4 query tokens, model llama-68m at core cycle 1234
[SPECDEC] finish draft at core cycle 5678 (4444 cycles)
[SPECDEC] verify step 1 done in 28775 cycles, accepted (req:a) 0:2 1:4 ...
[SPECDEC] ===== summary =====   per-phase cycles, accepted/verify, cycles per generated token
```

Run (the checkout is bind-mounted into the `onnxim` container):

```bash
docker exec -d <container> bash -c 'cd /workspace/ONNXim/build && env \
  ONNXIM_WEIGHT_SWIZZLE=1 RAMULATOR_REQBUF=1024 RAMULATOR_WEIGHT_LIMIT=800000000 ONNXIM_DRAM_OCC=1 \
  ONNXIM_KV_BLOCK=16 ONNXIM_KV_LAYOUT=head ONNXIM_KV_WRITES=1 \
  ONNXIM_DRAM_TRACE=/workspace/ONNXim/out/specdec/sd_v4m512.csv \
  ./bin/Simulator --config ../configs/_multi_16x32.json --mode language \
    --models_list ../example/sd_v4m512.json --trace_file sd_v4m512.csv \
  > /workspace/ONNXim/out/specdec/sd_v4m512.log 2>&1'
python3 scripts/specdec_analyze.py out/specdec/sd_v4m512.csv out/specdec/sd_v4m512.log \
    --ref out/specdec/sd_v4m512.ref.json
```

`ONNXIM_KV_WRITES=1` is new and needed: stock ONNXim marks the KV-concat tiles
`skip`, so the cache was never written and no trace could show the overwrite.

Trace-file conventions (`traces/*.csv`, `time, prompt_length, target_length, cached_length`):
rows resident after prefill = cached + prompt_length + 1; a request retires once
rows >= cached + prompt_length + target_length. `--emit-workload` writes
`0, 1, gen, ctx-1` so that the generator's `ctx+1` rows and `gen` tokens match.

## Open loop: `scripts/specdec_trace.py`

```bash
python3 scripts/specdec_trace.py --count-only --requests 8 --k 4 --steps 5          # counts only
python3 scripts/specdec_trace.py --scorer expand --count-only                          # V0 scorer
python3 scripts/specdec_trace.py --k 4 --steps 2 --out sd.csv --json sd.json           # addresses
python3 scripts/specdec_trace.py --emit-workload sd_v4m512 --requests 4 --max-prompt 512 --gen-cap 6
```

Row format `step,phase,stream,request,address,rw,waw`; stream is `weight`,
`kv`, `dweight`, `dkv` (d = draft). Weights are the tile-major walk of
`vllm_trace.py`, repeated once per token tile: with B*(k+1) tokens the GEMM
token dimension can exceed one tile (416 tokens on this config) and GemmWS
re-reads every weight tile per token tile. The tool prints how many walks the
plain and the verify pass take.

Per-step it prints the counts per phase and stream, and at the end the share,
requests per generated token, and the ratio to a plain-decode reference for
the same tokens at the same batch. 8 requests, k=4, alpha=0.8, one target layer:

```
share: target weights 57.1%  target KV 8.7%  draft weights 21.3%  draft KV 12.9%
speculative / plain = 0.435   (target-only: 0.286)
```

i.e. per generated token the target's DRAM traffic drops to 29% of
autoregressive decode (weights amortised over 3.5 tokens/verify), and the
draft model (2 layers, read k times per step) eats back a third of that.

## Fixes to stock ONNXim that this depends on

All three are upstream ONNXim behaviour, not local regressions; they change
KV addresses for every earlier run in FINDINGS.md (see the note there).

1. **`Attention.cc`: every attention head read KV head 0.** `kv_head_idx =
   head_idx / _nkvh` is 0 for all 32 heads of an MHA model, so all cores read
   the same 1/32 of the cache. Now `head_idx / (nh/nkvh)`. Address COUNTS
   were right (which is what the earlier validation checked); the footprint
   was 32x too small.
2. **`Attention.cc`: tiled long contexts re-read chunk 0.** When the context
   exceeds one tile (>2730 tokens at 4 MB spad) the per-tile address used the
   tile-local index, so every tile fetched rows [0, chunk). Now the global row.
   Also the output row of a tiled attention was never written (q index used
   the KV-tile index).
3. **`Attention.cc`: multi-query decode against a tiled context** forced
   q_len = 1 and re-read the cache once per query, i.e. batch expansion by
   accident. Kept only for q_len = 1 (bit-identical); for q_len > 1 the queries
   stay resident and the KV chunk shrinks.
4. **`KVCacheConcat.cc`: writes were skipped, and misplaced.** Tiles are
   `skip` (never executed); when executed (`ONNXIM_KV_WRITES=1`) the write went
   n_new rows past the end of the cache, request b's first token was written
   into request b-1's cache, and the per-tile token loop had no bound
   (`for (...; _inner_loops; ...)`), so the first tile staged every token and
   overflowed the scratchpad at batch 128. All fixed; reads never depended
   on this. With writes on, the staging chunk is half a scratchpad partition.

Single-token decode with contexts <= 2730 differs from before only through
fix 1 (which KV head each core reads). `out/specdec/base_hm16_v8m1024.log`
is the same run as `out/varhead/hm16_v8m1024.log` after the fixes.

## Validation

**Closed loop vs open loop, small models** (`sd_smoke2`: opt-125m target,
llama-68m draft, 2 requests at 64 tokens, k=2, `ONNXIM_KV_WRITES=1`,
`out/specdec/sd_smoke2.{csv,log}` vs `sd_smoke2.ref.json`):

| phase | stream | ONNXim | generator |
|---|---|---|---|
| draft0 / draft1 | draft KV reads | 25,728 / 26,112 | 25,728 / 26,112 |
| draft0 / draft1 | draft KV writes | 384 / 384 | 384 / 384 |
| verify | target KV reads | 13,248 | 13,248 |
| verify | target KV writes | 576 | 576 |
| draft / verify | weight matrices | +0.13% / +0.26% | -- |

Every KV count matches to the request. The weight surplus sits inside the
matrix regions (bias/LayerNorm tensors are split out as `wparam`) and is a
tile-edge alignment effect of ONNXim's address generation that the generator
does not model.

**Baseline regression with the fixes** (`hm16_v8m1024`): see FINDINGS.md §9.
Weight stream and runtime unchanged; KV row hit 96.1% -> 66.7%.

**LLaMA-2 7B, 4 requests (contexts 156-512), k=4, alpha 0.8, single-call
scorer** (`sd_v4m512`, 2 speculative steps, 21 tokens, 15 draft tokens
accepted; `out/specdec/sd_v4m512.analysis.txt`). Totals over both steps,
ONNXim trace vs generator:

| stream | ONNXim | generator | diff |
|---|---|---|---|
| target KV reads | 1,288,704 | 1,288,704 | 0 |
| target KV writes | 15,360 | 15,360 | 0 |
| target KV write-after-write (stale rows) | 3,584 | 3,584 | 0 |
| draft KV reads | 1,921,536 | 1,921,536 | 0 |
| draft KV writes / WAW | 4,608 / 960 | 4,608 / 960 | 0 |
| target weights | 25,359,112 | 25,296,896 | +0.25% |
| draft weights | 9,431,040 | 9,437,184 | -0.07% |

The weight surplus is `make_address_tiled` padding the K dimension to whole
tiles (the padded tail of fc2 lands in the unused lm-head region); the draft
deficit is requests queued in the per-channel buffer that reach DRAM after
the phase's finish cycle. Everything the generator claims to model matches
to the request.

Cycle level (core cycles, `_multi_16x32`, one target layer):

| phase | cycles | note |
|---|---|---|
| plain decode step, 4 requests | 602,301 | 150.6k per token |
| draft pass (llama-68m, 2 layers) | ~106,000 | 38 MB of weights at ~360 GB/s: small GEMMs under-fill 16 cores |
| verify, 4 requests x 5 tokens | 644,461 | +7% over a plain step for 5x the tokens |
| speculative step 0 (4 drafts + verify) | 1,068,000 | 13 tokens: 82k per token, **1.83x** over plain |
| whole run (draft+verify) | 95.9k per token | 1.57x; step 1 ran with 2 requests |

Row hit (arrival order) in the verify: weights 85.1%, target KV 85.0%,
draft KV 92.5%.

**Batch-expansion scorer, same requests and accepted counts**
(`sd_v4m512_expand`, vLLM V0 default): every KV count matches the generator
exactly (reads 6,412,800 = 5x the single-call scorer's 1,288,704; writes and
write-after-write identical), weights +0.25% as above. Verify takes 848,287
cycles for the batch of 4 versus 644,461 single-call, a **32% penalty** for
reading the cache k+1 times; the run costs 111.0k cycles per token (1.36x
over plain) versus 95.9k (1.57x). Target KV share rises from 3.8% to 11.9%
of DRAM requests.

Three-way summary, same 4 requests, one target layer:

| | plain decode | speculative, single-call verify | speculative, batch expansion |
|---|---|---|---|
| cycles per generated token | 151.0k | 95.9k | 111.0k |
| first step, batch of 4 | 150.6k | 82.2k | 97.9k |
| verify (or step) cycles | 602k | 644k | 848k |
| target KV reads per step 0 | 809k | 818k | 4,068k |
| DRAM share: target weights / target KV / draft | 92.2 / 5.9 / -- | 87.3 / 3.8 / 8.0 | 80.0 / 11.9 / 7.7 |

**Plain-decode reference, same requests** (`sd_v4m512_plain`, k=0, no
draft: 5 steps of 4 tokens): KV reads/writes match the generator exactly on
every step, weights +0.03%. Steps take 602-606k cycles each, so plain decode
costs 151k cycles per token here; the whole speculative run costs 95.9k
(1.57x) and its first step, at the same batch of 4, 82k (1.83x). DRAM
share in plain decode: weights 92.2%, KV 5.9%, activations 1.8%; in the
speculative run: target weights 87.3%, target KV 3.8%, draft weights+KV
8.0%, activations 4.9%.
