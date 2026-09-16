#ifndef SPECDEC_SCHEDULER_H
#define SPECDEC_SCHEDULER_H
#include <random>
#include <set>
#include "LanguageScheduler.h"

/* Speculative decoding, vLLM-style, as a closed loop inside ONNXim.
 *
 * One speculative step for the resident batch is
 *
 *     draft step 0 .. k-1   draft model, 1 query token each (2 after a step
 *                           in which every draft token was accepted, because
 *                           the bonus token has not been seen by the draft)
 *     verify                target model, k+1 query tokens against the
 *                           target KV cache
 *     accept                a ~ min(Geometric, k) tokens accepted, sampled
 *                           per request with per-token probability alpha;
 *                           the request advances by a+1 tokens
 *
 * KV bookkeeping follows vLLM: k+1 lookahead rows are written by verify, the
 * rows past the accepted prefix are stale and are overwritten by the next
 * step, and the draft KV is rolled back the same way.
 *
 * models_list scheduler_config keys
 *     max_batch_size     as for the other schedulers
 *     draft_model        name in models/language_models (omit: draft compute
 *                        and draft KV are NOT simulated; verify + accept only)
 *     spec_k             draft tokens per step (default 4; 0 = plain decode)
 *     spec_alpha         per-token acceptance probability (default 0.8)
 *     spec_seed          RNG seed for acceptance sampling (default 1)
 *     accept_file        optional: text file of accepted counts, consumed one
 *                        per (verify step, request) in batch order; overrides
 *                        the geometric model, wraps around at EOF
 *     scorer             "mqa" (default): one attention call with q_len=k+1,
 *                        KV read once (vLLM V1 / V0 MQA scorer)
 *                        "expand": batch expansion, k+1 single-token
 *                        sequences with contexts L..L+k, KV read k+1 times
 *                        (vLLM V0 default scorer)
 *     spec_disable_batch batch size above which the step is plain decode
 *                        (vLLM speculative_disable_by_batch_size; 0 = never)
 *
 * Admission is iteration-level (a request joins as soon as there is room);
 * newly admitted requests are prefilled -- on the target and then on the
 * draft -- before the next speculative step.
 */
class SpecDecScheduler : public LangScheduler {
  public:
    SpecDecScheduler(std::string name, std::string path,
                     std::unique_ptr<LanguageModel> model,
                     std::unique_ptr<LanguageModel> draft,
                     SimulationConfig config, json scheduler_config);
    void cycle() override;
    void finish_model(uint32_t model_id) override;
    uint64_t get_kv_memory_size() override;

  private:
    enum class Phase { FREE, PREFILL_TARGET, PREFILL_DRAFT, DRAFT, VERIFY, PLAIN };

    std::unique_ptr<LanguageModel> _draft_model;
    uint32_t _draft_sim_layers = 0;
    uint32_t _draft_cache_dim = 0;
    std::vector<uint32_t> _draft_max_dims;

    uint32_t _k;
    double _alpha;
    std::string _scorer;
    uint32_t _disable_batch;
    std::mt19937_64 _rng;
    std::vector<uint32_t> _accept_trace;
    size_t _accept_pos = 0;

    Phase _phase = Phase::FREE;          /* what is in flight */
    Phase _next = Phase::FREE;           /* what to launch when idle */
    std::vector<uint32_t> _batch;        /* requests in the current step */
    uint32_t _draft_step = 0;
    std::vector<std::unique_ptr<Tensor>> _expand_tensors;

    /* run statistics */
    uint64_t _verify_steps = 0, _draft_steps = 0, _plain_steps = 0, _prefills = 0;
    uint64_t _accepted_total = 0, _tokens_total = 0;
    uint64_t _phase_start_cycle = 0;
    uint64_t _cycles_draft = 0, _cycles_verify = 0, _cycles_prefill = 0, _cycles_plain = 0;

    void admit();
    void launch();
    void init_draft_request(std::unique_ptr<LangRequest>& request);
    void resize_caches(LangRequest& r, bool draft, uint32_t rows);
    uint32_t sample_accepted();
    void launch_model(std::unique_ptr<LanguageModel>& lm, std::vector<LangInput>& inputs, Phase phase);
    void retire_if_done(uint32_t req_id);
    void log_summary();
};

#endif
