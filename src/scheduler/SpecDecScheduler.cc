#include "SpecDecScheduler.h"
#include <fstream>

SpecDecScheduler::SpecDecScheduler(std::string name, std::string path,
                                   std::unique_ptr<LanguageModel> model,
                                   std::unique_ptr<LanguageModel> draft,
                                   SimulationConfig config, json sc)
    : LangScheduler(name, path, std::move(model), config, sc) {
  _draft_model = std::move(draft);
  _k = sc.value("spec_k", 4);
  _alpha = sc.value("spec_alpha", 0.8);
  _scorer = sc.value("scorer", std::string("mqa"));
  _disable_batch = sc.value("spec_disable_batch", 0);
  _rng.seed(sc.value("spec_seed", 1));
  if (_scorer != "mqa" && _scorer != "expand") {
    spdlog::error("[SPECDEC] unknown scorer '{}' (mqa|expand)", _scorer);
    throw std::runtime_error("bad scorer");
  }
  if (sc.contains("accept_file")) {
    std::string accept_path = sc["accept_file"];
    if (!accept_path.empty() && accept_path[0] != '/') {
      const char* home = std::getenv("ONNXIM_HOME");
      accept_path = std::string(home ? home : ".") + "/" + accept_path;
    }
    std::ifstream f(accept_path);
    if (!f.is_open()) {
      spdlog::error("[SPECDEC] cannot open accept_file {}", accept_path);
      throw std::runtime_error("accept_file");
    }
    std::string tok;
    while (f >> tok) {
      if (tok.empty() || !isdigit(tok[0])) continue;
      _accept_trace.push_back(std::min((uint32_t)std::stoul(tok), _k));
    }
    spdlog::info("[SPECDEC] accept_file: {} entries", _accept_trace.size());
  }
  if (_draft_model) {
    json dc = _draft_model->get_model_config();
    uint32_t layers = dc["num_hidden_layers"];
    layers /= (uint64_t)dc["pipeline_parallel_size"];
    _draft_sim_layers = _draft_model->is_run_single_layer() ? 1 : layers;
    uint32_t h = dc["hidden_size"], nh = dc["num_attention_heads"], nkvh = dc["num_kv_heads"];
    _draft_cache_dim = h / nh * nkvh;
    _draft_max_dims = {(uint32_t)dc["max_seq_length"], _draft_cache_dim};
  }
  spdlog::info("[SPECDEC] k {} alpha {} scorer {} draft {} disable_batch {} max_batch {}",
               _k, _alpha, _scorer, _draft_model ? _draft_model->get_name() : "none",
               _disable_batch, _max_batch_size);
}

void SpecDecScheduler::init_draft_request(std::unique_ptr<LangRequest>& request) {
  if (!_draft_model) return;
  request->draft_key_cache.resize(_draft_sim_layers);
  request->draft_value_cache.resize(_draft_sim_layers);
  request->draft_length = request->current_length;      /* cached prefix */
  std::vector<uint32_t> first_dims = {request->draft_length, _draft_cache_dim};
  for (uint32_t i = 0; i < _draft_sim_layers; i++) {
    request->draft_key_cache[i] = std::make_unique<Tensor>(
        _draft_model->get_root_node_id(), name_gen(LAYER(i), "DraftKeyCache"),
        _draft_max_dims, _config.precision, true);
    request->draft_key_cache[i]->resize_tensor(first_dims);
    request->draft_value_cache[i] = std::make_unique<Tensor>(
        _draft_model->get_root_node_id(), name_gen(LAYER(i), "DraftValueCache"),
        _draft_max_dims, _config.precision, true);
    request->draft_value_cache[i]->resize_tensor(first_dims);
  }
}

void SpecDecScheduler::resize_caches(LangRequest& r, bool draft, uint32_t rows) {
  if (draft) {
    if (!_draft_model) return;
    r.draft_length = rows;
    std::vector<uint32_t> d = {rows, _draft_cache_dim};
    for (uint32_t i = 0; i < _draft_sim_layers; i++) {
      r.draft_key_cache[i]->resize_tensor(d);
      r.draft_value_cache[i]->resize_tensor(d);
    }
  } else {
    r.current_length = rows;
    std::vector<uint32_t> d = {rows, _cache_dim};
    for (uint32_t i = 0; i < _num_sim_layers; i++) {
      r.key_cache[i]->resize_tensor(d);
      r.value_cache[i]->resize_tensor(d);
    }
  }
}

uint64_t SpecDecScheduler::get_kv_memory_size() {
  uint64_t kv = LangScheduler::get_kv_memory_size();
  if (!_draft_model) return kv;
  uint64_t d = 0;
  for (auto& it : _active_requests)
    for (uint32_t i = 0; i < _draft_sim_layers; i++)
      d += it.second->draft_key_cache[i]->get_size() + it.second->draft_value_cache[i]->get_size();
  if (_draft_model->is_run_single_layer()) d *= _draft_model->get_num_layers();
  return kv + d;
}

void SpecDecScheduler::admit() {
  while (!_request_queue.empty() && _request_queue.front()->request_time <= _cycle) {
    if (_max_batch_size > 0 && _active_requests.size() >= _max_batch_size) break;
    init_request(_request_queue.front());
    init_draft_request(_request_queue.front());
    uint32_t id = _request_queue.front()->request_id;
    {
      /* address ranges, one line per (model, layer), so a DRAM trace can be
         split into streams */
      auto& r = *_request_queue.front();
      uint64_t tspan = (uint64_t)_max_dims[0] * _max_dims[1] * _config.precision;
      for (uint32_t l = 0; l < _num_sim_layers; l++)
        spdlog::info("[SPECDEC] request {} cached {} kv target layer {} K {:#x}+{:#x} V {:#x}+{:#x}",
                     id, r.current_length, l, r.key_cache[l]->get_address(), tspan,
                     r.value_cache[l]->get_address(), tspan);
      if (_draft_model) {
        uint64_t dspan = (uint64_t)_draft_max_dims[0] * _draft_max_dims[1] * _config.precision;
        for (uint32_t l = 0; l < _draft_sim_layers; l++)
          spdlog::info("[SPECDEC] request {} cached {} kv draft layer {} K {:#x}+{:#x} V {:#x}+{:#x}",
                       id, r.current_length, l, r.draft_key_cache[l]->get_address(), dspan,
                       r.draft_value_cache[l]->get_address(), dspan);
      }
    }
    _active_requests[id] = std::move(_request_queue.front());
    _request_queue.pop();
  }
}

void SpecDecScheduler::cycle() {
  _cycle++;
  if (!_model_queue.empty() || !_requests_in_model.empty()) return;   /* a step is in flight */
  if (_next == Phase::FREE) admit();
  launch();
}

void SpecDecScheduler::launch_model(std::unique_ptr<LanguageModel>& lm,
                                    std::vector<LangInput>& inputs, Phase phase) {
  auto m = lm->generate_model(inputs);
  std::set<uint32_t> seen;
  uint32_t num_tokens = 0;
  for (auto& in : inputs) {
    num_tokens += in.seq_length;
    _active_requests[in.request_id]->running = true;
    if (seen.insert(in.request_id).second)
      _requests_in_model[m->get_id()].push_back(in.request_id);
  }
  _phase = phase;
  _phase_start_cycle = _cycle;
  static const char* names[] = {"free", "prefill_target", "prefill_draft", "draft", "verify", "plain"};
  spdlog::info("[SPECDEC] launch {} : {} requests, {} query tokens, model {} at core cycle {}",
               names[(int)phase], seen.size(), num_tokens, lm->get_name(), _cycle);
  _model_queue.push(std::move(m));

  float weight_size = _language_model->get_weight_size() / (1.0 GB)
                    + (_draft_model ? _draft_model->get_weight_size() / (1.0 GB) : 0.0f);
  float kv_size = get_kv_memory_size() / (1.0 GB);
  float act_size = lm->get_act_size() / (1.0 GB) * num_tokens;
  float tot = weight_size + kv_size + act_size;
  if (_config.dram_size < tot && _config.dram_size > 0) {
    if (_check_mem_size) {
      spdlog::error("Memory Usage exceeds the memory size limit {} GB/{} GB", tot, _config.dram_size);
      exit(EXIT_FAILURE);
    }
    spdlog::warn("Memory Usage exceeds the memory size limit {} GB/{} GB", tot, _config.dram_size);
  }
}

void SpecDecScheduler::launch() {
  if (_active_requests.empty()) return;
  std::vector<LangInput> inputs;

  auto target_input = [&](LangRequest& r, uint32_t seq, uint32_t ctx) {
    LangInput in;
    in.request_id = r.request_id; in.seq_length = seq; in.context_length = ctx;
    for (uint32_t i = 0; i < _num_sim_layers; i++) {
      in.key_cache.push_back(r.key_cache[i].get());
      in.value_cache.push_back(r.value_cache[i].get());
    }
    return in;
  };
  auto draft_input = [&](LangRequest& r, uint32_t seq, uint32_t ctx) {
    LangInput in;
    in.request_id = r.request_id; in.seq_length = seq; in.context_length = ctx;
    for (uint32_t i = 0; i < _draft_sim_layers; i++) {
      in.key_cache.push_back(r.draft_key_cache[i].get());
      in.value_cache.push_back(r.draft_value_cache[i].get());
    }
    return in;
  };

  if (_next == Phase::FREE) {
    /* 1. prefill anything newly admitted */
    _batch.clear();
    for (auto& it : _active_requests)
      if (!it.second->gen_phase) _batch.push_back(it.first);
    if (!_batch.empty()) {
      for (auto id : _batch) {
        auto& r = *_active_requests[id];
        inputs.push_back(target_input(r, r.prompt_length, r.current_length));
      }
      _prefills++;
      launch_model(_language_model, inputs, Phase::PREFILL_TARGET);
      return;
    }
    /* 2. a speculative (or plain) step for the resident batch */
    for (auto& it : _active_requests) {
      _batch.push_back(it.first);
      if (_max_batch_size > 0 && _batch.size() >= _max_batch_size) break;
    }
    bool spec = _k > 0 && (_disable_batch == 0 || _batch.size() <= _disable_batch);
    if (!spec) {
      for (auto id : _batch) {
        auto& r = *_active_requests[id];
        inputs.push_back(target_input(r, 1, r.current_length));
      }
      _plain_steps++;
      launch_model(_language_model, inputs, Phase::PLAIN);
      return;
    }
    _draft_step = 0;
    _next = _draft_model ? Phase::DRAFT : Phase::VERIFY;
  }

  if (_next == Phase::PREFILL_DRAFT) {
    for (auto id : _batch) {
      auto& r = *_active_requests[id];
      inputs.push_back(draft_input(r, r.prompt_length, r.draft_length));
    }
    launch_model(_draft_model, inputs, Phase::PREFILL_DRAFT);
    return;
  }

  if (_next == Phase::DRAFT) {
    for (auto id : _batch) {
      auto& r = *_active_requests[id];
      /* bring the draft KV up to target rows + step + 1: normally one query
         token, two right after a fully-accepted step (bonus token) */
      uint32_t want = r.current_length + _draft_step + 1;
      uint32_t seq = want > r.draft_length ? want - r.draft_length : 1;
      inputs.push_back(draft_input(r, seq, r.draft_length));
    }
    _draft_steps++;
    launch_model(_draft_model, inputs, Phase::DRAFT);
    return;
  }

  if (_next == Phase::VERIFY) {
    _expand_tensors.clear();
    for (auto id : _batch) {
      auto& r = *_active_requests[id];
      if (_scorer == "mqa") {
        inputs.push_back(target_input(r, _k + 1, r.current_length));
      } else {
        /* batch expansion: k+1 single-token sequences over the same cache,
           the i-th seeing context L+i and writing row L+i */
        for (uint32_t i = 0; i <= _k; i++) {
          LangInput in;
          in.request_id = id; in.seq_length = 1; in.context_length = r.current_length + i;
          std::vector<uint32_t> d = {r.current_length + i, _cache_dim};
          for (uint32_t l = 0; l < _num_sim_layers; l++) {
            auto kc = std::make_unique<Tensor>(*r.key_cache[l]);
            auto vc = std::make_unique<Tensor>(*r.value_cache[l]);
            kc->reassign_id(); vc->reassign_id();     /* distinct views, same cache */
            kc->resize_tensor(d); vc->resize_tensor(d);
            in.key_cache.push_back(kc.get()); in.value_cache.push_back(vc.get());
            _expand_tensors.push_back(std::move(kc));
            _expand_tensors.push_back(std::move(vc));
          }
          inputs.push_back(in);
        }
      }
    }
    _verify_steps++;
    launch_model(_language_model, inputs, Phase::VERIFY);
    return;
  }
}

uint32_t SpecDecScheduler::sample_accepted() {
  if (!_accept_trace.empty()) {
    uint32_t a = _accept_trace[_accept_pos % _accept_trace.size()];
    _accept_pos++;
    return a;
  }
  std::uniform_real_distribution<double> u(0.0, 1.0);
  uint32_t a = 0;
  while (a < _k && u(_rng) < _alpha) a++;
  return a;
}

void SpecDecScheduler::retire_if_done(uint32_t req_id) {
  auto& r = *_active_requests[req_id];
  if (r.current_length < r.target_length) return;
  r.finish_time = _cycle;
  uint32_t gen = r.current_length - r.prompt_length;
  spdlog::info("Request {} completed in {} cycles", req_id, r.finish_time - r.start_time);
  spdlog::info("[SPECDEC] request {} : {} verify steps, {} draft tokens accepted "
               "({:.2f} accepted/step, {:.2f} tokens/step), final length {}",
               req_id, r.spec_steps, r.spec_accepted,
               r.spec_steps ? (double)r.spec_accepted / r.spec_steps : 0.0,
               r.spec_steps ? (double)(r.spec_accepted + r.spec_steps) / r.spec_steps : 0.0,
               r.current_length);
  (void)gen;
  _active_requests.erase(req_id);
}

void SpecDecScheduler::finish_model(uint32_t model_id) {
  std::vector<uint32_t> ids = _requests_in_model[model_id];
  _requests_in_model.erase(model_id);
  uint64_t took = _cycle - _phase_start_cycle;
  static const char* names[] = {"free", "prefill_target", "prefill_draft", "draft", "verify", "plain"};
  spdlog::info("[SPECDEC] finish {} at core cycle {} ({} cycles)", names[(int)_phase], _cycle, took);
  for (auto id : ids) _active_requests[id]->running = false;

  switch (_phase) {
    case Phase::PREFILL_TARGET:
      _cycles_prefill += took;
      for (auto id : ids) {
        auto& r = *_active_requests[id];
        /* stock convention (LanguageScheduler::finish_model): rows = cached + prompt + 1 */
        resize_caches(r, false, r.current_length + r.prompt_length + 1);
      }
      if (_draft_model) {
        _next = Phase::PREFILL_DRAFT;
      } else {
        for (auto id : ids) { _active_requests[id]->gen_phase = true; retire_if_done(id); }
        _next = Phase::FREE;
      }
      break;

    case Phase::PREFILL_DRAFT:
      _cycles_prefill += took;
      for (auto id : ids) {
        auto& r = *_active_requests[id];
        resize_caches(r, true, r.current_length);      /* draft rows == target rows */
        r.gen_phase = true;
        retire_if_done(id);
      }
      _next = Phase::FREE;
      break;

    case Phase::DRAFT:
      _cycles_draft += took;
      for (auto id : ids) {
        auto& r = *_active_requests[id];
        resize_caches(r, true, r.current_length + _draft_step + 1);
      }
      _draft_step++;
      _next = _draft_step < _k ? Phase::DRAFT : Phase::VERIFY;
      break;

    case Phase::VERIFY: {
      _cycles_verify += took;
      _expand_tensors.clear();
      std::string acc_log;
      for (auto id : ids) {
        auto& r = *_active_requests[id];
        uint32_t a = sample_accepted();
        r.spec_steps++; r.spec_accepted += a;
        _accepted_total += a; _tokens_total += a + 1;
        uint32_t new_rows = r.current_length + a + 1;
        resize_caches(r, false, new_rows);
        /* draft rows past the accepted prefix are stale too */
        if (_draft_model && r.draft_length > new_rows) resize_caches(r, true, new_rows);
        acc_log += fmt::format("{}:{} ", id, a);
      }
      spdlog::info("[SPECDEC] verify step {} done in {} cycles, accepted (req:a) {}",
                   _verify_steps, took, acc_log);
      for (auto id : ids) retire_if_done(id);
      _next = Phase::FREE;
      break;
    }

    case Phase::PLAIN:
      _cycles_plain += took;
      for (auto id : ids) {
        auto& r = *_active_requests[id];
        resize_caches(r, false, r.current_length + 1);
        _tokens_total += 1;
      }
      for (auto id : ids) retire_if_done(id);
      _next = Phase::FREE;
      break;

    case Phase::FREE:
      break;
  }
  _phase = Phase::FREE;
  if (_active_requests.empty() && _request_queue.empty()) log_summary();
}

void SpecDecScheduler::log_summary() {
  spdlog::info("[SPECDEC] ===== summary =====");
  spdlog::info("[SPECDEC] prefills {} ({} cycles) | draft steps {} ({} cycles) | verify steps {} "
               "({} cycles) | plain steps {} ({} cycles)",
               _prefills, _cycles_prefill, _draft_steps, _cycles_draft,
               _verify_steps, _cycles_verify, _plain_steps, _cycles_plain);
  spdlog::info("[SPECDEC] tokens generated {} | draft tokens accepted {} | "
               "accepted per verify {:.3f} | tokens per verify {:.3f}",
               _tokens_total, _accepted_total,
               _verify_steps ? (double)_accepted_total / _verify_steps : 0.0,
               _verify_steps ? (double)_tokens_total / _verify_steps : 0.0);
  uint64_t spec_cycles = _cycles_draft + _cycles_verify;
  if (_tokens_total)
    spdlog::info("[SPECDEC] cycles per generated token: {:.1f} (draft+verify), {:.1f} (all phases)",
                 (double)spec_cycles / _tokens_total,
                 (double)(spec_cycles + _cycles_prefill + _cycles_plain) / _tokens_total);
}
