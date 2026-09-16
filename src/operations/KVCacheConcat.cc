#include "KVCacheConcat.h"
#include <cstdlib>
#include "../Model.h"
// const scalar_t* __restrict__ q,       // [num_seqs, num_heads, head_size]
// const cache_t* __restrict__ k_cache,  // [num_blocks, num_kv_heads,
//                                       // head_size/x, block_size, x]
// const cache_t* __restrict__ v_cache,  // [num_blocks, num_kv_heads,
//                                       // head_size, block_size]
// const int num_kv_heads,               // [num_heads]
KVCacheConcat::KVCacheConcat(SimulationConfig config, Model* model,
                             onnx::NodeProto& node_proto, uint32_t target_core)
    : Operation(config, model, node_proto, target_core) {
  spdlog::error("KVCacheConcat: Not implemented");
  throw std::runtime_error("KVCacheConcat: Not implemented");
}

KVCacheConcat::KVCacheConcat(const KVCacheConcat& src) : Operation(src) {
  spdlog::error("KVCacheConcat: Not implemented");
  throw std::runtime_error("KVCacheConcat: Not implemented");
}

KVCacheConcat::KVCacheConcat(SimulationConfig config, Model* model,
                             std::string name,
                             std::map<std::string, std::string>& attributes, uint32_t target_core)
    : Operation(config, model, name, attributes, target_core) {
  _input_token_lengths = parse_dims(get_attribute("input_token_lengths"));
  _num_kv_heads = std::stoi(get_attribute("num_kv_heads"));
  _num_attention_heads = std::stoi(get_attribute("num_heads"));
  _hidden_size = std::stoi(get_attribute("hidden_size"));
  _num_batches = _input_token_lengths.size();
  _cache_dim = _hidden_size / _num_attention_heads * _num_kv_heads;
  
  spdlog::debug("[KVCacheConcat] input_token_lengths: {}",
                _input_token_lengths);
  for (int batch = 0; batch < _num_batches; batch++) {
    std::vector<uint32_t> query_dim = {_input_token_lengths[batch], _hidden_size};
    auto query_out = std::make_unique<Tensor>(
        _id, name_gen(std::to_string(_id), "QueryOut", std::to_string(batch)),
        query_dim, _config.precision, false);
    //make temporal tensor for key and value
    auto key_out = std::make_unique<Tensor>(
        _id, name_gen(std::to_string(_id), "KeyOut", std::to_string(batch)),
        _config.precision);
    auto value_out = std::make_unique<Tensor>(
        _id, name_gen(std::to_string(_id), "ValueOut", std::to_string(batch)),
        _config.precision);
    _outputs.push_back(query_out.get()->get_id());
    _model->add_tensor(std::move(query_out));
    _outputs.push_back(key_out.get()->get_id());
    _model->add_tensor(std::move(key_out));
    _outputs.push_back(value_out.get()->get_id());
    _model->add_tensor(std::move(value_out));
  }
}

void KVCacheConcat::initialize_tiles(MappingTable& mapping_table) {
  auto qkv_out_tensor = _model->get_tensor(_inputs[0]);
  for(int batch = 0; batch < _num_batches; batch++){
    uint32_t key_tensor_id = _outputs[batch * 3 + 1];
    uint32_t value_tensor_id = _outputs[batch * 3 + 2];
    auto key_cache = _model->get_tensor(_inputs[batch * 2 + 1]);
    auto key_dims = key_cache->get_dims();
    key_dims[0] = key_dims[0] + _input_token_lengths[batch];
    _model->get_tensor(key_tensor_id)->define_tensor(key_cache->get_address(), key_dims); 
    auto value_cache = _model->get_tensor(_inputs[batch * 2 + 2]);
    auto value_dims = value_cache->get_dims();
    value_dims[0] = value_dims[0] + _input_token_lengths[batch];
    _model->get_tensor(value_tensor_id)->define_tensor(value_cache->get_address(), value_dims);
  }
  calculate_loops();
  /* Stock ONNXim marks these tiles `skip`: Core::push_tile() retires a skip
     tile without executing it, so the KV cache is NEVER written to DRAM and
     the only writes a decode/prefill trace shows are activations.
     ONNXIM_KV_WRITES=1 executes them (MOVIN of the QKV output, MOVOUT of the
     query and of the new K/V rows), which is what vLLM's reshape_and_cache
     kernel does. Needed to observe the stale-row overwrite (write-after-write)
     of speculative decoding. Default stays stock so baselines are unchanged. */
  const bool kv_writes = kv_writes_enabled();
  for(int outter = 0; outter < _outter_loops; outter++) {
    _tiles.push_back(std::make_unique<Tile>(Tile{.status = Tile::Status::INITIALIZED,
                      .optype = "KVCacheConcat",
                      .layer_id = _id,
                      .skip = !kv_writes}));
    initialize_instructions(_tiles.back().get(), outter);
  }
}

bool KVCacheConcat::kv_writes_enabled() {
  static const bool e = [] {
    const char* v = std::getenv("ONNXIM_KV_WRITES");
    return v && v[0] == '1';
  }();
  return e;
}

void KVCacheConcat::calculate_loops() {
  uint32_t per_token_size = _config.precision * ( _cache_dim * 2 + _hidden_size);
  /* Stock sizes the chunk to half the scratchpad, which is fine for a tile
     that is never executed. When the tiles DO run (ONNXIM_KV_WRITES=1) the
     MOVIN must fit one scratchpad partition (spad / tile_depth), so use half
     a partition; 128 requests x 24 KB of QKV output otherwise panics the
     core ("MVIN issue panic"). Skip-tile counts stay as before. */
  uint32_t chunk = kv_writes_enabled()
      ? (_config.core_config[target_core].spad_size KB) / _config.tile_depth / 2
      : (_config.core_config[target_core].spad_size KB) / 2;
  _outter_loops =  ceil_div(get_input(0)->get_size(), chunk);
  _inner_loops = ceil_div(chunk, per_token_size);
  spdlog::debug("[KVCacheConcat] number of tiles: {}", _outter_loops);
}

void KVCacheConcat::initialize_instructions(Tile* tile, uint32_t idx) {
  uint32_t per_token_size = _config.precision * ( _cache_dim * 2 + _hidden_size);
  std::set<addr_type> movin_addresses;
  std::set<addr_type> query_out_address;
  std::vector<std::set<addr_type>> key_out_addresses;
  std::vector<std::set<addr_type>> value_out_addresses;
  key_out_addresses.resize(_num_batches);
  value_out_addresses.resize(_num_batches);

  int currenet_batch = 0;
  int current_index = 0;
  /* FIX (upstream bug): the condition was `_inner_loops` (always true), so the
     first tile staged EVERY token and its MOVIN exceeded the scratchpad as soon
     as the tiles were executed with a real batch. */
  for(int inner = 0; inner < (int)_inner_loops; inner++) {
    int token_id = idx * _inner_loops + inner;
    if(token_id >= get_input(0)->get_dims()[0]) {
      break;
    }
    addr_type qkv_out_addr = get_input(0)->get_address() + token_id * per_token_size;
    addr_type query_out_addr = get_output(0)->get_address() + token_id * _hidden_size * _config.precision;
    for(addr_type offset = 0; offset < per_token_size; offset += _config.dram_req_size) {
      movin_addresses.insert(_config.align_address(qkv_out_addr + offset));
    }
    for(addr_type offset = 0; offset < _hidden_size * _config.precision; offset += _config.dram_req_size) {
      query_out_address.insert(_config.align_address(query_out_addr + offset));
    }
    /* Outputs are registered per batch as {query, key, value} -- LanguageModel.cc:245-247.
       value_out_address used to derive from key_cache_tensor, so every V write landed on
       the K cache's address and the V region was never written. Reads were always correct
       (Attention.cc uses separate key/value bases), so this affected WRITE traffic only:
       negligible in decode (1 token written per step) but material for prefill, which
       writes N tokens of KV. */
    /* FIX (upstream bugs), write placement only -- reads were never affected:
       1. the batch advanced AFTER the address was formed, so the first token of
          request b>0 was written into request b-1's cache;
       2. the offset used the OUTPUT tensor's size, which already includes the
          new tokens, so every write landed n_new rows past the end of the
          cache instead of at the first free row. With speculative decoding
          n_new = k+1 and the stale-row overwrite (write-after-write) is one of
          the things being measured, so the rows must be the real ones. */
    /* (batch, row) come from the GLOBAL token id, not a per-tile counter,
       so the mapping also holds for the second and later tiles of a
       multi-tile prefill. */
    currenet_batch = 0;
    current_index = token_id;
    while(currenet_batch < _num_batches - 1 &&
          current_index >= (int)_input_token_lengths[currenet_batch]) {
      current_index -= _input_token_lengths[currenet_batch];
      currenet_batch++;
    }
    auto key_cache_in = _model->get_tensor(_inputs[currenet_batch * 2 + 1]);
    auto value_cache_in = _model->get_tensor(_inputs[currenet_batch * 2 + 2]);
    addr_type key_out_address = key_cache_in->get_address() + key_cache_in->get_size() +
      current_index * _cache_dim * _config.precision;
    addr_type value_out_address = value_cache_in->get_address() + value_cache_in->get_size() +
      current_index * _cache_dim * _config.precision;
    for(addr_type offset = 0; offset < _cache_dim * _config.precision; offset += _config.dram_req_size) {
      key_out_addresses[currenet_batch].insert(_config.align_address(key_out_address + offset));
      value_out_addresses[currenet_batch].insert(_config.align_address(value_out_address + offset));
    }
  }
  
  tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
    .opcode = Opcode::MOVIN,
    .dest_addr = SPAD_BASE,
    .size = (uint32_t)movin_addresses.size(),
    .src_addrs = std::vector<addr_type>(movin_addresses.begin(), movin_addresses.end()),
    .operand_id = _INPUT_OPERAND
  }));
  tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
    .opcode = Opcode::MOVOUT,
    .dest_addr = SPAD_BASE,
    .size = (uint32_t)query_out_address.size(),
    .src_addrs = std::vector<addr_type>(query_out_address.begin(), query_out_address.end()),
    .operand_id = _OUTPUT_OPERAND
  }));
  for(int batch = 0; batch < _num_batches; batch++) {
    tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
      .opcode = Opcode::MOVOUT,
      .dest_addr = SPAD_BASE,
      .size = (uint32_t)key_out_addresses[batch].size(),
      .src_addrs = std::vector<addr_type>(key_out_addresses[batch].begin(), key_out_addresses[batch].end()),
      .operand_id = _OUTPUT_OPERAND
    }));
  }
  for(int batch = 0; batch < _num_batches; batch++){
    tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
      .opcode = Opcode::MOVOUT,
      .dest_addr = SPAD_BASE,
      .size = (uint32_t)value_out_addresses[batch].size(),
      .src_addrs = std::vector<addr_type>(value_out_addresses[batch].begin(), value_out_addresses[batch].end()),
      .operand_id = _OUTPUT_OPERAND
    }));
  }
}