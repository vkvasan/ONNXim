#include <random>
#include <algorithm>
#include <cstdlib>
#include <vector>
#include <string>
#include <map>
#include "Attention.h"
#include "../Model.h"
#include "../Tensor.h"
#include "GemmWS.h"
#include "Softmax.h"

Attention::Attention(SimulationConfig config, Model* model,
               onnx::NodeProto& node_proto, uint32_t target_core)
    : Operation(config, model, node_proto, target_core) {
    onnx = true;
    for (auto attribute : node_proto.attribute()) {
        if (attribute.name() == "num_heads") {
            _nh = attribute.i();
        }
    }

    /* Load weight info from node */
    _input_shape = get_input(0)->get_dims();
    _weight_shape = get_input(1)->get_dims();
    _bias_shape = get_input(2)->get_dims();
    if (_inputs.size() > 3)
        _mask_shape = get_input(3)->get_dims();
    if (node_proto.input().size()==5) {
        _kv_cache_shape = get_input(4)->get_dims();
        /* If "past_seq_len" is not 0 */
        if (_kv_cache_shape.at(3))
            has_kv_cache = true;
    }
    assert(_input_shape.size()==3);
    _batch_size = _input_shape.at(0);
    _dmodel = _weight_shape.at(0);
    _nkvh = _nh;
    _dk = _dmodel / _nh;
    _q_len = _input_shape.at(1);
    if (has_kv_cache)
        _seq = _kv_cache_shape.at(3) + 1;
    else
        _seq = _input_shape.at(1);

    _query_shape = std::vector<uint32_t>{_nh, _q_len, _dk};
    _key_shape = std::vector<uint32_t>{_nh, _seq, _dk};
    _value_shape = std::vector<uint32_t>{_nh, _seq, _dk};

    _output_shape = std::vector<uint32_t>{_batch_size, _q_len, _dmodel};
    _liner_output_shape = std::vector<uint32_t>{_batch_size, _q_len, _weight_shape[1]};
    _projection_output_shape = std::vector<uint32_t>{_batch_size, _q_len, _weight_shape[1]/3};
    spdlog::debug("Fused attention: input shape: [{}, {}, {}]", _input_shape.at(0), _input_shape.at(1), _input_shape.at(2));
    spdlog::debug("Fused attention: output shape: [{}, {}, {}]", _output_shape.at(0), _output_shape.at(1), _output_shape.at(2));
    spdlog::debug("Fused attention: query shape: [{}, {}, {}]", _query_shape.at(0), _query_shape.at(1), _query_shape.at(2));
    spdlog::debug("Fused attention: key shape: [{}, {}, {}]", _key_shape.at(0), _key_shape.at(1), _key_shape.at(2));
    spdlog::debug("Fused attention: value shape: [{}, {}, {}]", _value_shape.at(0), _value_shape.at(1), _value_shape.at(2));

    Tensor* pre_defind_tensor = _model->find_tensor(node_proto.output(0));
    if (pre_defind_tensor == nullptr) {
        std::unique_ptr<Tensor> output_tensor = std::make_unique<Tensor>(
            _id, node_proto.output(0), _output_shape, _config.precision, false);
            _outputs.push_back(output_tensor.get()->get_id());
        _model->add_tensor(std::move(output_tensor));
    } else {
        pre_defind_tensor->redefine_tensor(_id, _output_shape);
    }
}

Attention::Attention(SimulationConfig config, Model* model, 
        std::string name, std::map<std::string, std::string>& attributes, uint32_t target_core)
    :Operation(config, model, name, attributes, target_core) {
    _batch_size = 1;
    _q_len = std::stoi(get_attribute("num_tokens"));
    _nh = std::stoi(get_attribute("num_heads"));
    _nkvh = std::stoi(get_attribute("num_kv_heads"));
    _dmodel = std::stoi(get_attribute("hidden_size"));
    _dk = _dmodel / _nh;
}

void Attention::initialize_tiles(MappingTable& mapping_table) {
    if(_outputs.empty()) {
        _output_shape =  {_q_len, _dmodel};
        auto output_tensor = std::make_unique<Tensor> (
            _id, name_gen(_name, "output"), _output_shape, _config.precision, false);
        _outputs.push_back(output_tensor.get()->get_id());
        _model->add_tensor(std::move(output_tensor));
        _input_shape = get_input(0)->get_dims();
        _seq = get_input(1)->get_dims()[0]; //key first dim
        _weight_shape = get_input(1)->get_dims();
        _liner_output_shape = std::vector<uint32_t>{_q_len, _weight_shape[1]};
        _query_shape = std::vector<uint32_t>{_nh, _q_len, _dk};
        _key_shape = std::vector<uint32_t>{_nkvh, _seq, _dk};
        _value_shape = std::vector<uint32_t>{_nkvh, _seq, _dk};
    }
    Mapping mapping;
    calculate_loops(mapping);

    /* Check using fusion */
    /* The `&& onnx` is LOAD-BEARING, not just a feature gate. It makes the
       non-fused path unreachable from the language-model path, so LLMs always
       run FlashAttention. Removing it and setting ONNXIM_NONFUSED=1 segfaults
       (exit 139) immediately after the first head's Softmax init -- see the
       "Todo. dram addr" in initialize_non_fused_tiles(): the GemmWS/Softmax
       sub-ops it builds never get DRAM addresses assigned. Fixing that address
       assignment is a prerequisite for using non-fused attention with LLMs. */
    if (!use_fused && onnx) {
        initialize_non_fused_tiles(mapping_table);
        return;
    }
    /* Create linear node and tensors */
    uint32_t fused_op_id = 0;
    /* Fused Attention body */

    spdlog::info("Mapping info {}", mapping.to_string());
    /* Parallelism STRATEGY (ONNXIM_PAR_STRATEGY):
         head    (default, stock) rotate core per N -- N indexes head groups,
                 so head h lands on core h % num_cores, the same way for every
                 request. effectively head-parallel already.
         request pin this whole Attention op (= one request) to ONE core, so a
                 core walks one request's entire KV instead of a slice of many.
         seq     rotate per M (sequence chunk) instead of per head.
       Only the tile->core mapping changes; the addresses are identical. */
    static const std::string par_strategy = [] {
        const char* e = std::getenv("ONNXIM_PAR_STRATEGY");
        return std::string(e ? e : "head");
    }();
    static int op_counter = 0;
    const int op_core = (op_counter++) % _config.num_cores;
    int core_id = -1;
    for (uint32_t N = 0; N < mapping.tile_out_loop.N; N++) {
        int heads_per_kv = _nh / _nkvh;
        int qlen_offset = mapping.tile_out_loop.N / _nkvh;
        int head_off = N / qlen_offset * heads_per_kv;
        for(int M = 0; M < mapping.tile_out_loop.M; M++) {
            if (par_strategy == "request") {
                core_id = op_core;
            } else if (par_strategy == "seq") {
                core_id = M % _config.num_cores;
            } else if (M == 0) {
                core_id = (core_id + 1) % _config.num_cores;
            }
            std::unique_ptr<Tile> tile = std::make_unique<Tile>(Tile{
                .status = Tile::Status::INITIALIZED,
                .optype = get_name(),
                .layer_id = _id,
                .fused_op_id = fused_op_id++,
                .batch = N,
                .Q = 1,
                .P = 1, 
                .M = (uint32_t) M,
                .C = 1,
                .S = 1,
                .R = 1,
                .accum = M != 0,
                .core_id = core_id,
            });
            /* dummy mapping */
            _tiles.push_back(std::move(tile));
            initialize_instructions(_tiles.back().get(), mapping, head_off, heads_per_kv);
        }
    }
    float qk_flops = (2.0f * _q_len * _seq * _nh * _dk) / (float) 1e9;
    spdlog::info("[Attention] QK {} GFLOPs", qk_flops);
    float kv_flops = (2.0f * _q_len * _seq * _nh * _dk) / (float) 1e9;
    float softmax_flops = 5 * _q_len * _seq * _nh / (float) 1e9 * _config.full_precision / _config.precision;
        // 5: max, sub, exp, sum, div
    float tot_flops = qk_flops + softmax_flops + kv_flops;
    float kv_mem = _seq * _dk * _nkvh * 2  * _config.precision / (float) 1e9; //GB
    float q_mem = _q_len * _dk * _nh * 2 * _config.precision / (float) 1e9; //GB
    float total_mem = kv_mem + q_mem;
    float compute_time = (qk_flops + kv_flops) / _config.max_systolic_flops(target_core) * 1e3;
    compute_time += softmax_flops / _config.max_vector_flops(target_core) * 1e3;
    float mem_time = total_mem / _config.max_dram_bandwidth() * 1e3;
    float total_time = std::max(compute_time, mem_time);
    spdlog::info("[Attention] total {} GFLOPs, {} GB", tot_flops, total_mem);
    spdlog::info("[Attention] Theoretical time(ms): {} Compute time: {} Memory time: {}",
        total_time, compute_time, mem_time);
    spdlog::info("[Attention] QK compute {:.4f}ms Softmax compute {:.4f}ms SV compute {:.4f}ms",
        qk_flops / _config.max_systolic_flops(target_core) * 1e3,
        softmax_flops / _config.max_vector_flops(target_core) * 1e3,
        kv_flops / _config.max_systolic_flops(target_core) * 1e3);
}

void Attention::initialize_onnx_tiles(MappingTable& mapping_table) {
    calculate_loops();
    /* Check using fusion */
    if (!use_fused) {
        initialize_non_fused_tiles(mapping_table);
        return;
    }

    /* Create linear node and tensors */
    uint32_t fused_op_id = 0;
    _projection_node = new GemmWS(_config, mapping_table, _input_shape, _weight_shape, _liner_output_shape, target_core);
    std::unique_ptr<Tensor> key_projection = std::make_unique<Tensor>(
        _id, "", _projection_output_shape, _config.precision, false);
    std::unique_ptr<Tensor> query_projection = std::make_unique<Tensor>(
        _id, "", _projection_output_shape, _config.precision, false);
    std::unique_ptr<Tensor> value_projection = std::make_unique<Tensor>(
       _id, "", _projection_output_shape, _config.precision, false);

    /* Link tensors to linear node */
    _projection_node->set_model(_model);
    _projection_node->add_input(_inputs.at(0));
    _projection_node->add_input(_inputs.at(1));
    _projection_node->add_input(_inputs.at(2));
    _projection_node->add_output(key_projection.get()->get_id());
    _projection_node->add_output(query_projection.get()->get_id());
    _projection_node->add_output(value_projection.get()->get_id());
    get_input(0)->add_child_node(_projection_node);
    key_projection->add_child_node(this);
    query_projection->add_child_node(this);
    value_projection->add_child_node(this);

    /* Link key query value to attention node */
    _key_projection_id = _INPUT_OPERAND + _inputs.size();
    _inputs.push_back(key_projection.get()->get_id());
    _query_projection_id = _INPUT_OPERAND + _inputs.size();
    _inputs.push_back(query_projection.get()->get_id());
    _value_projection_id = _INPUT_OPERAND + _inputs.size();
    _inputs.push_back(value_projection.get()->get_id());

    /* Register tensor */
    _model->add_tensor(std::move(key_projection));
    _model->add_tensor(std::move(query_projection));
    _model->add_tensor(std::move(value_projection));

    /* Fused Attention body */
    for (int req_idx = 0; req_idx < _batch_size; req_idx++) {
        int heads_per_tile = _heads_per_tile[req_idx];
        for (int head_off=0; head_off<_nh; head_off+=heads_per_tile) {
            uint32_t remain_heads = std::min(_nh-head_off, (uint32_t)heads_per_tile);
            std::unique_ptr<Tile> tile = std::make_unique<Tile>(Tile{
                .status = Tile::Status::INITIALIZED,
                .optype = get_name(),
                .layer_id = _id,
                .fused_op_id = fused_op_id++,
                //.K = 0,
                .accum = false,
            });
            /* dummy mapping */
            _tiles.push_back(std::move(tile));
            initialize_instructions(_tiles.back().get(), head_off, heads_per_tile);
        }
    }
}

// 일단 한 tile에는 최대 하나의 request만 있는 경우부터.
void Attention::initialize_instructions(Tile* tile, int head_idx, int num_heads) {
    // head_idx # start idx
    // num_heads
    uint32_t q_len = _q_len;
    uint32_t seq_len = _seq;
    uint32_t value_offset = ceil_div(num_heads, _nkvh);
    addr_type sram_query_base = SPAD_BASE;
    addr_type sram_key_base = sram_query_base + q_len * _dk * num_heads * _config.precision;
    addr_type sram_value_base = sram_key_base + _dk * seq_len * value_offset * _config.precision;
    addr_type sram_logit_base = ACCUM_SPAD_BASE;  // for logits

    addr_type query_addr = get_operand_addr(_INPUT_OPERAND);
    addr_type key_addr = get_operand_addr(_INPUT_OPERAND + 1);
    addr_type value_addr = get_operand_addr(_INPUT_OPERAND + 2);
    addr_type ouput_addr = get_operand_addr(_OUTPUT_OPERAND);
    assert(num_heads <= _nkvh);
    /* FIX (upstream bug): head_idx / _nkvh maps every attention head of an
       MHA model to KV head 0, so all 32 heads of LLaMA-2 7B read the same
       1/32 of the cache. The KV head that serves query head h is h / (nh/nkvh). */
    const int heads_per_kv_ = std::max(1u, _nh / _nkvh);
    int kv_head_idx = head_idx / heads_per_kv_;
    addr_type sram_k_ofs = sram_key_base + kv_head_idx * (_dk * seq_len) * _config.precision;
    addr_type sram_v_ofs = sram_value_base + kv_head_idx * (_dk * seq_len) * _config.precision;
    /* std::set SORTS, so a whole head's KV read was issued in ascending address
       order rather than logical token order. Under paging that silently undoes
       the block scatter -- the effect being measured. ONNXIM_KV_UNSORTED=1
       keeps logical order (dedup preserved) to quantify the masking. */
    static const bool kv_unsorted = (std::getenv("ONNXIM_KV_UNSORTED") != nullptr);
    std::vector<addr_type> kv_seq;
    std::set<addr_type> dram_kv_addrs;    // = _key[req_idx]->get_all_addrs();
    _kv_tile_len = 0; _kv_tile_idx = 0;   /* single tile: no tile-bank offset */

    for(int seq_idx = 0; seq_idx < seq_len; seq_idx++) {
        for(int i = 0; i <_dk; i++) {
            addr_type a_ = kv_pool_enabled()
                ? kv_address_pooled((uint32_t)kv_head_idx, (uint32_t)seq_idx, (uint32_t)i, key_addr)
                : kv_address((uint32_t)kv_head_idx, (uint32_t)seq_idx, (uint32_t)i);
            if (dram_kv_addrs.insert(a_).second && kv_unsorted) kv_seq.push_back(a_);
        }
    }
    std::vector<addr_type> key_addrs, value_addrs;
    /* headbank: bases padded to a 32-chunk boundary so head bits hit the bank field */
    const addr_type key_base = key_addr + kv_headbank_pad(key_addr);
    const addr_type value_base = value_addr + kv_headbank_pad(value_addr) + kv_headbank_v_shift();
    if (kv_unsorted) {
        for (addr_type a_ : kv_seq) {
            key_addrs.push_back(key_base + a_);
            value_addrs.push_back(value_base + a_);
        }
    } else
    for(auto itr = dram_kv_addrs.begin(); itr != dram_kv_addrs.end(); itr++) {
        key_addrs.push_back(key_base + *itr);
        value_addrs.push_back(value_base + *itr);
    }
    tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
        .opcode = Opcode::MOVIN,
        .dest_addr = sram_k_ofs,
        .size = (uint32_t)value_addrs.size(),
        .src_addrs = key_addrs,
        .operand_id = _INPUT_OPERAND + 1,  // key
    }));
    tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
        .opcode = Opcode::MOVIN,
        .dest_addr = sram_v_ofs,
        .size = (uint32_t)value_addrs.size(),
        .src_addrs = value_addrs,
        .operand_id = _INPUT_OPERAND + 2,  // value
    }));
    for (int h_ofs = 0; h_ofs < num_heads; h_ofs++) {
        int h_idx = head_idx + h_ofs;
        addr_type sram_q_ofs = sram_query_base + h_ofs * (q_len * _dk) * _config.precision;
        addr_type sram_l_ofs = sram_logit_base + h_ofs * (q_len * seq_len) * _config.precision;
        std::set<addr_type> dram_query_addrs;  // = _query[req_idx]->get_all_addrs();
        std::set<addr_type> dram_output_addrs;
        for (int i = 0; i < _dk; i++) {
            for (int seq_idx = 0; seq_idx < q_len; seq_idx++) {
                // key:  h, d_k, seq_len
                std::vector<uint32_t> query_idx = {(uint32_t)(h_idx), (uint32_t)seq_idx, (uint32_t)i};
                std::vector<uint32_t> output_idx = {(uint32_t)(h_idx), (uint32_t)seq_idx, (uint32_t)i};
                dram_query_addrs.insert(act_lane(query_addr + make_address(query_idx, _query_shape)));
                dram_output_addrs.insert(act_lane(ouput_addr + make_address(output_idx, _query_shape))); // Used query_shape intentionally
            }
        }
        // -- load --
        // MOVIN query, key, value
        tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
            .opcode = Opcode::MOVIN,
            .dest_addr = sram_q_ofs,
            .size = (uint32_t)dram_query_addrs.size(),
            .src_addrs = std::vector<addr_type>(dram_query_addrs.begin(), dram_query_addrs.end()),
            .operand_id = _INPUT_OPERAND,  // query
        }));
        // -- compute --
        // GEMM (q*k -> l)
        tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
            .opcode = Opcode::GEMM,
            .dest_addr = sram_l_ofs,
            .size = q_len * seq_len * _config.precision / _config.dram_req_size,
            .compute_size =  ceil_div(q_len, _config.core_config[target_core].core_height) * seq_len,
            .src_addrs = std::vector<addr_type>{sram_q_ofs, sram_k_ofs},

            .tile_m = seq_len,
            .tile_k = _dk,
            .tile_n = q_len,
        }));
        // Softmax (l -> l)

        tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
            .opcode = Opcode::SOFTMAX,
            .dest_addr = sram_l_ofs,
            .size = q_len * seq_len * _config.precision / _config.dram_req_size,
            .compute_size = seq_len * _config.precision,
            .src_addrs = std::vector<addr_type>{sram_l_ofs},
            .tile_m = q_len,
            .src_from_accum = true,
        }));

        // [ ] change output offset
        // GEMM (l*v -> acc)
        tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
            .opcode = Opcode::GEMM,
            .dest_addr = sram_l_ofs,
            .size = q_len * _dk * _config.precision / _config.dram_req_size,
            .compute_size = q_len * _dk,
            .src_addrs = std::vector<addr_type>{sram_l_ofs, sram_v_ofs},

            .tile_m = _dk,
            .tile_k = seq_len,
            .tile_n = q_len,
            .src_from_accum = true,
        }));

        // MOVOUT
        tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
            .opcode = Opcode::MOVOUT,
            .dest_addr = sram_l_ofs,
            .size = (uint32_t)dram_output_addrs.size(),
            .src_addrs = std::vector<addr_type>(dram_output_addrs.begin(), dram_output_addrs.end()),
            .operand_id = _OUTPUT_OPERAND,
        }));
    }
}

void Attention::initialize_instructions(Tile* tile, Mapping mapping, int head_idx, int num_heads) {
    // head_idx # start idx
    // num_heads
    int qlen_offset = mapping.tile_out_loop.N / _nkvh;
    int q_ffset = tile->batch % qlen_offset;
    uint32_t q_len = mapping.tile_in_loop.N / num_heads;
    uint32_t seq_len = mapping.tile_in_loop.M;
    uint32_t value_offset = ceil_div(num_heads, _nkvh);
    addr_type sram_query_base = SPAD_BASE;
    addr_type sram_key_base = sram_query_base + q_len * _dk * num_heads * _config.precision;
    addr_type sram_value_base = sram_key_base + _dk * seq_len * value_offset * _config.precision;
    addr_type sram_logit_base = ACCUM_SPAD_BASE;  // for logits
    std::vector<uint32_t> output_shape = {_seq, _nh, _dk};
    addr_type query_addr = get_operand_addr(_INPUT_OPERAND);
    addr_type key_addr = get_operand_addr(_INPUT_OPERAND + 1);
    addr_type value_addr = get_operand_addr(_INPUT_OPERAND + 2);
    addr_type ouput_addr = get_operand_addr(_OUTPUT_OPERAND);
    addr_type logits_addr = get_operand_addr(_OUTPUT_OPERAND) + _dmodel * _q_len * _config.precision; // logits addr for scale
    assert(num_heads <= _nkvh);
    /* FIX (upstream bug): head_idx / _nkvh maps every attention head of an
       MHA model to KV head 0, so all 32 heads of LLaMA-2 7B read the same
       1/32 of the cache. The KV head that serves query head h is h / (nh/nkvh). */
    const int heads_per_kv_ = std::max(1u, _nh / _nkvh);
    int kv_head_idx = head_idx / heads_per_kv_;
    addr_type sram_k_ofs = sram_key_base + kv_head_idx * (_dk * seq_len) * _config.precision;
    addr_type sram_v_ofs = sram_value_base + kv_head_idx * (_dk * seq_len) * _config.precision;
    static const bool kv_unsorted2 = (std::getenv("ONNXIM_KV_UNSORTED") != nullptr);
    std::vector<addr_type> kv_seq2;
    std::set<addr_type> dram_kv_addrs;    // = _key[req_idx]->get_all_addrs();
    /* The bank offset must key on the M-TILE (tile->M), which is what gets
       dispatched to a core, not on seq_len: seq_len here is the spad-limited
       sub-chunk that one tile walks sequentially. Keying on the sub-chunk
       spread every chunk over stride-many banks (measured) and did nothing
       for the real collision, two M-tiles of one head on two cores. */
    _kv_tile_len = 0;
    _kv_tile_idx = (uint32_t)tile->M;
    for(int seq_idx = 0; seq_idx < seq_len; seq_idx++) {
        int kv_seq_index = tile->M * seq_len + seq_idx;
        if(kv_seq_index >= _seq) break;
        for(int i = 0; i <_dk; i++) {
            /* FIX (upstream bug): the address used the tile-local seq_idx, so
               when a long context is split over M tiles every tile re-read
               rows [0, seq_len) instead of its own chunk. */
            addr_type a_ = kv_pool_enabled()
                ? kv_address_pooled((uint32_t)kv_head_idx, (uint32_t)kv_seq_index, (uint32_t)i, key_addr)
                : kv_address((uint32_t)kv_head_idx, (uint32_t)kv_seq_index, (uint32_t)i);
            if (dram_kv_addrs.insert(a_).second && kv_unsorted2) kv_seq2.push_back(a_);
        }
    }

    std::vector<addr_type> key_addrs, value_addrs;
    /* headbank: bases padded to a 32-chunk boundary so head bits hit the bank field */
    const addr_type key_base = key_addr + kv_headbank_pad(key_addr);
    const addr_type value_base = value_addr + kv_headbank_pad(value_addr) + kv_headbank_v_shift();
    if (kv_unsorted2) {
        for (addr_type a_ : kv_seq2) {
            key_addrs.push_back(key_base + a_);
            value_addrs.push_back(value_base + a_);
        }
    } else {
        for(auto itr = dram_kv_addrs.begin(); itr != dram_kv_addrs.end(); itr++) {
            key_addrs.push_back(key_base + *itr);
            value_addrs.push_back(value_base + *itr);
        }
    }
    tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
        .opcode = Opcode::MOVIN,
        .dest_addr = sram_k_ofs,
        .size = (uint32_t)value_addrs.size(),
        .src_addrs = key_addrs,
        .operand_id = _INPUT_OPERAND + 1,  // key
    }));

    for (int h_ofs = 0; h_ofs < num_heads; h_ofs++) {
        int h_idx = head_idx + h_ofs;
        addr_type sram_q_ofs = sram_query_base + h_ofs * (q_len * _dk) * _config.precision;
        addr_type sram_l_ofs = sram_logit_base + h_ofs * (q_len * seq_len) * _config.precision;
        addr_type sram_logits_offset = sram_l_ofs + num_heads * (q_len * seq_len) * _config.precision;
        std::set<addr_type> dram_query_addrs;  // = _query[req_idx]->get_all_addrs();
        for (int seq_idx = 0; seq_idx < q_len; seq_idx++) {
            for (int i = 0; i < _dk; i++) {
                // key:  h, d_k, seq_len
                int q_index = q_ffset * q_len + seq_idx;
                if(q_index >= _q_len) break;
                std::vector<uint32_t> query_idx = {(uint32_t)(h_idx), (uint32_t)q_index, (uint32_t)i};
                dram_query_addrs.insert(act_lane(query_addr + make_address(query_idx, _query_shape)));
            }
        }
        // -- load --
        // MOVIN query, key, value
        tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
            .opcode = Opcode::MOVIN,
            .dest_addr = sram_q_ofs,
            .size = (uint32_t)dram_query_addrs.size(),
            .src_addrs = std::vector<addr_type>(dram_query_addrs.begin(), dram_query_addrs.end()),
            .operand_id = _INPUT_OPERAND,  // query
        }));
    }
    tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
        .opcode = Opcode::MOVIN,
        .dest_addr = sram_v_ofs,
        .size = (uint32_t)value_addrs.size(),
        .src_addrs = value_addrs,
        .operand_id = _INPUT_OPERAND + 2,  // value
    }));
     // -- compute -- //     
    // GEMM (q*k -> l)
    for(int sitr = 0; sitr < seq_len; sitr+=_config.core_config[target_core].core_height) {
        int s_loop = std::min(seq_len - sitr, _config.core_config[target_core].core_height);
        for(int kitr = 0; kitr < _dk; kitr+=_config.core_config[target_core].core_height) {
            int k_loop = std::min(_dk - kitr, _config.core_config[target_core].core_height);
                for (int h_ofs = 0; h_ofs < num_heads; h_ofs++) {
                Opcode op = h_ofs == 0 ? Opcode::GEMM_PRELOAD : Opcode::GEMM;
                int h_idx = head_idx + h_ofs;
                addr_type sram_q_ofs = sram_query_base + h_ofs * (q_len * _dk) * _config.precision;
                addr_type sram_l_ofs = sram_logit_base + h_ofs * (q_len * seq_len) * _config.precision;
                addr_type sram_logits_offset = sram_l_ofs + num_heads * (q_len * seq_len) * _config.precision;
                tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
                    .opcode = op,
                    .dest_addr = sram_l_ofs,
                    .size =  q_len * _config.precision / _config.dram_req_size,
                    .compute_size = q_len,
                    .src_addrs = std::vector<addr_type>{sram_q_ofs, sram_k_ofs},
                    .tile_m = static_cast<unsigned int>(s_loop),
                    .tile_k = static_cast<unsigned int>(k_loop),
                    .tile_n = static_cast<unsigned int>(q_len)
                }));
            }
        }
    }

    for (int h_ofs = 0; h_ofs < num_heads; h_ofs++) {
        int h_idx = head_idx + h_ofs;
        addr_type sram_q_ofs = sram_query_base + h_ofs * (q_len * _dk) * _config.precision;
        addr_type sram_l_ofs = sram_logit_base + h_ofs * (q_len * seq_len) * _config.precision;
        addr_type sram_logits_offset = sram_l_ofs + num_heads * (q_len * seq_len) * _config.precision;
        // Softmax (l -> l)
        tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
            .opcode = Opcode::ADDTREE,
            .dest_addr = sram_logits_offset,
            .size = q_len * seq_len * _config.precision / _config.dram_req_size,
            .compute_size = seq_len * _config.precision + 1, // 1 for prior max
            .src_addrs = std::vector<addr_type>{sram_l_ofs},
            .tile_m = q_len,
            .src_from_accum = true,
        }));// On chip, compute 𝑚(𝑗) = max(𝑚(𝑗−1),rowmax(S(𝑗))) ∈ R𝐵𝑟
        tile->instructions.push_back(std::make_unique<Instruction>(Instruction{ 
            .opcode = Opcode::ADD,
            .dest_addr = sram_l_ofs,
            .size = q_len * seq_len * _config.precision / _config.dram_req_size,
            .compute_size = q_len * (seq_len + 1) * _config.precision, // 1 for prior max
            .src_addrs = std::vector<addr_type>{sram_l_ofs, sram_logits_offset},
            .tile_m = q_len,
            .src_from_accum = true,
        })); // S(𝑗) −𝑚(𝑗)
        tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
            .opcode = Opcode::EXP,
            .dest_addr = sram_l_ofs,
            .size = q_len * seq_len * _config.precision / _config.dram_req_size,
            .compute_size = q_len * (seq_len + 1) * _config.full_precision, // 1 for prior max
            .src_addrs = std::vector<addr_type>{sram_l_ofs},
            .tile_m = q_len,
            .src_from_accum = true,
        })); // P(j) = exp(S(𝑗) −𝑚(𝑗)) , exp(m(j-1) - m(j))
        tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
            .opcode = Opcode::ADDTREE,
            .dest_addr = sram_logits_offset,
            .size = q_len * seq_len * _config.precision / _config.dram_req_size,
            .compute_size = seq_len * _config.full_precision,
            .src_addrs = std::vector<addr_type>{sram_l_ofs},
            .tile_m = q_len,
            .src_from_accum = true,
        }));// rowsum(P(𝑗)) ∈ R𝐵𝑟
        tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
            .opcode = Opcode::MAC,
            .dest_addr = sram_logits_offset,
            .size = q_len * seq_len * _config.precision / _config.dram_req_size,
            .compute_size = q_len * seq_len * _config.full_precision,
            .src_addrs = std::vector<addr_type>{sram_logits_offset},
            .tile_m = q_len,
            .src_from_accum = true,
        }));  // 𝑒^(𝑖m^(j-1) -m^(j))  +rowsum(P )∈R 
        tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
            .opcode = Opcode::DIV,
            .dest_addr = sram_l_ofs,
            .size = q_len * _config.precision / _config.dram_req_size,
            .compute_size = q_len  * _config.full_precision,
            .src_addrs = std::vector<addr_type>{sram_logits_offset},
            .tile_m = q_len,
            .src_from_accum = true,
        })); // diag(exp(m(j-1) - m(j))-1)
        tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
            .opcode = Opcode::MUL,
            .dest_addr = sram_l_ofs,
            .size = q_len * seq_len* _config.precision / _config.dram_req_size,
            .compute_size = q_len * seq_len * _config.full_precision,
            .src_addrs = std::vector<addr_type>{sram_l_ofs, sram_logits_offset},
            .tile_m = q_len,
            .src_from_accum = true,
        })); // diag(exp(m(j-1) - m(j))-1) * O(j-1)
    }
            // GEMM (l*v -> acc)
    
    for(int kitr = 0; kitr < _dk; kitr+=_config.core_config[target_core].core_height) {
        int k_loop = std::min(_dk - kitr, _config.core_config[target_core].core_height);
        for(int sitr = 0; sitr < seq_len; sitr+=_config.core_config[target_core].core_height) {
            int s_loop = std::min(seq_len - sitr, _config.core_config[target_core].core_height);
            for (int h_ofs = 0; h_ofs < num_heads; h_ofs++) {
                Opcode op = h_ofs == 0 ? Opcode::GEMM_PRELOAD : Opcode::GEMM;
                addr_type sram_l_ofs = sram_logit_base + h_ofs * (q_len * seq_len) * _config.precision;
                tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
                    .opcode = op,
                    .dest_addr = sram_l_ofs,
                    .size = q_len * _config.precision / _config.dram_req_size,
                    .compute_size = q_len,
                    .src_addrs = std::vector<addr_type>{sram_l_ofs, sram_v_ofs},
                    .tile_m = static_cast<unsigned int>(k_loop),
                    .tile_k = static_cast<unsigned int>(s_loop),
                    .tile_n = static_cast<unsigned int>(q_len),
                    .src_from_accum = true
                }));
            }
        }
    }

    for (int h_ofs = 0; h_ofs < num_heads; h_ofs++) {
        int h_idx = head_idx + h_ofs;
        addr_type sram_l_ofs = sram_logit_base + h_ofs * (q_len * seq_len) * _config.precision;
        addr_type sram_logits_offset = sram_l_ofs + num_heads * (q_len * seq_len) * _config.precision;
        if(tile->M == mapping.tile_out_loop.M -1) {
            std::set<addr_type> dram_output_addrs;
            for (int seq_idx = 0; seq_idx < q_len; seq_idx++) {
                for (int i = 0; i < _dk; i++) {
                    /* FIX (upstream bug): tile->M indexes the KV chunk, not the
                       query chunk; with M > 0 no output row was ever written. */
                    int q_index = q_ffset * q_len + seq_idx;
                    if(q_index >= _q_len) break;
                    std::vector<uint32_t> output_idx = {(uint32_t)q_index, (uint32_t)(h_idx), (uint32_t)i};
                    dram_output_addrs.insert(act_lane(ouput_addr + make_address(output_idx, output_shape))); // Used query_shape intentionally
                }
            }
            tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
                .opcode = Opcode::DIV,
                .dest_addr = sram_l_ofs,
                .size = _q_len * _config.precision / _config.dram_req_size,
                .compute_size = _q_len  * _config.full_precision,
                .src_addrs = std::vector<addr_type>{sram_logits_offset},
                .tile_m = q_len,
                .src_from_accum = true,
            })); // diag(l(Tc))-1
            tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
                .opcode = Opcode::MUL,
                .dest_addr = sram_l_ofs,
                .size = _q_len * _dk * _config.precision / _config.dram_req_size,
                .compute_size = _q_len * _dk * _config.full_precision,
                .src_addrs = std::vector<addr_type>{sram_l_ofs, sram_logits_offset},
                .tile_m = q_len,
                .src_from_accum = true,
            })); // diag(l(Tc))-1 * O(Tc)
            // MOVOUT
            tile->instructions.push_back(std::make_unique<Instruction>(Instruction{
                .opcode = Opcode::MOVOUT,
                .dest_addr = sram_l_ofs,
                .size = (uint32_t)dram_output_addrs.size(),
                .src_addrs = std::vector<addr_type>(dram_output_addrs.begin(), dram_output_addrs.end()),
                .operand_id = _OUTPUT_OPERAND,
            }));
        }
    }
}

/* ---- PagedAttention KV addressing (see Attention.h) ---------------------- */
uint32_t Attention::kv_block_tokens() {
    static uint32_t b = [] {
        const char* e = std::getenv("ONNXIM_KV_BLOCK");
        return e ? (uint32_t)std::strtoul(e, nullptr, 10) : 0u;
    }();
    return b;
}

const std::vector<uint32_t>& Attention::kv_block_table(uint32_t n_blocks) {
    /* One table per block-count, built once and reused, so every layer and head
       of a run sees the SAME physical layout -- as a real server would. */
    static std::map<uint32_t, std::vector<uint32_t>> cache;
    auto it = cache.find(n_blocks);
    if (it != cache.end()) return it->second;
    std::vector<uint32_t> table(n_blocks);
    for (uint32_t i = 0; i < n_blocks; i++) table[i] = i;
    const char* mode = std::getenv("ONNXIM_KV_ALLOC");
    if (!(mode && std::string(mode) == "seq")) {
        std::mt19937 rng(12345);            /* fixed seed: runs are reproducible */
        std::shuffle(table.begin(), table.end(), rng);
    }
    cache[n_blocks] = std::move(table);
    return cache[n_blocks];
}

bool Attention::kv_pool_enabled() {
    static bool e = [] { const char* v = std::getenv("ONNXIM_KV_POOL"); return v && v[0] == '1'; }();
    return e;
}

/* Register each distinct KV tensor base in first-seen order and remember the
   lowest one; that becomes the pool base. Returns this tensor's ordinal. */
uint32_t Attention::kv_pool_ordinal(addr_type tensor_base, addr_type& pool_base) {
    static std::map<addr_type, uint32_t> ordinal;
    static addr_type lowest = 0;
    auto it = ordinal.find(tensor_base);
    if (it == ordinal.end()) {
        uint32_t n = (uint32_t)ordinal.size();
        ordinal[tensor_base] = n;
        if (lowest == 0 || tensor_base < lowest) lowest = tensor_base;
        it = ordinal.find(tensor_base);
    }
    pool_base = lowest;
    return it->second;
}

addr_type Attention::kv_address_pooled(uint32_t head, uint32_t seq_idx, uint32_t d,
                                       addr_type tensor_base) {
    const uint32_t B = kv_block_tokens();
    addr_type pool_base = 0;
    const uint32_t r = kv_pool_ordinal(tensor_base, pool_base);
    static std::map<addr_type,uint32_t> dummy;
    const uint32_t n_blocks = (_seq + B - 1) / B;
    const uint32_t logical  = seq_idx / B;
    /* Round-robin across requests: request r takes every R-th slot, so blocks
       from different sequences sit next to each other in the pool. R is the
       number of tensors registered so far (>=1). */
    static uint32_t n_reg = 1;
    n_reg = std::max(n_reg, r + 1);
    const uint32_t slot_index = logical * n_reg + r;
    const uint32_t pool_slots = n_blocks * n_reg;
    const uint32_t phys = kv_block_table(pool_slots)[slot_index % pool_slots];
    const addr_type blk = (addr_type)_key_shape.at(0) * B * _dk;   /* elems per block */
    static const bool head_major = [] {
        const char* e = std::getenv("ONNXIM_KV_LAYOUT");
        return e && std::string(e) == "head";
    }();
    if (kv_headbank_enabled()) {
        /* see kv_address(): head index on the bank bits, pool-relative */
        const addr_type p = ((addr_type)phys * B * _dk
                           + (addr_type)(seq_idx % B) * _dk + d) * _config.precision;
        const uint32_t tile = _kv_tile_idx;
        return _config.align_address(kv_headbank_offset(head, p, tile)) + (pool_base - tensor_base);
    }
    addr_type off;
    if (head_major)
        /* each head owns a contiguous slice of the WHOLE pool */
        off = (addr_type)head * pool_slots * B * _dk
            + (addr_type)phys * B * _dk
            + (addr_type)(seq_idx % B) * _dk
            + d;
    else
        off = (addr_type)phys * blk
            + (addr_type)head * B * _dk
            + (addr_type)(seq_idx % B) * _dk
            + d;
    /* rebase: caller adds tensor_base, so subtract it and add the pool base */
    return _config.align_address(off * _config.precision) + (pool_base - tensor_base);
}

addr_type Attention::kv_address(uint32_t head, uint32_t seq_idx, uint32_t d) {
    const uint32_t B = kv_block_tokens();
    if (B == 0)
        return make_address({head, seq_idx, d}, _key_shape);
    const uint32_t n_blocks = (_seq + B - 1) / B;
    const uint32_t phys = kv_block_table(n_blocks)[seq_idx / B];
    /* Two layouts, same total size, same block table:
         BLOCK-major (default, vLLM-like)  [block][kv_head][token][dk]
           a block is one contiguous 128 KB region holding every head, so
           reading ONE head gives 4 KB fragments 128 KB apart -- even when
           blocks are physically adjacent, because the other 31 heads sit
           in between.
         HEAD-major (ONNXIM_KV_LAYOUT=head) [kv_head][block][token][dk]
           each head owns a contiguous region; a "block" is then 32 separate
           slices, one per head, all at the same slot index. Still allocatable
           and freeable as one unit (one block-table entry), but reading one
           head walks its own region -- contiguous if the table is identity. */
    static const bool head_major = [] {
        const char* e = std::getenv("ONNXIM_KV_LAYOUT");
        return e && std::string(e) == "head";
    }();
    if (kv_headbank_enabled()) {
        /* HEAD-BANK (ONNXIM_KV_LAYOUT=headbank): head-major's per-head stream,
           re-interleaved so the head index lands on the DRAM bank bits.
           Under RoBaRaCoCh with this config (16 B tx x 16 ch compaction,
           64-tx rows, 2 pch x 4 bg x 4 bank) the bank fields are global
           address bits [14:19): a 16 KB chunk is one row of one bank in every
           channel, and 32 consecutive chunks span the 32 banks. So head h
           owns every chunk with index == h (mod 32): its own bank, rows
           walked sequentially, and no other head's stream ever evicts its
           row. Footprint is identical to head-major (NH x per-head bytes),
           rounded up by < 32 chunks at the tensor end. */
        const addr_type p = ((addr_type)phys * B * _dk
                           + (addr_type)(seq_idx % B) * _dk + d) * _config.precision;
        const uint32_t tile = _kv_tile_idx;
        return _config.align_address(kv_headbank_offset(head, p, tile));
    }
    addr_type off;
    if (head_major)
        off = (addr_type)head * n_blocks * B * _dk
            + (addr_type)phys * B * _dk
            + (addr_type)(seq_idx % B) * _dk
            + d;
    else
        off = (addr_type)phys * _key_shape.at(0) * B * _dk
            + (addr_type)head * B * _dk
            + (addr_type)(seq_idx % B) * _dk
            + d;
    return _config.align_address(off * _config.precision);
}

bool Attention::kv_headbank_enabled() {
    static const bool e = [] {
        const char* v = std::getenv("ONNXIM_KV_LAYOUT");
        return v && std::string(v) == "headbank";
    }();
    return e;
}

/* Chunk = bytes per (bank, row) stripe across all channels. Derived for the
   _c128 config above; override with ONNXIM_KV_BANKCHUNK for other geometries. */
addr_type Attention::kv_headbank_chunk() {
    static const addr_type c = [] {
        const char* v = std::getenv("ONNXIM_KV_BANKCHUNK");
        return v ? (addr_type)std::strtoull(v, nullptr, 10) : (addr_type)16384;
    }();
    return c;
}

/* Banks per head, K (ONNXIM_KV_BANKS_PER_HEAD, default 1). K=1 gives each
   head exactly one bank: no cross-head row eviction, but a stream's next row
   is in the SAME bank so its PRE->ACT cannot overlap its own data. K>1 lets
   a head alternate rows over K banks (intra-stream overlap) at the price of
   NH*K/NB heads sharing each bank set on interleaved rows. */
uint32_t Attention::kv_headbank_k() {
    static const uint32_t k = [] {
        const char* v = std::getenv("ONNXIM_KV_BANKS_PER_HEAD");
        uint32_t k = v ? (uint32_t)std::strtoul(v, nullptr, 10) : 1u;
        return k ? k : 1u;
    }();
    return k;
}

/* Number of distinct bank-field values (pch x bg x bank) = chunks per row
   stripe; ONNXIM_KV_NBANKS, default 32 for the _c128 geometry. */
uint32_t Attention::kv_headbank_nbanks() {
    static const uint32_t n = [] {
        const char* v = std::getenv("ONNXIM_KV_NBANKS");
        uint32_t n = v ? (uint32_t)std::strtoul(v, nullptr, 10) : 32u;
        return n ? n : 32u;
    }();
    return n;
}

/* Byte offset of stream position p (bytes into head `head`'s own sequence):
     c = p / CH            chunk index = row index within the head's stream
     b = (head*K + c%K) % NB   bank-field value for this chunk
     r = c / K             row group; S = max(1, NH*K/NB) heads share a set,
                           told apart by the row field: row = r*S + head/(NB/K)
     off = row * (NB*CH) + b * CH + p % CH
   Footprint = NH x per-head bytes (rounded up by < NB chunks). K=1, NH=NB
   reduces to: head h owns every chunk == h (mod NB). */
/* Tile-bank stride (ONNXIM_KV_TILE_BANK_STRIDE, default 0). A long context is
   split into M tiles of ~450 tokens and dynamic dispatch runs them on
   different cores at the same time; with one bank per head they all land in
   bank h and ping-pong its row buffer (measured: 4 cores reading 4 disjoint
   row ranges of head 5 in lockstep, 98% of re-opens other-core). Offsetting
   tile m's bank by m*stride puts a head's concurrent tiles in different
   banks: stride 8 -> h, h+8, h+16, h+24 for four tiles in flight. */
uint32_t Attention::kv_tile_bank_stride() {
    static const uint32_t s = [] {
        const char* v = std::getenv("ONNXIM_KV_TILE_BANK_STRIDE");
        return v ? (uint32_t)std::strtoul(v, nullptr, 10) : 0u;
    }();
    return s;
}

addr_type Attention::kv_headbank_offset(uint32_t head, addr_type p, uint32_t tile) {
    const addr_type CH = kv_headbank_chunk();
    const addr_type NB = kv_headbank_nbanks();
    const addr_type K  = kv_headbank_k();
    const addr_type NH = _key_shape.at(0);
    const addr_type sets = std::max<addr_type>(1, NB / K);
    const addr_type S    = std::max<addr_type>(1, (NH * K) / NB);
    const addr_type c = p / CH;
    const addr_type b = ((addr_type)head * K + (c % K) + (addr_type)tile * kv_tile_bank_stride()) % NB;
    const addr_type r = c / K;
    const addr_type row = r * S + ((addr_type)head / sets);
    return row * (NB * CH) + b * CH + (p % CH);
}

/* V of head h shares K(h)'s bank under headbank, and the attention tile
   issues K and V MOVINs back to back, so the two streams ping-pong the same
   bank on different rows. ONNXIM_KV_V_BANK_OFFSET=<banks> (default 0) shifts
   every V tensor by that many bank slots; 16 with K=1 and head-parallel puts
   V(h) in bank h+16, disjoint from every K stream that can be in flight with
   it (heads c, c+4, c+8 on core c at tile_depth 3). */
addr_type Attention::kv_headbank_v_shift() {
    static const addr_type s = [] {
        const char* v = std::getenv("ONNXIM_KV_V_BANK_OFFSET");
        return v ? (addr_type)std::strtoull(v, nullptr, 10) : (addr_type)0;
    }();
    return kv_headbank_enabled() ? s * kv_headbank_chunk() : 0;
}

/* ACTIVATION LANE (ONNXIM_ACT_LANE=<slot>): the attention tile's query MOVIN
   and output MOVOUT read/write 256 B slices of tensors shared by every head,
   through plain row-major make_address -- so they hop every bank at 1 KB per
   channel and each 1-2 request visit closes a head's open row (measured:
   72% of KV row visits under headbank re-open a just-left row). Steer them
   into one reserved bank slot by folding the 5 bank bits into the row index:
       a = hi*2^19 + b*2^14 + lo   ->   (hi*32 + b)*2^19 + S*2^14 + lo
   Injective, keeps the within-chunk offset, costs the lane's own bank only.
   Shares bank S with head S (K=1) -- pick S to taste. Off unless set. */
addr_type Attention::act_lane(addr_type a) {
    static const long lane = [] {
        const char* v = std::getenv("ONNXIM_ACT_LANE");
        return v ? std::strtol(v, nullptr, 10) : -1L;
    }();
    if (lane < 0) return a;
    const addr_type CH = kv_headbank_chunk();               // 16 KB
    const addr_type NB = kv_headbank_nbanks();              // 32
    const addr_type lo = a % CH;
    const addr_type b  = (a / CH) % NB;
    const addr_type hi = a / (CH * NB);
    return (hi * NB + b) * (CH * NB) + (addr_type)lane * CH + lo;
}

/* The head->bank property needs the tensor base on an NB-chunk boundary;
   ONNXim packs tensors at 256 B. Pad the base up (spills < NB chunks past the
   tensor end, harmless to timing). Zero unless headbank is on. */
addr_type Attention::kv_headbank_pad(addr_type base) {
    if (!kv_headbank_enabled()) return 0;
    const addr_type CH  = kv_headbank_chunk();
    const addr_type NB  = kv_headbank_nbanks();
    const addr_type grp = CH * NB;
    /* ONNXIM_KV_REQ_ROTATE=1: attention ops of different requests run
       concurrently on different cores, so head h of request A and head h of
       request B otherwise share bank h and ping-pong it (measured: 72% of KV
       row visits re-open a just-left row; the intruding rows carried the
       slot's own head from another core). Rotate each tensor's bank origin by
       a per-tensor amount derived from its base, so the same head of
       different requests lands in different banks. K and V of one request
       get the same rotation (their bases differ by a multiple of grp). */
    static const bool rot_on = [] {
        const char* v = std::getenv("ONNXIM_KV_REQ_ROTATE");
        return v && v[0] == '1';
    }();
    const addr_type rot = rot_on ? (((base / grp) * 11 + (base / CH)) % NB) : 0;
    const addr_type target = rot * CH;                       /* desired base % grp */
    return (target + grp - (base % grp)) % grp;
}

void Attention::initialize_non_fused_tiles(MappingTable& mapping_table) {
    /* Create linear node and tensors */
    uint32_t fused_op_id = 0;
    std::vector<uint32_t> single_head_query_shape = std::vector<uint32_t>{_q_len, _dk};
    std::vector<uint32_t> single_head_key_shape = std::vector<uint32_t>{_dk, _seq};
    std::vector<uint32_t> single_head_value_shape = std::vector<uint32_t>{_seq, _dk};
    std::vector<uint32_t> single_output_shape = std::vector<uint32_t>{_q_len, _dk};
    std::vector<uint32_t> query_key_shape = std::vector<uint32_t>{_q_len, _seq};

    /* Fused Attention body */
    for (int req_idx = 0; req_idx < _batch_size; req_idx++) {
        for (int head_off=0; head_off<_nh; head_off++) {
            /* Key query matmul */
            GemmWS key_query = GemmWS(_config, mapping_table, single_head_query_shape, single_head_key_shape, query_key_shape, target_core);
            /* Todo. dram addr */
            key_query.has_bias = false;
            key_query.initialize_tiles(mapping_table);
            std::deque<std::unique_ptr<Tile>>& key_query_tiles = key_query.get_tiles();
            for (const auto& tile : key_query_tiles) {
                tile->layer_id = _id;
                tile->fused_op_id = fused_op_id;
            }
            _tiles.insert(
                _tiles.end(),
                std::make_move_iterator(key_query.get_tiles().begin()),
                std::make_move_iterator(key_query.get_tiles().end())
            );
            _tiles.push_back(std::make_unique<Tile>(Tile{.status = Tile::Status::BAR, .layer_id = _id}));
            fused_op_id++;

            /* Softmax */
            Softmax attention_score = Softmax(_config, mapping_table, query_key_shape);
            /* Todo. dram addr */
            attention_score.initialize_tiles(mapping_table);
            std::deque<std::unique_ptr<Tile>>& attention_score_tiles = key_query.get_tiles();
            for (const auto& tile : attention_score_tiles) {
                tile->layer_id = _id;
            }
            _tiles.insert(
                _tiles.end(),
                std::make_move_iterator(key_query.get_tiles().begin()),
                std::make_move_iterator(key_query.get_tiles().end())
            );
            _tiles.push_back(std::make_unique<Tile>(Tile{.status = Tile::Status::BAR, .layer_id = _id}));

            /* attention x value */
            GemmWS attention = GemmWS(_config, mapping_table, query_key_shape, single_head_value_shape, single_output_shape, target_core);
            /* Todo. dram addr */
            attention.has_bias = false;
            attention.initialize_tiles(mapping_table);
            std::deque<std::unique_ptr<Tile>>& attention_tiles = attention.get_tiles();
            for (const auto& tile : attention_tiles) {
                tile->layer_id = _id;
                tile->fused_op_id = fused_op_id;
            }
            _tiles.insert(
                _tiles.end(),
                std::make_move_iterator(attention.get_tiles().begin()),
                std::make_move_iterator(attention.get_tiles().end())
            );
            _tiles.push_back(std::make_unique<Tile>(Tile{.status = Tile::Status::BAR, .layer_id = _id}));
            fused_op_id++;
        }
    }
}

void Attention::calculate_loops() {
    for (int i = 0; i < _batch_size; i++) {
        uint32_t spad_capacity = _config.core_config[target_core].spad_size KB / _config.tile_depth;  // unit: byte
        uint32_t acc_spad_capacity = _config.core_config[target_core].accum_spad_size KB / _config.tile_depth;
        int heads_per_kv = _nh / _nkvh;
        uint32_t q_len = _q_len;
        uint32_t seq_len = _seq;

        uint32_t tiles_per_head = std::max(ceil_div(seq_len * _config.precision, acc_spad_capacity),
                                          ceil_div(2 * seq_len * _dk * _config.precision, 
                                                    spad_capacity - heads_per_kv * _dk * _config.precision));
        if(tiles_per_head > 1) {
            q_len = 1;
            seq_len = ceil_div(_seq, tiles_per_head);
        } else {
            int max_q_acc = acc_spad_capacity / (seq_len * _config.precision);
            int max_q_spad = (spad_capacity - 2* seq_len * _dk * _config.precision) / (heads_per_kv * _dk * _config.precision);
            q_len = std::min(q_len, (uint32_t)max_q_acc);
            q_len = std::min(q_len, (uint32_t)max_q_spad);
        }

        uint32_t total_spad_size_per_head = 2 * seq_len * _dk +  heads_per_kv * q_len * _dk;
        uint32_t total_acc_size_per_head = seq_len * q_len;
        total_spad_size_per_head *= _config.precision;
        total_acc_size_per_head *= _config.precision;
        spdlog::info("[Attention] total_spad_size_per_head: {}", total_spad_size_per_head);
        spdlog::info("[Attention] total_acc_size_per_head: {}", total_acc_size_per_head);
        spdlog::info("[Attention] q_len: {}, seq_len: {}, dk: {}", q_len, seq_len, _dk);
        spdlog::info("[Attention] Spad size {}", _config.core_config[target_core].spad_size KB / _config.tile_depth);
        _heads_per_tile.push_back(heads_per_kv);
    }
}

void Attention::calculate_loops(Mapping& mapping) {
    for (int i = 0; i < _batch_size; i++) {
        uint32_t spad_capacity = _config.core_config[target_core].spad_size KB / _config.tile_depth;  // unit: byte
        uint32_t acc_spad_capacity = _config.core_config[target_core].accum_spad_size KB / _config.tile_depth;
        int heads_per_kv = _nh / _nkvh;
        uint32_t q_len = _q_len;
        uint32_t seq_len = _seq;
        uint32_t per_query_size = (heads_per_kv * _dk * _config.precision);
        uint32_t tiles_per_head = std::max(ceil_div(seq_len * _config.precision, acc_spad_capacity),
                                          ceil_div(2 * seq_len * _dk * _config.precision, 
                                                    spad_capacity - per_query_size));
        int tile_out_loop = _nkvh;
        /* Multi-query decode against a context too long for one tile
           (speculative verify, chunked prefill). Stock ONNXim forced q_len=1
           here and gave every query its own pass over the KV cache, i.e. the
           cache was read _q_len times. A few query rows cost only
           per_query_size bytes each, so keep them all resident and shrink the
           KV chunk instead. Only taken for _q_len > 1, so single-token decode
           is bit-identical to before. */
        bool multi_q_tiled = false;
        if (tiles_per_head > 1 && _q_len > 1) {
            uint32_t q_bytes_all = _q_len * per_query_size;
            if (q_bytes_all <= spad_capacity / 4) {
                uint32_t chunk_spad = (spad_capacity - q_bytes_all) / (2 * _dk * _config.precision);
                uint32_t chunk_acc = acc_spad_capacity / ((_q_len * _config.precision + 2 * 4) * heads_per_kv);
                uint32_t chunk = std::min(chunk_spad, chunk_acc);
                if (chunk >= _config.core_config[target_core].core_height) {
                    tiles_per_head = ceil_div(_seq, chunk);
                    seq_len = ceil_div(_seq, tiles_per_head);
                    q_len = _q_len;
                    tile_out_loop = _nkvh;
                    multi_q_tiled = true;
                }
            }
        }
        if (multi_q_tiled) {
            /* nothing more to do */
        } else if(tiles_per_head > 1) {
            q_len = 1;
            seq_len = ceil_div(_seq, tiles_per_head);
            tile_out_loop = _q_len * _nkvh;
        } else {
            int max_q_acc = acc_spad_capacity / ((seq_len * _config.precision + 2 * 4) * heads_per_kv);
            int max_q_spad = (spad_capacity - 2* seq_len * _dk * _config.precision)
                            / per_query_size;
            q_len = std::min(q_len, (uint32_t)max_q_acc);
            q_len = std::min(q_len, (uint32_t)max_q_spad);
            /*
             * A small accumulator can drive max_q_acc to 0, which made q_len 0
             * and divided by zero in ceil_div below. At least one query must be
             * processed per tile.
             */
            q_len = std::max(q_len, 1u);
            tile_out_loop = ceil_div(_q_len, q_len) * _nkvh;
        }

        uint32_t total_spad_size_per_head = 2 * seq_len * _dk +  heads_per_kv * q_len * _dk;
        uint32_t total_acc_size_per_head = seq_len * q_len;
        total_spad_size_per_head *= _config.precision;
        total_acc_size_per_head *= _config.precision;
        spdlog::info("[Attention] total_spad_size_per_head: {} B", total_spad_size_per_head);
        spdlog::info("[Attention] total_acc_size_per_head: {} B", total_acc_size_per_head);
        spdlog::info("[Attention] q_len: {}, seq_len: {}, dk: {}, heads per tile {}", q_len, seq_len, _dk, heads_per_kv);
        spdlog::info("[Attention] Spad size {}", _config.core_config[target_core].spad_size KB / _config.tile_depth);
        spdlog::info("[Attention] Accum spad size {}", _config.core_config[target_core].accum_spad_size KB / _config.tile_depth);
        mapping.total_loop.C = _dk;
        mapping.total_loop.M = _seq;
        mapping.tile_out_loop.C = 1;
        mapping.tile_in_loop.C = _dk;
        mapping.tile_in_loop.N = q_len * heads_per_kv;
        mapping.tile_in_loop.M = seq_len;
        mapping.tile_out_loop.N = tile_out_loop;
        mapping.total_loop.N = mapping.tile_in_loop.N * tile_out_loop;
        mapping.tile_out_loop.M = ceil_div(mapping.total_loop.M, mapping.tile_in_loop.M);
    }
}

uint32_t Attention::sram_size_needed() { return 0; }