#pragma once
//#include "../tensor/NPUTensor.h"
#include <cstdlib>
#include "Operation.h"
#include "GemmWS.h"

class Attention : public Operation {
   public:
    Attention(SimulationConfig config, Model* model, onnx::NodeProto& node_proto, uint32_t target_core=0);
    Attention(SimulationConfig config, Model* model, std::string name, std::map<std::string, std::string>& attributes, uint32_t target_core=0);
    //std::vector<Ptr<BTensor>> get_outputs(std::vector<Ptr<BTensor>> inputs) override;

    uint32_t _batch_size;
    /* q,k,v shape : (nh,{1,l},dk) / (nh,{l,l+1},dk) / (nh,{l,l+1},dk) */
    std::vector<uint32_t> _query_shape;
    std::vector<uint32_t> _key_shape;
    std::vector<uint32_t> _value_shape;

    std::vector<uint32_t> _weight_shape;
    std::vector<uint32_t> _bias_shape;
    std::vector<uint32_t> _mask_shape;
    std::vector<uint32_t> _kv_cache_shape;
    std::vector<uint32_t> _input_shape;
    std::vector<uint32_t> _output_shape;
    std::vector<uint32_t> _liner_output_shape;
    std::vector<uint32_t> _projection_output_shape;

    GemmWS* _projection_node;
    uint32_t _seq;
    uint32_t _q_len;
    uint32_t _dmodel;
    uint32_t _nh;
    uint32_t _nkvh;
    uint32_t _dk;

    uint32_t _key_projection_id;
    uint32_t _query_projection_id;
    uint32_t _value_projection_id;
    /* For kv cache */
    bool onnx = false;
    bool has_kv_cache = false;
    /* PagedAttention (vLLM-style) KV addressing.
       Contiguous KV is one affine walk, so its DRAM addresses are statically
       derivable. Paging stores KV in fixed-size blocks drawn from a pool and
       reaches them through a block table, so the address sequence becomes
       data-dependent -- the one access class no dense model can produce.
         ONNXIM_KV_BLOCK  tokens per block (0/unset = contiguous, stock)
         ONNXIM_KV_ALLOC  "shuffle" (default) or "seq" (identity, sanity check)
       The table is a permutation WITHIN the already-allocated KV region, so
       addresses stay in bounds; this models a fully-utilised pool. */
    static uint32_t kv_block_tokens();
    static const std::vector<uint32_t>& kv_block_table(uint32_t n_blocks);
    addr_type kv_address(uint32_t head, uint32_t seq_idx, uint32_t d);
    /* ONNXIM_KV_LAYOUT=headbank: head index placed on the DRAM bank bits */
    static bool      kv_headbank_enabled();
    static addr_type kv_headbank_chunk();
    static uint32_t  kv_headbank_k();
    static uint32_t  kv_headbank_nbanks();
    static uint32_t  kv_tile_bank_stride();   /* ONNXIM_KV_TILE_BANK_STRIDE */
    uint32_t         _kv_tile_len = 0;        /* (unused now) tokens per sub-chunk while building a tiled attention op */
    uint32_t         _kv_tile_idx = 0;        /* tile->M of the M-tile being built; keys the tile-bank stride */
    addr_type        kv_headbank_offset(uint32_t head, addr_type p, uint32_t tile = 0);
    addr_type        kv_headbank_pad(addr_type base);
    static addr_type kv_headbank_v_shift();
    static addr_type act_lane(addr_type a);   /* ONNXIM_ACT_LANE: activation bank lane */
    /* Shared block pool (ONNXIM_KV_POOL=1). ONNXim gives every request its own
       KV tensor, so requests never interleave in DRAM -- unlike vLLM, where one
       global pool serves all sequences and a churned free list leaves each
       sequence's blocks scattered among its neighbours'. This rebases every
       request into one common pool and hands out slots round-robin across
       requests, so blocks from different sequences physically interleave. */
    static bool  kv_pool_enabled();
    static uint32_t kv_pool_ordinal(addr_type tensor_base, addr_type& pool_base);
    addr_type kv_address_pooled(uint32_t head, uint32_t seq_idx, uint32_t d,
                                addr_type tensor_base);

    /* FlashAttention (online softmax, S never leaves the chip) vs the classic
       path that materialises S = QK^T to DRAM and reads it back.
       ONNXIM_NONFUSED=1 selects the non-fused path. At decode S is 1xS so the
       two barely differ; at prefill S is SxS, so non-fused is the only workload
       here that generates substantial WRITE traffic. */
    bool use_fused = (std::getenv("ONNXIM_NONFUSED") == nullptr);
    bool need_scale = false;

    std::vector<uint32_t> _heads_per_tile;
    std::vector<uint32_t> _tiles_per_head;
    std::vector<uint32_t> _scale_tiles_per_head;

    void calculate_loops();
    void calculate_loops(Mapping& mapping);

    //void initialize_tiles();
    //void initialize_instructions(Tile &tile, int req_idx, int head_idx, int num_heads);
    void initialize_tiles(MappingTable& mapping_table) override;
    void initialize_onnx_tiles(MappingTable& mapping_table);
    void initialize_non_fused_tiles(MappingTable& mapping_table);
    void initialize_instructions(Tile* tile, Mapping mapping, int head_idx, int num_heads);
    void initialize_instructions(Tile* tile, int head_idx, int num_heads);

    void initialize_scale_instructions(Tile* tile, Mapping mapping, int head_idx, int num_tiles, int query_idx, int num_queries);
   protected:
    uint32_t sram_size_needed();
};