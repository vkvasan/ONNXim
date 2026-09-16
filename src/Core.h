#pragma once
#include <robin_hood.h>

#include <memory>
#include <vector>

#include "Dram.h"
#include "SimulationConfig.h"
#include "Sram.h"
#include "Stat.h"

class Core {
 public:
  static std::unique_ptr<Core> create(uint32_t id, SimulationConfig config);
  Core(uint32_t id, SimulationConfig config);
  virtual ~Core() = default;
  virtual bool running();
  virtual bool can_issue(bool is_accum_tile=false);
  virtual void issue(std::unique_ptr<Tile> tile);
  virtual std::unique_ptr<Tile> pop_finished_tile();

  virtual void cycle();

  virtual bool has_memory_request();
  virtual void pop_memory_request();
  virtual MemoryAccess* top_memory_request();
  virtual void push_memory_response(MemoryAccess* response);
  virtual void print_stats();
  virtual void print_current_stats();

  virtual cycle_type get_compute_cycles() { return _stat_tot_compute_cycle; }

 protected:
  virtual bool can_issue_compute(std::unique_ptr<Instruction>& inst);
  virtual cycle_type get_inst_compute_cycles(std::unique_ptr<Instruction>& inst) = 0;
  virtual void update_stats();
  virtual void finish_compute_pipeline();
  virtual void finish_vector_pipeline();
  virtual void handle_ld_inst_queue();
  virtual void handle_st_inst_queue();
  virtual cycle_type calculate_add_tree_iterations(uint32_t vector_size);
  virtual cycle_type calculate_vector_op_iterations(uint32_t vector_size);

  const uint32_t _id;
  const SimulationConfig _config;

  cycle_type _core_cycle;
  
  cycle_type _stat_idle_cycle;
  cycle_type _stat_tot_idle_cycle = 0;

  cycle_type _stat_systolic_bubble_cycle = 0;
  cycle_type _stat_tot_systolic_bubble_cycle = 0;

  cycle_type _stat_memory_idle_cycle;
  cycle_type _stat_tot_memory_idle_cycle = 0;

  cycle_type _stat_compute_cycle = 0;
  cycle_type _stat_tot_compute_cycle = 0;

  cycle_type _accum_request_rr_cycle;
  cycle_type _max_request_rr_cycle;
  cycle_type _min_request_rr_cycle;
  
  /* Vector Unit Params */
  cycle_type _stat_vec_compute_cycle;
  cycle_type _stat_tot_vec_compute_cycle = 0;

  cycle_type _stat_systolic_active_cycle = 0;
  cycle_type _stat_tot_systolic_active_cycle = 0;

  /*
   * Cycle accounting for the tile pipeline. These partition every core cycle
   * into exactly one bucket, so the sum equals _core_cycle. The existing
   * counters above overlap and leave most cycles unnamed.
   */
  cycle_type _stat_cyc_no_tile = 0;     // no tile resident: starved by scheduler
  cycle_type _stat_cyc_issued = 0;      // an instruction was issued
  cycle_type _stat_cyc_stall_dep = 0;   // tiles resident, nothing issuable
  cycle_type _stat_cyc_slot_free = 0;   // _tiles.size() < 2, could accept work

  /*
   * Load throttle. Without it, every resident tile dumps its whole MOVIN list
   * within a few cycles, so tiles finish loading together and their preloads
   * then run back-to-back with nothing left to fetch. Capping in-flight loads
   * staggers them: tile N+1 starts loading while tile N preloads.
   * 0 = unlimited (stock behaviour).
   */
  uint64_t _outstanding_loads = 0;
  uint64_t _max_outstanding_loads = 0;

  /*
   * Load-ahead gate. A tile may start issuing MOVINs only if fewer than
   * _load_ahead_tiles OLDER resident tiles are still loading (have MOVINs
   * pending or loads in flight). This staggers tiles so that tile N+1 fetches
   * while tile N preloads into the systolic array, instead of every resident
   * tile dumping its loads at once and finishing together.
   * 0 = disabled (stock behaviour).
   */
  uint64_t _load_ahead_tiles = 0;
  uint64_t _next_load_epoch = 1;
  cycle_type _stat_gate_blocks = 0;

  /*
   * Per-tile phase timestamps. Existing counters say nothing about where an
   * individual tile's cycles go, which is what is needed to explain tile
   * retirement latency. Enabled by ONNXIM_TILE_PHASES=<csv>.
   */
  struct TilePhase {
    uint32_t layer_id = 0;
    cycle_type issue = 0;         // handed to this core
    cycle_type first_load = 0;    // first MOVIN issued
    cycle_type loads_done = 0;    // last outstanding load returned
    cycle_type first_compute = 0; // first non-MOVIN/MOVOUT issued
    cycle_type last_inst = 0;     // instruction list emptied
    cycle_type retire = 0;        // moved to _finished_tiles
    uint32_t n_inst = 0;          // instructions the tile carried
    uint32_t n_movin = 0;
    uint32_t n_compute = 0;
    uint32_t n_gemm = 0;   // GEMM / GEMM_PRELOAD
    uint32_t n_vector = 0; // everything else on the vector unit
  };
  robin_hood::unordered_flat_map<uint64_t, TilePhase> _tile_phase;
  FILE* _phase_fp = nullptr;
  robin_hood::unordered_flat_map<uint64_t, uint64_t> _tile_outstanding;

  /* Memory round-trip latency, measured from MemoryAccess::start_cycle. */
  uint64_t _stat_mem_lat_count = 0;
  uint64_t _stat_mem_lat_sum = 0;
  cycle_type _stat_mem_lat_max = 0;
  std::vector<uint64_t> _stat_mem_lat_hist =
      std::vector<uint64_t>(12, 0);  // log2 buckets: <2,<4,...,>=2048
  double _stat_matmul_cycle = 0;
  double _stat_tot_matmul_cycle = 0;

  int _running_layer;
  uint32_t tile_rr = 0;
  /* Explicit bank rotation. Deriving it from _tiles[0] only works for
     tile_depth == 2; with more banks it collides with a live tile. */
  int _last_spad_id = 0;
  int _last_acc_spad_id = 0;
  std::deque<std::unique_ptr<Tile>> _tiles;
  std::queue<std::unique_ptr<Tile>> _finished_tiles;

  std::queue<std::unique_ptr<Instruction>> _compute_pipeline;
  std::queue<std::unique_ptr<Instruction>> _vector_pipeline;

  std::queue<std::unique_ptr<Instruction>> _ld_inst_queue;
  std::queue<std::unique_ptr<Instruction>> _st_inst_queue;
  std::queue<std::unique_ptr<Instruction>> _ex_inst_queue;

  std::queue<MemoryAccess*> _request_queue;
  /* ONNXIM_RR_TILE_ISSUE=1: one sub-queue per in-flight tile (keyed by
     load_epoch), drained round-robin, so several tiles' loads are outstanding
     at once instead of one tile's list draining to completion first. */
  std::map<uint64_t, std::queue<MemoryAccess*>> _rr_queues;
  std::map<uint64_t, std::queue<MemoryAccess*>>::iterator _rr_cursor = _rr_queues.end();
  size_t _rr_size = 0;
  static bool rr_tile_issue();
  std::queue<MemoryAccess*> _response_queue;
  uint32_t _waiting_write_reqs;

  uint32_t _current_layer_id;
  uint32_t _current_fused_op_id;
  Sram _spad;
  Sram _acc_spad;
};