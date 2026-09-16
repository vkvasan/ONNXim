#include "Core.h"
#include "SystolicWS.h"
#include "SystolicOS.h"

#include "helper/HelperFunctions.h"

std::unique_ptr<Core> Core::create(uint32_t id, SimulationConfig config) {
  if (config.core_config[id].core_type == CoreType::SYSTOLIC_WS) {
    return std::make_unique<SystolicWS>(id, config);
  } else if (config.core_config[id].core_type == CoreType::SYSTOLIC_OS) {
    return std::make_unique<SystolicOS>(id, config);
  } else {
      spdlog::error("[Configuration] Invalid core type...!");
    exit(EXIT_FAILURE);
  }
}

Core::Core(uint32_t id, SimulationConfig config)
    : _id(id),
      _config(config),
      _core_cycle(0),
      _stat_idle_cycle(0),
      _stat_memory_idle_cycle(0),
      _stat_vec_compute_cycle(0),
      _stat_matmul_cycle(0),
      _spad(Sram(config, _core_cycle, false, id)),
      _acc_spad(Sram(config, _core_cycle, true, id)) {
  if (const char* mo = std::getenv("ONNXIM_MAX_OUTSTANDING"))
    _max_outstanding_loads = std::strtoull(mo, nullptr, 10);
  if (const char* la = std::getenv("ONNXIM_LOAD_AHEAD_TILES"))
    _load_ahead_tiles = std::strtoull(la, nullptr, 10);
  if (const char* pf = std::getenv("ONNXIM_TILE_PHASES")) {
    std::string path = std::string(pf) + "." + std::to_string(id);
    _phase_fp = fopen(path.c_str(), "w");
    if (_phase_fp)
      fprintf(_phase_fp, "core,layer_id,epoch,issue,first_load,loads_done,"
                         "first_compute,last_inst,retire,n_inst,n_movin,n_compute,n_gemm,n_vector\n");
  }
  _waiting_write_reqs = 0;
  _running_layer = -1;
}

bool Core::can_issue(bool is_accum_tile) {
  return _tiles.size() < _config.tile_depth;  // N-deep tile pipeline
}

void Core::issue(std::unique_ptr<Tile> op) {
  /* ONNXIM_TILE_LOG=1: ground truth of tile->core dispatch (which core runs
     which (layer, N=head, M=sequence tile) and when). */
  static const bool tile_log = [] { const char* v = std::getenv("ONNXIM_TILE_LOG"); return v && v[0]=='1'; }();
  if (tile_log)
    spdlog::info("[TILE] cycle {} core {} layer {} optype {} N {} M {} accum {} ninst {}",
                 _core_cycle, _id, op->layer_id, op->optype, op->batch, op->M, op->accum,
                 op->instructions.size());
  op->stat = {.start_cycle = _core_cycle,
             .cycles = 0,
             .compute_cycles = 0,
             .memory_stall = 0,
             .sram_reads = 0,
             .sram_writes = 0};
  /* Rotate through the scratchpad banks */
  _last_spad_id = (_last_spad_id + 1) % _config.tile_depth;
  int spad_id = _last_spad_id;
  _spad.flush(spad_id);

  /* FIX (upstream bug): the accumulate-sharing test used to compare against
     _current_layer_id/_current_fused_op_id -- the LAST tile issued. When tiles
     from another op interleave between two accumulating tiles of the same op,
     the second one fails that test, rotates to a fresh bank and flushes it,
     while its partner's data sits in the bank it was MOVIN'd into. The tile
     then retires with accum_spad_id pointing at an empty bank and
     Sram::fill() throws robin_hood "key not found" (asserts are compiled out
     in release). Only reachable when memory stalls reorder tile issue, which
     is why it fires under block-major KV and at low core counts.
     Fix: (a) reuse the bank of any LIVE tile from the same fused op, not just
     the last-issued one; (b) never flush a bank a live tile still references. */
  int acc_spad_id = -1;
  if (op->accum) {
    for (auto& t : _tiles)
      if (t->accum && t->layer_id == op->layer_id &&
          t->fused_op_id == op->fused_op_id) {
        acc_spad_id = t->accum_spad_id;
        break;
      }
  }
  if (acc_spad_id < 0) {
    for (int k = 0; k < (int)_config.tile_depth; k++) {
      int cand = (_last_acc_spad_id + 1 + k) % _config.tile_depth;
      bool busy = false;
      for (auto& t : _tiles)
        if (t->accum_spad_id == cand) { busy = true; break; }
      if (!busy) { acc_spad_id = cand; break; }
    }
    if (acc_spad_id < 0)  /* every bank live: can_issue() should prevent this */
      acc_spad_id = (_last_acc_spad_id + 1) % _config.tile_depth;
    _last_acc_spad_id = acc_spad_id;
    _acc_spad.flush(acc_spad_id);
  }
  _current_layer_id = op->layer_id;
  _current_fused_op_id = op->fused_op_id;

  op->spad_id = spad_id;
  op->accum_spad_id = acc_spad_id;
  op->status = Tile::Status::RUNNING;
  op->load_epoch = _next_load_epoch++;
  if (_phase_fp) {
    TilePhase ph; ph.layer_id = op->layer_id; ph.issue = _core_cycle;
    ph.n_inst = op->instructions.size();
    for (auto& in : op->instructions) {
      if (in->opcode == Opcode::MOVIN) ph.n_movin++;
      else if (in->opcode != Opcode::MOVOUT && in->opcode != Opcode::MOVOUT_POOL) {
        ph.n_compute++;
        if (in->opcode == Opcode::GEMM || in->opcode == Opcode::GEMM_PRELOAD)
          ph.n_gemm++;
        else
          ph.n_vector++;
      }
    }
    _tile_phase[op->load_epoch] = ph;
  }
  if (op->skip) {
    op->status = Tile::Status::FINISH;
    _finished_tiles.push(std::move(op));
    return;
  }
  if (_running_layer != op->layer_id) {
    _running_layer = op->layer_id;
  }
  _tiles.push_back(std::move(op));
}

std::unique_ptr<Tile> Core::pop_finished_tile() {
  std::unique_ptr<Tile> result = std::make_unique<Tile>(Tile{});
  result->status = Tile::Status::EMPTY;
  if (_finished_tiles.size() > 0) {
    result = std::move(_finished_tiles.front());
    _finished_tiles.pop();
  }
  return result;
}

void Core::cycle() {
  _core_cycle++;
  _spad.cycle();
  _acc_spad.cycle();
  bool issued_this_cycle = false;
  if (_tiles.empty())
    _stat_cyc_no_tile++;
  if (_tiles.size() < _config.tile_depth)
    _stat_cyc_slot_free++;
  for (int tile_iter = 0; tile_iter < _tiles.size(); tile_iter++) {
    int i = (tile_iter + tile_rr) % _tiles.size();
    if(_tiles[i]->instructions.empty()) 
      continue;
    std::unique_ptr<Instruction>& inst = _tiles[i]->instructions.front();
    if(_tiles[i]->instructions.size() == 1) {
      inst->last_inst = true;
      inst->my_tile = _tiles[i].get();
    }
    inst->spad_id = _tiles[i]->spad_id;
    inst->accum_spad_id = _tiles[i]->accum_spad_id;
    inst->load_epoch = _tiles[i]->load_epoch;
    Sram *buffer;
    int buffer_id;
    if (inst->dest_addr >= ACCUM_SPAD_BASE) {
      buffer = &_acc_spad;
      buffer_id = _tiles[i]->accum_spad_id;
    } else {
      buffer = &_spad;
      buffer_id = _tiles[i]->spad_id;
    }
    bool issued = false;
    /* Capture before dispatch: `inst` is moved-from once issued. */
    const Opcode issued_opcode = inst->opcode;
    const bool was_last_inst = (_tiles[i]->instructions.size() == 1);
    if (inst->opcode == Opcode::MOVIN) {
      /* Throttle: hold this tile's loads until earlier ones have drained. */
      if (_max_outstanding_loads && _outstanding_loads >= _max_outstanding_loads)
        continue;
      /*
       * Load-ahead gate: count OLDER resident tiles that are still loading.
       * A tile is "still loading" if it has loads in flight or MOVINs left to
       * issue. Note this does not restrict THIS tile's own MOVINs at all, so
       * its memory-level parallelism is untouched.
       */
      if (_load_ahead_tiles) {
        uint64_t older_loading = 0;
        for (int j = 0; j < i; j++) {
          auto it = _tile_outstanding.find(_tiles[j]->load_epoch);
          bool in_flight = (it != _tile_outstanding.end() && it->second > 0);
          bool movin_left = !_tiles[j]->instructions.empty() &&
                            _tiles[j]->instructions.front()->opcode == Opcode::MOVIN;
          if (in_flight || movin_left) older_loading++;
        }
        if (older_loading >= _load_ahead_tiles) { _stat_gate_blocks++; continue; }
      }
      /*LD inst queue */
      if (inst->size == 0) {
        spdlog::error("[Core {}] MVIN issue addr: {:x}, size: {:x}", _id, inst->dest_addr, inst->size);
      }
      if (!buffer->check_allocated(inst->dest_addr, buffer_id) &&
          buffer->check_remain(inst->size, buffer_id)) {
        _ld_inst_queue.push(std::move(inst));
        issued = true;
      } else {
        /*Invalid state */
        spdlog::error("Destination allocated: {} Size remain: {}", buffer->check_allocated(inst->dest_addr, buffer_id), buffer->check_remain(inst->size, buffer_id));
        spdlog::error("[Core {}] MVIN issue panic addr: {:x}, size: {} B", _id, inst->dest_addr, inst->size*_config.dram_req_size);
        buffer->print_all(buffer_id);
        exit(EXIT_FAILURE);
      }
    } else if (inst->opcode == Opcode::MOVOUT ||
               inst->opcode == Opcode::MOVOUT_POOL) {
      /* ST inst queue */
      if (buffer->check_hit(inst->dest_addr, buffer_id)) {
        _st_inst_queue.push(std::move(inst));
        issued = true;
      }
    } else {
      /* Ex inst queue */
      if (inst.get() == 0) {
        spdlog::error("null instruction!");
      }
      if(_ex_inst_queue.empty() && can_issue_compute(inst)) {
        _ex_inst_queue.push(std::move(inst));
        issued = true;
      }
    }
    if (issued && _phase_fp) {
      auto ph = _tile_phase.find(_tiles[i]->load_epoch);
      if (ph != _tile_phase.end()) {
        if (issued_opcode == Opcode::MOVIN) {
          if (!ph->second.first_load) ph->second.first_load = _core_cycle;
        } else if (issued_opcode != Opcode::MOVOUT &&
                   issued_opcode != Opcode::MOVOUT_POOL) {
          if (!ph->second.first_compute) ph->second.first_compute = _core_cycle;
        }
        if (was_last_inst) ph->second.last_inst = _core_cycle;
      }
    }
    if (issued) {
      _tiles[i]->instructions.pop_front();
      tile_rr = i;
      issued_this_cycle = true;
      break;
    }
  }
  if (issued_this_cycle)
    _stat_cyc_issued++;
  else if (!_tiles.empty())
    _stat_cyc_stall_dep++;
  for (auto tile = _tiles.begin() ; tile < _tiles.end(); tile++) {
    if ((*tile)->instructions.empty() && (*tile)->inst_finished) {
      (*tile)->status = Tile::Status::FINISH;
      (*tile)->stat.cycles = _core_cycle - (*tile)->stat.start_cycle;
      (*tile)->stat.memory_stall =
          (*tile)->stat.cycles - (*tile)->stat.compute_cycles;
      if (_phase_fp) {
        auto ph = _tile_phase.find((*tile)->load_epoch);
        if (ph != _tile_phase.end()) {
          auto& q = ph->second;
          fprintf(_phase_fp, "%u,%u,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%u,%u,%u,%u,%u\n", _id,
                  q.layer_id, (*tile)->load_epoch, q.issue, q.first_load,
                  q.loads_done, q.first_compute, q.last_inst, _core_cycle,
                  q.n_inst, q.n_movin, q.n_compute, q.n_gemm, q.n_vector);
          _tile_phase.erase(ph);
        }
      }
      _finished_tiles.push(std::move(*tile));
      _tiles.erase(tile);
      break;
    }
  }
  if(_config.core_print_interval && _core_cycle % _config.core_print_interval == 0) {
    print_current_stats();
  }
}

bool Core::running() {
  bool running = false;
  running = running || _tiles.size() > 0;
  running = running || !_compute_pipeline.empty();
  running = running ||
            !_vector_pipeline.empty();  // Vector unit (Might need to modify)
  running = running || _waiting_write_reqs != 0;
  running = running || !_ld_inst_queue.empty();
  running = running || !_st_inst_queue.empty();
  running = running || !_ex_inst_queue.empty();
  return running;
}

/* ROUND-ROBIN TILE ISSUE (ONNXIM_RR_TILE_ISSUE=1).
   Stock: one FIFO per core, so a MOVIN's whole address list drains before the
   next tile's begins. With a bank-isolated KV layout that means exactly one
   head is streaming per core, hence one bank live per core (measured: 4 of 32
   at 4 cores), and the channel has too few open rows to hide a row switch.
   Here each in-flight tile gets its own sub-queue, keyed by load_epoch, and
   the core hands the interconnect one request from each in turn. Several
   heads are then outstanding at once; under head->bank placement they occupy
   different banks by construction, so bank parallelism rises with no loss of
   row locality. Address order within a tile is unchanged. */
bool Core::rr_tile_issue() {
  static const bool e = [] {
    const char* v = std::getenv("ONNXIM_RR_TILE_ISSUE");
    return v && v[0] == '1';
  }();
  return e;
}

bool Core::has_memory_request() {
  return rr_tile_issue() ? _rr_size > 0 : _request_queue.size() > 0;
}

MemoryAccess* Core::top_memory_request() {
  if (!rr_tile_issue()) return _request_queue.front();
  /* advance the cursor to the next non-empty sub-queue */
  if (_rr_cursor == _rr_queues.end()) _rr_cursor = _rr_queues.begin();
  for (size_t i = 0; i <= _rr_queues.size(); i++) {
    if (_rr_cursor == _rr_queues.end()) _rr_cursor = _rr_queues.begin();
    if (!_rr_cursor->second.empty()) return _rr_cursor->second.front();
    _rr_cursor++;
  }
  assert(0);
  return nullptr;
}

void Core::pop_memory_request() {
  assert(has_memory_request());
  if (!rr_tile_issue()) { _request_queue.pop(); return; }
  _rr_cursor->second.pop();
  _rr_size--;
  if (_rr_cursor->second.empty()) _rr_cursor = _rr_queues.erase(_rr_cursor);
  else _rr_cursor++;                       /* rotate after each grant */
}

void Core::push_memory_response(MemoryAccess *response) {
  assert(!response->request);  // can only push response
  /* Round-trip latency as the core sees it, including queueing at the DRAM. */
  cycle_type latency = _core_cycle - response->start_cycle;
  _stat_mem_lat_count++;
  _stat_mem_lat_sum += latency;
  _stat_mem_lat_max = MAX(_stat_mem_lat_max, latency);
  int bucket = 0;
  for (cycle_type edge = 2; bucket < (int)_stat_mem_lat_hist.size() - 1 && latency >= edge;
       edge <<= 1)
    bucket++;
  _stat_mem_lat_hist[bucket]++;
  if (response->write) {
    _waiting_write_reqs--;
  } else {
    /* A returning load frees throttle credit AND must still fill the SRAM. */
    if (_outstanding_loads > 0) _outstanding_loads--;
    if (response->load_epoch) {
      auto it = _tile_outstanding.find(response->load_epoch);
      if (it != _tile_outstanding.end() && --(it->second) == 0) {
        _tile_outstanding.erase(it);
        if (_phase_fp) {
          auto ph = _tile_phase.find(response->load_epoch);
          if (ph != _tile_phase.end()) ph->second.loads_done = _core_cycle;
        }
      }
    }
    if (response->spad_address >= ACCUM_SPAD_BASE) {
      _acc_spad.fill(response->spad_address, response->buffer_id);
    } else {
      assert(_spad.check_allocated(response->spad_address, response->buffer_id));
      _spad.fill(response->spad_address, response->buffer_id);
    }
  }
  delete response;
}

bool Core::can_issue_compute(std::unique_ptr<Instruction>& inst) {
  bool result = true;

  for (addr_type addr : inst->src_addrs) {
    if (inst->src_from_accum && addr >= ACCUM_SPAD_BASE) {
      result = result && _acc_spad.check_hit(addr, inst->accum_spad_id);
    } else {
      result = result && _spad.check_hit(addr, inst->spad_id);
    }
  }
  if (!result) {
    for (addr_type addr : inst->src_addrs) {
      spdlog::trace("Core[{}] Dependency fail : {} , {}", _id, addr,
                    _spad.check_hit(addr, inst->spad_id));
    }
  }
  return result;
}

void Core::print_stats() {
  update_stats();
  spdlog::info(
      "Core [{}] : MatMul active cycle {} Vector active cycle {} ",
      _id, _stat_tot_matmul_cycle, _stat_tot_vec_compute_cycle);

  spdlog::info(
      "Core [{}] : Memory unit idle cycle {} Systolic bubble cycle {} "
      "Core idle cycle {} ",
      _id, _stat_tot_memory_idle_cycle, _stat_tot_systolic_bubble_cycle, _stat_tot_idle_cycle);

  spdlog::info("Core [{}] : Systolic Array Utilization(%) {:.2f} ({:.2f}% PE util), Vector Unit Utilization(%) {:.2f}, Total cycle: {}",
      _id, static_cast<float>(_stat_tot_systolic_active_cycle * 100) / _core_cycle,
      static_cast<float>(_stat_tot_matmul_cycle * 100) / _core_cycle,
      static_cast<float>(_stat_tot_vec_compute_cycle * 100) / _core_cycle, _core_cycle);

  /* Partition of every core cycle -- these three sum to _core_cycle. */
  auto pct = [&](cycle_type c) { return static_cast<float>(c * 100) / _core_cycle; };
  spdlog::info(
      "Core [{}] : CYCLES inst-issued {} ({:.1f}%) | stall-no-issuable-inst {} "
      "({:.1f}%) | no-tile-resident {} ({:.1f}%) [sum {}]",
      _id, _stat_cyc_issued, pct(_stat_cyc_issued), _stat_cyc_stall_dep,
      pct(_stat_cyc_stall_dep), _stat_cyc_no_tile, pct(_stat_cyc_no_tile),
      _stat_cyc_issued + _stat_cyc_stall_dep + _stat_cyc_no_tile);
  spdlog::info("Core [{}] : tile slot free (<2 resident) {} ({:.1f}%)", _id,
               _stat_cyc_slot_free, pct(_stat_cyc_slot_free));
  spdlog::info("Core [{}] : load-ahead gate blocked a MOVIN on {} cycles ({:.1f}%)",
               _id, _stat_gate_blocks, pct(_stat_gate_blocks));

  if (_stat_mem_lat_count > 0) {
    spdlog::info(
        "Core [{}] : MEM round-trip latency avg {:.1f} max {} over {} accesses "
        "(core cycles)",
        _id, static_cast<double>(_stat_mem_lat_sum) / _stat_mem_lat_count,
        _stat_mem_lat_max, _stat_mem_lat_count);
    std::string hist;
    cycle_type edge = 1;
    for (size_t b = 0; b < _stat_mem_lat_hist.size(); b++) {
      if (_stat_mem_lat_hist[b])
        hist += fmt::format(" {}{}:{:.1f}%",
                            b + 1 == _stat_mem_lat_hist.size() ? ">=" : "<",
                            b + 1 == _stat_mem_lat_hist.size() ? edge : edge * 2,
                            static_cast<float>(_stat_mem_lat_hist[b] * 100) /
                                _stat_mem_lat_count);
      edge <<= 1;
    }
    spdlog::info("Core [{}] : MEM latency hist{}", _id, hist);
  }
}

void Core::print_current_stats() {
  auto level = spdlog::level::info;
  if(_id != 0) 
    level = spdlog::level::debug;
    spdlog::log(level,
      "Core [{}] : MatMul active cycle {} Vector active cycle {} ",
      _id, _stat_matmul_cycle, _stat_vec_compute_cycle);

  spdlog::log(level,
      "Core [{}] : issued tile {} ", _id, _tiles.size());

  spdlog::log(level,
      "Core [{}] : Memory unit idle cycle {} Systolic bubble cycle {} "
      "Core idle cycle {} ",
      _id, _stat_memory_idle_cycle, _stat_systolic_bubble_cycle, _stat_idle_cycle);
  spdlog::log(level,"Core [{}] : Systolic Array Utilization(%) {:.2f} ({:.2f}% PE util), Vector Unit Utilization(%) {:.2f}, Total cycle: {}",
      _id, static_cast<float>(_stat_systolic_active_cycle * 100) / _config.core_print_interval,
      static_cast<float>(_stat_matmul_cycle * 100) / _config.core_print_interval,
      static_cast<float>(_stat_vec_compute_cycle * 100) / _config.core_print_interval, _core_cycle);
  update_stats();
}

void Core::update_stats() {
  _stat_tot_compute_cycle += _stat_compute_cycle;
  _stat_tot_systolic_active_cycle += _stat_systolic_active_cycle;
  _stat_tot_systolic_bubble_cycle += _stat_systolic_bubble_cycle;
  _stat_tot_memory_idle_cycle += _stat_memory_idle_cycle;
  _stat_tot_idle_cycle += _stat_idle_cycle;
  _stat_tot_vec_compute_cycle += _stat_vec_compute_cycle;
  _stat_tot_matmul_cycle += _stat_matmul_cycle;
  _stat_compute_cycle = 0;
  _stat_systolic_active_cycle = 0;
  _stat_systolic_bubble_cycle = 0;
  _stat_memory_idle_cycle = 0;
  _stat_idle_cycle = 0;
  _stat_vec_compute_cycle = 0;
  _stat_matmul_cycle = 0;
}

void Core::finish_compute_pipeline(){
  if (!_compute_pipeline.empty() &&
      _compute_pipeline.front()->finish_cycle <= _core_cycle) {
    std::unique_ptr<Instruction> inst = std::move(_compute_pipeline.front());
    if (inst->dest_addr >= ACCUM_SPAD_BASE)
      _acc_spad.fill(inst->dest_addr, inst->accum_spad_id);
    else
      _spad.fill(inst->dest_addr, inst->spad_id);
    if(inst->last_inst) {
      spdlog::trace("Finished last GEMM {}", inst->spad_id);
      inst->my_tile->inst_finished = true;
    }
    double compute_size = inst->tile_k * inst->tile_m * inst->tile_n
                            / (_config.core_config[_id].core_height * _config.core_config[_id].core_width);
    spdlog::trace("Compute size {} tile m {} tile k {} tile n {}", inst->compute_size, inst->tile_m, inst->tile_k, inst->tile_n);
    spdlog::trace("Compute size {} , compute time {}", compute_size, inst->finish_cycle - inst->start_cycle);
    _stat_matmul_cycle += compute_size;
    _compute_pipeline.pop();
  }
}

void Core::finish_vector_pipeline() {
  if (!_vector_pipeline.empty() &&
      _vector_pipeline.front()->finish_cycle <= _core_cycle) {
    std::unique_ptr<Instruction> inst = std::move(_vector_pipeline.front());
    if (inst->dest_addr >= ACCUM_SPAD_BASE) {
      if(!_acc_spad.check_allocated(inst->dest_addr, inst->accum_spad_id)) {
        spdlog::error("Vector pipeline -> accum");
        spdlog::error("Destination not allocated {}", inst->dest_addr);
      }
      _acc_spad.fill(inst->dest_addr, inst->accum_spad_id);
    }
    else {
      if(!_spad.check_allocated(inst->dest_addr, inst->accum_spad_id)) {
        spdlog::error("Vector pipeline -> spad");
        spdlog::error("Destination not allocated {}", inst->dest_addr);
      }
      _spad.fill(inst->dest_addr, inst->spad_id);
    }
      
    if(inst->last_inst)
      inst->my_tile->inst_finished = true;
    _vector_pipeline.pop();
  }
}

void Core::handle_ld_inst_queue() {
  if (!_ld_inst_queue.empty()) {
    std::unique_ptr<Instruction> front = std::move(_ld_inst_queue.front());
    if (front->opcode == Opcode::MOVIN) {
      uint64_t epoch = front->load_epoch;
      bool prefetched = false;
      Sram *buffer;
      int buffer_id;
      if (front->dest_addr >= ACCUM_SPAD_BASE) {
        buffer = &_acc_spad;
        buffer_id = front->accum_spad_id;
      } else {
        buffer = &_spad;
        buffer_id = front->spad_id;
      }
      if (front->size==0) {
        spdlog::error("Destination size is 0! opcode: {}, addr: 0x{:x}", (int)front->opcode, front->dest_addr);
      }
      int ret = buffer->prefetch(front->dest_addr, buffer_id, front->size, front->size);
      if (!ret) {
        spdlog::error("Destination allocated: {} Size remain: {}", buffer->check_allocated(front->dest_addr, buffer_id), buffer->check_remain(front->size, buffer_id));
        spdlog::error("instruction panic opcode: {:x}, addr: {:x}, size: {} B", (int)front->opcode, front->dest_addr, front->size*_config.dram_req_size);
        std::exit(EXIT_FAILURE);
      }
      for (addr_type addr : front->src_addrs) {
        assert(front->base_addr != GARBEGE_ADDR);
        MemoryAccess *access =
            new MemoryAccess({.id = generate_mem_access_id(),
                              .dram_address = addr + front->base_addr,
                              .spad_address = front->dest_addr,
                              .size = _config.dram_req_size,
                              .write = false,
                              .request = true,
                              .core_id = _id,
                              .start_cycle = _core_cycle,
                              .buffer_id = buffer_id,
                              .load_epoch = epoch,
                              .operand = front->operand_id});
        if (rr_tile_issue()) { _rr_queues[epoch].push(access); _rr_size++; }
        else _request_queue.push(access);
        _outstanding_loads++;
        if (epoch) _tile_outstanding[epoch]++;
      }
      _ld_inst_queue.pop();
    } else {
      assert(0);
    }
  }
}

void Core::handle_st_inst_queue() {
  if (!_st_inst_queue.empty()) {
    std::unique_ptr<Instruction> front = std::move(_st_inst_queue.front());
    if (front->opcode == Opcode::MOVOUT || front->opcode == Opcode::MOVOUT_POOL) {
      Sram *buffer;
      int buffer_id;
      if (front->dest_addr >= ACCUM_SPAD_BASE) {
        buffer = &_acc_spad;
        buffer_id = front->accum_spad_id;
      } else {
        buffer = &_spad;
        buffer_id = front->spad_id;
      }
      if(buffer->check_hit(front->dest_addr, buffer_id)) {
        for (addr_type addr : front->src_addrs) {
          assert(front->base_addr != GARBEGE_ADDR);
          MemoryAccess *access =
              new MemoryAccess{.id = generate_mem_access_id(),
                              .dram_address = addr + front->base_addr,
                              .spad_address = front->dest_addr,
                              .size = _config.dram_req_size,
                              .write = true,
                              .request = true,
                              .core_id = _id,
                              .start_cycle = _core_cycle,
                              .buffer_id = buffer_id};
          _waiting_write_reqs++;
          /* stores share one sub-queue: they are a small share of traffic and
             carry no load_epoch */
          if (rr_tile_issue()) { _rr_queues[0].push(access); _rr_size++; }
          else _request_queue.push(access);
        }
        if(front->last_inst) {
          spdlog::trace("Finished last store {}", front->spad_id);
          front->my_tile->inst_finished = true;
        }
        _st_inst_queue.pop();
      }
    } else {
      assert(0);
    }
  }
}

cycle_type Core::calculate_add_tree_iterations(uint32_t vector_size) {
  uint32_t calculation_unit = _config.core_config[_id].vector_process_bit >> 3;
  if (vector_size <= calculation_unit) {
    return 1;
  }

  uint32_t ret = vector_size / calculation_unit;
  if (vector_size % calculation_unit != 0) {
    ret++;
  }
  return ret + calculate_add_tree_iterations(ret);
}

cycle_type Core::calculate_vector_op_iterations(uint32_t vector_size) {
  uint32_t calculation_unit = _config.core_config[_id].vector_process_bit >> 3;
  uint32_t ret = vector_size / calculation_unit;
  if (vector_size % calculation_unit != 0) {
    ret++;
  }
  return ret;
}