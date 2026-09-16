#include "Dram.h"

#include <cstdio>
#include <cstdlib>

#include "helper/HelperFunctions.h"
#include "Hashing.h"

uint32_t Dram::get_channel_id(MemoryAccess* access) {
  uint32_t channel_id;
  /* ipoly_hash_function only implements 16/32/64 sets and exits(1) silently
     for anything else, which killed every 48-channel (LPDDR6) run. Hash over
     the next power of two >= n_ch and fold with a modulo. */
  const new_addr_type blk = (new_addr_type)access->dram_address / _config.dram_req_size;
  if (_n_ch == 16 || _n_ch == 32 || _n_ch == 64)
    channel_id = ipoly_hash_function(blk, 0, _n_ch);
  else if (_n_ch < 16)
    channel_id = ipoly_hash_function(blk, 0, 16) % _n_ch;
  else
    channel_id = blk % _n_ch;   /* plain block interleave: uniform for any channel count (a hash mod 48 would double-load 16 channels) */
  return channel_id;
}

/* FIXME: Simple DRAM has bugs */
SimpleDram::SimpleDram(SimulationConfig config)
    : _latency(config.dram_latency) {
  _cycles = 0;
  _config = config;
  _n_ch = config.dram_channels;
  _waiting_queue.resize(_n_ch);
  _response_queue.resize(_n_ch);
}

bool SimpleDram::running() { return false; }

void SimpleDram::cycle() {
  for (uint32_t ch = 0; ch < _n_ch; ch++) {
    if (!_waiting_queue[ch].empty() &&
        _waiting_queue[ch].front().first <= _cycles) {
      _response_queue[ch].push(_waiting_queue[ch].front().second);
      _waiting_queue[ch].pop();
    }
  }

  _cycles++;
}

bool SimpleDram::is_full(uint32_t cid, MemoryAccess* request) { return false; }

void SimpleDram::push(uint32_t cid, MemoryAccess* request) {
  request->request = false;
  std::pair<uint64_t, MemoryAccess*> entity;
  entity.first = MAX(_cycles + _latency, _last_finish_cycle);
  _last_finish_cycle = entity.first;
  entity.second = request;
  _waiting_queue[cid].push(entity);
}

bool SimpleDram::is_empty(uint32_t cid) { return _response_queue[cid].empty(); }

MemoryAccess* SimpleDram::top(uint32_t cid) {
  assert(!is_empty(cid));
  return _response_queue[cid].front();
}

void SimpleDram::pop(uint32_t cid) {
  assert(!is_empty(cid));
  _response_queue[cid].pop();
}

DramRamulator::DramRamulator(SimulationConfig config)
    : _mem(std::make_unique<ram::Ramulator>(config.dram_config_path,
                                            config.num_cores, false)) {
  _n_ch = config.dram_channels;
  _config = config;
  _cycles = 0;
  _total_processed_requests.resize(_n_ch);
  _processed_requests.resize(_n_ch);
  for (int ch = 0; ch < _n_ch; ch++) {
    _total_processed_requests[ch] = 0;
    _processed_requests[ch] = 0;
  }
}

bool DramRamulator::running() { return false; }

void DramRamulator::cycle() {
  _mem->tick();
  _cycles++;
  int interval = _config.dram_print_interval? _config.dram_print_interval: INT32_MAX;
  int average = 0;
  if (_cycles % interval == 0) {
    for (int ch = 0; ch < _n_ch; ch++) {
      float util = ((float)_processed_requests[ch]) / interval * 100;
      _total_processed_requests[ch] += _processed_requests[ch];
      average += _processed_requests[ch];
      _processed_requests[ch] = 0;
    }
    spdlog::info("Avg DRAM: BW Util {:.2f}%", (float)average / (interval * _n_ch) * 100);
  }
}

bool DramRamulator::is_full(uint32_t cid, MemoryAccess* request) {
  return !_mem->isAvailable(cid, request->dram_address, request->write);
}

void DramRamulator::push(uint32_t cid, MemoryAccess* request) {
  const addr_type atomic_bytes = _mem->getAtomicBytes();
  const addr_type target_addr = request->dram_address;
  // align address
  const addr_type start_addr = target_addr - (target_addr % atomic_bytes);
  assert(start_addr == target_addr);
  assert(request->size == atomic_bytes);
  int count = 0;
  request->request = false;
  _mem->push(cid, target_addr, request->write, request->core_id, request);
}

bool DramRamulator::is_empty(uint32_t cid) { return _mem->isEmpty(cid); }

MemoryAccess* DramRamulator::top(uint32_t cid) {
  assert(!is_empty(cid));
  return (MemoryAccess*)_mem->top(cid);
}

void DramRamulator::pop(uint32_t cid) {
  assert(!is_empty(cid));
  _mem->pop(cid);
  _processed_requests[cid]++;
}

void DramRamulator::print_stat() {
  uint32_t total_reqs = 0;
  for (int ch = 0; ch < _n_ch; ch++) {
    _total_processed_requests[ch] += _processed_requests[ch];
    float util = ((float)_total_processed_requests[ch]) / _cycles * 100;
    spdlog::info("DRAM CH[{}]: AVG BW Util {:.2f}%", ch, util);
    total_reqs += _total_processed_requests[ch];
  }
  float util = ((float)total_reqs / _n_ch) / _cycles * 100;
  spdlog::info("DRAM: AVG BW Util {:.2f}%", util);
  _mem->print_stats();
}

DramRamulator2::DramRamulator2(SimulationConfig config) {
  _n_ch = config.dram_channels;
  _req_size = config.dram_req_size;
  _config = config;
  _mem.resize(_n_ch);
  for (int ch = 0; ch < _n_ch; ch++) {
    _mem[ch] = std::make_unique<NDPSim::Ramulator2>(
      ch, _n_ch, config.dram_config_path, "Ramulator2", _config.dram_print_interval,
      config.dram_nbl);   /* was hard-coded 1: a 32 B request on a 64-bit channel is 2 cycles,
                             so the per-interval BW print read 2x low (capped at ~50%) */
  }
  _tx_log2 = log2(_req_size);
  _tx_ch_log2 = log2(_n_ch) + _tx_log2;
  /* The base class member is otherwise never advanced for this backend */
  _cycles = 0;
  _outstanding.assign(_n_ch, 0);
  _ch_busy_cycles.assign(_n_ch, 0);
  _ch_unserved_cycles.assign(_n_ch, 0);
  open_trace();
}

DramRamulator2::~DramRamulator2() {
  if (_occ_fp != nullptr) { fclose(_occ_fp); _occ_fp = nullptr; }
  if (_trace_fp != nullptr) {
    fclose(_trace_fp);
    _trace_fp = nullptr;
  }
}

/*
 * Mirrors Ramulator2's LinearMapperBase::setup so that the decoded
 * (pseudochannel, bankgroup, bank, row, column) matches what the simulated
 * address mapper computes for the same address.
 */
void DramRamulator2::open_trace() {
  const char* path = std::getenv("ONNXIM_DRAM_TRACE");
  if (path == nullptr || path[0] == '\0') return;

  const char* limit = std::getenv("ONNXIM_DRAM_TRACE_LIMIT");
  if (limit != nullptr && limit[0] != '\0') _trace_limit = std::strtoull(limit, nullptr, 10);

  _trace_fp = fopen(path, "w");
  if (_trace_fp == nullptr) {
    spdlog::error("[DRAM] could not open access trace {}", path);
    return;
  }

  _bits_pch = log2(_config.dram_pseudochannels);
  _bits_bg = log2(_config.dram_bankgroups);
  _bits_ba = log2(_config.dram_banks);
  _bits_ro = log2(_config.dram_rows);
  /* Column is addressed at prefetch-size granularity */
  /* Floor each log2 separately, as Ramulator2's calc_log2 does: with a
     non-power-of-two prefetch (LPDDR6 BL24) log2(2048)-log2(24) = 6.4 would
     truncate to 6 while the mapper uses 11-4 = 7 column bits. */
  _bits_co = (int)log2(_config.dram_columns) - (int)log2(_config.dram_prefetch_size);
  uint32_t tx_bytes = _config.dram_prefetch_size * _config.dram_channel_width / 8;
  _tx_offset_log2 = log2(tx_bytes);

  if (tx_bytes != _config.dram_req_size) {
    /*
     * Not an error: stock ONNXim already pushes 32 B requests into a DRAM whose
     * transaction granularity is 16 B. Reported so the column field is read
     * with the right granularity in mind.
     */
    spdlog::info(
        "[DRAM] trace decode: Ramulator2 transaction size is {} B while "
        "dram_req_size is {} B",
        tx_bytes, _config.dram_req_size);
  }

  fprintf(_trace_fp, "cycle,channel,pseudochannel,bankgroup,bank,row,column,address,rw,core,operand\n");
  if (const char* op = std::getenv("ONNXIM_DRAM_OCC")) {
    _occ_fp = fopen(op, "w");
    if (_occ_fp) fprintf(_occ_fp, "dram_cycle,unserved,in_flight\n");
  }
  spdlog::info("[DRAM] logging access trace to {}", path);
}

void DramRamulator2::log_access(uint32_t cid, addr_type ram_addr,
                                MemoryAccess* request) {
  if (_trace_fp == nullptr) return;
  if (_trace_limit != 0 && _trace_rows >= _trace_limit) {
    if (!_trace_truncated) {
      spdlog::warn("[DRAM] access trace truncated at {} rows", _trace_limit);
      _trace_truncated = true;
    }
    return;
  }

  /*
   * RoBaRaCoCh consumes bits from the LSB in the order
   * channel, column, pseudochannel, bankgroup, bank, row. The channel field is
   * zero bits wide here because ONNXim instantiates one single-channel
   * Ramulator2 per channel, so the real channel is the ONNXim-side cid.
   */
  addr_type addr = ram_addr >> _tx_offset_log2;
  uint64_t column = addr & ((1ull << _bits_co) - 1);
  addr >>= _bits_co;
  uint64_t pseudochannel = addr & ((1ull << _bits_pch) - 1);
  addr >>= _bits_pch;
  uint64_t bankgroup = addr & ((1ull << _bits_bg) - 1);
  addr >>= _bits_bg;
  uint64_t bank = addr & ((1ull << _bits_ba) - 1);
  addr >>= _bits_ba;
  uint64_t row = addr & ((1ull << _bits_ro) - 1);

  /* operand: Instruction::operand_id of the MOVIN/MOVOUT that produced the
     request (_INPUT_OPERAND=100 + k for input k: attention Q=100, K=101,
     V=102; _OUTPUT_OPERAND=200+; GEMM weight = 101). 0 if untagged. */
  fprintf(_trace_fp, "%lu,%u,%lu,%lu,%lu,%lu,%lu,0x%lx,%c,%u,%u\n", _cycles, cid,
          pseudochannel, bankgroup, bank, row, column,
          (uint64_t)request->dram_address, request->write ? 'W' : 'R',
          request->core_id, request->operand);
  _trace_rows++;
}

bool DramRamulator2::running() {
  return false;
}

void DramRamulator2::cycle() {
  uint64_t in_flight = 0;
  for (int ch = 0; ch < _n_ch; ch++) {
    _mem[ch]->cycle();
    if (_outstanding[ch] > 0) {
      _ch_busy_cycles[ch]++;
      in_flight += _outstanding[ch];
    }
    /*
     * Work the DRAM still owes: everything handed to it minus what it has
     * already finished and parked in the return queue awaiting collection.
     * Only this counts as backlog; the return queue is completed work.
     */
    int64_t unserved = _outstanding[ch] - (int64_t)_mem[ch]->pending_returns();
    if (unserved > 0) {
      _ch_unserved_cycles[ch]++;
      _sum_unserved += unserved;
    }
    _sum_returns += _mem[ch]->pending_returns();
  }
  _sum_outstanding += in_flight;
  _peak_outstanding = MAX(_peak_outstanding, in_flight);
  /* Time-resolved occupancy: is the DRAM idle, or working a backlog? */
  if (_occ_fp != nullptr && (_cycles % 500) == 0) {
    int64_t unserved_now = 0;
    for (int ch = 0; ch < _n_ch; ch++)
      unserved_now += MAX((int64_t)0, _outstanding[ch] - (int64_t)_mem[ch]->pending_returns());
    fprintf(_occ_fp, "%lu,%ld,%lu\n", _cycles, unserved_now, in_flight);
  }
  _cycles++;
}

bool DramRamulator2::is_full(uint32_t cid, MemoryAccess* request) {
  return _mem[cid]->full();
}

void DramRamulator2::push(uint32_t cid, MemoryAccess* request) {
  addr_type atomic_bytes =_config.dram_req_size;
  addr_type target_addr = request->dram_address;
  // align address
  addr_type start_addr = target_addr - (target_addr % atomic_bytes);
  assert(start_addr == target_addr);
  assert(request->size == atomic_bytes);
  target_addr = (target_addr >> _tx_ch_log2) << _tx_log2;
  NDPSim::mem_fetch* mf = new NDPSim::mem_fetch();
  mf->addr = target_addr;
  mf->size = request->size;
  mf->write = request->write;
  mf->request = true;
  mf->origin_data = request;
  mf->operand = request->operand;
  log_access(cid, target_addr, request);
  _outstanding[cid]++;
  _mem[cid]->push(mf);
}

bool DramRamulator2::is_empty(uint32_t cid) { 
  return _mem[cid]->return_queue_top() == NULL;
}

MemoryAccess* DramRamulator2::top(uint32_t cid) {
  assert(!is_empty(cid));
  NDPSim::mem_fetch* mf = _mem[cid]->return_queue_top();
  ((MemoryAccess*)mf->origin_data)->request = false;
  return (MemoryAccess*)mf->origin_data;
}

void DramRamulator2::pop(uint32_t cid) {
  assert(!is_empty(cid));
  NDPSim::mem_fetch* mf = _mem[cid]->return_queue_pop();
  _outstanding[cid]--;
  delete mf;
}

void DramRamulator2::print_stat() {
  for (int ch = 0; ch < _n_ch; ch++) {
    _mem[ch]->print(stdout);
  }
  /* Was the DRAM starved, or backlogged? */
  uint64_t busy = 0;
  for (int ch = 0; ch < _n_ch; ch++) busy += _ch_busy_cycles[ch];
  double ch_busy_frac = (double)busy / ((double)_cycles * _n_ch) * 100.0;
  uint64_t unserved_cyc = 0;
  for (int ch = 0; ch < _n_ch; ch++) unserved_cyc += _ch_unserved_cycles[ch];
  double unserved_frac = (double)unserved_cyc / ((double)_cycles * _n_ch) * 100.0;
  spdlog::info(
      "[DRAM] channel-busy (>=1 request outstanding) {:.1f}% of DRAM cycles | "
      "mean in-flight {:.1f} | peak in-flight {} | dram cycles {}",
      ch_busy_frac, (double)_sum_outstanding / (double)_cycles,
      _peak_outstanding, _cycles);
  spdlog::info(
      "[DRAM] TRUE BACKLOG: channel has unserved work {:.1f}% of cycles | "
      "mean unserved {:.1f} | mean awaiting-collection {:.1f}",
      unserved_frac, (double)_sum_unserved / (double)_cycles,
      (double)_sum_returns / (double)_cycles);
}
