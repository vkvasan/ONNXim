#ifndef DRAM_H
#define DRAM_H
#include <robin_hood.h>
#include <cstdint>
#include <queue>
#include <utility>

#include "Common.h"
#include "ramulator/Ramulator.hpp"
#include "ramulator2.hh"


class Dram {
 public:
  virtual ~Dram() = default;
  virtual bool running() = 0;
  virtual void cycle() = 0;
  virtual bool is_full(uint32_t cid, MemoryAccess* request) = 0;
  virtual void push(uint32_t cid, MemoryAccess* request) = 0;
  virtual bool is_empty(uint32_t cid) = 0;
  virtual MemoryAccess* top(uint32_t cid) = 0;
  virtual void pop(uint32_t cid) = 0;
  uint32_t get_channel_id(MemoryAccess* request);
  virtual void print_stat() {}

 protected:
  SimulationConfig _config;
  uint32_t _n_ch;
  cycle_type _cycles;
};

class SimpleDram : public Dram {
 public:
  SimpleDram(SimulationConfig config);
  virtual bool running() override;
  virtual void cycle() override;
  virtual bool is_full(uint32_t cid, MemoryAccess* request) override;
  virtual void push(uint32_t cid, MemoryAccess* request) override;
  virtual bool is_empty(uint32_t cid) override;
  virtual MemoryAccess* top(uint32_t cid) override;
  virtual void pop(uint32_t cid) override;

 private:
  uint32_t _latency;
  double _bandwidth;

  uint64_t _last_finish_cycle;
  std::vector<std::queue<std::pair<addr_type, MemoryAccess*>>> _waiting_queue;
  std::vector<std::queue<MemoryAccess*>> _response_queue;
};

class DramRamulator : public Dram {
 public:
  DramRamulator(SimulationConfig config);

  virtual bool running() override;
  virtual void cycle() override;
  virtual bool is_full(uint32_t cid, MemoryAccess* request) override;
  virtual void push(uint32_t cid, MemoryAccess* request) override;
  virtual bool is_empty(uint32_t cid) override;
  virtual MemoryAccess* top(uint32_t cid) override;
  virtual void pop(uint32_t cid) override;
  virtual void print_stat() override;

 private:
  std::unique_ptr<ram::Ramulator> _mem;
  robin_hood::unordered_flat_map<uint64_t, MemoryAccess*> _waiting_mem_access;
  std::queue<MemoryAccess*> _responses;

  std::vector<uint64_t> _total_processed_requests;
  std::vector<uint64_t> _processed_requests;
};

class DramRamulator2 : public Dram {
 public:
  DramRamulator2(SimulationConfig config);
  ~DramRamulator2() override;

  virtual bool running() override;
  virtual void cycle() override;
  virtual bool is_full(uint32_t cid, MemoryAccess* request) override;
  virtual void push(uint32_t cid, MemoryAccess* request) override;
  virtual bool is_empty(uint32_t cid) override;
  virtual MemoryAccess* top(uint32_t cid) override;
  virtual void pop(uint32_t cid) override;
  virtual void print_stat() override;

 private:
  std::vector<std::unique_ptr<NDPSim::Ramulator2>> _mem;
  int _tx_ch_log2;
  int _tx_log2;
  int _req_size;

  /*
   * Access trace. Enabled by the ONNXIM_DRAM_TRACE env var, which names the
   * output CSV. ONNXIM_DRAM_TRACE_LIMIT caps the number of rows (0 = no cap).
   */
  void open_trace();
  void log_access(uint32_t cid, addr_type ram_addr, MemoryAccess* request);

  /*
   * Occupancy: is the DRAM starved of work, or does it have a backlog it cannot
   * drain fast enough? Counts requests handed to Ramulator2 but not yet
   * returned, per channel, sampled every DRAM cycle.
   */
  std::vector<int64_t> _outstanding;
  std::vector<uint64_t> _ch_busy_cycles;
  std::vector<uint64_t> _ch_unserved_cycles;
  uint64_t _sum_unserved = 0;
  uint64_t _sum_returns = 0;
  uint64_t _sum_outstanding = 0;
  uint64_t _peak_outstanding = 0;

  FILE* _trace_fp = nullptr;
  FILE* _occ_fp = nullptr;
  uint64_t _trace_limit = 0;
  uint64_t _trace_rows = 0;
  bool _trace_truncated = false;
  /* Bit widths of each level, mirroring Ramulator2's LinearMapperBase::setup */
  int _tx_offset_log2 = 0;
  int _bits_pch = 0;
  int _bits_bg = 0;
  int _bits_ba = 0;
  int _bits_ro = 0;
  int _bits_co = 0;
};
#endif
