#include "Sram.h"
#define NUM_PORTS 3

Sram::Sram(SimulationConfig config, const cycle_type& core_cycle, bool accum, uint32_t core_id)
    : _core_cycle(core_cycle) {
  uint32_t banks = config.tile_depth ? config.tile_depth : 2;
  if (!accum) {
    _size = config.core_config[core_id].spad_size KB / banks;
  } else {
    _size = config.core_config[core_id].accum_spad_size KB / banks;
  }
  _current_size.assign(banks, 0);
  _cache_table.resize(banks);
  _data_width = config.dram_req_size;
  int precision = config.precision;
  _accum = accum;
}

bool Sram::check_hit(addr_type address, int buffer_id) {
  if (_cache_table[buffer_id].find(address) == _cache_table[buffer_id].end())
    return false;
  _cache_table[buffer_id].at(address).timestamp = _core_cycle;
  return _cache_table[buffer_id].at(address).valid;
}

bool Sram::check_full(int buffer_id) {
  return _current_size[buffer_id] * _data_width < _size;
}

bool Sram::check_remain(size_t size, int buffer_id) {
  return (_current_size[buffer_id] + size) * _data_width <= _size;
}

bool Sram::check_allocated(addr_type address, int buffer_id) {
  return _cache_table[buffer_id].find(address) != _cache_table[buffer_id].end();
}

void Sram::cycle() {}

void Sram::flush(int buffer_id) {
  _current_size[buffer_id] = 0;
  _cache_table[buffer_id].clear();
  spdlog::trace("{}SRAM[{}] Flush", _accum? "Acc-": "", buffer_id);
}

int Sram::prefetch(addr_type address, int buffer_id, size_t allocated_size,
                    size_t count) {
  if (_cache_table[buffer_id].find(address) == _cache_table[buffer_id].end()) {
    if (!check_remain(allocated_size, buffer_id)) {
      print_all(buffer_id);
      assert(0);
      return 0;
    }
    _current_size[buffer_id] += allocated_size;
  } else if (_cache_table[buffer_id].find(address) !=
                 _cache_table[buffer_id].end() &&
             _accum) {
    assert(_cache_table[buffer_id].at(address).size == allocated_size);
    return 0;
  } else {
    assert(0);
    return 0;
  }

  _cache_table[buffer_id][address] = SramEntry{.valid = false,
                                               .size = allocated_size,
                                               .remain_req_count = count,
                                               .timestamp = _core_cycle};
  return 1;
}

void Sram::fill(addr_type address, int buffer_id) {
  /* DIAGNOSTIC: the asserts below are compiled out in release, so a missing
     key falls through to .at() and throws robin_hood "key not found" with no
     context. Report which bank (if any) actually holds the address: present
     elsewhere => the tile's spad_id disagrees with where the MOVIN allocated
     it (bank-rotation bug); present nowhere => never allocated at all. */
  if (_cache_table[buffer_id].find(address) == _cache_table[buffer_id].end()) {
    int found_in = -1;
    for (size_t b = 0; b < _cache_table.size(); b++)
      if (_cache_table[b].find(address) != _cache_table[b].end()) { found_in = (int)b; break; }
    spdlog::error("[SRAM] fill MISS buf={} addr=0x{:x} entries_in_buf={} "
                  "occupancy={} B / {} B  found_in_buf={}",
                  buffer_id, address, _cache_table[buffer_id].size(),
                  _current_size[buffer_id] * _data_width, _size, found_in);
    std::exit(EXIT_FAILURE);
  }
  assert(check_allocated(address, buffer_id));
  assert(_cache_table[buffer_id].at(address).remain_req_count > 0 &&
         !_cache_table[buffer_id].at(address).valid);
  _cache_table[buffer_id].at(address).remain_req_count--;
  if (_cache_table[buffer_id].at(address).remain_req_count == 0) {
    _cache_table[buffer_id].at(address).valid = true;
    spdlog::trace("MAKE valid {} {}F", buffer_id, address);
  }
}

void Sram::count_up(addr_type address, int buffer_id) {
  assert(check_allocated(address, buffer_id));
  _cache_table[buffer_id].at(address).remain_req_count++;
  if (_cache_table[buffer_id].at(address).valid) {
    _cache_table[buffer_id].at(address).valid = false;
    spdlog::trace("MAKE valid {} {}F", buffer_id, address);
  }
}

void Sram::print_all(int buffer_id) {
  for (auto& [key, val] : _cache_table[buffer_id]) {
    spdlog::info("0x{:x} : {} B", key, val.size * _data_width);
  }
  spdlog::info("Allocated size: {} B", _current_size[buffer_id]*_data_width);
  spdlog::info("Size: {} B", _size);
}