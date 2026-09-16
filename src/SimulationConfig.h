#pragma once

#include <nlohmann/json.hpp>
#include <string>

using json = nlohmann::json;

enum class CoreType { SYSTOLIC_OS, SYSTOLIC_WS };

enum class DramType { SIMPLE, RAMULATOR1, RAMULATOR2 };

enum class IcntType { SIMPLE, BOOKSIM2 };

struct CoreConfig {
  CoreType core_type;
  uint32_t core_width;
  uint32_t core_height;

  /* Vector config*/
  uint32_t vector_process_bit;
  uint32_t layernorm_latency = 1;
  uint32_t softmax_latency = 1;
  uint32_t add_latency = 1;
  uint32_t mul_latency = 1;
  uint32_t mac_latency = 1;
  uint32_t div_latency = 1;
  uint32_t exp_latency = 1;
  uint32_t gelu_latency = 1;
  uint32_t add_tree_latency = 1;
  uint32_t scalar_sqrt_latency = 1;
  uint32_t scalar_add_latency = 1;
  uint32_t scalar_mul_latency = 1;

  /* SRAM config */
  uint32_t sram_width;
  uint32_t spad_size;
  uint32_t accum_spad_size;
};

struct SimulationConfig {
  /* Core config */
  uint32_t num_cores;
  uint32_t core_freq;
  uint32_t core_print_interval;
  struct CoreConfig *core_config;

  /* DRAM config */
  DramType dram_type;
  uint32_t dram_freq;
  uint32_t dram_channels;
  uint32_t dram_req_size;
  uint32_t dram_latency;
  uint32_t dram_size; // in GB
  uint32_t dram_nbl = 1; // busrt length in clock cycles (bust_length 8 in DDR -> 4 nbl)
  uint32_t dram_print_interval;
  std::string dram_config_path;

  /*
   * DRAM organization, used only to decode (pseudochannel, bankgroup, bank,
   * row, column) for the access trace. These must match the org of the
   * Ramulator2 spec in dram_config_path; the defaults below are the shared
   * org of the HBM2_8Gb and HBM3_8Gb presets.
   */
  uint32_t dram_pseudochannels = 2;
  uint32_t dram_bankgroups = 4;
  uint32_t dram_banks = 4;  // banks per bankgroup
  uint32_t dram_rows = 1 << 15;
  uint32_t dram_columns = 1 << 6;
  uint32_t dram_prefetch_size = 2;  // m_internal_prefetch_size
  /*
   * Ramulator2's org.channel_width, which defaults to 64 in both the HBM2 and
   * HBM3 specs. This is NOT org.dq (128); the transaction size that sets the
   * address offset is prefetch_size * channel_width / 8 = 16 B.
   */
  uint32_t dram_channel_width = 64;

  /*
   * Number of tile slots / scratchpad banks. 2 = stock double buffering.
   * Deeper lets more tiles be resident so an independent tile's loads can
   * cover another tile's compute -- but each bank is spad_size/tile_depth,
   * so tiles get correspondingly smaller.
   */
  uint32_t tile_depth = 2;

  /* ICNT config */
  IcntType icnt_type;
  uint32_t icnt_injection_ports_per_core = 1;
  std::string icnt_config_path;
  uint32_t icnt_freq;
  uint32_t icnt_latency;
  uint32_t icnt_print_interval=0;

  /* Sheduler config */
  std::string scheduler_type;

  /* Other configs */
  uint32_t precision;
  uint32_t full_precision = 4;
  std::string layout;

  /*
   * This map stores the partition information: <partition_id, core_id>
   *
   * Note: Each core belongs to one partition. Through these partition IDs,
   * it is possible to assign a specific DNN model to a particular group of cores.
   */
  std::map<uint32_t, std::vector<uint32_t>> partiton_map;

  uint64_t align_address(uint64_t addr) {
    return addr - (addr % dram_req_size);
  }

  float max_systolic_flops(uint32_t id) {
    return core_config[id].core_width * core_config[id].core_height * core_freq * 2 * num_cores / 1000; // GFLOPS
  }

  float max_vector_flops(uint32_t id) {
    return (core_config[id].vector_process_bit >> 3) / precision * 2 * core_freq / 1000; // GFLOPS
  }

  float max_dram_bandwidth() {
    return dram_freq * dram_channels * dram_req_size / dram_nbl / 1000; // GB/s
  }

};