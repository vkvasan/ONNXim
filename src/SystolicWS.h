#include "Core.h"

class SystolicWS : public Core {
 protected:
  /* ONNXIM_VECTOR_DEPTH: how many vector instructions may be in flight.
     1 = stock (strictly one at a time). */
  uint64_t _vector_depth = 1;
  static uint32_t preload_div();
  /* ONNXIM_INST_TRACE=<csv>: per-compute-instruction timeline. */
  FILE* _inst_fp = nullptr;
  uint64_t _stat_preload_warm = 0;
  uint64_t _stat_preload_cold = 0;

 public:
  SystolicWS(uint32_t id, SimulationConfig config);
  virtual void cycle() override;
  virtual void print_stats() override;

 protected:
  virtual bool can_issue_compute(std::unique_ptr<Instruction>& inst) override;
  virtual cycle_type get_inst_compute_cycles(std::unique_ptr<Instruction>& inst) override;
  uint32_t _stat_systolic_inst_issue_count = 0;
  uint32_t _stat_systolic_preload_issue_count = 0;
  cycle_type get_vector_compute_cycles(std::unique_ptr<Instruction>& inst);
};