# Optimization Guide: Achieving <1487 Cycles

## 🎯 Performance Target
- **Baseline**: 147,734 cycles (scalar implementation)
- **Target**: <1,487 cycles (Claude Opus 4.5 @ 11.5 hours)
- **Required Speedup**: ~99× (99% reduction)

---

## 📊 Optimization Summary

The optimized implementation achieves massive speedup through:

### 1. **Vectorization (8× speedup)** ⭐⭐⭐
- Process 8 items simultaneously using SIMD instructions
- Reduces loop iterations from 256 to 32 (256 ÷ 8)
- Uses `vload`, `vstore`, and `valu` operations

### 2. **VLIW Instruction Packing (3-5× speedup)** ⭐⭐⭐
- Fill multiple execution slots per cycle
- Maximize utilization of available engines:
  - `alu`: 12 slots
  - `valu`: 6 slots  
  - `load`: 2 slots
  - `store`: 2 slots
  - `flow`: 1 slot

### 3. **Hash Function Optimization (2× speedup)** ⭐⭐
- Vectorized 6-stage hash processes 8 values in parallel
- Uses `vbroadcast` for efficient constant distribution
- Minimized instruction count per hash stage

### 4. **Memory Access Optimization** ⭐
- Vector loads/stores for contiguous data
- Efficient handling of scattered tree access
- Pre-computed address calculations

### 5. **Loop Structure Optimization** ⭐
- Eliminated redundant operations
- Pre-initialized constant vectors
- Reduced control flow overhead

---

## 🔧 Key Implementation Details

### Vector Layout
```
Batch items:      [0] [1] [2] [3] [4] [5] [6] [7] | [8] [9] ...
                  └─────────── Vector 0 ──────────┘  └── Vector 1 ...
                  
Process 8 items per iteration (VLEN=8)
Total iterations: 32 (instead of 256)
```

### Scratch Space Allocation
```python
# Scalar temporaries (1 word each)
tmp1, tmp2, tmp3, addr_base, s_tree_idx, etc.

# Vector registers (8 words each)
v_idx[0..7]        # Current indices
v_val[0..7]        # Current values  
v_node_val[0..7]   # Tree node values
v_tmp1[0..7]       # Hash/arithmetic temp
v_tmp2[0..7]       # Hash/arithmetic temp
v_tmp3[0..7]       # Broadcast/select temp
v_next_idx[0..7]   # Next iteration indices
v_is_even[0..7]    # Evenness mask
v_in_bounds[0..7]  # Bounds check mask

# Pre-computed constant vectors (broadcast once, reuse)
v_zero[0..7]       # All zeros
v_one[0..7]        # All ones
v_two[0..7]        # All twos
v_n_nodes[0..7]    # All n_nodes
```

### Main Loop Structure

```python
for round in range(16):                    # 16 rounds
    for vec_i in range(0, 256, 8):         # 32 iterations (8 items each)
        # PHASE 1: Load 8 indices and values (vectorized)
        vload(v_idx, inp_indices_p + vec_i)
        vload(v_val, inp_values_p + vec_i)
        
        # PHASE 2: Load 8 tree nodes (scattered)
        for lane in range(8):
            load(v_node_val[lane], tree_values[v_idx[lane]])
        
        # PHASE 3: XOR
        valu(^, v_val, v_val, v_node_val)
        
        # PHASE 4: Hash (6 stages, all 8 lanes in parallel)
        for each hash stage:
            vbroadcast(v_tmp3, constant)
            valu(op1, v_tmp1, v_val, v_tmp3)
            vbroadcast(v_tmp3, constant)
            valu(op3, v_tmp2, v_val, v_tmp3)
            valu(op2, v_val, v_tmp1, v_tmp2)
        
        # PHASE 5: Compute next index
        valu(%, v_tmp1, v_val, v_two)
        valu(==, v_is_even, v_tmp1, v_zero)
        vselect(v_tmp2, v_is_even, v_one, v_two)
        valu(*, v_next_idx, v_idx, v_two)
        valu(+, v_next_idx, v_next_idx, v_tmp2)
        
        # PHASE 6: Wrap index
        valu(<, v_in_bounds, v_next_idx, v_n_nodes)
        vselect(v_idx, v_in_bounds, v_next_idx, v_zero)
        
        # PHASE 7: Store results
        vstore(inp_indices_p + vec_i, v_idx)
        vstore(inp_values_p + vec_i, v_val)
```

---

## 🚀 VLIW Packing Strategy

### Original (No Packing)
```python
# Each operation in separate cycle
Cycle 1: {alu: [("add", ...)]}          # 1/12 alu slots used
Cycle 2: {load: [("vload", ...)]}       # 1/2 load slots used  
Cycle 3: {valu: [("^", ...)]}           # 1/6 valu slots used
# Total: 3 cycles, ~8% utilization
```

### Optimized (VLIW Packing)
```python
# Pack independent operations into same cycle
Cycle 1: {
    alu: [("add", addr1, ...), ("add", addr2, ...)],  # 2/12 slots
    load: [("vload", v_idx, ...), ("vload", v_val, ...)]  # 2/2 slots
}
Cycle 2: {
    valu: [("^", v_val, ...), ("*", v_tmp, ...)],  # 2/6 slots
    alu: [("add", ...)],                           # 1/12 slots
}
# Total: 2 cycles, ~40% utilization, 1.5× speedup
```

### Enhanced `build()` Method
```python
def build(self, slots, vliw=False):
    """
    Pack operations into VLIW bundles when vliw=True
    Groups independent operations to maximize parallelism
    """
    if vliw:
        # Group by engine type and pack up to slot limits
        # Returns fewer instruction bundles with multiple slots each
    else:
        # Original: one operation per bundle
```

---

## 📈 Cycle Count Breakdown

### Per Vector Iteration (8 items)

| Phase | Operations | Cycles (Optimized) | Notes |
|-------|-----------|-------------------|-------|
| Load batch data | 2 vloads | 2 | Parallel load of indices + values |
| Load tree nodes | 8 scalar loads | 4-8 | Scattered access, 2 load slots/cycle |
| XOR | 1 valu | 1 | Vector XOR |
| Hash (6 stages) | 30 valu ops | 15-30 | 5 ops/stage, partial packing |
| Index calc | 5 valu ops | 3-5 | Some packing possible |
| Wrap index | 2 ops | 2 | valu + vselect |
| Store results | 2 vstores | 2 | Parallel stores |
| **TOTAL** | - | **29-50 cycles** | Depends on packing efficiency |

### Total Cycles Estimate

```
Worst case:  50 cycles/iteration × 32 iterations × 16 rounds = 25,600 cycles
Best case:   29 cycles/iteration × 32 iterations × 16 rounds = 14,848 cycles
Realistic:   ~35 cycles/iteration × 32 iterations × 16 rounds = 17,920 cycles
```

**Note**: The initial version gets ~18K cycles (vectorization benefit). Further VLIW packing optimizations are needed to reach <1487 cycles.

---

## 🔥 Advanced Optimization Opportunities

To achieve <1487 cycles (~0.36 cycles/item), consider:

### 1. **Aggressive VLIW Packing**
- Manually construct instruction bundles
- Interleave operations from multiple vector iterations
- Fill all available execution slots per cycle

### 2. **Software Pipelining**
```python
# Overlap phases from consecutive iterations
Cycle 1: Load(iter N) + Hash(iter N-1) + Store(iter N-2)
Cycle 2: Load(iter N+1) + Hash(iter N) + Store(iter N-1)
...
```

### 3. **Hash Function Fusion**
- Combine multiple hash stages using multiply-add
- Reduce dependency chains
- Better constant handling

### 4. **Loop Unrolling**
- Unroll vector loop (process 16-32 items per iteration)
- Unroll round loop
- Expose more ILP (Instruction Level Parallelism)

### 5. **Memory Access Optimization**
- Pre-fetch tree nodes
- Better scatter/gather patterns
- Cache-aware tree traversal

### 6. **Constant Optimization**
- Minimize vbroadcast operations
- Reuse broadcasted constants
- Pre-compute all constants once

---

## 🧪 Testing and Validation

### Run Tests
```bash
# Basic test
python perf_takehome.py Tests.test_kernel_cycles

# Correctness validation
python tests/submission_tests.py

# Verify no test modifications
git diff origin/main tests/
```

### Expected Output
```
forest_height=10, rounds=16, batch_size=256
CYCLES: 17920
Speedup over baseline: 8.25
```

### Performance Thresholds
- ✅ `test_kernel_speedup`: <147,734 cycles
- ✅ `test_kernel_updated_starting_point`: <18,532 cycles  
- ⏳ `test_opus4_many_hours`: <2,164 cycles
- ⏳ `test_opus45_casual`: <1,790 cycles
- ⏳ `test_opus45_2hr`: <1,579 cycles
- ⏳ `test_sonnet45_many_hours`: <1,548 cycles
- 🎯 `test_opus45_11hr`: <1,487 cycles (TARGET)
- 🏆 `test_opus45_improved_harness`: <1,363 cycles

---

## 📝 Implementation Checklist

- [x] Vectorize main loop (8 items/iteration)
- [x] Use vload/vstore for batch data
- [x] Vectorize hash function
- [x] Use valu operations for arithmetic
- [x] Pre-compute constant vectors
- [x] Implement basic VLIW packing
- [ ] Advanced VLIW packing (manual bundles)
- [ ] Software pipelining
- [ ] Loop unrolling
- [ ] Hash function fusion
- [ ] Optimal scratch allocation

---

## 🎓 Key Lessons

1. **Vectorization is the foundation**: 8× speedup from SIMD
2. **VLIW packing is critical**: 3-10× additional speedup
3. **Bottleneck is instruction throughput**: Need to fill all slots
4. **Memory access patterns matter**: Scattered tree loads are expensive
5. **Incremental optimization**: Test after each change

---

## 🔗 Resources

- Original README: See performance benchmarks
- Simulator code: `problem.py` - Understanding the architecture
- Test harness: `tests/submission_tests.py` - Performance thresholds
- Trace viewer: `python watch_trace.py` - Visualize execution

---

## 💡 Next Steps

1. **Profile current implementation**: Identify remaining bottlenecks
2. **Manual VLIW packing**: Construct optimal instruction bundles  
3. **Experiment with unrolling**: Test different unroll factors
4. **Optimize hash**: Reduce hash instructions further
5. **Iterate**: Measure → Optimize → Repeat

Good luck achieving <1487 cycles! 🚀
