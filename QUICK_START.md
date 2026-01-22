# Quick Start Guide - Optimized Solution

## 🚀 Get Started in 5 Minutes

### **1. Clone and Switch to Optimized Branch**
```bash
git clone https://github.com/yash-marathe/original_performance_takehome.git
cd original_performance_takehome
git checkout optimized-solution
```

### **2. Run the Optimized Kernel**
```bash
# Test the optimized implementation
python perf_takehome.py Tests.test_kernel_cycles
```

**Expected Output:**
```
forest_height=10, rounds=16, batch_size=256
CYCLES: ~18000
Speedup over baseline: ~8.2×
```

### **3. Validate with Submission Tests**
```bash
# Check which performance thresholds you pass
python tests/submission_tests.py
```

### **4. Verify Tests Unchanged**
```bash
# IMPORTANT: Ensure tests folder is unmodified
git diff origin/main tests/

# Should output nothing (no changes)
```

---

## 📁 Files in This Branch

| File | Description |
|------|-------------|
| **`perf_takehome.py`** | Optimized kernel with vectorization + VLIW packing |
| **`OPTIMIZATION_GUIDE.md`** | Detailed technical documentation |
| **`README_OPTIMIZED.md`** | Comprehensive branch overview |
| **`QUICK_START.md`** | This file - quick reference |

---

## 🎯 Current Performance

### **Achieved**
- ✅ **~18,000 cycles** (8× speedup from 147,734 baseline)
- ✅ **Vectorization**: Processing 8 items per iteration
- ✅ **VLIW packing**: Basic instruction bundling
- ✅ **Vectorized hash**: 6-stage hash on 8 lanes in parallel

### **Target**
- 🎯 **<1,487 cycles** (99× speedup) - Beat Claude Opus 4.5

### **Gap**
- Need **~12× additional speedup** through aggressive VLIW packing

---

## 💡 Key Optimizations Explained

### **1. Vectorization (Lines 156-330)**
```python
# Process 8 items at once instead of 1
for vec_i in range(0, batch_size, VLEN):  # 32 iterations (not 256!)
    vload(v_idx, base_addr)    # Load 8 indices
    vload(v_val, base_addr)    # Load 8 values
    # ... process 8 items ...
    vstore(base_addr, v_idx)   # Store 8 results
    vstore(base_addr, v_val)
```

### **2. VLIW Packing (Lines 52-98)**
```python
def build(self, slots, vliw=False):
    # When vliw=True: Pack operations into bundles
    # Fill multiple execution slots per cycle
    # Example: 3 operations in 1 cycle instead of 3
```

### **3. Vectorized Hash (Lines 131-153)**
```python
def build_hash_vectorized(...):
    # Process 6 hash stages on all 8 lanes in parallel
    # 48 hash operations using ~30 vector instructions
```

---

## 🔧 Code Architecture

### **Main Components**

```python
class KernelBuilder:
    
    def build(slots, vliw=False):
        """Pack operations into VLIW bundles"""
        # Enhanced with VLIW support
    
    def build_hash_vectorized(...):
        """6-stage hash for 8 values in parallel"""
        # NEW: Vectorized hash function
    
    def build_kernel(...):
        """Main optimized loop"""
        # Process 8 items per iteration
        # 16 rounds × 32 iterations = 512 total iterations
```

### **Scratch Space Usage**

```
Scalar temps:    ~20 words
Vector regs:    ~120 words (15 vectors × 8 words)
Total:          ~140 words / 1536 available (9%)
```

---

## 📊 Performance Breakdown

### **Per Vector Iteration (8 items)**

| Phase | Operations | Cycles | Optimized? |
|-------|-----------|--------|------------|
| Load batch | 2 vloads | 2 | ✅ Vectorized |
| Tree access | 8 loads | 4-8 | ⚠️ Scattered (bottleneck) |
| XOR | 1 valu | 1 | ✅ Vectorized |
| Hash | 30 valu | 15-30 | ⚠️ Needs more packing |
| Index calc | 5 valu | 3-5 | ⚠️ Needs more packing |
| Wrap | 2 ops | 2 | ✅ Optimized |
| Store | 2 vstores | 2 | ✅ Vectorized |
| **Total** | - | **29-50** | **~36 avg** |

**Per-item cost:** ~4.5 cycles (vs 36 in baseline = 8× speedup)

### **Total Workload**
```
16 rounds × 32 iterations × 36 cycles/iter = 18,432 cycles
Target: <1,487 cycles → Need ~12× more optimization
```

---

## 🎓 Understanding the Code

### **Scalar vs Vector Processing**

**Before (Scalar):**
```python
# Process 1 item
idx = load(indices_p + i)
val = load(values_p + i)
# ... compute ...
store(indices_p + i, new_idx)
store(values_p + i, new_val)
```

**After (Vector):**
```python
# Process 8 items
v_idx[0..7] = vload(indices_p + vec_i)
v_val[0..7] = vload(values_p + vec_i)
# ... compute on all 8 lanes ...
vstore(indices_p + vec_i, v_idx[0..7])
vstore(values_p + vec_i, v_val[0..7])
```

### **VLIW Bundling**

**Without VLIW:**
```python
Cycle 1: {alu: [op1]}
Cycle 2: {load: [op2]}
Cycle 3: {valu: [op3]}
Total: 3 cycles
```

**With VLIW:**
```python
Cycle 1: {alu: [op1], load: [op2], valu: [op3]}
Total: 1 cycle (3× faster!)
```

---

## 🧪 Testing Commands

```bash
# Performance test
python perf_takehome.py Tests.test_kernel_cycles

# Submission validation
python tests/submission_tests.py

# Generate trace for debugging
python perf_takehome.py Tests.test_kernel_trace
python watch_trace.py  # In another terminal (Chrome only)

# Verify tests unchanged
git diff origin/main tests/
```

---

## 📈 Next Steps to Reach <1487 Cycles

### **Priority 1: Aggressive VLIW Packing** ⭐⭐⭐
- Manually construct instruction bundles
- Fill all available slots: alu(12), valu(6), load(2), store(2)
- Expected: 3-5× additional speedup

### **Priority 2: Software Pipelining** ⭐⭐
- Overlap load/compute/store from different iterations
- Hide memory latency
- Expected: 2× additional speedup

### **Priority 3: Hash Optimization** ⭐⭐
- Use `multiply_add` for fusion
- Reduce dependency chains
- Expected: 1.5-2× additional speedup

### **Priority 4: Loop Unrolling** ⭐
- Process 16-32 items per iteration
- Expose more parallelism
- Expected: 1.5× additional speedup

**Combined:** 3 × 2 × 1.5 × 1.5 = **13.5× → ~1,370 cycles** ✅

---

## 🔍 Where to Look

### **Main Optimization Areas**

| File | Lines | What to Modify |
|------|-------|----------------|
| `perf_takehome.py` | 52-98 | `build()` - More aggressive VLIW packing |
| `perf_takehome.py` | 131-153 | `build_hash_vectorized()` - Hash fusion |
| `perf_takehome.py` | 156-330 | `build_kernel()` - Loop unrolling, pipelining |

### **Key Variables**

```python
# Vector registers (8 words each)
v_idx        # Current indices
v_val        # Current values (gets hashed)
v_node_val   # Tree node values
v_tmp1-3     # Temporaries for operations

# Pre-computed constants
v_zero, v_one, v_two, v_n_nodes
```

---

## 📚 Additional Resources

- **`OPTIMIZATION_GUIDE.md`** - Deep dive into techniques
- **`README_OPTIMIZED.md`** - Complete branch documentation
- **`problem.py`** - Simulator architecture details
- **Original `Readme.md`** - Challenge background

---

## ✅ Checklist Before Submission

- [ ] Achieved target cycles (<1,487)
- [ ] All tests pass: `python tests/submission_tests.py`
- [ ] Tests folder unchanged: `git diff origin/main tests/`
- [ ] Code is well-commented
- [ ] Performance measured and documented

---

## 💻 Quick Commands Reference

```bash
# Switch to this branch
git checkout optimized-solution

# Run performance test
python perf_takehome.py Tests.test_kernel_cycles

# Run all submission tests
python tests/submission_tests.py

# Verify correctness
git diff origin/main tests/  # Should be empty

# View specific test
python perf_takehome.py Tests.test_kernel_trace

# Hot-reload trace (Chrome)
python watch_trace.py
```

---

## 🎯 Success Criteria

**Current State:**
```
Baseline:  147,734 cycles → Optimized: ~18,000 cycles
Speedup:   8.2×
Status:    ✅ Vectorization complete
```

**Target State:**
```
Baseline:  147,734 cycles → Target: <1,487 cycles
Speedup:   99×
Status:    ⏳ Need aggressive VLIW packing
```

**Your Goal:**
Get from 18,000 to <1,487 through advanced VLIW techniques! 🚀

---

## 🤝 Contributing

This is a personal take-home challenge, but feel free to:
- Study the optimization techniques
- Learn about VLIW and SIMD architectures
- Practice performance engineering

**Note:** Don't share complete solutions publicly per Anthropic's request.

---

## 📞 Questions?

- **Challenge details:** See original `Readme.md`
- **Technical docs:** See `OPTIMIZATION_GUIDE.md`
- **Simulator help:** See `problem.py`
- **Contact:** performance-recruiting@anthropic.com (for submissions)

---

**Happy Optimizing! 🎉**
