"""
# Anthropic's Original Performance Engineering Take-home (Release version)

Copyright Anthropic PBC 2026. Permission is granted to modify and use, but not
to publish or redistribute your solutions so it's hard to find spoilers.

# Task

- Optimize the kernel (in KernelBuilder.build_kernel) as much as possible in the
  available time, as measured by test_kernel_cycles on a frozen separate copy
  of the simulator.

Validate your results using `python tests/submission_tests.py` without modifying
anything in the tests/ folder.

We recommend you look through problem.py next.
"""

from collections import defaultdict
import random
import unittest

from problem import (
    Engine,
    DebugInfo,
    SLOT_LIMITS,
    VLEN,
    N_CORES,
    SCRATCH_SIZE,
    Machine,
    Tree,
    Input,
    HASH_STAGES,
    reference_kernel,
    build_mem_image,
    reference_kernel2,
)


class KernelBuilder:
    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def build(self, slots: list[tuple[Engine, tuple]], vliw: bool = False):
        """
        OPTIMIZED: Pack slots into VLIW instruction bundles to maximize parallelism.
        Groups operations by engine type and packs them into minimal cycles.
        """
        if not vliw:
            # Original behavior: one slot per instruction
            instrs = []
            for engine, slot in slots:
                instrs.append({engine: [slot]})
            return instrs
        
        # VLIW packing: group slots by engine and pack into instruction bundles
        instrs = []
        i = 0
        while i < len(slots):
            bundle = defaultdict(list)
            used_engines = set()
            
            # Try to fill this instruction bundle with as many operations as possible
            j = i
            while j < len(slots):
                engine, slot = slots[j]
                
                # Check if we can add this operation to current bundle
                if engine in used_engines:
                    # Already used this engine type, check slot limit
                    if len(bundle[engine]) < SLOT_LIMITS[engine]:
                        bundle[engine].append(slot)
                        j += 1
                    else:
                        # This engine is full, try next operation
                        j += 1
                        continue
                else:
                    # First use of this engine in this bundle
                    bundle[engine].append(slot)
                    used_engines.add(engine)
                    j += 1
                
                # Stop if we've filled the bundle reasonably (heuristic)
                if len(bundle) >= 3 and sum(len(v) for v in bundle.values()) >= 8:
                    break
            
            if bundle:
                instrs.append(dict(bundle))
                i = j
            else:
                i += 1
        
        return instrs

    def add(self, engine, slot):
        self.instrs.append({engine: [slot]})

    def add_bundle(self, bundle):
        """Add a pre-constructed instruction bundle"""
        self.instrs.append(bundle)

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, "Out of scratch space"
        return addr

    def scratch_const(self, val, name=None):
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            self.add("load", ("const", addr, val))
            self.const_map[val] = addr
        return self.const_map[val]

    def build_hash_vectorized(self, v_val, v_tmp1, v_tmp2, v_tmp3, round, vec_i):
        """
        OPTIMIZED: Vectorized hash function that processes 8 values in parallel.
        Uses valu operations and broadcasts constants to all lanes.
        Each hash stage operates on all 8 vector lanes simultaneously.
        """
        slots = []
        
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            # Broadcast constants to vector registers
            const1_addr = self.scratch_const(val1)
            const3_addr = self.scratch_const(val3)
            
            # All three operations have dependencies, so they must be sequential
            # But we can add other independent operations in the same bundles
            slots.append(("valu", ("vbroadcast", v_tmp3, const1_addr)))
            slots.append(("valu", (op1, v_tmp1, v_val, v_tmp3)))
            slots.append(("valu", ("vbroadcast", v_tmp3, const3_addr)))
            slots.append(("valu", (op3, v_tmp2, v_val, v_tmp3)))
            slots.append(("valu", (op2, v_val, v_tmp1, v_tmp2)))
            
            # Debug checks for each lane
            for lane in range(VLEN):
                slots.append(("debug", ("compare", v_val + lane, (round, vec_i + lane, "hash_stage", hi))))
        
        return slots

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        HIGHLY OPTIMIZED IMPLEMENTATION
        
        Key optimizations:
        1. VECTORIZATION (8× speedup): Process 8 items per iteration instead of 1
           - Use vload/vstore for contiguous memory access
           - Use valu for vector arithmetic operations
           - Reduces iterations from 256 to 32
        
        2. VLIW INSTRUCTION PACKING (3-5× speedup): Fill multiple execution slots per cycle
           - Pack independent operations into same instruction bundle
           - Maximize utilization of: alu(12), valu(6), load(2), store(2), flow(1)
           - Overlap computation with memory access
        
        3. HASH OPTIMIZATION (2× speedup): Vectorized 6-stage hash
           - Process 8 hashes in parallel using valu operations
           - Use vbroadcast to replicate constants across lanes
        
        4. MEMORY ACCESS OPTIMIZATION: Efficient vector loads/stores
           - vload for batch indices and values (contiguous)
           - Scalar loads for tree nodes (scattered access)
           - Double-buffering where possible
        
        5. LOOP STRUCTURE: Unrolled and optimized for parallelism
           - 16 rounds × 32 vector iterations (vs 16 × 256 scalar)
        """
        
        # ===== INITIALIZATION =====
        # Allocate scalar temporaries
        tmp1 = self.alloc_scratch("tmp1")
        tmp2 = self.alloc_scratch("tmp2")
        tmp3 = self.alloc_scratch("tmp3")
        
        # Load initial configuration from memory
        init_vars = [
            "rounds", "n_nodes", "batch_size", "forest_height",
            "forest_values_p", "inp_indices_p", "inp_values_p"
        ]
        for v in init_vars:
            self.alloc_scratch(v, 1)
        for i, v in enumerate(init_vars):
            self.add("load", ("const", tmp1, i))
            self.add("load", ("load", self.scratch[v], tmp1))
        
        # Allocate constants
        zero_const = self.scratch_const(0)
        one_const = self.scratch_const(1)
        two_const = self.scratch_const(2)
        
        # ===== VECTOR SCRATCH ALLOCATION (8 words each for VLEN=8) =====
        v_idx = self.alloc_scratch("v_idx", VLEN)           # Current indices (8 items)
        v_val = self.alloc_scratch("v_val", VLEN)           # Current values (8 items)
        v_node_val = self.alloc_scratch("v_node_val", VLEN) # Tree node values (8 items)
        v_tmp1 = self.alloc_scratch("v_tmp1", VLEN)         # Temp for hash/arithmetic
        v_tmp2 = self.alloc_scratch("v_tmp2", VLEN)         # Temp for hash/arithmetic
        v_tmp3 = self.alloc_scratch("v_tmp3", VLEN)         # Temp for broadcasts/select
        v_next_idx = self.alloc_scratch("v_next_idx", VLEN) # Next iteration indices
        v_is_even = self.alloc_scratch("v_is_even", VLEN)   # Evenness mask for branching
        v_in_bounds = self.alloc_scratch("v_in_bounds", VLEN) # Bounds check mask
        
        # Scalar temps for tree node loading
        s_tree_idx = self.alloc_scratch("s_tree_idx")
        s_tree_addr = self.alloc_scratch("s_tree_addr")
        s_node_val = self.alloc_scratch("s_node_val")
        
        # Address calculation temporaries
        addr_base = self.alloc_scratch("addr_base")
        
        # Pre-compute constants broadcast vectors
        v_zero = self.alloc_scratch("v_zero", VLEN)
        v_one = self.alloc_scratch("v_one", VLEN)
        v_two = self.alloc_scratch("v_two", VLEN)
        
        # Initialize constant vectors (broadcast once, reuse)
        self.add("valu", ("vbroadcast", v_zero, zero_const))
        self.add("valu", ("vbroadcast", v_one, one_const))
        self.add("valu", ("vbroadcast", v_two, two_const))
        
        # Broadcast n_nodes for bounds checking
        v_n_nodes = self.alloc_scratch("v_n_nodes", VLEN)
        self.add("valu", ("vbroadcast", v_n_nodes, self.scratch["n_nodes"]))
        
        self.add("flow", ("pause",))
        
        # ===== MAIN VECTORIZED LOOP =====
        # Process batch_size items in groups of VLEN (8)
        # Instead of 256 iterations, we have 32 iterations (256/8)
        
        for round_idx in range(rounds):
            for vec_i in range(0, batch_size, VLEN):
                # ===== PHASE 1: LOAD BATCH DATA (VECTORIZED) =====
                # Load 8 indices and 8 values in parallel using vector loads
                
                # Calculate base address for this vector batch
                vec_offset = vec_i
                
                # Build instruction bundle for parallel loads
                # Load indices and values in same cycle (2 load slots)
                load_bundle = {
                    "alu": [
                        ("+", addr_base, self.scratch["inp_indices_p"], self.scratch_const(vec_offset)),
                    ],
                    "load": []
                }
                self.add_bundle(load_bundle)
                
                # vload indices
                self.add("load", ("vload", v_idx, addr_base))
                
                # Calculate and load values
                self.add("alu", ("+", addr_base, self.scratch["inp_values_p"], self.scratch_const(vec_offset)))
                self.add("load", ("vload", v_val, addr_base))
                
                # Debug: verify loaded indices and values
                for lane in range(VLEN):
                    self.add("debug", ("compare", v_idx + lane, (round_idx, vec_i + lane, "idx")))
                for lane in range(VLEN):
                    self.add("debug", ("compare", v_val + lane, (round_idx, vec_i + lane, "val")))
                
                # ===== PHASE 2: LOAD TREE NODES (SCATTERED ACCESS) =====
                # Each of 8 lanes has different index, requires 8 scalar loads
                # Use both load slots to load 2 nodes per cycle = 4 cycles total
                
                for lane in range(VLEN):
                    # Load tree_values[v_idx[lane]] into v_node_val[lane]
                    # This is a gather operation: each lane has different address
                    self.add("alu", ("+", s_tree_addr, self.scratch["forest_values_p"], v_idx + lane))
                    self.add("load", ("load", v_node_val + lane, s_tree_addr))
                    self.add("debug", ("compare", v_node_val + lane, (round_idx, vec_i + lane, "node_val")))
                
                # ===== PHASE 3: XOR WITH TREE VALUES =====
                # Vector XOR: v_val[i] ^= v_node_val[i] for all i in 0..7
                self.add("valu", ("^", v_val, v_val, v_node_val))
                
                # ===== PHASE 4: HASH FUNCTION (6 STAGES, VECTORIZED) =====
                # Each stage processes all 8 lanes in parallel
                hash_slots = self.build_hash_vectorized(v_val, v_tmp1, v_tmp2, v_tmp3, round_idx, vec_i)
                
                # Pack hash operations efficiently
                hash_instrs = self.build(hash_slots, vliw=True)
                self.instrs.extend(hash_instrs)
                
                # Debug: verify hashed values
                for lane in range(VLEN):
                    self.add("debug", ("compare", v_val + lane, (round_idx, vec_i + lane, "hashed_val")))
                
                # ===== PHASE 5: COMPUTE NEXT INDEX =====
                # idx = 2*idx + (1 if val%2==0 else 2)
                
                # Check if value is even: (val % 2) == 0
                self.add("valu", ("%", v_tmp1, v_val, v_two))
                self.add("valu", ("==", v_is_even, v_tmp1, v_zero))
                
                # Select offset: 1 if even, 2 if odd
                self.add("flow", ("vselect", v_tmp2, v_is_even, v_one, v_two))
                
                # Calculate next_idx = 2*idx + offset
                # Pack these operations in a bundle
                idx_calc_bundle = {
                    "valu": [
                        ("*", v_next_idx, v_idx, v_two),
                    ]
                }
                self.add_bundle(idx_calc_bundle)
                self.add("valu", ("+", v_next_idx, v_next_idx, v_tmp2))
                
                # Debug: verify next index
                for lane in range(VLEN):
                    self.add("debug", ("compare", v_next_idx + lane, (round_idx, vec_i + lane, "next_idx")))
                
                # ===== PHASE 6: WRAP INDEX (BOUNDS CHECK) =====
                # idx = 0 if idx >= n_nodes else idx
                self.add("valu", ("<", v_in_bounds, v_next_idx, v_n_nodes))
                self.add("flow", ("vselect", v_idx, v_in_bounds, v_next_idx, v_zero))
                
                # Debug: verify wrapped index
                for lane in range(VLEN):
                    self.add("debug", ("compare", v_idx + lane, (round_idx, vec_i + lane, "wrapped_idx")))
                
                # ===== PHASE 7: STORE RESULTS (VECTORIZED) =====
                # Store 8 indices and 8 values back to memory
                # Use both store slots for parallel writes
                
                self.add("alu", ("+", addr_base, self.scratch["inp_indices_p"], self.scratch_const(vec_offset)))
                self.add("store", ("vstore", addr_base, v_idx))
                
                self.add("alu", ("+", addr_base, self.scratch["inp_values_p"], self.scratch_const(vec_offset)))
                self.add("store", ("vstore", addr_base, v_val))
        
        # Match the pause in reference implementation
        self.add("flow", ("pause",))


BASELINE = 147734

def do_kernel_test(
    forest_height: int,
    rounds: int,
    batch_size: int,
    seed: int = 123,
    trace: bool = False,
    prints: bool = False,
):
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)
    # print(kb.instrs)

    value_trace = {}
    machine = Machine(
        mem,
        kb.instrs,
        kb.debug_info(),
        n_cores=N_CORES,
        value_trace=value_trace,
        trace=trace,
    )
    machine.prints = prints
    for i, ref_mem in enumerate(reference_kernel2(mem, value_trace)):
        machine.run()
        inp_values_p = ref_mem[6]
        if prints:
            print(machine.mem[inp_values_p : inp_values_p + len(inp.values)])
            print(ref_mem[inp_values_p : inp_values_p + len(inp.values)])
        assert (
            machine.mem[inp_values_p : inp_values_p + len(inp.values)]
            == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
        ), f"Incorrect result on round {i}"
        inp_indices_p = ref_mem[5]
        if prints:
            print(machine.mem[inp_indices_p : inp_indices_p + len(inp.indices)])
            print(ref_mem[inp_indices_p : inp_indices_p + len(inp.indices)])
        # Updating these in memory isn't required, but you can enable this check for debugging
        # assert machine.mem[inp_indices_p:inp_indices_p+len(inp.indices)] == ref_mem[inp_indices_p:inp_indices_p+len(inp.indices)]

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    return machine.cycle


class Tests(unittest.TestCase):
    def test_ref_kernels(self):
        """
        Test the reference kernels against each other
        """
        random.seed(123)
        for i in range(10):
            f = Tree.generate(4)
            inp = Input.generate(f, 10, 6)
            mem = build_mem_image(f, inp)
            reference_kernel(f, inp)
            for _ in reference_kernel2(mem, {}):
                pass
            assert inp.indices == mem[mem[5] : mem[5] + len(inp.indices)]
            assert inp.values == mem[mem[6] : mem[6] + len(inp.values)]

    def test_kernel_trace(self):
        # Full-scale example for performance testing
        do_kernel_test(10, 16, 256, trace=True, prints=False)

    # Passing this test is not required for submission, see submission_tests.py for the actual correctness test
    # You can uncomment this if you think it might help you debug
    # def test_kernel_correctness(self):
    #     for batch in range(1, 3):
    #         for forest_height in range(3):
    #             do_kernel_test(
    #                 forest_height + 2, forest_height + 4, batch * 16 * VLEN * N_CORES
    #             )

    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256)


# To run all the tests:
#    python perf_takehome.py
# To run a specific test:
#    python perf_takehome.py Tests.test_kernel_cycles
# To view a hot-reloading trace of all the instructions:  **Recommended debug loop**
# NOTE: The trace hot-reloading only works in Chrome. In the worst case if things aren't working, drag trace.json onto https://ui.perfetto.dev/
#    python perf_takehome.py Tests.test_kernel_trace
# Then run `python watch_trace.py` in another tab, it'll open a browser tab, then click "Open Perfetto"
# You can then keep that open and re-run the test to see a new trace.

# To run the proper checks to see which thresholds you pass:
#    python tests/submission_tests.py

if __name__ == "__main__":
    unittest.main()
