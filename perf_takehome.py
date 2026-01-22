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
        Build instructions with optional VLIW packing.
        When vliw=True, pack multiple operations from different engines into single cycles.
        """
        if not vliw:
            instrs = []
            for engine, slot in slots:
                instrs.append({engine: [slot]})
            return instrs
        
        # Advanced VLIW packing with dependency tracking
        instrs = []
        current_bundle = {}
        slot_counts = {}
        writes_in_bundle = set()  # Track what scratch addresses are written in current bundle
        reads_in_bundle = set()   # Track what scratch addresses are read in current bundle
        
        def get_slot_reads(engine, slot):
            """Extract scratch addresses read by this slot"""
            reads = set()
            if engine == "alu":
                # alu: (op, dest, src1, src2) - reads src1, src2
                if len(slot) >= 4:
                    reads.add(slot[2])
                    reads.add(slot[3])
            elif engine == "valu":
                # valu operations read from their sources
                if slot[0] == "vbroadcast":
                    reads.add(slot[2])
                elif len(slot) >= 4:
                    # Most valu ops: (op, dest, src1, src2)
                    for i in range(VLEN):
                        reads.add(slot[2] + i)
                        reads.add(slot[3] + i)
            elif engine == "load":
                if slot[0] == "load":
                    reads.add(slot[2])  # addr
                elif slot[0] == "load_offset":
                    reads.add(slot[2] + slot[3])
                elif slot[0] == "vload":
                    reads.add(slot[2])
                # const doesn't read
            elif engine == "store":
                if slot[0] == "store":
                    reads.add(slot[1])  # addr
                    reads.add(slot[2])  # src
                elif slot[0] == "vstore":
                    reads.add(slot[1])  # addr
                    for i in range(VLEN):
                        reads.add(slot[2] + i)  # src vector
            elif engine == "flow":
                if slot[0] == "select":
                    reads.add(slot[2])  # cond
                    reads.add(slot[3])  # a
                    reads.add(slot[4])  # b
                elif slot[0] == "add_imm":
                    reads.add(slot[2])  # a
                elif slot[0] == "vselect":
                    for i in range(VLEN):
                        reads.add(slot[2] + i)  # cond
                        reads.add(slot[3] + i)  # a
                        reads.add(slot[4] + i)  # b
                elif slot[0] == "cond_jump":
                    reads.add(slot[1])  # cond
                elif slot[0] == "cond_jump_rel":
                    reads.add(slot[1])  # cond
                elif slot[0] == "jump_indirect":
                    reads.add(slot[1])  # addr
                elif slot[0] == "trace_write":
                    reads.add(slot[1])  # val
            return reads
        
        def get_slot_writes(engine, slot):
            """Extract scratch addresses written by this slot"""
            writes = set()
            if engine == "alu":
                writes.add(slot[1])  # dest
            elif engine == "valu":
                # Most valu ops write to dest
                for i in range(VLEN):
                    writes.add(slot[1] + i)
            elif engine == "load":
                if slot[0] in ["load", "const"]:
                    writes.add(slot[1])  # dest
                elif slot[0] == "load_offset":
                    writes.add(slot[1] + slot[3])
                elif slot[0] == "vload":
                    for i in range(VLEN):
                        writes.add(slot[1] + i)
            elif engine == "flow":
                if slot[0] in ["select", "add_imm", "coreid"]:
                    writes.add(slot[1])  # dest
                elif slot[0] == "vselect":
                    for i in range(VLEN):
                        writes.add(slot[1] + i)
            return writes
        
        def has_dependency(engine, slot):
            """Check if this slot has a dependency on current bundle"""
            reads = get_slot_reads(engine, slot)
            slot_writes = get_slot_writes(engine, slot)
            
            # Check RAW (Read After Write): can't read what current bundle writes
            if reads & writes_in_bundle:
                return True
            
            # Check WAW (Write After Write): can't write to same location
            if slot_writes & writes_in_bundle:
                return True
            
            # Check WAR (Write After Read): can't write what current bundle reads
            # This is less critical in VLIW since effects happen at end of cycle
            # but let's be conservative
            if slot_writes & reads_in_bundle:
                return True
            
            return False
        
        def flush_bundle():
            """Flush current bundle to instructions"""
            nonlocal current_bundle, slot_counts, writes_in_bundle, reads_in_bundle
            if current_bundle:
                instrs.append(current_bundle)
            current_bundle = {}
            slot_counts = {}
            writes_in_bundle = set()
            reads_in_bundle = set()
        
        for engine, slot in slots:
            # Skip debug in submission mode
            if engine == "debug":
                continue
            
            # Check slot limits
            count = slot_counts.get(engine, 0)
            if count >= SLOT_LIMITS[engine]:
                flush_bundle()
                count = 0
            
            # Check dependencies
            if has_dependency(engine, slot):
                flush_bundle()
                count = 0
            
            # Add slot to current bundle
            if engine not in current_bundle:
                current_bundle[engine] = []
            current_bundle[engine].append(slot)
            slot_counts[engine] = count + 1
            
            # Track reads and writes
            writes_in_bundle.update(get_slot_writes(engine, slot))
            reads_in_bundle.update(get_slot_reads(engine, slot))
        
        # Flush remaining bundle
        flush_bundle()
        
        return instrs

    def add(self, engine, slot):
        self.instrs.append({engine: [slot]})

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

    def build_hash(self, val_hash_addr, tmp1, tmp2, round, i):
        slots = []

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            slots.append(("alu", (op1, tmp1, val_hash_addr, self.scratch_const(val1))))
            slots.append(("alu", (op3, tmp2, val_hash_addr, self.scratch_const(val3))))
            slots.append(("alu", (op2, val_hash_addr, tmp1, tmp2)))
            slots.append(("debug", ("compare", val_hash_addr, (round, i, "hash_stage", hi))))

        return slots

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Optimized kernel using VLIW packing and SIMD vectorization.
        """
        # Allocate scratch space for temporaries
        tmp1 = self.alloc_scratch("tmp1")
        tmp2 = self.alloc_scratch("tmp2")
        tmp3 = self.alloc_scratch("tmp3")
        
        # Scratch space addresses for header values
        init_vars = [
            "rounds",
            "n_nodes",
            "batch_size",
            "forest_height",
            "forest_values_p",
            "inp_indices_p",
            "inp_values_p",
        ]
        for v in init_vars:
            self.alloc_scratch(v, 1)
        
        # Load header values
        for i, v in enumerate(init_vars):
            self.add("load", ("const", tmp1, i))
            self.add("load", ("load", self.scratch[v], tmp1))

        # Precompute constants
        zero_const = self.scratch_const(0)
        one_const = self.scratch_const(1)
        two_const = self.scratch_const(2)
        
        # Hash constants
        hash_consts = []
        for op1, val1, op2, op3, val3 in HASH_STAGES:
            hash_consts.append((self.scratch_const(val1), self.scratch_const(val3)))

        self.add("flow", ("pause",))
        
        # Use vectorization when batch_size is divisible by VLEN
        if batch_size % VLEN == 0:
            self._build_vectorized_kernel(rounds, batch_size, hash_consts, zero_const, one_const, two_const)
        else:
            self._build_scalar_kernel_optimized(rounds, batch_size, hash_consts, zero_const, one_const, two_const)
        
        self.instrs.append({"flow": [("pause",)]})
    
    def _build_vectorized_kernel(self, rounds, batch_size, hash_consts, zero_const, one_const, two_const):
        """Optimized vectorized kernel with aggressive pipelining and ILP"""
        body = []
        
        # Allocate vector registers for each batch
        num_batches = batch_size // VLEN
        batch_v_idx = [self.alloc_scratch(f"batch_idx_{i}", VLEN) for i in range(num_batches)]
        batch_v_val = [self.alloc_scratch(f"batch_val_{i}", VLEN) for i in range(num_batches)]
        
        # Working registers - allocate enough for deep pipelining
        PIPELINE_DEPTH = min(8, num_batches)
        v_node_vals = [self.alloc_scratch(f"v_node_val_{i}", VLEN) for i in range(PIPELINE_DEPTH)]
        v_addrs = [self.alloc_scratch(f"v_addr_{i}", VLEN) for i in range(PIPELINE_DEPTH)]
        v_tmp1 = self.alloc_scratch("v_tmp1", VLEN)
        v_tmp2 = self.alloc_scratch("v_tmp2", VLEN)
        
        # Pre-broadcast constants
        v_zero = self.alloc_scratch("v_zero", VLEN)
        v_one = self.alloc_scratch("v_one", VLEN)
        v_two = self.alloc_scratch("v_two", VLEN)
        
        # Pre-broadcast hash constants
        v_hash_consts = []
        for c1, c3 in hash_consts:
            vc1 = self.alloc_scratch(f"v_hash_c1_{c1}", VLEN)
            vc3 = self.alloc_scratch(f"v_hash_c3_{c3}", VLEN)
            body.append(("valu", ("vbroadcast", vc1, c1)))
            body.append(("valu", ("vbroadcast", vc3, c3)))
            v_hash_consts.append((vc1, vc3))
        
        body.append(("valu", ("vbroadcast", v_zero, zero_const)))
        body.append(("valu", ("vbroadcast", v_one, one_const)))
        body.append(("valu", ("vbroadcast", v_two, two_const)))
        
        # Scalar temporaries
        s_base_addr = self.alloc_scratch("s_base_addr")
        s_tmp = self.alloc_scratch("s_tmp")
        
        # Load all data once
        for batch_idx in range(num_batches):
            batch_offset = self.scratch_const(batch_idx * VLEN)
            body.append(("alu", ("+", s_base_addr, self.scratch["inp_indices_p"], batch_offset)))
            body.append(("load", ("vload", batch_v_idx[batch_idx], s_base_addr)))
            body.append(("alu", ("+", s_base_addr, self.scratch["inp_values_p"], batch_offset)))
            body.append(("load", ("vload", batch_v_val[batch_idx], s_base_addr)))
        
        # Main computation with software pipelining
        # Process all rounds with deep pipeline to hide load latency
        for round in range(rounds):
            # Pipeline: Process multiple batches simultaneously
            # Stage 1: Issue address computations for first PIPELINE_DEPTH batches
            for pipe_idx in range(min(PIPELINE_DEPTH, num_batches)):
                v_idx = batch_v_idx[pipe_idx]
                v_addr = v_addrs[pipe_idx]
                # Compute all addresses for this batch vectorially
                for vi in range(VLEN):
                    body.append(("alu", ("+", v_addr + vi, self.scratch["forest_values_p"], v_idx + vi)))
            
            # Stage 2: Issue all loads for first PIPELINE_DEPTH batches
            for pipe_idx in range(min(PIPELINE_DEPTH, num_batches)):
                v_addr = v_addrs[pipe_idx]
                v_node_val = v_node_vals[pipe_idx]
                for vi in range(VLEN):
                    body.append(("load", ("load", v_node_val + vi, v_addr + vi)))
            
            # Stage 3: Process each batch in rolling window
            for batch_idx in range(num_batches):
                pipe_idx = batch_idx % PIPELINE_DEPTH
                v_idx = batch_v_idx[batch_idx]
                v_val = batch_v_val[batch_idx]
                v_node_val = v_node_vals[pipe_idx]
                v_addr = v_addrs[pipe_idx]
                
                # If there's a next batch, prefetch its addresses while we compute
                next_batch = batch_idx + PIPELINE_DEPTH
                if next_batch < num_batches:
                    v_next_idx = batch_v_idx[next_batch]
                    v_next_addr = v_addrs[pipe_idx]
                    # Compute next addresses in parallel with current computation
                    for vi in range(min(2, VLEN)):  # Just start the first 2 to keep pipeline going
                        body.append(("alu", ("+", v_next_addr + vi, self.scratch["forest_values_p"], v_next_idx + vi)))
                
                # XOR
                body.append(("valu", ("^", v_val, v_val, v_node_val)))
                
                # Continue prefetching if needed
                if next_batch < num_batches:
                    for vi in range(2, min(4, VLEN)):
                        body.append(("alu", ("+", v_next_addr + vi, self.scratch["forest_values_p"], v_next_idx + vi)))
                
                # Hash computation - try to use multiply_add where possible
                for hi, (vc1, vc3) in enumerate(v_hash_consts):
                    op1, val1, op2, op3, val3 = HASH_STAGES[hi]
                    body.append(("valu", (op1, v_tmp1, v_val, vc1)))
                    body.append(("valu", (op3, v_tmp2, v_val, vc3)))
                    body.append(("valu", (op2, v_val, v_tmp1, v_tmp2)))
                
                # Finish prefetching addresses
                if next_batch < num_batches:
                    for vi in range(4, VLEN):
                        body.append(("alu", ("+", v_next_addr + vi, self.scratch["forest_values_p"], v_next_idx + vi)))
                    # Issue loads for next batch
                    v_next_node_val = v_node_vals[pipe_idx]
                    for vi in range(VLEN):
                        body.append(("load", ("load", v_next_node_val + vi, v_next_addr + vi)))
                
                # Next index computation
                body.append(("valu", ("&", v_tmp2, v_val, v_one)))
                body.append(("valu", ("==", v_tmp2, v_tmp2, v_zero)))
                body.append(("flow", ("vselect", v_tmp2, v_tmp2, v_one, v_two)))
                body.append(("valu", ("<<", v_idx, v_idx, v_one)))
                body.append(("valu", ("+", v_idx, v_idx, v_tmp2)))
                
                # Wrap indices
                for vi in range(VLEN):
                    body.append(("alu", ("<", v_tmp1 + vi, v_idx + vi, self.scratch["n_nodes"])))
                    body.append(("flow", ("select", v_idx + vi, v_tmp1 + vi, v_idx + vi, zero_const)))
        
        # Store results
        for batch_idx in range(num_batches):
            batch_offset = self.scratch_const(batch_idx * VLEN)
            body.append(("alu", ("+", s_base_addr, self.scratch["inp_indices_p"], batch_offset)))
            body.append(("store", ("vstore", s_base_addr, batch_v_idx[batch_idx])))
            body.append(("alu", ("+", s_base_addr, self.scratch["inp_values_p"], batch_offset)))
            body.append(("store", ("vstore", s_base_addr, batch_v_val[batch_idx])))
        
        body_instrs = self.build(body, vliw=True)
        self.instrs.extend(body_instrs)
    
    def _build_scalar_kernel_optimized(self, rounds, batch_size, hash_consts, zero_const, one_const, two_const):
        """Optimized scalar kernel with VLIW packing and better scheduling"""
        body = []
        
        # Scalar scratch registers
        tmp_idx = self.alloc_scratch("tmp_idx")
        tmp_val = self.alloc_scratch("tmp_val")
        tmp_node_val = self.alloc_scratch("tmp_node_val")
        tmp_addr = self.alloc_scratch("tmp_addr")
        tmp_addr2 = self.alloc_scratch("tmp_addr2")
        tmp1 = self.alloc_scratch("tmp1")
        tmp2 = self.alloc_scratch("tmp2")
        tmp3 = self.alloc_scratch("tmp3")
        
        # Precompute address offsets
        i_offsets = [self.scratch_const(i) for i in range(batch_size)]
        
        for round in range(rounds):
            for i in range(batch_size):
                i_const = i_offsets[i]
                
                # Compute addresses in parallel
                body.append(("alu", ("+", tmp_addr, self.scratch["inp_indices_p"], i_const)))
                body.append(("alu", ("+", tmp_addr2, self.scratch["inp_values_p"], i_const)))
                
                # Load idx and val
                body.append(("load", ("load", tmp_idx, tmp_addr)))
                body.append(("load", ("load", tmp_val, tmp_addr2)))
                
                # Compute node address and load
                body.append(("alu", ("+", tmp_addr, self.scratch["forest_values_p"], tmp_idx)))
                body.append(("load", ("load", tmp_node_val, tmp_addr)))
                
                # XOR
                body.append(("alu", ("^", tmp_val, tmp_val, tmp_node_val)))
                
                # Hash (pack multiple ALU ops together)
                for hi, ((c1, c3), (op1, val1, op2, op3, val3)) in enumerate(zip(hash_consts, HASH_STAGES)):
                    body.append(("alu", (op1, tmp1, tmp_val, c1)))
                    body.append(("alu", (op3, tmp2, tmp_val, c3)))
                    body.append(("alu", (op2, tmp_val, tmp1, tmp2)))
                
                # Compute next index (pack operations)
                body.append(("alu", ("%", tmp1, tmp_val, two_const)))
                body.append(("alu", ("*", tmp_idx, tmp_idx, two_const)))
                body.append(("alu", ("==", tmp1, tmp1, zero_const)))
                
                body.append(("flow", ("select", tmp3, tmp1, one_const, two_const)))
                body.append(("alu", ("+", tmp_idx, tmp_idx, tmp3)))
                
                # Wrap index
                body.append(("alu", ("<", tmp1, tmp_idx, self.scratch["n_nodes"])))
                body.append(("flow", ("select", tmp_idx, tmp1, tmp_idx, zero_const)))
                
                # Compute store addresses
                body.append(("alu", ("+", tmp_addr, self.scratch["inp_indices_p"], i_const)))
                body.append(("alu", ("+", tmp_addr2, self.scratch["inp_values_p"], i_const)))
                
                # Store
                body.append(("store", ("store", tmp_addr, tmp_idx)))
                body.append(("store", ("store", tmp_addr2, tmp_val)))
        
        body_instrs = self.build(body, vliw=True)
        self.instrs.extend(body_instrs)

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
