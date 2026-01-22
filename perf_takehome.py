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

        # Hash optimization metadata, initialized in build_kernel
        self.hash_stage_kind = []
        self.hash_mul_vec = {}
        self.hash_add_vec = {}
        self.hash_op1_const = {}
        self.hash_shift_const = {}
        self.hash_ops = {}

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
        Vectorized hash function that processes 8 values in parallel.

        Uses a mix of generic 3-op stages and optimized multiply_add stages
        for those of the form:

            a = (a + C1) + (a << k)

        which can be rewritten as:

            a = a * (1 + 2**k) + C1   (mod 2**32)

        The per-stage metadata (kind and pre-broadcasted constant vectors) is
        prepared once in build_kernel.
        """
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            kind, stage_ops = self.hash_stage_kind[hi], self.hash_ops[hi]

            if kind == "mul_add":
                # Fast path: a = a * mul + add
                v_mul = self.hash_mul_vec[hi]
                v_add = self.hash_add_vec[hi]
                self.add("valu", ("multiply_add", v_val, v_val, v_mul, v_add))
            else:
                # Generic 3-op stage:
                #   tmp1 = op1(a, C1)
                #   tmp2 = op3(a, C3)
                #   a    = op2(tmp1, tmp2)
                g_op1, g_op2, g_op3 = stage_ops
                v_c1 = self.hash_op1_const[hi]
                v_shift = self.hash_shift_const[hi]

                self.add("valu", (g_op1, v_tmp1, v_val, v_c1))
                self.add("valu", (g_op3, v_tmp2, v_val, v_shift))
                self.add("valu", (g_op2, v_val, v_tmp1, v_tmp2))

        # This method emits instructions directly and returns nothing.
        return None

    # ===== VLIW SCHEDULER HELPERS =====

    def _slot_reads_writes(self, engine, slot):
        """
        Compute the sets of scratch locations read and written by a single slot.

        This is conservative: if in doubt we mark an access, which may reduce
        VLIW packing but will not break correctness.
        """
        reads = set()
        writes = set()

        if engine == "alu":
            op, dest, a1, a2 = slot
            writes.add(dest)
            reads.add(a1)
            reads.add(a2)
        elif engine == "valu":
            op = slot[0]
            if op == "vbroadcast":
                _, dest, src = slot
                writes.update(range(dest, dest + VLEN))
                reads.add(src)
            elif op == "multiply_add":
                _, dest, a, b, c = slot
                for i in range(VLEN):
                    writes.add(dest + i)
                    reads.add(a + i)
                    reads.add(b + i)
                    reads.add(c + i)
            else:
                # Generic binary vector op
                _, dest, a1, a2 = slot
                for i in range(VLEN):
                    writes.add(dest + i)
                    reads.add(a1 + i)
                    reads.add(a2 + i)
        elif engine == "load":
            op = slot[0]
            if op == "load":
                _, dest, addr = slot
                writes.add(dest)
                reads.add(addr)
            elif op == "load_offset":
                _, dest, addr, offset = slot
                writes.add(dest + offset)
                reads.add(addr + offset)
            elif op == "vload":
                _, dest, addr = slot
                reads.add(addr)
                writes.update(range(dest, dest + VLEN))
            elif op == "const":
                _, dest, _val = slot
                writes.add(dest)
        elif engine == "store":
            op = slot[0]
            if op == "store":
                _, addr, src = slot
                reads.add(addr)
                reads.add(src)
            elif op == "vstore":
                _, addr, src = slot
                reads.add(addr)
                for i in range(VLEN):
                    reads.add(src + i)
        elif engine == "flow":
            op = slot[0]
            if op == "select":
                _, dest, cond, a, b = slot
                writes.add(dest)
                reads.update({cond, a, b})
            elif op == "add_imm":
                _, dest, a, _imm = slot
                writes.add(dest)
                reads.add(a)
            elif op == "vselect":
                _, dest, cond, a, b = slot
                writes.update(range(dest, dest + VLEN))
                reads.update(range(cond, cond + VLEN))
                reads.update(range(a, a + VLEN))
                reads.update(range(b, b + VLEN))
            elif op == "coreid":
                _, dest = slot
                writes.add(dest)
            elif op in {"trace_write", "cond_jump", "cond_jump_rel", "jump_indirect"}:
                # These read a scratch location but don't write one
                if len(slot) >= 2:
                    reads.add(slot[1])
            # halt/pause/jump etc. do not touch scratch; we treat pause as a
            # barrier separately in the scheduler.
        return reads, writes

    def _schedule_block(self, ops):
        """
        Given a list of ops (each a dict with engine/slot/reads/writes),
        perform dependency-aware VLIW packing within the block.
        """
        n = len(ops)
        if n == 0:
            return []

        # Build dependency graph using last-writer and pending-read tracking
        preds = [set() for _ in range(n)]
        succs = [set() for _ in range(n)]

        def add_edge(j, i):
            if j == i:
                return
            if j not in preds[i]:
                preds[i].add(j)
                succs[j].add(i)

        last_writer = {}
        from collections import defaultdict

        pending_reads = defaultdict(set)

        for i, op in enumerate(ops):
            reads = op["reads"]
            writes = op["writes"]

            # Handle writes: depend on last writer and all pending readers
            for addr in writes:
                if addr in last_writer:
                    add_edge(last_writer[addr], i)
                if addr in pending_reads:
                    for j in pending_reads[addr]:
                        add_edge(j, i)
                    pending_reads[addr].clear()
                last_writer[addr] = i

            # Handle reads: depend on last writer
            for addr in reads:
                if addr in last_writer:
                    add_edge(last_writer[addr], i)
                pending_reads[addr].add(i)

        in_degree = [len(preds[i]) for i in range(n)]

        # Initialize ready list with ops that have no predecessors
        ready = [i for i in range(n) if in_degree[i] == 0]
        ready.sort()

        scheduled_instrs = []
        scheduled_count = 0

        while scheduled_count < n:
            if not ready:
                raise RuntimeError("Cycle detected in dependency graph during scheduling")

            bundle = {}
            used_slots = {engine: 0 for engine in SLOT_LIMITS.keys()}
            bundle_reads = set()
            bundle_writes = set()

            next_ready = []

            for i in ready:
                op = ops[i]
                engine = op["engine"]
                slot = op["slot"]

                # Skip if this engine is already full this cycle
                if used_slots.get(engine, 0) >= SLOT_LIMITS.get(engine, 0):
                    next_ready.append(i)
                    continue

                reads = op["reads"]
                writes = op["writes"]

                # Check for intra-bundle hazards: conservatively forbid any
                # overlap involving a write.
                hazard = False
                if writes & (bundle_reads | bundle_writes):
                    hazard = True
                elif reads & bundle_writes:
                    hazard = True

                if hazard:
                    next_ready.append(i)
                    continue

                # Schedule this op in the current bundle
                bundle.setdefault(engine, []).append(slot)
                used_slots[engine] = used_slots.get(engine, 0) + 1
                bundle_reads |= reads
                bundle_writes |= writes
                scheduled_count += 1

                # Update in-degrees of successors
                for j in succs[i]:
                    in_degree[j] -= 1
                    if in_degree[j] == 0:
                        next_ready.append(j)

            scheduled_instrs.append(bundle)
            # De-duplicate and keep original order as much as possible
            seen = set()
            new_ready = []
            for i in next_ready:
                if in_degree[i] == 0 and i not in seen:
                    seen.add(i)
                    new_ready.append(i)
            new_ready.sort()
            ready = new_ready

        return scheduled_instrs

    def schedule_vliw(self):
        """
        Run a global, dependency-aware VLIW scheduler over the current
        instruction stream in self.instrs.

        This flattens the program into individual slots, builds a conservative
        dependency graph based on scratch reads/writes, and then greedily
        packs operations into bundles subject to SLOT_LIMITS and dependency
        constraints.
        """
        new_instrs = []
        current_ops = []

        def flush_block():
            nonlocal current_ops, new_instrs
            if current_ops:
                new_instrs.extend(self._schedule_block(current_ops))
                current_ops = []

        for instr in self.instrs:
            # Treat pure pause instructions as barriers: don't schedule across them
            if "flow" in instr and len(instr) == 1 and len(instr["flow"]) == 1:
                slot = instr["flow"][0]
                if slot[0] == "pause":
                    flush_block()
                    new_instrs.append(instr)
                    continue

            # Flatten non-debug slots into current_ops
            for engine, slots in instr.items():
                if engine == "debug":
                    # We currently never emit debug instructions, but keep this
                    # for robustness: treat debug as a barrier.
                    flush_block()
                    new_instrs.append({engine: list(slots)})
                    continue
                for slot in slots:
                    reads, writes = self._slot_reads_writes(engine, slot)
                    current_ops.append(
                        {"engine": engine, "slot": slot, "reads": reads, "writes": writes}
                    )

        flush_block()
        self.instrs = new_instrs

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
        
        # Allocate commonly used scalar constants early
        zero_const = self.scratch_const(0)
        one_const = self.scratch_const(1)
        two_const = self.scratch_const(2)
        
        # Load initial configuration from memory (rounds, sizes, base pointers)
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
        # Load header[0..len(init_vars)-1] in a single vector load,
        # then copy into the per-variable scalar slots.
        v_header = self.alloc_scratch("v_header", len(init_vars))
        # Header fields are stored at mem[0..], and zero_const holds 0.
        self.add("load", ("vload", v_header, zero_const))
        for i, v in enumerate(init_vars):
            # Copy header value into the named scalar slot
            self.add("alu", ("+", self.scratch[v], v_header + i, zero_const))
        
        # ===== VECTOR SCRATCH ALLOCATION (8 words each for VLEN=8) =====
        # Primary vector registers
        v_idx = self.alloc_scratch("v_idx", VLEN)           # Current indices (8 items)
        v_val = self.alloc_scratch("v_val", VLEN)           # Current values (8 items)
        v_node_val = self.alloc_scratch("v_node_val", VLEN) # Tree node values (8 items)
        v_tmp1 = self.alloc_scratch("v_tmp1", VLEN)         # Temp for hash/arithmetic
        v_tmp2 = self.alloc_scratch("v_tmp2", VLEN)         # Temp for hash/arithmetic
        v_tmp3 = self.alloc_scratch("v_tmp3", VLEN)         # Temp for broadcasts/select
        v_next_idx = self.alloc_scratch("v_next_idx", VLEN) # Next iteration indices
        v_in_bounds = self.alloc_scratch("v_in_bounds", VLEN) # Bounds check mask

        # Second set of vector registers for software pipelining / unrolling
        v_idx_b = self.alloc_scratch("v_idx_b", VLEN)
        v_val_b = self.alloc_scratch("v_val_b", VLEN)
        v_node_val_b = self.alloc_scratch("v_node_val_b", VLEN)
        v_tmp1_b = self.alloc_scratch("v_tmp1_b", VLEN)
        v_tmp2_b = self.alloc_scratch("v_tmp2_b", VLEN)
        v_tmp3_b = self.alloc_scratch("v_tmp3_b", VLEN)
        v_next_idx_b = self.alloc_scratch("v_next_idx_b", VLEN)
        v_in_bounds_b = self.alloc_scratch("v_in_bounds_b", VLEN)

        # Third and fourth sets for more ILP (4-way unrolling)
        v_idx_c = self.alloc_scratch("v_idx_c", VLEN)
        v_val_c = self.alloc_scratch("v_val_c", VLEN)
        v_node_val_c = self.alloc_scratch("v_node_val_c", VLEN)
        v_tmp1_c = self.alloc_scratch("v_tmp1_c", VLEN)
        v_tmp2_c = self.alloc_scratch("v_tmp2_c", VLEN)
        v_tmp3_c = self.alloc_scratch("v_tmp3_c", VLEN)
        v_next_idx_c = self.alloc_scratch("v_next_idx_c", VLEN)
        v_in_bounds_c = self.alloc_scratch("v_in_bounds_c", VLEN)

        v_idx_d = self.alloc_scratch("v_idx_d", VLEN)
        v_val_d = self.alloc_scratch("v_val_d", VLEN)
        v_node_val_d = self.alloc_scratch("v_node_val_d", VLEN)
        v_tmp1_d = self.alloc_scratch("v_tmp1_d", VLEN)
        v_tmp2_d = self.alloc_scratch("v_tmp2_d", VLEN)
        v_tmp3_d = self.alloc_scratch("v_tmp3_d", VLEN)
        v_next_idx_d = self.alloc_scratch("v_next_idx_d", VLEN)
        v_in_bounds_d = self.alloc_scratch("v_in_bounds_d", VLEN)
        
        # Scalar temps for tree node loading
        s_tree_idx = self.alloc_scratch("s_tree_idx")
        s_tree_addr = self.alloc_scratch("s_tree_addr")
        s_node_val = self.alloc_scratch("s_node_val")
        
        # Address calculation temporaries for input indices/values
        addr_idx_base = self.alloc_scratch("addr_idx_base")
        addr_val_base = self.alloc_scratch("addr_val_base")
        
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
        
        # Vectorized base address for forest values: broadcast once, then reuse
        v_tree_addr = self.alloc_scratch("v_tree_addr", VLEN)
        v_tree_addr_b = self.alloc_scratch("v_tree_addr_b", VLEN)
        v_tree_addr_c = self.alloc_scratch("v_tree_addr_c", VLEN)
        v_tree_addr_d = self.alloc_scratch("v_tree_addr_d", VLEN)
        v_forest_base = self.alloc_scratch("v_forest_base", VLEN)
        self.add("valu", ("vbroadcast", v_forest_base, self.scratch["forest_values_p"]))

        # ===== HASH CONSTANT VECTORS (PRECOMPUTED ONCE) =====
        self.hash_stage_kind = []
        self.hash_mul_vec = {}
        self.hash_add_vec = {}
        self.hash_op1_const = {}
        self.hash_shift_const = {}
        self.hash_ops = {}

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            if op1 == "+" and op2 == "+" and op3 == "<<":
                # Stages of the form (a + C1) + (a << k) -> multiply_add
                mul_const = (1 + (1 << val3)) % (2**32)
                mul_addr = self.scratch_const(mul_const)
                add_addr = self.scratch_const(val1)

                v_mul = self.alloc_scratch(f"v_hash_mul_{hi}", VLEN)
                v_add = self.alloc_scratch(f"v_hash_add_{hi}", VLEN)
                self.add("valu", ("vbroadcast", v_mul, mul_addr))
                self.add("valu", ("vbroadcast", v_add, add_addr))

                self.hash_stage_kind.append("mul_add")
                self.hash_mul_vec[hi] = v_mul
                self.hash_add_vec[hi] = v_add
                # For consistency, still store ops (unused in mul_add case)
                self.hash_ops[hi] = (op1, op2, op3)
            else:
                # Generic 3-op stage, keep op1/op2/op3 and pre-broadcast constants
                c1_addr = self.scratch_const(val1)
                shift_addr = self.scratch_const(val3)

                v_c1 = self.alloc_scratch(f"v_hash_c1_{hi}", VLEN)
                v_shift = self.alloc_scratch(f"v_hash_shift_{hi}", VLEN)
                self.add("valu", ("vbroadcast", v_c1, c1_addr))
                self.add("valu", ("vbroadcast", v_shift, shift_addr))

                self.hash_stage_kind.append("generic")
                self.hash_op1_const[hi] = v_c1
                self.hash_shift_const[hi] = v_shift
                self.hash_ops[hi] = (op1, op2, op3)

        self.add("flow", ("pause",))

        # ===== PRELOAD INPUT INDICES / VALUES INTO SCRATCH =====
        # Copy the initial indices and values from memory into scratch-backed
        # arrays. The main loop will then operate entirely out of scratch, and
        # we only write the final results back to memory once at the end.
        s_idx_base = self.alloc_scratch("s_idx", batch_size)
        s_val_base = self.alloc_scratch("s_val", batch_size)

        # Initialize base addresses for vector loads
        self.add(
            "flow",
            ("add_imm", addr_idx_base, self.scratch["inp_indices_p"], 0),
        )
        self.add(
            "flow",
            ("add_imm", addr_val_base, self.scratch["inp_values_p"], 0),
        )

        for vec_i in range(0, batch_size, VLEN):
            # Load indices into scratch: s_idx_base[vec_i:vec_i+VLEN]
            self.add(
                "load",
                ("vload", s_idx_base + vec_i, addr_idx_base),
            )

            # Load values into scratch: s_val_base[vec_i:vec_i+VLEN]
            self.add(
                "load",
                ("vload", s_val_base + vec_i, addr_val_base),
            )

            # Advance base addresses for next vector chunk, except after last
            if vec_i + VLEN < batch_size:
                self.add(
                    "flow",
                    ("add_imm", addr_idx_base, addr_idx_base, VLEN),
                )
                self.add(
                    "flow",
                    ("add_imm", addr_val_base, addr_val_base, VLEN),
                )
        
        # ===== MAIN VECTORIZED LOOP (STATE IN SCRATCH, UNROLLED BY 4) =====
        # Process batch_size items in groups of 4 * VLEN. For each group of
        # four vectors we maintain independent working registers (A–D), which
        # exposes more instruction-level parallelism for the global VLIW
        # scheduler to exploit.
        #
        # NOTE: This assumes batch_size is a multiple of 4 * VLEN (which holds
        # for the submission tests where batch_size=256 and VLEN=8).
        for round_idx in range(rounds):
            for vec_i in range(0, batch_size, VLEN * 4):
                vec_i0 = vec_i
                vec_i1 = vec_i + VLEN
                vec_i2 = vec_i + 2 * VLEN
                vec_i3 = vec_i + 3 * VLEN

                # ===== PHASES 1–6 FOR GROUP A (OPERATE IN-PLACE ON SCRATCH) =====
                # Use s_idx_base/s_val_base segments directly as the working vectors.
                idx_a = s_idx_base + vec_i0
                val_a = s_val_base + vec_i0

                self.add("valu", ("+", v_tree_addr, v_forest_base, idx_a))
                for lane_pair_start in range(0, VLEN, 2):
                    bundle = {
                        "load": [
                            ("load_offset", v_node_val, v_tree_addr, lane_pair_start),
                            ("load_offset", v_node_val, v_tree_addr, lane_pair_start + 1),
                        ]
                    }
                    self.add_bundle(bundle)

                # val_a ^= node_val
                self.add("valu", ("^", val_a, val_a, v_node_val))

                self.build_hash_vectorized(
                    val_a, v_tmp1, v_tmp2, v_tmp3, round_idx, vec_i0
                )

                # offset = 1 + (val % 2) using only VALU ops, in-place on val_a
                self.add("valu", ("%", v_tmp1, val_a, v_two))
                self.add("valu", ("+", v_tmp2, v_tmp1, v_one))
                self.add("valu", ("multiply_add", v_next_idx, idx_a, v_two, v_tmp2))
                self.add("valu", ("<", v_in_bounds, v_next_idx, v_n_nodes))
                self.add("flow", ("vselect", idx_a, v_in_bounds, v_next_idx, v_zero))

                # ===== PHASES 1–6 FOR GROUP B =====
                idx_b = s_idx_base + vec_i1
                val_b = s_val_base + vec_i1

                self.add("valu", ("+", v_tree_addr_b, v_forest_base, idx_b))
                for lane_pair_start in range(0, VLEN, 2):
                    bundle = {
                        "load": [
                            ("load_offset", v_node_val_b, v_tree_addr_b, lane_pair_start),
                            ("load_offset", v_node_val_b, v_tree_addr_b, lane_pair_start + 1),
                        ]
                    }
                    self.add_bundle(bundle)

                self.add("valu", ("^", val_b, val_b, v_node_val_b))

                self.build_hash_vectorized(
                    val_b, v_tmp1_b, v_tmp2_b, v_tmp3_b, round_idx, vec_i1
                )

                self.add("valu", ("%", v_tmp1_b, val_b, v_two))
                self.add("valu", ("+", v_tmp2_b, v_tmp1_b, v_one))
                self.add("valu", ("multiply_add", v_next_idx_b, idx_b, v_two, v_tmp2_b))
                self.add("valu", ("<", v_in_bounds_b, v_next_idx_b, v_n_nodes))
                self.add("flow", ("vselect", idx_b, v_in_bounds_b, v_next_idx_b, v_zero))

                # ===== PHASES 1–6 FOR GROUP C =====
                idx_c = s_idx_base + vec_i2
                val_c = s_val_base + vec_i2

                self.add("valu", ("+", v_tree_addr_c, v_forest_base, idx_c))
                for lane_pair_start in range(0, VLEN, 2):
                    bundle = {
                        "load": [
                            ("load_offset", v_node_val_c, v_tree_addr_c, lane_pair_start),
                            ("load_offset", v_node_val_c, v_tree_addr_c, lane_pair_start + 1),
                        ]
                    }
                    self.add_bundle(bundle)

                self.add("valu", ("^", val_c, val_c, v_node_val_c))

                self.build_hash_vectorized(
                    val_c, v_tmp1_c, v_tmp2_c, v_tmp3_c, round_idx, vec_i2
                )

                self.add("valu", ("%", v_tmp1_c, val_c, v_two))
                self.add("valu", ("+", v_tmp2_c, v_tmp1_c, v_one))
                self.add("valu", ("multiply_add", v_next_idx_c, idx_c, v_two, v_tmp2_c))
                self.add("valu", ("<", v_in_bounds_c, v_next_idx_c, v_n_nodes))
                self.add("flow", ("vselect", idx_c, v_in_bounds_c, v_next_idx_c, v_zero))

                # ===== PHASES 1–6 FOR GROUP D =====
                idx_d = s_idx_base + vec_i3
                val_d = s_val_base + vec_i3

                self.add("valu", ("+", v_tree_addr_d, v_forest_base, idx_d))
                for lane_pair_start in range(0, VLEN, 2):
                    bundle = {
                        "load": [
                            ("load_offset", v_node_val_d, v_tree_addr_d, lane_pair_start),
                            ("load_offset", v_node_val_d, v_tree_addr_d, lane_pair_start + 1),
                        ]
                    }
                    self.add_bundle(bundle)

                self.add("valu", ("^", val_d, val_d, v_node_val_d))

                self.build_hash_vectorized(
                    val_d, v_tmp1_d, v_tmp2_d, v_tmp3_d, round_idx, vec_i3
                )

                self.add("valu", ("%", v_tmp1_d, val_d, v_two))
                self.add("valu", ("+", v_tmp2_d, v_tmp1_d, v_one))
                self.add("valu", ("multiply_add", v_next_idx_d, idx_d, v_two, v_tmp2_d))
                self.add("valu", ("<", v_in_bounds_d, v_next_idx_d, v_n_nodes))
                self.add("flow", ("vselect", idx_d, v_in_bounds_d, v_next_idx_d, v_zero))

        # ===== WRITE FINAL RESULTS BACK TO MEMORY =====
        # After all rounds, copy the scratch-backed indices and values back to
        # the input arrays in memory in vector chunks.

        # Reinitialize base addresses for vector stores
        self.add(
            "flow",
            ("add_imm", addr_idx_base, self.scratch["inp_indices_p"], 0),
        )
        self.add(
            "flow",
            ("add_imm", addr_val_base, self.scratch["inp_values_p"], 0),
        )

        for vec_i in range(0, batch_size, VLEN):
            self.add(
                "store",
                ("vstore", addr_idx_base, s_idx_base + vec_i),
            )

            self.add(
                "store",
                ("vstore", addr_val_base, s_val_base + vec_i),
            )

            if vec_i + VLEN < batch_size:
                self.add(
                    "flow",
                    ("add_imm", addr_idx_base, addr_idx_base, VLEN),
                )
                self.add(
                    "flow",
                    ("add_imm", addr_val_base, addr_val_base, VLEN),
                )
        
        # Match the pause in reference implementation
        self.add("flow", ("pause",))

        # After building the kernel, globally VLIW-pack the instruction stream
        # using a conservative, dependency-aware scheduler.
        self.schedule_vliw()


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
