import random

from problem import Tree, Input, build_mem_image, reference_kernel2, HASH_STAGES
from perf_takehome import (
    KernelBuilder,
    Machine,
    N_CORES,
)


class WrongFastKernelBuilder(KernelBuilder):
    """
    Experimental "wrong but fast" kernel.

    This intentionally does NOT implement the full reference semantics:
    - It only runs half the requested number of rounds.
    - It skips the hash and uses a trivial transformation.

    This file is for experimentation only and is NOT used by the tests.
    """

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        # Intentionally use fewer rounds to reduce work
        effective_rounds = rounds // 2

        # Simple scalar walk: idx = 2*idx + 1, val ^= node_val
        # This is NOT equal to reference_kernel2; it is just to see
        # what kind of cycle count is achievable if we cheat.
        tmp1 = self.alloc_scratch("tmp1")
        tmp2 = self.alloc_scratch("tmp2")

        # Load basic configuration
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
        for i, v in enumerate(init_vars):
            self.add("load", ("const", tmp1, i))
            self.add("load", ("load", self.scratch[v], tmp1))

        # For simplicity, operate directly from memory each round, scalar style.
        self.add("flow", ("pause",))

        for r in range(effective_rounds):
            # Loop over all batch elements
            for i in range(batch_size):
                self.add("load", ("const", tmp1, i))
                # idx = mem[inp_indices_p + i]
                self.add("alu", ("+", tmp2, self.scratch["inp_indices_p"], tmp1))
                self.add("load", ("load", tmp2, tmp2))

                # node_val = mem[forest_values_p + idx]
                self.add("alu", ("+", tmp1, self.scratch["forest_values_p"], tmp2))
                self.add("load", ("load", tmp1, tmp1))

                # val ^= node_val
                self.add(
                    "alu",
                    (
                        "+",
                        tmp2,
                        self.scratch["inp_values_p"],
                        self.scratch_const(i),
                    ),
                )
                self.add("load", ("load", tmp2, tmp2))
                self.add("alu", ("^", tmp2, tmp2, tmp1))

                # idx = (2*idx + 1) % n_nodes (not the true rule)
                self.add("alu", ("+", tmp1, tmp2, tmp2))  # tmp1 = 2*idx (wrong)
                self.add("alu", ("+", tmp1, tmp1, self.scratch_const(1)))

                # Bounds wrap:
                self.add("alu", ("<", tmp2, tmp1, self.scratch["n_nodes"]))
                # If in bounds, keep tmp1; else 0 (very approximate)
                self.add(
                    "flow",
                    ("select", tmp1, tmp2, tmp1, self.scratch_const(0)),
                )

                # Store back idx and val
                self.add(
                    "alu",
                    (
                        "+",
                        self.scratch["inp_indices_p"],
                        self.scratch["inp_indices_p"],
                        self.scratch_const(0),
                    ),
                )
                self.add(
                    "alu",
                    (
                        "+",
                        self.scratch["inp_values_p"],
                        self.scratch["inp_values_p"],
                        self.scratch_const(0),
                    ),
                )
                # We skip actual stores here; this kernel is purely for timing.

        self.add("flow", ("pause",))
        self.schedule_vliw()


class NodeCentricKernelBuilder(KernelBuilder):
    """
    Experimental *node-centric* kernel (A1 idea).

    For each round, this kernel iterates over all tree nodes, and for each node
    it scans all lanes and updates only the lanes that *started the round* at
    that node. This is correctness-preserving but extremely expensive; it is
    intended only for small problem sizes to study behavior.
    """

    def build_hash_scalar(self, a_addr: int, tmp1: int, tmp2: int):
        """
        Scalar version of myhash(a), operating entirely in scratch.

        a_addr: scratch addr holding 'a' (updated in place).
        tmp1, tmp2: scratch temporaries.
        """
        for op1, val1, op2, op3, val3 in HASH_STAGES:
            c1 = self.scratch_const(val1)
            c3 = self.scratch_const(val3)
            # tmp1 = op1(a, C1)
            self.add("alu", (op1, tmp1, a_addr, c1))
            # tmp2 = op3(a, C3)
            self.add("alu", (op3, tmp2, a_addr, c3))
            # a    = op2(tmp1, tmp2)
            self.add("alu", (op2, a_addr, tmp1, tmp2))

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        # This experimental kernel is only tractable for small instances.
        assert (
            n_nodes <= 64 and batch_size <= 32
        ), "NodeCentricKernelBuilder is only intended for small sizes"

        # Basic temporaries
        tmp1 = self.alloc_scratch("tmp1")
        tmp2 = self.alloc_scratch("tmp2")

        # Load configuration from memory into scratch
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
        for i, v in enumerate(init_vars):
            self.add("load", ("const", tmp1, i))
            self.add("load", ("load", self.scratch[v], tmp1))

        # Constants
        zero = self.scratch_const(0)
        one = self.scratch_const(1)
        two = self.scratch_const(2)

        # Scratch arrays for indices and values at the start of each round
        s_idx_cur = self.alloc_scratch("s_idx_cur", batch_size)
        s_val_cur = self.alloc_scratch("s_val_cur", batch_size)

        # Scratch arrays for indices and values at the end of each round
        s_idx_next = self.alloc_scratch("s_idx_next", batch_size)
        s_val_next = self.alloc_scratch("s_val_next", batch_size)

        # Preload initial indices and values from memory into s_*_cur
        for i in range(batch_size):
            off = self.scratch_const(i)

            # Load idx[i]
            self.add("alu", ("+", tmp1, self.scratch["inp_indices_p"], off))
            self.add("load", ("load", s_idx_cur + i, tmp1))

            # Load val[i]
            self.add("alu", ("+", tmp1, self.scratch["inp_values_p"], off))
            self.add("load", ("load", s_val_cur + i, tmp1))

        # Hash and index-update temporaries
        a_addr = self.alloc_scratch("hash_a")
        t1 = self.alloc_scratch("hash_t1")
        t2 = self.alloc_scratch("hash_t2")
        parity = self.alloc_scratch("parity")
        offset = self.alloc_scratch("offset")
        next_idx = self.alloc_scratch("next_idx")
        in_bounds = self.alloc_scratch("in_bounds")
        cond = self.alloc_scratch("cond")
        node_val = self.alloc_scratch("node_val")

        # Node-centric processing:
        # for each round:
        #   snapshot s_idx_cur/s_val_cur into s_idx_next/s_val_next
        #   for each node j:
        #       node_val = forest_values[j]
        #       for each lane i:
        #           if idx_cur[i] == j: apply hash + index update, writing to s_*_next
        for h in range(rounds):
            # Start-of-round snapshot: next = cur (will be overwritten selectively)
            for i in range(batch_size):
                self.add("alu", ("+", s_idx_next + i, s_idx_cur + i, zero))
                self.add("alu", ("+", s_val_next + i, s_val_cur + i, zero))

            for node_j in range(n_nodes):
                off_node = self.scratch_const(node_j)

                # node_val = mem[forest_values_p + node_j]
                self.add("alu", ("+", tmp1, self.scratch["forest_values_p"], off_node))
                self.add("load", ("load", node_val, tmp1))

                # Use node_j as the index to compare against for this round snapshot
                node_idx_const = off_node

                for i in range(batch_size):
                    idx_cur_addr = s_idx_cur + i
                    val_cur_addr = s_val_cur + i
                    idx_next_addr = s_idx_next + i
                    val_next_addr = s_val_next + i

                    # cond = (idx_cur[i] == node_j)
                    self.add("alu", ("==", cond, idx_cur_addr, node_idx_const))

                    # a = val_cur[i] ^ node_val
                    self.add("alu", ("^", a_addr, val_cur_addr, node_val))

                    # a = myhash(a)
                    self.build_hash_scalar(a_addr, t1, t2)

                    # Compute next_idx = 2*idx_cur + (1 + (a % 2))
                    self.add("alu", ("%", parity, a_addr, two))
                    self.add("alu", ("+", offset, parity, one))
                    self.add("alu", ("+", next_idx, idx_cur_addr, idx_cur_addr))
                    self.add("alu", ("+", next_idx, next_idx, offset))

                    # Bounds check: if next_idx >= n_nodes -> idx_next = 0
                    self.add(
                        "alu",
                        ("<", in_bounds, next_idx, self.scratch["n_nodes"]),
                    )
                    self.add("flow", ("select", next_idx, in_bounds, next_idx, zero))

                    # Apply updates only when cond==1, writing into *_next
                    self.add(
                        "flow",
                        ("select", idx_next_addr, cond, next_idx, idx_next_addr),
                    )
                    self.add(
                        "flow",
                        ("select", val_next_addr, cond, a_addr, val_next_addr),
                    )

            # End of round: cur = next
            for i in range(batch_size):
                self.add("alu", ("+", s_idx_cur + i, s_idx_next + i, zero))
                self.add("alu", ("+", s_val_cur + i, s_val_next + i, zero))

        # Write final indices and values (s_*_cur) back to memory
        for i in range(batch_size):
            off = self.scratch_const(i)

            # Store idx[i]
            self.add("alu", ("+", tmp1, self.scratch["inp_indices_p"], off))
            self.add("store", ("store", tmp1, s_idx_cur + i))

            # Store val[i]
            self.add("alu", ("+", tmp1, self.scratch["inp_values_p"], off))
            self.add("store", ("store", tmp1, s_val_cur + i))


class PipelinedRoundsKernelBuilder(KernelBuilder):
    """
    Experimental *lane-wise software pipelined* kernel (A2 idea).

    For each round, we treat the per-lane update as a two-stage pipeline:
      - Stage A: load node_val for a lane's current index.
      - Stage B: hash and update using the node_val loaded in the previous
        iteration.

    Concretely, within each round we:
      - Pre-load node_val for lane 0.
      - For i = 1..batch_size-1:
          * Stage A: load node_val for lane i into `node_val_curr`.
          * Stage B: process lane i-1 using `node_val_prev`.
          * node_val_prev = node_val_curr
      - Finally, flush lane batch_size-1 using the last node_val_prev.

    This overlaps "current" loads with "previous" hashes conceptually.
    It is scalar and intended only for small problem sizes.
    """

    def build_hash_scalar(self, a_addr: int, tmp1: int, tmp2: int):
        """
        Scalar myhash(a) implemented with ALU ops over scratch.
        """
        for op1, val1, op2, op3, val3 in HASH_STAGES:
            c1 = self.scratch_const(val1)
            c3 = self.scratch_const(val3)
            self.add("alu", (op1, tmp1, a_addr, c1))
            self.add("alu", (op3, tmp2, a_addr, c3))
            self.add("alu", (op2, a_addr, tmp1, tmp2))

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        # This experimental kernel is only tractable for small instances.
        assert (
            n_nodes <= 64 and batch_size <= 32 and rounds <= 8
        ), "PipelinedRoundsKernelBuilder is only intended for small sizes"

        # Basic temporaries
        tmp1 = self.alloc_scratch("tmp1")
        tmp2 = self.alloc_scratch("tmp2")

        # Load configuration from memory into scratch
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
        for i, v in enumerate(init_vars):
            self.add("load", ("const", tmp1, i))
            self.add("load", ("load", self.scratch[v], tmp1))

        # Constants
        zero = self.scratch_const(0)
        one = self.scratch_const(1)
        two = self.scratch_const(2)

        # State arrays for indices and values
        s_idx_base = self.alloc_scratch("s_idx", batch_size)
        s_val_base = self.alloc_scratch("s_val", batch_size)

        # Preload indices and values from memory into scratch
        for i in range(batch_size):
            off = self.scratch_const(i)

            # idx[i]
            self.add("alu", ("+", tmp1, self.scratch["inp_indices_p"], off))
            self.add("load", ("load", s_idx_base + i, tmp1))

            # val[i]
            self.add("alu", ("+", tmp1, self.scratch["inp_values_p"], off))
            self.add("load", ("load", s_val_base + i, tmp1))

        # Hash and update temporaries
        a_addr = self.alloc_scratch("hash_a")
        t1 = self.alloc_scratch("hash_t1")
        t2 = self.alloc_scratch("hash_t2")
        parity = self.alloc_scratch("parity")
        offset = self.alloc_scratch("offset")
        next_idx = self.alloc_scratch("next_idx")
        in_bounds = self.alloc_scratch("in_bounds")
        node_val_prev = self.alloc_scratch("node_val_prev")
        node_val_curr = self.alloc_scratch("node_val_curr")

        # Helper to emit "stage B" for a single lane j using node_val_prev
        def emit_stage_b(j: int):
            idx_addr = s_idx_base + j
            val_addr = s_val_base + j

            # a = val[j] ^ node_val_prev
            self.add("alu", ("^", a_addr, val_addr, node_val_prev))

            # a = myhash(a)
            self.build_hash_scalar(a_addr, t1, t2)

            # offset = 1 + (a % 2)
            self.add("alu", ("%", parity, a_addr, two))
            self.add("alu", ("+", offset, parity, one))

            # next_idx = 2*idx + offset
            self.add("alu", ("+", next_idx, idx_addr, idx_addr))
            self.add("alu", ("+", next_idx, next_idx, offset))

            # bounds wrap
            self.add(
                "alu",
                ("<", in_bounds, next_idx, self.scratch["n_nodes"]),
            )
            self.add("flow", ("select", next_idx, in_bounds, next_idx, zero))

            # idx[j] = next_idx
            self.add("flow", ("select", idx_addr, one, next_idx, zero))

            # val[j] = a
            self.add("alu", ("+", val_addr, a_addr, zero))

        # Main rounds, each with lane-wise software pipelining
        for h in range(rounds):
            if batch_size == 0:
                continue

            # Warmup: load node_val_prev for lane 0
            # addr = forest_values_p + idx[0]
            idx0_addr = s_idx_base + 0
            self.add("alu", ("+", tmp1, self.scratch["forest_values_p"], idx0_addr))
            self.add("load", ("load", node_val_prev, tmp1))

            # Pipeline over lanes 1..batch_size-1
            for i in range(1, batch_size):
                # Stage A: load node_val_curr for lane i
                idx_i_addr = s_idx_base + i
                self.add(
                    "alu",
                    ("+", tmp1, self.scratch["forest_values_p"], idx_i_addr),
                )
                self.add("load", ("load", node_val_curr, tmp1))

                # Stage B: process lane i-1 using node_val_prev
                emit_stage_b(i - 1)

                # Advance pipeline: node_val_prev = node_val_curr
                self.add("alu", ("+", node_val_prev, node_val_curr, zero))

            # Flush last lane (batch_size-1) using final node_val_prev
            emit_stage_b(batch_size - 1)

        # Write final indices and values back to memory
        for i in range(batch_size):
            off = self.scratch_const(i)

            # idx[i]
            self.add("alu", ("+", tmp1, self.scratch["inp_indices_p"], off))
            self.add("store", ("store", tmp1, s_idx_base + i))

            # val[i]
            self.add("alu", ("+", tmp1, self.scratch["inp_values_p"], off))
            self.add("store", ("store", tmp1, s_val_base + i))


class UniformLoadsKernelBuilder(KernelBuilder):
    """
    Experimental vector kernel that *conditionally* shares node loads
    across 8 lanes when all indices in the vector are equal.

    For each 8-lane chunk in a round:
      - Compute all_equal = 1 if idx[0] == ... == idx[7], else 0.
      - If all_equal:
           * Load node_val once: node_val = forest_values[idx[0]]
           * vbroadcast node_val across a vector register
        Else:
           * Load node_val[lane] individually for each lane.

    Then perform the usual vectorized hash and index update.

    This kernel:
      - Uses the same hash as the main kernel (via build_hash_vectorized).
      - Keeps all state in scratch (s_idx/s_val) and writes back once.
      - Does NOT call schedule_vliw, so control flow (jumps) is preserved.
    """

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        # For simplicity, require batch_size multiple of VLEN (8)
        assert batch_size % 8 == 0, "UniformLoadsKernelBuilder requires batch_size % 8 == 0"

        VLEN = 8

        # --- Scalar temporaries and configuration ---
        tmp1 = self.alloc_scratch("tmp1")
        tmp2 = self.alloc_scratch("tmp2")

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
        for i, v in enumerate(init_vars):
            self.add("load", ("const", tmp1, i))
            self.add("load", ("load", self.scratch[v], tmp1))

        # Debug: record n_nodes and rounds into the core's trace buffer
        self.add("flow", ("trace_write", self.scratch["n_nodes"]))
        self.add("flow", ("trace_write", self.scratch["rounds"]))

        zero_const = self.scratch_const(0)
        one_const = self.scratch_const(1)
        two_const = self.scratch_const(2)

        # --- Vector scratch allocation ---
        v_zero = self.alloc_scratch("v_zero", VLEN)
        v_one = self.alloc_scratch("v_one", VLEN)
        v_two = self.alloc_scratch("v_two", VLEN)
        self.add("valu", ("vbroadcast", v_zero, zero_const))
        self.add("valu", ("vbroadcast", v_one, one_const))
        self.add("valu", ("vbroadcast", v_two, two_const))

        v_n_nodes = self.alloc_scratch("v_n_nodes", VLEN)
        self.add("valu", ("vbroadcast", v_n_nodes, self.scratch["n_nodes"]))

        v_forest_base = self.alloc_scratch("v_forest_base", VLEN)
        self.add("valu", ("vbroadcast", v_forest_base, self.scratch["forest_values_p"]))

        v_node_val = self.alloc_scratch("v_node_val", VLEN)
        v_tmp1 = self.alloc_scratch("v_tmp1", VLEN)
        v_tmp2 = self.alloc_scratch("v_tmp2", VLEN)
        v_next_idx = self.alloc_scratch("v_next_idx", VLEN)
        v_in_bounds = self.alloc_scratch("v_in_bounds", VLEN)

        # Scalar temporaries for uniform detection and address computation
        all_equal = self.alloc_scratch("all_equal")
        eq_tmp = self.alloc_scratch("eq_tmp")
        s_node_val_scalar = self.alloc_scratch("s_node_val_scalar")
        s_tree_addr = self.alloc_scratch("s_tree_addr")

        # --- Hash metadata (reuse main-kernel vectorized hash) ---
        self.hash_stage_kind = []
        self.hash_mul_vec = {}
        self.hash_add_vec = {}
        self.hash_op1_const = {}
        self.hash_shift_const = {}
        self.hash_ops = {}

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            if op1 == "+" and op2 == "+" and op3 == "<<":
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
                self.hash_ops[hi] = (op1, op2, op3)
            else:
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

        # Align with reference harness pause
        self.add("flow", ("pause",))

        # --- Preload indices and values into scratch arrays ---
        s_idx_base = self.alloc_scratch("s_idx", batch_size)
        s_val_base = self.alloc_scratch("s_val", batch_size)

        for i in range(batch_size):
            off = self.scratch_const(i)

            # idx[i]
            self.add("alu", ("+", tmp1, self.scratch["inp_indices_p"], off))
            self.add("load", ("load", s_idx_base + i, tmp1))

            # val[i]
            self.add("alu", ("+", tmp1, self.scratch["inp_values_p"], off))
            self.add("load", ("load", s_val_base + i, tmp1))

        # --- Helper to emit loads that fill v_node_val ---
        def emit_uniform_or_fallback_block(idx_vec_base: int):
            """
            Simpler version for now: always perform per-lane loads

                v_node_val[lane] = forest_values[idx[lane]]

            This ignores the all_equal flag and does not attempt to share
            loads, but keeps the structure similar so we can validate the
            rest of the vector logic.
            """
            for lane in range(VLEN):
                idx_addr = idx_vec_base + lane
                # tmp1 = forest_values_p + idx[lane]
                self.add("alu", ("+", tmp1, self.scratch["forest_values_p"], idx_addr))
                # v_node_val[lane] = mem[tmp1]
                self.add("load", ("load", v_node_val + lane, tmp1))

        # --- Main rounds: vectorized with conditional load sharing ---
        for h in range(rounds):
            for base in range(0, batch_size, VLEN):
                idx_vec = s_idx_base + base
                val_vec = s_val_base + base

                # Compute all_equal flag for this 8-lane chunk
                # all_equal = 1
                self.add("alu", ("+", all_equal, one_const, zero_const))
                # Compare idx[0] to idx[1..7]
                for lane in range(1, VLEN):
                    self.add("alu", ("==", eq_tmp, idx_vec, idx_vec + lane))
                    self.add("alu", ("&", all_equal, all_equal, eq_tmp))

                # Conditionally fill v_node_val either via shared or per-lane loads
                emit_uniform_or_fallback_block(idx_vec)

                # val ^= node_val (vector)
                self.add("valu", ("^", val_vec, val_vec, v_node_val))

                # Hash vectorized
                self.build_hash_vectorized(
                    val_vec, v_tmp1, v_tmp2, v_zero, h, base
                )

                # offset = 1 + (val % 2)
                self.add("valu", ("%", v_tmp1, val_vec, v_two))
                self.add("valu", ("+", v_tmp2, v_tmp1, v_one))

                # next_idx = 2*idx + offset via multiply_add: 2*idx + offset
                self.add("valu", ("multiply_add", v_next_idx, idx_vec, v_two, v_tmp2))

                # bounds check: next_idx < n_nodes
                self.add("valu", ("<", v_in_bounds, v_next_idx, v_n_nodes))

                # Debug: record lane-0 next_idx, n_nodes, in_bounds, and idx before update
                self.add("flow", ("trace_write", v_next_idx))
                self.add("flow", ("trace_write", v_n_nodes))
                self.add("flow", ("trace_write", v_in_bounds))
                self.add("flow", ("trace_write", idx_vec))

                # idx = next_idx if in bounds else 0
                self.add("flow", ("vselect", idx_vec, v_in_bounds, v_next_idx, v_zero))

        # --- Write final indices and values back to memory ---
        for base in range(0, batch_size, VLEN):
            off_vec = self.scratch_const(base)

            # idx chunk
            self.add("alu", ("+", tmp1, self.scratch["inp_indices_p"], off_vec))
            self.add("store", ("vstore", tmp1, s_idx_base + base))

            # val chunk
            self.add("alu", ("+", tmp1, self.scratch["inp_values_p"], off_vec))
            self.add("store", ("vstore", tmp1, s_val_base + base))

        # Final pause to match reference harness
        self.add("flow", ("pause",))


def run_wrong_fast_kernel():
    random.seed(123)
    forest_height, rounds, batch_size = 10, 16, 256
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = WrongFastKernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)

    machine = Machine(
        mem,
        kb.instrs,
        kb.debug_info(),
        n_cores=N_CORES,
        trace=False,
        value_trace={},
    )
    machine.run()
    print("Wrong-fast kernel cycles:", machine.cycle)

    # Compare against reference to show it is indeed wrong
    mem_ref = build_mem_image(forest, inp)
    value_trace = {}
    for _ in reference_kernel2(mem_ref, value_trace):
        pass
    match = mem == mem_ref
    print("Wrong-fast kernel matches reference:", match)


def run_node_centric_kernel_small():
    """
    Run the experimental node-centric kernel on a small instance to
    measure its behavior and verify correctness.
    """
    random.seed(123)
    forest_height, rounds, batch_size = 4, 4, 16
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = NodeCentricKernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)

    machine = Machine(
        mem,
        kb.instrs,
        kb.debug_info(),
        n_cores=N_CORES,
        trace=False,
        value_trace={},
    )
    machine.run()
    print(
        "Node-centric kernel cycles (height=4, rounds=4, batch=16):",
        machine.cycle,
    )

    # Compare against reference to confirm correctness
    mem_ref = build_mem_image(forest, inp)
    value_trace = {}
    for _ in reference_kernel2(mem_ref, value_trace):
        pass

    batch_size_ref = mem_ref[2]
    inp_indices_p = mem_ref[5]
    inp_values_p = mem_ref[6]

    idx_machine = machine.mem[inp_indices_p : inp_indices_p + batch_size_ref]
    val_machine = machine.mem[inp_values_p : inp_values_p + batch_size_ref]
    idx_ref = mem_ref[inp_indices_p : inp_indices_p + batch_size_ref]
    val_ref = mem_ref[inp_values_p : inp_values_p + batch_size_ref]

    match = (idx_machine == idx_ref) and (val_machine == val_ref)
    print("Node-centric kernel matches reference:", match)
    if not match:
        print("Indices (machine):", idx_machine)
        print("Indices (ref):    ", idx_ref)
        print("Values (machine): ", val_machine)
        print("Values (ref):     ", val_ref)


def run_pipelined_kernel_small():
    """
    Run the PipelinedRoundsKernelBuilder on a small instance and compare
    against the reference implementation.
    """
    random.seed(123)
    forest_height, rounds, batch_size = 4, 4, 16
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = PipelinedRoundsKernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)

    machine = Machine(
        mem,
        kb.instrs,
        kb.debug_info(),
        n_cores=N_CORES,
        trace=False,
        value_trace={},
    )
    machine.run()
    print("PipelinedRounds kernel cycles:", machine.cycle)

    # Reference
    mem_ref = build_mem_image(forest, inp)
    value_trace = {}
    for _ in reference_kernel2(mem_ref, value_trace):
        pass

    assert mem == mem_ref, "PipelinedRounds kernel does not match reference"
    print("PipelinedRounds kernel matches reference on small test.")


def run_uniform_kernel_small():
    """
    Run the UniformLoadsKernelBuilder on a small instance and compare
    against the reference implementation. Print differences if any.
    """
    random.seed(456)
    forest_height, rounds, batch_size = 4, 4, 16
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = UniformLoadsKernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)

    machine = Machine(
        mem,
        kb.instrs,
        kb.debug_info(),
        n_cores=N_CORES,
        trace=False,
        value_trace={},
    )
    machine.run()
    print("UniformLoads kernel cycles (small):", machine.cycle)
    tb = machine.cores[0].trace_buf
    print("UniformLoads trace buffer (first 32):", tb[:32])

    # Reference
    mem_ref = build_mem_image(forest, inp)
    value_trace = {}
    for _ in reference_kernel2(mem_ref, value_trace):
        pass

    batch_size_ref = mem_ref[2]
    inp_indices_p = mem_ref[5]
    inp_values_p = mem_ref[6]

    idx_machine = machine.mem[inp_indices_p : inp_indices_p + batch_size_ref]
    val_machine = machine.mem[inp_values_p : inp_values_p + batch_size_ref]
    idx_ref = mem_ref[inp_indices_p : inp_indices_p + batch_size_ref]
    val_ref = mem_ref[inp_values_p : inp_values_p + batch_size_ref]

    match = (idx_machine == idx_ref) and (val_machine == val_ref)
    print("UniformLoads kernel matches reference on small test:", match)
    if not match:
        print("Indices (machine):", idx_machine)
        print("Indices (ref):    ", idx_ref)
        print("Values (machine): ", val_machine)
        print("Values (ref):     ", val_ref)


def run_uniform_kernel_main_config():
    """
    Run the UniformLoadsKernelBuilder on the main config
    (forest_height=10, rounds=16, batch_size=256) and compare against
    the reference implementation. This is a 'real test' in the sense of
    using the full-sized problem, but does not integrate with the
    submission_tests harness.
    """
    random.seed(789)
    forest_height, rounds, batch_size = 10, 16, 256
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = UniformLoadsKernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)

    machine = Machine(
        mem,
        kb.instrs,
        kb.debug_info(),
        n_cores=N_CORES,
        trace=False,
        value_trace={},
    )
    machine.run()
    print("UniformLoads kernel cycles (main config):", machine.cycle)

    # Reference
    mem_ref = build_mem_image(forest, inp)
    value_trace = {}
    for _ in reference_kernel2(mem_ref, value_trace):
        pass

    assert mem == mem_ref, "UniformLoads kernel does not match reference on main config"
    print("UniformLoads kernel matches reference on main config.")


if __name__ == "__main__":
    run_wrong_fast_kernel()
    run_node_centric_kernel_small()
    # PipelinedRoundsKernelBuilder is experimental and currently not exact;
    # leave it out of the default run sequence.
    # run_pipelined_kernel_small()
    run_uniform_kernel_small()
    run_uniform_kernel_main_config()