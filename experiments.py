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

    rounds_ref = mem_ref[0]
    n_nodes_ref = mem_ref[1]
    batch_size_ref = mem_ref[2]
    forest_values_p = mem_ref[4]
    inp_indices_p = mem_ref[5]
    inp_values_p = mem_ref[6]

    idx_machine = mem[inp_indices_p : inp_indices_p + batch_size_ref]
    val_machine = mem[inp_values_p : inp_values_p + batch_size_ref]
    idx_ref = mem_ref[inp_indices_p : inp_indices_p + batch_size_ref]
    val_ref = mem_ref[inp_values_p : inp_values_p + batch_size_ref]

    match = (idx_machine == idx_ref) and (val_machine == val_ref)
    print("Node-centric kernel matches reference:", match)
    if not match:
        print("Indices (machine):", idx_machine)
        print("Indices (ref):    ", idx_ref)
        print("Values (machine): ", val_machine)
        print("Values (ref):     ", val_ref)


if __name__ == "__main__":
    run_wrong_fast_kernel()
    run_node_centric_kernel_small()