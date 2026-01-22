import random

from problem import Tree, Input, build_mem_image, reference_kernel2
from perf_takehome import (
    KernelBuilder,
    Machine,
    N_CORES,
)


class WrongFastKernelBuilder(KernelBuilder):
    """
    Experimental \"wrong but fast\" kernel.

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
                self.add("alu", ("+", tmp2, self.scratch["inp_values_p"], self.scratch_const(i)))
                self.add("load", ("load", tmp2, tmp2))
                self.add("alu", ("^", tmp2, tmp2, tmp1))

                # idx = (2*idx + 1) % n_nodes (not the true rule)
                self.add("alu", ("+", tmp1, tmp2, tmp2))  # tmp1 = 2*idx (wrong)
                self.add("alu", ("+", tmp1, tmp1, self.scratch_const(1)))

                # Bounds wrap:
                self.add("alu", ("<", tmp2, tmp1, self.scratch["n_nodes"]))
                # If in bounds, keep tmp1; else 0 (very approximate)
                self.add("flow", ("select", tmp1, tmp2, tmp1, self.scratch_const(0)))

                # Store back idx and val
                self.add("alu", ("+", self.scratch["inp_indices_p"], self.scratch["inp_indices_p"], self.scratch_const(0)))
                self.add("alu", ("+", self.scratch["inp_values_p"], self.scratch["inp_values_p"], self.scratch_const(0)))
                # We skip actual stores here; this kernel is purely for timing.

        self.add("flow", ("pause",))
        self.schedule_vliw()


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


if __name__ == "__main__":
    run_wrong_fast_kernel()