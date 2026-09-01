from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys
import unittest


TRITON_EXPERIMENTS = Path(__file__).resolve().parents[1] / "triton"
sys.path.insert(0, str(TRITON_EXPERIMENTS))

from stage1_common import (
    assembly_opcode_counts,
    compiled_codegen_statistics,
    is_load_opcode,
    is_store_opcode,
)


class Stage1CodegenTests(unittest.TestCase):
    def test_ptx_global_memory_opcodes_are_classified(self) -> None:
        assembly = """
        .visible .entry kernel() {
            ld.global.b32 %r1, [%rd1];
            @%p1 ld.global.nc.v4.b32 {%r2, %r3, %r4, %r5}, [%rd2];
            st.global.b32 [%rd3], %r1;
            @%p2 st.global.v2.b32 [%rd4], {%r2, %r3};
        }
        """

        opcodes = assembly_opcode_counts(assembly)

        self.assertEqual(opcodes["ld.global.b32"], 1)
        self.assertEqual(opcodes["ld.global.nc.v4.b32"], 1)
        self.assertEqual(opcodes["st.global.b32"], 1)
        self.assertEqual(opcodes["st.global.v2.b32"], 1)
        self.assertTrue(is_load_opcode("ld.global.nc.v4.b32"))
        self.assertTrue(is_store_opcode("st.global.v2.b32"))

    def test_codegen_statistics_count_ptx_loads_and_stores(self) -> None:
        compiled = SimpleNamespace(
            asm={
                "ptx": """
                    ld.global.b32 %r1, [%rd1];
                    ld.global.nc.b32 %r2, [%rd2];
                    st.global.b32 [%rd3], %r1;
                """,
                "cubin": b"binary",
            },
            metadata=SimpleNamespace(shared=0),
            n_regs=8,
            n_spills=0,
            n_max_threads=1024,
        )

        statistics = compiled_codegen_statistics(compiled)

        self.assertEqual(statistics["load_instruction_count"], 2)
        self.assertEqual(statistics["store_instruction_count"], 1)


if __name__ == "__main__":
    unittest.main()
