"""CPU-only physical-layout records and conservative realization checks."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Mapping

@dataclass(frozen=True)
class RuntimeLayout:
    name: str
    argument: int
    shape: tuple[int, ...]
    strides: tuple[int, ...]
    envelope_shape: tuple[int, ...]
    rows: tuple[int, ...]

    def pass_argument(self) -> str:
        fields = (
            str(self.argument),
            ",".join(map(str, self.shape)),
            ",".join(map(str, self.strides)),
            ",".join(map(str, self.rows)),
        )
        return "|".join(fields)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "argument": self.argument,
            "shape": list(self.shape),
            "strides": list(self.strides),
            "envelope_shape": list(self.envelope_shape),
            "rows": list(self.rows),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RuntimeLayout":
        return cls(
            str(value["name"]),
            int(value["argument"]),
            tuple(map(int, value["shape"])),
            tuple(map(int, value["strides"])),
            tuple(map(int, value["envelope_shape"])),
            tuple(map(int, value["rows"])),
        )


def preserves_vector_bits(rows, ordinary_rows, bits):
    protected = sum(ordinary_rows[:bits])
    return (tuple(rows[:bits]) == tuple(ordinary_rows[:bits])
            and not any(row & protected for row in rows[bits:]))


def realization_rejections(baseline, selected):
    """Check actual compiler output, without consulting evaluation timings."""
    reasons = []
    for key in ("n_regs", "n_spills", "shared_bytes", "load_instruction_count"):
        if key not in baseline or key not in selected:
            reasons.append(f"missing compiler statistic: {key}")
        elif selected[key] > baseline[key]:
            reasons.append(f"{key} increased: {baseline[key]} -> {selected[key]}")
    if not baseline.get("final_assembly") or not selected.get("final_assembly"):
        reasons.append("final device assembly unavailable")
    if not baseline.get("memory_instruction_widths") or not selected.get("memory_instruction_widths"):
        reasons.append("global memory instructions unavailable")
    if selected.get("execution_encodings") != baseline.get("execution_encodings"):
        reasons.append("execution layout encodings changed")
    if selected.get("final_instruction_count", 0) > 1.10 * baseline.get("final_instruction_count", 0):
        reasons.append("instruction count increased by more than the declared 10% budget")
    if selected.get("matrix_instruction_forms") != baseline.get("matrix_instruction_forms"):
        reasons.append("matrix instruction forms changed")
    if selected.get("memory_instruction_widths") != baseline.get("memory_instruction_widths"):
        reasons.append("lowered memory instruction widths/counts changed")
    if selected.get("matrix_instruction_count") != baseline.get("matrix_instruction_count"):
        reasons.append("matrix instruction count changed")
    return reasons


def compiler_statistics(compiled, directory, label):
    """Save lowered code and compare opcode forms as well as launch resources."""
    from collections import Counter
    import hashlib
    from pathlib import Path
    import re
    import shutil
    import subprocess

    from stage1_common import assembly_opcode_counts, compiled_codegen_statistics

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    record = compiled_codegen_statistics(compiled)
    artifacts = {}
    for name, value in compiled.asm.items():
        if isinstance(value, (str, bytes)):
            path = directory / f"{label}.{name}"
            path.write_bytes(value.encode() if isinstance(value, str) else value)
            artifacts[name] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    assembly = compiled.asm.get("amdgcn")
    if assembly is not None:
        opcodes = assembly_opcode_counts(assembly)
        record["final_assembly"] = "amdgcn"
    else:
        assembly = compiled.asm.get("sass")
        cuobjdump = shutil.which("cuobjdump")
        if assembly is None and cuobjdump and "cubin" in artifacts:
            assembly = subprocess.check_output([cuobjdump, "--dump-sass",
                        artifacts["cubin"]["path"]], text=True)
            path = directory / f"{label}.sass"
            path.write_text(assembly)
            artifacts["sass"] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        if assembly is None:
            record["final_assembly"] = None
            opcodes = Counter()
        else:
            # cuobjdump prefixes instructions with address comments and may
            # append hexadecimal encodings. Keep only the instruction text.
            lines = [re.sub(r"/\*.*?\*/", "", line).strip() for line in assembly.splitlines()]
            opcodes = assembly_opcode_counts("\n".join(lines))
            record["final_assembly"] = "sass"
    memory = {k: v for k, v in opcodes.items() if re.search(
        r"(?:global|flat|buffer)_(?:load|store)|^(?:LDG|STG|LDGSTS|LDSM|TMA)", k)}
    matrix = {k: v for k, v in opcodes.items() if re.search(r"mfma|wmma|mma|MMA|HMMA|IMMA|WGMMA", k)}
    record["memory_instruction_widths"] = dict(sorted(memory.items()))
    record["matrix_instruction_count"] = sum(matrix.values())
    record["matrix_instruction_forms"] = dict(sorted(matrix.items()))
    record["final_instruction_count"] = sum(opcodes.values())
    record["execution_encodings"] = sorted(re.findall(
        r"#(?:triton_gpu|ttg)\.\w+<[^\n]*>", compiled.asm.get("ttgir", "")))
    record["artifacts"] = artifacts
    return record
