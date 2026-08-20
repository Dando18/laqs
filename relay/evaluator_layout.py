"""Layout-descriptor parsing and HIP offset code generation for evaluators."""

from __future__ import annotations

from dataclasses import dataclass

from .gf2 import rank


@dataclass(frozen=True)
class EvaluatorLayout:
    """A canonical word or a full-rank binary inner-layout matrix."""

    word: str
    first_bits: int
    second_bits: int
    a_rows: tuple[int, ...] | None = None

    @property
    def tile_rows(self) -> int:
        return 1 << self.first_bits

    @property
    def tile_columns(self) -> int:
        return 1 << self.second_bits

    @property
    def width(self) -> int:
        return self.first_bits + self.second_bits


CanonicalLayout = EvaluatorLayout


def parse_layout(value: str, option: str) -> EvaluatorLayout:
    """Parse an ``i/j`` word or ``linear:i_bits,j_bits:hex_rows``."""

    normalized = value.lower()
    if not normalized:
        raise ValueError(f"{option} must not be empty")
    if not normalized.startswith("linear:"):
        invalid = sorted(set(normalized) - {"i", "j"})
        if invalid:
            raise ValueError(
                f"{option} must be an i/j word or linear descriptor "
                f"(found {''.join(invalid)!r})"
            )
        first_bits = normalized.count("i")
        second_bits = normalized.count("j")
        if first_bits > 31 or second_bits > 31 or len(normalized) > 62:
            raise ValueError(f"{option} is too wide for the generated index code")
        return EvaluatorLayout(normalized, first_bits, second_bits)

    fields = normalized.split(":")
    if len(fields) != 3:
        raise ValueError(
            f"{option} linear descriptor must be linear:i_bits,j_bits:hex_rows"
        )
    try:
        exponent_fields = fields[1].split(",")
        if len(exponent_fields) != 2:
            raise ValueError
        first_bits, second_bits = (int(field) for field in exponent_fields)
        rows = tuple(int(field, 16) for field in fields[2].split(","))
    except ValueError as error:
        raise ValueError(f"{option} contains an invalid linear descriptor") from error
    width = first_bits + second_bits
    if (
        first_bits < 0
        or second_bits < 0
        or first_bits > 31
        or second_bits > 31
        or width > 62
    ):
        raise ValueError(f"{option} has invalid linear tile exponents")
    mask = (1 << width) - 1
    if len(rows) != width or any(row <= 0 or row & ~mask for row in rows):
        raise ValueError(f"{option} must contain exactly {width} valid matrix rows")
    if rank(rows) != width:
        raise ValueError(f"{option} linear matrix is singular")
    return EvaluatorLayout(normalized, first_bits, second_bits, rows)


canonical_layout = parse_layout


def _mask(bits: int) -> str:
    return "0ull" if bits == 0 else f"0x{(1 << bits) - 1:x}ull"


def _linear_inner_terms(layout: EvaluatorLayout) -> list[str]:
    assert layout.a_rows is not None
    terms: list[str] = []
    physical_bit = 0
    while physical_bit < layout.width:
        row = layout.a_rows[physical_bit]
        source_bit = row.bit_length() - 1
        if row.bit_count() == 1:
            end = physical_bit + 1
            while end < layout.width:
                next_row = layout.a_rows[end]
                if next_row != 1 << (source_bit + end - physical_bit):
                    break
                start_mode = source_bit < layout.first_bits
                next_source = source_bit + end - physical_bit
                if (next_source < layout.first_bits) != start_mode:
                    break
                end += 1
            width = end - physical_bit
            if source_bit < layout.first_bits:
                coordinate = "first"
                shift = source_bit
            else:
                coordinate = "second"
                shift = source_bit - layout.first_bits
            term = (
                f"((static_cast<uint64_t>({coordinate}) >> {shift}) & "
                f"{_mask(width)})"
            )
            if physical_bit:
                term += f" << {physical_bit}"
            terms.append(term)
            physical_bit = end
            continue

        bit_terms = []
        for logical_bit in range(layout.width):
            if not ((row >> logical_bit) & 1):
                continue
            if logical_bit < layout.first_bits:
                bit_terms.append(
                    f"(static_cast<uint64_t>(first) >> {logical_bit})"
                )
            else:
                bit_terms.append(
                    f"(static_cast<uint64_t>(second) >> "
                    f"{logical_bit - layout.first_bits})"
                )
        term = f"(({' ^ '.join(bit_terms)}) & 1ull)"
        if physical_bit:
            term += f" << {physical_bit}"
        terms.append(term)
        physical_bit += 1
    return terms


def layout_function(name: str, layout: EvaluatorLayout) -> str:
    """Emit a host/device offset function for either descriptor form."""

    if layout.a_rows is None:
        used = {"i": 0, "j": 0}
        inner_terms: list[str] = []
        physical_bit = 0
        while physical_bit < len(layout.word):
            mode = layout.word[physical_bit]
            end = physical_bit + 1
            while end < len(layout.word) and layout.word[end] == mode:
                end += 1
            width = end - physical_bit
            logical_bit = used[mode]
            used[mode] += width
            coordinate = "first" if mode == "i" else "second"
            term = (
                f"((static_cast<uint64_t>({coordinate}) >> {logical_bit}) & "
                f"{_mask(width)})"
            )
            if physical_bit:
                term += f" << {physical_bit}"
            inner_terms.append(term)
            physical_bit = end
    else:
        inner_terms = _linear_inner_terms(layout)
    inner = " |\n        ".join(inner_terms) if inner_terms else "0ull"
    description = (
        f"word(low -> high)={layout.word}"
        if layout.a_rows is None
        else f"linear A rows(low -> high)={layout.word}"
    )
    return f"""// {description}, tile={layout.tile_rows}x{layout.tile_columns}
__host__ __device__ static __forceinline__ uint64_t {name}(
    uint32_t first, uint32_t second, uint32_t n) {{
  const uint64_t outer =
      static_cast<uint64_t>(first >> {layout.first_bits}) *
          (n >> {layout.second_bits}) +
      (second >> {layout.second_bits});
  const uint64_t inner =
      {inner};
  return (outer << {layout.width}) | inner;
}}"""
