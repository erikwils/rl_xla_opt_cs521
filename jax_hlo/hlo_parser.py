import argparse
import re
from hlo_representations import HloInstruction, HloComputation, HloModuleIR
from typing import Dict, List, Optional

_HEADER_RE = re.compile(
    r"^HloModule\s+([^\s,]+),\s*"
    r"entry_computation_layout=\{(.+)\}$"
)
_COMP_START_RE = re.compile(r"^\s*(?:ENTRY\s+)?(\w[\w\.]*)\s*\{")
_COMP_END_RE = re.compile(r"^}\s*$")
_INST_RE = re.compile(r'''
    ^\s*
    (?:ROOT\s+)?                    # optional ROOT
    (?P<name>\S+)\s*=\s*            # instruction name
    (?P<shape>                      # the shape, either…
       \([^)]*\)                    #   a parenthesized tuple of shapes, or
       |                            # —or—
       [^\s(]+                      #   any non-space, non-paren token
    )
    \s+
    (?P<opcode>[\w\-]+)             # opcode (letters, digits, underscore, hyphen)
    \(
       (?P<operands>[^)]*)          # comma-separated operand list
    \)
    (?:,\s*(?P<rest>.*))?          # optional “, key=…,” attributes
    \s*$
''', re.VERBOSE)
_OPERAND_PAREN_RE = re.compile(r"\(([^)]*)\)")


def _parse_instruction(line: str) -> HloInstruction:
    m = _INST_RE.match(line)
    if not m:
        raise ValueError(f"Cannot parse instruction: {line}")

    is_root = line.strip().startswith("ROOT")
    name = m.group("name")
    opcode = m.group("opcode")
    shape = (m.group("shape") or "").strip() or None

    rest = m.group("rest")
    operands: List[str] = []
    if rest:
        paren = _OPERAND_PAREN_RE.search(rest)
        if paren:
            operands = [s.strip() for s in paren.group(1).split(",") if s.strip()]

    raw_attrs: Optional[str] = None
    if rest:
        idx = rest.find(")")
        if idx != -1 and idx + 1 < len(rest):
            raw_attrs = rest[idx + 1 :].strip()
        elif idx == -1:
            raw_attrs = rest.strip()

    return HloInstruction(
        name=name,
        opcode=opcode,
        shape=shape,
        operands=operands,
        raw_attrs=raw_attrs,
        is_root=is_root,
    )


def parse_hlo_text(text: str) -> HloModuleIR:
    lines = text.splitlines()
    if not lines:
        raise ValueError("Empty input")

    header_m = _HEADER_RE.match(lines[0])
    if not header_m:
        raise ValueError("First line is not a valid HloModule header")

    module_name = header_m.group(1)
    entry_layout = header_m.group(2)

    comps: Dict[str, HloComputation] = {}
    current: Optional[HloComputation] = None

    for line in lines[1:]:
        start = _COMP_START_RE.match(line)
        if start:
            name = start.group(1)
            current = HloComputation(name=name, is_entry=line.lstrip().startswith("ENTRY"))
            comps[name] = current
            continue

        if _COMP_END_RE.match(line):
            current = None
            continue

        if current is not None and line.strip():
            current.instructions.append(_parse_instruction(line))

    return HloModuleIR(module_name=module_name, entry_layout=entry_layout, computations=comps)

def main(file_path: str):
    # Read the HLO text from the given file
    with open(file_path, 'r') as f:
        hlo_text = f.read()

    hlo_module = parse_hlo_text(hlo_text)
    print(hlo_module)            # e.g. module name, entry computation, etc.

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load and parse an HLO text file into an HLOModuleIR"
    )
    parser.add_argument(
        "file_path",
        help="Path to the .txt file containing HLO IR"
    )
    args = parser.parse_args()
    main(args.file_path)