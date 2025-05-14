import argparse
import re
import os
import sys

# Add the jax_hlo directory to the path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

from hlo_representations import HloInstruction, HloComputation, HloModuleIR
from typing import Dict, List, Optional

# Regex for parsing HLO generated with help of Gemini
_HEADER_RE = re.compile(
    r"^HloModule\s+([^\s,]+),\s*"
    r"entry_computation_layout=\{(.+)\}$"
)
_COMP_START_RE = re.compile(
    r"^\s*(?:ENTRY\s+)?(?P<comp_name>%?[\w\.-]+)\s*"  # Capture name 
    r"(?:\([^)]*\)\s*->\s*.*?)?"                      # Optional signature like (params) -> return_type
    r"\{"                                             # Opening brace
)
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


def _parse_instruction(line: str) -> HloInstruction:
    m = _INST_RE.match(line)
    if not m:
        raise ValueError(f"Cannot parse instruction: {line}")

    is_root = line.strip().startswith("ROOT")
    name = m.group("name")
    opcode = m.group("opcode")
    shape = (m.group("shape") or "").strip() or None
    operands_str = m.group("operands") # Get the raw operands string
    rest = m.group("rest")

    operands: List[str] = []
    if operands_str:
        # Split operands by comma, then clean each one to get just the name
        raw_operands = operands_str.split(',')
        for op_str in raw_operands:
            op_str = op_str.strip()
            if not op_str:
                continue
            # Assume name is the part before the first colon ':', or the whole string if no colon
            operand_name = op_str.split(':', 1)[0].strip()
            operands.append(operand_name)

    raw_attrs: Optional[str] = None
    if rest:
        paren_match = re.search(r"^\s*\((.*)\)", rest) # Check if 'rest' starts with (...)
        if paren_match:
             attr_start_index = rest.find(')') + 1
             if attr_start_index > 0 and attr_start_index < len(rest) :
                 raw_attrs = rest[attr_start_index:].strip().lstrip(',') # Remove leading comma if any
             else: # No attributes after parenthesis
                 raw_attrs = None
        else:
             raw_attrs = rest.strip().lstrip(',') # Remove leading comma if any

        # Ensure raw_attrs is None if empty string after stripping
        if raw_attrs == "":
            raw_attrs = None

    return HloInstruction(
        name=name,
        opcode=opcode,
        shape=shape,
        operands=operands, # Use the cleaned list of operand names
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

def parse_hlo_from_filepath(file_path: str) -> str:
    # Read the HLO text from the given file
    with open(file_path, 'r') as f:
        hlo_text = f.read()

    return parse_hlo_text(hlo_text)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load and parse an HLO text file into an HLOModuleIR"
    )
    parser.add_argument(
        "file_path",
        help="Path to the .txt file containing HLO IR"
    )
    args = parser.parse_args()
    print(parse_hlo_from_filepath(args.file_path))
