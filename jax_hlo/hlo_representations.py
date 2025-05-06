import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

@dataclass
class HloInstruction:
    name: str
    opcode: str
    shape: Optional[str]
    operands: List[str] = field(default_factory=list)
    raw_attrs: Optional[str] = None
    is_root: bool = False

    def __str__(self) -> str:  # pragma: no cover
        flag = "ROOT " if self.is_root else ""
        return f"{flag}{self.name} ({self.opcode}) | shape={self.shape} | ops={self.operands} | attrs={self.raw_attrs}"


@dataclass
class HloComputation:
    name: str
    instructions: List[HloInstruction] = field(default_factory=list)
    is_entry: bool = False

    _index: Dict[str, HloInstruction] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        for inst in self.instructions:
            self._index[inst.name] = inst

    def instruction(self, name: str) -> HloInstruction:
        return self._index[name]

    def root(self) -> HloInstruction:
        for inst in self.instructions:
            if inst.is_root:
                return inst
        raise ValueError(f"Computation {self.name} has no ROOT")

    def __str__(self) -> str:  # pragma: no cover
        tag = " (ENTRY)" if self.is_entry else ""
        body = "\n".join(f"  {i}" for i in self.instructions)
        return f"Computation {self.name}{tag}:\n{body}"


@dataclass
class HloModuleIR:
    module_name: str
    entry_layout: Optional[str]
    computations: Dict[str, HloComputation]

    # convenience ---------------------------------------------------------
    def entry(self) -> HloComputation:
        for c in self.computations.values():
            if c.is_entry:
                return c
        raise ValueError("Module missing ENTRY computation")

    def all_instructions(self) -> List[HloInstruction]:
        return [i for c in self.computations.values() for i in c.instructions]

    def __str__(self) -> str:  # pragma: no cover
        comps = "\n\n".join(str(c) for c in self.computations.values())
        return f"HloModule {self.module_name}\nentry_layout={{ {self.entry_layout} }}\n\n{comps}"