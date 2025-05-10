import re
import math 
from typing import Dict, List, Optional

_DTYPE_SIZE_MAP: Dict[str, int] = {  # Created map with the help of Gemini
    "pred": 1,
    "s8": 1, "u8": 1,
    "s16": 2, "u16": 2, "bf16": 2, "f16": 2,
    "s32": 4, "u32": 4, "f32": 4,
    "s64": 8, "u64": 8, "f64": 8,
    "c64": 8,    # complex64 (2x f32)
    "c128": 16,  # complex128 (2x f64)
    "token": 0,  # Opaque token, typically no data payload relevant for size
    # Note: "tuple" itself is not a dtype here; tuple shapes are parsed structurally.
    # "opaque" type also results in 0 size, handled in _calculate_shape_size.
}

# Regex to parse an elemental shape string like "f32[1,2,3]{0,1,2}" or "s32[]"
# Captures: 1. dtype (e.g., "f32"), 2. dimensions string (e.g., "1,2,3" or "" for scalar)
_ELEMENTAL_SHAPE_RE = re.compile(r"([a-zA-Z0-9_]+)(?:\[([\d,]*)\])?(?:\{[^}]*?\})?")
_ELEMENTAL_SHAPE_DETAILS_RE = re.compile(r"([a-zA-Z0-9_]+)(?:\[([\d,]*)\])?(?:\{([^}]*)\})?")


def parse_tuple_elements(content: str) -> List[str]:
    """
    Parses the content of a tuple shape string into a list of element shape strings.
    Example: "f32[1,2]{0,1}, (s32[], u8[3])" -> ["f32[1,2]{0,1}", "(s32[], u8[3])"]
    Handles commas within dimensions [...] and layouts {...} correctly.
    """
    elements: List[str] = []
    if not content.strip():  # Handles empty tuple content like in "()"
        return []

    paren_balance = 0
    square_bracket_balance = 0
    curly_bracket_balance = 0
    current_element_start = 0

    for i, char in enumerate(content):
        if char == '(':
            paren_balance += 1
        elif char == ')':
            paren_balance -= 1
            if paren_balance < 0:
                # This indicates a malformed tuple string, e.g., "f32[1])"
                raise ValueError(f"Unbalanced parentheses in tuple content: '{content}' at index {i}")
        elif char == '[':
            # Only track square brackets if we are not inside an already open parenthesis pair
            if paren_balance == 0:
                square_bracket_balance += 1
        elif char == ']':
            if paren_balance == 0:
                square_bracket_balance -= 1
                if square_bracket_balance < 0:
                    raise ValueError(f"Unbalanced square brackets in tuple content: '{content}' at index {i}")
        elif char == '{':
            # Only track curly braces if we are not inside an already open parenthesis pair
            if paren_balance == 0:
                curly_bracket_balance += 1
        elif char == '}':
            if paren_balance == 0:
                curly_bracket_balance -= 1
                if curly_bracket_balance < 0:
                    raise ValueError(f"Unbalanced curly brackets in tuple content: '{content}' at index {i}")
        elif char == ',' and paren_balance == 0 and square_bracket_balance == 0 and curly_bracket_balance == 0:
            # Split only if all relevant brackets are balanced at this top level
            elements.append(content[current_element_start:i].strip())
            current_element_start = i + 1

    # Add the last element
    elements.append(content[current_element_start:].strip())

    # Final sanity checks for the whole content string
    if paren_balance != 0:
        raise ValueError(f"Unbalanced parentheses at the end of tuple content: '{content}'")
    if square_bracket_balance != 0: # Should be 0 if parsing valid tuple content
        raise ValueError(f"Unbalanced square brackets at the end of tuple content: '{content}'")
    if curly_bracket_balance != 0: # Should be 0 if parsing valid tuple content
        raise ValueError(f"Unbalanced curly brackets at the end of tuple content: '{content}'")
        
    return [e for e in elements if e] # Filter out potential empty strings


def calculate_shape_size(shape_str: Optional[str]) -> int:
    """
    Calculates the size in bytes of a tensor given its HLO shape string.
    Handles elemental types (e.g., "f32[1,2,3]") and tuple types (e.g., "(f32[], s32[10])").
    """
    if shape_str is None:
        return 0

    shape_str = shape_str.strip()

    # Handle tuple shapes like "(f32[1,2], s32[])" or "tuple(f32[1,2], s32[])"
    is_tuple_keyword = shape_str.lower().startswith("tuple(")
    if (shape_str.startswith("(") and shape_str.endswith(")")) or \
       (is_tuple_keyword and shape_str.endswith(")")):
        
        content_start = shape_str.find("(") + 1
        content_end = shape_str.rfind(")")
        if content_start > content_end : # Malformed or empty tuple like "()" or "tuple()"
             return 0

        content = shape_str[content_start:content_end].strip()
        element_shape_strs = parse_tuple_elements(content)
        
        total_size = 0
        for elem_str in element_shape_strs:
            total_size += calculate_shape_size(elem_str)  # Recursive call
        return total_size

    # Handle Opaque type (often used for control dependencies, size 0)
    if shape_str.lower() == "opaque[]" or shape_str.lower() == "opaque":
        return 0

    # Handle elemental shapes like "f32[1,2,3]" or "s32[]"
    match = _ELEMENTAL_SHAPE_RE.fullmatch(shape_str)
    if not match:
        raise ValueError(f"Cannot parse elemental shape string: '{shape_str}'")

    dtype = match.group(1)
    dims_str = match.group(2)

    if dtype not in _DTYPE_SIZE_MAP:
        raise ValueError(f"Unknown dtype: '{dtype}' in shape string: '{shape_str}'")

    dtype_size_bytes = _DTYPE_SIZE_MAP[dtype]
    if dtype_size_bytes == 0:  # E.g., "token" type
        return 0

    if dims_str is None or dims_str == "":  # Scalar case (e.g., "f32[]")
        dimensions = []
    else:
        dimensions = [int(d) for d in dims_str.split(",") if d.strip()]

    num_elements = math.prod(dimensions) if dimensions else 1
    return dtype_size_bytes * num_elements


def extract_details_recursive(shape_str: Optional[str], dtypes: List[str], layouts: List[str]) -> None:
    """
    Recursively parses a shape string to extract elemental data types and layout strings.
    Dtypes like 'token' or 'opaque' are ignored for the dtypes list.
    """
    if not shape_str:
        return

    shape_str = shape_str.strip()
    is_tuple_keyword = shape_str.lower().startswith("tuple(")

    # Check for tuple structure
    if (shape_str.startswith("(") and shape_str.endswith(")")) or \
       (is_tuple_keyword and shape_str.endswith(")")):
        
        content_start = shape_str.find("(") + 1
        content_end = shape_str.rfind(")")
        
        if content_start > content_end:  # Malformed or empty tuple like "()" or "tuple()"
            return

        content = shape_str[content_start:content_end].strip()
        if not content: # Handles empty tuple content after stripping
            return
            
        element_shape_strs = parse_tuple_elements(content)
        for elem_str in element_shape_strs:
            extract_details_recursive(elem_str, dtypes, layouts)
    else:
        # Elemental shape
        match = _ELEMENTAL_SHAPE_DETAILS_RE.fullmatch(shape_str)
        if match:
            dtype = match.group(1)
            # Add dtype if it's a data-carrying type
            if dtype and dtype.lower() not in ["token", "opaque"]:
                dtypes.append(dtype)
            
            layout_content = match.group(3)  # Content of layout, e.g., "0,1,2"
            if layout_content is not None:
                layouts.append(layout_content)
        