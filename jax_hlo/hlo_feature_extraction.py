import argparse
import glob
import json
import os
from typing import Dict, Any, List, Optional
from collections import Counter
from hlo_feature_extraction_helpers import calculate_shape_size, extract_details_recursive
from hlo_parser import parse_hlo_from_filepath
from hlo_representations import HloModuleIR

# Define opcode categories - Created with help of Gemini 2.5
ARITHMETIC_OPS = {
    "maximum", "add", "subtract", "divide", "multiply", "exponential", "log", "convert", "and", "compare", "abs", 
    "negate", "sign", "ceil", "floor", "round-nearest-afz", "or", "xor", "not", "clz", "popcnt", "shift-left", 
    "shift-right-arithmetic", "shift-right-logical", "remainder", "power", "sqrt", "rsqrt", "cbrt", "is-finite", 
    "atan2", "clamp", "cos", "sin", "tan", "tanh"
}
REDUCTION_OPS = {
    "reduce", "all-reduce", "reduce-window", "select-and-scatter", "all-gather", "reduce-scatter"
}
DATA_MOVE_OPS = {
    "parameter", "constant", "broadcast", "reshape", "convolution", "dot", "transpose", "get-tuple-element", 
    "gather", "select", "scatter", "tuple", "slice", "dynamic-slice", "dynamic-update-slice", "pad", "concatenate", 
    "iota", "bitcast", "bitcast-convert", "copy", "domain", "optimization-barrier"
}
CONTROL_FLOW_OPS = {
    "call", "while", "conditional", "custom-call", "infeed", "outfeed", "after-all", "all-to-all", 
    "collective-permute", "partition-id", "replica-id", "send", "recv", "fusion"
}

MASTER_OP_LIST = [op for op in ARITHMETIC_OPS | REDUCTION_OPS | DATA_MOVE_OPS | CONTROL_FLOW_OPS]
ELEMENT_WISE_OPCODES = ARITHMETIC_OPS | {"select"}


def feat_op_counts(ir: HloModuleIR) -> Dict[str, int]:
    """Operation‑level histogram.

    Features Returned
    -----------------
    Dict[str, int]
        Keys are op mnemonics; values are counts. **Variable‑length** vector.
    """
    op_counts: Dict[str, int] = {key: 0 for key in MASTER_OP_LIST}
    
    # Count occurrences of each opcode
    for instruction in ir.all_instructions():
        if instruction.opcode in op_counts:
            op_counts[instruction.opcode] += 1
            
    return op_counts


def feat_op_categories(ir: HloModuleIR) -> Dict[str, int]:
    """Coarse categories (arithmetic, reduction, data‑move, control).

    Returns a dict of **4 fixed features**.
    """
    categories = {
        "arithmetic": 0,
        "reduction": 0,
        "data_move": 0,
        "control_flow": 0, 
        "other": 0 # For ops not fitting into the above
    }

    for instruction in ir.all_instructions():
        opcode = instruction.opcode
        if opcode in ARITHMETIC_OPS:
            categories["arithmetic"] += 1
        elif opcode in REDUCTION_OPS:
            categories["reduction"] += 1
        elif opcode in DATA_MOVE_OPS:
            categories["data_move"] += 1
        elif opcode in CONTROL_FLOW_OPS:
            categories["control_flow"] += 1
        else:
            categories["other"] += 1 

    return categories


def feat_static_footprint(ir: HloModuleIR) -> Dict[str, int]:
    """Worst‑case static memory consumption.

    Returns **1 feature**: total_bytes (int).
    """
    total_bytes = 0
    for instruction in ir.all_instructions():
        total_bytes += calculate_shape_size(instruction.shape)
    return {"total_bytes": total_bytes}


def feat_io_footprint(ir: HloModuleIR) -> Dict[str, int]:
    """Input/Output tensor footprint.

    Returns **2 features**:
    * input_bytes
    * output_bytes
    """
    input_bytes = 0
    output_bytes = 0

    try:
        entry_comp = ir.entry()
    except ValueError:
        # No entry computation defined in the module
        return {"input_bytes": 0, "output_bytes": 0}

    # Calculate input bytes from parameter instructions
    for instruction in entry_comp.instructions:
        if instruction.opcode == "parameter":
            input_bytes += calculate_shape_size(instruction.shape)

    # Calculate output bytes from the ROOT instruction
    try:
        root_instruction = entry_comp.root()
        output_bytes = calculate_shape_size(root_instruction.shape)
    except ValueError:
        # Entry computation has no ROOT instruction
        output_bytes = 0
        
    return {"input_bytes": input_bytes, "output_bytes": output_bytes}


def feat_parallel_ratios(ir: HloModuleIR) -> Dict[str, float]:
    """Fraction of element‑wise ops.

    Returns **1 feature**: elemwise_ratio (float 0‑1).
    """
    instructions = ir.all_instructions()
    if not instructions:
        return {"elemwise_ratio": 0.0}

    elementwise_op_count = 0
    for instruction in instructions:
        if instruction.opcode in ELEMENT_WISE_OPCODES:
            elementwise_op_count += 1
    
    total_ops = len(instructions)
    ratio = float(elementwise_op_count) / total_ops if total_ops > 0 else 0.0
    return {"elemwise_ratio": ratio}


def feat_mixed_precision_flag(ir: HloModuleIR) -> Dict[str, int]:
    """Indicates multiple dtypes present.

    Returns **1 binary feature**: mixed_precision (0/1).
    """
    all_dtypes: List[str] = []
    for instruction in ir.all_instructions():
        extract_details_recursive(instruction.shape, all_dtypes, [])
    
    unique_dtypes = set(all_dtypes)
    return {"mixed_precision": 1 if len(unique_dtypes) > 1 else 0}


# Orchestration:
# The main helper that a training script can import.
def extract_all_features(hlo_rep: HloModuleIR) -> Dict[str, Any]:
    """Run **all** feature extractors and merge into one flat dictionary.

    Parameters
    ----------
    hlo_text : str
        Raw HLO dump.

    Returns
    -------
    Dict[str, Any]
        Key‑value map suitable for tabular ML or JSON serialization.
    """

    feature_funcs = [
        feat_op_counts,
        feat_op_categories,
        feat_static_footprint,
        feat_io_footprint,
        feat_parallel_ratios,
        feat_mixed_precision_flag,
    ]

    all_feats: Dict[str, Any] = {}
    for fn in feature_funcs:
        all_feats.update(fn(hlo_rep))

    return all_feats

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Load and parse HLO text file(s) into HloModuleIR and extract features. "
            "If no file_path is provided, processes all .txt files in ./hlo_data/"
        )
    )
    parser.add_argument(
        "file_path",
        nargs="?",  # Makes the argument optional
        default=None, # Default value if not provided
        help="Path to the .txt file containing HLO IR. If not provided, scans ./hlo_data/ directory."
    )

    args = parser.parse_args()
    
    files_to_process = []
    if args.file_path:
        if not os.path.isfile(args.file_path):
            print(f"Error: File not found: {args.file_path}")
            exit(1)
        files_to_process.append(args.file_path)
    else:
        hlo_data_dir = "hlo_data"
        if not os.path.isdir(hlo_data_dir):
            print(f"Error: Directory '{hlo_data_dir}' not found. Please create it or provide a specific file path.")
            exit(1)
        
        files_to_process = glob.glob(os.path.join(hlo_data_dir, "*.txt"))
        if not files_to_process:
            print(f"No .txt files found in '{hlo_data_dir}'.")
            exit(0)
        print(f"Found {len(files_to_process)} files in '{hlo_data_dir}'. Processing...")


    for hlo_file_path in files_to_process:
        print(f"\n--- Processing file: {hlo_file_path} ---")
        try:
            hlo_module = parse_hlo_from_filepath(hlo_file_path)
            feats = extract_all_features(hlo_module)
            print(json.dumps(feats, indent=2))
        except ValueError as ve:
            print(f"Could not parse or process HLO file {hlo_file_path}: {ve}")
        except Exception as e: # pragma: no cover
            print(f"An unexpected error occurred while processing {hlo_file_path}: {e}")


