import argparse
import glob
import json
import os
from typing import Dict, Any, List
from hlo_feature_extraction_helpers import calculate_shape_size, extract_details_recursive
from hlo_parser import parse_hlo_from_filepath
from hlo_representations import HloModuleIR, HloComputation

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

MASTER_OP_LIST = sorted(list(ARITHMETIC_OPS | REDUCTION_OPS | DATA_MOVE_OPS | CONTROL_FLOW_OPS))
OPCODE_TO_INDEX = {op: i for i, op in enumerate(MASTER_OP_LIST)}
NUM_OPCODES = len(MASTER_OP_LIST)

ELEMENT_WISE_OPCODES = ARITHMETIC_OPS | {"select"}


def feat_op_counts(ir: HloModuleIR) -> Dict[str, int]:
    """Operation‑level histogram.

    Features Returned
    -----------------
    Dict[str, int]
        Keys are op mnemonics; values are counts. Vector length equal to that of all
        the available op codes.
    """
    op_counts: Dict[str, int] = {key: 0 for key in MASTER_OP_LIST}
    
    # Count occurrences of each opcode
    for instruction in ir.all_instructions():
        if instruction.opcode in op_counts:
            op_counts[instruction.opcode] += 1
            
    return op_counts


def feat_op_categories(ir: HloModuleIR) -> Dict[str, int]:
    """Coarse categories (arithmetic, reduction, data‑move, control, other).
    Other serves as a catch all.

    Returns a dict of 5 fixed features.
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

    Returns 1 feature: total_bytes (int).
    """
    total_bytes = 0
    for instruction in ir.all_instructions():
        total_bytes += calculate_shape_size(instruction.shape)
    return {"total_bytes": total_bytes}


def feat_io_footprint(ir: HloModuleIR) -> Dict[str, int]:
    """Input/Output tensor footprint.

    Returns 2 features:
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

    Returns 1 feature: elemwise_ratio (float 0‑1).
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

    Returns 1 binary feature: mixed_precision (0/1).
    """
    all_dtypes: List[str] = []
    for instruction in ir.all_instructions():
        extract_details_recursive(instruction.shape, all_dtypes, [])
    
    unique_dtypes = set(all_dtypes)
    return {"mixed_precision": 1 if len(unique_dtypes) > 1 else 0}


def feat_graph_structure(ir: HloModuleIR) -> Dict[str, Any]:
    """Extracts graph structure of the entry computation.

    Returns a dict containing:
    * graph_nodes: List[List[float]], features for each node (instruction).
                   Features: [one_hot_opcode, output_size, is_param, is_const, is_root]
    * graph_edge_links: List[List[int]], pairs of [source_node_idx, target_node_idx].
    """
    node_features: List[List[float]] = []
    edge_links: List[List[int]] = []
    default_result = {"graph_nodes": [], "graph_edge_links": []}

    try:
        entry_comp: HloComputation = ir.entry()
    except ValueError:
        return default_result # No entry computation

    instructions = entry_comp.instructions
    if not instructions:
        return default_result # Entry computation is empty

    node_name_to_index: Dict[str, int] = {
        inst.name: i for i, inst in enumerate(instructions)
    }

    # Extract Node Features
    for i, instruction in enumerate(instructions):
        # Use -1 or another indicator if opcode is unknown, though MASTER_OP_LIST aims to be exhaustive
        opcode_idx = float(OPCODE_TO_INDEX.get(instruction.opcode, -1.0))
        output_size = float(calculate_shape_size(instruction.shape))

        is_param = 1.0 if instruction.opcode == "parameter" else 0.0
        is_const = 1.0 if instruction.opcode == "constant" else 0.0
        is_root_flag = 1.0 if instruction.is_root else 0.0

        features = [opcode_idx, output_size, is_param, is_const, is_root_flag]
        node_features.append(features)

    # Extract Edge Links
    for j, instruction in enumerate(instructions): 
        for operand_name in instruction.operands:
            # Check if the operand is an instruction within this entry computation
            if operand_name in node_name_to_index:
                source_index_i = node_name_to_index[operand_name]
                edge_links.append([source_index_i, j])

    return {
        "graph_nodes": node_features,
        "graph_edge_links": edge_links
    }


# Orchestration:
# The main helper that a training script can import.
def extract_all_features(hlo_rep: HloModuleIR) -> Dict[str, Any]:
    """Run all feature extractors and merge into one flat dictionary.

    Parameters
    ----------
    hlo_text : HloModuleIR
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
        feat_graph_structure,
    ]

    all_feats: Dict[str, Any] = {}
    for fn in feature_funcs:
        all_feats.update(fn(hlo_rep))

    return all_feats

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Load and parse HLO file(s) into HloModuleIR and extract features. "
            "If no file_path is provided, processes all .hlo files in ./hlo_data/"
        )
    )
    parser.add_argument(
        "file_path",
        nargs="?",  # Makes the argument optional
        default=None, # Default value if not provided
        help="Path to the .hlo file containing HLO IR. If not provided, scans ./hlo_data/ directory."
    )
    parser.add_argument(
        "--output_jsonl",
        type=str,
        default=None,
        help="Path to the output JSON Lines file. If provided, features for each HLO file will be appended as a new JSON line. Otherwise, features are printed to stdout."
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
        
        files_to_process = glob.glob(os.path.join(hlo_data_dir, "*.hlo"))
        if not files_to_process:
            print(f"No .hlo files found in '{hlo_data_dir}'.")
            exit(0)
        print(f"Found {len(files_to_process)} files in '{hlo_data_dir}'. Processing...")

    output_file_handle = None
    if args.output_jsonl:
        try:
            # Open in append mode so multiple runs can add to the same file if desired
            output_file_handle = open(args.output_jsonl, 'a')
            print(f"Outputting features to JSONL file: {args.output_jsonl}")
        except IOError as e:
            print(f"Error: Could not open output file {args.output_jsonl}: {e}")
            exit(1)

    try:
        for hlo_file_path in files_to_process:
            if not args.output_jsonl: # Print file processing message only if not writing to file or for general verbosity
                 print(f"\n--- Processing file: {hlo_file_path} ---")
            elif files_to_process.index(hlo_file_path) % 10 == 0 and files_to_process.index(hlo_file_path) > 0 : # Progress update for many files
                print(f"Processed {files_to_process.index(hlo_file_path)}/{len(files_to_process)} files...")


            try:
                hlo_module = parse_hlo_from_filepath(hlo_file_path)
                extracted_features = extract_all_features(hlo_module)
                
                # Combine source file information with extracted features
                # record = {"source_file": os.path.basename(hlo_file_path), **extracted_features}
                record = {"source_file": os.path.normpath(hlo_file_path), **extracted_features}

                if output_file_handle:
                    json_line = json.dumps(record)
                    output_file_handle.write(json_line + "\n")
                else:
                    print(json.dumps(record, indent=2))

            except ValueError as ve:
                print(f"Could not parse or process HLO file {hlo_file_path}: {ve}")
            except Exception as e: 
                print(f"An unexpected error occurred while processing {hlo_file_path}: {e}")
    finally:
        if output_file_handle:
            output_file_handle.close()
            print(f"Finished writing features to {args.output_jsonl}")
