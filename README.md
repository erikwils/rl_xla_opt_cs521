# XLA Compiler Optimizations with RL

This repo is the final project of Erik Wilson, Rayaan Farqui, and Ahyush Kaul for CS 521 ML Compilers at UIUC MCS program. Our project is to implement RL for finding optimal XLA compiler pass ordering.

## HLO IR Feature Extraction

See the `/jax_hlo` directory for these files and details. This repository provides Python scripts to parse HLO (Hierarchical Linear Optimizer) Intermediate Representation (IR) text files and extract a variety of numerical and categorical features. The primary output format is JSON Lines (JSONL), where each line in the output file is a JSON object representing the features of a single HLO input file.

### File Structure

- `hlo_feature_extraction.py`: The main script to run for feature extraction.
- `hlo_parser.py`: Contains the HLO text parsing logic.
- `hlo_representations.py`: Defines data classes for HLO Module, Computation, and Instruction.
- `hlo_feature_extraction_helpers.py`: Contains helper functions used during feature calculation (e.g., shape size calculation, tuple parsing).
- `hlo_data/` (convention): A directory where you can place your input `.txt` HLO files. The script can scan this directory.

### Usage

`python hlo_feature_extraction.py [OPTIONS] [FILE_PATH]`

`FILE_PATH` (Positional, Optional):

- Path to a specific .txt file containing HLO IR.
- If omitted, the script will automatically scan and process all .txt files in the ./hlo_data/ directory (this directory must exist if FILE_PATH is not provided).

`--output_jsonl <OUTPUT_FILE_PATH>` (Optional Flag):

- Specifies the path to an output JSON Lines file.
- If provided, features for each processed HLO file will be appended as a new JSON line to this file.
- If omitted, features are printed to the standard output in a prettified JSON format.

### Output

Let's assume this is an example feature vector for now:
```
{"source_file": "hlo_data/conv_relu_hlo.txt", "abs": 0, "add": 2, "after-all": 0, "all-gather": 0, "all-reduce": 0, "all-to-all": 0, "and": 0, "atan2": 0, "bitcast": 0, "bitcast-convert": 0, "broadcast": 6, "call": 2, "cbrt": 0, "ceil": 0, "clamp": 0, "clz": 0, "collective-permute": 0, "compare": 0, "concatenate": 0, "conditional": 0, "constant": 2, "convert": 8, "convolution": 2, "copy": 0, "cos": 0, "custom-call": 0, "divide": 0, "domain": 0, "dot": 0, "dynamic-slice": 0, "dynamic-update-slice": 0, "exponential": 0, "floor": 0, "fusion": 0, "gather": 0, "get-tuple-element": 0, "infeed": 0, "iota": 0, "is-finite": 0, "log": 0, "maximum": 2, "multiply": 0, "negate": 0, "not": 0, "optimization-barrier": 0, "or": 0, "outfeed": 0, "pad": 0, "parameter": 7, "partition-id": 0, "popcnt": 0, "power": 0, "recv": 0, "reduce": 0, "reduce-scatter": 0, "reduce-window": 0, "remainder": 0, "replica-id": 0, "reshape": 4, "round-nearest-afz": 0, "rsqrt": 0, "scatter": 0, "select": 0, "select-and-scatter": 0, "send": 0, "shift-left": 0, "shift-right-arithmetic": 0, "shift-right-logical": 0, "sign": 0, "sin": 0, "slice": 0, "sqrt": 0, "subtract": 0, "tan": 0, "tanh": 0, "transpose": 0, "tuple": 0, "while": 0, "xor": 0, "arithmetic": 12, "reduction": 0, "data_move": 21, "control_flow": 2, "other": 0, "total_bytes": 721000, "input_bytes": 32640, "output_bytes": 32768, "elemwise_ratio": 0.34285714285714286, "mixed_precision": 1, "graph_nodes": [[48.0, 12288.0, 1.0, 0.0, 0.0], [21.0, 6144.0, 0.0, 0.0, 0.0], [48.0, 1728.0, 1.0, 0.0, 0.0], [21.0, 864.0, 0.0, 0.0, 0.0], [22.0, 32768.0, 0.0, 0.0, 0.0], [48.0, 64.0, 1.0, 0.0, 0.0], [21.0, 32.0, 0.0, 0.0, 0.0], [58.0, 32.0, 0.0, 0.0, 0.0], [10.0, 32.0, 0.0, 0.0, 0.0], [58.0, 32.0, 0.0, 0.0, 0.0], [10.0, 32768.0, 0.0, 0.0, 0.0], [1.0, 32768.0, 0.0, 0.0, 0.0], [21.0, 65536.0, 0.0, 0.0, 0.0], [11.0, 65536.0, 0.0, 0.0, 0.0], [21.0, 32768.0, 0.0, 0.0, 0.0], [48.0, 18432.0, 1.0, 0.0, 0.0], [21.0, 9216.0, 0.0, 0.0, 0.0], [22.0, 16384.0, 0.0, 0.0, 0.0], [48.0, 128.0, 1.0, 0.0, 0.0], [21.0, 64.0, 0.0, 0.0, 0.0], [58.0, 64.0, 0.0, 0.0, 0.0], [10.0, 64.0, 0.0, 0.0, 0.0], [58.0, 64.0, 0.0, 0.0, 0.0], [10.0, 16384.0, 0.0, 0.0, 0.0], [1.0, 16384.0, 0.0, 0.0, 0.0], [21.0, 32768.0, 0.0, 0.0, 0.0], [11.0, 32768.0, 0.0, 0.0, 1.0]], "graph_edge_links": [[0, 1], [2, 3], [1, 4], [3, 4], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [4, 11], [10, 11], [11, 12], [12, 13], [13, 14], [15, 16], [14, 17], [16, 17], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23], [17, 24], [23, 24], [24, 25], [25, 26]]}
```
