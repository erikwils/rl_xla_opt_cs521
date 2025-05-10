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
{
"power": 0,
"select-and-scatter": 0,
"pad": 0,
"rsqrt": 0,
"all-gather": 0,
"after-all": 0,
"gather": 0,
"send": 0,
"clamp": 0,
"constant": 2,
"transpose": 0,
"bitcast": 0,
"scatter": 0,
"recv": 0,
"is-finite": 0,
"clz": 0,
"shift-right-arithmetic": 0,
"cos": 0,
"tuple": 0,
"shift-right-logical": 0,
"convolution": 2,
"dynamic-update-slice": 0,
"broadcast": 6,
"while": 0,
"replica-id": 0,
"divide": 0,
"negate": 0,
"tanh": 0,
"all-reduce": 0,
"domain": 0,
"or": 0,
"subtract": 0,
"infeed": 0,
"conditional": 0,
"select": 0,
"slice": 0,
"abs": 0,
"exponential": 0,
"round-nearest-afz": 0,
"atan2": 0,
"add": 2,
"sign": 0,
"reshape": 4,
"multiply": 0,
"convert": 8,
"outfeed": 0,
"xor": 0,
"partition-id": 0,
"maximum": 2,
"copy": 0,
"not": 0,
"call": 2,
"log": 0,
"reduce-window": 0,
"get-tuple-element": 0,
"optimization-barrier": 0,
"popcnt": 0,
"reduce": 0,
"remainder": 0,
"custom-call": 0,
"reduce-scatter": 0,
"sqrt": 0,
"ceil": 0,
"all-to-all": 0,
"tan": 0,
"compare": 0,
"shift-left": 0,
"cbrt": 0,
"floor": 0,
"dot": 0,
"concatenate": 0,
"fusion": 0,
"parameter": 7,
"collective-permute": 0,
"sin": 0,
"bitcast-convert": 0,
"iota": 0,
"dynamic-slice": 0,
"and": 0,
"arithmetic": 12,
"reduction": 0,
"data_move": 21,
"control_flow": 2,
"other": 0,
"total_bytes": 721000,
"input_bytes": 32640,
"output_bytes": 32768,
"elemwise_ratio": 0.34285714285714286,
"mixed_precision": 1
}
```
