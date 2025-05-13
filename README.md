# XLA Compiler Optimizations with RL

This repo is the final project of Erik Wilson, Rayaan Faruqi, and Ahyush Kaul for CS 521 ML Compilers at UIUC MCS program. Our project is to implement RL for finding optimal XLA compiler pass ordering.

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

## XLA Compiler Setup

This section provides instructions for building and using the XLA compiler with this project. These steps are crucial for running the reinforcement learning environment that optimizes compiler pass ordering.

### Prerequisites

- Python 3.8+ (with venv or conda)
- Bazel 6.x (we use bazelisk to manage versions)
- Git
- A C++ compiler (Clang recommended for macOS)
- At least 16GB of RAM and 20GB of disk space for the build

### Building XLA from Source

1. Clone the XLA repository next to this repository:
   ```bash
   cd ..
   git clone https://github.com/openxla/xla.git
   cd xla
   ```

2. Configure the build for macOS:
   ```bash
   # Create a .bazelrc file with macOS configuration
   cat > .bazelrc << EOF
   build --apple_platform_type=macos
   build --apple_crosstool_top=@local_config_apple_cc//:toolchain
   build --crosstool_top=@local_config_apple_cc//:toolchain
   build --host_crosstool_top=@local_config_apple_cc//:toolchain
   build --copt=-DGRPC_BAZEL_BUILD
   build --features=archive_param_file
   build --copt=-w
   build --define=PREFIX=/usr
   build --define=LIBDIR=\$(PREFIX)/lib
   build --define=INCLUDEDIR=\$(PREFIX)/include
   build --define=PROTOBUF_INCLUDE_PATH=\$(PREFIX)/include
   build --cxxopt=-std=c++17
   build --host_cxxopt=-std=c++17
   build --macos_minimum_os=12.3
   build --macos_sdk_version=12.3
   build --cpu=darwin_x86_64
   build --host_cpu=darwin_x86_64
   build --noenable_bzlmod
   EOF
   ```

3. Build the hlo-opt tool:
   ```bash
   # Build the lightweight version for hardware-independent passes
   bazelisk build -c opt //xla/hlo/tools:hlo-opt
   
   # Or build the full version with platform-specific passes (CPU, GPU, etc.)
   bazelisk build -c opt --config=[CPU|GPU] //xla/tools:hlo-opt
   ```

4. Verify the built tool:
   ```bash
   # List all available passes
   ./bazel-bin/xla/hlo/tools/hlo-opt --list-passes
   ```

### Integrating XLA with Reinforcement Learning

The `xla_interface.py` script in the reinforcement learning directory provides an interface between the XLA compiler and our RL environment. It handles:

1. Running optimization passes on HLO graphs
2. Extracting features from optimized HLO
3. Computing metrics to evaluate optimization quality

To use the RL environment with XLA:

1. Make sure your environment has access to the XLA build directory
2. Use the `xla_opt_env.py` environment with the `--xla_dir` parameter pointing to your XLA build
3. Passes are applied using the names as defined in the XLA compiler

Example:
```bash
# Train the RL agent with XLA passes
python reinforcement_learning/train_agent.py --xla_dir=../xla
```

### Using the hlo-opt Tool

The hlo-opt tool allows you to:

1. Apply individual passes to HLO files:
   ```bash
   ./bazel-bin/xla/hlo/tools/hlo-opt --passes=algebraic_simplifier input.hlo -o output.hlo
   ```

2. Apply multiple passes in sequence:
   ```bash
   ./bazel-bin/xla/hlo/tools/hlo-opt --passes=hlo_dce,algebraic_simplifier,cse input.hlo -o output.hlo
   ```

3. Convert between HLO formats:
   ```bash
   # Convert HLO text to proto
   ./bazel-bin/xla/hlo/tools/hlo-opt --emit-proto input.hlo -o output.pb
   
   # Convert proto to text
   ./bazel-bin/xla/hlo/tools/hlo-opt input.pbtxt -o output.hlo
   ```

### Available Optimization Passes

Below is a subset of key optimization passes available through hlo-opt:

1. **Core Simplifiers**:
   - `algebraic_simplifier` - Performs algebraic simplifications
   - `hlo_dce` - Dead code elimination
   - `cse` - Common subexpression elimination
   - `zero_sized_hlo_elimination` - Removes zero-sized operations

2. **Transformation Passes**:
   - `bfloat16_propagation` - Numerical precision optimization
   - `memory_space_propagation` - Memory optimization
   - `flatten_call_graph` - Flattens the call hierarchy

3. **Structure Passes**:
   - `tuple_simplifier` - Simplifies tuple operations
   - `reshape_mover` - Optimizes placement of reshape operations
   - `tree_reduction_rewriter` - Optimizes reductions
   - `defuser` - Performs defusion of operations

To get a complete list of available passes, run:
```bash
./bazel-bin/xla/hlo/tools/hlo-opt --list-passes
```

## Reinforcement Learning Environment

The RL environment in this project learns to find optimal compiler pass orderings by:

1. Treating compiler optimization as a sequential decision problem
2. Taking actions (applying optimization passes) on HLO IR
3. Measuring the quality of optimizations through defined metrics
4. Learning which sequences of passes lead to the best optimization outcomes

### Environment Setup

The environment is defined in `reinforcement_learning/xla_opt_env.py` and implements the standard Gym interface:

```python
class XLAOptimizationEnv(gym.Env):
    def __init__(
            self,
            hlo_features: np.ndarray,
            available_passes: List[str],
            max_sequence_length: int = 10,
            verbose: bool = False,
            hlo_file_path: str = None
    ):
        # ...
```

Key components:
- **State**: The current HLO IR features vector
- **Actions**: Available compiler optimization passes
- **Rewards**: Improvement in code quality metrics (e.g., operation count reduction, memory usage reduction)
- **Terminal State**: Reached after applying a fixed number of optimization passes

### RL Agent

We implement a simple Q-learning agent in `reinforcement_learning/simple_agent.py`:

```python
class SimpleQLearningAgent:
    def __init__(
            self,
            action_space_size: int, 
            learning_rate: float = 0.1,
            discount_factor: float = 0.99,
            exploration_rate: float = 1.0,
            exploration_decay: float = 0.995,
            min_exploration_rate: float = 0.01
    ):
        # ...
```

The agent:
- Uses a tabular Q-learning approach with discretized state space
- Balances exploration and exploitation with an epsilon-greedy policy
- Tracks the best optimization sequences found during training

### Training Process

The training loop in `reinforcement_learning/train_agent.py` manages the RL workflow:

1. For each episode:
   - Reset the environment with a fresh HLO module
   - The agent selects and applies optimization passes sequentially
   - After each step, the environment returns the new state and reward
   - The agent updates its Q-values based on the rewards
   - Exploration rate decays over time to focus on exploitation

2. After training:
   - The best sequence of passes is identified
   - Performance metrics are plotted
   - The trained agent can be used for new HLO modules

### XLA Integration

The `xla_interface.py` module bridges the gap between the RL environment and XLA compiler:

```python
class XLAInterface:
    def __init__(self, xla_dir: str = None, verbose: bool = False):
        # ...
        
    def apply_pass(self, hlo_file: str, pass_name: str) -> Tuple[bool, Optional[str]]:
        # ...
        
    def extract_features(self, hlo_file: str) -> np.ndarray:
        # ...
```

This interface:
- Manages HLO file I/O
- Calls the hlo-opt tool to apply compiler passes on HLO modules
- Extracts feature vectors from optimized HLO files
- Computes metrics to determine the rewards for the RL agent

## Running the Project

To run the complete RL optimization pipeline:

1. Build XLA as described in the "XLA Compiler Setup" section
2. Prepare HLO IR files in the `jax_hlo/hlo_data/` directory
3. Train the RL agent:

```bash
cd reinforcement_learning
python train_agent.py
```

4. Analyze the results in the generated plots and logs


## Future Directions

- Implement more sophisticated RL algorithms (DQN, PPO)
- Add more compiler metrics for multi-objective optimization
- Support for multi-device targets (CPU/GPU/TPU)
- Integration with JAX for end-to-end optimization
