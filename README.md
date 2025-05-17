# XLA Compiler Optimizations with RL

This repository contains our final project for CS 521 ML Compilers at UIUC MCS program by Erik Wilson, Rayaan Faruqi, and Ahyush Kaul. We address the challenging problem of finding optimal compiler optimization pass orderings using reinforcement learning.

## Table of Contents

- [Introduction](#introduction)
- [System Architecture](#system-architecture)
- [Quick Start](#quick-start)
- [HLO IR Feature Extraction](#hlo-ir-feature-extraction)
- [XLA Compiler Setup](#xla-compiler-setup)
- [Reinforcement Learning Environment](#reinforcement-learning-environment)
- [Training the RL Agent](#training-the-rl-agent)
- [Testing the Trained Agent](#testing-the-trained-agent)
- [Project Structure](#project-structure)
- [Example Workflow](#example-workflow)
- [Future Directions](#future-directions)

## Introduction

Compiler optimization is a critical component of achieving high-performance code execution. Modern compilers like XLA (Accelerated Linear Algebra) apply numerous optimization passes to transform program representations into more efficient forms. However, the effectiveness of these optimizations depends heavily on the order in which they are applied, creating a combinatorial challenge:

- **Combinatorial Explosion**: With dozens of possible optimization passes, the number of potential orderings is astronomical
- **Context Sensitivity**: The best sequence varies based on the specific code being optimized
- **Irregular Rewards**: Some optimizations only yield benefits after other transformations have been applied
- **Cost Measurement Challenges**: The impact of each pass is difficult to predict without actual execution

Reinforcement Learning (RL) offers a powerful paradigm for tackling this problem, allowing an agent to:
1. Learn from experience which pass sequences work well for different code patterns
2. Adapt to the unique characteristics of each input program
3. Discover non-intuitive optimization sequences that human experts might not consider
4. Balance exploration of novel strategies with exploitation of known effective sequences

Our project implements a Q-learning-based RL agent that interacts with the XLA compiler to discover effective optimization pass sequences for HLO (High Level Optimizer) programs, with the goal of reducing computational cost metrics.

## System Architecture

Our reinforcement learning system for XLA optimization follows this architecture:

```
                                    ┌─────────────────┐
                                    │   HLO Program   │
                                    └────────┬────────┘
                                             │
                                             ▼
┌─────────────────┐  action  ┌─────────────────────────┐  step()   ┌─────────────────┐
│   RL Agent      │ ─────────► XLA Optimization Env    │◄───────────│  XLA Interface  │
│ (Q-Learning)    │◄────────┐└─────────────────────────┘           │                 │
└─────────────────┘  reward │            │                         └────────┬────────┘
                             │            │ state                        ↗ ↓ │
                             └────────────┘                     feature   │  │
                                                              extraction  │  │
                                                                    ┌─────┘  │
                                         ┌───────────────────┐     │        │
                                         │ jax_hlo module    │◄────┘        │
                                         │ (Feature Extractor)│              │
                                         └───────────────────┘               │
                                                                             ▼
                                                                   ┌─────────────────┐
                                                                   │  XLA Compiler   │
                                                                   │  (hlo-opt)      │
                                                                   └─────────────────┘
```

The workflow:
1. The agent chooses an optimization pass (action) based on the current program state
2. The environment applies this pass via the XLA interface, which calls the XLA compiler
3. XLA compiler performs the optimization and produces a modified HLO program
4. The XLA interface uses the jax_hlo module to extract features from the optimized HLO program
5. These features form the state representation passed back to the environment
6. The environment calculates the reward (cost improvement) based on the new state
7. The agent updates its Q-table based on the reward and new state
8. This cycle repeats until a terminal condition (max steps or no improvement)

## Quick Start

Get up and running with the RL agent for XLA compiler pass optimization in just a few steps:

### Prerequisites
- Python 3.8+ with pip
- Bazel/Bazelisk for building XLA
- Git

### Installation

1. **Clone repositories**:
   ```bash
   git clone https://github.com/yourusername/rl_xla_opt_cs521.git
   cd rl_xla_opt_cs521
   git clone https://github.com/openxla/xla.git ../xla
   ```

2. **Install Python dependencies**:
   ```bash
   pip install numpy matplotlib pandas gymnasium
   ```

3. **Build XLA compiler**:
   ```bash
   cd ../xla
   # For macOS users
   echo "build --apple_platform_type=macos" > .bazelrc
   echo "build --cxxopt=-std=c++17" >> .bazelrc
   # Build the hlo-opt tool
   bazelisk build -c opt //xla/hlo/tools:hlo-opt
   cd ../rl_xla_opt_cs521
   ```

### Training & Testing

1. **Train the agent**:
   ```bash
   cd reinforcement_learning
   python train_agent.py
   # Trained model will be saved to outputs/run_TIMESTAMP/models/trained_agent.pkl
   ```

2. **Test the agent**:
   ```bash
   python test_agent.py --model outputs/run_TIMESTAMP/models/trained_agent.pkl --hlo_dir ../jax_hlo/test_set/ --xla_dir ../../xla
   # Results saved to test_output/run_TIMESTAMP/
   ```

3. **Visualize results**:
   - Check plots in `test_output/run_TIMESTAMP/plots/`
   - View metrics in `test_output/run_TIMESTAMP/csv_data/`

### What's Next?
- Explore detailed options in the Training and Testing sections below
- Customize the agent behavior with different parameters
- Analyze optimization patterns in the results

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

An example feature vector:
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

2. Configure the build:
- For macOS (different computers will need different configs):
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
- Other systems will require different configurations.

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

## Example List of XLA Passes (Subset Only):

1. AssumeGatherIndicesInBoundRewriteToCopy
2. ReduceWindowToReduceAndBroadcast
3. add-original-value
4. algsimp
5. all-gather-bcast-reorder
6. all-gather-combiner
7. all-gather-cse
8. all-reduce-combiner
9. all-reduce-contiguous
10. all-reduce-folder
11. ar-crs-combiner
12. async-collective-creator
13. batch-dot-simplification
14. bf16-mixed-precision-removal
15. bfloat16-fold
16. bfloat16-propagation
17. broadcast_canonicalizer
18. cholesky_expander
19. collective-quantizer
20. collective-transformation-reorderer
21. collectives-schedule-linearizer
22. comparison-expander
23. computation-deduplicator
24. conditional-canonicalizer
25. constant_folding
26. control-dep-remover
27. convert-async-collectives-to-sync
28. convert-memory-placement-to-internal-annotations
29. convert-mover
30. convert_operand_folding
31. convolution-group-converter
32. convolution-pred-expander
33. convolution_4d_expander
34. cse_barrier_expander
35. dce
36. defuser
37. despecializer
38. dot-merger
39. dot_decomposer
40. dot_dimension_merger
41. dynamic-dimension-simplifier
42. dynamic-index-splitter
43. eigh_expander
44. element_type_converter
45. flatten-call-graph
46. float-normalization-bf16
47. fusion_constant_sinking
48. gather_simplifier
49. hlo-constant-splitter
50. hlo-descheduler
51. hlo-memory-scheduler
52. hlo-trivial-scheduler
53. host-memory-transfer-asyncifier
54. host-offload-legalize
55. host-offloader
56. host-offloading-prepare-elide-move-to-host
57. indexed-array-analysis-printer-pass
58. infeed-token-propagation
59. instruction-hoister
60. literal-canonicalizer
61. logistic-expander
62. memory-space-propagation
63. operand_upcaster
64. optimize_input_output_buffer_alias
65. qr_expander
66. real_imag_expander
67. reduce-decomposer
68. reduce-window-rewriter
69. reorder-convert-reduce-add
70. reorder-reduce-transpose
71. reshape-decomposer
72. reshape-mover
73. result_caster
74. rng-bit-generator-expander
75. rng-expander
76. root-instruction-sinker
77. simplify-fp-conversions
78. simplify-sorts
79. slice-sinker
80. stable-sort-expander
81. stochastic_convert_decomposer
82. sub-byte-size-setter
83. test-only-algebraic-simplifier-with-onednn-enabled
84. test-only-bar2hello
85. test-only-foo2bar
86. test-only-xla-builder
87. tree_reduction_rewriter
88. tuple-simplifier
89. while-loop-trip-count-annotator
90. zero_sized_hlo_elimination


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
python reinforcement_learning/train_agent.py
```
- Note: In order to run on different systems, the paths within main() of the train_agent system will need to be changed to the appropriate directories.

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
   
Our interface internally uses the same structure as the first command to run individual passes.

### Optimization Pass Groups

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
            xla_dir: str,
            initial_hlo_file_path: str,
            max_sequence_length: int = 30,
            no_improvement_threshold: int = 5,
            verbose: bool = False
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

## Training the RL Agent

Ensure that within the main() function of the train_agent.py script, that you have pointed to the appropriate XLA directory. To train the reinforcement learning agent:

```bash
cd rl_xla_opt_cs521/reinforcement_learning
python train_agent.py
```

### Training Options

The training script automatically scans the `jax_hlo/hlo_data/` directory for HLO files and trains the agent on all valid files. Some key parameters in the script that you can modify:

- `episodes_per_file`: Number of training episodes for each HLO file (default: 25)
- `max_steps_per_episode`: Maximum optimization passes per episode (default: 25)
- `xla_dir`: Path to your XLA build directory
- `output_dirs`: Where you would like for statistics, output, and the model to be stored

You can modify these parameters in the `main()` function of `train_agent.py`.

The training process:
1. Initializes the XLA environment and the Q-learning agent
2. For each HLO file, runs multiple episodes of training
3. In each episode, the agent learns to select optimization passes that maximize reward (cost reduction)
4. After training, saves the agent's Q-table and parameters to `models/trained_agent.pkl`

Training progress is logged to the console, and reward plots are generated for each file.

### Customizing the Agent

The agent's behavior can be customized by modifying:

1. **State representation**: Edit the `discretize_state` method in `simple_agent.py` to change how graph states are represented
2. **Reward function**: Modify the `_calculate_cost` method in `xla_opt_env.py` to change how reward is calculated
3. **Exploration strategy**: Adjust the exploration parameters in `train_agent.py`, inside the train_on_multiple_files() method:

   ```python
   agent = SimpleQLearningAgent(
       action_space_size=len(available_passes),
       learning_rate=0.2,
       discount_factor=0.95,
       exploration_rate=1.0,
       exploration_decay=0.995,
       min_exploration_rate=0.01
   )
   ```

## Testing the Trained Agent

After training, you can test the agent on individual HLO files or directories of files. The test results are organized in a structured directory with timestamped outputs.

### Testing on a Single File

```bash
python test_agent.py --model outputs/run_TIMESTAMP/models/trained_agent.pkl --hlo ../jax_hlo/test_set/your_file.hlo --xla_dir ../../xla [--verbose]
```

### Testing on Multiple Files

```bash
python test_agent.py --model outputs/run_TIMESTAMP/models/trained_agent.pkl --hlo_dir ../jax_hlo/test_set/ --xla_dir ../../xla [--verbose]
```

### Test Options

- `--model`: Path to the saved model file (required)
- `--hlo`: Path to a single HLO file to test (optional if `--hlo_dir` is provided)
- `--hlo_dir`: Directory containing multiple HLO files to test (optional if `--hlo` is provided)
- `--xla_dir`: Path to the XLA directory (required)
- `--max_steps`: Maximum number of steps (default: 30)
- `--verbose`: Print detailed progress (optional)
- `--try_all`: Try all passes and report effectiveness (optional)
- `--manual`: Try a manually curated sequence of passes (optional)
- `--no_plots`: Disable plotting (optional)
- `--output_dir`: Custom output directory path (optional)

### Test Output Structure

Test results are organized in a structured directory similar to training outputs:

```
test_output/
└── run_TIMESTAMP/
    ├── csv_data/
    │   ├── file1_name/
    │   │   ├── metrics_file1_name.csv     # Detailed metrics for file1
    │   │   ├── passes_file1_name.csv      # Applied passes sequence
    │   │   └── cost_history_file1_name.csv # Cost at each step
    │   ├── file2_name/
    │   │   └── ...
    │   └── test_results_summary.csv       # Summary metrics across all files
    ├── plots/
    │   ├── file1_name/
    │   │   ├── cost_reduction_file1_name.png    # Cost reduction plot
    │   │   └── percent_reduction_file1_name.png # Percent reduction plot
    │   ├── file2_name/
    │   │   └── ...
    │   ├── summary_cost_reduction.png     # Comparison across files
    │   ├── summary_percent_reduction.png  # Percent comparison
    │   └── unique_actions_vs_reduction.png # Correlation analysis
    └── analysis/
        └── ...                            # Additional analysis data
```

### Understanding Test Results

The test script outputs:
1. For single file testing:
   - The sequence of applied optimization passes
   - Initial and final computational cost
   - Cost reduction and percentage improvement
   - Visualization of cost reduction over steps

2. For multiple file testing:
   - Summary statistics for all tested files
   - Visualizations comparing performance across files
   - CSV files with detailed metrics for each file
   - Summary across all tested files

Sample output metrics include:
- Total reward (cost reduction)
- Initial and final computational cost
- Percentage cost reduction
- Number of unique optimization passes applied
- Number of unique states visited
- Step-by-step progression of cost reduction

### Project Structure

- `reinforcement_learning/`: Contains all RL-related code
  - `simple_agent.py`: Implementation of the Q-learning agent
  - `xla_opt_env.py`: Gymnasium environment for XLA optimization
  - `XLA_interface.py`: Interface between Python and XLA compiler
  - `train_agent.py`: Script for training the agent with comprehensive metrics collection
  - `test_agent.py`: Script for testing the trained agent on new HLO files
  - `utils.py`: Utility functions
  - `models/`: Directory for saved model files
  - `outputs/`: Contains timestamped output directories from training runs
    - `run_TIMESTAMP/`: Each run gets its own directory
      - `csv_data/`: CSV files with training statistics
      - `plots/`: Visualizations of training progress
      - `models/`: Saved model checkpoints
      - `analysis/`: Additional analysis data
  - `test_output/`: Contains timestamped output directories from test runs
    - `run_TIMESTAMP/`: Each test run gets its own directory
      - `csv_data/`: CSV files with test metrics for each HLO file
      - `plots/`: Visualizations of test results
      - `analysis/`: Additional analysis data
  - `optimized_hlo/`: Temporary directory for storing HLO files during optimization
  - `unparseable_hlo/`: Directory with problematic HLO files for debugging

- `jax_hlo/`: Contains HLO-related code and data
  - `hlo_feature_extraction.py`: Functions to extract graph features from HLO modules
  - `hlo_parser.py`: Parser for HLO files
  - `hlo_data/`: Directory containing training HLO files
  - `test_set/`: Directory containing test HLO files

- `xla/`: XLA compiler codebase (not included in repository, must be set up separately)

### Example Workflow

Here's a complete example workflow, provided prerequisites as discussed previously are met.

```bash
# Build XLA (if not done already)
cd ../xla
bazelisk build -c opt //xla/hlo/tools:hlo-opt

# Train the agent
cd ../rl_xla_opt_cs521/reinforcement_learning
python train_agent.py
# Training outputs will be stored in outputs/run_TIMESTAMP/

# After successful training, find the model file
MODEL_PATH="outputs/run_TIMESTAMP/models/trained_agent.pkl"

# Test on a single file with detailed output
python test_agent.py --model $MODEL_PATH --hlo ../jax_hlo/test_set/black_scholes_call_option.hlo --xla_dir ../../xla --verbose

# Test on multiple files and generate summary
python test_agent.py --model $MODEL_PATH --hlo_dir ../jax_hlo/test_set/ --xla_dir ../../xla
# Test results will be stored in test_output/run_TIMESTAMP/

# Examine results
# - Check test_output/run_TIMESTAMP/csv_data/ for detailed metrics
# - View visualizations in test_output/run_TIMESTAMP/plots/
# - Review summaries in test_output/run_TIMESTAMP/csv_data/test_results_summary.csv
```

This workflow will train the agent, test it on sample HLO files, and provide comprehensive visualizations and metrics in an organized directory structure for analysis.

The test results can be used to:
1. Analyze which compiler passes are most effective for different HLO files
2. Compare optimization strategies across different program types
3. Identify patterns of successful optimization sequences
4. Quantify the effectiveness of the trained reinforcement learning agent

## Future Directions

- Implement more sophisticated RL algorithms (DQN, PPO)
- Update state discretization strategy
- Add more compiler metrics for multi-objective optimization
- Support for multi-device targets (CPU/GPU/TPU)
- Integration with JAX for end-to-end optimization