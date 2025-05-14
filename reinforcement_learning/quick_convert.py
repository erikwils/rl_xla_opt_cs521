import os
import sys
from XLA_interface import XLAInterface


xla_dir = "/Users/rayaanfaruqi/Documents/CS521/Final_Project/xla"
file_path = "/Users/rayaanfaruqi/Documents/CS521/Final_Project/rl_xla_opt_cs521/jax_hlo/hlo_data/mha.hlo"
interface = XLAInterface(xla_dir=xla_dir)
passes = interface.get_available_passes()


# This checks the initial costs of an HLO module.

features = interface.extract_features(file_path)

# Combine various metrics for cost: 1. Total number of ops, 2. Memory footprint, 3. Graph complexity
excluded_keys = ['source_file', 'graph_nodes', 'graph_edge_links', 'total_bytes',
                'input_bytes', 'output_bytes', 'elemwise_ratio', 'mixed_precision']

total_ops = 0
for op_name, value in features.items():
    if op_name not in excluded_keys:
        # make sure we're only summing numeric values
        if isinstance(value, (int, float)):
            total_ops += value

memory_cost = features.get('total_bytes', 0) / 1000.0
graph_nodes = features.get('graph_nodes', [])
graph_complexity = len(graph_nodes) * 10 # weight by importance

# Simple weighted sum without normalization
# Use scaling factors to bring components to similar magnitude
weighted_total_ops = total_ops * 10.0
weighted_memory_cost = memory_cost * 1.0
weighted_graph_complexity = graph_complexity * 0.1
cost = weighted_total_ops + weighted_graph_complexity + weighted_memory_cost

print(f"\nCost components: ops={total_ops}, memory={memory_cost}, complexity={graph_complexity}")
print(f"Weighted components: ops={weighted_total_ops}, memory={weighted_memory_cost}, complexity={weighted_graph_complexity}")
print(f"Total cost: {cost}\n")

interface.apply_pass(file_path, passes[2])
