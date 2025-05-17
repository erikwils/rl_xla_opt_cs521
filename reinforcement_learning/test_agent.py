import os
import pickle
import argparse
import numpy as np
from simple_agent import SimpleQLearningAgent
from XLA_interface import XLAInterface
from xla_opt_env import XLAOptimizationEnv
from collections import defaultdict as DefaultDict
import matplotlib.pyplot as plt
import glob
import pandas as pd
from typing import List, Dict, Tuple, Any
import datetime

def test_agent(agent : SimpleQLearningAgent, env : XLAOptimizationEnv, max_steps=30, verbose=True, use_manual=False):
    """Test our trained agent on a single HLO file."""
    state, info = env.reset()

    total_reward = 0
    done = False
    step = 0
    applied_passes = []
    
    # Setup tracking for state and actions
    state_tracker = StateTracker(verbose)
    action_manager = ActionManager(env.available_passes, verbose)
    
    # Define commonly beneficial passes to try when state not in q-table
    beneficial_passes = get_beneficial_passes()

    while not done and step < max_steps:
        # Get discretized state and update state tracker
        current_state = agent.discretize_state(state)
        state_tracker.track_state(current_state, state, step)
        
        # Select action based on state and Q-values
        action = select_action(
            agent=agent, 
            current_state=current_state,
            env=env, 
            state_tracker=state_tracker,
            action_manager=action_manager,
            beneficial_passes=beneficial_passes,
            use_manual=use_manual,
            manual_step=step,
            verbose=verbose
        )
        
        # Track the selected action as tried
        action_manager.mark_action_tried(action)
        
        # Get pass name and apply it
        selected_pass = env.available_passes[action]
        applied_passes.append(selected_pass)

        # Take step in environment
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Update for next iteration
        state_tracker.update_previous_state(state)
        state = next_state
        total_reward += reward
        step += 1
        
        if verbose:
            print(f"Step {step}: Applied pass {selected_pass}, Reward: {reward:.4f}")
            
            # Display more detailed graph info after multiple unchanged states
            if state_tracker.same_state_count > 2:
                analyze_unchanged_graph(env, action_manager, verbose)
    
    # Analyze state changes throughout the episode
    state_stats = state_tracker.get_statistics()
    
    if verbose and len(state_tracker.state_history) > 1:
        print(f"\nState change analysis:")
        print(f"Total steps: {step}, Unique states visited: {state_stats['unique_states']}")
        print(f"State uniqueness ratio: {state_stats['state_uniqueness_ratio']:.2f}")
    
    # Calculate metrics for this test
    metrics = calculate_test_metrics(env, total_reward, step, applied_passes, state_tracker, action_manager)
    
    # Display final results
    if verbose:
        print_test_results(total_reward, applied_passes, env, state_tracker)
    
    return total_reward, applied_passes, env.cost_history, metrics


class StateTracker:
    """Tracks state changes during testing."""
    
    def __init__(self, verbose=True):
        self.previous_state = None
        self.previous_discretized_state = None
        self.same_state_count = 0
        self.state_history = []
        self.verbose = verbose
        
    def track_state(self, current_state, raw_state, step):
        """Track the current state and check if it has changed since the previous step."""
        self.state_history.append(current_state)
        
        if self.previous_discretized_state is not None:
            if current_state == self.previous_discretized_state:
                self.same_state_count += 1
                if self.verbose:
                    self._report_unchanged_state(current_state, raw_state)
            else:
                # State changed, reset the counter
                self.same_state_count = 0
                if self.verbose:
                    self._report_state_change(current_state)
    
    def _report_unchanged_state(self, current_state, raw_state):
        """Report information about an unchanged state."""
        print(f"WARNING: Discretized state unchanged after previous action (count: {self.same_state_count})")
        
        # Provide more detailed analysis of why states appear the same
        print(f"Discretized state: {current_state[:5]}... (truncated, {len(current_state)} features total)")
        
        # Check if raw graph features changed even if discretized state didn't
        self._analyze_raw_state_changes(raw_state)
    
    def _analyze_raw_state_changes(self, raw_state):
        """Analyze whether raw state changed even if discretized state didn't."""
        nodes_changed = False
        edges_changed = False
        
        if self.previous_state is not None and 'nodes' in raw_state and 'nodes' in self.previous_state:
            if len(raw_state['nodes']) != len(self.previous_state['nodes']):
                nodes_changed = True
            elif not np.array_equal(np.array(raw_state['nodes']), np.array(self.previous_state['nodes'])):
                nodes_changed = True
        
        if self.previous_state is not None and 'edge_links' in raw_state and 'edge_links' in self.previous_state:
            if len(raw_state['edge_links']) != len(self.previous_state['edge_links']):
                edges_changed = True
            elif not np.array_equal(np.array(raw_state['edge_links']), np.array(self.previous_state['edge_links'])):
                edges_changed = True
        
        if nodes_changed or edges_changed:
            print(f"NOTE: Raw graph changed but discretized to same state!")
            print(f"  Nodes changed: {nodes_changed}, Edges changed: {edges_changed}")
            if self.previous_state is not None:
                print(f"  Current nodes: {len(raw_state.get('nodes', []))}, Previous: {len(self.previous_state.get('nodes', []))}")
                print(f"  Current edges: {len(raw_state.get('edge_links', []))}, Previous: {len(self.previous_state.get('edge_links', []))}")
        else:
            print(f"Graph structure truly unchanged after previous action")
    
    def _report_state_change(self, current_state):
        """Report information about a state change."""
        if self.previous_discretized_state is None:
            print(f"Initial state: {current_state[:3]}...")
            print(f"Full current state: {current_state}")
            return
            
        print(f"State changed! From {self.previous_discretized_state[:3]}... to {current_state[:3]}...")
        print(f"Full current state: {current_state}")
        
        # Calculate number of changed features
        if len(self.previous_discretized_state) == len(current_state):
            changes = sum(1 for a, b in zip(self.previous_discretized_state, current_state) if a != b)
            print(f"Number of changed features: {changes}/{len(current_state)}")
    
    def update_previous_state(self, state):
        """Update the previous state with the current state."""
        self.previous_state = state.copy() if state is not None else None
    
    def get_statistics(self):
        """Get statistics about state changes."""
        unique_states = len(set(self.state_history))
        total_steps = len(self.state_history)
        
        return {
            'unique_states': unique_states,
            'total_states': total_steps,
            'state_uniqueness_ratio': unique_states / total_steps if total_steps > 0 else 0
        }


class ActionManager:
    """Manages action selection and tracking during testing."""
    
    def __init__(self, available_passes, verbose=True):
        self.available_passes = available_passes
        self.tried_actions = []
        self.verbose = verbose
    
    def mark_action_tried(self, action):
        """Mark an action as tried."""
        self.tried_actions.append(action)
    
    def get_untried_actions(self, total_actions):
        """Get a list of actions that haven't been tried yet."""
        return [i for i in range(total_actions) if i not in self.tried_actions]
    
    def get_random_untried_action(self, total_actions):
        """Get a random action that hasn't been tried yet."""
        untried_actions = self.get_untried_actions(total_actions)
        if untried_actions:
            action = np.random.choice(untried_actions)
            if self.verbose:
                print(f"Selecting random untried action: {self.available_passes[action]}")
            return action
        else:
            # All actions have been tried, just pick any random action
            action = np.random.randint(total_actions)
            if self.verbose:
                print(f"All actions tried, selecting random action: {self.available_passes[action]}")
            return action
    
    def get_best_untried_positive_action(self, q_values):
        """Get the highest Q-value action that hasn't been tried yet."""
        positive_q_indices = np.where(q_values > 0)[0]
        untried_positive_indices = [i for i in positive_q_indices if i not in self.tried_actions]
        
        if untried_positive_indices:
            best_untried_idx = max(untried_positive_indices, key=lambda i: q_values[i])
            return best_untried_idx
        
        return None


def get_beneficial_passes():
    """Return a list of commonly beneficial passes to try when state not in q-table."""
    return [
        "algsimp", 
        "all-reduce-folder", 
        "computation-deduplicator", 
        "dot-merger",
        "host-offloader",
        "all-gather-combiner",
        "simplify-fp-conversions",
        "slice-sinker", 
        "tuple-simplifier"
    ]


def select_action(agent, current_state, env, state_tracker, action_manager, 
                 beneficial_passes, use_manual=False, manual_step=0, verbose=True):
    """Select the next action based on the current state and policy."""
    if not use_manual:
        return select_action_using_policy(
            agent=agent,
            current_state=current_state,
            env=env,
            state_tracker=state_tracker,
            action_manager=action_manager,
            beneficial_passes=beneficial_passes,
            verbose=verbose
        )
    else:
        return select_action_manually(
            env=env,
            beneficial_passes=beneficial_passes,
            manual_step=manual_step,
            verbose=verbose
        )


def select_action_using_policy(agent, current_state, env, state_tracker, 
                              action_manager, beneficial_passes, verbose=True):
    """Select an action using the agent's policy (q-table or exploration)."""
    if current_state in agent.q_table:
        return select_action_from_qtable(
            agent=agent,
            current_state=current_state,
            env=env,
            state_tracker=state_tracker,
            action_manager=action_manager,
            verbose=verbose
        )
    else:
        # State not in q-table, try beneficial passes or random
        return select_action_for_unknown_state(
            env=env,
            beneficial_passes=beneficial_passes,
            action_manager=action_manager,
            verbose=verbose
        )


def select_action_from_qtable(agent, current_state, env, state_tracker, action_manager, verbose=True):
    """Select action based on Q-values when state is in Q-table."""
    q_values = agent.q_table[current_state].copy()
    
    # Mark already tried actions as very negative
    for action_idx in action_manager.tried_actions:
        q_values[action_idx] = -float('inf')
    
    # Get positive Q-values that haven't been tried yet
    positive_q_indices = np.where(q_values > 0)[0]
    untried_positive_indices = [i for i in positive_q_indices if i not in action_manager.tried_actions]
    
    if state_tracker.same_state_count > 2 and len(untried_positive_indices) > 0:
        # If state hasn't changed but we have untried positive actions, choose the best one
        best_untried_idx = max(untried_positive_indices, key=lambda i: q_values[i])
        action = best_untried_idx
        if verbose:
            print(f"Trying alternative action with q-value {q_values[action]:.4f}: {env.available_passes[action]}")
    elif state_tracker.same_state_count > 2:
        # If state hasn't changed and no positive untried actions, pick a random action
        action = action_manager.get_random_untried_action(len(env.available_passes))
    else:
        # Normal case - use the best q-value if positive, otherwise random
        best_action_idx = int(np.argmax(q_values))
        best_q_value = q_values[best_action_idx]
        
        if verbose:
            print(f"Found state in q-table, best q-value: {best_q_value:.4f} at index {best_action_idx}")
            print(f"Best action is: {env.available_passes[best_action_idx]}")
            if len(positive_q_indices) > 0:
                print(f"Positive q-value indices: {positive_q_indices}")
                print(f"Corresponding passes: {[env.available_passes[i] for i in positive_q_indices]}")
        
        if best_q_value > 0:
            action = best_action_idx
        else:
            # No positive q-values, select random action
            action = np.random.randint(len(env.available_passes))
            if verbose:
                print(f"No positive q-values, selecting random action: {env.available_passes[action]}")
    
    return action


def select_action_for_unknown_state(env, beneficial_passes, action_manager, verbose=True):
    """Select action when state is not in the Q-table."""
    if verbose:
        print(f"State not found in q-table")
    
    if beneficial_passes and np.random.random() < 0.7:  # 70% chance to try beneficial pass
        for pass_name in beneficial_passes:
            if pass_name in env.available_passes:
                action = env.available_passes.index(pass_name)
                if verbose:
                    print(f"Trying known beneficial pass: {pass_name}")
                return action
        
        # No beneficial pass found, pick random action
        action = np.random.randint(len(env.available_passes))
        if verbose:
            print(f"No beneficial pass available, choosing random action: {env.available_passes[action]}")
        return action
    else:
        # Simply pick random action
        action = np.random.randint(len(env.available_passes))
        if verbose:
            print(f"Choosing random action: {env.available_passes[action]}")
        return action


def select_action_manually(env, beneficial_passes, manual_step, verbose=True):
    """Select action based on a manual sequence (for benchmarking)."""
    current_pass_idx = manual_step % len(beneficial_passes)
    pass_name = beneficial_passes[current_pass_idx]
    
    if pass_name in env.available_passes:
        action = env.available_passes.index(pass_name)
        if verbose:
            print(f"Manual mode: Trying pass {pass_name} from beneficial list")
    else:
        # If pass not available, use the first available pass
        action = 0
        if verbose:
            print(f"Manual mode: Pass {pass_name} not available, using {env.available_passes[0]}")
    
    return action


def analyze_unchanged_graph(env, action_manager, verbose=True):
    """Analyze and print information about an unchanged graph state."""
    if not verbose:
        return
    
    print("\nDetailed graph structure after multiple unchanged states:")
    graph_features = env.current_features
    
    # Safely access graph nodes
    node_count = len(graph_features.get('graph_nodes', []))
    print(f"Number of nodes: {node_count}")
    if 'graph_nodes' in graph_features and len(graph_features['graph_nodes']) > 0:
        print(f"First few nodes: {graph_features['graph_nodes'][:3]}")
    
    # Safely check for graph edges/links
    if 'graph_edges' in graph_features:
        print(f"Number of edges: {len(graph_features['graph_edges'])}")
        if len(graph_features['graph_edges']) > 0:
            print(f"First few edges: {graph_features['graph_edges'][:3]}")
    elif 'graph_edge_links' in graph_features:
        print(f"Number of edge links: {len(graph_features['graph_edge_links'])}")
        if len(graph_features['graph_edge_links']) > 0:
            print(f"First few edge links: {graph_features['graph_edge_links'][:3]}")
    
    print(f"Graph hash: {hash(str(graph_features))}")
    print(f"Tried actions: {[env.available_passes[i] for i in action_manager.tried_actions]}")


def calculate_test_metrics(env, total_reward, step, applied_passes, state_tracker, action_manager):
    """Calculate metrics for the test run."""
    state_stats = state_tracker.get_statistics()
    
    return {
        "total_reward": total_reward,
        "initial_cost": env.cost_history[0],
        "final_cost": env.cost_history[-1],
        "cost_reduction": env.cost_history[0] - env.cost_history[-1],
        "percent_reduction": ((env.cost_history[0] - env.cost_history[-1]) / env.cost_history[0]) * 100 if env.cost_history[0] > 0 else 0,
        "steps_taken": step,
        "same_state_count": state_tracker.same_state_count,
        "unique_actions": len(set(applied_passes)),
        "unique_states": state_stats['unique_states'],
        "state_uniqueness_ratio": state_stats['state_uniqueness_ratio']
    }


def print_test_results(total_reward, applied_passes, env, state_tracker):
    """Print the final results of the test."""
    state_stats = state_tracker.get_statistics()
    
    print(f"\nTest Complete:")
    print(f"Total Reward: {total_reward:.4f}")
    print(f"Applied Passes: {applied_passes}")
    print(f"Initial Cost: {env.cost_history[0]:.4f}")
    print(f"Final Cost: {env.cost_history[-1]:.4f}")
    print(f"Cost Reduction: {env.cost_history[0] - env.cost_history[-1]:.4f}")
    print(f"Same state count: {state_tracker.same_state_count}")
    print(f"Total unique actions tried: {len(set(applied_passes))}")
    
    if state_stats['unique_states'] > 0:
        print(f"Unique states: {state_stats['unique_states']}/{state_stats['total_states']} (ratio: {state_stats['state_uniqueness_ratio']:.2f})")

def test_on_multiple_files(agent: SimpleQLearningAgent, hlo_files: List[str], xla_dir: str, 
                           max_steps: int = 30, verbose: bool = False, use_manual: bool = False,
                           plot_results: bool = True, output_dir: str = None) -> Dict:
    """
    Test the agent on multiple HLO files and aggregate results.
    
    Args:
        agent: The trained RL agent
        hlo_files: List of HLO file paths to test on
        xla_dir: Path to the XLA directory
        max_steps: Maximum steps for each test
        verbose: Whether to print detailed progress
        use_manual: Whether to use the manual pass sequence
        plot_results: Whether to generate summary plots
        output_dir: Base output directory (default: creates test_output directory)
        
    Returns:
        Dictionary of summary results and metrics for each file
    """
    # Create structured output directories
    if output_dir is None:
        # Get script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Create a timestamp for unique output directories
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Main output directory
        output_dir = os.path.join(script_dir, "test_output", f"run_{timestamp}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create subdirectories
    csv_dir = os.path.join(output_dir, "csv_data")
    plots_dir = os.path.join(output_dir, "plots")
    analysis_dir = os.path.join(output_dir, "analysis")
    
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(analysis_dir, exist_ok=True)
    
    print(f"Test results will be saved to: {output_dir}")
    
    all_results = {}
    summary_metrics = {
        "total_files": len(hlo_files),
        "files_with_improvement": 0,
        "total_cost_reduction": 0,
        "avg_cost_reduction": 0,
        "avg_percent_reduction": 0,
        "max_cost_reduction": 0,
        "max_percent_reduction": 0,
        "best_file": "",
        "worst_file": "",
        "all_file_metrics": {}
    }
    
    file_metrics = []
    
    print(f"\nTesting agent on {len(hlo_files)} HLO files...\n")
    
    # Ensure XLA directory is absolute path
    xla_dir = os.path.abspath(xla_dir)
    
    for i, hlo_file in enumerate(hlo_files):
        # Ensure HLO file path is absolute
        hlo_file = os.path.abspath(hlo_file)
        file_name = os.path.basename(hlo_file)
        file_base_name = os.path.splitext(file_name)[0]
        
        # Create file-specific directories
        file_csv_dir = os.path.join(csv_dir, file_base_name)
        file_plots_dir = os.path.join(plots_dir, file_base_name)
        os.makedirs(file_csv_dir, exist_ok=True)
        os.makedirs(file_plots_dir, exist_ok=True)
        
        print(f"[{i+1}/{len(hlo_files)}] Testing on: {file_name}")
        
        if not os.path.exists(hlo_file):
            print(f"  ERROR: File not found: {hlo_file}")
            continue
            
        # Create environment for this file
        try:
            env = XLAOptimizationEnv(
                xla_dir=xla_dir,
                initial_hlo_file_path=hlo_file,
                max_sequence_length=max_steps,
                verbose=verbose
            )
            
            # Run test on this file
            _, applied_passes, cost_history, metrics = test_agent(
                agent=agent, 
                env=env, 
                max_steps=max_steps,
                verbose=verbose,
                use_manual=use_manual
            )
            
            # Save metrics for this file
            metrics["file_name"] = file_name
            metrics["file_path"] = hlo_file
            file_metrics.append(metrics)
            summary_metrics["all_file_metrics"][file_name] = metrics
            
            # Save detailed metrics to CSV
            file_metrics_path = os.path.join(file_csv_dir, f"metrics_{file_base_name}.csv")
            metrics_df = pd.DataFrame([metrics])
            metrics_df.to_csv(file_metrics_path, index=False)
            
            # Save pass sequence to CSV
            passes_df = pd.DataFrame({
                'step': range(len(applied_passes)),
                'pass_name': applied_passes
            })
            passes_path = os.path.join(file_csv_dir, f"passes_{file_base_name}.csv")
            passes_df.to_csv(passes_path, index=False)
            
            # Save cost history to CSV
            cost_df = pd.DataFrame({
                'step': range(len(cost_history)),
                'cost': cost_history,
                'reduction': [cost_history[0] - c for c in cost_history],
                'percent_reduction': [(cost_history[0] - c) / cost_history[0] * 100 if cost_history[0] > 0 else 0 
                                      for c in cost_history]
            })
            cost_path = os.path.join(file_csv_dir, f"cost_history_{file_base_name}.csv")
            cost_df.to_csv(cost_path, index=False)
            
            # Update summary metrics
            cost_reduction = metrics["cost_reduction"]
            percent_reduction = metrics["percent_reduction"]
            
            if cost_reduction > 0:
                summary_metrics["files_with_improvement"] += 1
            
            summary_metrics["total_cost_reduction"] += cost_reduction
            
            # Track best and worst files
            if cost_reduction > summary_metrics["max_cost_reduction"]:
                summary_metrics["max_cost_reduction"] = cost_reduction
                summary_metrics["max_percent_reduction"] = percent_reduction
                summary_metrics["best_file"] = file_name
                
            # Generate individual file plot if requested
            if plot_results:
                plot_cost_history(
                    cost_history, 
                    os.path.join(file_plots_dir, f"cost_reduction_{file_base_name}.png")
                )
                
                # Additional plot for percent reduction
                plt.figure(figsize=(10, 5))
                plt.plot(cost_df['step'], cost_df['percent_reduction'], 'g-')
                plt.title(f"Percent Cost Reduction During Testing - {file_base_name}")
                plt.xlabel("Step")
                plt.ylabel("Cost Reduction (%)")
                plt.grid(True)
                plt.savefig(os.path.join(file_plots_dir, f"percent_reduction_{file_base_name}.png"))
                plt.close()
            
            print(f"  Initial cost: {metrics['initial_cost']:.4f}")
            print(f"  Final cost: {metrics['final_cost']:.4f}")
            print(f"  Reduction: {metrics['cost_reduction']:.4f} ({metrics['percent_reduction']:.2f}%)")
            print(f"  Unique actions: {metrics['unique_actions']}")
            print()
        
        except Exception as e:
            print(f"  Error testing file {file_name}: {str(e)}")
            print()
            continue
    
    # Check if we have any successful results
    if not file_metrics:
        print("No successful test results to report.")
        return summary_metrics
        
    # Calculate final summary statistics
    if len(file_metrics) > 0:
        summary_metrics["avg_cost_reduction"] = summary_metrics["total_cost_reduction"] / len(file_metrics)
        
        # Calculate average percent reduction across all files
        total_percent = sum(metrics["percent_reduction"] for metrics in file_metrics)
        summary_metrics["avg_percent_reduction"] = total_percent / len(file_metrics)
    
    # Create DataFrame for easier analysis
    metrics_df = pd.DataFrame(file_metrics)
    summary_metrics["metrics_df"] = metrics_df
    
    # Save summary metrics to CSV
    summary_csv_path = os.path.join(csv_dir, 'test_results_summary.csv')
    metrics_df.to_csv(summary_csv_path, index=False)
    
    # Print summary
    print("\n===== Summary Results =====")
    print(f"Total files tested: {summary_metrics['total_files']}")
    print(f"Files with successful tests: {len(file_metrics)}")
    print(f"Files with cost reduction: {summary_metrics['files_with_improvement']} ({summary_metrics['files_with_improvement'] / len(file_metrics) * 100:.2f}% of successful tests)")
    print(f"Total cost reduction: {summary_metrics['total_cost_reduction']:.4f}")
    print(f"Average cost reduction: {summary_metrics['avg_cost_reduction']:.4f}")
    print(f"Average percent reduction: {summary_metrics['avg_percent_reduction']:.2f}%")
    print(f"Maximum cost reduction: {summary_metrics['max_cost_reduction']:.4f} ({summary_metrics['max_percent_reduction']:.2f}%)")
    print(f"Best performing file: {summary_metrics['best_file']}")
    print(f"Results saved to: {output_dir}")
    
    # Generate summary plots
    if plot_results and len(file_metrics) > 1:
        plot_summary_results(metrics_df, plots_dir)
    
    return summary_metrics

def plot_summary_results(metrics_df: pd.DataFrame, plots_dir: str) -> None:
    """Generate summary plots for multiple file test results."""
    # Ensure plots_dir is a string
    plots_dir = str(plots_dir) if plots_dir is not None else "plots"
    
    # Sort by cost reduction for better visualization
    metrics_df_sorted = metrics_df.sort_values('cost_reduction', ascending=False)
    
    # Plot 1: Cost reduction by file
    plt.figure(figsize=(12, 6))
    plt.bar(metrics_df_sorted['file_name'], metrics_df_sorted['cost_reduction'])
    plt.title('Cost Reduction by File')
    plt.xlabel('HLO File')
    plt.ylabel('Cost Reduction')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'summary_cost_reduction.png'))
    plt.close()
    
    # Plot 2: Percent reduction by file
    plt.figure(figsize=(12, 6))
    plt.bar(metrics_df_sorted['file_name'], metrics_df_sorted['percent_reduction'])
    plt.title('Percent Cost Reduction by File')
    plt.xlabel('HLO File')
    plt.ylabel('Percent Reduction')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'summary_percent_reduction.png'))
    plt.close()
    
    # Plot 3: Unique actions vs. cost reduction scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(metrics_df['unique_actions'], metrics_df['cost_reduction'])
    plt.title('Cost Reduction vs. Unique Actions Applied')
    plt.xlabel('Number of Unique Actions')
    plt.ylabel('Cost Reduction')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'unique_actions_vs_reduction.png'))
    plt.close()
    
    print("Summary plots saved: summary_cost_reduction.png, summary_percent_reduction.png, unique_actions_vs_reduction.png")

def plot_cost_history(cost_history, output_path):
    """Plot the cost history during testing."""
    plt.figure(figsize=(10, 5))
    plt.plot(cost_history)
    plt.title("Cost Reduction During Testing")
    plt.xlabel("Step")
    plt.ylabel("Cost")
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()
    print(f"Cost history plot saved to {output_path}")

def try_all_passes(env, verbose=True):
    """Try each optimization pass and report which ones reduce cost."""
    original_cost = env.cost_history[0]
    effective_passes = []
    
    for i, pass_name in enumerate(env.available_passes):
        # Reset environment to initial state
        state, _ = env.reset()
        
        # Apply the pass
        next_state, reward, _, _, _ = env.step(i)
        
        new_cost = env.cost_history[-1]
        if new_cost < original_cost:
            reduction = original_cost - new_cost
            effective_passes.append((pass_name, reduction))
            if verbose:
                print(f"Pass '{pass_name}' reduces cost by {reduction:.4f} (from {original_cost:.4f} to {new_cost:.4f})")
    
    # Sort by cost reduction (largest first)
    effective_passes.sort(key=lambda x: x[1], reverse=True)
    
    if verbose:
        print("\nMost effective passes:")
        for pass_name, reduction in effective_passes[:10]:  # Show top 10
            print(f"{pass_name}: reduces cost by {reduction:.4f}")
    
    return effective_passes

def main():
    """Main function to run the test script."""
    parser = argparse.ArgumentParser(description="Test a trained RL agent on XLA optimization")
    parser.add_argument("--model", required=True, help="Path to the saved agent model (.pkl)")
    parser.add_argument("--hlo", help="Path to a single HLO file to test on")
    parser.add_argument("--hlo_dir", help="Directory containing multiple HLO files to test on")
    parser.add_argument("--xla_dir", required=True, help="Path to the XLA directory")
    parser.add_argument("--max_steps", type=int, default=30, help="Maximum number of steps")
    parser.add_argument("--verbose", action="store_true", help="Print detailed progress")
    parser.add_argument("--try_all", action="store_true", help="Try all passes and report effectiveness")
    parser.add_argument("--manual", action="store_true", help="Try a manually curated sequence of passes")
    parser.add_argument("--no_plots", action="store_true", help="Disable plotting")
    parser.add_argument("--output_dir", help="Custom output directory path")
    
    args = parser.parse_args()
    
    # Ensure either a single HLO file or directory is specified
    if not args.hlo and not args.hlo_dir:
        print("Error: Either --hlo (single file) or --hlo_dir (directory) must be specified")
        return
    
    if args.hlo and args.hlo_dir:
        print("Warning: Both --hlo and --hlo_dir specified. Will use --hlo_dir for batch testing.")
    
    # Convert all paths to absolute paths
    if args.model:
        args.model = os.path.abspath(args.model)
    if args.hlo:
        args.hlo = os.path.abspath(args.hlo)
    if args.hlo_dir:
        args.hlo_dir = os.path.abspath(args.hlo_dir)
    if args.xla_dir:
        args.xla_dir = os.path.abspath(args.xla_dir)
    if args.output_dir:
        args.output_dir = os.path.abspath(args.output_dir)
    
    # Load the trained agent data
    with open(args.model, "rb") as f:
        agent_data = pickle.load(f)

    # Create a new agent with the loaded data
    agent = SimpleQLearningAgent(
        action_space_size=len(agent_data["available_passes"]),
        learning_rate=agent_data["learning_rate"],
        discount_factor=agent_data["discount_factor"],
        exploration_rate=agent_data["exploration_rate"]
    )

    # Convert the dict back to defaultdict
    agent.q_table = DefaultDict(lambda: np.zeros(len(agent_data["available_passes"])))
    for k, v in agent_data["q_table"].items():
        agent.q_table[k] = v
    
    print(f"Loaded agent model from {args.model}")
    print(f"Q-table entries: {len(agent.q_table)}")
    
    # Create output directories
    script_dir = os.path.dirname(os.path.abspath(__file__))
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir if args.output_dir else os.path.join(script_dir, "test_output", f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create subdirectories
    csv_dir = os.path.join(output_dir, "csv_data")
    plots_dir = os.path.join(output_dir, "plots")
    analysis_dir = os.path.join(output_dir, "analysis")
    
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(analysis_dir, exist_ok=True)
    
    print(f"Test results will be saved to: {output_dir}")
    
    # Testing on multiple HLO files
    if args.hlo_dir:
        # Get all HLO files in the directory
        hlo_files = glob.glob(os.path.join(args.hlo_dir, "*.hlo"))
        
        if not hlo_files:
            print(f"No HLO files found in directory: {args.hlo_dir}")
            return
        
        print(f"Found {len(hlo_files)} HLO files in {args.hlo_dir}")
        
        # Run tests on all files
        summary_metrics = test_on_multiple_files(
            agent=agent,
            hlo_files=hlo_files,
            xla_dir=args.xla_dir,
            max_steps=args.max_steps,
            verbose=args.verbose,
            use_manual=args.manual,
            plot_results=not args.no_plots,
            output_dir=output_dir
        )
        
        # Save summary results to CSV
        metrics_df = summary_metrics['metrics_df']
        metrics_df.to_csv(os.path.join(csv_dir, 'test_results_summary.csv'), index=False)
        print("Summary metrics saved to test_results_summary.csv")
        
        return summary_metrics
    
    # Testing on a single HLO file
    else:
        # Verify file exists
        if not os.path.exists(args.hlo):
            print(f"Error: HLO file not found: {args.hlo}")
            return
            
        # Create the environment with the test HLO file
        env = XLAOptimizationEnv(
            xla_dir=args.xla_dir,
            initial_hlo_file_path=args.hlo,
            max_sequence_length=args.max_steps,
            verbose=args.verbose
        )
        
        if args.try_all:
            print(f"Testing all passes on {args.hlo}")
            effective_passes = try_all_passes(env, verbose=args.verbose)
            return effective_passes
        
        # Ensure environment and agent use the same passes
        if args.verbose and "available_passes" in agent_data:
            print(f"Checking compatibility between model and environment...")
            assert len(agent_data["available_passes"]) == len(env.available_passes), "Pass count mismatch"
            mismatches = 0
            for i, (p1, p2) in enumerate(zip(agent_data["available_passes"], env.available_passes)):
                if p1 != p2:
                    mismatches += 1
                    if mismatches <= 5:  # Only show first few mismatches
                        print(f"Pass mismatch at index {i}: {p1} vs {p2}")
            if mismatches > 0:
                print(f"Found {mismatches} mismatched passes between model and environment")
        
        # Create file-specific output directory
        file_base_name = os.path.splitext(os.path.basename(args.hlo))[0]
        file_plots_dir = os.path.join(plots_dir, file_base_name)
        file_csv_dir = os.path.join(csv_dir, file_base_name)
        os.makedirs(file_plots_dir, exist_ok=True)
        os.makedirs(file_csv_dir, exist_ok=True)
        
        # Test the agent
        reward, passes, cost_history, metrics = test_agent(
            agent=agent,
            env=env,
            max_steps=args.max_steps,
            verbose=args.verbose,
            use_manual=args.manual
        )
        
        # Save metrics to CSV
        metrics["file_name"] = os.path.basename(args.hlo)
        metrics["file_path"] = args.hlo
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(os.path.join(file_csv_dir, f"metrics_{file_base_name}.csv"), index=False)
        
        # Save pass sequence to CSV
        passes_df = pd.DataFrame({
            'step': range(len(passes)),
            'pass_name': passes
        })
        passes_df.to_csv(os.path.join(file_csv_dir, f"passes_{file_base_name}.csv"), index=False)
        
        # Save cost history to CSV
        cost_df = pd.DataFrame({
            'step': range(len(cost_history)),
            'cost': cost_history,
            'reduction': [cost_history[0] - c for c in cost_history],
            'percent_reduction': [(cost_history[0] - c) / cost_history[0] * 100 if cost_history[0] > 0 else 0 
                                  for c in cost_history]
        })
        cost_df.to_csv(os.path.join(file_csv_dir, f"cost_history_{file_base_name}.csv"), index=False)
        
        # Plot the cost history
        if not args.no_plots:
            # Cost reduction plot
            plot_cost_history(
                cost_history=cost_history,
                output_path=os.path.join(file_plots_dir, f"cost_reduction_{file_base_name}.png")
            )
            
            # Percent reduction plot
            plt.figure(figsize=(10, 5))
            plt.plot(cost_df['step'], cost_df['percent_reduction'], 'g-')
            plt.title(f"Percent Cost Reduction During Testing - {file_base_name}")
            plt.xlabel("Step")
            plt.ylabel("Cost Reduction (%)")
            plt.grid(True)
            plt.savefig(os.path.join(file_plots_dir, f"percent_reduction_{file_base_name}.png"))
            plt.close()
        
        print(f"Results saved to: {output_dir}")
        return reward, passes, metrics

if __name__ == "__main__":
    main()
    # python test_agent.py --model /Users/rayaanfaruqi/Documents/CS521/Final_Project/rl_xla_opt_cs521/reinforcement_learning/models/trained_agent.pkl --hlo /Users/rayaanfaruqi/Documents/CS521/Final_Project/rl_xla_opt_cs521/jax_hlo/hlo_data/conv_relu.hlo --xla_dir /Users/rayaanfaruqi/Documents/CS521/Final_Project/xla --verbose