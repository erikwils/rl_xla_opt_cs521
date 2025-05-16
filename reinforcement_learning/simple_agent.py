import numpy as np
from collections import defaultdict
from typing import Tuple, List, Dict, Any

class SimpleQLearningAgent:
    """
    Simple Q-learning agent for XLA optimization environment

    Agent uses tabular representation (dictionary) to store Q-Values for state-action pairs,
    which is a simple approach for initial testing
    """

    def __init__(
            self,
            action_space_size: int,
            learning_rate: float = 0.1,
            discount_factor: float = 0.99,
            exploration_rate: float = 1.0,
            exploration_decay: float = 0.995,
            min_exploration_rate: float = 0.01
    ):
        """
        Initialize the Q-learning agent.

        Args:
            action_space_size: Number of available actions (optimization passes)
            learning_rate: learning rate (alpha) for Q-val updates
            discount_factor: gamme for future rewards
            exploration_rate: initial exploration rate (epsilon)
            exploration_decay: factor to decay exploration rate each episode
            min_exploration_rate: minimum exploration rate
        """

        self.action_space_size = action_space_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = min_exploration_rate

        # Initialize Q-table as dict, default vals 0
        # for continuous states -- use simple discretization strategy
        self.q_table = defaultdict(lambda: np.zeros(action_space_size))

        # track training progress
        self.episode_rewards = []
        self.best_sequence = []
        self.best_reward = float('-inf')

    def discretize_state(self, state: Dict[str, Any]) -> Tuple:
        """
        Discretize a graph state for the Q-table

        Args:
            state: graph observation with nodes, edges, and edge_links

        Returns:
            a tuple representing the discretized state
        """
        # Extract features from the graph observation
        nodes = state["nodes"]
        edges = state.get("edges", [])
        edge_links = state.get("edge_links", [])
        
        features = []
        
        # Graph size features (more precise than before)
        node_count = len(nodes)
        edge_count = len(edges)
        features.append(node_count)  # Use exact count instead of rounding
        features.append(edge_count)  # Add edge count as a feature
        
        # Graph density/connectivity
        if node_count > 1:
            density = edge_count / (node_count * (node_count - 1) / 2)
            features.append(np.round(density * 20) / 20)  # Discretize with 0.05 intervals
        else:
            features.append(0.0)
        
        # Node features statistics
        if node_count > 0:
            node_array = np.array(nodes)
            
            # For each feature dimension, compute multiple statistics
            for dim in range(node_array.shape[1]):
                col_data = node_array[:, dim]
                
                # More granular statistics with finer discretization
                mean_val = np.mean(col_data)
                median_val = np.median(col_data)
                max_val = np.max(col_data)
                min_val = np.min(col_data)
                std_val = np.std(col_data)
                p25 = np.percentile(col_data, 25)
                p75 = np.percentile(col_data, 75)
                
                # Use different precision for different ranges
                if max_val > 1000:
                    # For large values use coarser discretization
                    features.append(np.round(mean_val / 50) * 50)
                    features.append(np.round(median_val / 50) * 50)
                    features.append(np.round(max_val / 100) * 100)
                    features.append(np.round(min_val / 20) * 20)
                else:
                    # For smaller values use finer discretization
                    features.append(np.round(mean_val * 10) / 10)
                    features.append(np.round(median_val * 10) / 10)
                    features.append(np.round(max_val * 5) / 5)
                    features.append(np.round(min_val * 5) / 5)
                
                # Add standard deviation and quartiles
                features.append(np.round(std_val * 5) / 5)
                features.append(np.round(p25 * 5) / 5)
                features.append(np.round(p75 * 5) / 5)
                
                # Add histogram features (capture distribution better)
                if max_val > min_val:
                    hist_bins = 8  # More bins than before
                    histogram, _ = np.histogram(col_data, bins=hist_bins, range=(min_val, max_val))
                    # Normalize and discretize histogram bins
                    if np.sum(histogram) > 0:
                        norm_hist = histogram / np.sum(histogram)
                        for bin_val in norm_hist:
                            features.append(np.round(bin_val * 20) / 20)
                    else:
                        features.extend([0.0] * hist_bins)
        else:
            # Handle empty graph case
            features.extend([0.0] * 20)  # Pad with zeros
        
        # Edge feature statistics if available
        if len(edges) > 0:
            edge_array = np.array(edges)
            features.append(np.mean(edge_array))
            features.append(np.std(edge_array))
        else:
            features.extend([0.0, 0.0])
        
        # Graph structural hash - use edge patterns
        if len(edge_links) > 0:
            # Count nodes by their connectivity (in-degree and out-degree)
            # This helps distinguish different graph structures
            in_degrees = np.zeros(node_count)
            out_degrees = np.zeros(node_count)
            
            for edge in edge_links:
                if len(edge) >= 2:
                    src, dst = edge[0], edge[1]
                    if 0 <= src < node_count and 0 <= dst < node_count:
                        out_degrees[src] += 1
                        in_degrees[dst] += 1
            
            # Discretize degree distributions
            in_degree_hist, _ = np.histogram(in_degrees, bins=5)
            out_degree_hist, _ = np.histogram(out_degrees, bins=5)
            
            # Add normalized and discretized degree histograms
            if np.sum(in_degree_hist) > 0:
                norm_in_hist = in_degree_hist / np.sum(in_degree_hist)
                for val in norm_in_hist:
                    features.append(np.round(val * 10) / 10)
            else:
                features.extend([0.0] * 5)
                
            if np.sum(out_degree_hist) > 0:
                norm_out_hist = out_degree_hist / np.sum(out_degree_hist)
                for val in norm_out_hist:
                    features.append(np.round(val * 10) / 10)
            else:
                features.extend([0.0] * 5)
                
            # Add count of nodes with specific connectivity patterns
            # These act as graph motif detectors
            isolated_nodes = np.sum((in_degrees == 0) & (out_degrees == 0))
            source_nodes = np.sum((in_degrees == 0) & (out_degrees > 0))
            sink_nodes = np.sum((in_degrees > 0) & (out_degrees == 0))
            transfer_nodes = np.sum((in_degrees == 1) & (out_degrees == 1))
            computation_nodes = np.sum((in_degrees > 1) & (out_degrees >= 1))
            
            features.append(isolated_nodes / node_count if node_count > 0 else 0)
            features.append(source_nodes / node_count if node_count > 0 else 0)
            features.append(sink_nodes / node_count if node_count > 0 else 0)
            features.append(transfer_nodes / node_count if node_count > 0 else 0)
            features.append(computation_nodes / node_count if node_count > 0 else 0)
        
        # Include a graph signature that captures the rough graph structure
        if node_count > 0 and node_array.shape[1] > 0:
            # Use the top 3 most common node feature values as a signature
            for dim in range(min(3, node_array.shape[1])):
                if dim < node_array.shape[1]:
                    col_data = node_array[:, dim]
                    unique, counts = np.unique(np.round(col_data), return_counts=True)
                    if len(unique) > 0:
                        # Get top 3 most frequent values
                        sorted_indices = np.argsort(-counts)
                        for i in range(min(3, len(sorted_indices))):
                            features.append(unique[sorted_indices[i]])
        
        # Convert to tuple and return
        # Make all values hashable (convert numpy types to Python types)
        hashable_features = tuple(float(f) for f in features)
        return hashable_features

    def select_action(self, state: Dict[str, Any], training: bool = True) -> int:
        """
        select action with epsilon-greedy policy

        Args:
            state: current state
            training: if training, use exploration. If not, don't and evaluate

        Returns:
            Selected action index
        """

        discretized_state = self.discretize_state(state)

        # for exploration, just choose an action at random
        if training and np.random.random() < self.exploration_rate:
            return np.random.randint(self.action_space_size)

        # exploitation: choose best action according to q-vals
        q_values = self.q_table[discretized_state]
        return int(np.argmax(q_values))

    def update(self, state: Dict[str, Any], action: int, reward: float,
               next_state: Dict[str, Any], done: bool) -> None:
        """
        Update q-value in the q-table for state-action pair

        Args:
            state: the current state
            action: the selected action
            reward: the received reward
            next_state: the next state
            done: whether the episode is done
        """

        discretized_state = self.discretize_state(state)
        discretized_next_state = self.discretize_state(next_state)

        # Current Q-value
        current_q = self.q_table[discretized_state][action]

        # Max q-val for next state
        max_next_q = np.max(self.q_table[discretized_next_state]) if not done else 0

        # Compute target Q-value using Q-learning update rule
        target_q = reward + self.discount_factor * max_next_q

        # Update Q-value
        self.q_table[discretized_state][action] += self.learning_rate * (target_q - current_q)

    def end_episode(self, episode_reward: float, actions_taken: List[int]) -> None:
        """
        process end of an episode

        Args:
            episode_reward: total reward for the episode
            actions_taken: list of actions taken during the episode
        """

        # track episode reward
        self.episode_rewards.append(episode_reward)

        # track best sequence
        if episode_reward > self.best_reward:
            self.best_reward = episode_reward
            self.best_sequence = actions_taken.copy()

        # decay exploration rate
        self.exploration_rate = max(
            self.min_exploration_rate,
            self.exploration_rate * self.exploration_decay
        )

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get training statistics.

        Returns:
            A dictionary with training statistics
        """

        return {
            "episode_rewards": self.episode_rewards,
            "best_reward": self.best_reward,
            "best_sequence": self.best_sequence,
            "current_exploration_rate": self.exploration_rate
        }
