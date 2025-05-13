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
    
    def discretize_state(self, state: np.ndarray) -> Tuple:
        """
        Discretize a continuous state for the Q-table

        # TODO: Change function to fn approximation instead??

        Args: 
            state: continuous state vector
        
        Returns: 
            a tuple representing the discretized state
        """

        # discretizes to nearest 0.5. May need something more sophisticated like fn approx.
        discretized = tuple(np.round(state * 2) / 2)
        return discretized
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
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

    def update(self, state: np.ndarray, action: int, reward: float,
               next_state: np.ndarray, done: bool) -> None:
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