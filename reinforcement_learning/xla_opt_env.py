import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional, Any

class XLAOptimizationEnv(gym.Env):

    def __init__(
            self,
            hlo_features: np.ndarray,
            available_passes: List[str],
            max_sequence_length: int = 10,
            verbose: bool = False
    ):
        
        super().__init__() # what does this do?

        # store initial HLO features and available passes:
        self.initial_features = hlo_features
        self.available_passes = available_passes
        self.max_sequence_length = max_sequence_length
        self.verbose = verbose
        
        # Define action and observation spaces
        # actions/passes implicitly mapped to int here (in spaces.Discrete())
        # hence, action 0 will correspond to first action available_passes[0]
        self.action_space = spaces.Discrete(len(available_passes))
        self.observation_space = spaces.Box( # defines space in which we hold or observe features
            low = 0,
            high= np.inf,
            shape = hlo_features.shape,
            dtype= np.float32
        )

        # if we change this to a graph representation...
        # self.observation_space = spaces.Graph(...)

        # initialize state variables
        self.reset()

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        '''
        Reset env to initial state.
        
        Returns:
            observation: Initial observation (HLO Features)
            info: additional info in dictionary
        '''

        super().reset(seed=seed)

        # reset state vars
        self.current_features = self.initial_features.copy()
        self.current_step = 0
        self.applied_passes = []
        self.cost_history = [self._calculate_cost(self.current_features)]


        return self.current_features, {}
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Apply selected optimization pass, return new state
        
        Args:
            action: index of the optimization pass to apply
        Returns:
            observation: new observation after applying the pass
            reward: reward for this step
            terminated: whether episode has ended
            truncated: whether episode was truncated
            info: additional info
        """

        # Get selected pass
        selected_pass = self.available_passes[action] # passes implicitly mapped to int in init
        self.applied_passes.append(selected_pass)

        if self.verbose:
            print(f"Step {self.current_step}: Applying pass {selected_pass}")

        # apply selected pass # TODO: Currently this is a no-op; needs to be implemented later

        self._apply_pass(selected_pass)
        self.current_step += 1

        # calculate current cost:
        current_cost = self._calculate_cost(self.current_features)
        self.cost_history.append(current_cost)

        # calculate reward (cost improvement)
        prev_cost = self.cost_history[-2]
        reward = prev_cost - current_cost

        done = (self.current_step >= self.max_sequence_length)

        return self.current_features, reward, done, False, {
            "applied_passes" : self.applied_passes,
            "cost_history" : self.cost_history,
            "cumulative_improvement" : self.cost_history[0] - current_cost
        }
    
    def _apply_pass(self, pass_name : str) -> None:
        """
        Apply specificed optimization pass to current HLO graph

        # TODO: currently a no-op placeholder. Need to integrate with XLA
        """
        # print("[FAKE] Applied Pass")
    
    def _calculate_cost(self, features: np.ndarray) -> float: 
        """
        Calculate some cost metric from the HLO features
        # TODO: get some kind of cost metric.
        """

        return np.sum(features)
