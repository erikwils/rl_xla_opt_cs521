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
        
        # original box space implementation
        '''
        self.observation_space = spaces.Box( # defines space in which we hold or observe features
            low = 0,
            high= np.inf,
            shape = hlo_features.shape,
            dtype= np.float32
        )
        '''

        # if we change this to a graph representation...
        # https://gymnasium.farama.org/api/spaces/composite/#gymnasium.spaces.Graph
        
        num_nodes = len(hlo_features["graph_nodes"]) if "graph_nodes" in hlo_features else 0
        edge_links = hlo_features["graph_edge_links"] if "graph_edge_links" in hlo_features else []

        node_feature_dim = 5
        edge_feature_dim = 1

        node_space = spaces.Box(
            low= -np.inf,
            high= np.inf,
            shape= (node_feature_dim,),
            dtype= np.float32
        )

        edge_space = spaces.Box(
            low= 0,
            high= 1,
            shape= (edge_feature_dim,),
            dtype= np.float32
        )

        self.observation_space = spaces.Graph(
            node_space=node_space,
            edge_space=edge_space,
        )

        # initialize state variables
        self.reset()

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        '''
        Reset env to initial state.
        
        Returns:
            observation: Initial observation (HLO Features as graph)
            info: additional info in dictionary
        '''

        super().reset(seed=seed)

        # reset state vars
        self.current_features = self.initial_features.copy()
        self.current_step = 0
        self.applied_passes = []
        self.cost_history = [self._calculate_cost(self.current_features)]


        return self.current_features, {}
    
    
    def _features_to_graph(self, features):
        """
        Convert features to graph observation format

        Args:
            features: HLO feature vector with graph_nodes and graph_edge_links

        Returns:
            graph observation dictionary
        """

        # extract graph components
        nodes = np.array(features["graph_nodes"], dtype=np.float32)

        # create edge indices from edge_links
        edge_links = features["graph_edge_links"]
        edge_indices = np.array(edge_links, dtype=np.int64)

        # create simple edge features (all 1s for connectivity)
        edge_features = np.ones((len(edge_links), 1), dtype=np.float32)

        return_dict = {
            "nodes": nodes,
            "edges": edge_features,
            "edge_links": edge_indices
        }

        return return_dict
    
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

        # convert features to graph observation
        graph_observation = self._features_to_graph(self.current_features)

        return graph_observation, reward, done, False, {
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
