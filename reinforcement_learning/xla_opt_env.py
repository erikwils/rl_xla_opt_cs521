import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional, Any
from XLA_interface import XLAInterface

class XLAOptimizationEnv(gym.Env):

    def __init__(
            self,
            xla_dir: str,
            initial_hlo_file_path: str,
            max_sequence_length: int = 30,
            no_improvement_threshold: int = 5,
            verbose: bool = False
    ):

        super().__init__() # what does this do?

        # Initialize XLA interface
        print("Initializing XLA Interface...")
        self.xla_interface = XLAInterface(xla_dir=xla_dir, verbose=verbose)

        # store params
        self.max_sequence_length = max_sequence_length
        self.no_improvement_threshold = no_improvement_threshold
        self.verbose = verbose

        self.available_passes = self.xla_interface.get_available_passes()
        self.cost_history = []
        self.long_term_cost_history = []
        self.long_term_pass_history = []
        self.applied_passes = []

        # set up base HLO file (note this must come AFTER the other initializations as we call reset() inside here)
        self.set_base_hlo_file(initial_hlo_file_path)

        if verbose:
            print(f"\nAvailable Passes: \n{self.available_passes}\n")

        # Define action and observation spaces
        # actions/passes implicitly mapped to int here (in spaces.Discrete())
        # hence, action 0 will correspond to first action available_passes[0]
        self.action_space = spaces.Discrete(len(self.available_passes))

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

    def set_base_hlo_file(self, hlo_file_path : str) -> None:
        """
        Set a new base HLO file to optimize.

        Args:
            hlo_file_path: Features of the new HLO file to optimize
        """
        # Extract features from the HLO file:
        self.base_hlo_file_path = hlo_file_path
        self.base_features = self.xla_interface.extract_features(hlo_file=hlo_file_path)
        
        # initialize current_hlo_file and features
        self.current_hlo_file = hlo_file_path
        self.current_features = self.base_features.copy()

        # set initial cost
        self.cost_history = [self._calculate_cost(self.base_features)]

        self.reset()
    
    def reset(self, seed=None, options=None) -> Tuple[Dict[str, Any], Dict]:
        '''
        Reset env to initial state.

        Returns:
            observation: Initial observation (HLO Features as graph)
            info: additional info in dictionary
        '''

        super().reset(seed=seed)

        # append episode cost to our long term cost history:
        episode_cost_history = self.cost_history
        self.long_term_cost_history.append(episode_cost_history)

        # track our pass history as well:
        episode_pass_history = self.applied_passes
        self.long_term_pass_history.append(episode_pass_history)

        # reset state vars
        self.current_features = self.base_features.copy()
        self.current_step = 0
        self.applied_passes = []
        self.cost_history = [self._calculate_cost(self.base_features)]

        graph_observation = self._features_to_graph(self.current_features)
        return graph_observation, {}

    def _features_to_graph(self, features) -> Dict[str, Any]:
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

    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, bool, Dict]:
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
        
        # if len(self.cost_history) > self.no_improvement_threshold:
        #     # check if there's been no improvement for several steps
        #     recent_costs = self.cost_history[-self.no_improvement_threshold:]
        #     no_improvement = all(abs(cost - recent_costs[0]) < 1e-6 for cost in recent_costs)
        #     if no_improvement:
        #         done = True
        #         if self.verbose:
        #             print(f"Terminating early: No improvement for {self.no_improvement_threshold} steps")

        # convert features to graph observation
        graph_observation = self._features_to_graph(self.current_features)

        return graph_observation, reward, done, False, {
            "applied_passes" : self.applied_passes,
            "cost_history" : self.cost_history,
            "cumulative_improvement" : self.cost_history[0] - current_cost
        }

    def _apply_pass(self, pass_name : str) -> None:
        """
        Apply specificed optimization pass to current HLO graph.
        Uses XLA Interface to apply a compilier pass to the current HLO file,
        and then updates the current features with the optimized version.

        Args:
            pass_name: name of the optimization pass to apply

        """
        if self.verbose:
            print(f"Applying pass {pass_name} to file {self.current_hlo_file}")
        
        # Apply using XLA interface:
        success, optimized_file_path = self.xla_interface.apply_pass(
            hlo_file=self.current_hlo_file,
            pass_name=pass_name
        )

        if not success:
            if self.verbose:
                print("Failed to apply pass! Keeping current features.")
            return
        
        # update current HLO file to point to optimized file, extract features
        self.current_hlo_file = optimized_file_path
        self.current_features = self.xla_interface.extract_features(hlo_file=optimized_file_path) # type: ignore

        if self.verbose:
            print(f"Successfully applied pass {pass_name}. Updated features.")

    def _calculate_cost(self, features: Dict[str, Any]) -> float:
        """
        Calculate some cost metric from the HLO features dictionary.

        This function extracts metrics from the features dictionary to determine the "cost" of the current HLO module.
        Lower cost is better.

        Args:
            features: Dictionary of HLO features from extract_features()

        Returns:
            A cost value (lower is better)

         # TODO: Possibly re-evaluate cost metric?
        """
        # Combine various metrics for cost: 1. Total number of ops, 2. Memory footprint, 3. Graph complexity
        excluded_keys = [['source_file', 'graph_nodes', 'graph_edge_links', 'total_bytes',
                    'input_bytes', 'output_bytes', 'elemwise_ratio', 'mixed_precision']]

        # print(f"\n [FEATURES]: {features}\n\n")

        total_ops = 0
        for op_name, value in features.items():
            if op_name not in excluded_keys:
                # make sure we're only summing numeric values
                if isinstance(value, (int, float)):
                    total_ops += value

        memory_cost = features.get('total_bytes', 0) / 1000.0
        graph_nodes = features.get('graph_nodes', [])
        graph_complexity = len(graph_nodes) * 10 # weight by importance

        # combine metrics:
        cost = total_ops + memory_cost + graph_complexity

        return cost

