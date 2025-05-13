# reinforcement_learning/test_env.py

import numpy as np
from xla_opt_env import XLAOptimizationEnv

def dummy_test_environment():
    """
    Test the XLA optimization environment with random actions.
    """
    # Create a dummy set of HLO features (random values for now)
    # In a real implementation, these would come from the HLO graph
    feature_dim = 20
    initial_features = np.random.normal(0, 1, (feature_dim,))
    
    # Define a list of dummy optimization passes
    available_passes = [
        "algebraic-simplifier",
        "cse",
        "fusion",
        "hlo-dce",
        "hlo-cse",
        "inliner",
        "reshape-mover",
        "transpose-folding",
        "tree-reduction-rewriter",
        "while-loop-simplifier"
    ]
    
    # Create the environment
    env = XLAOptimizationEnv(
        hlo_features=initial_features,
        available_passes=available_passes,
        max_sequence_length=5,
        verbose=True
    )
    
    # Run a random agent for one episode
    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    print(f"Initial cost: {env._calculate_cost(obs)}")
    
    done = False
    total_reward = 0
    
    while not done:
        # Choose a random action
        action = env.action_space.sample()
        
        # Take a step in the environment
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        total_reward += reward
        print(f"Action: {available_passes[action]}, Reward: {reward:.4f}")
    
    print("\nEpisode summary:")
    print(f"Total reward: {total_reward:.4f}")
    print(f"Applied passes: {info['applied_passes']}")
    print(f"Initial cost: {info['cost_history'][0]:.4f}")
    print(f"Final cost: {info['cost_history'][-1]:.4f}")
    print(f"Cumulative improvement: {info['cumulative_improvement']:.4f}")

if __name__ == "__main__":
    dummy_test_environment()