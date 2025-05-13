import numpy as np
import matplotlib.pyplot as plt
from xla_opt_env import XLAOptimizationEnv
from simple_agent import SimpleQLearningAgent

def train_agent(
        env,
        agent,
        num_episodes=1000,
        max_steps_per_episode=10,
        print_interval=100
):
    """
    Args:
        env: XLA optimization environment
        agent: the RL agent we are training
        num_episodes: number of episodes to train for
        max_steps_per_episode: maximum steps per episode
        print_interval: how often to print our progress
    """

    for episode in range(num_episodes):
         # Reset the environment
        state, info = env.reset()
        done = False
        episode_reward = 0
        actions_taken = []
        
        # Run the episode
        step = 0
        while not done and step < max_steps_per_episode:
            # Select an action
            action = agent.select_action(state)
            actions_taken.append(action)
            
            # Take the action
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Update the agent
            agent.update(state, action, reward, next_state, done)
            
            # Update variables for next iteration
            state = next_state
            episode_reward += reward
            step += 1
        
        # End of episode processing
        agent.end_episode(episode_reward, actions_taken)
        
        # Print progress
        if (episode + 1) % print_interval == 0:
            mean_reward = np.mean(agent.episode_rewards[-print_interval:])
            print(f"Episode {episode + 1}/{num_episodes} | "
                  f"Mean Reward: {mean_reward:.4f} | "
                  f"Exploration Rate: {agent.exploration_rate:.4f}")
    
    # Return the trained agent and its statistics
    return agent, agent.get_statistics()


def plot_training_results(stats):
    """
    Plot the training results.
    
    Args:
        stats: Statistics from agent training
    """
    # Plot episode rewards
    plt.figure(figsize=(10, 5))
    plt.plot(stats["episode_rewards"])
    plt.title("Episode Rewards During Training")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid(True)
    plt.savefig("training_rewards.png")
    plt.close()
    
    # Print best sequence
    print("\nBest Pass Sequence Found:")
    print(stats["best_sequence"])
    print(f"Best Total Reward: {stats['best_reward']:.4f}")


def main():
    """
    Set up and run training process
    """

    # create dummy set of HLO features (random values for now)
    feature_dim = 20
    initial_features = np.random.normal(0, 1, (feature_dim,))

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

    # create env
    env = XLAOptimizationEnv(
        hlo_features=initial_features,
        available_passes=available_passes,
        max_sequence_length=10,
        verbose=False
    )

    # create agent and begin training it
    agent = SimpleQLearningAgent(
        action_space_size=len(available_passes),
        learning_rate=0.1,
        discount_factor=0.99,
        exploration_rate=1.0,
        exploration_decay=0.995,
        min_exploration_rate=0.01
    )

    trained_agent, stats = train_agent(
        env=env,
        agent=agent,
        num_episodes=1000,
        max_steps_per_episode=10,
        print_interval=100
    )

    plot_training_results(stats)

if __name__ == "__main__":
    main()