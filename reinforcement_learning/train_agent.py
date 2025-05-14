import numpy as np
import matplotlib.pyplot as plt
from xla_opt_env import XLAOptimizationEnv
from simple_agent import SimpleQLearningAgent
from XLA_interface import XLAInterface
import os
import glob

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


def train_on_multiple_files(
        hlo_files,
        xla_dir,
        episodes_per_file = 100,
        max_steps_per_episode = 30,
        print_interval = 10,
        verbose = True
):
    """
    Train the agent on multiple HLO files sequentially

    Args:
        hlo_files: List of HLO file paths to train on
        xla_dir: Path to XLA directory
        episodes_per_file: Number of episodes to train on each HLO file
        max_steps_per_episode: Maximum optimization passes per episode
        print_interval: How often to print progress
        verbose: Whether to print detailed progress

    Returns:
        Agent and statistics for each file.
    """

    results = {}

    # Initialize environment instance
    env = XLAOptimizationEnv(
        xla_dir=xla_dir,
        initial_hlo_file_path=hlo_files[0],
        max_sequence_length=max_steps_per_episode,
        verbose=verbose
    )

    # Get available passes
    available_passes = env.available_passes

    # create new agent:
    agent = SimpleQLearningAgent(
        action_space_size=len(available_passes),
        learning_rate=0.1,
        discount_factor=0.99,
        exploration_rate=1.0,
        exploration_decay=0.995, # quite a large exploration decay?
        min_exploration_rate=0.01
    )

    # train on each file
    for i, hlo_file in enumerate(hlo_files):
        if verbose:
            print(f"\n\n{'='*50}")
            print(f"Training on {hlo_file} [{i+1}/{len(hlo_files)}]")
            print(f"{'='*50}\n")
    
        # Only set the new HLO file if it's not the first one (already set in initialization):
        if i > 0:
            env.set_base_hlo_file(hlo_file_path=hlo_file)
        
        # train agent on this file
        trained_agent, stats = train_agent(
            env=env,
            agent=agent,
            num_episodes=episodes_per_file,
            max_steps_per_episode=max_steps_per_episode,
            print_interval=print_interval
        )

        file_basename = os.path.basename(hlo_file)
        if verbose:
            # plot results for this file:
            best_sequence = plot_training_results(stats=stats, available_passes=available_passes, hlo_file_name=file_basename)
        
        # store results for this file
        results[file_basename] = {
            'best_sequence': best_sequence,
            'best_reward': stats['best_reward'],
            'episode_rewards': stats['episode_rewards']
        }
        # clear out the optimized directory between transitions to other files
        clear_optimized_directory(env.xla_interface.optimized_dir)

    return trained_agent, results, available_passes

def clear_optimized_directory(dir_path):
    """Clear all files in the optimized_hlo directory"""
    for file in os.listdir(dir_path):
        file_path = os.path.join(dir_path, file)
        if os.path.isfile(file_path):
            os.remove(file_path)
    print(f"Cleared contents of {dir_path}")

def plot_training_results(stats, available_passes, hlo_file_name=None):
    """
    Plot the training results.

    Args:
        stats: Statistics from agent training
        hlo_file_name: Name of the HLO file trained on
    """
    # Plot episode rewards
    plt.figure(figsize=(10, 5))
    plt.plot(stats["episode_rewards"])
    
    title = "Episode Rewards During Training"
    if hlo_file_name:
        title += f" - {hlo_file_name}"
    
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid(True)
    
    output_file = "training_rewards"
    if hlo_file_name:
        output_file += f"_{os.path.basename(hlo_file_name).split('.')[0]}"
    output_file += ".png"
    
    plt.savefig(output_file)
    plt.close()

    # Print best sequence
    print("\nBest Pass Sequence Found:")
    best_sequence_passes = [available_passes[i] for i in stats["best_sequence"]]
    print(best_sequence_passes)
    print(f"Best Total Reward: {stats['best_reward']:.4f}")
    
    return best_sequence_passes

def verify_hlo_files(hlo_files, xla_interface):
    """
    Check each HLO file to make sure it can be parsed properly.
    Returns a list of valid HLO files.
    """
    valid_files = []
    print()
    
    for hlo_file in hlo_files:
        try:
            # Try to extract features
            features = xla_interface.extract_features(hlo_file)
            valid_files.append(hlo_file)
            print(f"✅ Valid HLO file: {os.path.basename(hlo_file)}")
        except Exception as e:
            print(f"❌ Invalid HLO file: {os.path.basename(hlo_file)}, Error: {str(e)}")
    
    return valid_files


def main():
    """
    Set up and run training process on multiple HLO files using a single environment
    """
    # XLA directory
    xla_dir = "/Users/rayaanfaruqi/Documents/CS521/Final_Project/xla"
    
    # Find all HLO files to train on
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    
    # Find both .hlo and .txt files in the data directory
    hlo_data_dir = os.path.join(project_dir, "jax_hlo", "hlo_data")
    hlo_files = glob.glob(os.path.join(hlo_data_dir, "*.hlo"))
    hlo_files.extend(glob.glob(os.path.join(hlo_data_dir, "*.txt")))
    
    if not hlo_files:
        print(f"No HLO files found in {hlo_data_dir}")
        return
    
    print(f"Found {len(hlo_files)} HLO files: {[os.path.basename(f) for f in hlo_files]}")
    
    # Initialize XLA interface for validation
    xla_interface = XLAInterface(xla_dir=xla_dir, verbose=True)
    
    # Verify files before training
    valid_hlo_files = verify_hlo_files(hlo_files, xla_interface)
    
    if not valid_hlo_files:
        print("No valid HLO files found. Cannot proceed with training.")
        return
    
    print(f"\nProceeding with {len(valid_hlo_files)} valid HLO files")
    
    # Train on valid HLO files using a single environment
    trained_agent, results, available_passes = train_on_multiple_files(
        hlo_files=valid_hlo_files,
        xla_dir=xla_dir,
        episodes_per_file=7,
        max_steps_per_episode=30,
        print_interval=10,
        verbose=True
    )
    
    # Print summary of results
    print("\n\nTraining Summary Across All Files:")
    print("=" * 50)
    for file_name, file_results in results.items():
        print(f"\nFile: {file_name}")
        print(f"Best Pass Sequence: {file_results['best_sequence']}")
        print(f"Best Reward: {file_results['best_reward']:.4f}")


if __name__ == "__main__":
    main()
