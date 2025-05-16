import numpy as np
import matplotlib.pyplot as plt
from xla_opt_env import XLAOptimizationEnv
from simple_agent import SimpleQLearningAgent
from XLA_interface import XLAInterface
import os
import pickle
import glob
import pandas as pd

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
    # Track detailed metrics for each episode
    episode_metrics = []

    for episode in range(num_episodes):
        # Reset the environment
        state, info = env.reset()
        done = False
        episode_reward = 0
        actions_taken = []
        step_rewards = []
        step_costs = []
        unique_actions = set()
        step_count = 0
        initial_cost = env.cost_history[0]

        # Run the episode
        step = 0
        while not done and step < max_steps_per_episode:
            # Select an action
            action = agent.select_action(state)
            actions_taken.append(action)
            unique_actions.add(action)

            # Take the action
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Track detailed metrics
            step_rewards.append(reward)
            step_costs.append(info["cost_history"][-1])
            step_count += 1

            # Update the agent
            agent.update(state, action, reward, next_state, done)

            # Update variables for next iteration
            state = next_state
            episode_reward += reward
            step += 1

        # End of episode processing
        agent.end_episode(episode_reward, actions_taken)
        
        # Record episode metrics
        final_cost = env.cost_history[-1]
        cost_reduction = initial_cost - final_cost
        percent_reduction = (cost_reduction / initial_cost) * 100 if initial_cost > 0 else 0
        
        episode_data = {
            "episode": episode + 1,
            "total_reward": episode_reward,
            "initial_cost": initial_cost,
            "final_cost": final_cost,
            "cost_reduction": cost_reduction,
            "percent_reduction": percent_reduction,
            "exploration_rate": agent.exploration_rate,
            "unique_actions": len(unique_actions),
            "steps_taken": step_count,
            "action_sequence": ",".join(str(a) for a in actions_taken)
        }
        episode_metrics.append(episode_data)

        # Print progress
        if (episode + 1) % print_interval == 0:
            mean_reward = np.mean(agent.episode_rewards[-print_interval:])
            print(f"Episode {episode + 1}/{num_episodes} | "
                  f"Mean Reward: {mean_reward:.4f} | "
                  f"Exploration Rate: {agent.exploration_rate:.4f}")

    # Return the trained agent, its statistics, and detailed metrics
    return agent, agent.get_statistics(), episode_metrics


def save_training_data_to_csv(episode_metrics, hlo_file_name, available_passes):
    """
    Save training metrics to CSV file for analysis.
    
    Args:
        episode_metrics: List of dictionaries containing episode data
        hlo_file_name: Name of the HLO file trained on
        available_passes: List of available compiler passes
    """
    # Create a DataFrame from the episode metrics
    df = pd.DataFrame(episode_metrics)
    
    # Prepare filename
    base_name = os.path.basename(hlo_file_name).split('.')[0] if hlo_file_name else "training"
    csv_file = f"training_metrics_{base_name}.csv"
    
    # Save to CSV
    df.to_csv(csv_file, index=False)
    print(f"Training metrics saved to {csv_file}")
    
    # Create a pass name mapping DataFrame
    pass_map = pd.DataFrame({
        'pass_index': list(range(len(available_passes))),
        'pass_name': available_passes
    })
    pass_map_file = f"pass_mapping_{base_name}.csv"
    pass_map.to_csv(pass_map_file, index=False)
    print(f"Pass mapping saved to {pass_map_file}")
    
    # For the most successful episodes, create detailed sequence analysis
    if len(df) > 0:
        # Get the top 5 episodes by reward (or all if less than 5)
        top_n = min(5, len(df))
        top_episodes = df.nlargest(top_n, 'total_reward')
        
        # Create a detailed breakdown of pass sequences for top episodes
        top_sequences = []
        for _, row in top_episodes.iterrows():
            action_indices = [int(a) for a in row['action_sequence'].split(',') if a]
            action_names = [available_passes[idx] for idx in action_indices]
            
            sequence_data = {
                'episode': row['episode'],
                'total_reward': row['total_reward'],
                'percent_reduction': row['percent_reduction'],
                'pass_sequence': ', '.join(action_names)
            }
            top_sequences.append(sequence_data)
        
        # Save top sequences to a separate CSV
        top_seq_df = pd.DataFrame(top_sequences)
        top_seq_file = f"top_sequences_{base_name}.csv"
        top_seq_df.to_csv(top_seq_file, index=False)
        print(f"Top sequences saved to {top_seq_file}")
    
    return csv_file


def train_on_multiple_files(
        hlo_files,
        xla_dir,
        episodes_per_file = 10,
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
    all_csv_files = []

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
        learning_rate=0.2,
        discount_factor=0.95,
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
        trained_agent, stats, episode_metrics = train_agent(
            env=env,
            agent=agent,
            num_episodes=episodes_per_file,
            max_steps_per_episode=max_steps_per_episode,
            print_interval=print_interval
        )

        file_basename = os.path.basename(hlo_file)
        
        # Save detailed metrics to CSV
        csv_file = save_training_data_to_csv(
            episode_metrics=episode_metrics, 
            hlo_file_name=file_basename,
            available_passes=available_passes
        )
        all_csv_files.append(csv_file)
        
        if verbose:
            # plot results for this file:
            best_sequence = plot_training_results(stats=stats, available_passes=available_passes, hlo_file_name=file_basename)
        
        # store results for this file
        results[file_basename] = {
            'best_sequence': best_sequence,
            'best_reward': stats['best_reward'],
            'episode_rewards': stats['episode_rewards'],
            'csv_file': csv_file
        }
        # clear out the optimized directory between transitions to other files
        clear_optimized_directory(env.xla_interface.optimized_dir)

    # Create a summary CSV that combines key metrics across all files
    if all_csv_files:
        create_training_summary(results, available_passes)

    return trained_agent, results, available_passes


def create_training_summary(results, available_passes):
    """Create a summary CSV that shows metrics across all files."""
    summary_data = []
    
    for file_name, file_results in results.items():
        # Load the CSV data
        csv_file = file_results.get('csv_file')
        if not csv_file or not os.path.exists(csv_file):
            continue
            
        df = pd.read_csv(csv_file)
        
        # Get metrics from the last episode (final state of training)
        if len(df) > 0:
            last_episode = df.iloc[-1]
            
            # Get metrics from the best episode
            best_episode = df.loc[df['total_reward'].idxmax()]
            
            # Count frequency of each pass in the best performing episode
            best_actions = [int(a) for a in best_episode['action_sequence'].split(',') if a]
            pass_counts = {i: best_actions.count(i) for i in range(len(available_passes))}
            most_used_idx = max(pass_counts.items(), key=lambda x: x[1])[0] if pass_counts else 0
            most_used_pass = available_passes[most_used_idx]
            
            # Add to summary
            summary_data.append({
                'file_name': file_name,
                'final_exploration_rate': last_episode['exploration_rate'],
                'best_reward': best_episode['total_reward'],
                'best_percent_reduction': best_episode['percent_reduction'],
                'most_used_pass': most_used_pass,
                'unique_passes_in_best': best_episode['unique_actions'],
                'best_episode_number': best_episode['episode']
            })
    
    if summary_data:
        # Create and save summary DataFrame
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv('training_summary_all_files.csv', index=False)
        print("Training summary saved to training_summary_all_files.csv")


def plot_training_results(stats, available_passes, hlo_file_name=None):
    """
    Plot the training results.

    Args:
        stats: Statistics from agent training
        available_passes: List of available optimization passes
        hlo_file_name: Name of the HLO file trained on
    """
    # Create a directory for plots if it doesn't exist
    plots_dir = "training_plots"
    os.makedirs(plots_dir, exist_ok=True)
    
    # Base filename for plots
    base_name = os.path.basename(hlo_file_name).split('.')[0] if hlo_file_name else "training"
    
    # Load the CSV data for additional plots
    csv_file = f"training_metrics_{base_name}.csv"
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        
        # 1. Plot episode rewards
        plt.figure(figsize=(10, 6))
        plt.plot(df['episode'], df['total_reward'])
        
        title = "Episode Rewards During Training"
        if hlo_file_name:
            title += f" - {hlo_file_name}"
        
        plt.title(title)
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.grid(True)
        
        reward_plot = os.path.join(plots_dir, f"rewards_{base_name}.png")
        plt.savefig(reward_plot)
        plt.close()
        
        # 2. Plot cost reduction over episodes
        plt.figure(figsize=(10, 6))
        plt.plot(df['episode'], df['percent_reduction'], 'g-')
        
        plt.title(f"Cost Reduction Percentage Over Episodes - {base_name}")
        plt.xlabel("Episode")
        plt.ylabel("Cost Reduction (%)")
        plt.grid(True)
        
        cost_plot = os.path.join(plots_dir, f"cost_reduction_{base_name}.png")
        plt.savefig(cost_plot)
        plt.close()
        
        # 3. Plot exploration rate decay
        plt.figure(figsize=(10, 6))
        plt.plot(df['episode'], df['exploration_rate'], 'r-')
        
        plt.title(f"Exploration Rate Decay - {base_name}")
        plt.xlabel("Episode")
        plt.ylabel("Exploration Rate (ε)")
        plt.grid(True)
        
        exploration_plot = os.path.join(plots_dir, f"exploration_rate_{base_name}.png")
        plt.savefig(exploration_plot)
        plt.close()
        
        # 4. Plot relationship between unique actions and reward
        plt.figure(figsize=(10, 6))
        plt.scatter(df['unique_actions'], df['total_reward'], alpha=0.6)
        
        # Add trend line
        if len(df) > 1:
            z = np.polyfit(df['unique_actions'], df['total_reward'], 1)
            p = np.poly1d(z)
            plt.plot(df['unique_actions'], p(df['unique_actions']), "r--")
        
        plt.title(f"Relationship Between Unique Actions and Reward - {base_name}")
        plt.xlabel("Number of Unique Actions")
        plt.ylabel("Total Reward")
        plt.grid(True)
        
        actions_plot = os.path.join(plots_dir, f"actions_vs_reward_{base_name}.png")
        plt.savefig(actions_plot)
        plt.close()
        
        # 5. Get best episode data for pass analysis
        if len(df) > 0:
            best_episode = df.loc[df['total_reward'].idxmax()]
            best_actions = [int(a) for a in best_episode['action_sequence'].split(',') if a]
            
            # Count passes used in best episode
            pass_counts = {}
            for action in best_actions:
                pass_name = available_passes[action]
                pass_counts[pass_name] = pass_counts.get(pass_name, 0) + 1
            
            # Plot histogram of passes in best episode
            if pass_counts:
                plt.figure(figsize=(12, 8))
                passes, counts = zip(*sorted(pass_counts.items(), key=lambda x: x[1], reverse=True))
                plt.bar(passes, counts)
                plt.xticks(rotation=90)
                plt.title(f"Passes Used in Best Episode ({best_episode['episode']}) - {base_name}")
                plt.xlabel("Pass Name")
                plt.ylabel("Frequency")
                plt.tight_layout()
                
                pass_hist_plot = os.path.join(plots_dir, f"best_passes_{base_name}.png")
                plt.savefig(pass_hist_plot)
                plt.close()
    else:
        # Fallback to simpler plot if CSV not available
        plt.figure(figsize=(10, 5))
        plt.plot(stats["episode_rewards"])
        
        title = "Episode Rewards During Training"
        if hlo_file_name:
            title += f" - {hlo_file_name}"
        
        plt.title(title)
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.grid(True)
        
        output_file = os.path.join(plots_dir, f"rewards_{base_name}.png")
        plt.savefig(output_file)
        plt.close()

    # Print best sequence
    print("\nBest Pass Sequence Found:")
    best_sequence_passes = [available_passes[i] for i in stats["best_sequence"]]
    print(best_sequence_passes)
    print(f"Best Total Reward: {stats['best_reward']:.4f}")
    
    return best_sequence_passes

def clear_optimized_directory(dir_path):
    """Clear all files in the optimized_hlo directory"""
    for file in os.listdir(dir_path):
        file_path = os.path.join(dir_path, file)
        if os.path.isfile(file_path):
            os.remove(file_path)
    print(f"Cleared contents of {dir_path}")

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

def analyze_state_space(agent, available_passes):
    """
    Analyze the agent's state space and Q-table to extract insights useful for a report.
    
    Args:
        agent: Trained Q-learning agent
        available_passes: List of available optimization passes
    """
    print("\nAnalyzing state space representation...")
    
    # Create a directory for state space analysis
    analysis_dir = "state_space_analysis"
    os.makedirs(analysis_dir, exist_ok=True)
    
    # Get Q-table information
    q_table = agent.q_table
    num_states = len(q_table)
    
    print(f"Q-table contains {num_states} unique states")
    
    # If no states were visited, nothing to analyze
    if num_states == 0:
        print("No states were visited during training. Cannot perform state space analysis.")
        return
    
    # 1. Analyze Q-value distribution
    all_q_values = []
    max_q_values = []
    for state, q_values in q_table.items():
        all_q_values.extend(q_values)
        max_q_values.append(np.max(q_values))
    
    # Plot Q-value distribution
    plt.figure(figsize=(10, 6))
    plt.hist(all_q_values, bins=50, alpha=0.7)
    plt.title("Distribution of All Q-values")
    plt.xlabel("Q-value")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.savefig(os.path.join(analysis_dir, "q_value_distribution.png"))
    plt.close()
    
    # Plot max Q-value distribution
    plt.figure(figsize=(10, 6))
    plt.hist(max_q_values, bins=30, alpha=0.7, color='green')
    plt.title("Distribution of Maximum Q-values per State")
    plt.xlabel("Maximum Q-value")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.savefig(os.path.join(analysis_dir, "max_q_value_distribution.png"))
    plt.close()
    
    # 2. Analyze which passes are preferred across the state space
    preferred_actions = {}
    confidence_values = []
    
    for state, q_values in q_table.items():
        max_q = np.max(q_values)
        if max_q > 0:  # Only consider states with positive Q-values
            best_action = np.argmax(q_values)
            pass_name = available_passes[best_action]
            preferred_actions[pass_name] = preferred_actions.get(pass_name, 0) + 1
            
            # Calculate confidence: how much better is the best action compared to the average?
            avg_q = np.mean(q_values)
            if avg_q != 0:
                confidence = (max_q - avg_q) / max(abs(avg_q), 1e-5)
                confidence_values.append(confidence)
    
    # Plot preferred passes
    if preferred_actions:
        plt.figure(figsize=(12, 8))
        passes, counts = zip(*sorted(preferred_actions.items(), key=lambda x: x[1], reverse=True))
        plt.bar(passes, counts)
        plt.xticks(rotation=90)
        plt.title("Preferred Optimization Passes Across State Space")
        plt.xlabel("Pass Name")
        plt.ylabel("Number of States Where Pass is Preferred")
        plt.tight_layout()
        plt.savefig(os.path.join(analysis_dir, "preferred_passes.png"))
        plt.close()
    
    # Plot confidence distribution
    if confidence_values:
        plt.figure(figsize=(10, 6))
        plt.hist(confidence_values, bins=30, alpha=0.7, color='orange')
        plt.title("Decision Confidence Distribution")
        plt.xlabel("Confidence (how much better the best action is)")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.savefig(os.path.join(analysis_dir, "decision_confidence.png"))
        plt.close()
    
    # 3. Dimensionality of state space
    # Get a sample of state tuples to analyze their structure
    state_samples = list(q_table.keys())[:min(100, len(q_table))]
    
    # Calculate average length of state tuples
    if state_samples:
        state_lengths = [len(state) for state in state_samples]
        avg_length = np.mean(state_lengths)
        
        # Create a report file with state space statistics
        with open(os.path.join(analysis_dir, "state_space_report.txt"), "w") as f:
            f.write("State Space Analysis Report\n")
            f.write("==========================\n\n")
            f.write(f"Number of unique states: {num_states}\n")
            f.write(f"Average state representation length: {avg_length:.2f} features\n")
            f.write(f"Range of Q-values: [{min(all_q_values):.4f}, {max(all_q_values):.4f}]\n")
            f.write(f"Average max Q-value: {np.mean(max_q_values):.4f}\n\n")
            
            f.write("Top 10 preferred passes:\n")
            for i, (pass_name, count) in enumerate(sorted(preferred_actions.items(), key=lambda x: x[1], reverse=True)[:10]):
                f.write(f"{i+1}. {pass_name}: preferred in {count} states ({count/num_states*100:.2f}%)\n")
            
            f.write("\nSample state representations:\n")
            for i, state in enumerate(state_samples[:5]):
                f.write(f"State {i+1}: {state}\n")
    
    print("State space analysis complete. Results saved to the 'state_space_analysis' directory.")

def main():
    """
    Set up and run training process on multiple HLO files using a single environment
    """
    # XLA directory
    xla_dir = "/Users/rayaanfaruqi/Documents/CS521/Final_Project/xla" # OTHER USERS CHANGE HERE
    
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
        episodes_per_file=3,
        max_steps_per_episode=3,
        print_interval=50,
        verbose=True
    )
    
    # Analyze the state space - this is a new addition
    analyze_state_space(trained_agent, available_passes)
    

    agent_data = {
        "q_table": dict(trained_agent.q_table),
        "learning_rate": trained_agent.learning_rate,
        "discount_factor": trained_agent.discount_factor,
        "exploration_rate": trained_agent.exploration_rate,
        "available_passes": available_passes
    }

    script_dir = os.path.dirname(os.path.abspath(__file__))  # Gets the reinforcement_learning directory
    models_dir = os.path.join(script_dir, "models")
    os.makedirs(models_dir, exist_ok=True)  # Create models directory if it doesn't exist
    model_path = os.path.join(models_dir, "trained_agent.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(agent_data, f)

    
    # Print summary of results
    print("\n\nTraining Summary Across All Files:")
    print("=" * 50)
    for file_name, file_results in results.items():
        print(f"\nFile: {file_name}")
        print(f"Best Pass Sequence: {file_results['best_sequence']}")
        print(f"Best Reward: {file_results['best_reward']:.4f}")


if __name__ == "__main__":
    main()
