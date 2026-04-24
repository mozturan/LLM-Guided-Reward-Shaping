from memory import retrieve_relevant_attempts, store_attempt
from generator import generate_reward_function, should_stop
from evaluator import evaluate_reward_function

def run_reward_design_agent(goal:str,
                            environment_description: str,
                            max_iterations: int = 10):
    
    print(f"Goal: {goal}")
    print("Starting autonomous reward design loop...\n")

    best_result = None
    best_reward_code = None

    for iteration in range(max_iterations):
        print(f"--- Iteration {iteration + 1} ---")

                # Retrieve relevant past attempts from memory
        relevant_history = retrieve_relevant_attempts(
            current_problem=f"{goal} - iteration {iteration}",
            n=3
        ) if iteration > 0 else []

        # Generate reward function
        print("Generating reward function...")
        reward_code = generate_reward_function(
            goal_description=goal,
            environment_description=environment_description,
            previous_attempt=best_reward_code,
            previous_result=str(best_result),
            relevant_history=relevant_history
        )

        print("Generated reward function:")
        print(reward_code)

        # Evaluate it
        print("\nTraining and evaluating...")
        metrics = evaluate_reward_function(reward_code)
        
        print(f"Results: {metrics['behavioral_description']}")
        print(f"Mean reward: {metrics['mean_episode_reward']:.2f}")

        # Store in memory
        store_attempt(reward_code, metrics, iteration)

        # Update best
        if best_result is None or metrics['mean_episode_reward'] > best_result['mean_episode_reward']:
            best_result = metrics
            best_reward_code = reward_code
            print("New best reward function found!")

        # Agent decides whether to continue or stop
        if should_stop(metrics, goal):
            print("\nAgent determined goal is sufficiently achieved.")
            break
        
    print("\n=== Final Result ===")
    print(f"Best reward function:\n{best_reward_code}")
    return best_reward_code, best_result

