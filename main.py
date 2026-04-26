from memory import retrieve_relevant_attempts, store_attempt
from generator import generate_reward_function, summarize_approach
from evaluator import evaluate_reward_function

def run_reward_design_agent(goal:str,
                            environment_description: str,
                            max_iterations: int = 10):
    
    print(f"Goal: {goal}")
    print("Starting autonomous reward design loop...\n")

    all_attempts = []  # grows each iteration

    for iteration in range(max_iterations):
        print(f"--- Iteration {iteration + 1} ---")

        # Generate reward function
        print("Generating reward function...")
        reward_code = generate_reward_function(
            goal_description=goal,
            environment_description=environment_description,
            all_previous_attempts=retrieve_relevant_attempts(goal)
        )

        # Summarize the approach
        approach_summary = summarize_approach(reward_code, backend=backend)
        print(f"Approach: {approach_summary}")


        # Evaluate it
        print("\nTraining and evaluating...")
        metrics = evaluate_reward_function(reward_code)
        
        print(f"Results: {metrics['behavioral_description']}")
        print(f"Mean reward: {metrics['mean_episode_reward']:.2f}")

        # Store in memory
        store_attempt(reward_code, metrics, iteration)
