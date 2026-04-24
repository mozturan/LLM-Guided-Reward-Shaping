

def evaluate_reward_function(reward_function_code: str,
                             training_steps: int = 10000):
    
    """
    TODO: 
    - Train agent with the given reward function,return metrics the LLM can understand.
    """

    # Dynamically execute the LLM-generated reward function
    exec(reward_function_code, globals())

    # Train for a short evaluation period
    metrics = train_and_evaluate(
        reward_fn=compute_reward,
        steps=training_steps
    )
    
    return {
        "mean_episode_reward": metrics["mean_reward"],
        "mean_episode_length": metrics["mean_length"],
        "action_smoothness": metrics["action_variance"],
        "lane_deviation": metrics["lane_error"],
        "collision_rate": metrics["collisions"],
        "behavioral_description": describe_behavior(metrics)
    }

def describe_behavior(metrics):
    """Convert raw metrics into natural language for the LLM."""
    description = []
    if metrics["action_variance"] > 0.3:
        description.append("agent movements are jerky and oscillating")
    if metrics["lane_error"] > 0.5:
        description.append("agent frequently leaves the lane")
    if metrics["collisions"] > 0.1:
        description.append("agent collides with obstacles regularly")
    return ". ".join(description) if description else "agent behavior looks reasonable"

