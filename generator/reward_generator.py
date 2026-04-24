from .llm import client

def generate_reward_function(goal_description: str,
                             environment_description: str,
                             previous_attempt,
                             previous_result,
                             relevant_history):
    

    system_prompt = """ 
    You are an expert reinforcement learning engineer.
    Your job is to write Python reward functions for RL agents.
    The reward function must be a Python function called compute_reward(state, action, next_state, info) that returns a float.
    Output ONLY the Python function code, nothing else.
    """
    
    user_prompt =f"""
    Environment: {environment_description}
    Goal : {goal_description}
    """

    if previous_attempt:
        user_prompt += f"""
        Previous reward function attempt: {previous_attempt}
        Training results: {previous_result}
        the previous attempt has problems. Generate an improved version.        
        """

    response = client.models.generate_content(
        model="gemini-3.1-flash-lite-preview",
        config={
            "system_instruction": system_prompt
        },
        contents=[user_prompt]
    )

def should_stop(metrics, goal):
    """Let the LLM decide if the goal is achieved."""

    user_prompt = f"""
            Goal: {goal}
            Current metrics: {metrics}
            Has the goal been sufficiently achieved? 
            Answer only YES or NO.
            """
    
    response = client.models.generate_content(
        model="gemini-3.1-flash-lite-preview",
        contents=[user_prompt] 
    )

    return True if response == "YES" else False
