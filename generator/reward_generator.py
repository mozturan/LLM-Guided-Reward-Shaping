from .llm import client

def generate_reward_function(goal_description: str,
                             environment_description: str,
                             previous_attempt: str,
                             previous_result: str):
    

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
