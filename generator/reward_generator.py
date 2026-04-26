from .llm import client

def generate_reward_function(goal_description: str,
                             environment_description: str,
                             all_previous_attempts: list | None = None, # list of {"approach": ..., "mean_reward": ..., "behavioral_description": ...}
                             ) -> str:
    

    # Set System Prompt
    system_prompt = """ 
    You are an expert reinforcement learning engineer.
    Your job is to write Python reward functions for RL agents.

    Rules: 

    - The reward function must be a Python function called compute_reward
    - Signature MUST be: def compute_reward(state, action, next_state, info) -> float
    - You may use numpy as np (already imported)
    - Output ONLY the Python function code, no explanation, no markdown, no backticks.
    """
    
    # Pass Environment and Goal Descriptions
    user_prompt =f"""
    Environment: {environment_description}
    Goal : {goal_description}
    """

    # All previous approaches — compact summaries to prevent repetition
    if all_previous_attempts:
        user_prompt+= "\nAll approaches tried so far: \n"
        for item in enumerate(all_previous_attempts):
            user_prompt += f"{item}\n"               

        user_prompt += """
            TASK:
            Generate an IMPROVED reward function that:
            1. Addresses the weaknesses described above
            2. Tries a meaningfully different approach from all previous attempts listed
            3. Builds on what worked in the best attempt
            4. Is not identical or near-identical to any previous attempt
            """
    else:
        user_prompt += """
            TASK:
            Generate an initial reward function for this goal.
            Start with a well-structured approach that balances the key objectives.
            """
    
    user_prompt += "Output ONLY the Python function. Start with 'def compute_reward'."

    response = client.models.generate_content(
        model="gemini-3.1-flash-lite-preview",
        config={
            "system_instruction": system_prompt
        },
        contents=[user_prompt]
    )

    return response

def summarize_approach(reward_code:str):

    prompt = f"""Summarize what approach this reward function uses in ONE short sentence.
    Focus on the key components and weighting strategy, not implementation details.

    {reward_code}

    One sentence summary:"""

    response = client.models.generate_content(
        model="gemini-3.1-flash-lite-preview",
        config={
            "system_instruction": prompt
        },
        contents=[prompt]
    )

    return response
