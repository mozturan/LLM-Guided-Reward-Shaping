from .llm import client

def generate_reward_function(goal_description: str,
                             environment_description: str,
                             best_attempt: dict | None = None, # {"code": ..., "mean_reward": 123.4, "approach": ...}
                             last_attempt: dict | None = None, # {"code": ..., "mean_reward": 123.4, "approach": ...}
                             all_previous_attempts: list | None = None, # list of {"approach": ..., "mean_reward": ...}
                             relevant_history: list | None = None # retrieved from ChromaDB
                             ):
    

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
        for i, attempt in enumerate(all_previous_attempts):
            user_prompt += f"Iteration {i+1}: mean_reward = {attempt['mean_reward']} | approach = {attempt['approach']} \n"        
        user_prompt += "\nDo not repeat the same approaches. Use the information about what was tried and what worked to generate a new, different approach. Be creative and try something meaningfully different from past attempts.\n"
        
    # Pass Relevant History
    if relevant_history:
        user_prompt += f"""
        Relevant past reward function attempts and results retrieved from memory:):"""
        for item in relevant_history:
            user_prompt += f"{item}\n\n"

    ## ! We need to get this part together
    # Pass Best Attempt
    if best_attempt:
        user_prompt += f"""
        Current best attempt: {best_attempt['code']}
        Training results of best attempt: {best_attempt['mean_reward']}
        Approach used in best attempt: {best_attempt['approach']}
        Weaknesses of best attempt: {best_attempt['behavioral_description']}
    """

    # Pass Last Attempt
    if last_attempt and last_attempt != best_attempt:
        user_prompt += f"""
        Most recent attempt: {last_attempt['code']}
        Training results of most recent attempt: {last_attempt['mean_reward']}
        Approach used in most recent attempt: {last_attempt['approach']}
        Weaknesses of most recent attempt: {last_attempt['behavioral_description']}
    """
    ## ! TO this point    

    # Core instruction
    if best_attempt or last_attempt:
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
