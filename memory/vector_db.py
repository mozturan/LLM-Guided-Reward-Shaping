import chromadb
from chromadb.utils import embedding_functions

client = chromadb.Client()
collection = client.create_collection("reward_function_memory")

def store_attempt(reward_code: str,
                  mertrics: dict,
                  iteration: int):
    
    """Store a reward function attempt and its results."""
    document = f"""
    Reward function: 
    {reward_code}

    Results:
    - Mean reward: {metrics['mean_episode_reward']}
    - Action smoothness: {metrics['action_smoothness']}
    - Lane deviation: {metrics['lane_deviation']}
    - Behavioral description: {metrics['behavioral_description']}
    """

    collection.add(
        documents=[document],
        ids = [f"attempt_{iteration}"],
        metadatas=[{
            "iteration": iteration,
            "mean_reward": metrics['mean_episode_reward']
        }]
    )


def retrieve_relevant_attempts(current_problem: str, n=3):
    """Find past attempts most relevant to the current situation."""
    results = collection.query(
        query_texts=[current_problem],
        n_results=n
    )
    return results['documents'][0]
