import os
from dotenv import load_dotenv
from google import genai

# Load variables from .env into the environment
load_dotenv()

# Access the key
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("API Key not found. Did you set it in the .env file?")

client = genai.Client(api_key=api_key)

system_prompt = "Sen insanların sorularına cevap veren bilge birisin ve Yoda gibi konuşuyorsun."
user_prompt = "Hayatın anlamını nasıl bulabilirim?"

# response = client.models.generate_content(
#     model="gemini-3.1-flash-lite-preview",
#         config={
#             "system_instruction": system_prompt
#         },
#         contents=[user_prompt]
# )

# print(response.text)

