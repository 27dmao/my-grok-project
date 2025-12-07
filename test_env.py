from dotenv import load_dotenv
import os

# Load variables from .env in this folder
load_dotenv()

api_key = os.getenv("XAI_API_KEY")
print("API key from .env:", api_key)
