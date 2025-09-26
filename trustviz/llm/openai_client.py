# trustviz/llm/openai_client.py
import os
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv

def get_client() -> OpenAI:
    # Always (re)load .env so this works from scripts, tests, server, etc.
    load_dotenv(find_dotenv())

    key = os.environ.get("OPENAI_API_KEY")
    if not key or not key.startswith("sk-"):
        # Keep the message precise but safe (don't print the key)
        raise RuntimeError(f"OPENAI_API_KEY not set or malformed; visible={bool(key)}")
    return OpenAI(api_key=key)
