# trustviz/llm/openai_client.py
import os

try:
    from openai import OpenAI
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "The OpenAI Python SDK is missing or too old. "
        "Run:  pip install -U openai"
    ) from e

_client = None

def get_openai_client() -> OpenAI:
    """Return a singleton OpenAI client. Requires OPENAI_API_KEY in env."""
    global _client
    if _client is not None:
        return _client

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set in your environment (.env).")

    _client = OpenAI(api_key=api_key)  # SDK 1.x reads env too; passing explicitly is fine
    return _client

# Backward-compat alias so existing imports keep working
def get_client() -> OpenAI:
    return get_openai_client()

__all__ = ["get_openai_client", "get_client"]
