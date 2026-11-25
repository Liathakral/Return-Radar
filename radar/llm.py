import os

# Models are read from the environment so you can swap them without code changes.
# Check the OpenAI docs for current model names before running.
LLM_MODEL = os.getenv("RADAR_LLM_MODEL", "gpt-4o-mini")
EMBED_MODEL = os.getenv("RADAR_EMBED_MODEL", "text-embedding-3-small")


def get_client():
    """Return an OpenAI client, or None when no API key is set."""
    if not os.getenv("OPENAI_API_KEY"):
        return None
    from openai import OpenAI

    return OpenAI()
