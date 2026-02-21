"""Shared OpenRouter API client for Hermes tools.

Provides a single lazy-initialized AsyncOpenAI client that all tool modules
can share, eliminating the duplicated _get_openrouter_client() / 
_get_summarizer_client() pattern previously copy-pasted across web_tools,
vision_tools, mixture_of_agents_tool, and session_search_tool.
"""

import os

from openai import AsyncOpenAI
from hermes_constants import OPENROUTER_BASE_URL

_client: AsyncOpenAI | None = None


def get_async_client() -> AsyncOpenAI:
    """Return a shared AsyncOpenAI client pointed at OpenRouter.

    The client is created lazily on first call and reused thereafter.
    Raises ValueError if OPENROUTER_API_KEY is not set.
    """
    global _client
    if _client is None:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable not set")
        _client = AsyncOpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL)
    return _client


def check_api_key() -> bool:
    """Check whether the OpenRouter API key is present."""
    return bool(os.getenv("OPENROUTER_API_KEY"))
