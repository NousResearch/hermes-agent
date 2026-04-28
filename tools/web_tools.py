#!/usr/bin/env python3
"""
Standalone Web Tools Module

This module provides generic web tools that work with multiple backend providers.
Backend is selected during ``hermes tools`` setup (web.backend in config.yaml).
When available, Hermes can route Firecrawl calls through a Nous-hosted tool-gateway
for Nous Subscribers only.

Available tools:
- web_search_tool: Search the web for information
- web_extract_tool: Extract content from specific web pages
- web_crawl_tool: Crawl websites with specific instructions

Backend compatibility:
- Exa: https://exa.ai (search, extract)
- Firecrawl: https://docs.firecrawl.dev/introduction (search, extract, crawl; direct or derived firecrawl-gateway.<domain> for Nous Subscribers)
- Parallel: https://docs.parallel.ai (search, extract)
- Tavily: https://tavily.com (search, extract, crawl)

LLM Processing:
- Uses OpenRouter API with Gemini 3 Flash Preview for intelligent content extraction
- Extracts key excerpts and creates markdown summaries to reduce token usage

Debug Mode:
- Set WEB_TOOLS_DEBUG=true to enable detailed logging
- Creates web_tools_debug_UUID.json in ./logs directory
- Captures all tool calls, results, and compression metrics

Usage:
    from web_tools import web_search_tool, web_extract_tool, web_crawl_tool
    
    # Search the web
    results = web_search_tool("Python machine learning libraries", limit=3)
    
    # Extract content from URLs  
    content = web_extract_tool(["https://example.com"], format="markdown")
    
    # Crawl a website
    crawl_data = web_crawl_tool("example.com", "Find contact information")
"""

import json
import logging
import os
import re
import asyncio
from typing import List, Dict, Any, Optional
import httpx
# NOTE: `from firecrawl import Firecrawl` is deliberately NOT at module top —
# the SDK pulls ~200 ms of imports (httpcore, firecrawl.v1/v2 type trees) and
# we only need it when the backend is actually "firecrawl". See
# _get_firecrawl_client() below for the lazy import.
from agent.auxiliary_client import (
    async_call_llm,
    extract_content_or_reasoning,
    get_async_text_auxiliary_client,
)
from tools.debug_helpers import DebugSession
from tools.managed_tool_gateway import (
    build_vendor_gateway_url,
    read_nous_access_token as _read_nous_access_token,
    resolve_managed_tool_gateway,
)
from tools.tool_backend_helpers import managed_nous_tools_enabled, prefers_gateway
from tools.url_safety import is_safe_url
from tools.website_policy import check_website_access

logger = logging.getLogger(__name__)


# ─── Backend Selection ────────────────────────────────────────────────────────

def _has_env(name: str) -> bool:
    val = os.getenv(name)
    return bool(val and val.strip())

def _load_web_config() -> dict:
    """Load the ``web:`` section from ~/.hermes/config.yaml."""
    try:
        from hermes_cli.config import load_config
        return load_config().get("web", {})
    except (ImportError, Exception):
        return {}

def _get_backend() -> str:
    """Determine which web backend to use.

    Reads ``web.backend`` from config.yaml (set by ``hermes tools``).
    Falls back to whichever API key is present for users who configured
    keys manually without running setup.
    """
    configured = (_load_web_config().get("backend") or "").lower().strip()
    if configured in ("parallel", "firecrawl", "tavily", "exa"):
        return configured

    # Fallback for manual / legacy config — pick the highest-priority
    # available backend. Firecrawl also counts as available when the managed
    # tool gateway is configured for Nous subscribers.
    backend_candidates = (
        ("firecrawl", _has_env("FIRECRAWL_API_KEY") or _has_env("FIRECRAWL_API_URL") or _is_tool_gateway_ready()),
        ("parallel", _has_env("PARALLEL_API_KEY")),
        ("tavily", _has_env("TAVILY_API_KEY")),
        ("exa", _has_env("EXA_API_KEY")),
    )
    for backend, available in backend_candidates:
        if available:
            return backend

    return "firecrawl"  # default (backward compat)


def _is_backend_available(backend: str) -> bool:
    """Return True when the selected backend is currently usable."""
    if backend == "exa":
        return _has_env("EXA_API_KEY")
    if backend == "parallel":
        return _has_env("PARALLEL_API_KEY")
    if backend == "firecrawl":
        return check_firecrawl_api_key()
    if backend == "tavily":
        return _has_env("TAVILY_API_KEY")
    return False

# ─── Firecrawl Client ────────────────────────────────────────────────────────

_firecrawl_client = None
_firecrawl_client_config = None


def _get_direct_firecrawl_config() -> Optional[tuple[Dict[str, str], tuple[str, Optional[str], Optional[str]]]]:
    """Return explicit direct Firecrawl kwargs + cache key, or None when unset."""
    api_key = os.getenv("FIRECRAWL_API_KEY", "").strip()
    api_url = os.getenv("FIRECRAWL_API_URL", "").strip().rstrip("/")

    if not api_key and not api_url:
        return None

    kwargs: Dict[str, str] = {}
    if api_key:
        kwargs["api_key"] = api_key
    if api_url:
        kwargs["api_url"] = api_url

    return kwargs, ("direct", api_url or None, api_key or None)


def _get_firecrawl_gateway_url() -> str:
    """Return configured Firecrawl gateway URL."""
    return build_vendor_gateway_url("firecrawl")


def _is_tool_gateway_ready() -> bool:
    """Return True when gateway URL and a Nous Subscriber token are available."""
    return resolve_managed_tool_gateway("firecrawl", token_reader=_read_nous_access_token) is not None


def _has_direct_firecrawl_config() -> bool:
    """Return True when direct Firecrawl config is explicitly configured."""
    return _get_direct_firecrawl_config() is not None


def _raise_web_backend_configuration_error() -> None:
    """Raise a clear error for unsupported web backend configuration."""
    message = (
        "Web tools are not configured. "
        "Set FIRECRAWL_API_KEY for cloud Firecrawl or set FIRECRAWL_API_URL for a self-hosted Firecrawl instance."
    )
    if managed_nous_tools_enabled():
        message += (
            " With your Nous subscription you can also use the Tool Gateway — "
            "run `hermes tools` and select Nous Subscription as the web provider."
        )
    raise ValueError(message)


def _firecrawl_backend_help_suffix() -> str:
    """Return optional managed-gateway guidance for Firecrawl help text."""
    if not managed_nous_tools_enabled():
        return ""
    return (
        ", or use the Nous Tool Gateway via your subscription "
        "(FIRECRAWL_GATEWAY_URL or TOOL_GATEWAY_DOMAIN)"
    )


def _web_requires_env() -> list[str]:
    """Return tool metadata env vars for the currently enabled web backends."""
    requires = [
        "EXA_API_KEY",
        "PARALLEL_API_KEY",
        "TAVILY_API_KEY",
        "FIRECRAWL_API_KEY",
        "FIRECRAWL_API_URL",
    ]
    if managed_nous_tools_enabled():
        requires.extend(
            [
                "FIRECRAWL_GATEWAY_URL",
                "TOOL_GATEWAY_DOMAIN",
                "TOOL_GATEWAY_SCHEME",
                "TOOL_GATEWAY_USER_TOKEN",
            ]
        )
    return requires


def _get_firecrawl_client():
    """Get or create Firecrawl client.

    When ``web.use_gateway`` is set in config, the Tool Gateway is preferred
    even if direct Firecrawl credentials are present.  Otherwise direct
    Firecrawl takes precedence when explicitly configured.
    """
    global _firecrawl_client, _firecrawl_client_config

    direct_config = _get_direct_firecrawl_config()
    if direct_config is not None and not prefers_gateway("web"):
        kwargs, client_config = direct_config
    else:
        managed_gateway = resolve_managed_tool_gateway(
            "firecrawl",
            token_reader=_read_nous_access_token,
        )
        if managed_gateway is None:
            logger.error("Firecrawl client initialization failed: missing direct config and tool-gateway auth.")
            _raise_web_backend_configuration_error()

        kwargs = {
            "api_key": managed_gateway.nous_user_token,
            "api_url": managed_gateway.gateway_origin,
        }
        client_config = (
            "tool-gateway",
            kwargs["api_url"],
            managed_gateway.nous_user_token,
        )

    if _firecrawl_client is not None and _firecrawl_client_config == client_config:
        return _firecrawl_client

    # Lazy import — ~200 ms of SDK init, only paid when firecrawl is actually used.
    from firecrawl import Firecrawl  # noqa: E402
    _firecrawl_client = Firecrawl(**kwargs)
    _firecrawl_client_config = client_config
    return _firecrawl_client

# ─── Parallel Client ─────────────────────────────────────────────────────────

_parallel_client = None
_async_parallel_client = None

def _get_parallel_client():
    """Get or create the Parallel sync client (lazy initialization).

    Requires PARALLEL_API_KEY environment variable.
    """
    from parallel import Parallel
    global _parallel_client
    if _parallel_client is None:
        api_key = os.getenv("PARALLEL_API_KEY")
        if not api_key:
            raise ValueError(
                "PARALLEL_API_KEY environment variable not set. "
                "Get your API key at https://parallel.ai"
            )
        _parallel_client = Parallel(api_key=api_key)
    return _parallel_client


def _get_async_parallel_client():
    """Get or create the Parallel async client (lazy initialization).

    Requires PARALLEL_API_KEY environment variable.
    """
    from parallel import AsyncParallel
    global _async_parallel_client
    if _async_parallel_client is None:
        api_key = os.getenv("PARALLEL_API_KEY")
        if not api_key:
            raise ValueError(
                "PARALLEL_API_KEY environment variable not set. "
                "Get your API key at https://parallel.ai"
            )
        _async_parallel_client = AsyncParallel(api_key=api_key)
    return _async_parallel_client

# ─── Tavily Client ───────────────────────────────────────────────────────────

_TAVILY_BASE_URL = os.getenv("TAVILY_BASE_URL", "https://api.tavily.com")


def _tavily_request(endpoint: str, payload: dict) -> dict:
    """Send a POST request to the Tavily API.

    Auth is provided via ``api_key`` in the JSON body (no header-based auth).
    Raises ``ValueError`` if ``TAVILY_API_KEY`` is not set.
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ValueError(
            "TAVILY_API_KEY environment variable not set. "
            "Get your API key at https://app.tavily.com/home"
        )
    payload["api_key"] = api_key
    url = f"{_TAVILY_BASE_URL}/{endpoint.lstrip('/')}"
    logger.info("Tavily %s request to %s", endpoint, url)
    response = httpx.post(url, json=payload, timeout=60)
    response.raise_for_status()
    return response.json()


def _normalize_tavily_search_results(response: dict) -> dict:
    """Normalize Tavily /search response to the standard web search format.

    Tavily returns ``{results: [{title, url, content, score, ...}]}``.
    We map to ``{success, data: {web: [{title, url, description, position}]}}``.
    """
    web_results = []
    for i, result in enumerate(response.get("results", [])):
        web_results.append({
            "title": result.get("title", ""),
            "url": result.get("url", ""),
            "description": result.get("content", ""),
            "position": i + 1,
        })
    return {"success": True, "data": {"web": web_results}}


def _normalize_tavily_documents(response: dict, fallback_url: str = "") -> List[Dict[str, Any]]:
    """Normalize Tavily /extract or /crawl response to the standard document format.

    Maps results to ``{url, title, content, raw_content, metadata}`` and
    includes any ``failed_results`` / ``failed_urls`` as error entries.
    """
    documents: List[Dict[str, Any]] = []
    for result in response.get("results", []):
        url = result.get("url", fallback_url)
        raw = result.get("raw_content", "") or result.get("content", "")
        documents.append({
            "url": url,
            "title": result.get("title", ""),
            "content": raw,
            "raw_content": raw,
            "metadata": {"sourceURL": url, "title": result.get("title", "")},
        })
    # Handle failed results
    for fail in response.get("failed_results", []):
        documents.append({
            "url": fail.get("url", fallback_url),
            "title": "",
            "content": "",
            "raw_content": "",
            "error": fail.get("error", "extraction failed"),
            "metadata": {"sourceURL": fail.get("url", fallback_url)},
        })
    for fail_url in response.get("failed_urls", []):
        url_str = fail_url if isinstance(fail_url, str) else str(fail_url)
        documents.append({
            "url": url_str,
            "title": "",
            "content": "",
            "raw_content": "",
            "error": "extraction failed",
            "metadata": {"sourceURL": url_str},
        })
    return documents


def _to_plain_object(value: Any) -> Any:
    """Convert SDK objects to plain python data structures when possible."""
    if value is None:
        return None

    if isinstance(value, (dict, list, str, int, float, bool)):
        return value

    if hasattr(value, "model_dump"):
        try:
            return value.model_dump()
        except Exception:
            pass

    if hasattr(value, "__dict__"):
        try:
            return {k: v for k, v in value.__dict__.items() if not k.startswith("_")}
        except Exception:
            pass

    return value


def _normalize_result_list(values: Any) -> List[Dict[str, Any]]:
    """Normalize mixed SDK/list payloads into a list of dicts."""
    if not isinstance(values, list):
        return []

    normalized: List[Dict[str, Any]] = []
    for item in values:
        plain = _to_plain_object(item)
        if isinstance(plain, dict):
            normalized.append(plain)
    return normalized


def _extract_web_search_results(response: Any) -> List[Dict[str, Any]]:
    """Extract Firecrawl search results across SDK/direct/gateway response shapes."""
    response_plain = _to_plain_object(response)

    if isinstance(response_plain, dict):
        data = response_plain.get("data")
        if isinstance(data, list):
            return _normalize_result_list(data)

        if isinstance(data, dict):
            data_web = _normalize_result_list(data.get("web"))
            if data_web:
                return data_web
            data_results = _normalize_result_list(data.get("results"))
            if data_results:
                return data_results

        top_web = _normalize_result_list(response_plain.get("web"))
        if top_web:
            return top_web

        top_results = _normalize_result_list(response_plain.get("results"))
        if top_results:
            return top_results

    if hasattr(response, "web"):
        return _normalize_result_list(getattr(response, "web", []))

    return []


def _extract_scrape_payload(scrape_result: Any) -> Dict[str, Any]:
    """Normalize Firecrawl scrape payload shape across SDK and gateway variants."""
    result_plain = _to_plain_object(scrape_result)
    if not isinstance(result_plain, dict):
        return {}

    nested = result_plain.get("data")
    if isinstance(nested, dict):
        return nested

    return result_plain


DEFAULT_MIN_LENGTH_FOR_SUMMARIZATION = 5000

def _is_nous_auxiliary_client(client: Any) -> bool:
    """Return True when the resolved auxiliary backend is Nous Portal."""
    from urllib.parse import urlparse

    base_url = str(getattr(client, "base_url", "") or "")
    host = (urlparse(base_url).hostname or "").lower()
    return host == "nousresearch.com" or host.endswith(".nousresearch.com")


def _resolve_web_extract_auxiliary(model: Optional[str] = None) -> tuple[Optional[Any], Optional[str], Dict[str, Any]]:
    """Resolve the current web-extract auxiliary client, model, and extra body."""
    client, default_model = get_async_text_auxiliary_client("web_extract")
    configured_model = os.getenv("AUXILIARY_WEB_EXTRACT_MODEL", "").strip()
    effective_model = model or configured_model or default_model

    extra_body: Dict[str, Any] = {}
    if client is not None and _is_nous_auxiliary_client(client):
        from agent.auxiliary_client import get_auxiliary_extra_body
        extra_body = get_auxiliary_extra_body() or {"tags": ["product=hermes-agent"]}

    return client, effective_model, extra_body


def _get_default_summarizer_model() -> Optional[str]:
    """Return the current default model for web extraction summarization."""
    _, model, _ = _resolve_web_extract_auxiliary()
    return model

_debug = DebugSession("web_tools", env_var="WEB_TOOLS_DEBUG")


async def process_content_with_llm(
    content: str, 
    url: str = "", 
    title: str = "",
    model: Optional[str] = None,
    min_length: int = DEFAULT_MIN_LENGTH_FOR_SUMMARIZATION
) -> Optional[str]:
    """
    Process web content using LLM to create intelligent summaries with key excerpts.
    
    This function uses Gemini 3 Flash Preview (or specified model) via OpenRouter API 
    to intelligently extract key information and create markdown summaries,
    significantly reducing token usage while preserving all important information.
    
    For very large content (>500k chars), uses chunked processing with synthesis.
    For extremely large content (>2M chars), refuses to process entirely.
    
    Args:
        content (str): The raw content to process
        url (str): The source URL (for context, optional)
        title (str): The page title (for context, optional)
        model (str): The model to use for processing (default: google/gemini-3-flash-preview)
        min_length (int): Minimum content length to trigger processing (default: 5000)
        
    Returns:
        Optional[str]: Processed markdown content, or None if content too short or processing fails
    """
    # Size thresholds
    MAX_CONTENT_SIZE = 2_000_000  # 2M chars - refuse entirely above this
    CHUNK_THRESHOLD = 500_000     # 500k chars - use chunked processing above this
    CHUNK_SIZE = 100_000          # 100k chars per chunk
    MAX_OUTPUT_SIZE = 5000        # Hard cap on final output size
    
    try:
        content_len = len(content)
        
        # Refuse if content is absurdly large
        if content_len > MAX_CONTENT_SIZE:
            size_mb = content_len / 1_000_000
            logger.warning("Content too large (%.1fMB > 2MB limit). Refusing to process.", size_mb)
            return f"[Content too large to process: {size_mb:.1f}MB. Try using web_crawl with specific extraction instructions, or search for a more focused source.]"
        
        # Skip processing if content is too short
        if content_len < min_length:
            logger.debug("Content too short (%d < %d chars), skipping LLM processing", content_len, min_length)
            return None
        
        # Create context information
        context_info = []
        if title:
            context_info.append(f"Title: {title}")
        if url:
            context_info.append(f"Source: {url}")
        context_str = "\n".join(context_info) + "\n\n" if context_info else ""
        
        # Check if we need chunked processing
        if content_len > CHUNK_THRESHOLD:
            logger.info("Content large (%d chars). Using chunked processing...", content_len)
            return await _process_large_content_chunked(
                content, context_str, model, CHUNK_SIZE, MAX_OUTPUT_SIZE
            )
        
        # Standard single-pass processing for normal content
        logger.info("Processing content with LLM (%d characters)", content_len)
        
        processed_content = await _call_summarizer_llm(content, context_str, model)
        
        if processed_content:
            # Enforce output cap
            if len(processed_content) > MAX_OUTPUT_SIZE:
                processed_content = processed_content[:MAX_OUTPUT_SIZE] + "\n\n[... summary truncated for context management ...]"
            
            # Log compression metrics
            processed_length = len(processed_content)
            compression_ratio = processed_length / content_len if content_len > 0 else 1.0
            logger.info("Content processed: %d -> %d chars (%.1f%%)", content_len, processed_length, compression_ratio * 100)
        
        return processed_content
        
    except Exception as e:
        logger.warning(
            "web_extract LLM summarization failed (%s). "
            "Tip: increase auxiliary.web_extract.timeout in config.yaml "
            "or switch to a faster auxiliary model.",
            str(e)[:120],
        )
        # Fall back to truncated raw content instead of returning a useless
        # error message.  The first ~5000 chars are almost always more useful
        # to the model than "[Failed to process content: ...]".
        truncated = content[:MAX_OUTPUT_SIZE]
        if len(content) > MAX_OUTPUT_SIZE:
            truncated += (
                f"\n\n[Content truncated — showing first {MAX_OUTPUT_SIZE:,} of "
                f"{len(content):,} chars. LLM summarization timed out. "
                f"To fix: increase auxiliary.web_extract.timeout in config.yaml, "
                f"or use a faster auxiliary model. Use browser_navigate for the full page.]"
            )
        return truncated


async def _call_summarizer_llm(
    content: str, 
    context_str: str, 
    model: Optional[str], 
    max_tokens: int = 20000,
    is_chunk: bool = False,
    chunk_info: str = ""
) -> Optional[str]:
    """
    Make a single LLM call to summarize content.
    
    Args:
        content: The content to summarize
        context_str: Context information (title, URL)
        model: Model to use
        max_tokens: Maximum output tokens
        is_chunk: Whether this is a chunk of a larger document
        chunk_info: Information about chunk position (e.g., "Chunk 2/5")
        
    Returns:
        Summarized content or None on failure
    """
    if is_chunk:
        # Chunk-specific prompt - aware that this is partial content
        system_prompt = """You are an expert content analyst processing a SECTION of a larger document. Your job is to extract and summarize the key information from THIS SECTION ONLY.

Important guidelines for chunk processing:
1. Do NOT write introductions or conclusions - this is a partial document
2. Focus on extracting ALL key facts, figures, data points, and insights from this section
3. Preserve important quotes, code snippets, and specific details verbatim
4. Use bullet points and structured formatting for easy synthesis later
5. Note any references to other sections (e.g., "as mentioned earlier", "see below") without trying to resolve them

Your output will be combined with summaries of other sections, so focus on thorough extraction rather than narrative flow."""

        user_prompt = f"""Extract key information from this SECTION of a larger document:

{context_str}{chunk_info}

SECTION CONTENT:
{content}

Extract all important information from this section in a structured format. Focus on facts, data, insights, and key details. Do not add introductions or conclusions."""

    else:
        # Standard full-document prompt
        system_prompt = """You are an expert content analyst. Your job is to process web content and create a comprehensive yet concise summary that preserves all important information while dramatically reducing bulk.

Create a well-structured markdown summary that includes:
1. Key excerpts (quotes, code snippets, important facts) in their original format
2. Comprehensive summary of all other important information
3. Proper markdown formatting with headers, bullets, and emphasis

Your goal is to preserve ALL important information while reducing length. Never lose key facts, figures, insights, or actionable information. Make it scannable and well-organized."""

        user_prompt = f"""Please process this web content and create a comprehensive markdown summary:

{context_str}CONTENT TO PROCESS:
{content}

Create a markdown summary that captures all key information in a well-organized, scannable format. Include important quotes and code snippets in their original formatting. Focus on actionable information, specific details, and unique insights."""

    # Call the LLM with retry logic — keep retries low since summarization
    # is a nice-to-have; the caller falls back to truncated content on failure.
    max_retries = 2
    retry_delay = 2
    last_error = None

    for attempt in range(max_retries):
        try:
            aux_client, effective_model, extra_body = _resolve_web_extract_auxiliary(model)
            if aux_client is None or not effective_model:
                logger.warning("No auxiliary model available for web content processing")
                return None
            call_kwargs = {
                "task": "web_extract",
                "model": effective_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.1,
                "max_tokens": max_tokens,
                # No explicit timeout — async_call_llm reads auxiliary.web_extract.timeout
                # from config (default 360s / 6min).  Users with slow local models can
                # increase it in config.yaml.
            }
            if extra_body:
                call_kwargs["extra_body"] = extra_body
            response = await async_call_llm(**call_kwargs)
            content = extract_content_or_reasoning(response)
            if content:
                return content
            # Reasoning-only / empty response — let the retry loop handle it
            logger.warning("LLM returned empty content (attempt %d/%d), retrying", attempt + 1, max_retries)
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, 60)
                continue
            return content  # Return whatever we got after exhausting retries
        except RuntimeError:
            logger.warning("No auxiliary model available for web content processing")
            return None
        except Exception as api_error:
            last_error = api_error
            if attempt < max_retries - 1:
                logger.warning("LLM API call failed (attempt %d/%d): %s", attempt + 1, max_retries, str(api_error)[:100])
                logger.warning("Retrying in %ds...", retry_delay)
                await asyncio.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, 60)
            else:
                raise last_error
    
    return None


async def _process_large_content_chunked(
    content: str, 
    context_str: str, 
    model: Optional[str], 
    chunk_size: int,
    max_output_size: int
) -> Optional[str]:
    """
    Process large content by chunking, summarizing each chunk in parallel,
    then synthesizing the summaries.
    
    Args:
        content: The large content to process
        context_str: Context information
        model: Model to use
        chunk_size: Size of each chunk in characters
        max_output_size: Maximum final output size
        
    Returns:
        Synthesized summary or None on failure
    """
    # Split content into chunks
    chunks = []
    for i in range(0, len(content), chunk_size):
        chunk = content[i:i + chunk_size]
        chunks.append(chunk)
    
    logger.info("Split into %d chunks of ~%d chars each", len(chunks), chunk_size)
    
    # Summarize each chunk in parallel
    async def summarize_chunk(chunk_idx: int, chunk_content: str) -> tuple[int, Optional[str]]:
        """Summarize a single chunk."""
        try:
            chunk_info = f"[Processing chunk {chunk_idx + 1} of {len(chunks)}]"
            summary = await _call_summarizer_llm(
                chunk_content, 
                context_str, 
                model, 
                max_tokens=10000,
                is_chunk=True,
                chunk_info=chunk_info
            )
            if summary:
                logger.info("Chunk %d/%d summarized: %d -> %d chars", chunk_idx + 1, len(chunks), len(chunk_content), len(summary))
            return chunk_idx, summary
        except Exception as e:
            logger.warning("Chunk %d/%d failed: %s", chunk_idx + 1, len(chunks), str(e)[:50])
            return chunk_idx, None
    
    # Run all chunk summarizations in parallel
    tasks = [summarize_chunk(i, chunk) for i, chunk in enumerate(chunks)]
    results = await asyncio.gather(*tasks)
    
    # Collect successful summaries in order
    summaries = []
    for chunk_idx, summary in sorted(results, key=lambda x: x[0]):
        if summary:
            summaries.append(f"## Section {chunk_idx + 1}\n{summary}")
    
    if not summaries:
        logger.debug("All chunk summarizations failed")
        return "[Failed to process large content: all chunk summarizations failed]"
    
    logger.info("Got %d/%d chunk summaries", len(summaries), len(chunks))
    
    # If only one chunk succeeded, just return it (with cap)
    if len(summaries) == 1:
        result = summaries[0]
        if len(result) > max_output_size:
            result = result[:max_output_size] + "\n\n[... truncated ...]"
        return result
    
    # Synthesize the summaries into a final summary
    logger.info("Synthesizing %d summaries...", len(summaries))
    
    combined_summaries = "\n\n---\n\n".join(summaries)
    
    synthesis_prompt = f"""You have been given summaries of different sections of a large document. 
Synthesize these into ONE cohesive, comprehensive summary that:
1. Removes redundancy between sections
2. Preserves all key facts, figures, and actionable information
3. Is well-organized with clear structure
4. Is under {max_output_size} characters

{context_str}SECTION SUMMARIES:
{combined_summaries}

Create a single, unified markdown summary."""

    try:
        aux_client, effective_model, extra_body = _resolve_web_extract_auxiliary(model)
        if aux_client is None or not effective_model:
            logger.warning("No auxiliary model for synthesis, concatenating summaries")
            fallback = "\n\n".join(summaries)
            if len(fallback) > max_output_size:
                fallback = fallback[:max_output_size] + "\n\n[... truncated ...]"
            return fallback

        call_kwargs = {
            "task": "web_extract",
            "model": effective_model,
            "messages": [
                {"role": "system", "content": "You synthesize multiple summaries into one cohesive, comprehensive summary. Be thorough but concise."},
                {"role": "user", "content": synthesis_prompt},
            ],
            "temperature": 0.1,
            "max_tokens": 20000,
        }
        if extra_body:
            call_kwargs["extra_body"] = extra_body
        response = await async_call_llm(**call_kwargs)
        final_summary = extract_content_or_reasoning(response)

        # Retry once on empty content (reasoning-only response)
        if not final_summary:
            logger.warning("Synthesis LLM returned empty content, retrying once")
            response = await async_call_llm(**call_kwargs)
            final_summary = extract_content_or_reasoning(response)

        # If still None after retry, fall back to concatenated summaries
        if not final_summary:
            logger.warning("Synthesis failed after retry — concatenating chunk summaries")
            fallback = "\n\n".join(summaries)
            if len(fallback) > max_output_size:
                fallback = fallback[:max_output_size] + "\n\n[... truncated ...]"
            return fallback

        # Enforce hard cap
        if len(final_summary) > max_output_size:
            final_summary = final_summary[:max_output_size] + "\n\n[... summary truncated for context management ...]"
        
        original_len = len(content)
        final_len = len(final_summary)
        compression = final_len / original_len if original_len > 0 else 1.0
        
        logger.info("Synthesis complete: %d -> %d chars (%.2f%%)", original_len, final_len, compression * 100)
        return final_summary
        
    except Exception as e:
        logger.warning("Synthesis failed: %s", str(e)[:100])
        # Fall back to concatenated summaries with truncation
        fallback = "\n\n".join(summaries)
        if len(fallback) > max_output_size:
            fallback = fallback[:max_output_size] + "\n\n[... truncated due to synthesis failure ...]"
        return fallback


def clean_base64_images(text: str) -> str:
    """
    Remove base64 encoded images from text to reduce token count and clutter.
    
    This function finds and removes base64 encoded images in various formats:
    - (data:image/png;base64,...)
    - (data:image/jpeg;base64,...)
    - (data:image/svg+xml;base64,...)
    - data:image/[type];base64,... (without parentheses)
    
    Args:
        text: The text content to clean
        
    Returns:
        Cleaned text with base64 images replaced with placeholders
    """
    # Pattern to match base64 encoded images wrapped in parentheses
    # Matches: (data:image/[type];base64,[base64-string])
    base64_with_parens_pattern = r'\(data:image/[^;]+;base64,[A-Za-z0-9+/=]+\)'
    
    # Pattern to match base64 encoded images without parentheses
    # Matches: data:image/[type];base64,[base64-string]
    base64_pattern = r'data:image/[^;]+;base64,[A-Za-z0-9+/=]+'
    
    # Replace parentheses-wrapped images first
    cleaned_text = re.sub(base64_with_parens_pattern, '[BASE64_IMAGE_REMOVED]', text)
    
    # Then replace any remaining non-parentheses images
    cleaned_text = re.sub(base64_pattern, '[BASE64_IMAGE_REMOVED]', cleaned_text)
    
    return cleaned_text


# ─── Exa Client ──────────────────────────────────────────────────────────────

_exa_client = None

def _get_exa_client():
    """Get or create the Exa client (lazy initialization).

    Requires EXA_API_KEY environment variable.
    """
    from exa_py import Exa
    global _exa_client
    if _exa_client is None:
        api_key = os.getenv("EXA_API_KEY")
        if not api_key:
            raise ValueError(
                "EXA_API_KEY environment variable not set. "
                "Get your API key at https://exa.ai"
            )
        _exa_client = Exa(api_key=api_key)
        _exa_client.headers["x-exa-integration"] = "hermes-agent"
    return _exa_client


# ─── Exa Search & Extract Helpers ─────────────────────────────────────────────

def _exa_search(query: str, limit: int = 10) -> dict:
    """Search using the Exa SDK and return results as a dict."""
    from tools.interrupt import is_interrupted
    if is_interrupted():
        return {"error": "Interrupted", "success": False}

    logger.info("Exa search: '%s' (limit=%d)", query, limit)
    response = _get_exa_client().search(
        query,
        num_results=limit,
        contents={
            "highlights": True,
        },
    )

    web_results = []
    for i, result in enumerate(response.results or []):
        highlights = result.highlights or []
        web_results.append({
            "url": result.url or "",
            "title": result.title or "",
            "description": " ".join(highlights) if highlights else "",
            "position": i + 1,
        })

    return {"success": True, "data": {"web": web_results}}


def _exa_extract(urls: List[str]) -> List[Dict[str, Any]]:
    """Extract content from URLs using the Exa SDK.

    Returns a list of result dicts matching the structure expected by the
    LLM post-processing pipeline (url, title, content, metadata).
    """
    from tools.interrupt import is_interrupted
    if is_interrupted():
        return [{"url": u, "error": "Interrupted", "title": ""} for u in urls]

    logger.info("Exa extract: %d URL(s)", len(urls))
    response = _get_exa_client().get_contents(
        urls,
        text=True,
    )

    results = []
    for result in response.results or []:
        content = result.text or ""
        url = result.url or ""
        title = result.title or ""
        results.append({
            "url": url,
            "title": title,
            "content": content,
            "raw_content": content,
            "metadata": {"sourceURL": url, "title": title},
        })

    return results


# ─── Parallel Search & Extract Helpers ────────────────────────────────────────

def _parallel_search(query: str, limit: int = 5) -> dict:
    """Search using the Parallel SDK and return results as a dict."""
    from tools.interrupt import is_interrupted
    if is_interrupted():
        return {"error": "Interrupted", "success": False}

    mode = os.getenv("PARALLEL_SEARCH_MODE", "agentic").lower().strip()
    if mode not in ("fast", "one-shot", "agentic"):
        mode = "agentic"

    logger.info("Parallel search: '%s' (mode=%s, limit=%d)", query, mode, limit)
    response = _get_parallel_client().beta.search(
        search_queries=[query],
        objective=query,
        mode=mode,
        max_results=min(limit, 20),
    )

    web_results = []
    for i, result in enumerate(response.results or []):
        excerpts = result.excerpts or []
        web_results.append({
            "url": result.url or "",
            "title": result.title or "",
            "description": " ".join(excerpts) if excerpts else "",
            "position": i + 1,
        })

    return {"success": True, "data": {"web": web_results}}


async def _parallel_extract(urls: List[str]) -> List[Dict[str, Any]]:
    """Extract content from URLs using the Parallel async SDK.

    Returns a list of result dicts matching the structure expected by the
    LLM post-processing pipeline (url, title, content, metadata).
    """
    from tools.interrupt import is_interrupted
    if is_interrupted():
        return [{"url": u, "error": "Interrupted", "title": ""} for u in urls]

    logger.info("Parallel extract: %d URL(s)", len(urls))
    response = await _get_async_parallel_client().beta.extract(
        urls=urls,
        full_content=True,
    )

    results = []
    for result in response.results or []:
        content = result.full_content or ""
        if not content:
            content = "\n\n".join(result.excerpts or [])
        url = result.url or ""
        title = result.title or ""
        results.append({
            "url": url,
            "title": title,
            "content": content,
            "raw_content": content,
            "metadata": {"sourceURL": url, "title": title},
        })

    for error in response.errors or []:
        results.append({
            "url": error.url or "",
            "title": "",
            "content": "",
            "error": error.content or error.error_type or "extraction failed",
            "metadata": {"sourceURL": error.url or ""},
        })

    return results


def web_search_tool(query: str, limit: int = 5) -> str:
    """
    Search the web for information using available search API backend.

    This function provides a generic interface for web search that can work
    with multiple backends (Parallel or Firecrawl).

    Note: This function returns search result metadata only (URLs, titles, descriptions).
    Use web_extract_tool to get full content from specific URLs.
    
    Args:
        query (str): The search query to look up
        limit (int): Maximum number of results to return (default: 5)
    
    Returns:
        str: JSON string containing search results with the following structure:
             {
                 "success": bool,
                 "data": {
                     "web": [
                         {
                             "title": str,
                             "url": str,
                             "description": str,
                             "position": int
                         },
                         ...
                     ]
                 }
             }
    
    Raises:
        Exception: If search fails or API key is not set
    """
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        limit = 5
    limit = min(max(limit, 1), 100)

    debug_call_data = {
        "parameters": {
            "query": query,
            "limit": limit
        },
        "error": None,
        "results_count": 0,
        "original_response_size": 0,
        "final_response_size": 0
    }
    
    try:
        from tools.interrupt import is_interrupted
        if is_interrupted():
            return tool_error("Interrupted", success=False)

        # Dispatch to the configured backend
        backend = _get_backend()
        if backend == "parallel":
            response_data = _parallel_search(query, limit)
            debug_call_data["results_count"] = len(response_data.get("data", {}).get("web", []))
            result_json = json.dumps(response_data, indent=2, ensure_ascii=False)
            debug_call_data["final_response_size"] = len(result_json)
            _debug.log_call("web_search_tool", debug_call_data)
            _debug.save()
            return result_json

        if backend == "exa":
            response_data = _exa_search(query, limit)
            debug_call_data["results_count"] = len(response_data.get("data", {}).get("web", []))
            result_json = json.dumps(response_data, indent=2, ensure_ascii=False)
            debug_call_data["final_response_size"] = len(result_json)
            _debug.log_call("web_search_tool", debug_call_data)
            _debug.save()
            return result_json

        if backend == "tavily":
            logger.info("Tavily search: '%s' (limit: %d)", query, limit)
            raw = _tavily_request("search", {
                "query": query,
                "max_results": min(limit, 20),
                "include_raw_content": False,
                "include_images": False,
            })
            response_data = _normalize_tavily_search_results(raw)
            debug_call_data["results_count"] = len(response_data.get("data", {}).get("web", []))
            result_json = json.dumps(response_data, indent=2, ensure_ascii=False)
            debug_call_data["final_response_size"] = len(result_json)
            _debug.log_call("web_search_tool", debug_call_data)
            _debug.save()
            return result_json

        logger.info("Searching the web for: '%s' (limit: %d)", query, limit)

        response = _get_firecrawl_client().search(
            query=query,
            limit=limit
        )

        web_results = _extract_web_search_results(response)
        results_count = len(web_results)
        logger.info("Found %d search results", results_count)
        
        # Build response with just search metadata (URLs, titles, descriptions)
        response_data = {
            "success": True,
            "data": {
                "web": web_results
            }
        }
        
        # Capture debug information
        debug_call_data["results_count"] = results_count
        
        # Convert to JSON
        result_json = json.dumps(response_data, indent=2, ensure_ascii=False)
        
        debug_call_data["final_response_size"] = len(result_json)
        
        # Log debug information
        _debug.log_call("web_search_tool", debug_call_data)
        _debug.save()
        
        return result_json
        
    except Exception as e:
        error_msg = f"Error searching web: {str(e)}"
        logger.debug("%s", error_msg)

        debug_call_data["error"] = error_msg
        _debug.log_call("web_search_tool", debug_call_data)
        _debug.save()

        return tool_error(error_msg)


async def web_extract_tool(
    urls: List[str],
    format: str = None,
    use_llm_processing: bool = True,
    model: Optional[str] = None,
    min_length: int = DEFAULT_MIN_LENGTH_FOR_SUMMARIZATION
) -> str:
    """
    Extract content from specific web pages using available extraction API backend.

    This function provides a generic interface for web content extraction that
    can work with multiple backends. Currently uses Firecrawl.

    Args:
        urls (List[str]): List of URLs to extract content from
        format (str): Desired output format ("markdown" or "html", optional)
        use_llm_processing (bool): Whether to process content with LLM for summarization (default: True)
        model (Optional[str]): The model to use for LLM processing (defaults to current auxiliary backend model)
        min_length (int): Minimum content length to trigger LLM processing (default: 5000)

    Security: URLs are checked for embedded secrets before fetching.
    
    Returns:
        str: JSON string containing extracted content. If LLM processing is enabled and successful,
             the 'content' field will contain the processed markdown summary instead of raw content.
    
    Raises:
        Exception: If extraction fails or API key is not set
    """
    # Block URLs containing embedded secrets (exfiltration prevention).
    # URL-decode first so percent-encoded secrets (%73k- = sk-) are caught.
    from agent.redact import _PREFIX_RE
    from urllib.parse import unquote
    for _url in urls:
        if _PREFIX_RE.search(_url) or _PREFIX_RE.search(unquote(_url)):
            return json.dumps({
                "success": False,
                "error": "Blocked: URL contains what appears to be an API key or token. "
                         "Secrets must not be sent in URLs.",
            })

    debug_call_data = {
        "parameters": {
            "urls": urls,
            "format": format,
            "use_llm_processing": use_llm_processing,
            "model": model,
            "min_length": min_length
        },
        "error": None,
        "pages_extracted": 0,
        "pages_processed_with_llm": 0,
        "original_response_size": 0,
        "final_response_size": 0,
        "compression_metrics": [],
        "processing_applied": []
    }
    
    try:
        logger.info("Extracting content from %d URL(s)", len(urls))

        # ── SSRF protection — filter out private/internal URLs before any backend ──
        safe_urls = []
        ssrf_blocked: List[Dict[str, Any]] = []
        for url in urls:
            if not is_safe_url(url):
                ssrf_blocked.append({
                    "url": url, "title": "", "content": "",
                    "error": "Blocked: URL targets a private or internal network address",
                })
            else:
                safe_urls.append(url)

        # Dispatch only safe URLs to the configured backend
        if not safe_urls:
            results = []
        else:
            backend = _get_backend()

            if backend == "parallel":
                results = await _parallel_extract(safe_urls)
            elif backend == "exa":
                results = _exa_extract(safe_urls)
            elif backend == "tavily":
                logger.info("Tavily extract: %d URL(s)", len(safe_urls))
                raw = _tavily_request("extract", {
                    "urls": safe_urls,
                    "include_images": False,
                })
                results = _normalize_tavily_documents(raw, fallback_url=safe_urls[0] if safe_urls else "")
            else:
                # ── Firecrawl extraction ──
                # Determine requested formats for Firecrawl v2
                formats: List[str] = []
                if format == "markdown":
                    formats = ["markdown"]
                elif format == "html":
                    formats = ["html"]
                else:
                    # Default: request markdown for LLM-readiness and include html as backup
                    formats = ["markdown", "html"]

                # Always use individual scraping for simplicity and reliability
                # Batch scraping adds complexity without much benefit for small numbers of URLs
                results: List[Dict[str, Any]] = []

                from tools.interrupt import is_interrupted as _is_interrupted
                for url in safe_urls:
                    if _is_interrupted():
                        results.append({"url": url, "error": "Interrupted", "title": ""})
                        continue

                    # Website policy check — block before fetching
                    blocked = check_website_access(url)
                    if blocked:
                        logger.info("Blocked web_extract for %s by rule %s", blocked["host"], blocked["rule"])
                        results.append({
                            "url": url, "title": "", "content": "",
                            "error": blocked["message"],
                            "blocked_by_policy": {"host": blocked["host"], "rule": blocked["rule"], "source": blocked["source"]},
                        })
                        continue

                    try:
                        logger.info("Scraping: %s", url)
                        # Run synchronous Firecrawl scrape in a thread with a
                        # 60s timeout so a hung fetch doesn't block the session.
                        try:
                            scrape_result = await asyncio.wait_for(
                                asyncio.to_thread(
                                    _get_firecrawl_client().scrape,
                                    url=url,
                                    formats=formats,
                                ),
                                timeout=60,
                            )
                        except asyncio.TimeoutError:
                            logger.warning("Firecrawl scrape timed out for %s", url)
                            results.append({
                                "url": url, "title": "", "content": "",
                                "error": "Scrape timed out after 60s — page may be too large or unresponsive. Try browser_navigate instead.",
                            })
                            continue

                        scrape_payload = _extract_scrape_payload(scrape_result)
                        metadata = scrape_payload.get("metadata", {})
                        title = ""
                        content_markdown = scrape_payload.get("markdown")
                        content_html = scrape_payload.get("html")

                        # Ensure metadata is a dict (not an object)
                        if not isinstance(metadata, dict):
                            if hasattr(metadata, 'model_dump'):
                                metadata = metadata.model_dump()
                            elif hasattr(metadata, '__dict__'):
                                metadata = metadata.__dict__
                            else:
                                metadata = {}

                        # Get title from metadata
                        title = metadata.get("title", "")

                        # Re-check final URL after redirect
                        final_url = metadata.get("sourceURL", url)
                        final_blocked = check_website_access(final_url)
                        if final_blocked:
                            logger.info("Blocked redirected web_extract for %s by rule %s", final_blocked["host"], final_blocked["rule"])
                            results.append({
                                "url": final_url, "title": title, "content": "", "raw_content": "",
                                "error": final_blocked["message"],
                                "blocked_by_policy": {"host": final_blocked["host"], "rule": final_blocked["rule"], "source": final_blocked["source"]},
                            })
                            continue

                        # Choose content based on requested format
                        chosen_content = content_markdown if (format == "markdown" or (format is None and content_markdown)) else content_html or content_markdown or ""

                        results.append({
                            "url": final_url,
                            "title": title,
                            "content": chosen_content,
                            "raw_content": chosen_content,
                            "metadata": metadata  # Now guaranteed to be a dict
                        })

                    except Exception as scrape_err:
                        logger.debug("Scrape failed for %s: %s", url, scrape_err)
                        results.append({
                            "url": url,
                            "title": "",
                            "content": "",
                            "raw_content": "",
                            "error": str(scrape_err)
                        })

        # Merge any SSRF-blocked results back in
        if ssrf_blocked:
            results = ssrf_blocked + results

        response = {"results": results}
        
        pages_extracted = len(response.get('results', []))
        logger.info("Extracted content from %d pages", pages_extracted)
        
        debug_call_data["pages_extracted"] = pages_extracted
        debug_call_data["original_response_size"] = len(json.dumps(response))
        effective_model = model or _get_default_summarizer_model()
        auxiliary_available = check_auxiliary_model()
        
        # Process each result with LLM if enabled
        if use_llm_processing and auxiliary_available:
            logger.info("Processing extracted content with LLM (parallel)...")
            debug_call_data["processing_applied"].append("llm_processing")
            
            # Prepare tasks for parallel processing
            async def process_single_result(result):
                """Process a single result with LLM and return updated result with metrics."""
                url = result.get('url', 'Unknown URL')
                title = result.get('title', '')
                raw_content = result.get('raw_content', '') or result.get('content', '')
                
                if not raw_content:
                    return result, None, "no_content"
                
                original_size = len(raw_content)
                
                # Process content with LLM
                processed = await process_content_with_llm(
                    raw_content, url, title, effective_model, min_length
                )
                
                if processed:
                    processed_size = len(processed)
                    compression_ratio = processed_size / original_size if original_size > 0 else 1.0
                    
                    # Update result with processed content
                    result['content'] = processed
                    result['raw_content'] = raw_content
                    
                    metrics = {
                        "url": url,
                        "original_size": original_size,
                        "processed_size": processed_size,
                        "compression_ratio": compression_ratio,
                        "model_used": effective_model
                    }
                    return result, metrics, "processed"
                else:
                    metrics = {
                        "url": url,
                        "original_size": original_size,
                        "processed_size": original_size,
                        "compression_ratio": 1.0,
                        "model_used": None,
                        "reason": "content_too_short"
                    }
                    return result, metrics, "too_short"
            
            # Run all LLM processing in parallel
            results_list = response.get('results', [])
            tasks = [process_single_result(result) for result in results_list]
            processed_results = await asyncio.gather(*tasks)
            
            # Collect metrics and print results
            for result, metrics, status in processed_results:
                url = result.get('url', 'Unknown URL')
                if status == "processed":
                    debug_call_data["compression_metrics"].append(metrics)
                    debug_call_data["pages_processed_with_llm"] += 1
                    logger.info("%s (processed)", url)
                elif status == "too_short":
                    debug_call_data["compression_metrics"].append(metrics)
                    logger.info("%s (no processing - content too short)", url)
                else:
                    logger.warning("%s (no content to process)", url)
        else:
            if use_llm_processing and not auxiliary_available:
                logger.warning("LLM processing requested but no auxiliary model available, returning raw content")
                debug_call_data["processing_applied"].append("llm_processing_unavailable")
            # Print summary of extracted pages for debugging (original behavior)
            for result in response.get('results', []):
                url = result.get('url', 'Unknown URL')
                content_length = len(result.get('raw_content', ''))
                logger.info("%s (%d characters)", url, content_length)
        
        # Trim output to minimal fields per entry: title, content, error
        trimmed_results = [
            {
                "url": r.get("url", ""),
                "title": r.get("title", ""),
                "content": r.get("content", ""),
                "error": r.get("error"),
                **({  "blocked_by_policy": r["blocked_by_policy"]} if "blocked_by_policy" in r else {}),
            }
            for r in response.get("results", [])
        ]
        trimmed_response = {"results": trimmed_results}

        if trimmed_response.get("results") == []:
            result_json = tool_error("Content was inaccessible or not found")

            cleaned_result = clean_base64_images(result_json)
        
        else:
            result_json = json.dumps(trimmed_response, indent=2, ensure_ascii=False)
            
            cleaned_result = clean_base64_images(result_json)
        
        debug_call_data["final_response_size"] = len(cleaned_result)
        debug_call_data["processing_applied"].append("base64_image_removal")
        
        # Log debug information
        _debug.log_call("web_extract_tool", debug_call_data)
        _debug.save()
        
        return cleaned_result
            
    except Exception as e:
        error_msg = f"Error extracting content: {str(e)}"
        logger.debug("%s", error_msg)
        
        debug_call_data["error"] = error_msg
        _debug.log_call("web_extract_tool", debug_call_data)
        _debug.save()
        
        return tool_error(error_msg)


async def web_crawl_tool(
    url: str, 
    instructions: str = None, 
    depth: str = "basic", 
    use_llm_processing: bool = True,
    model: Optional[str] = None,
    min_length: int = DEFAULT_MIN_LENGTH_FOR_SUMMARIZATION
) -> str:
    """
    Crawl a website with specific instructions using available crawling API backend.
    
    This function provides a generic interface for web crawling that can work
    with multiple backends. Currently uses Firecrawl.
    
    Args:
        url (str): The base URL to crawl (can include or exclude https://)
        instructions (str): Instructions for what to crawl/extract using LLM intelligence (optional)
        depth (str): Depth of extraction ("basic" or "advanced", default: "basic")
        use_llm_processing (bool): Whether to process content with LLM for summarization (default: True)
        model (Optional[str]): The model to use for LLM processing (defaults to current auxiliary backend model)
        min_length (int): Minimum content length to trigger LLM processing (default: 5000)
    
    Returns:
        str: JSON string containing crawled content. If LLM processing is enabled and successful,
             the 'content' field will contain the processed markdown summary instead of raw content.
             Each page is processed individually.
    
    Raises:
        Exception: If crawling fails or API key is not set
    """
    debug_call_data = {
        "parameters": {
            "url": url,
            "instructions": instructions,
            "depth": depth,
            "use_llm_processing": use_llm_processing,
            "model": model,
            "min_length": min_length
        },
        "error": None,
        "pages_crawled": 0,
        "pages_processed_with_llm": 0,
        "original_response_size": 0,
        "final_response_size": 0,
        "compression_metrics": [],
        "processing_applied": []
    }
    
    try:
        effective_model = model or _get_default_summarizer_model()
        auxiliary_available = check_auxiliary_model()
        backend = _get_backend()

        # Tavily supports crawl via its /crawl endpoint
        if backend == "tavily":
            # Ensure URL has protocol
            if not url.startswith(('http://', 'https://')):
                url = f'https://{url}'

            # SSRF protection — block private/internal addresses
            if not is_safe_url(url):
                return json.dumps({"results": [{"url": url, "title": "", "content": "",
                    "error": "Blocked: URL targets a private or internal network address"}]}, ensure_ascii=False)

            # Website policy check
            blocked = check_website_access(url)
            if blocked:
                logger.info("Blocked web_crawl for %s by rule %s", blocked["host"], blocked["rule"])
                return json.dumps({"results": [{"url": url, "title": "", "content": "", "error": blocked["message"],
                    "blocked_by_policy": {"host": blocked["host"], "rule": blocked["rule"], "source": blocked["source"]}}]}, ensure_ascii=False)

            from tools.interrupt import is_interrupted as _is_int
            if _is_int():
                return tool_error("Interrupted", success=False)

            logger.info("Tavily crawl: %s", url)
            payload: Dict[str, Any] = {
                "url": url,
                "limit": 20,
                "extract_depth": depth,
            }
            if instructions:
                payload["instructions"] = instructions
            raw = _tavily_request("crawl", payload)
            results = _normalize_tavily_documents(raw, fallback_url=url)

            response = {"results": results}
            # Fall through to the shared LLM processing and trimming below
            # (skip the Firecrawl-specific crawl logic)
            pages_crawled = len(response.get('results', []))
            logger.info("Crawled %d pages", pages_crawled)
            debug_call_data["pages_crawled"] = pages_crawled
            debug_call_data["original_response_size"] = len(json.dumps(response))

            # Process each result with LLM if enabled
            if use_llm_processing and auxiliary_available:
                logger.info("Processing crawled content with LLM (parallel)...")
                debug_call_data["processing_applied"].append("llm_processing")

                async def _process_tavily_crawl(result):
                    page_url = result.get('url', 'Unknown URL')
                    title = result.get('title', '')
                    content = result.get('content', '')
                    if not content:
                        return result, None, "no_content"
                    original_size = len(content)
                    processed = await process_content_with_llm(content, page_url, title, effective_model, min_length)
                    if processed:
                        result['raw_content'] = content
                        result['content'] = processed
                        metrics = {"url": page_url, "original_size": original_size, "processed_size": len(processed),
                                   "compression_ratio": len(processed) / original_size if original_size else 1.0, "model_used": effective_model}
                        return result, metrics, "processed"
                    metrics = {"url": page_url, "original_size": original_size, "processed_size": original_size,
                               "compression_ratio": 1.0, "model_used": None, "reason": "content_too_short"}
                    return result, metrics, "too_short"

                tasks = [_process_tavily_crawl(r) for r in response.get('results', [])]
                processed_results = await asyncio.gather(*tasks)
                for result, metrics, status in processed_results:
                    if status == "processed":
                        debug_call_data["compression_metrics"].append(metrics)
                        debug_call_data["pages_processed_with_llm"] += 1

            if use_llm_processing and not auxiliary_available:
                logger.warning("LLM processing requested but no auxiliary model available, returning raw content")
                debug_call_data["processing_applied"].append("llm_processing_unavailable")

            trimmed_results = [{"url": r.get("url", ""), "title": r.get("title", ""), "content": r.get("content", ""), "error": r.get("error"),
                **({  "blocked_by_policy": r["blocked_by_policy"]} if "blocked_by_policy" in r else {})} for r in response.get("results", [])]
            result_json = json.dumps({"results": trimmed_results}, indent=2, ensure_ascii=False)
            cleaned_result = clean_base64_images(result_json)
            debug_call_data["final_response_size"] = len(cleaned_result)
            _debug.log_call("web_crawl_tool", debug_call_data)
            _debug.save()
            return cleaned_result

        # web_crawl requires Firecrawl or the Firecrawl tool-gateway — Parallel has no crawl API
        if not check_firecrawl_api_key():
            return json.dumps({
                "error": "web_crawl requires Firecrawl. Set FIRECRAWL_API_KEY, FIRECRAWL_API_URL"
                         f"{_firecrawl_backend_help_suffix()}, or use web_search + web_extract instead.",
                "success": False,
            }, ensure_ascii=False)

        # Ensure URL has protocol
        if not url.startswith(('http://', 'https://')):
            url = f'https://{url}'
            logger.info("Added https:// prefix to URL: %s", url)
        
        instructions_text = f" with instructions: '{instructions}'" if instructions else ""
        logger.info("Crawling %s%s", url, instructions_text)
        
        # SSRF protection — block private/internal addresses
        if not is_safe_url(url):
            return json.dumps({"results": [{"url": url, "title": "", "content": "",
                "error": "Blocked: URL targets a private or internal network address"}]}, ensure_ascii=False)

        # Website policy check — block before crawling
        blocked = check_website_access(url)
        if blocked:
            logger.info("Blocked web_crawl for %s by rule %s", blocked["host"], blocked["rule"])
            return json.dumps({"results": [{"url": url, "title": "", "content": "", "error": blocked["message"],
                "blocked_by_policy": {"host": blocked["host"], "rule": blocked["rule"], "source": blocked["source"]}}]}, ensure_ascii=False)

        # Use Firecrawl's v2 crawl functionality
        # Docs: https://docs.firecrawl.dev/features/crawl
        # The crawl() method automatically waits for completion and returns all data
        
        # Build crawl parameters - keep it simple
        crawl_params = {
            "limit": 20,  # Limit number of pages to crawl
            "scrape_options": {
                "formats": ["markdown"]  # Just markdown for simplicity
            }
        }
        
        # Note: The 'prompt' parameter is not documented for crawl
        # Instructions are typically used with the Extract endpoint, not Crawl
        if instructions:
            logger.info("Instructions parameter ignored (not supported in crawl API)")
        
        from tools.interrupt import is_interrupted as _is_int
        if _is_int():
            return tool_error("Interrupted", success=False)

        try:
            crawl_result = _get_firecrawl_client().crawl(
                url=url,
                **crawl_params
            )
        except Exception as e:
            logger.debug("Crawl API call failed: %s", e)
            raise

        pages: List[Dict[str, Any]] = []
        
        # Process crawl results - the crawl method returns a CrawlJob object with data attribute
        data_list = []
        
        # The crawl_result is a CrawlJob object with a 'data' attribute containing list of Document objects
        if hasattr(crawl_result, 'data'):
            data_list = crawl_result.data if crawl_result.data else []
            logger.info("Status: %s", getattr(crawl_result, 'status', 'unknown'))
            logger.info("Retrieved %d pages", len(data_list))
            
            # Debug: Check other attributes if no data
            if not data_list:
                logger.debug("CrawlJob attributes: %s", [attr for attr in dir(crawl_result) if not attr.startswith('_')])
                logger.debug("Status: %s", getattr(crawl_result, 'status', 'N/A'))
                logger.debug("Total: %s", getattr(crawl_result, 'total', 'N/A'))
                logger.debug("Completed: %s", getattr(crawl_result, 'completed', 'N/A'))
                
        elif isinstance(crawl_result, dict) and 'data' in crawl_result:
            data_list = crawl_result.get("data", [])
        else:
            logger.warning("Unexpected crawl result type")
            logger.debug("Result type: %s", type(crawl_result))
            if hasattr(crawl_result, '__dict__'):
                logger.debug("Result attributes: %s", list(crawl_result.__dict__.keys()))
        
        for item in data_list:
            # Process each crawled page - properly handle object serialization
            page_url = "Unknown URL"
            title = ""
            content_markdown = None
            content_html = None
            metadata = {}
            
            # Extract data from the item
            if hasattr(item, 'model_dump'):
                # Pydantic model - use model_dump to get dict
                item_dict = item.model_dump()
                content_markdown = item_dict.get('markdown')
                content_html = item_dict.get('html')
                metadata = item_dict.get('metadata', {})
            elif hasattr(item, '__dict__'):
                # Regular object with attributes
                content_markdown = getattr(item, 'markdown', None)
                content_html = getattr(item, 'html', None)
                
                # Handle metadata - convert to dict if it's an object
                metadata_obj = getattr(item, 'metadata', {})
                if hasattr(metadata_obj, 'model_dump'):
                    metadata = metadata_obj.model_dump()
                elif hasattr(metadata_obj, '__dict__'):
                    metadata = metadata_obj.__dict__
                elif isinstance(metadata_obj, dict):
                    metadata = metadata_obj
                else:
                    metadata = {}
            elif isinstance(item, dict):
                # Already a dictionary
                content_markdown = item.get('markdown')
                content_html = item.get('html')
                metadata = item.get('metadata', {})
            
            # Ensure metadata is a dict (not an object)
            if not isinstance(metadata, dict):
                if hasattr(metadata, 'model_dump'):
                    metadata = metadata.model_dump()
                elif hasattr(metadata, '__dict__'):
                    metadata = metadata.__dict__
                else:
                    metadata = {}
            
            # Extract URL and title from metadata
            page_url = metadata.get("sourceURL", metadata.get("url", "Unknown URL"))
            title = metadata.get("title", "")
            
            # Re-check crawled page URL against policy
            page_blocked = check_website_access(page_url)
            if page_blocked:
                logger.info("Blocked crawled page %s by rule %s", page_blocked["host"], page_blocked["rule"])
                pages.append({
                    "url": page_url, "title": title, "content": "", "raw_content": "",
                    "error": page_blocked["message"],
                    "blocked_by_policy": {"host": page_blocked["host"], "rule": page_blocked["rule"], "source": page_blocked["source"]},
                })
                continue

            # Choose content (prefer markdown)
            content = content_markdown or content_html or ""
            
            pages.append({
                "url": page_url,
                "title": title,
                "content": content,
                "raw_content": content,
                "metadata": metadata  # Now guaranteed to be a dict
            })

        response = {"results": pages}
        
        pages_crawled = len(response.get('results', []))
        logger.info("Crawled %d pages", pages_crawled)
        
        debug_call_data["pages_crawled"] = pages_crawled
        debug_call_data["original_response_size"] = len(json.dumps(response))
        
        # Process each result with LLM if enabled
        if use_llm_processing and auxiliary_available:
            logger.info("Processing crawled content with LLM (parallel)...")
            debug_call_data["processing_applied"].append("llm_processing")
            
            # Prepare tasks for parallel processing
            async def process_single_crawl_result(result):
                """Process a single crawl result with LLM and return updated result with metrics."""
                page_url = result.get('url', 'Unknown URL')
                title = result.get('title', '')
                content = result.get('content', '')
                
                if not content:
                    return result, None, "no_content"
                
                original_size = len(content)
                
                # Process content with LLM
                processed = await process_content_with_llm(
                    content, page_url, title, effective_model, min_length
                )
                
                if processed:
                    processed_size = len(processed)
                    compression_ratio = processed_size / original_size if original_size > 0 else 1.0
                    
                    # Update result with processed content
                    result['raw_content'] = content
                    result['content'] = processed
                    
                    metrics = {
                        "url": page_url,
                        "original_size": original_size,
                        "processed_size": processed_size,
                        "compression_ratio": compression_ratio,
                        "model_used": effective_model
                    }
                    return result, metrics, "processed"
                else:
                    metrics = {
                        "url": page_url,
                        "original_size": original_size,
                        "processed_size": original_size,
                        "compression_ratio": 1.0,
                        "model_used": None,
                        "reason": "content_too_short"
                    }
                    return result, metrics, "too_short"
            
            # Run all LLM processing in parallel
            results_list = response.get('results', [])
            tasks = [process_single_crawl_result(result) for result in results_list]
            processed_results = await asyncio.gather(*tasks)
            
            # Collect metrics and print results
            for result, metrics, status in processed_results:
                page_url = result.get('url', 'Unknown URL')
                if status == "processed":
                    debug_call_data["compression_metrics"].append(metrics)
                    debug_call_data["pages_processed_with_llm"] += 1
                    logger.info("%s (processed)", page_url)
                elif status == "too_short":
                    debug_call_data["compression_metrics"].append(metrics)
                    logger.info("%s (no processing - content too short)", page_url)
                else:
                    logger.warning("%s (no content to process)", page_url)
        else:
            if use_llm_processing and not auxiliary_available:
                logger.warning("LLM processing requested but no auxiliary model available, returning raw content")
                debug_call_data["processing_applied"].append("llm_processing_unavailable")
            # Print summary of crawled pages for debugging (original behavior)
            for result in response.get('results', []):
                page_url = result.get('url', 'Unknown URL')
                content_length = len(result.get('content', ''))
                logger.info("%s (%d characters)", page_url, content_length)
        
        # Trim output to minimal fields per entry: title, content, error
        trimmed_results = [
            {
                "url": r.get("url", ""),
                "title": r.get("title", ""),
                "content": r.get("content", ""),
                "error": r.get("error"),
                **({  "blocked_by_policy": r["blocked_by_policy"]} if "blocked_by_policy" in r else {}),
            }
            for r in response.get("results", [])
        ]
        trimmed_response = {"results": trimmed_results}
        
        result_json = json.dumps(trimmed_response, indent=2, ensure_ascii=False)
        # Clean base64 images from crawled content
        cleaned_result = clean_base64_images(result_json)
        
        debug_call_data["final_response_size"] = len(cleaned_result)
        debug_call_data["processing_applied"].append("base64_image_removal")
        
        # Log debug information
        _debug.log_call("web_crawl_tool", debug_call_data)
        _debug.save()
        
        return cleaned_result
        
    except Exception as e:
        error_msg = f"Error crawling website: {str(e)}"
        logger.debug("%s", error_msg)
        
        debug_call_data["error"] = error_msg
        _debug.log_call("web_crawl_tool", debug_call_data)
        _debug.save()
        
        return tool_error(error_msg)


# Convenience function to check Firecrawl credentials
def check_firecrawl_api_key() -> bool:
    """
    Check whether the Firecrawl backend is available.

    Availability is true when either:
    1) direct Firecrawl config (`FIRECRAWL_API_KEY` or `FIRECRAWL_API_URL`), or
    2) Firecrawl gateway origin + Nous Subscriber access token
       (fallback when direct Firecrawl is not configured).

    Returns:
        bool: True if direct Firecrawl or the tool-gateway can be used.
    """
    return _has_direct_firecrawl_config() or _is_tool_gateway_ready()


def check_web_api_key() -> bool:
    """Check whether the configured web backend is available."""
    configured = _load_web_config().get("backend", "").lower().strip()
    if configured in ("exa", "parallel", "firecrawl", "tavily"):
        return _is_backend_available(configured)
    return any(_is_backend_available(backend) for backend in ("exa", "parallel", "firecrawl", "tavily"))


def check_auxiliary_model() -> bool:
    """Check if an auxiliary text model is available for LLM content processing."""
    client, _, _ = _resolve_web_extract_auxiliary()
    return client is not None




if __name__ == "__main__":
    """
    Simple test/demo when run directly
    """
    print("🌐 Standalone Web Tools Module")
    print("=" * 40)
    
    # Check if API keys are available
    web_available = check_web_api_key()
    tool_gateway_available = _is_tool_gateway_ready()
    firecrawl_key_available = bool(os.getenv("FIRECRAWL_API_KEY", "").strip())
    firecrawl_url_available = bool(os.getenv("FIRECRAWL_API_URL", "").strip())
    nous_available = check_auxiliary_model()
    default_summarizer_model = _get_default_summarizer_model()

    if web_available:
        backend = _get_backend()
        print(f"✅ Web backend: {backend}")
        if backend == "exa":
            print("   Using Exa API (https://exa.ai)")
        elif backend == "parallel":
            print("   Using Parallel API (https://parallel.ai)")
        elif backend == "tavily":
            print("   Using Tavily API (https://tavily.com)")
        else:
            if firecrawl_url_available:
                print(f"   Using self-hosted Firecrawl: {os.getenv('FIRECRAWL_API_URL').strip().rstrip('/')}")
            elif firecrawl_key_available:
                print("   Using direct Firecrawl cloud API")
            elif tool_gateway_available:
                print(f"   Using Firecrawl tool-gateway: {_get_firecrawl_gateway_url()}")
            else:
                print("   Firecrawl backend selected but not configured")
    else:
        print("❌ No web search backend configured")
        print(
            "Set EXA_API_KEY, PARALLEL_API_KEY, TAVILY_API_KEY, FIRECRAWL_API_KEY, FIRECRAWL_API_URL"
            f"{_firecrawl_backend_help_suffix()}"
        )

    if not nous_available:
        print("❌ No auxiliary model available for LLM content processing")
        print("Set OPENROUTER_API_KEY, configure Nous Portal, or set OPENAI_BASE_URL + OPENAI_API_KEY")
        print("⚠️  Without an auxiliary model, LLM content processing will be disabled")
    else:
        print(f"✅ Auxiliary model available: {default_summarizer_model}")

    if not web_available:
        exit(1)

    print("🛠️  Web tools ready for use!")
    
    if nous_available:
        print(f"🧠 LLM content processing available with {default_summarizer_model}")
        print(f"   Default min length for processing: {DEFAULT_MIN_LENGTH_FOR_SUMMARIZATION} chars")
    
    # Show debug mode status
    if _debug.active:
        print(f"🐛 Debug mode ENABLED - Session ID: {_debug.session_id}")
        print(f"   Debug logs will be saved to: {_debug.log_dir}/web_tools_debug_{_debug.session_id}.json")
    else:
        print("🐛 Debug mode disabled (set WEB_TOOLS_DEBUG=true to enable)")
    
    print("\nBasic usage:")
    print("  from web_tools import web_search_tool, web_extract_tool, web_crawl_tool")
    print("  import asyncio")
    print("")
    print("  # Search (synchronous)")
    print("  results = web_search_tool('Python tutorials')")
    print("")
    print("  # Extract and crawl (asynchronous)")
    print("  async def main():")
    print("      content = await web_extract_tool(['https://example.com'])")
    print("      crawl_data = await web_crawl_tool('example.com', 'Find docs')")
    print("  asyncio.run(main())")
    
    if nous_available:
        print("\nLLM-enhanced usage:")
        print("  # Content automatically processed for pages >5000 chars (default)")
        print("  content = await web_extract_tool(['https://python.org/about/'])")
        print("")
        print("  # Customize processing parameters")
        print("  crawl_data = await web_crawl_tool(")
        print("      'docs.python.org',")
        print("      'Find key concepts',")
        print("      model='google/gemini-3-flash-preview',")
        print("      min_length=3000")
        print("  )")
        print("")
        print("  # Disable LLM processing")
        print("  raw_content = await web_extract_tool(['https://example.com'], use_llm_processing=False)")
    
    print("\nDebug mode:")
    print("  # Enable debug logging")
    print("  export WEB_TOOLS_DEBUG=true")
    print("  # Debug logs capture:")
    print("  # - All tool calls with parameters")
    print("  # - Original API responses")
    print("  # - LLM compression metrics")
    print("  # - Final processed results")
    print("  # Logs saved to: ./logs/web_tools_debug_UUID.json")
    
    print("\n📝 Run 'python test_web_tools_llm.py' to test LLM processing capabilities")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
from tools.registry import registry, tool_error

WEB_SEARCH_SCHEMA = {
    "name": "web_search",
    "description": "Search the web for information. Returns up to 5 results by default with titles, URLs, and descriptions. The query is passed through to the configured backend, so operators such as site:domain, filetype:pdf, intitle:word, -term, and \"exact phrase\" may work when the backend supports them.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query to look up on the web. You may include backend-supported operators such as site:example.com, filetype:pdf, intitle:word, -term, or \"exact phrase\"."
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of results to return. Defaults to 5.",
                "minimum": 1,
                "maximum": 100,
                "default": 5
            }
        },
        "required": ["query"]
    }
}

WEB_EXTRACT_SCHEMA = {
    "name": "web_extract",
    "description": "Extract content from web page URLs. Returns page content in markdown format. Also works with PDF URLs (arxiv papers, documents, etc.) — pass the PDF link directly and it converts to markdown text. Pages under 5000 chars return full markdown; larger pages are LLM-summarized and capped at ~5000 chars per page. Pages over 2M chars are refused. If a URL fails or times out, use the browser tool to access it instead.",
    "parameters": {
        "type": "object",
        "properties": {
            "urls": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of URLs to extract content from (max 5 URLs per call)",
                "maxItems": 5
            }
        },
        "required": ["urls"]
    }
}

registry.register(
    name="web_search",
    toolset="web",
    schema=WEB_SEARCH_SCHEMA,
    handler=lambda args, **kw: web_search_tool(args.get("query", ""), limit=args.get("limit", 5)),
    check_fn=check_web_api_key,
    requires_env=_web_requires_env(),
    emoji="🔍",
    max_result_size_chars=100_000,
)
registry.register(
    name="web_extract",
    toolset="web",
    schema=WEB_EXTRACT_SCHEMA,
    handler=lambda args, **kw: web_extract_tool(
        args.get("urls", [])[:5] if isinstance(args.get("urls"), list) else [], "markdown"),
    check_fn=check_web_api_key,
    requires_env=_web_requires_env(),
    is_async=True,
    emoji="📄",
    max_result_size_chars=100_000,
)
