#!/usr/bin/env python3
"""
Standalone Web Tools Module

This module provides generic web tools that work with multiple backend providers.
Currently uses Firecrawl as the backend, and the interface makes it easy to swap
providers without changing the function signatures.

Available tools:
- web_search_tool: Search the web for information
- web_extract_tool: Extract content from specific web pages
- web_crawl_tool: Crawl websites with specific instructions

Backend compatibility:
- Firecrawl: https://docs.firecrawl.dev/introduction

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

#TODO: Search Capabilities over the scraped pages
#TODO: Store the pages in something
#TODO: Tool to see what pages are available/saved to search over

import json
import os
import re
import asyncio
import uuid
import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from firecrawl import Firecrawl
from openai import AsyncOpenAI

# Initialize Firecrawl client once at module level
firecrawl_client = Firecrawl(api_key=os.getenv("FIRECRAWL_API_KEY"))

# Initialize OpenRouter API client for LLM processing (async)
summarizer_client = AsyncOpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

# Configuration for LLM processing
DEFAULT_SUMMARIZER_MODEL = "google/gemini-3-flash-preview"
DEFAULT_MIN_LENGTH_FOR_SUMMARIZATION = 5000

# Debug mode configuration
DEBUG_MODE = os.getenv("WEB_TOOLS_DEBUG", "false").lower() == "true"
DEBUG_SESSION_ID = str(uuid.uuid4())
DEBUG_LOG_PATH = Path("./logs")
DEBUG_DATA = {
    "session_id": DEBUG_SESSION_ID,
    "start_time": datetime.datetime.now().isoformat(),
    "debug_enabled": DEBUG_MODE,
    "tool_calls": []
} if DEBUG_MODE else None

# Create logs directory if debug mode is enabled
if DEBUG_MODE:
    DEBUG_LOG_PATH.mkdir(exist_ok=True)
    print(f"🐛 Debug mode enabled - Session ID: {DEBUG_SESSION_ID}")


def _log_debug_call(tool_name: str, call_data: Dict[str, Any]) -> None:
    """
    Log a debug call entry to the global debug data structure.
    
    Args:
        tool_name (str): Name of the tool being called
        call_data (Dict[str, Any]): Data about the call including parameters and results
    """
    if not DEBUG_MODE or not DEBUG_DATA:
        return
    
    call_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "tool_name": tool_name,
        **call_data
    }
    
    DEBUG_DATA["tool_calls"].append(call_entry)


def _save_debug_log() -> None:
    """
    Save the current debug data to a JSON file in the logs directory.
    """
    if not DEBUG_MODE or not DEBUG_DATA:
        return
    
    try:
        debug_filename = f"web_tools_debug_{DEBUG_SESSION_ID}.json"
        debug_filepath = DEBUG_LOG_PATH / debug_filename
        
        # Update end time
        DEBUG_DATA["end_time"] = datetime.datetime.now().isoformat()
        DEBUG_DATA["total_calls"] = len(DEBUG_DATA["tool_calls"])
        
        with open(debug_filepath, 'w', encoding='utf-8') as f:
            json.dump(DEBUG_DATA, f, indent=2, ensure_ascii=False)
        
        print(f"🐛 Debug log saved: {debug_filepath}")
        
    except Exception as e:
        print(f"❌ Error saving debug log: {str(e)}")


async def process_content_with_llm(
    content: str, 
    url: str = "", 
    title: str = "",
    model: str = DEFAULT_SUMMARIZER_MODEL,
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
            print(f"🚫 Content too large ({size_mb:.1f}MB > 2MB limit). Refusing to process.")
            return f"[Content too large to process: {size_mb:.1f}MB. Try using web_crawl with specific extraction instructions, or search for a more focused source.]"
        
        # Skip processing if content is too short
        if content_len < min_length:
            print(f"📏 Content too short ({content_len} < {min_length} chars), skipping LLM processing")
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
            print(f"📦 Content large ({content_len:,} chars). Using chunked processing...")
            return await _process_large_content_chunked(
                content, context_str, model, CHUNK_SIZE, MAX_OUTPUT_SIZE
            )
        
        # Standard single-pass processing for normal content
        print(f"🧠 Processing content with LLM ({content_len} characters)")
        
        processed_content = await _call_summarizer_llm(content, context_str, model)
        
        if processed_content:
            # Enforce output cap
            if len(processed_content) > MAX_OUTPUT_SIZE:
                processed_content = processed_content[:MAX_OUTPUT_SIZE] + "\n\n[... summary truncated for context management ...]"
            
            # Log compression metrics
            processed_length = len(processed_content)
            compression_ratio = processed_length / content_len if content_len > 0 else 1.0
            print(f"✅ Content processed: {content_len} → {processed_length} chars ({compression_ratio:.1%})")
        
        return processed_content
        
    except Exception as e:
        print(f"❌ Error processing content with LLM: {str(e)}")
        return f"[Failed to process content: {str(e)[:100]}. Content size: {len(content):,} chars]"


async def _call_summarizer_llm(
    content: str, 
    context_str: str, 
    model: str, 
    max_tokens: int = 4000,
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

    # Call the LLM with retry logic
    max_retries = 6
    retry_delay = 2
    last_error = None

    for attempt in range(max_retries):
        try:
            response = await summarizer_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content.strip()
        except Exception as api_error:
            last_error = api_error
            if attempt < max_retries - 1:
                print(f"⚠️  LLM API call failed (attempt {attempt + 1}/{max_retries}): {str(api_error)[:100]}")
                print(f"   Retrying in {retry_delay}s...")
                await asyncio.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, 60)
            else:
                raise last_error
    
    return None


async def _process_large_content_chunked(
    content: str, 
    context_str: str, 
    model: str, 
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
    
    print(f"   📦 Split into {len(chunks)} chunks of ~{chunk_size:,} chars each")
    
    # Summarize each chunk in parallel
    async def summarize_chunk(chunk_idx: int, chunk_content: str) -> tuple[int, Optional[str]]:
        """Summarize a single chunk."""
        try:
            chunk_info = f"[Processing chunk {chunk_idx + 1} of {len(chunks)}]"
            summary = await _call_summarizer_llm(
                chunk_content, 
                context_str, 
                model, 
                max_tokens=2000,
                is_chunk=True,
                chunk_info=chunk_info
            )
            if summary:
                print(f"   ✅ Chunk {chunk_idx + 1}/{len(chunks)} summarized: {len(chunk_content):,} → {len(summary):,} chars")
            return chunk_idx, summary
        except Exception as e:
            print(f"   ⚠️  Chunk {chunk_idx + 1}/{len(chunks)} failed: {str(e)[:50]}")
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
        print(f"   ❌ All chunk summarizations failed")
        return "[Failed to process large content: all chunk summarizations failed]"
    
    print(f"   📊 Got {len(summaries)}/{len(chunks)} chunk summaries")
    
    # If only one chunk succeeded, just return it (with cap)
    if len(summaries) == 1:
        result = summaries[0]
        if len(result) > max_output_size:
            result = result[:max_output_size] + "\n\n[... truncated ...]"
        return result
    
    # Synthesize the summaries into a final summary
    print(f"   🔗 Synthesizing {len(summaries)} summaries...")
    
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
        response = await summarizer_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You synthesize multiple summaries into one cohesive, comprehensive summary. Be thorough but concise."},
                {"role": "user", "content": synthesis_prompt}
            ],
            temperature=0.1,
            max_tokens=4000
        )
        final_summary = response.choices[0].message.content.strip()
        
        # Enforce hard cap
        if len(final_summary) > max_output_size:
            final_summary = final_summary[:max_output_size] + "\n\n[... summary truncated for context management ...]"
        
        original_len = len(content)
        final_len = len(final_summary)
        compression = final_len / original_len if original_len > 0 else 1.0
        
        print(f"   ✅ Synthesis complete: {original_len:,} → {final_len:,} chars ({compression:.2%})")
        return final_summary
        
    except Exception as e:
        print(f"   ⚠️  Synthesis failed: {str(e)[:100]}")
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


def web_search_tool(query: str, limit: int = 5) -> str:
    """
    Search the web for information using available search API backend.
    
    This function provides a generic interface for web search that can work
    with multiple backends. Currently uses Firecrawl.
    
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
        print(f"🔍 Searching the web for: '{query}' (limit: {limit})")
        
        # Use Firecrawl's v2 search functionality WITHOUT scraping
        # We only want search result metadata, not scraped content
        # Docs: https://docs.firecrawl.dev/features/search
        response = firecrawl_client.search(
            query=query,
            limit=limit
        )
        
        # The response is a SearchData object with web, news, and images attributes
        # When not scraping, the results are directly in these attributes
        web_results = []
        
        # Check if response has web attribute (SearchData object)
        if hasattr(response, 'web'):
            # Response is a SearchData object with web attribute
            if response.web:
                # Convert each SearchResultWeb object to dict
                for result in response.web:
                    if hasattr(result, 'model_dump'):
                        # Pydantic model - use model_dump
                        web_results.append(result.model_dump())
                    elif hasattr(result, '__dict__'):
                        # Regular object - use __dict__
                        web_results.append(result.__dict__)
                    elif isinstance(result, dict):
                        # Already a dict
                        web_results.append(result)
        elif hasattr(response, 'model_dump'):
            # Response has model_dump method - use it to get dict
            response_dict = response.model_dump()
            if 'web' in response_dict and response_dict['web']:
                web_results = response_dict['web']
        elif isinstance(response, dict):
            # Response is already a dictionary
            if 'web' in response and response['web']:
                web_results = response['web']
        
        results_count = len(web_results)
        print(f"✅ Found {results_count} search results")
        
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
        _log_debug_call("web_search_tool", debug_call_data)
        _save_debug_log()
        
        return result_json
        
    except Exception as e:
        error_msg = f"Error searching web: {str(e)}"
        print(f"❌ {error_msg}")
        
        debug_call_data["error"] = error_msg
        _log_debug_call("web_search_tool", debug_call_data)
        _save_debug_log()
        
        return json.dumps({"error": error_msg}, ensure_ascii=False)


async def web_extract_tool(
    urls: List[str], 
    format: str = None, 
    use_llm_processing: bool = True,
    model: str = DEFAULT_SUMMARIZER_MODEL,
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
        model (str): The model to use for LLM processing (default: google/gemini-3-flash-preview)
        min_length (int): Minimum content length to trigger LLM processing (default: 5000)
    
    Returns:
        str: JSON string containing extracted content. If LLM processing is enabled and successful,
             the 'content' field will contain the processed markdown summary instead of raw content.
    
    Raises:
        Exception: If extraction fails or API key is not set
    """
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
        print(f"📄 Extracting content from {len(urls)} URL(s)")
        
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
        
        for url in urls:
            try:
                print(f"  📄 Scraping: {url}")
                scrape_result = firecrawl_client.scrape(
                    url=url,
                    formats=formats
                )
                
                # Process the result - properly handle object serialization
                metadata = {}
                title = ""
                content_markdown = None
                content_html = None
                
                # Extract data from the scrape result
                if hasattr(scrape_result, 'model_dump'):
                    # Pydantic model - use model_dump to get dict
                    result_dict = scrape_result.model_dump()
                    content_markdown = result_dict.get('markdown')
                    content_html = result_dict.get('html')
                    metadata = result_dict.get('metadata', {})
                elif hasattr(scrape_result, '__dict__'):
                    # Regular object with attributes
                    content_markdown = getattr(scrape_result, 'markdown', None)
                    content_html = getattr(scrape_result, 'html', None)
                    
                    # Handle metadata - convert to dict if it's an object
                    metadata_obj = getattr(scrape_result, 'metadata', {})
                    if hasattr(metadata_obj, 'model_dump'):
                        metadata = metadata_obj.model_dump()
                    elif hasattr(metadata_obj, '__dict__'):
                        metadata = metadata_obj.__dict__
                    elif isinstance(metadata_obj, dict):
                        metadata = metadata_obj
                    else:
                        metadata = {}
                elif isinstance(scrape_result, dict):
                    # Already a dictionary
                    content_markdown = scrape_result.get('markdown')
                    content_html = scrape_result.get('html')
                    metadata = scrape_result.get('metadata', {})
                
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
                
                # Choose content based on requested format
                chosen_content = content_markdown if (format == "markdown" or (format is None and content_markdown)) else content_html or content_markdown or ""
                
                results.append({
                    "url": metadata.get("sourceURL", url),
                    "title": title,
                    "content": chosen_content,
                    "raw_content": chosen_content,
                    "metadata": metadata  # Now guaranteed to be a dict
                })
                
            except Exception as scrape_err:
                print(f"  ❌ Error scraping {url}: {str(scrape_err)}")
                results.append({
                    "url": url,
                    "title": "",
                    "content": "",
                    "raw_content": "",
                    "error": str(scrape_err)
                })

        response = {"results": results}
        
        pages_extracted = len(response.get('results', []))
        print(f"✅ Extracted content from {pages_extracted} pages")
        
        debug_call_data["pages_extracted"] = pages_extracted
        debug_call_data["original_response_size"] = len(json.dumps(response))
        
        # Process each result with LLM if enabled
        if use_llm_processing and os.getenv("OPENROUTER_API_KEY"):
            print("🧠 Processing extracted content with LLM (parallel)...")
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
                    raw_content, url, title, model, min_length
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
                        "model_used": model
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
                    print(f"  📝 {url} (processed)")
                elif status == "too_short":
                    debug_call_data["compression_metrics"].append(metrics)
                    print(f"  📝 {url} (no processing - content too short)")
                else:
                    print(f"  ⚠️  {url} (no content to process)")
        else:
            if use_llm_processing and not os.getenv("OPENROUTER_API_KEY"):
                print("⚠️  LLM processing requested but OPENROUTER_API_KEY not set, returning raw content")
                debug_call_data["processing_applied"].append("llm_processing_unavailable")
            
            # Print summary of extracted pages for debugging (original behavior)
            for result in response.get('results', []):
                url = result.get('url', 'Unknown URL')
                content_length = len(result.get('raw_content', ''))
                print(f"  📝 {url} ({content_length} characters)")
        
        # Trim output to minimal fields per entry: title, content, error
        trimmed_results = [
            {
                "title": r.get("title", ""),
                "content": r.get("content", ""),
                "error": r.get("error"),
            }
            for r in response.get("results", [])
        ]
        trimmed_response = {"results": trimmed_results}

        if trimmed_response.get("results") == []:
            result_json = json.dumps({"error": "Content was inaccessible or not found"}, ensure_ascii=False)

            cleaned_result = clean_base64_images(result_json)
        
        else:
            result_json = json.dumps(trimmed_response, indent=2, ensure_ascii=False)
            
            cleaned_result = clean_base64_images(result_json)
        
        debug_call_data["final_response_size"] = len(cleaned_result)
        debug_call_data["processing_applied"].append("base64_image_removal")
        
        # Log debug information
        _log_debug_call("web_extract_tool", debug_call_data)
        _save_debug_log()
        
        return cleaned_result
            
    except Exception as e:
        error_msg = f"Error extracting content: {str(e)}"
        print(f"❌ {error_msg}")
        
        debug_call_data["error"] = error_msg
        _log_debug_call("web_extract_tool", debug_call_data)
        _save_debug_log()
        
        return json.dumps({"error": error_msg}, ensure_ascii=False)


async def web_crawl_tool(
    url: str, 
    instructions: str = None, 
    depth: str = "basic", 
    use_llm_processing: bool = True,
    model: str = DEFAULT_SUMMARIZER_MODEL,
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
        model (str): The model to use for LLM processing (default: google/gemini-3-flash-preview)
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
        # Ensure URL has protocol
        if not url.startswith(('http://', 'https://')):
            url = f'https://{url}'
            print(f"  📝 Added https:// prefix to URL: {url}")
        
        instructions_text = f" with instructions: '{instructions}'" if instructions else ""
        print(f"🕷️ Crawling {url}{instructions_text}")
        
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
            print(f"  ℹ️  Note: Instructions parameter ignored (not supported in crawl API)")
        
        # Use the crawl method which waits for completion automatically
        try:
            crawl_result = firecrawl_client.crawl(
                url=url,
                **crawl_params
            )
        except Exception as e:
            print(f"  ❌ Crawl API call failed: {e}")
            raise

        pages: List[Dict[str, Any]] = []
        
        # Process crawl results - the crawl method returns a CrawlJob object with data attribute
        data_list = []
        
        # The crawl_result is a CrawlJob object with a 'data' attribute containing list of Document objects
        if hasattr(crawl_result, 'data'):
            data_list = crawl_result.data if crawl_result.data else []
            print(f"  📊 Status: {getattr(crawl_result, 'status', 'unknown')}")
            print(f"  📄 Retrieved {len(data_list)} pages")
            
            # Debug: Check other attributes if no data
            if not data_list:
                print(f"  🔍 Debug - CrawlJob attributes: {[attr for attr in dir(crawl_result) if not attr.startswith('_')]}")
                print(f"  🔍 Debug - Status: {getattr(crawl_result, 'status', 'N/A')}")
                print(f"  🔍 Debug - Total: {getattr(crawl_result, 'total', 'N/A')}")
                print(f"  🔍 Debug - Completed: {getattr(crawl_result, 'completed', 'N/A')}")
                
        elif isinstance(crawl_result, dict) and 'data' in crawl_result:
            data_list = crawl_result.get("data", [])
        else:
            print("  ⚠️  Unexpected crawl result type")
            print(f"  🔍 Debug - Result type: {type(crawl_result)}")
            if hasattr(crawl_result, '__dict__'):
                print(f"  🔍 Debug - Result attributes: {list(crawl_result.__dict__.keys())}")
        
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
        print(f"✅ Crawled {pages_crawled} pages")
        
        debug_call_data["pages_crawled"] = pages_crawled
        debug_call_data["original_response_size"] = len(json.dumps(response))
        
        # Process each result with LLM if enabled
        if use_llm_processing and os.getenv("OPENROUTER_API_KEY"):
            print("🧠 Processing crawled content with LLM (parallel)...")
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
                    content, page_url, title, model, min_length
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
                        "model_used": model
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
                    print(f"  🌐 {page_url} (processed)")
                elif status == "too_short":
                    debug_call_data["compression_metrics"].append(metrics)
                    print(f"  🌐 {page_url} (no processing - content too short)")
                else:
                    print(f"  ⚠️  {page_url} (no content to process)")
        else:
            if use_llm_processing and not os.getenv("OPENROUTER_API_KEY"):
                print("⚠️  LLM processing requested but OPENROUTER_API_KEY not set, returning raw content")
                debug_call_data["processing_applied"].append("llm_processing_unavailable")
            
            # Print summary of crawled pages for debugging (original behavior)
            for result in response.get('results', []):
                page_url = result.get('url', 'Unknown URL')
                content_length = len(result.get('content', ''))
                print(f"  🌐 {page_url} ({content_length} characters)")
        
        # Trim output to minimal fields per entry: title, content, error
        trimmed_results = [
            {
                "title": r.get("title", ""),
                "content": r.get("content", ""),
                "error": r.get("error")
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
        _log_debug_call("web_crawl_tool", debug_call_data)
        _save_debug_log()
        
        return cleaned_result
        
    except Exception as e:
        error_msg = f"Error crawling website: {str(e)}"
        print(f"❌ {error_msg}")
        
        debug_call_data["error"] = error_msg
        _log_debug_call("web_crawl_tool", debug_call_data)
        _save_debug_log()
        
        return json.dumps({"error": error_msg}, ensure_ascii=False)


# Convenience function to check if API key is available
def check_firecrawl_api_key() -> bool:
    """
    Check if the Firecrawl API key is available in environment variables.
    
    Returns:
        bool: True if API key is set, False otherwise
    """
    return bool(os.getenv("FIRECRAWL_API_KEY"))


def check_nous_api_key() -> bool:
    """
    Check if the Nous Research API key is available in environment variables.
    
    Returns:
        bool: True if API key is set, False otherwise
    """
    return bool(os.getenv("OPENROUTER_API_KEY"))


def get_debug_session_info() -> Dict[str, Any]:
    """
    Get information about the current debug session.
    
    Returns:
        Dict[str, Any]: Dictionary containing debug session information:
                       - enabled: Whether debug mode is enabled
                       - session_id: Current session UUID (if enabled)
                       - log_path: Path where debug logs are saved (if enabled)
                       - total_calls: Number of tool calls logged so far (if enabled)
    """
    if not DEBUG_MODE or not DEBUG_DATA:
        return {
            "enabled": False,
            "session_id": None,
            "log_path": None,
            "total_calls": 0
        }
    
    return {
        "enabled": True,
        "session_id": DEBUG_SESSION_ID,
        "log_path": str(DEBUG_LOG_PATH / f"web_tools_debug_{DEBUG_SESSION_ID}.json"),
        "total_calls": len(DEBUG_DATA["tool_calls"])
    }


if __name__ == "__main__":
    """
    Simple test/demo when run directly
    """
    print("🌐 Standalone Web Tools Module")
    print("=" * 40)
    
    # Check if API keys are available
    firecrawl_available = check_firecrawl_api_key()
    nous_available = check_nous_api_key()
    
    if not firecrawl_available:
        print("❌ FIRECRAWL_API_KEY environment variable not set")
        print("Please set your API key: export FIRECRAWL_API_KEY='your-key-here'")
        print("Get API key at: https://firecrawl.dev/")
    else:
        print("✅ Firecrawl API key found")
    
    if not nous_available:
        print("❌ OPENROUTER_API_KEY environment variable not set")
        print("Please set your API key: export OPENROUTER_API_KEY='your-key-here'")  
        print("Get API key at: https://inference-api.nousresearch.com/")
        print("⚠️  Without Nous API key, LLM content processing will be disabled")
    else:
        print("✅ Nous Research API key found")
    
    if not firecrawl_available:
        exit(1)
    
    print("🛠️  Web tools ready for use!")
    
    if nous_available:
        print("🧠 LLM content processing available with Gemini 3 Flash Preview via OpenRouter")
        print(f"   Default min length for processing: {DEFAULT_MIN_LENGTH_FOR_SUMMARIZATION} chars")
    
    # Show debug mode status
    if DEBUG_MODE:
        print(f"🐛 Debug mode ENABLED - Session ID: {DEBUG_SESSION_ID}")
        print(f"   Debug logs will be saved to: ./logs/web_tools_debug_{DEBUG_SESSION_ID}.json")
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
    
    print(f"\n📝 Run 'python test_web_tools_llm.py' to test LLM processing capabilities")
