#!/usr/bin/env python3
"""
Standalone Web Tools Module

This module provides generic web tools that work with multiple backend providers.
Currently uses Tavily as the backend, but the interface makes it easy to swap
to other providers like Firecrawl without changing the function signatures.

Available tools:
- web_search_tool: Search the web for information
- web_extract_tool: Extract content from specific web pages
- web_crawl_tool: Crawl websites with specific instructions

Backend compatibility:
- Tavily: https://docs.tavily.com/
- Firecrawl: https://docs.firecrawl.dev/features/search

LLM Processing:
- Uses Nous Research API with Gemini 2.5 Flash for intelligent content extraction
- Extracts key excerpts and creates markdown summaries to reduce token usage

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
from typing import List, Dict, Any, Optional
from tavily import TavilyClient
from openai import AsyncOpenAI

# Initialize Tavily client once at module level
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

# Initialize Nous Research API client for LLM processing (async)
nous_client = AsyncOpenAI(
    api_key=os.getenv("NOUS_API_KEY"),
    base_url="https://inference-api.nousresearch.com/v1"
)

# Configuration for LLM processing
DEFAULT_SUMMARIZER_MODEL = "gemini-2.5-flash"
DEFAULT_MIN_LENGTH_FOR_SUMMARIZATION = 5000


async def process_content_with_llm(
    content: str, 
    url: str = "", 
    title: str = "",
    model: str = DEFAULT_SUMMARIZER_MODEL,
    min_length: int = DEFAULT_MIN_LENGTH_FOR_SUMMARIZATION
) -> Optional[str]:
    """
    Process web content using LLM to create intelligent summaries with key excerpts.
    
    This function uses Gemini 2.5 Flash (or specified model) via Nous Research API 
    to intelligently extract key information and create markdown summaries,
    significantly reducing token usage while preserving all important information.
    
    Args:
        content (str): The raw content to process
        url (str): The source URL (for context, optional)
        title (str): The page title (for context, optional)
        model (str): The model to use for processing (default: gemini-2.5-flash)
        min_length (int): Minimum content length to trigger processing (default: 5000)
        
    Returns:
        Optional[str]: Processed markdown content, or None if content too short or processing fails
    """
    try:
        # Skip processing if content is too short
        if len(content) < min_length:
            print(f"📏 Content too short ({len(content)} < {min_length} chars), skipping LLM processing")
            return None
        
        print(f"🧠 Processing content with LLM ({len(content)} characters)")
        
        # Create context information
        context_info = []
        if title:
            context_info.append(f"Title: {title}")
        if url:
            context_info.append(f"Source: {url}")
        
        context_str = "\n".join(context_info) + "\n\n" if context_info else ""
        
        # Simplified prompt for better quality markdown output
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

        # Call the LLM asynchronously
        response = await nous_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,  # Low temperature for consistent extraction
            max_tokens=4000   # Generous limit for comprehensive processing
        )
        
        # Get the markdown response directly
        processed_content = response.choices[0].message.content.strip()
        
        # Calculate compression metrics for logging
        original_length = len(content)
        processed_length = len(processed_content)
        compression_ratio = processed_length / original_length if original_length > 0 else 1.0
        
        print(f"✅ Content processed: {original_length} → {processed_length} chars ({compression_ratio:.1%})")
        
        return processed_content
        
    except Exception as e:
        print(f"❌ Error processing content with LLM: {str(e)}")
        return None


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
    with multiple backends. Currently uses Tavily but can be easily swapped.
    
    Note: Search results are already concise snippets, so no LLM processing is applied.
    
    Args:
        query (str): The search query to look up
        limit (int): Maximum number of results to return (default: 5)
    
    Returns:
        str: JSON string containing search results with the following structure:
             {
                 "query": str,
                 "results": [
                     {
                         "title": str,
                         "url": str,
                         "content": str,
                         "score": float
                     },
                     ...
                 ]
             }
    
    Raises:
        Exception: If search fails or API key is not set
    """
    try:
        print(f"🔍 Searching the web for: '{query}' (limit: {limit})")
        
        # Use Tavily's search functionality
        response = tavily_client.search(query=query, max_results=limit, search_depth="advanced")
        
        print(f"✅ Found {len(response.get('results', []))} results")
        
        result_json = json.dumps(response, indent=2)
        # Clean base64 images from search results
        return clean_base64_images(result_json)
        
    except Exception as e:
        error_msg = f"Error searching web: {str(e)}"
        print(f"❌ {error_msg}")
        return json.dumps({"error": error_msg})


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
    can work with multiple backends. Currently uses Tavily but can be easily swapped.
    
    Args:
        urls (List[str]): List of URLs to extract content from
        format (str): Desired output format ("markdown" or "html", optional)
        use_llm_processing (bool): Whether to process content with LLM for summarization (default: True)
        model (str): The model to use for LLM processing (default: gemini-2.5-flash)
        min_length (int): Minimum content length to trigger LLM processing (default: 5000)
    
    Returns:
        str: JSON string containing extracted content. If LLM processing is enabled and successful,
             the 'content' field will contain the processed markdown summary instead of raw content.
    
    Raises:
        Exception: If extraction fails or API key is not set
    """
    try:
        print(f"📄 Extracting content from {len(urls)} URL(s)")
        
        # Use Tavily's extract functionality
        response = tavily_client.extract(urls=urls, format=format)
        
        print(f"✅ Extracted content from {len(response.get('results', []))} pages")
        
        # Process each result with LLM if enabled
        if use_llm_processing and os.getenv("NOUS_API_KEY"):
            print("🧠 Processing extracted content with LLM...")
            
            for result in response.get('results', []):
                url = result.get('url', 'Unknown URL')
                title = result.get('title', '')
                raw_content = result.get('raw_content', '') or result.get('content', '')
                
                if raw_content:
                    # Process content with LLM
                    processed = await process_content_with_llm(
                        raw_content, url, title, model, min_length
                    )
                    
                    if processed:
                        # Replace content with processed version
                        result['content'] = processed
                        # Keep raw content in separate field for reference
                        result['raw_content'] = raw_content
                        print(f"  📝 {url} (processed)")
                    else:
                        print(f"  📝 {url} (no processing - content too short)")
                else:
                    print(f"  ⚠️  {url} (no content to process)")
        else:
            if use_llm_processing and not os.getenv("NOUS_API_KEY"):
                print("⚠️  LLM processing requested but NOUS_API_KEY not set, returning raw content")
            
            # Print summary of extracted pages for debugging (original behavior)
            for result in response.get('results', []):
                url = result.get('url', 'Unknown URL')
                content_length = len(result.get('raw_content', ''))
                print(f"  📝 {url} ({content_length} characters)")
        
        result_json = json.dumps(response, indent=2)
        # Clean base64 images from extracted content
        return clean_base64_images(result_json)
            
    except Exception as e:
        error_msg = f"Error extracting content: {str(e)}"
        print(f"❌ {error_msg}")
        return json.dumps({"error": error_msg})


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
    with multiple backends. Currently uses Tavily but can be easily swapped.
    
    Args:
        url (str): The base URL to crawl (can include or exclude https://)
        instructions (str): Instructions for what to crawl/extract using LLM intelligence (optional)
        depth (str): Depth of extraction ("basic" or "advanced", default: "basic")
        use_llm_processing (bool): Whether to process content with LLM for summarization (default: True)
        model (str): The model to use for LLM processing (default: gemini-2.5-flash)
        min_length (int): Minimum content length to trigger LLM processing (default: 5000)
    
    Returns:
        str: JSON string containing crawled content. If LLM processing is enabled and successful,
             the 'content' field will contain the processed markdown summary instead of raw content.
             Each page is processed individually.
    
    Raises:
        Exception: If crawling fails or API key is not set
    """
    try:
        instructions_text = f" with instructions: '{instructions}'" if instructions else ""
        print(f"🕷️ Crawling {url}{instructions_text}")
        
        # Use Tavily's crawl functionality
        response = tavily_client.crawl(
            url=url,
            limit=20,  # Reasonable limit for most use cases
            instructions=instructions or "Get all available content",
            extract_depth=depth
        )
        
        print(f"✅ Crawled {len(response.get('results', []))} pages")
        
        # Process each result with LLM if enabled
        if use_llm_processing and os.getenv("NOUS_API_KEY"):
            print("🧠 Processing crawled content with LLM...")
            
            for result in response.get('results', []):
                page_url = result.get('url', 'Unknown URL')
                title = result.get('title', '')
                content = result.get('content', '')
                
                if content:
                    # Process content with LLM
                    processed = await process_content_with_llm(
                        content, page_url, title, model, min_length
                    )
                    
                    if processed:
                        # Keep original content in raw_content field
                        result['raw_content'] = content
                        # Replace content with processed version
                        result['content'] = processed
                        print(f"  🌐 {page_url} (processed)")
                    else:
                        print(f"  🌐 {page_url} (no processing - content too short)")
                else:
                    print(f"  ⚠️  {page_url} (no content to process)")
        else:
            if use_llm_processing and not os.getenv("NOUS_API_KEY"):
                print("⚠️  LLM processing requested but NOUS_API_KEY not set, returning raw content")
            
            # Print summary of crawled pages for debugging (original behavior)
            for result in response.get('results', []):
                page_url = result.get('url', 'Unknown URL')
                content_length = len(result.get('content', ''))
                print(f"  🌐 {page_url} ({content_length} characters)")
        
        result_json = json.dumps(response, indent=2)
        # Clean base64 images from crawled content
        return clean_base64_images(result_json)
        
    except Exception as e:
        error_msg = f"Error crawling website: {str(e)}"
        print(f"❌ {error_msg}")
        return json.dumps({"error": error_msg})


# Convenience function to check if API key is available
def check_tavily_api_key() -> bool:
    """
    Check if the Tavily API key is available in environment variables.
    
    Returns:
        bool: True if API key is set, False otherwise
    """
    return bool(os.getenv("TAVILY_API_KEY"))


def check_nous_api_key() -> bool:
    """
    Check if the Nous Research API key is available in environment variables.
    
    Returns:
        bool: True if API key is set, False otherwise
    """
    return bool(os.getenv("NOUS_API_KEY"))


if __name__ == "__main__":
    """
    Simple test/demo when run directly
    """
    print("🌐 Standalone Web Tools Module")
    print("=" * 40)
    
    # Check if API keys are available
    tavily_available = check_tavily_api_key()
    nous_available = check_nous_api_key()
    
    if not tavily_available:
        print("❌ TAVILY_API_KEY environment variable not set")
        print("Please set your API key: export TAVILY_API_KEY='your-key-here'")
        print("Get API key at: https://tavily.com/")
    else:
        print("✅ Tavily API key found")
    
    if not nous_available:
        print("❌ NOUS_API_KEY environment variable not set")
        print("Please set your API key: export NOUS_API_KEY='your-key-here'")  
        print("Get API key at: https://inference-api.nousresearch.com/")
        print("⚠️  Without Nous API key, LLM content processing will be disabled")
    else:
        print("✅ Nous Research API key found")
    
    if not tavily_available:
        exit(1)
    
    print("🛠️  Web tools ready for use!")
    
    if nous_available:
        print("🧠 LLM content processing available with Gemini 2.5 Flash")
        print(f"   Default min length for processing: {DEFAULT_MIN_LENGTH_FOR_SUMMARIZATION} chars")
    
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
        print("      model='gemini-2.5-flash',")
        print("      min_length=3000")
        print("  )")
        print("")
        print("  # Disable LLM processing")
        print("  raw_content = await web_extract_tool(['https://example.com'], use_llm_processing=False)")
    
    print(f"\n📝 Run 'python test_web_tools_llm.py' to test LLM processing capabilities")
