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
from typing import List
from tavily import TavilyClient

# Initialize Tavily client once at module level
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


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


def web_extract_tool(urls: List[str], format: str = None) -> str:
    """
    Extract content from specific web pages using available extraction API backend.
    
    This function provides a generic interface for web content extraction that
    can work with multiple backends. Currently uses Tavily but can be easily swapped.
    
    Args:
        urls (List[str]): List of URLs to extract content from
        format (str): Desired output format ("markdown" or "html", optional)
    
    Returns:
        str: JSON string containing extracted content with the following structure:
             {
                 "results": [
                     {
                         "url": str,
                         "title": str,
                         "raw_content": str,
                         "content": str
                     },
                     ...
                 ]
             }
    
    Raises:
        Exception: If extraction fails or API key is not set
    """
    try:
        print(f"📄 Extracting content from {len(urls)} URL(s)")
        
        # Use Tavily's extract functionality
        response = tavily_client.extract(urls=urls, format=format)
        
        print(f"✅ Extracted content from {len(response.get('results', []))} pages")
        
        # Print summary of extracted pages for debugging
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


def web_crawl_tool(url: str, instructions: str = None, depth: str = "basic") -> str:
    """
    Crawl a website with specific instructions using available crawling API backend.
    
    This function provides a generic interface for web crawling that can work
    with multiple backends. Currently uses Tavily but can be easily swapped.
    
    Args:
        url (str): The base URL to crawl (can include or exclude https://)
        instructions (str): Instructions for what to crawl/extract using LLM intelligence (optional)
        depth (str): Depth of extraction ("basic" or "advanced", default: "basic")
    
    Returns:
        str: JSON string containing crawled content with the following structure:
             {
                 "results": [
                     {
                         "url": str,
                         "title": str,
                         "content": str
                     },
                     ...
                 ]
             }
    
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
        
        # Print summary of crawled pages for debugging
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


if __name__ == "__main__":
    """
    Simple test/demo when run directly
    """
    print("🌐 Standalone Web Tools Module")
    print("=" * 40)
    
    # Check if API key is available
    if not check_tavily_api_key():
        print("❌ TAVILY_API_KEY environment variable not set")
        print("Please set your API key: export TAVILY_API_KEY='your-key-here'")
        print("Get API key at: https://tavily.com/")
        exit(1)
    
    print("✅ Tavily API key found")
    print("🛠️  Web tools ready for use!")
    print("\nExample usage:")
    print("  from web_tools import web_search_tool, web_extract_tool, web_crawl_tool")
    print("  results = web_search_tool('Python tutorials')")
    print("  content = web_extract_tool(['https://example.com'])")
    print("  crawl_data = web_crawl_tool('example.com', 'Find documentation')")
