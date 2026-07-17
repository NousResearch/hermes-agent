#!/usr/bin/env python3
"""
网页检索工具 - 通过搜索引擎 API 搜索公开信息

用途：技术调研、资料收集、信息验证等合法用途
依赖：pip install requests

使用前需要设置搜索引擎 API：
1. Google Custom Search: https://programmablesearchengine.google.com/
2. 或 SerpAPI: https://serpapi.com/ (免费版每月100次)
"""

import json
import urllib.parse
import urllib.request
import sys
from datetime import datetime


# ============================================================
# 方式一：Google Custom Search API（需注册）
# ============================================================
def search_google(query, api_key=None, cx=None, num=10):
    """使用 Google Custom Search JSON API"""
    if not api_key or not cx:
        return None
    
    url = (
        f"https://www.googleapis.com/customsearch/v1"
        f"?key={api_key}&cx={cx}&q={urllib.parse.quote(query)}&num={min(num, 10)}"
    )
    
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            data = json.loads(resp.read())
        
        results = []
        for item in data.get("items", []):
            results.append({
                "title": item.get("title"),
                "url": item.get("link"),
                "snippet": item.get("snippet"),
            })
        return results
    except Exception as e:
        print(f"[Google API Error] {e}", file=sys.stderr)
        return None


# ============================================================
# 方式二：SerpAPI（推荐，无需 Google CX 配置）
# ============================================================
def search_serpapi(query, api_key=None, num=10):
    """使用 SerpAPI（支持 Google/Bing/Baidu 等引擎）"""
    if not api_key:
        return None
    
    params = {
        "q": query,
        "api_key": api_key,
        "engine": "google",
        "num": min(num, 10),
    }
    url = f"https://serpapi.com/search?{urllib.parse.urlencode(params)}"
    
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            data = json.loads(resp.read())
        
        results = []
        for item in data.get("organic_results", []):
            results.append({
                "title": item.get("title"),
                "url": item.get("link"),
                "snippet": item.get("snippet"),
            })
        return results
    except Exception as e:
        print(f"[SerpAPI Error] {e}", file=sys.stderr)
        return None


# ============================================================
# 方式三：DuckDuckGo（免费，无需 API Key）
# ============================================================
def search_duckduckgo(query, max_results=10):
    """使用 DuckDuckGo Lite API（免费，无需注册）"""
    url = "https://lite.duckduckgo.com/lite/"
    data = urllib.parse.urlencode({"q": query}).encode()
    
    try:
        req = urllib.request.Request(url, data=data,
            headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            html = resp.read().decode()
        
        # 简单解析结果
        results = []
        import re
        # 提取结果链接和标题
        links = re.findall(r'class="result-link".*?href="(.*?)".*?>(.*?)</a>', html, re.DOTALL)
        snippets = re.findall(r'class="result-snippet".*?>(.*?)</td>', html, re.DOTALL)
        
        for i, (url, title) in enumerate(links[:max_results]):
            snippet = snippets[i] if i < len(snippets) else ""
            results.append({
                "title": re.sub(r'<[^>]+>', '', title).strip(),
                "url": url,
                "snippet": re.sub(r'<[^>]+>', '', snippet).strip(),
            })
        return results
    except Exception as e:
        print(f"[DuckDuckGo Error] {e}", file=sys.stderr)
        return None


# ============================================================
# 主入口
# ============================================================
def main():
    if len(sys.argv) < 2:
        print("用法: python web_search.py <搜索词>")
        print("示例: python web_search.py Python async 教程")
        sys.exit(1)
    
    query = " ".join(sys.argv[1:])
    print(f"\n🔍 搜索: {query}")
    print(f"⏰ 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # 优先尝试 DuckDuckGo（免费）
    results = search_duckduckgo(query)
    
    if not results:
        # 如果安装了 SerpAPI key，尝试 SerpAPI
        # results = search_serpapi(query, api_key="YOUR_SERPAPI_KEY")
        print("⚠️  搜索无结果，请设置 API Key 或检查网络\n")
        print("💡 免费方案: 安装 duckduckgo-search 库")
        print("   pip install duckduckgo-search")
        print("   from duckduckgo_search import DDGS")
        print("   with DDGS() as ddgs:")
        print("       for r in ddgs.text(query, max_results=10):")
        print("           print(r['title'], r['href'])")
        return
    
    print(f"共找到 {len(results)} 条结果:\n")
    for i, r in enumerate(results, 1):
        print(f"{i:2d}. {r['title']}")
        print(f"    {r['url']}")
        if r['snippet']:
            print(f"    {r['snippet'][:150]}")
        print()


if __name__ == "__main__":
    main()
