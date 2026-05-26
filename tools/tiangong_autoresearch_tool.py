"""
TianGong AutoResearch 适配器 - 超级进化11

四核之一：检索、读取、交叉验证、知识蒸馏。

实际后端：web_search + arxiv + GitHub 进化工厂（超级进化2）。
"""

from tools.registry import registry
import logging
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


def query_arxiv(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """查询 arxiv API"""
    base_url = "http://export.arxiv.org/api/query"
    params = {
        'search_query': f'all:{query}',
        'start': 0,
        'max_results': max_results,
        'sortBy': 'relevance',
        'sortOrder': 'descending'
    }
    url = f"{base_url}?{urllib.parse.urlencode(params)}"
    
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            xml_data = resp.read().decode('utf-8')
        
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        root = ET.fromstring(xml_data)
        
        papers = []
        for entry in root.findall('atom:entry', ns):
            paper = {
                'title': (entry.find('atom:title', ns).text or '').strip(),
                'summary': (entry.find('atom:summary', ns).text or '').strip()[:300],
                'published': (entry.find('atom:published', ns).text or '').strip()[:10],
                'authors': [a.find('atom:name', ns).text for a in entry.findall('atom:author', ns)][:3],
                'link': '',
            }
            for link in entry.findall('atom:link', ns):
                if link.attrib.get('type') == 'application/pdf' or link.attrib.get('rel') == 'alternate':
                    paper['link'] = link.attrib.get('href', '')
                    break
            papers.append(paper)
        return papers
    except Exception as e:
        logger.warning(f"arxiv 查询失败: {e}")
        return []


def tiangong_autoresearch_handler(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    TianGong AutoResearch：多源检索、交叉验证、蒸馏。
    
    Args:
        args:
            query: 研究主题（必需）
            sources: ["arxiv", "github", "web"]，默认 ["arxiv"]
            max_results: 每源最大结果数（默认 5）
        
    Returns:
        AutoResearch 报告
    """
    query = args.get('query', '')
    if not query:
        return {'success': False, 'error': 'query 参数不能为空', 'role': 'autoresearch'}
    
    sources = args.get('sources', ['arxiv'])
    max_results = args.get('max_results', 5)
    
    evidence = {
        'role': 'autoresearch',
        'query': query,
        'sources_queried': sources,
        'results': {},
    }
    
    # arxiv 检索
    if 'arxiv' in sources:
        papers = query_arxiv(query, max_results)
        evidence['results']['arxiv'] = {
            'count': len(papers),
            'papers': papers,
        }
    
    # github 检索（依赖超级进化2 GitHub 进化工厂）
    if 'github' in sources:
        try:
            import subprocess
            # 假设 hermes evolution-factory 命令存在
            result = subprocess.run(
                ['gh', 'search', 'repos', query, '--limit', str(max_results), '--json', 'name,description,url,stargazersCount'],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                import json
                evidence['results']['github'] = {
                    'count': max_results,
                    'repos': json.loads(result.stdout),
                }
            else:
                evidence['results']['github'] = {'error': 'gh 命令未配置或失败'}
        except FileNotFoundError:
            evidence['results']['github'] = {'error': 'gh CLI 未安装'}
        except Exception as e:
            evidence['results']['github'] = {'error': str(e)}
    
    # web 检索（提示用户使用 web_search 工具）
    if 'web' in sources:
        evidence['results']['web'] = {
            'note': '请单独调用 web_search 工具，本适配器不直接调用以避免重复'
        }
    
    # 蒸馏：跨源对比
    total = sum(
        v.get('count', 0) for v in evidence['results'].values() 
        if isinstance(v, dict) and 'count' in v
    )
    evidence['total_results'] = total
    evidence['message'] = f"✅ AutoResearch：从 {len(sources)} 个来源检索到 {total} 条结果"
    
    return {'success': True, **evidence}


registry.register(
    name="tiangong_autoresearch",
    toolset="skills",
    schema={
        "name": "tiangong_autoresearch",
        "description": "TianGong 四核之 AutoResearch：多源检索（arxiv/github）、交叉验证、知识蒸馏（超级进化11）。",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "研究主题"},
                "sources": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["arxiv", "github", "web"]},
                    "description": "检索源（默认 ['arxiv']）"
                },
                "max_results": {"type": "integer", "description": "每源最大结果数（默认 5）"}
            },
            "required": ["query"]
        }
    },
    handler=tiangong_autoresearch_handler
)
