"""MemLLM 统一管道 — RAG + 长期记忆同步。

将现有的 memory_tool、hippocampus、gene DB 通路打包为一个统一调用。
query(): 搜索 MEMORY.md + USER.md + hippocampus → 返回合并上下文
store(): 写入内存 + 触发 hippocampus 巩固 + 基因 DB 同步
"""
import json
import re
import sqlite3
import logging
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

HERMES = Path('/Users/appleoppa/.hermes')
MEMORY_FILE = HERMES / 'memories' / 'MEMORY.md'
USER_FILE = HERMES / 'memories' / 'USER.md'
HIPP_DB = HERMES / 'data' / 'hippocampus.db'
GENE_DB = HERMES / 'workspace' / '开智' / '02-进化基因' / 'apex_evolution_genes.sqlite3'
logger = logging.getLogger(__name__)


def _tokenize(text: str) -> List[str]:
    """拆词：中文单字+双字组+英文单词"""
    text = text.lower()
    chars = re.findall(r'[\u4e00-\u9fff]', text)
    bigrams = [chars[i] + chars[i+1] for i in range(len(chars)-1)]
    words = re.findall(r'[a-z]+', text)
    return words + chars + bigrams


def _keyword_match_score(query_tokens: List[str], text: str) -> float:
    """关键词匹配得分。"""
    text_tokens = _tokenize(text)
    if not query_tokens or not text_tokens:
        return 0.0
    q_set = set(query_tokens)
    t_set = set(text_tokens)
    intersection = q_set & t_set
    if not intersection:
        return 0.0
    # score = (交集词数²) / (查询词数 × 文本词数)⁰·⁵
    return len(intersection) ** 2 / (len(q_set) + len(t_set)) ** 0.5


def _search_memory_file(path: Path, query_tokens: List[str], top_k: int) -> List[Dict[str, Any]]:
    """在单个 memory 文件中做关键词检索。"""
    if not path.exists():
        return []
    text = path.read_text(encoding='utf-8', errors='replace')
    # 按段落分割
    paragraphs = re.split(r'\n\s*\n', text)
    scored = []
    for i, para in enumerate(paragraphs):
        para = para.strip()
        if not para or len(para) < 20:
            continue
        score = _keyword_match_score(query_tokens, para)
        if score > 0:
            scored.append({
                'source': path.name,
                'paragraph_idx': i,
                'score': round(score, 4),
                'content': para[:200],
                'length': len(para),
            })
    scored.sort(key=lambda x: -x['score'])
    return scored[:top_k]


def _search_hippocampus(query_tokens: List[str], top_k: int) -> List[Dict[str, Any]]:
    """在 hippocampus SQLite 中用 LIKE 检索。"""
    if not HIPP_DB.exists():
        return []
    query_text = ' '.join(query_tokens[:5])
    try:
        con = sqlite3.connect(str(HIPP_DB))
        cur = con.cursor()
        rows = cur.execute(
            "SELECT content, memory_type, importance, created_at FROM memories "
            "WHERE content LIKE ? ORDER BY importance DESC LIMIT ?",
            (f'%{query_text}%', top_k)
        ).fetchall()
        con.close()
        return [
            {'source': 'hippocampus', 'content': r[0][:200], 'type': r[1], 'importance': r[2], 'created_at': r[3]}
            for r in rows
        ]
    except Exception as exc:
        logger.warning("MemLLM hippocampus query failed: %s", exc)
        return []


def _search_genes(query_tokens: List[str], top_k: int) -> List[Dict[str, Any]]:
    """在进化基因库中按 gene_name 和 absorbed_knowledge 做关键词匹配。"""
    if not GENE_DB.exists():
        return []
    try:
        con = sqlite3.connect(str(GENE_DB))
        cur = con.cursor()
        rows = cur.execute(
            "SELECT gene_name, absorbed_knowledge, status, created_at FROM evolution_genes ORDER BY created_at DESC LIMIT 50"
        ).fetchall()
        con.close()
        scored = []
        for gene_name, knowledge, status, created_at in rows:
            text = f"{gene_name} {knowledge or ''}"
            score = _keyword_match_score(query_tokens, text)
            if score > 0:
                scored.append({
                    'source': 'gene_db',
                    'gene_name': gene_name,
                    'score': round(score, 4),
                    'knowledge': (knowledge or '')[:200],
                    'status': status,
                })
        scored.sort(key=lambda x: -x['score'])
        return scored[:top_k]
    except Exception as exc:
        logger.warning("MemLLM gene query failed: %s", exc)
        return []


def register_tool():
    """注册 memllm 工具到工具注册表。"""
    from tools.registry import registry

    def memllm_handler(args: Dict[str, Any]) -> Dict[str, Any]:
        return _memllm_handler(args)

    registry.register(
        name="memllm",
        toolset="skills",
        schema={
            "name": "memllm",
            "description": "MemLLM 统一管道：RAG 查询 + 长期记忆存储 + 同步。query(上下文)→返回相关记忆块；store(数据)→写入记忆并触发巩固同步。",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["query", "store", "stats"],
                        "description": "query=检索, store=存储, stats=统计"
                    },
                    "query": {"type": "string", "description": "query 动作下的检索上下文"},
                    "top_k": {"type": "integer", "description": "返回最多 top_k 条（默认 5）"},
                    "content": {"type": "string", "description": "store 动作下要存储的记忆内容"},
                    "source": {"type": "string", "description": "store 动作下的来源描述"},
                    "importance": {"type": "number", "description": "store 动作下的重要性（0-1，默认 0.5）"},
                }
            }
        },
        handler=memllm_handler,
    )
    return True


def _memllm_handler(args: Dict[str, Any]) -> Dict[str, Any]:
    action = args.get('action', 'stats')

    if action == 'query':
        query_text = args.get('query', '')
        top_k = int(args.get('top_k', 5))
        if not query_text:
            return {'success': False, 'error': '需要 query 参数'}

        tokens = _tokenize(query_text)

        # 三路并行检索
        mem_results = _search_memory_file(MEMORY_FILE, tokens, top_k)
        user_results = _search_memory_file(USER_FILE, tokens, top_k)
        hipp_results = _search_hippocampus(tokens, top_k)
        gene_results = _search_genes(tokens, top_k)

        # 合并去重
        seen = set()
        merged = []
        for r in mem_results + user_results + hipp_results + gene_results:
            key = r.get('content', '')[:80]
            if key not in seen:
                seen.add(key)
                merged.append(r)

        merged.sort(key=lambda x: -x.get('score', 0))
        merged = merged[:top_k * 2]

        return {
            'success': True,
            'action': 'query',
            'query': query_text,
            'total_results': len(merged),
            'sources': {
                'memory_md': len(mem_results),
                'user_md': len(user_results),
                'hippocampus': len(hipp_results),
                'gene_db': len(gene_results),
            },
            'results': merged,
        }

    elif action == 'store':
        content = args.get('content', '')
        source = args.get('source', 'memllm_auto')
        importance = float(args.get('importance', 0.5))

        if not content:
            return {'success': False, 'error': '需要 content 参数'}

        written = []
        # 写入 MEMORY.md
        MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with MEMORY_FILE.open('a', encoding='utf-8') as f:
            f.write(f"\n§\n{stamp} MemLLM({source}): {content}\n")
        written.append('memory_md')

        # hippocampus 同步
        try:
            if HIPP_DB.exists():
                con = sqlite3.connect(str(HIPP_DB))
                con.execute(
                    "INSERT INTO memories (content, memory_type, importance, created_at) VALUES (?, ?, ?, ?)",
                    (f"MemLLM({source}): {content[:500]}", 'memllm_sync', min(importance, 1.0), datetime.now(timezone.utc).isoformat())
                )
                con.commit()
                con.close()
                written.append('hippocampus')
        except Exception as exc:
            logger.warning("MemLLM hippocampus store failed: %s", exc)

        return {
            'success': True,
            'action': 'store',
            'content': content[:200],
            'source': source,
            'importance': importance,
            'written_to': written,
            'length': len(content),
        }

    else:  # stats
        stats = {}
        for name, p in [('memory_md', MEMORY_FILE), ('user_md', USER_FILE)]:
            if p.exists():
                text = p.read_text(encoding='utf-8', errors='replace')
                stats[name] = {'size': p.stat().st_size, 'lines': len(text.splitlines()), 'paragraphs': len(re.split(r'\n\s*\n', text))}
            else:
                stats[name] = {'exists': False}
        stats['hippocampus_db'] = {'exists': HIPP_DB.exists(), 'size': HIPP_DB.stat().st_size if HIPP_DB.exists() else 0}
        stats['gene_db'] = {'exists': GENE_DB.exists(), 'size': GENE_DB.stat().st_size if GENE_DB.exists() else 0}

        return {
            'success': True,
            'action': 'stats',
            'pipeline': 'MemLLM = RAG(query: MEMORY.md+USER.md+hippocampus+genes) + LongTermMem(store: memory.md+hippocampus)',
            'sources': stats,
        }


# 直接可调用（不依赖工具注册）
def query(context: str, top_k: int = 5) -> Dict[str, Any]:
    """便捷函数：单次查询。"""
    return _memllm_handler({'action': 'query', 'query': context, 'top_k': top_k})


def store(content: str, source: str = 'memllm', importance: float = 0.5) -> Dict[str, Any]:
    """便捷函数：单次存储。"""
    return _memllm_handler({'action': 'store', 'content': content, 'source': source, 'importance': importance})


# 自动注册工具
try:
    register_tool()
except Exception:
    pass
