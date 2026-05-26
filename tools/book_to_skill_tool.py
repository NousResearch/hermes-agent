"""Book-to-Skill 编译器 — 超级进化16 过目不忘。

ApexBookSkill = DoclingParse × SkillStruct × LazyLoad × MemLLM × ParallelAgent

保守可追溯版：
- 不声称安装 Docling；按可用解析器降级（txt/md -> direct, pdf -> pymupdf if available）
- 每个 chunk 保留 source/hash/offset
- 生成 skill draft + manifest + evidence map
- 默认 draft，不自动发布正式 skill
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from tools.registry import registry

HERMES = Path('/Users/appleoppa/.hermes')
OUT_ROOT = HERMES / 'workspace' / 'book_to_skill'
SKILL_DRAFT_ROOT = HERMES / 'skills' / 'workflow' / 'book-derived'


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8', errors='replace')).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def parse_document(path: Path) -> Dict[str, Any]:
    """解析文档为统一 DocumentAST。"""
    suffix = path.suffix.lower()
    source_hash = sha256_file(path)
    parser = 'direct_text'
    text = ''
    warnings = []

    if suffix in ('.md', '.txt'):
        text = path.read_text(encoding='utf-8', errors='replace')
    elif suffix == '.pdf':
        try:
            import fitz  # PyMuPDF
            parser = 'pymupdf'
            doc = fitz.open(str(path))
            pages = []
            for i, page in enumerate(doc, 1):
                pages.append(f"\n\n[PAGE {i}]\n" + page.get_text())
            text = ''.join(pages)
        except Exception as exc:
            return {'success': False, 'error': f'PDF解析失败: {exc}', 'parser': 'pymupdf'}
    else:
        return {'success': False, 'error': f'暂不支持格式: {suffix}', 'supported': ['.md', '.txt', '.pdf']}

    return {
        'success': True,
        'schema': 'book_to_skill_document_ast_v1',
        'source_file': str(path),
        'source_hash': source_hash,
        'parser': parser,
        'text_length': len(text),
        'warnings': warnings,
        'text': text,
    }


def chunk_text(text: str, max_chars: int = 1800) -> List[Dict[str, Any]]:
    """按标题/段落切块，保留 offset/hash。"""
    paras = re.split(r'\n\s*\n', text)
    chunks = []
    buf = []
    buf_start = 0
    cursor = 0

    for para in paras:
        p = para.strip()
        if not p:
            cursor += len(para) + 2
            continue
        if not buf:
            buf_start = cursor
        if sum(len(x) for x in buf) + len(p) > max_chars and buf:
            content = '\n\n'.join(buf)
            chunks.append({'idx': len(chunks), 'start': buf_start, 'end': buf_start + len(content), 'hash': sha256_text(content), 'content': content})
            buf = [p]
            buf_start = cursor
        else:
            buf.append(p)
        cursor += len(para) + 2

    if buf:
        content = '\n\n'.join(buf)
        chunks.append({'idx': len(chunks), 'start': buf_start, 'end': buf_start + len(content), 'hash': sha256_text(content), 'content': content})

    return chunks


def extract_terms(chunks: List[Dict[str, Any]], top_k: int = 30) -> List[str]:
    """轻量术语抽取：英文词/中文2-4字高频片段。"""
    from collections import Counter
    c = Counter()
    for ch in chunks:
        text = ch['content']
        c.update(re.findall(r'[A-Za-z][A-Za-z0-9_\-]{2,}', text))
        chars = re.findall(r'[\u4e00-\u9fff]', text)
        for n in (2, 3, 4):
            for i in range(len(chars)-n+1):
                token = ''.join(chars[i:i+n])
                c[token] += 1
    stop = {'的','了','和','是','在','一个','可以','进行','通过','使用','需要'}
    terms = [t for t, _ in c.most_common(top_k * 2) if t not in stop and len(t) >= 2]
    return terms[:top_k]


def make_skill_name(title: str) -> str:
    safe = re.sub(r'[^a-zA-Z0-9\u4e00-\u9fff_-]+', '-', title).strip('-')
    if not safe:
        safe = 'book-derived-skill'
    return safe[:80]


def compile_skill(source_path: Path, title: str | None = None, dry_run: bool = True) -> Dict[str, Any]:
    ast = parse_document(source_path)
    if not ast.get('success'):
        return ast

    title = title or source_path.stem.strip()
    skill_id = make_skill_name(title)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = OUT_ROOT / f'{skill_id}_{stamp}'
    chunks = chunk_text(ast['text'])
    terms = extract_terms(chunks)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'document_ast.json').write_text(json.dumps({k:v for k,v in ast.items() if k != 'text'}, ensure_ascii=False, indent=2), encoding='utf-8')
    (out_dir / 'chunks.jsonl').write_text('\n'.join(json.dumps(c, ensure_ascii=False) for c in chunks), encoding='utf-8')

    evidence = {
        'source_file': ast['source_file'],
        'source_hash': ast['source_hash'],
        'parser': ast['parser'],
        'chunks': [{'idx': c['idx'], 'hash': c['hash'], 'start': c['start'], 'end': c['end']} for c in chunks],
    }
    (out_dir / 'evidence_map.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding='utf-8')

    skill_md = f"""---
name: {skill_id}
description: 从书籍/文档《{title}》编译生成的技能草案。状态: draft，需要人工验证后发布。
version: 0.1.0
metadata:
  source_file: {ast['source_file']}
  source_hash: {ast['source_hash']}
  parser: {ast['parser']}
  status: draft
---

# {title}

## 来源与边界

| 项 | 值 |
|---|---|
| 来源文件 | `{ast['source_file']}` |
| SHA256 | `{ast['source_hash']}` |
| 解析器 | `{ast['parser']}` |
| 状态 | draft，未人工验证 |

## 核心术语

{chr(10).join('- ' + t for t in terms[:20])}

## 章节/知识块索引

本技能采用 LazyLoad：默认只加载本摘要，原文块保存在：

`{out_dir}/chunks.jsonl`

## 可执行流程

1. 先根据任务关键词查询 manifest。
2. 命中后读取 evidence_map 和相关 chunk。
3. 只引用有 source_hash/chunk_hash 的内容。
4. 低置信度内容不得作为结论。

## 验证清单

- [ ] 每条 claim 能回链 source chunk。
- [ ] 至少有一个可运行示例或人工验收项。
- [ ] 已人工复核解析错误。
- [ ] 已确认未泄露版权敏感长段落。
"""
    (out_dir / 'SKILL_DRAFT.md').write_text(skill_md, encoding='utf-8')

    manifest = {
        'schema': 'book_to_skill_manifest_v1',
        'created_at': datetime.now(timezone.utc).isoformat(),
        'title': title,
        'skill_id': skill_id,
        'status': 'draft',
        'lazy_load': True,
        'source_file': ast['source_file'],
        'source_hash': ast['source_hash'],
        'parser': ast['parser'],
        'chunk_count': len(chunks),
        'terms': terms,
        'artifacts': {
            'out_dir': str(out_dir),
            'skill_draft': str(out_dir / 'SKILL_DRAFT.md'),
            'chunks': str(out_dir / 'chunks.jsonl'),
            'evidence': str(out_dir / 'evidence_map.json'),
        },
        'boundary': 'draft_only_not_published_without_manual_validation',
    }
    (out_dir / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')

    published = False
    publish_path = SKILL_DRAFT_ROOT / skill_id / 'SKILL.md'
    if not dry_run:
        publish_path.parent.mkdir(parents=True, exist_ok=True)
        publish_path.write_text(skill_md, encoding='utf-8')
        published = True

    return {'success': True, 'manifest': manifest, 'published': published, 'publish_path': str(publish_path) if published else None}


def book_to_skill_handler(args: Dict[str, Any]) -> Dict[str, Any]:
    action = args.get('action', 'compile')
    if action == 'compile':
        source = Path(args.get('source_file', '')).expanduser()
        if not source.exists():
            return {'success': False, 'error': f'文件不存在: {source}'}
        return compile_skill(source, title=args.get('title'), dry_run=bool(args.get('dry_run', True)))
    if action == 'query_manifest':
        manifests = sorted(OUT_ROOT.glob('*/manifest.json'), key=lambda p: p.stat().st_mtime, reverse=True)
        return {'success': True, 'manifests': [json.loads(p.read_text(encoding='utf-8')) for p in manifests[:20]]}
    return {'success': False, 'error': f'unknown action: {action}'}


registry.register(
    name='book_to_skill',
    toolset='skills',
    schema={
        'name': 'book_to_skill',
        'description': 'Book-to-Skill 编译器：解析文档→生成skill草案→manifest/evidence懒加载索引。默认dry-run，不发布正式skill。',
        'parameters': {
            'type': 'object',
            'properties': {
                'action': {'type': 'string', 'enum': ['compile', 'query_manifest']},
                'source_file': {'type': 'string'},
                'title': {'type': 'string'},
                'dry_run': {'type': 'boolean'},
            }
        }
    },
    handler=book_to_skill_handler,
)
