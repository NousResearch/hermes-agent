#!/usr/bin/env python3
"""ArXiv and Semantic Scholar literature search tool."""

from __future__ import annotations

import json
import time
import xml.etree.ElementTree as ET
from typing import Any

import httpx

from research.config import get_research_config
from research.models import LiteratureRecord, now_utc
from research.state_store import ResearchStateStore
from tools.registry import registry

_ARXIV_API = "https://export.arxiv.org/api/query"
_S2_API = "https://api.semanticscholar.org/graph/v1"

_last_arxiv_call: float = 0.0
_last_s2_call: float = 0.0


def check_literature_requirements() -> bool:
    """Literature tool is always available."""
    return True


def _resolve_project_id(store: ResearchStateStore, project_id: str | None) -> str:
    resolved = project_id or store.latest_project_id()
    if not resolved:
        raise ValueError("No project_id provided and no existing project found")
    return resolved


def _arxiv_rate_limit() -> None:
    global _last_arxiv_call
    cfg = get_research_config()
    delay = float(cfg.get("literature", {}).get("arxiv_rate_limit_seconds", 3))
    elapsed = time.monotonic() - _last_arxiv_call
    if elapsed < delay:
        time.sleep(delay - elapsed)
    _last_arxiv_call = time.monotonic()


def _s2_rate_limit() -> None:
    global _last_s2_call
    cfg = get_research_config()
    delay = float(cfg.get("literature", {}).get("s2_rate_limit_seconds", 1))
    elapsed = time.monotonic() - _last_s2_call
    if elapsed < delay:
        time.sleep(delay - elapsed)
    _last_s2_call = time.monotonic()


def _parse_arxiv_entry(entry: ET.Element) -> dict[str, Any]:
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    title = (entry.findtext("atom:title", "", ns) or "").strip().replace("\n", " ")
    summary = (entry.findtext("atom:summary", "", ns) or "").strip()
    authors = [
        (a.findtext("atom:name", "", ns) or "").strip()
        for a in entry.findall("atom:author", ns)
    ]
    published = entry.findtext("atom:published", "", ns) or ""
    year = int(published[:4]) if len(published) >= 4 else None

    arxiv_id = ""
    for link in entry.findall("atom:link", ns):
        href = link.get("href", "")
        if "abs" in href:
            arxiv_id = href.split("/")[-1]
            break
    if not arxiv_id:
        raw_id = entry.findtext("atom:id", "", ns) or ""
        arxiv_id = raw_id.split("/")[-1]

    url = f"https://arxiv.org/abs/{arxiv_id}"

    return {
        "paper_id": arxiv_id,
        "title": title,
        "authors": authors,
        "year": year,
        "abstract": summary,
        "url": url,
        "source": "arxiv",
    }


def _auto_record(store: ResearchStateStore, project_id: str, paper: dict[str, Any]) -> None:
    record = LiteratureRecord(
        paper_id=paper.get("paper_id", ""),
        title=paper.get("title", ""),
        authors=paper.get("authors", []),
        year=paper.get("year"),
        abstract=paper.get("abstract", ""),
        url=paper.get("url", ""),
        citation_count=paper.get("citation_count"),
        source=paper.get("source", ""),
        tags=paper.get("tags", []),
        summary=paper.get("abstract", "")[:200],
    )
    store.append_jsonl(project_id, "literature", record.model_dump(mode="json"))


def literature_tool(action: str, cwd: str | None = None, **kwargs: Any) -> str:
    """Perform literature search and retrieval."""
    store = ResearchStateStore(cwd=cwd)
    action = (action or "").strip()

    if action == "search_arxiv":
        query = kwargs.get("query", "")
        if not query:
            raise ValueError("query is required for search_arxiv")
        max_results = min(int(kwargs.get("max_results", 10)), 50)
        sort_by = kwargs.get("sort_by", "relevance")

        _arxiv_rate_limit()
        params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": max_results,
            "sortBy": sort_by,
            "sortOrder": "descending",
        }

        try:
            resp = httpx.get(_ARXIV_API, params=params, timeout=30.0)
            resp.raise_for_status()
        except httpx.HTTPError as e:
            return json.dumps({"success": False, "error": f"ArXiv API error: {e}"}, ensure_ascii=False, indent=2)

        root = ET.fromstring(resp.text)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        entries = root.findall("atom:entry", ns)
        papers = [_parse_arxiv_entry(entry) for entry in entries]

        project_id = kwargs.get("project_id")
        if project_id and kwargs.get("auto_record", False):
            for paper in papers:
                _auto_record(store, project_id, paper)

        return json.dumps({"success": True, "papers": papers, "count": len(papers)}, ensure_ascii=False, indent=2)

    if action == "search_semantic_scholar":
        query = kwargs.get("query", "")
        if not query:
            raise ValueError("query is required for search_semantic_scholar")
        max_results = min(int(kwargs.get("max_results", 10)), 100)

        _s2_rate_limit()
        params = {
            "query": query,
            "limit": max_results,
            "fields": "paperId,title,authors,year,abstract,url,citationCount",
        }

        try:
            resp = httpx.get(f"{_S2_API}/paper/search", params=params, timeout=30.0)
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPError as e:
            return json.dumps({"success": False, "error": f"Semantic Scholar API error: {e}"}, ensure_ascii=False, indent=2)

        papers = []
        for item in data.get("data", []):
            papers.append({
                "paper_id": item.get("paperId", ""),
                "title": item.get("title", ""),
                "authors": [a.get("name", "") for a in (item.get("authors") or [])],
                "year": item.get("year"),
                "abstract": item.get("abstract", ""),
                "url": item.get("url", ""),
                "citation_count": item.get("citationCount"),
                "source": "semantic_scholar",
            })

        project_id = kwargs.get("project_id")
        if project_id and kwargs.get("auto_record", False):
            for paper in papers:
                _auto_record(store, project_id, paper)

        return json.dumps({"success": True, "papers": papers, "count": len(papers)}, ensure_ascii=False, indent=2)

    if action == "get_paper_details":
        paper_id = kwargs.get("paper_id") or kwargs.get("arxiv_id")
        if not paper_id:
            raise ValueError("paper_id or arxiv_id is required")

        _s2_rate_limit()
        # S2 accepts arXiv IDs with ARXIV: prefix
        if "/" in str(paper_id) or "." in str(paper_id):
            lookup_id = f"ARXIV:{paper_id}"
        else:
            lookup_id = str(paper_id)

        fields = "paperId,title,authors,year,abstract,url,citationCount,references,citations"
        try:
            resp = httpx.get(
                f"{_S2_API}/paper/{lookup_id}",
                params={"fields": fields},
                timeout=30.0,
            )
            resp.raise_for_status()
            item = resp.json()
        except httpx.HTTPError as e:
            return json.dumps({"success": False, "error": f"S2 API error: {e}"}, ensure_ascii=False, indent=2)

        paper = {
            "paper_id": item.get("paperId", ""),
            "title": item.get("title", ""),
            "authors": [a.get("name", "") for a in (item.get("authors") or [])],
            "year": item.get("year"),
            "abstract": item.get("abstract", ""),
            "url": item.get("url", ""),
            "citation_count": item.get("citationCount"),
            "reference_count": len(item.get("references") or []),
            "citation_list_count": len(item.get("citations") or []),
            "source": "semantic_scholar",
        }

        project_id = kwargs.get("project_id")
        if project_id and kwargs.get("auto_record", False):
            _auto_record(store, project_id, paper)

        return json.dumps({"success": True, "paper": paper}, ensure_ascii=False, indent=2)

    if action == "get_citations":
        paper_id = kwargs.get("paper_id")
        if not paper_id:
            raise ValueError("paper_id is required")
        max_results = min(int(kwargs.get("max_results", 20)), 100)

        _s2_rate_limit()
        fields = "paperId,title,authors,year,citationCount"
        try:
            resp = httpx.get(
                f"{_S2_API}/paper/{paper_id}/citations",
                params={"fields": fields, "limit": max_results},
                timeout=30.0,
            )
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPError as e:
            return json.dumps({"success": False, "error": f"S2 API error: {e}"}, ensure_ascii=False, indent=2)

        citations = []
        for item in data.get("data", []):
            citing = item.get("citingPaper", {})
            citations.append({
                "paper_id": citing.get("paperId", ""),
                "title": citing.get("title", ""),
                "authors": [a.get("name", "") for a in (citing.get("authors") or [])],
                "year": citing.get("year"),
                "citation_count": citing.get("citationCount"),
            })

        return json.dumps({"success": True, "citations": citations, "count": len(citations)}, ensure_ascii=False, indent=2)

    if action == "get_recommendations":
        positive_paper_ids = kwargs.get("positive_paper_ids") or []
        negative_paper_ids = kwargs.get("negative_paper_ids") or []
        if not positive_paper_ids:
            raise ValueError("positive_paper_ids is required")

        _s2_rate_limit()
        payload = {"positivePaperIds": positive_paper_ids}
        if negative_paper_ids:
            payload["negativePaperIds"] = negative_paper_ids

        fields = "paperId,title,authors,year,abstract,citationCount"
        try:
            resp = httpx.post(
                f"{_S2_API}/paper/recommendations",
                json=payload,
                params={"fields": fields, "limit": min(int(kwargs.get("max_results", 10)), 50)},
                timeout=30.0,
            )
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPError as e:
            return json.dumps({"success": False, "error": f"S2 API error: {e}"}, ensure_ascii=False, indent=2)

        papers = []
        for item in data.get("recommendedPapers", []):
            papers.append({
                "paper_id": item.get("paperId", ""),
                "title": item.get("title", ""),
                "authors": [a.get("name", "") for a in (item.get("authors") or [])],
                "year": item.get("year"),
                "abstract": item.get("abstract", ""),
                "citation_count": item.get("citationCount"),
                "source": "semantic_scholar_recommendations",
            })

        project_id = kwargs.get("project_id")
        if project_id and kwargs.get("auto_record", False):
            for paper in papers:
                _auto_record(store, project_id, paper)

        return json.dumps({"success": True, "papers": papers, "count": len(papers)}, ensure_ascii=False, indent=2)

    raise ValueError(f"Unsupported literature action: {action}")


LITERATURE_SCHEMA = {
    "name": "literature",
    "description": (
        "Search academic literature via ArXiv and Semantic Scholar APIs. "
        "Supports paper search, citation lookup, paper details, and recommendations. "
        "Can auto-record findings to project literature notes."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "search_arxiv",
                    "search_semantic_scholar",
                    "get_paper_details",
                    "get_citations",
                    "get_recommendations",
                ],
            },
            "cwd": {"type": "string"},
            "project_id": {"type": "string"},
            "query": {"type": "string"},
            "arxiv_id": {"type": "string"},
            "paper_id": {"type": "string"},
            "max_results": {"type": "integer"},
            "sort_by": {"type": "string"},
            "auto_record": {"type": "boolean"},
            "positive_paper_ids": {"type": "array", "items": {"type": "string"}},
            "negative_paper_ids": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["action"],
    },
}


registry.register(
    name="literature",
    toolset="research",
    schema=LITERATURE_SCHEMA,
    handler=lambda args, **kw: literature_tool(**args),
    check_fn=check_literature_requirements,
)
