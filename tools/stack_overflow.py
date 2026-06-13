import os
import json
import time
import hashlib
import re
from html.parser import HTMLParser
from pathlib import Path
from typing import List, Dict, Any, Optional
import httpx

from tools.registry import registry, tool_result, tool_error

CACHE_DIR = Path.home() / ".hermes" / "cache"
CACHE_FILE = CACHE_DIR / "stack_overflow_cache.json"
BACKOFF_FILE = CACHE_DIR / "stack_overflow_backoff.json"

class HTMLToMarkdown(HTMLParser):
    def __init__(self):
        super().__init__()
        self.result = []
        self.in_code = False
        self.in_pre = False
        self.current_href = None

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        if tag == 'pre':
            self.in_pre = True
            self.result.append('\n```\n')
        elif tag == 'code':
            self.in_code = True
            if not self.in_pre:
                self.result.append('`')
        elif tag == 'a':
            self.current_href = attrs_dict.get('href', '')
            self.result.append('[')
        elif tag in ['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'div']:
            self.result.append('\n\n')
        elif tag == 'br':
            self.result.append('\n')
        elif tag == 'li':
            self.result.append('\n- ')

    def handle_endtag(self, tag):
        if tag == 'pre':
            self.in_pre = False
            self.result.append('\n```\n')
        elif tag == 'code':
            self.in_code = False
            if not self.in_pre:
                self.result.append('`')
        elif tag == 'a':
            href = self.current_href or ''
            self.result.append(f']({href})')
            self.current_href = None
        elif tag in ['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'div']:
            self.result.append('\n')

    def handle_data(self, data):
        self.result.append(data)

def html_to_markdown(html: str) -> str:
    parser = HTMLToMarkdown()
    parser.feed(html)
    text = "".join(parser.result)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def get_cache_key(query: str, tags: Optional[List[str]] = None) -> str:
    tags_str = ";".join(sorted(tags)) if tags else ""
    return hashlib.sha256(f"{query}||{tags_str}".encode('utf-8')).hexdigest()

def check_backoff() -> bool:
    if not BACKOFF_FILE.exists():
        return False
    try:
        data = json.loads(BACKOFF_FILE.read_text())
        until = data.get("backoff_until", 0)
        if time.time() < until:
            return True
    except Exception:
        pass
    return False

def set_backoff(seconds: int):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    until = time.time() + seconds
    BACKOFF_FILE.write_text(json.dumps({"backoff_until": until}))

def get_cached_result(key: str) -> Optional[Dict[str, Any]]:
    if not CACHE_FILE.exists():
        return None
    try:
        cache = json.loads(CACHE_FILE.read_text())
        entry = cache.get(key)
        if entry:
            if time.time() - entry.get("timestamp", 0) < 86400:
                return entry.get("data")
    except Exception:
        pass
    return None

def save_to_cache(key: str, data: Dict[str, Any]):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = {}
    if CACHE_FILE.exists():
        try:
            cache = json.loads(CACHE_FILE.read_text())
        except Exception:
            pass
    cache[key] = {
        "timestamp": time.time(),
        "data": data
    }
    now = time.time()
    cache = {k: v for k, v in cache.items() if now - v.get("timestamp", 0) < 86400}
    try:
        CACHE_FILE.write_text(json.dumps(cache, indent=2))
    except Exception:
        pass

def search_stack_overflow(query: str, tags: Optional[List[str]] = None, api_key: Optional[str] = None) -> Dict[str, Any]:
    if check_backoff():
        return {"error": "Backoff active. Must wait before making API requests."}

    cache_key = get_cache_key(query, tags)
    cached = get_cached_result(cache_key)
    if cached:
        return {**cached, "source": "cache"}

    base_url = "https://api.stackexchange.com/2.3"
    client_params = {
        "site": "stackoverflow",
        "pagesize": 3,
        "order": "desc",
        "sort": "relevance",
    }

    if not api_key:
        api_key = os.environ.get("STACK_EXCHANGE_APP_KEY")
    if api_key:
        client_params["key"] = api_key

    if tags:
        client_params["tagged"] = ";".join(tags)

    client_params["q"] = query

    headers = {
        "User-Agent": "HermesAgent/0.15.1 (https://github.com/NousResearch/hermes-agent)"
    }

    try:
        response = httpx.get(f"{base_url}/search/advanced", params=client_params, headers=headers, timeout=10.0)
        data = response.json()

        if "backoff" in data:
            set_backoff(int(data["backoff"]))

        if response.status_code != 200:
            return {"error": f"API Search returned status {response.status_code}", "details": data}

        items = data.get("items", [])
        if not items:
            result = {"questions": [], "quota_remaining": data.get("quota_remaining")}
            save_to_cache(cache_key, result)
            return result

        question_ids = []
        questions_map = {}
        for q in items:
            q_id = str(q["question_id"])
            question_ids.append(q_id)
            questions_map[q_id] = {
                "question_id": q["question_id"],
                "title": q["title"],
                "link": q["link"],
                "score": q["score"],
                "is_answered": q["is_answered"],
                "creation_date": q.get("creation_date"),
                "answers": []
            }

        ids_str = ";".join(question_ids)
        answers_params = {
            "site": "stackoverflow",
            "sort": "votes",
            "order": "desc",
            "filter": "withbody",
            "pagesize": 10
        }
        if api_key:
            answers_params["key"] = api_key

        ans_response = httpx.get(f"{base_url}/questions/{ids_str}/answers", params=answers_params, headers=headers, timeout=10.0)
        ans_data = ans_response.json()

        if "backoff" in ans_data:
            set_backoff(int(ans_data["backoff"]))

        if ans_response.status_code == 200:
            ans_items = ans_data.get("items", [])
            for ans in ans_items:
                q_id = str(ans["question_id"])
                if q_id in questions_map:
                    score = ans.get("score", 0)
                    if score < 0 and not ans.get("is_accepted"):
                        continue
                    questions_map[q_id]["answers"].append({
                        "answer_id": ans["answer_id"],
                        "score": score,
                        "is_accepted": ans.get("is_accepted", False),
                        "creation_date": ans.get("creation_date"),
                        "body_markdown": html_to_markdown(ans.get("body", ""))
                    })

        questions_list = list(questions_map.values())
        for q in questions_list:
            q["answers"].sort(key=lambda x: (x["is_accepted"], x["score"]), reverse=True)

        result = {
            "questions": questions_list,
            "quota_remaining": data.get("quota_remaining"),
            "quota_max": data.get("quota_max")
        }
        save_to_cache(cache_key, result)
        return result

    except Exception as e:
        return {"error": f"Request failed: {str(e)}"}

def _stack_overflow_search_handler(args: dict, **kwargs) -> str:
    query = args.get("query", "")
    tags = args.get("tags")

    if not query:
        return tool_error("Query must not be empty")

    result = search_stack_overflow(query, tags)
    if "error" in result:
        return tool_error(result["error"], **{k: v for k, v in result.items() if k != "error"})

    return tool_result(result)

STACK_OVERFLOW_SEARCH_SCHEMA = {
    "name": "stack_overflow_search",
    "description": "Search Stack Overflow for answered questions matching a technical error message or query. Use only for software debugging, stack traces, runtime errors, or dependency/API issues.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Error message or technical query to search for, for example: ValueError: Columns must be same length as key"
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional Stack Overflow tags to filter results, for example: python, pandas"
            }
        },
        "required": ["query"]
    }
}

registry.register(
    name="stack_overflow_search",
    toolset="web",
    schema=STACK_OVERFLOW_SEARCH_SCHEMA,
    handler=_stack_overflow_search_handler,
    description="Search Stack Overflow for answered questions matching a technical error message or query, returning parsed markdown answers.",
    emoji="💬",
    max_result_size_chars=100_000
)
