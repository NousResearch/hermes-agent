#!/usr/bin/env python3
"""
Knowledge Retrieval Tool - 知识库检索

通过知识库 API 从知识库中检索相关信息，并支持列出知识库详情。

API 说明：
- 检索接口: POST {retrieval_url}
- 列表接口: POST {kb_list_url}

配置方式：
  config.yaml 中添加：
    knowledge_retrieval:
      retrieval_url: "http://<host>/documents_store/v1/retrieval"
      kb_list_url: "http://<host>/documents_store/v1/knowledge_bases/list"
      default_dataset_ids: ["kb-xxxxxxxxxxxx"]
      top_k: 5
"""

import json
import logging
from typing import Dict, Any, Optional
import urllib.request
import urllib.error

from tools.registry import registry, tool_error, tool_result

logger = logging.getLogger(__name__)


def _get_dataset_ids(config: Dict[str, Any]) -> list:
    """从配置中提取并规范化 default_dataset_ids，确保返回 list[str]。
    
    兼容环境变量注入的逗号分隔字符串（如 "kb-1,kb-2,kb-3"）和原生列表格式。
    """
    raw = config.get("default_dataset_ids", [])
    if isinstance(raw, str):
        return [x.strip() for x in raw.split(",") if x.strip()]
    return raw


def _get_config() -> Dict[str, Any]:
    """获取知识库配置，从 config.yaml 加载"""
    logger.info("📋 正在加载知识库配置...")
    config = {
        "retrieval_url": "",
        "kb_list_url": "",
        "default_dataset_ids": [],
        "top_k": 5,
    }

    # 尝试从 Hermes 配置加载
    try:
        from hermes_cli.config import load_config
        hermes_config = load_config()
        kr_config = hermes_config.get("knowledge_retrieval", {})
        if kr_config:
            # 兼容旧配置：如果只有 api_url/base_url，自动推导
            api_url = kr_config.get("api_url", "")
            base_url = kr_config.get("base_url", "")
            retrieval_url = kr_config.get("retrieval_url", "")
            kb_list_url = kr_config.get("kb_list_url", "")
            if not retrieval_url and base_url:
                retrieval_url = f"{base_url.rstrip('/')}/retrieval"
            if not retrieval_url and api_url:
                retrieval_url = api_url
            if not kb_list_url and base_url:
                kb_list_url = f"{base_url.rstrip('/')}/knowledge_bases/list"
            raw_ids = _get_dataset_ids(kr_config)
            default_ids = _get_dataset_ids(config)
            dataset_ids = raw_ids if raw_ids else default_ids
            
            config.update({
                "retrieval_url": retrieval_url,
                "kb_list_url": kb_list_url,
                "default_dataset_ids": dataset_ids,
                "top_k": kr_config.get("top_k", config["top_k"]),
            })
            logger.info("✅ 已从 config.yaml 加载知识库配置")
    except Exception as e:
        logger.debug("Failed to load knowledge_retrieval config from hermes config: %s", e)
        logger.warning("⚠️ 无法从 config.yaml 加载配置，将使用默认值")

    return config


def _http_request(method: str, url: str, body: Optional[Dict] = None) -> Dict:
    """发送 HTTP 请求"""
    if not url:
        raise ValueError("API URL 未配置")

    logger.info(f"🌐 正在请求: {method} {url}")
    logger.debug(f"📤 请求体: {json.dumps(body, ensure_ascii=False)}" if body else "📤 请求体: 无")
    headers = {"Content-Type": "application/json"}

    data = json.dumps(body).encode("utf-8") if body else None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            response_data = json.loads(resp.read().decode("utf-8"))
            logger.info(f"✅ 请求成功，状态码: {resp.status}")
            return response_data
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8", errors="replace")
        logger.error(f"❌ API 错误 (HTTP {e.code}): {error_body}")
        raise RuntimeError(f"API error {e.code}: {error_body}")
    except urllib.error.URLError as e:
        logger.error(f"❌ API 连接失败: {e.reason}")
        raise RuntimeError(f"API connection failed: {e.reason}")


def knowledge_retrieval(query: str, kb_ids: Optional[list] = None) -> str:
    """从知识库中检索相关信息。"""
    if not query or not query.strip():
        logger.warning("⚠️ 检索失败：查询词为空")
        return tool_error("查询词不能为空")

    config = _get_config()
    retrieval_url = config.get("retrieval_url", "")
    if not retrieval_url:
        return tool_error("知识库未配置。请在 config.yaml 中配置 retrieval_url")

    # 动态选择数据集 ID：优先使用传入的参数，否则使用默认配置
    target_dataset_ids = kb_ids if kb_ids else _get_dataset_ids(config)
    if not target_dataset_ids:
        return tool_error("未指定数据集。请在 config.yaml 中配置 default_dataset_ids 或传入 kb_ids 参数")

    top_k = config.get("top_k", 5)
    logger.info(f"🔍 开始检索知识库，查询: \"{query}\"")
    logger.info(f"📚 目标知识库: {target_dataset_ids}")
    logger.info(f"📊 返回数量: {top_k}")

    try:
        request_body = {
            "kb_ids": target_dataset_ids,
            "query": query.strip(),
            "top_k": top_k,
            "engines": [],
        }

        response = _http_request("POST", retrieval_url, body=request_body)

        # 兼容多种响应格式
        # 格式 1: {"status": 200, "data": {"results": [...]}}
        # 格式 2: {"code": 0, "data": {"chunks": [...]}}
        is_success = response.get("code") == 0 or response.get("status") == 200
        
        if is_success:
            data = response.get("data", {})
            # 兼容 results/chunks 字段
            chunks = data.get("results", data.get("chunks", []))
            logger.info(f"📝 检索到 {len(chunks)} 条结果")
            formatted_results = []
            for i, chunk in enumerate(chunks, 1):
                formatted_results.append({
                    "index": i,
                    "content": chunk.get("content", ""),
                    "document_name": chunk.get("doc_name", chunk.get("docnm_kwd", "")),
                    "doc_id": chunk.get("doc_id", ""),
                    "chunk_id": chunk.get("chunk_id", ""),
                    "score": chunk.get("score", chunk.get("similarity", 0)),
                    "start_page": chunk.get("start_page", ""),
                    "end_page": chunk.get("end_page", ""),
                })
            logger.info("✅ 知识库检索完成")
            return tool_result({
                "success": True,
                "query": query,
                "dataset_ids": target_dataset_ids,
                "total_results": len(formatted_results),
                "results": formatted_results,
            })
        else:
            error_msg = response.get("message", "未知错误")
            logger.error(f"❌ 返回错误: {error_msg}")
            return tool_error(f"检索失败: {error_msg}")

    except Exception as e:
        logger.exception("Knowledge retrieval error")
        return tool_error(f"检索异常: {type(e).__name__}: {e}")


def list_datasets() -> str:
    """列出配置的知识库信息"""
    logger.info("📚 正在获取知识库列表...")
    config = _get_config()
    kb_list_url = config.get("kb_list_url", "")
    if not kb_list_url:
        return tool_error("知识库未配置。请在 config.yaml 中配置 kb_list_url")

    target_dataset_ids = _get_dataset_ids(config)
    if not target_dataset_ids:
        return tool_error("未指定数据集。请在 config.yaml 中配置 default_dataset_ids")

    request_body = {"kb_ids": target_dataset_ids, "offset": 0, "limit": 20}

    try:
        response = _http_request("POST", kb_list_url, body=request_body)

        if response.get("status") == 200:
            data = response.get("data", {})
            items = data.get("items", [])
            total = data.get("total", 0)
            logger.info(f"📚 共找到 {total} 个知识库")

            formatted = []
            for item in items:
                formatted.append({
                    "id": item.get("id", ""),
                    "name": item.get("name", ""),
                    "description": item.get("description", ""),
                    "indexers": item.get("indexers", []),
                    "split_method": item.get("split_method", ""),
                    "created_at": item.get("created_at", ""),
                    "updated_at": item.get("updated_at", ""),
                })
                logger.info(f"  📄 {item.get('name', 'unknown')} (ID: {item.get('id', '')})")

            logger.info("✅ 知识库列表获取完成")
            return tool_result({
                "success": True,
                "total": total,
                "datasets": formatted,
            })
        else:
            return tool_error(f"获取数据集失败: {response.get('message', '未知错误')}")

    except Exception as e:
        logger.exception("List datasets error")
        return tool_error(f"获取数据集异常: {type(e).__name__}: {e}")


def check_knowledge_retrieval_requirements() -> bool:
    """检查工具可用性"""
    config = _get_config()
    available = bool(config.get("retrieval_url", ""))
    if available:
        logger.info("✅ 知识库检索工具已就绪")
    else:
        logger.warning("⚠️ 知识库检索工具未配置")
    return available


# =============================================================================
# OpenAI Function-Calling Schema
# =============================================================================

KNOWLEDGE_RETRIEVAL_SCHEMA = {
    "name": "knowledge_retrieval",
    "description": (
        "从知识库中检索相关信息。用于回答用户问题时查找参考资料、"
        "文档内容、FAQ 等。\n\n"
        "使用场景：\n"
        "- 用户询问公司政策、流程时检索内部文档\n"
        "- 回答技术问题时查找相关资料\n"
        "- 需要引用特定文档内容时"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "检索查询词（自然语言或关键词）"
            }
        },
        "required": ["query"]
    }
}

LIST_DATASETS_SCHEMA = {
    "name": "list_datasets",
    "description": "列出知识库基本信息（名称、ID、描述、索引器、更新时间等）",
    "parameters": {
        "type": "object",
        "properties": {},
        "required": []
    }
}


# =============================================================================
# Dynamic Schema Update (Runtime)
# =============================================================================

def _update_knowledge_retrieval_schema():
    """尝试动态获取知识库列表并更新工具 Schema 的描述和参数"""
    try:
        config = _get_config()
        kb_list_url = config.get("kb_list_url", "")
        target_dataset_ids = _get_dataset_ids(config)
        
        kb_ids_enum = []
        kb_desc_lines = []

        if kb_list_url and target_dataset_ids:
            request_body = {"kb_ids": target_dataset_ids, "offset": 0, "limit": 50}
            response = _http_request("POST", kb_list_url, body=request_body)
            
            if response.get("status") == 200:
                items = response.get("data", {}).get("items", [])
                for item in items:
                    ds_id = item.get("id", "")
                    ds_name = item.get("name", "")
                    ds_desc = item.get("description", "暂无描述")
                    kb_ids_enum.append(ds_id)
                    kb_desc_lines.append(f"- `{ds_id}` ({ds_name}): {ds_desc}")

        # 1. 更新 description
        base_desc = "从知识库中检索相关信息。用于回答用户问题时查找参考资料、文档内容、FAQ 等。"
        if kb_desc_lines:
            base_desc += f"\n\n**当前支持以下知识库检索，请根据问题匹配对应的 kb_ids：**\n" + "\n".join(kb_desc_lines)
        KNOWLEDGE_RETRIEVAL_SCHEMA["description"] = base_desc

        # 2. 更新 parameters (添加 kb_ids)
        if kb_ids_enum:
            KNOWLEDGE_RETRIEVAL_SCHEMA["parameters"]["properties"]["kb_ids"] = {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": kb_ids_enum
                },
                "description": "目标知识库 ID 列表。请根据用户问题选择匹配的知识库 ID。若不传或留空，将搜索所有已配置的默认知识库。"
            }
            # Ensure 'query' is still the only required param
            KNOWLEDGE_RETRIEVAL_SCHEMA["parameters"]["required"] = ["query"]

    except Exception as e:
        logger.debug("动态更新知识库 Schema 失败，使用默认配置: %s", e)
        # Fallback: keep default static schema

# Execute update at import time
_update_knowledge_retrieval_schema()


# --- Registry ---


def handle_knowledge_retrieval(args: Dict[str, Any], **kwargs) -> str:
    """Handle the knowledge_retrieval tool call."""
    return knowledge_retrieval(
        query=args.get("query", ""),
        kb_ids=args.get("kb_ids"),
    )


registry.register(
    name="knowledge_retrieval",
    toolset="knowledge_retrieval",
    schema=KNOWLEDGE_RETRIEVAL_SCHEMA,
    handler=handle_knowledge_retrieval,
    check_fn=check_knowledge_retrieval_requirements,
    emoji="🔍",
)

registry.register(
    name="list_datasets",
    toolset="knowledge_retrieval",
    schema=LIST_DATASETS_SCHEMA,
    handler=lambda args, **kw: list_datasets(),
    check_fn=check_knowledge_retrieval_requirements,
    emoji="📚",
)
