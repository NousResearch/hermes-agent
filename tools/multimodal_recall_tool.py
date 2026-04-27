#!/usr/bin/env python3
import json
from typing import Any, Dict

from tools.registry import registry
from tools import mcp_tool


def _local_mmrag_connected() -> bool:
    with mcp_tool._lock:
        server = mcp_tool._servers.get('local_mmrag')
    return server is not None and getattr(server, 'session', None) is not None


def _call_local_mmrag(tool_name: str, args: Dict[str, Any]) -> str:
    with mcp_tool._lock:
        server = mcp_tool._servers.get('local_mmrag')
    if not server or not getattr(server, 'session', None):
        return json.dumps({'error': "local_mmrag MCP server is not connected"}, ensure_ascii=False)

    async def _call():
        result = await server.session.call_tool(tool_name, arguments=args)
        if result.isError:
            text = ''.join(block.text for block in (result.content or []) if hasattr(block, 'text'))
            return json.dumps({'error': text or 'MCP tool returned an error'}, ensure_ascii=False)
        parts = [block.text for block in (result.content or []) if hasattr(block, 'text')]
        if getattr(result, 'structuredContent', None) is not None:
            if parts:
                return json.dumps({'result': '\n'.join(parts), 'structuredContent': result.structuredContent}, ensure_ascii=False)
            return json.dumps({'result': result.structuredContent}, ensure_ascii=False)
        text = '\n'.join(parts) if parts else ''
        return text if text else json.dumps({'result': ''}, ensure_ascii=False)

    return mcp_tool._run_on_mcp_loop(_call(), timeout=60)


def multimodal_recall(action: str = 'evidence', query: str = '', source_ref: str = '', source_type: str = '', modality: str = '', collection: str = '', top_k: int = 5) -> str:
    action = (action or 'evidence').strip().lower()
    if action == 'recent':
        return _call_local_mmrag('mm_recall_recent_artifacts', {'limit': top_k})

    args = {'query': query, 'top_k': top_k}
    if source_ref:
        args['source_ref'] = source_ref
    if source_type:
        args['source_type'] = source_type
    if modality:
        args['modality'] = modality
    if collection:
        args['collection'] = collection

    if action == 'search':
        return _call_local_mmrag('mm_recall_search', args)
    return _call_local_mmrag('mm_recall_get_evidence_pack', args)


MULTIMODAL_RECALL_SCHEMA = {
    'name': 'multimodal_recall',
    'description': 'Recall multimodal evidence (PDFs, OCR, screenshots, evidence packs) through the local_mmrag MCP layer. Use this when the user refers to a prior report, screenshot, attachment, OCR result, or evidence pack rather than transcript-only memory. If the user likely needs both past conversation context and artifact evidence, prefer recall_with_artifacts instead.',
    'parameters': {
        'type': 'object',
        'properties': {
            'action': {
                'type': 'string',
                'description': "One of: 'evidence' (default), 'search', or 'recent'."
            },
            'query': {
                'type': 'string',
                'description': 'Natural-language recall query, e.g. customer waiting eta, prior pentest PDF, screenshot with error message.'
            },
            'source_ref': {
                'type': 'string',
                'description': 'Optional stable source reference filter, e.g. issue:123 or project:iid.'
            },
            'source_type': {
                'type': 'string',
                'description': 'Optional filter such as issue, attachment, wiki, local_file.'
            },
            'modality': {
                'type': 'string',
                'description': 'Optional modality filter such as text, image, pdf, image_ocr.'
            },
            'collection': {
                'type': 'string',
                'description': 'Optional collection name to limit recall to a known local-mmrag collection.'
            },
            'top_k': {
                'type': 'integer',
                'description': 'Max number of recalled results/artifacts to return (default 5).',
                'default': 5,
            },
        },
        'required': [],
    },
}

registry.register(
    name='multimodal_recall',
    toolset='memory',
    schema=MULTIMODAL_RECALL_SCHEMA,
    handler=lambda args, **kw: multimodal_recall(
        action=args.get('action', 'evidence'),
        query=args.get('query', ''),
        source_ref=args.get('source_ref', ''),
        source_type=args.get('source_type', ''),
        modality=args.get('modality', ''),
        collection=args.get('collection', ''),
        top_k=args.get('top_k', 5),
    ),
    check_fn=_local_mmrag_connected,
    emoji='🧠',
)
