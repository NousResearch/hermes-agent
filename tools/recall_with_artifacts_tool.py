#!/usr/bin/env python3
import json
import re
from typing import Any, Dict

from tools.registry import registry
from tools.multimodal_recall_tool import _local_mmrag_connected, multimodal_recall
from tools.session_search_tool import check_session_search_requirements, session_search


def _session_search_available() -> bool:
    return check_session_search_requirements()


def _safe_json_loads(text: str) -> Dict[str, Any]:
    if not text:
        return {}
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {"result": parsed}
    except Exception:
        return {"result": text}


def _run_session_search(query: str, role_filter: str | None = None, limit: int = 3) -> Dict[str, Any]:
    try:
        from hermes_state import SessionDB
    except Exception as e:
        return {"error": f"SessionDB unavailable: {e}"}

    try:
        db = SessionDB()
    except Exception as e:
        return {"error": f"Failed to open session database: {e}"}

    try:
        raw = session_search(query=query, role_filter=role_filter, limit=limit, db=db)
        return _safe_json_loads(raw)
    finally:
        try:
            db.close()
        except Exception:
            pass


def _run_multimodal_recall(**kwargs) -> Dict[str, Any]:
    raw = multimodal_recall(**kwargs)
    parsed = _safe_json_loads(raw)
    structured = parsed.get('structuredContent')
    if isinstance(structured, dict):
        return structured
    return parsed


def _compact_text(text: Any, max_chars: int = 320) -> str:
    raw = '' if text is None else str(text)
    compact = ' '.join(raw.split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 1].rstrip() + '…'


def _normalize_hint(value: str) -> str:
    value = (value or '').strip().strip('"\'`.,;:()[]{}')
    if not value:
        return ''
    if not re.search(r'[A-Za-z0-9]', value):
        return ''
    return value


def _extract_hints_from_text(text: str) -> Dict[str, str]:
    text = text or ''
    hints: Dict[str, str] = {}

    source_ref_patterns = [
        r'\bsource_ref\s*[:=]\s*["\']?([A-Za-z0-9_.:-]+)',
        r'\b(issue:\d+)\b',
        r'\b(project:[A-Za-z0-9_.:-]+)\b',
    ]
    for pattern in source_ref_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            normalized = _normalize_hint(match.group(1))
            if normalized:
                hints['source_ref'] = normalized
                break

    collection_patterns = [
        r'\bcollection_name\s*[:=]\s*["\']?([A-Za-z0-9_.-]+)',
        r'\bcollection\s*[:=]\s*["\']?([A-Za-z0-9_.-]+)',
        r'\b(gitlab_radar_[A-Za-z0-9]+)\b',
    ]
    for pattern in collection_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            normalized = _normalize_hint(match.group(1))
            if normalized:
                hints['collection'] = normalized
                break

    source_type_match = re.search(r'\bsource_type\s*[:=]\s*["\']?([A-Za-z0-9_.-]+)', text, re.IGNORECASE)
    if source_type_match:
        normalized = _normalize_hint(source_type_match.group(1))
        if normalized:
            hints['source_type'] = normalized

    return hints


def _merge_hint_confidence(current: Dict[str, str], key: str, value: str, confidence: str) -> None:
    if not value:
        return
    existing = current.get(key)
    if existing == 'high':
        return
    current[key] = confidence


def _derive_recall_hints(transcript_recall: Dict[str, Any], explicit_source_type: str = '') -> tuple[Dict[str, str], Dict[str, str]]:
    hints: Dict[str, str] = {}
    confidence: Dict[str, str] = {}
    results = transcript_recall.get('results') if isinstance(transcript_recall, dict) else None
    if not isinstance(results, list):
        return hints, confidence

    normalized_explicit_source_type = (explicit_source_type or '').strip().lower()

    for item in results:
        if not isinstance(item, dict):
            continue
        candidate_text = ' '.join(
            str(item.get(key, '') or '')
            for key in ('summary', 'preview')
        )
        extracted = _extract_hints_from_text(candidate_text)
        source_type_from_text = (extracted.get('source_type') or '').lower()

        if 'source_ref' not in hints and extracted.get('source_ref'):
            hints['source_ref'] = extracted['source_ref']
            confidence['source_ref'] = 'high'

        if 'collection' not in hints and extracted.get('collection'):
            hints['collection'] = extracted['collection']
            collection_confidence = 'low'
            if normalized_explicit_source_type and source_type_from_text and source_type_from_text == normalized_explicit_source_type:
                collection_confidence = 'high'
            confidence['collection'] = collection_confidence

        if 'source_type' not in hints and extracted.get('source_type'):
            hints['source_type'] = extracted['source_type']
            _merge_hint_confidence(confidence, 'source_type', extracted['source_type'], 'medium')

        if 'source_ref' in hints and 'collection' in hints and confidence.get('collection') == 'high':
            break

    return hints, confidence


def _augment_query_with_soft_hints(query: str, hints: Dict[str, str], confidence: Dict[str, str]) -> str:
    base = (query or '').strip()
    additions = []
    lowered_query = base.lower()
    if hints.get('collection') and confidence.get('collection') != 'high':
        additions.append(f'collection {hints["collection"]}')
    if hints.get('source_type') and confidence.get('source_type') == 'medium':
        source_type_hint = hints['source_type']
        generic_attachment = source_type_hint.lower() in {'attachment', 'attachments'}
        broad_pdf_screenshot = any(term in lowered_query for term in ['pdf', 'screenshot', '截圖', '附件', '證據'])
        if not (generic_attachment and broad_pdf_screenshot):
            additions.append(f'source_type {source_type_hint}')
    if not additions:
        return base
    suffix = ' '.join(additions)
    return f'{base} {suffix}'.strip()


def _query_requires_source_ref_corroboration(query: str) -> bool:
    lowered = (query or '').lower()
    strong_multimodal_terms = [
        'screenshot', 'pdf', 'attachment', 'attachments', 'image', 'ocr', 'artifact', 'artifacts',
        '截圖', '附件', '圖片', '影像', '證據', '文件', '檔案', '報告',
    ]
    return any(term in lowered for term in strong_multimodal_terms)


def _query_is_broad_pdf_screenshot(query: str) -> bool:
    lowered = (query or '').lower()
    has_pdf = 'pdf' in lowered
    has_visual = any(term in lowered for term in ['screenshot', '截圖', '附件', '證據'])
    return has_pdf and has_visual


def _resolve_effective_source_ref(
    query: str,
    explicit_source_ref: str,
    derived_hints: Dict[str, str],
    hint_confidence: Dict[str, str],
) -> str:
    if explicit_source_ref:
        return explicit_source_ref
    derived_source_ref = derived_hints.get('source_ref', '')
    if not derived_source_ref:
        return ''
    lowered_query = (query or '').lower()
    if derived_source_ref.lower() in lowered_query:
        return derived_source_ref
    if not _query_requires_source_ref_corroboration(query):
        return derived_source_ref
    if hint_confidence.get('collection') == 'high':
        return derived_source_ref
    return ''


def _extract_blocker_signals(text: str) -> list[str]:
    lowered = (text or '').lower()
    signals: list[str] = []
    blocker_patterns = [
        (r'ocr validation', 'ocr validation'),
        (r'blocker[:\s]+([^.;\n]+)', None),
        (r'waiting customer reply', 'waiting customer reply'),
        (r'customer waiting', 'customer waiting'),
    ]
    for pattern, label in blocker_patterns:
        match = re.search(pattern, lowered, re.IGNORECASE)
        if not match:
            continue
        value = label or _normalize_hint(match.group(1))
        if value and value not in signals:
            signals.append(value)
    return signals


def _query_looks_multimodal(query: str) -> bool:
    lowered = (query or '').lower()
    if not lowered.strip():
        return False
    strong_terms = [
        'screenshot', 'pdf', 'attachment', 'attachments', 'image', 'ocr', 'evidence', 'artifact', 'artifacts',
        'report', 'document', 'file', 'files', 'photo', 'photos', 'scan', 'scanned',
        '截圖', '附件', '圖片', '影像', '照片', '證據', '文件', '檔案', '報告', '掃描', 'pdf',
    ]
    if any(term in lowered for term in strong_terms):
        return True
    status_terms = ['customer waiting', 'waiting customer', 'eta', 'blocker', '客戶等待', '等待客戶', '阻塞', '卡住']
    matched = [term for term in status_terms if term in lowered]
    return len(matched) >= 2


def _score_evidence_alignment(query: str, item: Dict[str, Any], resolved_source_ref: str = '') -> float:
    lowered_query = (query or '').lower()
    text = ' '.join(
        str(item.get(key, '') or '')
        for key in ('text', 'source_type', 'modality', 'source_ref', 'collection_name', 'source_path')
    ).lower()
    score = float(item.get('score') or 0.0)

    if resolved_source_ref and (item.get('source_ref') or '') == resolved_source_ref:
        score += 5.0

    keyword_weights = {
        'screenshot': 3.0,
        'image': 2.5,
        'attachment': 2.0,
        'pdf': 2.0,
        'ocr': 2.0,
        'evidence': 1.5,
        'eta': 1.5,
        'blocker': 1.5,
    }
    for term, weight in keyword_weights.items():
        if term in lowered_query and term in text:
            score += weight

    for term in ['截圖', '圖片', '附件', 'pdf', '證據', 'eta', 'blocker']:
        if term.lower() in lowered_query and term.lower() in text:
            score += 1.5

    issue_refs = re.findall(r'issue:\d+', lowered_query)
    for ref in issue_refs:
        if ref in text:
            score += 4.0

    if 'screenshot' in lowered_query and (item.get('source_type') == 'image' or item.get('modality') == 'image'):
        score += 2.0
    if 'pdf' in lowered_query and (item.get('source_type') == 'pdf' or item.get('modality') == 'pdf'):
        score += 2.0

    ocr_error_focused = any(term in lowered_query for term in ['ocr', '錯誤訊息', 'error'])
    if ocr_error_focused:
        if any(term in text for term in ['錯誤訊息', 'technical error', 'error message', 'technical detail', 'custom error', 'non-customized errors', 'detail leakage']):
            score += 4.0
        if (item.get('source_type') == 'pdf' or item.get('modality') == 'pdf'):
            score += 2.0
        if 'status waiting customer reply' in text or 'waiting customer reply' in text:
            score -= 2.0

    return score


def _rank_top_evidence_for_prompt(query: str, top_evidence: list[Dict[str, Any]], resolved_source_ref: str = '') -> list[Dict[str, Any]]:
    ranked = [item for item in top_evidence if isinstance(item, dict)]
    ranked.sort(
        key=lambda item: _score_evidence_alignment(query, item, resolved_source_ref=resolved_source_ref),
        reverse=True,
    )
    return ranked


def _cluster_evidence_items(top_evidence: list[Dict[str, Any]]) -> list[list[Dict[str, Any]]]:
    clusters: dict[tuple[str, str], list[Dict[str, Any]]] = {}
    for item in top_evidence:
        if not isinstance(item, dict):
            continue
        source_ref = item.get('source_ref') or ''
        collection = item.get('collection_name') or ''
        key = (source_ref, collection)
        clusters.setdefault(key, []).append(item)
    return list(clusters.values())


def _score_evidence_cluster(query: str, cluster: list[Dict[str, Any]], resolved_source_ref: str = '') -> float:
    if not cluster:
        return float('-inf')
    scores = [_score_evidence_alignment(query, item, resolved_source_ref=resolved_source_ref) for item in cluster]
    return max(scores) + (0.05 * len(cluster))


def _select_focus_cluster(query: str, top_evidence: list[Dict[str, Any]], resolved_source_ref: str = '') -> list[Dict[str, Any]]:
    clusters = _cluster_evidence_items(top_evidence)
    if not clusters:
        return []
    clusters.sort(key=lambda cluster: _score_evidence_cluster(query, cluster, resolved_source_ref=resolved_source_ref), reverse=True)
    focus = clusters[0]
    focus.sort(key=lambda item: _score_evidence_alignment(query, item, resolved_source_ref=resolved_source_ref), reverse=True)
    return focus


def _build_issue_evidence_brief(
    artifact_recall: Dict[str, Any],
    *,
    query: str = '',
    source_ref: str = '',
    derived_hints: Dict[str, str] | None = None,
) -> Dict[str, Any]:
    derived_hints = derived_hints or {}
    if not isinstance(artifact_recall, dict):
        return {}

    top_evidence = artifact_recall.get('top_evidence')
    if not isinstance(top_evidence, list) or not top_evidence:
        return {}

    collections: list[str] = []
    modalities_seen: list[str] = []
    top_sources: list[str] = []
    blocker_signals: list[str] = []
    combined_text_parts: list[str] = []
    resolved_source_ref = source_ref or ''
    derived_source_ref = derived_hints.get('source_ref', '')
    if not resolved_source_ref and derived_source_ref:
        if any((item.get('source_ref') or '') == derived_source_ref for item in top_evidence if isinstance(item, dict)):
            resolved_source_ref = derived_source_ref

    ranked_evidence = _rank_top_evidence_for_prompt(query, top_evidence, resolved_source_ref=resolved_source_ref)
    screenshot_focused = any(term in (query or '').lower() for term in ['screenshot', '截圖'])
    ocr_error_focused = any(term in (query or '').lower() for term in ['ocr', '錯誤訊息', 'error'])
    focus_source_ref = ''
    focus_collection = ''
    focus_cluster = _select_focus_cluster(query, ranked_evidence, resolved_source_ref=resolved_source_ref)
    if focus_cluster:
        focus_source_ref = focus_cluster[0].get('source_ref') or ''
        focus_collection = focus_cluster[0].get('collection_name') or artifact_recall.get('retrieval_notes', {}).get('collection_name', '')
    if screenshot_focused and focus_source_ref:
        resolved_source_ref = focus_source_ref

    evidence_for_brief = focus_cluster or ranked_evidence

    for item in evidence_for_brief:
        if not isinstance(item, dict):
            continue
        item_source_ref = item.get('source_ref') or ''
        item_collection = item.get('collection_name') or artifact_recall.get('retrieval_notes', {}).get('collection_name', '')
        if screenshot_focused:
            if focus_source_ref and item_source_ref and item_source_ref != focus_source_ref:
                continue
            if focus_collection and item_collection and item_collection != focus_collection and item_source_ref != focus_source_ref:
                continue
        if not resolved_source_ref and item_source_ref:
            resolved_source_ref = item_source_ref
        collection_name = item_collection
        if collection_name and collection_name not in collections:
            collections.append(collection_name)
        modality = item.get('modality') or ''
        if modality and modality not in modalities_seen:
            modalities_seen.append(modality)
        source_path = item.get('source_path') or ''
        if source_path and source_path not in top_sources:
            top_sources.append(source_path)
        text = item.get('text') or ''
        if text:
            combined_text_parts.append(text)
            for signal in _extract_blocker_signals(text):
                if signal not in blocker_signals:
                    blocker_signals.append(signal)

    extracted_fields = artifact_recall.get('extracted_fields') if isinstance(artifact_recall.get('extracted_fields'), dict) else {}
    status_signals = list(extracted_fields.get('customer_waiting_signals') or [])
    eta_value = extracted_fields.get('eta')
    eta_signals = [eta_value] if eta_value else []

    if not eta_signals:
        joined_lower = ' '.join(combined_text_parts).lower()
        for pattern in [r'eta[:\s]+([^.;\n]+)', r'next tuesday', r'next monday']:
            match = re.search(pattern, joined_lower)
            if not match:
                continue
            eta_candidate = pattern if pattern in {'next tuesday', 'next monday'} else _normalize_hint(match.group(1))
            if eta_candidate:
                eta_signals = [eta_candidate]
                break

    preferred_blocker = ''
    for candidate in ['ocr validation']:
        if candidate in blocker_signals:
            preferred_blocker = candidate
            break
    if not preferred_blocker and blocker_signals:
        preferred_blocker = blocker_signals[0]

    brief_bits = []
    if resolved_source_ref:
        brief_bits.append(f'{resolved_source_ref} evidence')
    if status_signals:
        brief_bits.append('shows customer waiting status')
    if eta_signals:
        brief_bits.append(f'ETA {eta_signals[0]}')
    if preferred_blocker:
        brief_bits.append(f'blocker {preferred_blocker}')
    if screenshot_focused and combined_text_parts:
        focus_text = _compact_text(combined_text_parts[0], 120)
        if focus_text:
            brief_bits.append(f'focus {focus_text}')
    if ocr_error_focused and combined_text_parts:
        focus_text = _compact_text(combined_text_parts[0], 120)
        if focus_text and focus_text not in brief_bits:
            brief_bits.append(f'focus {focus_text}')
    evidence_brief = '; '.join(brief_bits).strip()

    return {
        'source_ref': resolved_source_ref,
        'collections': collections,
        'modalities_seen': modalities_seen,
        'status_signals': status_signals,
        'eta_signals': eta_signals,
        'blocker_signals': blocker_signals,
        'top_sources': top_sources[:3],
        'evidence_brief': evidence_brief,
        'preferred_blocker': preferred_blocker,
    }


def _build_radar_ready_summary(issue_evidence_brief: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(issue_evidence_brief, dict) or not issue_evidence_brief:
        return {}

    source_ref = issue_evidence_brief.get('source_ref', '')
    status_signals = list(issue_evidence_brief.get('status_signals') or [])
    eta_signals = list(issue_evidence_brief.get('eta_signals') or [])
    blocker = issue_evidence_brief.get('preferred_blocker') or ''
    modalities_seen = list(issue_evidence_brief.get('modalities_seen') or [])
    collections = list(issue_evidence_brief.get('collections') or [])
    top_sources = list(issue_evidence_brief.get('top_sources') or [])

    current_status = status_signals[0] if status_signals else ''
    eta = eta_signals[0] if eta_signals else ''

    next_step = ''
    if current_status == 'customer_waiting' and eta:
        next_step = f'Update the tracking issue with ETA {eta} and follow up with the customer.'
    elif blocker:
        next_step = f'Clarify or resolve blocker: {blocker}.'
    elif top_sources:
        next_step = 'Review the top evidence sources and update the tracking issue.'

    return {
        'source_ref': source_ref,
        'current_evidence_backed_status': current_status,
        'eta': eta,
        'blocker': blocker,
        'evidence_basis': modalities_seen,
        'suggested_next_step': next_step,
        'provenance': {
            'collection': collections[0] if collections else '',
            'top_sources': top_sources[:3],
        },
    }


def _build_comment_ready_summary(radar_ready_summary: Dict[str, Any]) -> str:
    if not isinstance(radar_ready_summary, dict) or not radar_ready_summary:
        return ''

    lines = []
    status = radar_ready_summary.get('current_evidence_backed_status') or ''
    eta = radar_ready_summary.get('eta') or ''
    blocker = radar_ready_summary.get('blocker') or ''
    evidence_basis = ', '.join(radar_ready_summary.get('evidence_basis') or [])
    next_step = radar_ready_summary.get('suggested_next_step') or ''
    source_ref = radar_ready_summary.get('source_ref') or ''
    provenance = radar_ready_summary.get('provenance') or {}
    collection = provenance.get('collection') or ''
    top_sources = provenance.get('top_sources') or []

    if status:
        lines.append(f'Current evidence-backed status: {status}')
    if eta:
        lines.append(f'ETA: {eta}')
    if blocker:
        lines.append(f'Blocker: {blocker}')
    if evidence_basis:
        lines.append(f'Evidence basis: {evidence_basis}')
    if next_step:
        lines.append(f'Suggested next step: {next_step}')
    if source_ref:
        lines.append(f'Source ref: {source_ref}')
    if collection:
        lines.append(f'Collection: {collection}')
    if top_sources:
        lines.append('Top sources:')
        lines.extend(f'- {path}' for path in top_sources)

    return '\n'.join(lines).strip()


def _build_evidence_channel_summary(
    query: str,
    transcript_recall: Dict[str, Any],
    artifact_recall: Dict[str, Any],
) -> Dict[str, Any]:
    lowered_query = (query or '').lower()
    ambiguity_terms = ['來自對話', '來自附件', 'transcript', 'artifact', '附件', '對話']
    is_ambiguity_query = any(term in lowered_query for term in ambiguity_terms)

    transcript_support = ''
    results = transcript_recall.get('results') if isinstance(transcript_recall, dict) else None
    if isinstance(results, list) and results:
        first = results[0]
        if isinstance(first, dict):
            transcript_support = _compact_text(first.get('summary') or first.get('preview') or '', 220)

    artifact_support = ''
    if isinstance(artifact_recall, dict):
        artifact_support = _compact_text(artifact_recall.get('summary') or artifact_recall.get('result') or '', 220)
        if not artifact_support:
            top = artifact_recall.get('top_evidence')
            if isinstance(top, list) and top:
                artifact_support = _compact_text((top[0] or {}).get('text') or '', 220)

    if not is_ambiguity_query and not (transcript_support and artifact_support):
        return {}

    if transcript_support and artifact_support:
        preferred = 'mixed'
    elif artifact_support:
        preferred = 'artifact'
    elif transcript_support:
        preferred = 'transcript'
    else:
        preferred = 'none'

    return {
        'transcript_support': transcript_support,
        'artifact_support': artifact_support,
        'preferred_evidence_channel': preferred,
    }


def _build_combined_summary(query: str, transcript_recall: Dict[str, Any], artifact_recall: Dict[str, Any]) -> str:
    parts = []
    if query:
        parts.append(f"Hybrid recall for query: {query}.")

    transcript_results = transcript_recall.get('results') if isinstance(transcript_recall, dict) else None
    if isinstance(transcript_results, list) and transcript_results:
        first = transcript_results[0]
        if isinstance(first, dict):
            summary = _compact_text(first.get('summary') or first.get('preview') or '')
            if summary:
                parts.append(f"Transcript recall found {len(transcript_results)} relevant session(s); top match: {summary}")
    elif isinstance(transcript_recall, dict) and transcript_recall.get('error'):
        parts.append(f"Transcript recall unavailable: {_compact_text(transcript_recall['error'], 180)}")

    if isinstance(artifact_recall, dict):
        artifact_summary = _compact_text(artifact_recall.get('summary') or artifact_recall.get('result') or '')
        if artifact_summary:
            parts.append(f"Artifact recall: {artifact_summary}")
        elif artifact_recall.get('artifacts'):
            parts.append(f"Artifact recall returned {artifact_recall.get('count', len(artifact_recall.get('artifacts', [])))} recent artifact(s).")
        elif artifact_recall.get('error'):
            parts.append(f"Artifact recall unavailable: {_compact_text(artifact_recall['error'], 180)}")

    return ' '.join(p.strip() for p in parts if p and str(p).strip())


def recall_with_artifacts(
    query: str = '',
    role_filter: str = '',
    session_limit: int = 3,
    artifact_top_k: int = 5,
    source_ref: str = '',
    source_type: str = '',
    modality: str = '',
    collection: str = '',
) -> str:
    transcript_recall: Dict[str, Any]
    artifact_recall: Dict[str, Any]

    if _session_search_available():
        transcript_recall = _run_session_search(query=query, role_filter=role_filter or None, limit=session_limit)
    else:
        transcript_recall = {'error': 'session_search is not available'}

    derived_hints, hint_confidence = _derive_recall_hints(transcript_recall, explicit_source_type=source_type)
    effective_source_ref = _resolve_effective_source_ref(
        query,
        source_ref,
        derived_hints,
        hint_confidence,
    )
    use_collection_hard_filter = bool(collection)
    if not use_collection_hard_filter and derived_hints.get('collection'):
        if hint_confidence.get('collection') == 'high':
            use_collection_hard_filter = True
        elif _query_is_broad_pdf_screenshot(query) and not effective_source_ref:
            use_collection_hard_filter = True
    effective_collection = collection or (derived_hints.get('collection', '') if use_collection_hard_filter else '')
    effective_query = _augment_query_with_soft_hints(query, derived_hints, hint_confidence)
    has_explicit_artifact_scope = any(
        (value or '').strip()
        for value in [source_ref, source_type, modality, collection, effective_source_ref, effective_collection]
    )
    should_run_artifact_recall = (not (query or '').strip()) or has_explicit_artifact_scope or _query_looks_multimodal(query)

    if _local_mmrag_connected() and should_run_artifact_recall:
        mm_action = 'recent' if not (query or '').strip() else 'evidence'
        artifact_recall = _run_multimodal_recall(
            action=mm_action,
            query=effective_query,
            top_k=artifact_top_k,
            source_ref=effective_source_ref,
            source_type=source_type,
            modality=modality,
            collection=effective_collection,
        )
    elif _local_mmrag_connected():
        artifact_recall = {
            'skipped': True,
            'reason': 'query does not appear multimodal and no artifact filters were provided',
        }
    else:
        artifact_recall = {'error': 'local_mmrag MCP server is not connected'}

    combined = {
        'query': query,
        'transcript_recall': transcript_recall,
        'artifact_recall': artifact_recall,
        'derived_hints': derived_hints,
        'hint_confidence': hint_confidence,
        'effective_artifact_query': effective_query,
        'issue_evidence_brief': _build_issue_evidence_brief(
            artifact_recall,
            query=query,
            source_ref=effective_source_ref,
            derived_hints=derived_hints,
        ),
        'combined_summary': _build_combined_summary(query, transcript_recall, artifact_recall),
    }
    combined['evidence_channel_summary'] = _build_evidence_channel_summary(query, transcript_recall, artifact_recall)
    combined['radar_ready_summary'] = _build_radar_ready_summary(combined['issue_evidence_brief'])
    combined['comment_ready_summary'] = _build_comment_ready_summary(combined['radar_ready_summary'])
    return json.dumps(combined, ensure_ascii=False)


RECALL_WITH_ARTIFACTS_SCHEMA = {
    'name': 'recall_with_artifacts',
    'description': 'Hybrid recall that combines transcript-first session_search with multimodal_recall artifact evidence. Use this first when the user refers to previous work plus screenshots, PDFs, attachments, OCR, evidence packs, or wants both conversation context and supporting artifacts.',
    'parameters': {
        'type': 'object',
        'properties': {
            'query': {'type': 'string', 'description': 'Recall query spanning prior conversations and artifacts.'},
            'role_filter': {'type': 'string', 'description': 'Optional role filter passed to session_search, e.g. user,assistant.'},
            'session_limit': {'type': 'integer', 'description': 'Max transcript sessions to summarize (default 3).', 'default': 3},
            'artifact_top_k': {'type': 'integer', 'description': 'Max artifact hits or recent artifacts to return (default 5).', 'default': 5},
            'source_ref': {'type': 'string', 'description': 'Optional source reference filter for multimodal recall.'},
            'source_type': {'type': 'string', 'description': 'Optional source type filter for multimodal recall.'},
            'modality': {'type': 'string', 'description': 'Optional modality filter for multimodal recall.'},
            'collection': {'type': 'string', 'description': 'Optional local-mmrag collection filter.'},
        },
        'required': [],
    },
}


registry.register(
    name='recall_with_artifacts',
    toolset='memory',
    schema=RECALL_WITH_ARTIFACTS_SCHEMA,
    handler=lambda args, **kw: recall_with_artifacts(
        query=args.get('query', ''),
        role_filter=args.get('role_filter', ''),
        session_limit=args.get('session_limit', 3),
        artifact_top_k=args.get('artifact_top_k', 5),
        source_ref=args.get('source_ref', ''),
        source_type=args.get('source_type', ''),
        modality=args.get('modality', ''),
        collection=args.get('collection', ''),
    ),
    check_fn=lambda: _session_search_available() or _local_mmrag_connected(),
    emoji='🧩',
)
