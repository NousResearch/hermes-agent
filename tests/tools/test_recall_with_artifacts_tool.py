import json
from tools.recall_with_artifacts_tool import RECALL_WITH_ARTIFACTS_SCHEMA


def test_recall_with_artifacts_schema_mentions_combined_context():
    description = RECALL_WITH_ARTIFACTS_SCHEMA['description']
    assert 'session_search' in description
    assert 'multimodal_recall' in description
    assert 'Use this first' in description


def test_recall_with_artifacts_combines_session_and_multimodal(monkeypatch):
    import tools.recall_with_artifacts_tool as rwt

    monkeypatch.setattr(rwt, '_session_search_available', lambda: True)
    monkeypatch.setattr(rwt, '_local_mmrag_connected', lambda: True)
    monkeypatch.setattr(
        rwt,
        '_run_session_search',
        lambda query, role_filter=None, limit=3: {
            'success': True,
            'query': query,
            'results': [
                {
                    'session_id': 's1',
                    'when': 'April 18, 2026 at 01:39 AM',
                    'summary': 'We used multimodal recall to inspect customer waiting ETA evidence.',
                }
            ],
            'count': 1,
        },
    )
    monkeypatch.setattr(
        rwt,
        '_run_multimodal_recall',
        lambda **kwargs: {
            'query': kwargs['query'],
            'summary': 'OCR shows waiting customer reply and ETA next Monday.',
            'top_evidence': [
                {'source_path': '/tmp/status.png', 'source_type': 'image', 'score': 0.99}
            ],
            'retrieval_notes': {'collection_name': 'manual_cli_check'},
        },
    )

    result = json.loads(rwt.recall_with_artifacts(query='customer waiting ETA', session_limit=2, artifact_top_k=3))
    assert result['query'] == 'customer waiting ETA'
    assert result['transcript_recall']['count'] == 1
    assert result['artifact_recall']['retrieval_notes']['collection_name'] == 'manual_cli_check'
    assert 'customer waiting ETA' in result['combined_summary']
    assert 'next Monday' in result['combined_summary']


def test_recall_with_artifacts_handles_missing_session_search(monkeypatch):
    import tools.recall_with_artifacts_tool as rwt

    monkeypatch.setattr(rwt, '_session_search_available', lambda: False)
    monkeypatch.setattr(rwt, '_local_mmrag_connected', lambda: True)
    monkeypatch.setattr(
        rwt,
        '_run_multimodal_recall',
        lambda **kwargs: {'summary': 'Artifact-only evidence', 'top_evidence': [], 'retrieval_notes': {}},
    )

    result = json.loads(rwt.recall_with_artifacts(query='artifact only'))
    assert 'error' in result['transcript_recall']
    assert result['artifact_recall']['summary'] == 'Artifact-only evidence'


def test_recall_with_artifacts_recent_mode(monkeypatch):
    import tools.recall_with_artifacts_tool as rwt

    monkeypatch.setattr(rwt, '_session_search_available', lambda: True)
    monkeypatch.setattr(rwt, '_local_mmrag_connected', lambda: True)
    monkeypatch.setattr(
        rwt,
        '_run_session_search',
        lambda query, role_filter=None, limit=3: {
            'success': True,
            'mode': 'recent',
            'results': [{'session_id': 's1'}],
            'count': 1,
        },
    )
    monkeypatch.setattr(
        rwt,
        '_run_multimodal_recall',
        lambda **kwargs: {'artifacts': [{'source_path': '/tmp/a.png'}], 'count': 1},
    )

    result = json.loads(rwt.recall_with_artifacts(query='', session_limit=1, artifact_top_k=1))
    assert result['transcript_recall']['mode'] == 'recent'
    assert result['artifact_recall']['count'] == 1


def test_recall_with_artifacts_compacts_long_session_summary(monkeypatch):
    import tools.recall_with_artifacts_tool as rwt

    monkeypatch.setattr(rwt, '_session_search_available', lambda: True)
    monkeypatch.setattr(rwt, '_local_mmrag_connected', lambda: True)
    monkeypatch.setattr(
        rwt,
        '_run_session_search',
        lambda query, role_filter=None, limit=3: {
            'success': True,
            'query': query,
            'results': [{'summary': 'A' * 1200}],
            'count': 1,
        },
    )
    monkeypatch.setattr(
        rwt,
        '_run_multimodal_recall',
        lambda **kwargs: {'summary': 'B' * 1200, 'top_evidence': []},
    )

    result = json.loads(rwt.recall_with_artifacts(query='very long'))
    assert len(result['combined_summary']) < 1000


def test_recall_with_artifacts_derives_high_confidence_source_ref_and_soft_collection_hint(monkeypatch):
    import tools.recall_with_artifacts_tool as rwt

    captured = {}
    monkeypatch.setattr(rwt, '_session_search_available', lambda: True)
    monkeypatch.setattr(rwt, '_local_mmrag_connected', lambda: True)
    monkeypatch.setattr(
        rwt,
        '_run_session_search',
        lambda query, role_filter=None, limit=3: {
            'success': True,
            'query': query,
            'results': [
                {
                    'summary': 'We used source_ref issue:123 and collection_name: "gitlab_radar_ae4adda6785fd089" while reviewing attachment evidence.'
                }
            ],
            'count': 1,
        },
    )

    def _fake_mm(**kwargs):
        captured.update(kwargs)
        return {'summary': 'artifact summary', 'top_evidence': []}

    monkeypatch.setattr(rwt, '_run_multimodal_recall', _fake_mm)

    result = json.loads(rwt.recall_with_artifacts(query='customer waiting ETA'))
    assert captured['source_ref'] == 'issue:123'
    assert captured['collection'] == ''
    assert 'gitlab_radar_ae4adda6785fd089' in captured['query']
    assert result['derived_hints']['source_ref'] == 'issue:123'
    assert result['derived_hints']['collection'] == 'gitlab_radar_ae4adda6785fd089'
    assert result['hint_confidence']['source_ref'] == 'high'
    assert result['hint_confidence']['collection'] == 'low'


def test_explicit_hints_override_derived_hints(monkeypatch):
    import tools.recall_with_artifacts_tool as rwt

    captured = {}
    monkeypatch.setattr(rwt, '_session_search_available', lambda: True)
    monkeypatch.setattr(rwt, '_local_mmrag_connected', lambda: True)
    monkeypatch.setattr(
        rwt,
        '_run_session_search',
        lambda query, role_filter=None, limit=3: {
            'success': True,
            'results': [{'summary': 'source_ref issue:123 collection_name: "gitlab_radar_from_session"'}],
            'count': 1,
        },
    )

    def _fake_mm(**kwargs):
        captured.update(kwargs)
        return {'summary': 'artifact summary'}

    monkeypatch.setattr(rwt, '_run_multimodal_recall', _fake_mm)

    json.loads(
        rwt.recall_with_artifacts(
            query='customer waiting ETA',
            source_ref='issue:999',
            collection='manual_cli_check',
        )
    )
    assert captured['source_ref'] == 'issue:999'
    assert captured['collection'] == 'manual_cli_check'


def test_derived_hints_ignore_ellipsis_like_placeholders(monkeypatch):
    import tools.recall_with_artifacts_tool as rwt

    captured = {}
    monkeypatch.setattr(rwt, '_session_search_available', lambda: True)
    monkeypatch.setattr(rwt, '_local_mmrag_connected', lambda: True)
    monkeypatch.setattr(
        rwt,
        '_run_session_search',
        lambda query, role_filter=None, limit=3: {
            'success': True,
            'results': [{'summary': 'related source_ref: ... collection_name: test_retrieve_returns_ranked_candidates'}],
            'count': 1,
        },
    )

    def _fake_mm(**kwargs):
        captured.update(kwargs)
        return {'summary': 'artifact summary'}

    monkeypatch.setattr(rwt, '_run_multimodal_recall', _fake_mm)

    result = json.loads(rwt.recall_with_artifacts(query='customer waiting ETA'))
    assert captured['source_ref'] == ''
    assert captured['collection'] == ''
    assert 'test_retrieve_returns_ranked_candidates' in captured['query']
    assert result['derived_hints'].get('source_ref', '') == ''


def test_high_confidence_collection_hint_is_used_as_hard_filter(monkeypatch):
    import tools.recall_with_artifacts_tool as rwt

    captured = {}
    monkeypatch.setattr(rwt, '_session_search_available', lambda: True)
    monkeypatch.setattr(rwt, '_local_mmrag_connected', lambda: True)
    monkeypatch.setattr(
        rwt,
        '_run_session_search',
        lambda query, role_filter=None, limit=3: {
            'success': True,
            'results': [{'summary': 'collection_name: "manual_cli_check" source_type: attachment'}],
            'count': 1,
        },
    )

    def _fake_mm(**kwargs):
        captured.update(kwargs)
        return {'summary': 'artifact summary'}

    monkeypatch.setattr(rwt, '_run_multimodal_recall', _fake_mm)

    result = json.loads(rwt.recall_with_artifacts(query='customer waiting ETA', source_type='attachment'))
    assert captured['collection'] == 'manual_cli_check'
    assert result['hint_confidence']['collection'] == 'high'


def test_recall_with_artifacts_skips_artifact_recall_for_non_multimodal_query_without_hints(monkeypatch):
    import tools.recall_with_artifacts_tool as rwt

    called = {'value': False}
    monkeypatch.setattr(rwt, '_session_search_available', lambda: True)
    monkeypatch.setattr(rwt, '_local_mmrag_connected', lambda: True)
    monkeypatch.setattr(
        rwt,
        '_run_session_search',
        lambda query, role_filter=None, limit=3: {
            'success': True,
            'results': [
                {'summary': 'We enabled multimodal provider integration, but did not discuss any screenshot or attachment for this branch question.'}
            ],
            'count': 1,
        },
    )

    def _fake_mm(**kwargs):
        called['value'] = True
        return {'summary': 'should not run'}

    monkeypatch.setattr(rwt, '_run_multimodal_recall', _fake_mm)

    result = json.loads(rwt.recall_with_artifacts(query='what branch name did we use'))
    assert called['value'] is False
    assert result['artifact_recall']['skipped'] is True
    assert result['artifact_recall']['reason'] == 'query does not appear multimodal and no artifact filters were provided'


def test_transcript_derived_source_ref_stays_soft_for_screenshot_query_without_corroboration(monkeypatch):
    import tools.recall_with_artifacts_tool as rwt

    captured = {}
    monkeypatch.setattr(rwt, '_session_search_available', lambda: True)
    monkeypatch.setattr(rwt, '_local_mmrag_connected', lambda: True)
    monkeypatch.setattr(
        rwt,
        '_run_session_search',
        lambda query, role_filter=None, limit=3: {
            'success': True,
            'results': [
                {
                    'summary': 'Earlier drifted session mentioned source_ref issue:987 and collection_name: "gitlab_radar_49942185f88bf27f" while discussing screenshot evidence.'
                }
            ],
            'count': 1,
        },
    )

    def _fake_mm(**kwargs):
        captured.update(kwargs)
        return {'summary': 'artifact summary', 'top_evidence': []}

    monkeypatch.setattr(rwt, '_run_multimodal_recall', _fake_mm)

    result = json.loads(rwt.recall_with_artifacts(query='那張截圖顯示現在卡在哪裡？'))
    assert captured['source_ref'] == ''
    assert 'gitlab_radar_49942185f88bf27f' in captured['query']
    assert result['derived_hints']['source_ref'] == 'issue:987'
    assert result['hint_confidence']['source_ref'] == 'high'


def test_transcript_derived_source_ref_stays_soft_for_ocr_query_without_corroboration(monkeypatch):
    import tools.recall_with_artifacts_tool as rwt

    captured = {}
    monkeypatch.setattr(rwt, '_session_search_available', lambda: True)
    monkeypatch.setattr(rwt, '_local_mmrag_connected', lambda: True)
    monkeypatch.setattr(
        rwt,
        '_run_session_search',
        lambda query, role_filter=None, limit=3: {
            'success': True,
            'results': [
                {
                    'summary': 'Earlier drifted session mentioned source_ref issue:987 and collection_name: "gitlab_radar_ae4adda6785fd089" source_type: pdf.'
                }
            ],
            'count': 1,
        },
    )

    def _fake_mm(**kwargs):
        captured.update(kwargs)
        return {'summary': 'artifact summary', 'top_evidence': []}

    monkeypatch.setattr(rwt, '_run_multimodal_recall', _fake_mm)

    result = json.loads(rwt.recall_with_artifacts(query='之前 OCR 抓到的錯誤訊息是什麼？'))
    assert captured['source_ref'] == ''
    assert 'gitlab_radar_ae4adda6785fd089' in captured['query']
    assert 'source_type pdf' in captured['query']
    assert result['derived_hints']['source_ref'] == 'issue:987'
    assert result['hint_confidence']['source_ref'] == 'high'


def test_broad_pdf_screenshot_query_does_not_append_transcript_attachment_source_type_hint(monkeypatch):
    import tools.recall_with_artifacts_tool as rwt

    captured = {}
    monkeypatch.setattr(rwt, '_session_search_available', lambda: True)
    monkeypatch.setattr(rwt, '_local_mmrag_connected', lambda: True)
    monkeypatch.setattr(
        rwt,
        '_run_session_search',
        lambda query, role_filter=None, limit=3: {
            'success': True,
            'results': [
                {
                    'summary': 'Prior session mentioned collection_name: "gitlab_radar_ae4adda6785fd089" source_type: attachment while discussing PDF/screenshot evidence.'
                }
            ],
            'count': 1,
        },
    )

    def _fake_mm(**kwargs):
        captured.update(kwargs)
        return {'summary': 'artifact summary', 'top_evidence': []}

    monkeypatch.setattr(rwt, '_run_multimodal_recall', _fake_mm)

    result = json.loads(rwt.recall_with_artifacts(query='先前那個 PDF 附件和截圖證據做到哪了？'))
    assert captured['collection'] == 'gitlab_radar_ae4adda6785fd089'
    assert 'source_type attachment' not in captured['query']
    assert captured['source_type'] == ''
    assert result['derived_hints']['source_type'] == 'attachment'
    assert result['hint_confidence']['source_type'] == 'medium'


def test_soft_transcript_source_ref_does_not_contaminate_issue_brief_provenance(monkeypatch):
    import tools.recall_with_artifacts_tool as rwt

    monkeypatch.setattr(rwt, '_session_search_available', lambda: True)
    monkeypatch.setattr(rwt, '_local_mmrag_connected', lambda: True)
    monkeypatch.setattr(
        rwt,
        '_run_session_search',
        lambda query, role_filter=None, limit=3: {
            'success': True,
            'results': [
                {
                    'summary': 'Earlier drifted session mentioned source_ref issue:987 and collection_name: "gitlab_radar_49942185f88bf27f" while discussing screenshot evidence.'
                }
            ],
            'count': 1,
        },
    )
    monkeypatch.setattr(
        rwt,
        '_run_multimodal_recall',
        lambda **kwargs: {
            'summary': '系统有资料, 看来是洗资料有问题',
            'top_evidence': [
                {
                    'source_path': '/tmp/image.png',
                    'source_type': 'image',
                    'source_ref': None,
                    'modality': 'image',
                    'collection_name': 'gitlab_radar_49942185f88bf27f',
                    'text': '系统有资料, 看来是洗资料有问题',
                    'score': 0.9,
                },
                {
                    'source_path': '/tmp/issue_context.md',
                    'source_type': 'text',
                    'source_ref': None,
                    'modality': 'text',
                    'collection_name': 'gitlab_radar_49942185f88bf27f',
                    'text': '# Issue context\n\n- title: 客戶反應 2019 以前的學校資料都不存在 => 洗資料有問題',
                    'score': 0.8,
                },
            ],
            'extracted_fields': {},
            'retrieval_notes': {'collection_name': 'gitlab_radar_49942185f88bf27f'},
        },
    )

    result = json.loads(rwt.recall_with_artifacts(query='那張截圖顯示現在卡在哪裡？'))
    assert result['issue_evidence_brief']['source_ref'] == ''
    assert result['issue_evidence_brief']['collections'] == ['gitlab_radar_49942185f88bf27f']
    assert 'issue:987' not in result['issue_evidence_brief']['evidence_brief']


def test_recall_with_artifacts_prefers_prompt_aligned_artifacts_when_multiple_sets_match(monkeypatch):
    import tools.recall_with_artifacts_tool as rwt

    monkeypatch.setattr(rwt, '_session_search_available', lambda: True)
    monkeypatch.setattr(rwt, '_local_mmrag_connected', lambda: True)
    monkeypatch.setattr(
        rwt,
        '_run_session_search',
        lambda query, role_filter=None, limit=3: {
            'success': True,
            'results': [{'summary': 'We discussed screenshot evidence for issue:123.'}],
            'count': 1,
        },
    )
    monkeypatch.setattr(
        rwt,
        '_run_multimodal_recall',
        lambda **kwargs: {
            'summary': 'mixed artifact summary',
            'top_evidence': [
                {
                    'source_path': '/tmp/adjacent-note.md',
                    'source_type': 'text',
                    'source_ref': 'issue:999',
                    'collection_name': 'adjacent_collection',
                    'text': 'Customer waiting notes with blocker details but no screenshot context.',
                    'score': 0.99,
                },
                {
                    'source_path': '/tmp/status.png',
                    'source_type': 'image',
                    'source_ref': 'issue:123',
                    'collection_name': 'target_collection',
                    'text': 'Screenshot evidence for issue:123 showing ETA next Monday and blocker OCR validation.',
                    'score': 0.80,
                },
            ],
            'retrieval_notes': {'collection_name': 'adjacent_collection'},
        },
    )

    result = json.loads(rwt.recall_with_artifacts(query='issue:123 screenshot blocker ETA'))
    brief = result['issue_evidence_brief']
    assert brief['source_ref'] == 'issue:123'
    assert brief['top_sources'][0] == '/tmp/status.png'
    assert brief['collections'][0] == 'target_collection'
    assert 'ETA next monday' in brief['evidence_brief']


def test_screenshot_only_followup_reduces_cross_artifact_blending(monkeypatch):
    import tools.recall_with_artifacts_tool as rwt

    monkeypatch.setattr(rwt, '_session_search_available', lambda: True)
    monkeypatch.setattr(rwt, '_local_mmrag_connected', lambda: True)
    monkeypatch.setattr(
        rwt,
        '_run_session_search',
        lambda query, role_filter=None, limit=3: {
            'success': True,
            'results': [{'summary': 'We discussed a screenshot in issue:286 and another customer-waiting thread in issue:987.'}],
            'count': 1,
        },
    )
    monkeypatch.setattr(
        rwt,
        '_run_multimodal_recall',
        lambda **kwargs: {
            'summary': 'mixed screenshot summary',
            'top_evidence': [
                {
                    'source_path': '/tmp/issue286-shot.png',
                    'source_type': 'image',
                    'source_ref': 'issue:286',
                    'collection_name': 'issue286_collection',
                    'text': 'Screenshot shows data exists and washing flow looks wrong.',
                    'score': 0.90,
                },
                {
                    'source_path': '/tmp/issue987-note.md',
                    'source_type': 'text',
                    'source_ref': 'issue:987',
                    'collection_name': 'issue987_collection',
                    'text': 'Customer waiting note with OCR validation blocker and next Tuesday ETA.',
                    'score': 0.91,
                },
                {
                    'source_path': '/tmp/issue987-shot.png',
                    'source_type': 'image',
                    'source_ref': 'issue:987',
                    'collection_name': 'issue987_collection',
                    'text': 'Status waiting customer reply; ETA next Tuesday; screenshot from support handoff.',
                    'score': 0.89,
                },
            ],
            'retrieval_notes': {'collection_name': 'issue987_collection'},
        },
    )

    result = json.loads(rwt.recall_with_artifacts(query='那張截圖顯示現在卡在哪裡？'))
    brief = result['issue_evidence_brief']
    assert brief['top_sources'][0] == '/tmp/issue286-shot.png'
    assert brief['collections'][0] == 'issue286_collection'
    assert brief['source_ref'] == 'issue:286'
    assert 'washing flow' in brief['evidence_brief'].lower() or 'data exists' in brief['evidence_brief'].lower()


def test_recall_with_artifacts_surfaces_transcript_vs_artifact_attribution_for_ambiguity_queries(monkeypatch):
    import tools.recall_with_artifacts_tool as rwt

    monkeypatch.setattr(rwt, '_session_search_available', lambda: True)
    monkeypatch.setattr(rwt, '_local_mmrag_connected', lambda: True)
    monkeypatch.setattr(
        rwt,
        '_run_session_search',
        lambda query, role_filter=None, limit=3: {
            'success': True,
            'results': [
                {'summary': 'Transcript says the customer was waiting for an ETA update in prior discussion.'}
            ],
            'count': 1,
        },
    )
    monkeypatch.setattr(
        rwt,
        '_run_multimodal_recall',
        lambda **kwargs: {
            'summary': 'Artifact evidence from screenshot and attachment confirms waiting customer status.',
            'top_evidence': [
                {
                    'source_path': '/tmp/status.png',
                    'source_type': 'image',
                    'source_ref': 'issue:987',
                    'collection_name': 'artifact_collection',
                    'text': 'Status: waiting customer reply; ETA: next Tuesday.',
                    'score': 0.9,
                }
            ],
            'retrieval_notes': {'collection_name': 'artifact_collection'},
        },
    )

    result = json.loads(rwt.recall_with_artifacts(query='那個 waiting customer 的事情，證據是來自對話還是附件？'))
    summary = result['evidence_channel_summary']
    assert summary['preferred_evidence_channel'] == 'mixed'
    assert 'Transcript says the customer was waiting' in summary['transcript_support']
    assert 'Artifact evidence from screenshot and attachment confirms waiting customer status.' in summary['artifact_support']


def test_ocr_error_prompt_prefers_error_message_over_adjacent_status_text(monkeypatch):
    import tools.recall_with_artifacts_tool as rwt

    monkeypatch.setattr(rwt, '_session_search_available', lambda: True)
    monkeypatch.setattr(rwt, '_local_mmrag_connected', lambda: True)
    monkeypatch.setattr(
        rwt,
        '_run_session_search',
        lambda query, role_filter=None, limit=3: {
            'success': True,
            'results': [{'summary': 'We reviewed OCR output and a document finding about technical error messages.'}],
            'count': 1,
        },
    )
    monkeypatch.setattr(
        rwt,
        '_run_multimodal_recall',
        lambda **kwargs: {
            'summary': 'ocr/document mixed summary',
            'top_evidence': [
                {
                    'source_path': '/tmp/status.png',
                    'source_type': 'image',
                    'modality': 'image',
                    'collection_name': 'status_collection',
                    'text': 'Status waiting customer reply',
                    'score': 0.95,
                },
                {
                    'source_path': '/tmp/report.pdf',
                    'source_type': 'pdf',
                    'modality': 'pdf',
                    'collection_name': 'report_collection',
                    'text': '錯誤訊息非客製化結果而夾帶太多技術訊息',
                    'score': 0.80,
                },
            ],
            'retrieval_notes': {'collection_name': 'status_collection'},
        },
    )

    result = json.loads(rwt.recall_with_artifacts(query='之前 OCR 抓到的錯誤訊息是什麼？'))
    brief = result['issue_evidence_brief']
    assert brief['top_sources'][0] == '/tmp/report.pdf'
    assert brief['collections'][0] == 'report_collection'
    assert '錯誤訊息非客製化結果而夾帶太多技術訊息' in brief['evidence_brief']


def test_phase3_baseline_pdf_screenshot_query_preserves_provenance(monkeypatch):
    import tools.recall_with_artifacts_tool as rwt

    monkeypatch.setattr(rwt, '_session_search_available', lambda: True)
    monkeypatch.setattr(rwt, '_local_mmrag_connected', lambda: True)
    monkeypatch.setattr(
        rwt,
        '_run_session_search',
        lambda query, role_filter=None, limit=3: {
            'success': True,
            'results': [{'summary': 'We previously checked the PDF attachment and screenshot evidence for issue:123.'}],
            'count': 1,
        },
    )
    monkeypatch.setattr(
        rwt,
        '_run_multimodal_recall',
        lambda **kwargs: {
            'summary': 'GitLab radar automation tracks customer waiting items.',
            'top_evidence': [
                {
                    'source_path': '/tmp/status.png',
                    'source_type': 'image',
                    'source_ref': 'issue:123',
                    'collection_name': 'test_mm_recall_search_filters',
                    'text': 'Project hermes ETA next Monday status waiting customer reply.',
                    'score': 0.8,
                },
                {
                    'source_path': '/tmp/report.pdf',
                    'source_type': 'pdf',
                    'source_ref': 'issue:123',
                    'collection_name': 'test_mm_recall_search_filters',
                    'text': 'Supporting PDF attachment for the same issue context.',
                    'score': 0.7,
                },
            ],
            'extracted_fields': {'customer_waiting_signals': ['customer_waiting'], 'eta': 'next monday'},
            'retrieval_notes': {'collection_name': 'test_mm_recall_search_filters'},
        },
    )

    result = json.loads(rwt.recall_with_artifacts(query='先前那個 PDF 附件和截圖證據做到哪了？'))
    assert result['issue_evidence_brief']['source_ref'] == 'issue:123'
    assert result['radar_ready_summary']['provenance']['collection'] == 'test_mm_recall_search_filters'
    assert result['radar_ready_summary']['eta'] == 'next monday'


def test_phase3_baseline_cross_session_continuity_query_returns_transcript_and_artifact_support(monkeypatch):
    import tools.recall_with_artifacts_tool as rwt

    monkeypatch.setattr(rwt, '_session_search_available', lambda: True)
    monkeypatch.setattr(rwt, '_local_mmrag_connected', lambda: True)
    monkeypatch.setattr(
        rwt,
        '_run_session_search',
        lambda query, role_filter=None, limit=3: {
            'success': True,
            'results': [{'summary': 'Prior discussion said the customer was still waiting for an ETA update.'}],
            'count': 1,
        },
    )
    monkeypatch.setattr(
        rwt,
        '_run_multimodal_recall',
        lambda **kwargs: {
            'summary': 'Artifact evidence shows waiting customer status and ETA next Tuesday.',
            'top_evidence': [
                {
                    'source_path': '/tmp/status.png',
                    'source_type': 'image',
                    'source_ref': 'issue:987',
                    'collection_name': 'test_realistic_mm_smoke_customer_waiting',
                    'text': 'Status waiting customer reply; ETA next Tuesday.',
                    'score': 0.9,
                }
            ],
            'extracted_fields': {'customer_waiting_signals': ['customer_waiting'], 'eta': 'next tuesday'},
            'retrieval_notes': {'collection_name': 'test_realistic_mm_smoke_customer_waiting'},
        },
    )

    result = json.loads(rwt.recall_with_artifacts(query='上次那個客戶等待的 ETA 後來有更新嗎？'))
    summary = result['evidence_channel_summary']
    assert summary['preferred_evidence_channel'] == 'mixed'
    assert 'customer was still waiting' in summary['transcript_support']
    assert 'Artifact evidence shows waiting customer status' in summary['artifact_support']
    assert result['radar_ready_summary']['eta'] == 'next tuesday'


def test_cluster_ranking_prefers_query_aligned_cluster_over_larger_adjacent_cluster(monkeypatch):
    import tools.recall_with_artifacts_tool as rwt

    clusters = [
        [
            {
                'source_path': '/tmp/adjacent-note.md',
                'source_type': 'text',
                'source_ref': 'issue:999',
                'collection_name': 'adjacent_collection',
                'text': 'Customer waiting notes and blocker updates.',
                'score': 0.95,
            },
            {
                'source_path': '/tmp/adjacent-shot.png',
                'source_type': 'image',
                'source_ref': 'issue:999',
                'collection_name': 'adjacent_collection',
                'text': 'Status waiting customer reply; ETA next Tuesday.',
                'score': 0.90,
            },
        ],
        [
            {
                'source_path': '/tmp/target-shot.png',
                'source_type': 'image',
                'source_ref': 'issue:123',
                'collection_name': 'target_collection',
                'text': 'Screenshot evidence for issue:123 showing ETA next Monday and blocker OCR validation.',
                'score': 0.80,
            },
            {
                'source_path': '/tmp/target-report.pdf',
                'source_type': 'pdf',
                'source_ref': 'issue:123',
                'collection_name': 'target_collection',
                'text': 'Supporting PDF attachment for the same issue context.',
                'score': 0.70,
            },
        ],
    ]

    ranked = sorted(
        clusters,
        key=lambda cluster: sum(float(item.get('score') or 0.0) for item in cluster),
        reverse=True,
    )
    assert ranked[0][0]['source_ref'] == 'issue:999'

    focus = rwt._select_focus_cluster('issue:123 screenshot blocker ETA', [item for cluster in clusters for item in cluster], resolved_source_ref='issue:123')
    assert focus
    assert focus[0]['source_ref'] == 'issue:123'
    assert all((item.get('source_ref') or '') == 'issue:123' for item in focus)


def test_issue_brief_uses_single_focus_cluster_for_live_screenshot_followup(monkeypatch):
    import tools.recall_with_artifacts_tool as rwt

    monkeypatch.setattr(rwt, '_session_search_available', lambda: True)
    monkeypatch.setattr(rwt, '_local_mmrag_connected', lambda: True)
    monkeypatch.setattr(
        rwt,
        '_run_session_search',
        lambda query, role_filter=None, limit=3: {
            'success': True,
            'results': [{'summary': 'We saw screenshot-heavy evidence across issue:286 and issue:987.'}],
            'count': 1,
        },
    )
    monkeypatch.setattr(
        rwt,
        '_run_multimodal_recall',
        lambda **kwargs: {
            'summary': 'live screenshot mix',
            'top_evidence': [
                {
                    'source_path': '/tmp/issue286-shot.png',
                    'source_type': 'image',
                    'modality': 'image',
                    'source_ref': 'issue:286',
                    'collection_name': 'issue286_collection',
                    'text': 'Screenshot shows data exists and washing flow looks wrong.',
                    'score': 0.86,
                },
                {
                    'source_path': '/tmp/issue286-note.md',
                    'source_type': 'text',
                    'modality': 'text',
                    'source_ref': 'issue:286',
                    'collection_name': 'issue286_collection',
                    'text': 'Issue 286 notes: likely data washing flow problem.',
                    'score': 0.70,
                },
                {
                    'source_path': '/tmp/issue987-shot.png',
                    'source_type': 'image',
                    'modality': 'image',
                    'source_ref': 'issue:987',
                    'collection_name': 'issue987_collection',
                    'text': 'Status waiting customer reply; ETA next Tuesday; screenshot from support handoff.',
                    'score': 0.89,
                },
                {
                    'source_path': '/tmp/issue987-note.md',
                    'source_type': 'text',
                    'modality': 'text',
                    'source_ref': 'issue:987',
                    'collection_name': 'issue987_collection',
                    'text': 'Customer waiting note with OCR validation blocker.',
                    'score': 0.88,
                },
            ],
            'retrieval_notes': {'collection_name': 'issue987_collection'},
        },
    )

    result = json.loads(rwt.recall_with_artifacts(query='那張截圖顯示現在卡在哪裡？'))
    brief = result['issue_evidence_brief']
    assert brief['source_ref'] == 'issue:286'
    assert brief['collections'] == ['issue286_collection']
    assert brief['top_sources'] == ['/tmp/issue286-shot.png', '/tmp/issue286-note.md']
    assert 'ocr validation' not in brief['evidence_brief'].lower()
    assert 'washing flow' in brief['evidence_brief'].lower() or 'data exists' in brief['evidence_brief'].lower()


def test_ocr_error_query_prefers_error_text_cluster_over_status_cluster(monkeypatch):
    import tools.recall_with_artifacts_tool as rwt

    evidence = [
        {
            'source_path': '/tmp/status.png',
            'source_type': 'image',
            'modality': 'image',
            'source_ref': 'issue:987',
            'collection_name': 'status_collection',
            'text': 'Status waiting customer reply; ETA next Tuesday.',
            'score': 0.95,
        },
        {
            'source_path': '/tmp/status-note.md',
            'source_type': 'text',
            'modality': 'text',
            'source_ref': 'issue:987',
            'collection_name': 'status_collection',
            'text': 'Customer waiting follow-up and OCR validation blocker.',
            'score': 0.92,
        },
        {
            'source_path': '/tmp/error-report.pdf',
            'source_type': 'pdf',
            'modality': 'pdf',
            'source_ref': 'issue:555',
            'collection_name': 'error_collection',
            'text': 'Custom error pages expose too much technical detail to users.',
            'score': 0.80,
        },
        {
            'source_path': '/tmp/error-note.md',
            'source_type': 'text',
            'modality': 'text',
            'source_ref': 'issue:555',
            'collection_name': 'error_collection',
            'text': 'Document finding confirms technical detail leakage through non-customized errors.',
            'score': 0.70,
        },
    ]

    focus = rwt._select_focus_cluster('之前 OCR 抓到的錯誤訊息是什麼？', evidence)
    assert focus
    assert focus[0]['source_ref'] == 'issue:555'
    assert all((item.get('source_ref') or '') == 'issue:555' for item in focus)



def test_phase3_broad_pdf_screenshot_query_prefers_pdf_screenshot_cluster_over_adjacent_status_cluster(monkeypatch):
    import tools.recall_with_artifacts_tool as rwt

    evidence = [
        {
            'source_path': '/tmp/adjacent-note.md',
            'source_type': 'text',
            'modality': 'text',
            'source_ref': 'issue:123',
            'collection_name': 'adjacent_status_collection',
            'text': 'Customer waiting note and blocker follow-up without any PDF or screenshot focus.',
            'score': 0.97,
        },
        {
            'source_path': '/tmp/adjacent-status.png',
            'source_type': 'image',
            'modality': 'image',
            'source_ref': 'issue:123',
            'collection_name': 'adjacent_status_collection',
            'text': 'Status waiting customer reply; ETA next Monday.',
            'score': 0.96,
        },
        {
            'source_path': '/tmp/target-shot.png',
            'source_type': 'image',
            'modality': 'image',
            'source_ref': 'issue:123',
            'collection_name': 'target_pdf_screenshot_collection',
            'text': 'Screenshot evidence for issue:123 progress check with key visual context.',
            'score': 0.82,
        },
        {
            'source_path': '/tmp/target-report.pdf',
            'source_type': 'pdf',
            'modality': 'pdf',
            'source_ref': 'issue:123',
            'collection_name': 'target_pdf_screenshot_collection',
            'text': 'PDF attachment for issue:123 containing progress evidence and document context.',
            'score': 0.81,
        },
    ]

    focus = rwt._select_focus_cluster('先前那個 PDF 附件和截圖證據做到哪了？', evidence, resolved_source_ref='issue:123')
    assert focus
    assert focus[0]['collection_name'] == 'target_pdf_screenshot_collection'
    assert all((item.get('collection_name') or '') == 'target_pdf_screenshot_collection' for item in focus)



def test_recall_with_artifacts_builds_issue_evidence_brief_from_issue_scoped_artifacts(monkeypatch):
    import tools.recall_with_artifacts_tool as rwt

    monkeypatch.setattr(rwt, '_session_search_available', lambda: True)
    monkeypatch.setattr(rwt, '_local_mmrag_connected', lambda: True)
    monkeypatch.setattr(
        rwt,
        '_run_session_search',
        lambda query, role_filter=None, limit=3: {
            'success': True,
            'results': [
                {'summary': 'Issue issue:987 discusses customer waiting and attachment evidence.'}
            ],
            'count': 1,
        },
    )
    monkeypatch.setattr(
        rwt,
        '_run_multimodal_recall',
        lambda **kwargs: {
            'summary': 'ETA next Tuesday. Blocker is OCR validation. Customer is waiting.',
            'top_evidence': [
                {
                    'source_path': '/tmp/status.png',
                    'source_type': 'attachment',
                    'source_ref': 'issue:987',
                    'modality': 'image',
                    'collection_name': 'gitlab_radar_demo',
                    'text': 'Status: waiting customer reply. ETA: next Tuesday. Evidence: screenshot.',
                },
                {
                    'source_path': '/tmp/summary.pdf',
                    'source_type': 'attachment',
                    'source_ref': 'issue:987',
                    'modality': 'pdf',
                    'collection_name': 'gitlab_radar_demo',
                    'text': 'Blocker: OCR validation for rollout evidence.',
                },
                {
                    'source_path': '/tmp/note.md',
                    'source_type': 'issue-note',
                    'source_ref': 'issue:987',
                    'modality': 'text',
                    'collection_name': 'gitlab_radar_demo',
                    'text': 'Customer is waiting for ETA update on the rollout.',
                },
            ],
            'extracted_fields': {
                'customer_waiting_signals': ['customer_waiting'],
                'eta': 'next tuesday',
            },
            'retrieval_notes': {'collection_name': 'gitlab_radar_demo'},
        },
    )

    result = json.loads(rwt.recall_with_artifacts(query='issue:987 customer waiting ETA blocker', source_ref='issue:987'))
    brief = result['issue_evidence_brief']
    assert brief['source_ref'] == 'issue:987'
    assert brief['collections'] == ['gitlab_radar_demo']
    assert brief['modalities_seen'] == ['image', 'pdf', 'text']
    assert brief['status_signals'] == ['customer_waiting']
    assert brief['eta_signals'] == ['next tuesday']
    assert 'ocr validation' in brief['blocker_signals']
    assert brief['top_sources'] == ['/tmp/status.png', '/tmp/summary.pdf', '/tmp/note.md']
    assert 'next tuesday' in brief['evidence_brief'].lower()
    assert 'ocr validation' in brief['evidence_brief'].lower()
    assert 'customer waiting' in brief['evidence_brief'].lower()


def test_recall_with_artifacts_builds_radar_ready_summary_from_issue_evidence_brief(monkeypatch):
    import tools.recall_with_artifacts_tool as rwt

    monkeypatch.setattr(rwt, '_session_search_available', lambda: True)
    monkeypatch.setattr(rwt, '_local_mmrag_connected', lambda: True)
    monkeypatch.setattr(
        rwt,
        '_run_session_search',
        lambda query, role_filter=None, limit=3: {
            'success': True,
            'results': [{'summary': 'Issue issue:987 discusses customer waiting and attachment evidence.'}],
            'count': 1,
        },
    )
    monkeypatch.setattr(
        rwt,
        '_run_multimodal_recall',
        lambda **kwargs: {
            'summary': 'ETA next Tuesday. Blocker is OCR validation. Customer is waiting.',
            'top_evidence': [
                {
                    'source_path': '/tmp/status.png',
                    'source_type': 'attachment',
                    'source_ref': 'issue:987',
                    'modality': 'image',
                    'collection_name': 'gitlab_radar_demo',
                    'text': 'Status: waiting customer reply. ETA: next Tuesday. Evidence: screenshot.',
                },
                {
                    'source_path': '/tmp/summary.pdf',
                    'source_type': 'attachment',
                    'source_ref': 'issue:987',
                    'modality': 'pdf',
                    'collection_name': 'gitlab_radar_demo',
                    'text': 'Blocker: OCR validation for rollout evidence.',
                },
            ],
            'extracted_fields': {
                'customer_waiting_signals': ['customer_waiting'],
                'eta': 'next tuesday',
            },
            'retrieval_notes': {'collection_name': 'gitlab_radar_demo'},
        },
    )

    result = json.loads(rwt.recall_with_artifacts(query='issue:987 customer waiting ETA blocker', source_ref='issue:987'))
    summary = result['radar_ready_summary']
    assert summary['source_ref'] == 'issue:987'
    assert summary['current_evidence_backed_status'] == 'customer_waiting'
    assert summary['eta'] == 'next tuesday'
    assert summary['blocker'] == 'ocr validation'
    assert summary['evidence_basis'] == ['image', 'pdf']
    assert summary['suggested_next_step']
    assert 'customer' in summary['suggested_next_step'].lower()
    assert summary['provenance']['collection'] == 'gitlab_radar_demo'
    assert summary['provenance']['top_sources'] == ['/tmp/status.png', '/tmp/summary.pdf']


def test_recall_with_artifacts_builds_comment_ready_summary_from_radar_ready_summary(monkeypatch):
    import tools.recall_with_artifacts_tool as rwt

    monkeypatch.setattr(rwt, '_session_search_available', lambda: True)
    monkeypatch.setattr(rwt, '_local_mmrag_connected', lambda: True)
    monkeypatch.setattr(
        rwt,
        '_run_session_search',
        lambda query, role_filter=None, limit=3: {
            'success': True,
            'results': [{'summary': 'Issue issue:987 discusses customer waiting and attachment evidence.'}],
            'count': 1,
        },
    )
    monkeypatch.setattr(
        rwt,
        '_run_multimodal_recall',
        lambda **kwargs: {
            'summary': 'ETA next Tuesday. Blocker is OCR validation. Customer is waiting.',
            'top_evidence': [
                {
                    'source_path': '/tmp/status.png',
                    'source_type': 'attachment',
                    'source_ref': 'issue:987',
                    'modality': 'image',
                    'collection_name': 'gitlab_radar_demo',
                    'text': 'Status: waiting customer reply. ETA: next Tuesday. Evidence: screenshot.',
                },
                {
                    'source_path': '/tmp/summary.pdf',
                    'source_type': 'attachment',
                    'source_ref': 'issue:987',
                    'modality': 'pdf',
                    'collection_name': 'gitlab_radar_demo',
                    'text': 'Blocker: OCR validation for rollout evidence.',
                },
            ],
            'extracted_fields': {
                'customer_waiting_signals': ['customer_waiting'],
                'eta': 'next tuesday',
            },
            'retrieval_notes': {'collection_name': 'gitlab_radar_demo'},
        },
    )

    result = json.loads(rwt.recall_with_artifacts(query='issue:987 customer waiting ETA blocker', source_ref='issue:987'))
    comment = result['comment_ready_summary']
    assert 'Current evidence-backed status: customer_waiting' in comment
    assert 'ETA: next tuesday' in comment
    assert 'Blocker: ocr validation' in comment
    assert 'Evidence basis: image, pdf' in comment
    assert 'Suggested next step:' in comment
    assert 'Source ref: issue:987' in comment
    assert 'Collection: gitlab_radar_demo' in comment
    assert '/tmp/status.png' in comment
