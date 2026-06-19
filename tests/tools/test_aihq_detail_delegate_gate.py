from tools.delegate_tool import (
    _aihq_detail_delegate_missing_requirements,
    _aihq_pipeline_delegate_missing_requirements,
)


def test_aihq_detail_delegate_gate_blocks_copy_task_without_intake_context():
    missing = _aihq_detail_delegate_missing_requirements(
        [
            {
                "goal": "쿠팡용 햄버거 카피바라 봉제인형 2종 공용 상세페이지의 상품명 후보와 핵심 카피를 제안하라.",
                "context": "",
            }
        ]
    )

    assert "pipeline_id" in missing
    assert "/mywiki" in missing
    assert "실제 작업 에이전트 팀장 회의" in missing


def test_aihq_detail_delegate_gate_allows_task_with_full_intake_context():
    context = """
    pipeline_id: detail_page
    /mywiki 조회 완료
    My Note 기록 예정
    실제 작업 에이전트 팀장 회의 완료
    superpowers 계획 적용
    GStack 검수 적용
    김팀장 최종 취합 예정
    """

    missing = _aihq_detail_delegate_missing_requirements(
        [
            {
                "goal": "쿠팡 780px 상세페이지 섹션 카피 검토",
                "context": context,
            }
        ]
    )

    assert missing == []


def test_aihq_detail_delegate_gate_skips_unrelated_tasks():
    missing = _aihq_detail_delegate_missing_requirements(
        [{"goal": "오늘 재고 확인 기준을 정리하라.", "context": ""}]
    )

    assert missing == []


def test_aihq_pipeline_delegate_gate_blocks_proposal_without_intake_context():
    missing = _aihq_pipeline_delegate_missing_requirements(
        [{"goal": "이마트24 제안서 핵심 카피와 섹션 구성을 작성하라.", "context": ""}]
    )

    assert "pipeline_id" in missing
    assert "/mywiki" in missing
    assert "김팀장 최종 취합" in missing


def test_aihq_pipeline_delegate_gate_blocks_market_research_without_intake_context():
    missing = _aihq_pipeline_delegate_missing_requirements(
        [{"goal": "경쟁조사 보고서 초안을 작성하라.", "context": ""}]
    )

    assert "pipeline_id" in missing
    assert "실제 작업 에이전트 팀장 회의" in missing


def test_aihq_pipeline_delegate_gate_blocks_automation_without_intake_context():
    missing = _aihq_pipeline_delegate_missing_requirements(
        [{"goal": "Slack 자동화 개발복구 작업을 하위 에이전트에게 맡겨라.", "context": ""}]
    )

    assert "superpowers" in missing
    assert "GStack" in missing
