"""Experience to Skill Learning Loop infrastructure.

Captures anomalies and unexpected structures discovered during execution,
formats candidate lessons with raw evidence, and links them to regression
fixtures so skills evolve predictably and securely.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class CandidateLesson:
    lesson_id: str
    skill_name: str
    issue_description: str
    raw_evidence_ref: str
    regression_fixture: Dict[str, Any]
    proposed_fix_description: str
    status: str = "candidate"  # candidate, verified, rejected, promoted
    created_at: str = field(default_factory=_utc_now)
    promoted_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ExperienceLessonCompiler:
    """Compiles execution experiences into candidate skill updates and regression fixtures."""

    def __init__(self, lessons_dir: Optional[Path] = None) -> None:
        if lessons_dir is not None:
            self.lessons_dir = Path(lessons_dir)
        else:
            self.lessons_dir = get_hermes_home() / "workstation" / "candidate_lessons"
        self.lessons_dir.mkdir(parents=True, exist_ok=True)

    def record_candidate_lesson(
        self,
        *,
        skill_name: str,
        issue_description: str,
        raw_evidence_ref: str,
        regression_fixture: Dict[str, Any],
        proposed_fix_description: str,
    ) -> CandidateLesson:
        lesson_id = f"lesson_{uuid4().hex[:10]}"
        lesson = CandidateLesson(
            lesson_id=lesson_id,
            skill_name=skill_name,
            issue_description=issue_description,
            raw_evidence_ref=raw_evidence_ref,
            regression_fixture=regression_fixture,
            proposed_fix_description=proposed_fix_description,
        )

        path = self.lessons_dir / f"{lesson_id}.json"
        path.write_text(json.dumps(lesson.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Recorded candidate lesson %s for skill %s", lesson_id, skill_name)
        return lesson

    def list_candidate_lessons(self, skill_name: Optional[str] = None) -> List[CandidateLesson]:
        lessons: List[CandidateLesson] = []
        for file in self.lessons_dir.glob("*.json"):
            try:
                data = json.loads(file.read_text(encoding="utf-8"))
                if skill_name is None or data.get("skill_name") == skill_name:
                    lessons.append(CandidateLesson(**data))
            except Exception:
                continue
        return lessons
