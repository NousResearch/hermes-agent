"""Economic Topic-Based Skill Loader for Hermes Workstation.

Allows selective loading of specific sub-topics/sections within extensive skill documents
instead of injecting thousands of lines into the LLM context.
"""

from __future__ import annotations

import logging
from pathlib import Path
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class TopicSkillLoader:
    """Parses skill markdown documents into discrete topic sections for lean loading."""

    def __init__(self) -> None:
        self._parsed_cache: Dict[str, Dict[str, str]] = {}

    def parse_topics(self, content: str) -> Dict[str, str]:
        """Split markdown content into topics based on Level 2/3 headings."""
        topics: Dict[str, str] = {}
        current_topic = "overview"
        lines_buffer: List[str] = []

        for line in content.splitlines():
            heading_match = re.match(r"^(?:##|###)\s+(?:(?:\d+\.|\d+\))\s*)?(.+)$", line)
            if heading_match:
                if lines_buffer:
                    topics[current_topic] = "\n".join(lines_buffer).strip()
                    lines_buffer = []
                topic_title = heading_match.group(1).strip()
                # Normalize topic slug
                current_topic = re.sub(r"[^\w]+", "_", topic_title.lower()).strip("_")
            lines_buffer.append(line)

        if lines_buffer:
            topics[current_topic] = "\n".join(lines_buffer).strip()

        return topics

    def load_topic(
        self,
        skill_file_path: Path,
        topic: str,
        *,
        max_tokens_approx: int = 1500,
    ) -> Dict[str, Any]:
        """Load only the requested topic from the skill file."""
        file_key = str(skill_file_path.resolve())
        if file_key not in self._parsed_cache:
            if not skill_file_path.exists():
                raise FileNotFoundError(f"Skill file not found: {skill_file_path}")
            content = skill_file_path.read_text(encoding="utf-8")
            self._parsed_cache[file_key] = self.parse_topics(content)

        topics = self._parsed_cache[file_key]
        normalized_requested = re.sub(r"[^\w]+", "_", topic.lower()).strip("_")

        # Exact match or substring match
        matched_content = topics.get(normalized_requested)
        if not matched_content:
            for k, v in topics.items():
                if normalized_requested in k or k in normalized_requested:
                    matched_content = v
                    break

        if not matched_content:
            available = list(topics.keys())
            return {
                "found": False,
                "error": f"Topic '{topic}' not found in skill.",
                "available_topics": available,
            }

        return {
            "found": True,
            "topic": normalized_requested,
            "content": matched_content,
            "available_topics": list(topics.keys()),
            "approx_chars": len(matched_content),
        }
