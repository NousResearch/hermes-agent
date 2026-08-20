"""task_context.py — Structured input for Claude Code sessions.

The core data structure for the context pipeline:
Hermes fills TaskContext → ClaudeSession formats it → Claude executes.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class FileContext:
    """A file relevant to the task."""
    path: str
    content: Optional[str] = None
    description: Optional[str] = None


@dataclass
class TaskContext:
    """Structured task description for Claude Code execution.

    Designed to capture what Claude needs without requiring
    exploration: clear goal, relevant code, constraints, and
    acceptance criteria.
    """

    task_description: str
    file_contexts: list[FileContext] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    acceptance_criteria: list[str] = field(default_factory=list)
    project_conventions: Optional[str] = None

    def to_prompt(self) -> str:
        """Format into a high-density prompt for Claude Code."""
        parts = [f"## Task\n{self.task_description}"]

        if self.file_contexts:
            parts.append("\n## Relevant Files")
            for fc in self.file_contexts:
                header = f"### {fc.path}"
                if fc.description:
                    header += f" — {fc.description}"
                parts.append(header)
                if fc.content:
                    parts.append(f"```\n{fc.content}\n```")

        if self.constraints:
            parts.append("\n## Constraints")
            for c in self.constraints:
                parts.append(f"- {c}")

        if self.acceptance_criteria:
            parts.append("\n## Acceptance Criteria")
            for ac in self.acceptance_criteria:
                parts.append(f"- {ac}")

        if self.project_conventions:
            parts.append(f"\n## Project Conventions\n{self.project_conventions}")

        return "\n".join(parts)
