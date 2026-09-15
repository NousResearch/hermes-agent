"""Semantic Validation and Field Provenance for Data Workflows.

Provides invariant checking (e.g. views < likes -> SUSPECT, conflicting cross-section
labels) and field-level provenance tracking (value, source section, raw reference, confidence).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import logging
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class FieldProvenance:
    value: Any
    source: str
    section: str
    raw_ref: Optional[str] = None
    confidence: float = 1.0
    extracted_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SemanticValidator:
    """Configurable invariant checker for data items and extracted metrics."""

    def __init__(
        self,
        *,
        min_text_length: Optional[int] = None,
        required_fields: Optional[List[str]] = None,
        field_ratios: Optional[List[Dict[str, str]]] = None,
    ) -> None:
        self.min_text_length = min_text_length
        self.required_fields = required_fields or []
        self.field_ratios = field_ratios or []

    def validate_item(self, item_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate an extracted item against domain invariants.

        Returns {valid: bool, suspect: bool, reason: str, issues: list}.
        """
        issues: List[str] = []
        is_suspect = False

        # Check required fields
        for req in self.required_fields:
            if req not in item_data or item_data[req] is None or item_data[req] == "":
                issues.append(f"Missing required field: '{req}'")

        # Check field ratios / comparisons (e.g. greater_field >= smaller_field)
        for rule in self.field_ratios:
            greater_key = rule.get("greater")
            smaller_key = rule.get("smaller")
            if greater_key and smaller_key:
                val_greater = item_data.get(greater_key)
                val_smaller = item_data.get(smaller_key)
                if isinstance(val_greater, (int, float)) and isinstance(val_smaller, (int, float)):
                    if val_greater < val_smaller:
                        is_suspect = True
                        issues.append(
                            f"Invariant violation: {greater_key} ({val_greater}) < {smaller_key} ({val_smaller})"
                        )

        # Check for conflicting duplicate sections if structured with provenance
        seen_sections: Dict[str, Any] = {}
        for key, val in item_data.items():
            if isinstance(val, dict) and "section" in val and "value" in val:
                sec = val["section"]
                if sec in seen_sections and seen_sections[sec] != val["value"]:
                    is_suspect = True
                    issues.append(f"Conflicting values across section '{sec}': {seen_sections[sec]} vs {val['value']}")
                seen_sections[sec] = val["value"]

        valid = len(issues) == 0 or (is_suspect and not any("Missing" in i for i in issues))
        reason = "; ".join(issues) if issues else "Item satisfied all validation invariants"

        return {
            "valid": valid,
            "suspect": is_suspect,
            "reason": reason,
            "issues": issues,
        }
