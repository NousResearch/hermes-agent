"""
Bid Engine - Part Activation and Bid Generation

The bid system is the core mechanism through which Dynamic Parts influence agent behavior.
Standalone module without dependencies on tools/__init__.py
"""

import logging
import uuid
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class Bid:
    part_id: str
    part_name: str
    what_i_want: str
    recommendation: str
    prediction: str = ""
    confidence: str = "Medium"
    urgency: int = 5
    triggers: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "part_id": self.part_id,
            "part_name": self.part_name,
            "what_i_want": self.what_i_want,
            "recommendation": self.recommendation,
            "prediction": self.prediction,
            "confidence": self.confidence,
            "urgency": self.urgency,
            "triggers": self.triggers,
            "timestamp": self.timestamp,
        }


# Minimal Part class for bid engine
@dataclass
class Part:
    name: str
    description: str
    triggers: List[str] = field(default_factory=list)
    wants: List[str] = field(default_factory=list)
    phrases: List[str] = field(default_factory=list)
    personality: str = ""
    emotion: str = ""
    intensity: str = ""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    archived: bool = False


class BidEngine:
    def __init__(self, min_urgency_threshold: int = 3):
        self.min_urgency_threshold = min_urgency_threshold
    
    def check_triggers(self, part: Part, context: str) -> List[str]:
        context_lower = context.lower()
        matched = []
        for trigger in part.triggers:
            trigger_lower = trigger.lower()
            if trigger_lower in context_lower:
                matched.append(trigger)
        return matched
    
    def calculate_urgency(self, part: Part, matched_triggers: List[str]) -> int:
        urgency = 5
        if len(matched_triggers) > 2:
            urgency += 2
        elif len(matched_triggers) > 0:
            urgency += 1
        intensity_map = {"High": 2, "Medium": 1, "Low": 0}
        urgency += intensity_map.get(part.intensity, 0)
        return min(10, urgency)
    
    def generate_bid(self, part: Part, context: str, matched_triggers: List[str]) -> Bid:
        urgency = self.calculate_urgency(part, matched_triggers)
        what_i_want = ", ".join(part.wants[:2]) if part.wants else "attention"
        recommendation = part.phrases[0] if part.phrases else f"I want: {what_i_want}"
        return Bid(
            part_id=part.id,
            part_name=part.name,
            what_i_want=what_i_want,
            recommendation=recommendation,
            confidence=part.intensity or "Medium",
            urgency=urgency,
            triggers=matched_triggers,
        )
    
    def activate_parts(self, parts: List[Part], context: str) -> List[Bid]:
        bids = []
        for part in parts:
            if part.archived:
                continue
            matched = self.check_triggers(part, context)
            if matched:
                bids.append(self.generate_bid(part, context, matched))
        bids.sort(key=lambda b: b.urgency, reverse=True)
        return bids
    
    def filter_bids(self, bids: List[Bid], max_bids: int = 5) -> List[Bid]:
        filtered = [b for b in bids if b.urgency >= self.min_urgency_threshold]
        return filtered[:max_bids]
    
    def get_active_bids(self, parts: List[Part], context: str, max_bids: int = 5) -> List[Bid]:
        all_bids = self.activate_parts(parts, context)
        return self.filter_bids(all_bids, max_bids)
    
    def get_bids_summary(self, bids: List[Bid]) -> str:
        if not bids:
            return "No active bids."
        lines = ["Active bids:"]
        for i, bid in enumerate(bids, 1):
            lines.append(f"{i}. {bid.part_name}: {bid.recommendation}")
            lines.append(f"   Urgency: {bid.urgency}/10")
        return "\n".join(lines)


def get_bid_engine() -> BidEngine:
    return BidEngine()
