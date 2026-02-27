"""
Persona Layer - Self/Persona Decision System

The Persona Layer is the "Self" that receives bids from parts and decides which ones to follow.
This is where the agent's identity and values are expressed.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
import importlib.util
from pathlib import Path

logger = logging.getLogger(__name__)

# Load models directly
_parts_models_path = Path(__file__).parent / "models.py"
spec = importlib.util.spec_from_file_location("parts_models", _parts_models_path)
_parts_models = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_parts_models)

Part = _parts_models.Part

# Load bid engine
_bid_engine_path = Path(__file__).parent / "bid_engine.py"
spec2 = importlib.util.spec_from_file_location("bid_engine", _bid_engine_path)
_bid_engine_mod = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(_bid_engine_mod)

Bid = _bid_engine_mod.Bid


@dataclass
class Decision:
    """
    Represents a decision made by the Self/Persona.
    
    Attributes:
        chosen_bid: The bid that was chosen to follow
        rejected_bids: Bids that were considered but not chosen
        reasoning: Why this decision was made
        final_action: The action to take
    """
    chosen_bid: Optional[Bid]
    rejected_bids: List[Bid]
    reasoning: str
    final_action: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "chosen_bid": self.chosen_bid.to_dict() if self.chosen_bid else None,
            "rejected_bids": [b.to_dict() for b in self.rejected_bids],
            "reasoning": self.reasoning,
            "final_action": self.final_action,
            "timestamp": self.timestamp,
        }


@dataclass
class PersonaConfig:
    """
    Configuration for the Self/Persona.
    
    Attributes:
        name: Name of the persona/self
        core_values: What the self values most
        trust_levels: Which parts the self trusts in which contexts
        avoidance_triggers: What makes the self ignore a bid
    """
    name: str = "Hermes"
    core_values: List[str] = field(default_factory=lambda: [
        "helpfulness", "honesty", "respect", "learning"
    ])
    trust_levels: Dict[str, float] = field(default_factory=dict)
    avoidance_triggers: List[str] = field(default_factory=lambda: [
        "harmful", "illegal", "deceptive"
    ])


class PersonaLayer:
    """
    The Self/Persona layer that decides which bids to follow.
    
    This is the "I" that receives proposals from parts and makes decisions.
    It maintains identity, values, and decides based on trust and context.
    """
    
    def __init__(self, config: Optional[PersonaConfig] = None):
        self.config = config or PersonaConfig()
        self.decision_history: List[Decision] = []
    
    def check_avoidance(self, bid: Bid) -> tuple[bool, str]:
        """
        Check if a bid should be avoided based on triggers.
        
        Args:
            bid: The bid to check
            
        Returns:
            Tuple of (should_avoid, reason)
        """
        bid_text = (bid.what_i_want + " " + bid.recommendation).lower()
        
        for trigger in self.config.avoidance_triggers:
            if trigger.lower() in bid_text:
                return True, f"Avoided due to trigger: {trigger}"
        
        return False, ""
    
    def calculate_trust_score(self, bid: Bid) -> float:
        """
        Calculate how much the self trusts this bid.
        
        Args:
            bid: The bid to evaluate
            
        Returns:
            Trust score (0-1)
        """
        score = 0.5  # Base trust
        
        # Higher confidence = higher trust
        confidence_map = {"High": 0.3, "Medium": 0.0, "Low": -0.2}
        score += confidence_map.get(bid.confidence, 0)
        
        # Higher urgency = more attention (but not necessarily trust)
        score += (bid.urgency - 5) * 0.05
        
        # Core parts are more trusted
        # (This would need part lookup - simplified here)
        
        return max(0.0, min(1.0, score))
    
    def evaluate_bid(self, bid: Bid) -> Dict[str, Any]:
        """
        Evaluate a single bid.
        
        Args:
            bid: The bid to evaluate
            
        Returns:
            Evaluation results with scores
        """
        # Check avoidance triggers
        avoid, avoid_reason = self.check_avoidance(bid)
        if avoid:
            return {
                "accepted": False,
                "reason": avoid_reason,
                "trust_score": 0.0,
            }
        
        # Calculate trust
        trust_score = self.calculate_trust_score(bid)
        
        # Evaluate based on values alignment
        bid_text = (bid.what_i_want + " " + bid.recommendation).lower()
        values_aligned = any(
            value in bid_text 
            for value in self.config.core_values
        )
        
        return {
            "accepted": trust_score > 0.3 and values_aligned,
            "reason": f"Trust: {trust_score:.2f}, Values aligned: {values_aligned}",
            "trust_score": trust_score,
        }
    
    def make_decision(self, bids: List[Bid]) -> Decision:
        """
        Make a decision from available bids.
        
        Args:
            bids: List of bids from activated parts
            
        Returns:
            A Decision object
        """
        if not bids:
            return Decision(
                chosen_bid=None,
                rejected_bids=[],
                reasoning="No bids available",
                final_action="Continue normally"
            )
        
        # Evaluate all bids
        evaluations = []
        for bid in bids:
            eval_result = self.evaluate_bid(bid)
            eval_result["bid"] = bid
            evaluations.append(eval_result)
        
        # Sort by trust score
        evaluations.sort(key=lambda e: e["trust_score"], reverse=True)
        
        # Get accepted bids
        accepted = [e for e in evaluations if e["accepted"]]
        
        if not accepted:
            # No bids accepted - follow the highest trust anyway but note it
            best = evaluations[0]
            return Decision(
                chosen_bid=best["bid"],
                rejected_bids=[e["bid"] for e in evaluations[1:]],
                reasoning=f"No bids met acceptance criteria. Following highest trust: {best['reason']}",
                final_action=best["bid"].recommendation
            )
        
        # Choose the best accepted bid
        chosen = accepted[0]
        rejected = [e["bid"] for e in evaluations if e != chosen]
        
        return Decision(
            chosen_bid=chosen["bid"],
            rejected_bids=rejected,
            reasoning=f"Chose '{chosen['bid'].part_name}' - {chosen['reason']}",
            final_action=chosen["bid"].recommendation
        )
    
    def decide(self, bids: List[Bid], context: str = "") -> Decision:
        """
        Make a decision from bids with context.
        
        Args:
            bids: List of bids
            context: Additional context
            
        Returns:
            Decision
        """
        decision = self.make_decision(bids)
        self.decision_history.append(decision)
        
        return decision
    
    def get_decision_summary(self, decision: Decision) -> str:
        """
        Generate a human-readable summary of a decision.
        
        Args:
            decision: The decision to summarize
            
        Returns:
            Formatted summary string
        """
        if not decision.chosen_bid:
            return "No active parts influencing this turn."
        
        lines = [
            f"Persona: {self.config.name}",
            f"Decision: Following '{decision.chosen_bid.part_name}'",
            f"Reasoning: {decision.reasoning}",
            f"Action: {decision.final_action}",
        ]
        
        if decision.rejected_bids:
            lines.append(f"Rejected {len(decision.rejected_bids)} other bids")
        
        return "\n".join(lines)
    
    def get_recent_decisions(self, n: int = 5) -> List[Decision]:
        """Get the n most recent decisions."""
        return self.decision_history[-n:]
    
    def update_config(self, **kwargs):
        """Update persona configuration."""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)


def get_persona_layer(config: Optional[PersonaConfig] = None) -> PersonaLayer:
    """Get or create a PersonaLayer instance."""
    return PersonaLayer(config)
