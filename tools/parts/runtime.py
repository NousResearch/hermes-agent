"""
Dynamic Parts System - Runtime Integration

This module provides the runtime loop integration for Dynamic Parts.
It connects the parts system with the agent's turn processing.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class PartsContext:
    """Context passed to the agent with parts information."""
    active_bids: List[Dict[str, Any]]
    due_evaluations: List[Dict[str, Any]]
    relevant_parts: List[Dict[str, Any]]
    persona_decision: Optional[Dict[str, Any]]


class PartsRuntime:
    """
    Runtime integration for Dynamic Parts.
    
    This class manages the core loop:
    1. On each turn, retrieve relevant parts via vector search
    2. Generate bids from activated parts
    3. Have Self/Persona decide which bids to follow
    4. Surface due predictions
    5. Return enriched context to the agent
    """
    
    def __init__(self, storage, vector_store=None, bid_engine=None, persona_layer=None):
        self.storage = storage
        self.vector_store = vector_store
        self.bid_engine = bid_engine
        self.persona_layer = persona_layer
        self.enabled = True
    
    def process_turn(self, context: str, user_message: str) -> PartsContext:
        """
        Process a single turn with Dynamic Parts.
        
        Args:
            context: Current conversation context
            user_message: The user's latest message
            
        Returns:
            PartsContext with bids, evaluations, and decisions
        """
        if not self.enabled:
            return PartsContext(
                active_bids=[],
                due_evaluations=[],
                relevant_parts=[],
                persona_decision=None
            )
        
        try:
            # 1. Get relevant parts (from vector search or triggers)
            relevant_parts = self._get_relevant_parts(context)
            
            # 2. Generate bids from activated parts
            active_bids = self._generate_bids(relevant_parts, context)
            
            # 3. Get due evaluations
            due_evaluations = self._get_due_evaluations()
            
            # 4. Have Self/Persona decide
            persona_decision = self._make_decision(active_bids, context)
            
            return PartsContext(
                active_bids=active_bids,
                due_evaluations=due_evaluations,
                relevant_parts=[p.to_dict() if hasattr(p, 'to_dict') else p for p in relevant_parts],
                persona_decision=persona_decision
            )
            
        except Exception as e:
            logger.error(f"Error in parts runtime: {e}")
            return PartsContext(
                active_bids=[],
                due_evaluations=[],
                relevant_parts=[],
                persona_decision=None
            )
    
    def _get_relevant_parts(self, context: str) -> List:
        """Get parts relevant to current context."""
        parts = self.storage.list_parts(include_archived=False)
        
        # Try vector search first if available
        if self.vector_store and self.vector_store.client:
            try:
                vector_results = self.vector_store.search(context, limit=10)
                if vector_results:
                    # Get full part objects for vector results
                    relevant_ids = [r['part_id'] for r in vector_results]
                    relevant_parts = [p for p in parts if p.id in relevant_ids]
                    # Also include parts with matching triggers
                    trigger_parts = self._get_trigger_matched_parts(parts, context)
                    # Combine and deduplicate
                    all_parts = {p.id: p for p in relevant_parts + trigger_parts}
                    return list(all_parts.values())
            except Exception as e:
                logger.warning(f"Vector search failed: {e}")
        
        # Fallback to trigger matching
        return self._get_trigger_matched_parts(parts, context)
    
    def _get_trigger_matched_parts(self, parts: List, context: str) -> List:
        """Get parts whose triggers match the context."""
        context_lower = context.lower()
        matched = []
        
        for part in parts:
            if part.archived:
                continue
            for trigger in part.triggers:
                if trigger.lower() in context_lower:
                    matched.append(part)
                    break
        
        return matched
    
    def _generate_bids(self, parts: List, context: str) -> List[Dict[str, Any]]:
        """Generate bids from relevant parts."""
        if not self.bid_engine:
            return []
        
        try:
            bids = self.bid_engine.get_active_bids(parts, context, max_bids=5)
            return [b.to_dict() for b in bids]
        except Exception as e:
            logger.error(f"Error generating bids: {e}")
            return []
    
    def _get_due_evaluations(self) -> List[Dict[str, Any]]:
        """Get parts with due predictions."""
        try:
            due_parts = self.storage.get_due_evaluations()
            return [
                {
                    "part_id": p.id,
                    "part_name": p.name,
                    "suggestions": [
                        {
                            "predicted_result": s.predicted_result,
                            "your_suggestion": s.your_suggestion,
                            "timestamp": s.timestamp,
                            "timeframe_seconds": s.predicted_result_timeframe_seconds
                        }
                        for s in p.suggestions_and_results
                        if s.result is None
                    ]
                }
                for p in due_parts
                if p.suggestions_and_results
            ]
        except Exception as e:
            logger.error(f"Error getting due evaluations: {e}")
            return []
    
    def _make_decision(self, bids: List[Dict[str, Any]], context: str) -> Optional[Dict[str, Any]]:
        """Have Self/Persona make a decision."""
        if not self.persona_layer or not bids:
            return None

        try:
            # Convert dicts back to Bid objects for persona
            from tools.parts.bid_engine import Bid
            bid_objects = [Bid(**b) for b in bids]
            decision = self.persona_layer.decide(bid_objects, context)
            return decision.to_dict()
        except Exception as e:
            logger.error(f"Error making decision: {e}")
            return None
    
    def get_system_prompt_addition(self, parts_context: PartsContext) -> str:
        """
        Generate the system prompt addition based on active parts.
        
        This is injected into the agent's context to inform it about active parts.
        """
        if not parts_context.active_bids and not parts_context.due_evaluations:
            return ""
        
        sections = []
        
        # Active bids section
        if parts_context.active_bids:
            sections.append("## Active Parts (bidding for attention)")
            for bid in parts_context.active_bids[:3]:
                sections.append(f"- **{bid['part_name']}**: {bid['recommendation']}")
                if bid.get('triggers'):
                    sections.append(f"  Triggers: {', '.join(bid['triggers'])}")
            sections.append("")
        
        # Due evaluations section
        if parts_context.due_evaluations:
            sections.append("## Parts Needing Evaluation")
            sections.append("These predictions are due - consider what happened:")
            for eval in parts_context.due_evaluations[:2]:
                sections.append(f"- **{eval['part_name']}**: {eval['suggestions'][0]['your_suggestion']}")
            sections.append("")
        
        # Persona decision section
        if parts_context.persona_decision and parts_context.persona_decision.get('chosen_bid'):
            chosen = parts_context.persona_decision['chosen_bid']
            sections.append(f"## Self Decision")
            sections.append(f"Following: **{chosen['part_name']}** - {parts_context.persona_decision.get('final_action', '')}")
        
        return "\n".join(sections)


def create_parts_runtime(storage) -> PartsRuntime:
    """Create a PartsRuntime instance with all components."""
    import importlib.util
    from pathlib import Path

    bid_engine = None
    persona_layer = None
    vector_store = None

    # Try to create bid engine directly (without going through tools/__init__.py)
    try:
        _bid_engine_path = Path(__file__).parent / "bid_engine.py"
        spec = importlib.util.spec_from_file_location("bid_engine", _bid_engine_path)
        _bid_engine_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_bid_engine_mod)
        bid_engine = _bid_engine_mod.BidEngine()
    except Exception as e:
        logger.warning(f"Could not create bid engine: {e}")

    # Try to create persona layer directly
    try:
        _persona_path = Path(__file__).parent / "persona_layer.py"
        spec = importlib.util.spec_from_file_location("persona_layer", _persona_path)
        _persona_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_persona_mod)
        persona_layer = _persona_mod.PersonaLayer()
    except Exception as e:
        logger.warning(f"Could not create persona layer: {e}")

    # Try to create vector store directly
    try:
        _vector_path = Path(__file__).parent / "vector_store.py"
        spec = importlib.util.spec_from_file_location("vector_store", _vector_path)
        _vector_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_vector_mod)
        vector_store = _vector_mod.PartsVectorStore()
    except Exception as e:
        logger.warning(f"Could not create vector store: {e}")

    return PartsRuntime(
        storage=storage,
        vector_store=vector_store,
        bid_engine=bid_engine,
        persona_layer=persona_layer
    )
