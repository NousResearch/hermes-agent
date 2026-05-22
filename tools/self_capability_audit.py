"""
Fix for self_capability_audit (Issue #596)
==========================================
Problem: Agent doesn't know what it can do - no self-discovery of capabilities

Root Cause Analysis:
- Agent relies on tools being declared in tool definitions
- No introspection mechanism to discover "what can I do in situation X"
- No capability registry that tracks what the agent has actually demonstrated
- No self-awareness of skill limitations or boundaries

Fix Implementation:
1. Add CapabilityRegistry that tracks demonstrated capabilities
2. Add self-discovery mechanism that probes for capability boundaries
3. Add introspection tools: capability_query, capability_test, capability_learn
4. Build dynamic capability profile based on successful executions
"""

from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from enum import Enum
import json
import threading

# ============================================================================
# Capability Data Models
# ============================================================================

class CapabilityCategory(Enum):
    """Broad categories of capabilities."""
    CODE = "code"                    # Programming, debugging, code generation
    FILE = "file"                    # File operations, reading, writing
    TERMINAL = "terminal"           # Command execution
    WEB = "web"                     # Web search, browsing
    DATA = "data"                   # Data processing, analysis
    CREATIVE = "creative"           # Writing, design, art
    REASONING = "reasoning"         # Analysis, problem-solving
    PLANNING = "planning"           # Task planning, scheduling
    COMMUNICATION = "communication"  # User interaction, documentation
    TOOL = "tool"                  # Using external tools, APIs


class CapabilityConfidence(Enum):
    """How confident the agent is in a capability."""
    UNTESTED = "untested"          # Never tried
    FAILED = "failed"              # Tried and failed
    UNCERTAIN = "uncertain"        # Some successes, some failures
    CONFIDENT = "confident"         # Consistently successful
    EXPERT = "expert"              # Highly refined, handles edge cases


@dataclass
class Capability:
    """A single capability with metadata."""
    name: str
    category: str
    description: str
    confidence: str = "untested"
    success_count: int = 0
    failure_count: int = 0
    last_tested: Optional[str] = None
    last_success: Optional[str] = None
    context_patterns: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    related_tools: List[str] = field(default_factory=list)
    
    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        if total == 0:
            return 0.0
        return self.success_count / total
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass 
class CapabilityProfile:
    """Complete capability profile for an agent."""
    agent_id: str
    capabilities: Dict[str, Capability] = field(default_factory=dict)
    skill_knowledge: Dict[str, str] = field(default_factory=dict)  # skill_name -> skill_id
    tool_mastery: Dict[str, float] = field(default_factory=dict)   # tool_name -> mastery (0-1)
    boundaries: List[str] = field(default_factory=list)            # Known limitations
    updated_at: Optional[str] = None
    
    def get_capability(self, name: str) -> Optional[Capability]:
        return self.capabilities.get(name)
    
    def by_category(self, category: str) -> List[Capability]:
        return [c for c in self.capabilities.values() if c.category == category]
    
    def confident_capabilities(self) -> List[Capability]:
        return [c for c in self.capabilities.values() 
                if c.confidence in ("confident", "expert")]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "capabilities": {k: v.to_dict() for k, v in self.capabilities.items()},
            "skill_knowledge": self.skill_knowledge,
            "tool_mastery": self.tool_mastery,
            "boundaries": self.boundaries,
            "updated_at": self.updated_at,
        }


# ============================================================================
# Capability Registry
# ============================================================================

_capability_registry_lock = threading.Lock()
_capability_registry: Dict[str, CapabilityProfile] = {}


def get_capability_registry(agent_id: str = "default") -> CapabilityProfile:
    """Get or create the capability profile for an agent."""
    with _capability_registry_lock:
        if agent_id not in _capability_registry:
            _capability_registry[agent_id] = CapabilityProfile(agent_id=agent_id)
        return _capability_registry[agent_id]


def _default_capabilities() -> Dict[str, Capability]:
    """Return the default built-in capabilities."""
    return {
        "code_generation": Capability(
            name="code_generation",
            category=CapabilityCategory.CODE.value,
            description="Generate code in various programming languages",
            confidence="confident",
            related_tools=["execute_code", "skills_tool"],
        ),
        "code_debugging": Capability(
            name="code_debugging", 
            category=CapabilityCategory.CODE.value,
            description="Debug code issues, read error messages, propose fixes",
            confidence="confident",
            related_tools=["execute_code", "debug_helpers"],
        ),
        "file_reading": Capability(
            name="file_reading",
            category=CapabilityCategory.FILE.value,
            description="Read files from the filesystem",
            confidence="confident",
            related_tools=["read_file", "terminal"],
        ),
        "file_writing": Capability(
            name="file_writing",
            category=CapabilityCategory.FILE.value,
            description="Write or modify files on the filesystem",
            confidence="confident",
            related_tools=["write_file", "patch"],
        ),
        "terminal_execution": Capability(
            name="terminal_execution",
            category=CapabilityCategory.TERMINAL.value,
            description="Execute shell commands in terminal",
            confidence="confident",
            related_tools=["terminal", "execute_code"],
        ),
        "web_search": Capability(
            name="web_search",
            category=CapabilityCategory.WEB.value,
            description="Search the web for information",
            confidence="confident",
            related_tools=["web_search_provider", "browser_tool"],
        ),
        "skill_management": Capability(
            name="skill_management",
            category=CapabilityCategory.TOOL.value,
            description="Create, view, and manage skills",
            confidence="confident",
            related_tools=["skill_manage", "skill_view", "skills_list"],
        ),
        "delegate_to_subagent": Capability(
            name="delegate_to_subagent",
            category=CapabilityCategory.PLANNING.value,
            description="Delegate tasks to subagents for parallel processing",
            confidence="confident",
            related_tools=["delegate_task"],
        ),
    }


def initialize_capability_profile(
    agent_id: str = "default",
    use_defaults: bool = True,
) -> CapabilityProfile:
    """
    Initialize a new capability profile for an agent.
    
    Args:
        agent_id: Unique identifier for the agent
        use_defaults: If True, populate with built-in default capabilities
    
    Returns:
        The initialized CapabilityProfile
    """
    profile = get_capability_registry(agent_id)
    
    if use_defaults and not profile.capabilities:
        for name, cap in _default_capabilities().items():
            profile.capabilities[name] = cap
    
    profile.updated_at = datetime.now(timezone.utc).isoformat()
    return profile


# ============================================================================
# Capability Self-Discovery
# ============================================================================

SELF_DISCOVERY_PROMPT = """You are performing a SELF-CAPABILITY DISCOVERY session.

Your goal is to honestly assess what you can and cannot do. This is NOT about
being modest or confident - it's about ACCURATE self-assessment.

For each capability area, consider:
1. What have you successfully done in this area?
2. What have you tried and failed at?
3. What are the boundaries of your ability?
4. What context makes this easier/harder?

Evaluate these capability dimensions:

1. CODE - Programming abilities
   - What languages can you write confidently?
   - What frameworks have you successfully used?
   - What kinds of bugs can you reliably find and fix?

2. REASONING - Analysis and problem-solving
   - What complex reasoning tasks can you handle?
   - Where do you tend to make mistakes?

3. TOOL USE - Your ability to use tools effectively
   - Which tools do you use masterfully?
   - Which tools give you trouble?

4. CREATIVE - Generation and ideation
   - What creative tasks do you excel at?
   - What creative tasks are challenging?

5. BOUNDARIES - What you CANNOT do
   - What limitations have you discovered?
   - What tasks should NOT be delegated to you?

Output your assessment in this JSON format:
```json
{{
  "self_assessment": {{
    "code": {{
      "strengths": ["list of strengths"],
      "weaknesses": ["list of weaknesses"],
      "confidence": "high|medium|low",
      "examples": ["successful task examples"]
    }},
    "reasoning": {{...}},
    "tool_use": {{...}},
    "creative": {{...}},
    "boundaries": ["known limitations"]
  }},
  "capabilities_to_add": ["capability names that should be added to profile"],
  "capabilities_to_remove": ["capability names that are inaccurate"],
  "confidence_adjustments": {{
    "capability_name": "new_confidence_level"
  }}
}}
```
"""


def run_self_capability_discovery(
    agent_id: str = "default",
    llm_callback: Optional[Callable[[str], str]] = None,
    use_defaults: bool = True,
) -> Dict[str, Any]:
    """
    Run a self-discovery session to build honest capability profile.
    
    This uses introspection and, optionally, LLM assistance to build
    an accurate picture of what the agent can do.
    
    Args:
        agent_id: Agent identifier
        llm_callback: Optional LLM callback for assisted discovery
        use_defaults: Whether to initialize with default capabilities
    
    Returns:
        Discovery results with profile updates
    """
    profile = initialize_capability_profile(agent_id, use_defaults)
    
    results = {
        "agent_id": agent_id,
        "discovery_method": "llm_assisted" if llm_callback else "introspection",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "profile_snapshot": profile.to_dict(),
    }
    
    if llm_callback:
        try:
            response = llm_callback(SELF_DISCOVERY_PROMPT)
            discovery = _parse_discovery_response(response)
            results["discovery"] = discovery
            
            # Apply discoveries to profile
            _apply_discovery_to_profile(profile, discovery)
        except Exception as e:
            results["error"] = str(e)
            results["discovery"] = None
    
    profile.updated_at = datetime.now(timezone.utc).isoformat()
    results["profile_snapshot"] = profile.to_dict()
    
    return results


def _parse_discovery_response(response: str) -> Dict[str, Any]:
    """Parse LLM self-discovery response."""
    try:
        start = response.find("```json")
        if start != -1:
            start += 7
            end = response.find("```", start)
            if end != -1:
                json_str = response[start:end].strip()
            else:
                json_str = response[start:].strip()
        else:
            json_str = response.strip()
        
        return json.loads(json_str)
    except json.JSONDecodeError:
        return {}


def _apply_discovery_to_profile(
    profile: CapabilityProfile,
    discovery: Dict[str, Any],
) -> None:
    """Apply self-discovery results to capability profile."""
    if not discovery:
        return
    
    # Adjust confidence levels
    adjustments = discovery.get("confidence_adjustments", {})
    for cap_name, new_confidence in adjustments.items():
        if cap_name in profile.capabilities:
            profile.capabilities[cap_name].confidence = new_confidence
    
    # Add new capabilities
    for cap_name in discovery.get("capabilities_to_add", []):
        if cap_name not in profile.capabilities:
            profile.capabilities[cap_name] = Capability(
                name=cap_name,
                category=CapabilityCategory.REASONING.value,  # Default
                description=f"Capability discovered via self-assessment: {cap_name}",
                confidence="uncertain",
            )
    
    # Update boundaries
    for boundary in discovery.get("boundaries", []):
        if boundary not in profile.boundaries:
            profile.boundaries.append(boundary)


# ============================================================================
# Runtime Capability Tracking
# ============================================================================

def record_capability_outcome(
    capability_name: str,
    success: bool,
    context: Optional[str] = None,
    agent_id: str = "default",
) -> None:
    """
    Record the outcome of a capability attempt.
    
    This updates the confidence level based on success/failure patterns.
    """
    profile = get_capability_registry(agent_id)
    
    if capability_name not in profile.capabilities:
        profile.capabilities[capability_name] = Capability(
            name=capability_name,
            category=CapabilityCategory.REASONING.value,
            description=f"Discovered capability: {capability_name}",
            confidence="untested",
        )
    
    cap = profile.capabilities[capability_name]
    cap.last_tested = datetime.now(timezone.utc).isoformat()
    
    if success:
        cap.success_count += 1
        cap.last_success = cap.last_tested
        if context and context not in cap.context_patterns:
            cap.context_patterns.append(context)
        # Update confidence based on success rate
        rate = cap.success_rate()
        if rate >= 0.9 and cap.success_count >= 5:
            cap.confidence = CapabilityConfidence.EXPERT.value
        elif rate >= 0.75 and cap.success_count >= 3:
            cap.confidence = CapabilityConfidence.CONFIDENT.value
        elif rate >= 0.5:
            cap.confidence = CapabilityConfidence.UNCERTAIN.value
    else:
        cap.failure_count += 1
        if context and context not in cap.limitations:
            cap.limitations.append(context)
        # Downgrade confidence on failures
        if cap.success_count == 0 and cap.failure_count >= 3:
            cap.confidence = CapabilityConfidence.FAILED.value
    
    profile.updated_at = datetime.now(timezone.utc).isoformat()


def query_capabilities(
    query: str,
    agent_id: str = "default",
) -> Dict[str, Any]:
    """
    Query the capability profile for relevant capabilities.
    
    Args:
        query: Natural language query about what the agent can do
        agent_id: Agent identifier
    
    Returns:
        Relevant capabilities and a response to the query
    """
    profile = get_capability_registry(agent_id)
    
    query_lower = query.lower()
    
    # Find relevant capabilities
    relevant = []
    for cap in profile.capabilities.values():
        score = 0.0
        if any(word in cap.description.lower() for word in query_lower.split()):
            score += 0.5
        if any(word in cap.name.lower() for word in query_lower.split()):
            score += 0.3
        if cap.category in query_lower:
            score += 0.2
        if score > 0:
            relevant.append((score, cap))
    
    relevant.sort(key=lambda x: x[0], reverse=True)
    
    # Build response
    if relevant:
        response_parts = []
        for score, cap in relevant[:5]:
            conf_emoji = {
                "expert": "🟢 Expert",
                "confident": "🟢 Confident", 
                "uncertain": "🟡 Uncertain",
                "failed": "🔴 Unreliable",
                "untested": "⚪ Unknown",
            }.get(cap.confidence, cap.confidence)
            
            response_parts.append(
                f"- **{cap.name}** ({conf_emoji}): {cap.description}"
            )
        
        response = "Based on my capability profile:\n" + "\n".join(response_parts)
    else:
        response = "I don't have specific capabilities matching that query. Could you be more specific?"
    
    return {
        "query": query,
        "response": response,
        "relevant_capabilities": [cap.to_dict() for _, cap in relevant[:5]],
        "profile_summary": {
            "total_capabilities": len(profile.capabilities),
            "confident_count": len(profile.confident_capabilities()),
            "boundaries": profile.boundaries,
        },
    }


# ============================================================================
# INTEGRATION PATCHES
# ============================================================================
"""
To integrate these fixes, add to the agent's toolset:

1. New tool: capability_query
   - Input: query string
   - Output: JSON with relevant capabilities and natural language response
   - Uses query_capabilities() function

2. New tool: capability_learn  
   - Input: capability_name, success (bool), context (optional)
   - Output: Confirmation of recorded outcome
   - Uses record_capability_outcome() function

3. New tool: capability_discovery
   - Input: None or llm_callback flag
   - Output: Self-discovery results
   - Uses run_self_capability_discovery() function

4. New tool: capability_profile
   - Input: None
   - Output: Full capability profile as JSON
   - Uses get_capability_registry() function

5. In the main agent loop, add capability recording:
   - After successful tool execution: record_capability_outcome(tool_name, True)
   - After failed tool execution: record_capability_outcome(tool_name, False)

6. Add capability introspection to system prompt building:
   - Include relevant capabilities in context when agent asks "what can I do"
"""
