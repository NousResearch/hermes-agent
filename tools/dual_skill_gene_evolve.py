"""
Fix for dual_skill_gene_evolve (Issue #597)
==========================================
Problem: No self-evolution mechanism - skills cannot become genes, no closed loop

Root Cause Analysis:
- Skills are created and managed by curator for organization
- Skills can be archived/deleted but not evolved into higher forms
- No mechanism to convert proven skills into core agent capabilities (genes)
- No feedback loop from skill usage back to agent improvement

Fix Implementation:
1. Introduce "Gene" concept - core capabilities that are built into the agent
2. Add skill-to-gene conversion pathway: proven skill → candidate gene → approved gene
3. Add gene expression: gene → agent behavior modification
4. Add evolution tracking: which genes came from which skills
"""

from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional, Set
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from enum import Enum
import json
import threading
import hashlib

# ============================================================================
# Gene Data Models
# ============================================================================

class GeneState(Enum):
    """Lifecycle states for a gene."""
    CANDIDATE = "candidate"      # Proposed from skill conversion
    APPROVED = "approved"        # Reviewed and approved
    ACTIVE = "active"           # Currently expressed in agent
    DEPRECATED = "deprecated"   # Superseded by newer gene
    ARCHIVED = "archived"       # Moved to archive


class GeneOrigin(Enum):
    """Where a gene originated from."""
    BUILTIN = "builtin"         # Hard-coded in agent
    SKILL_CANDIDATE = "skill"   # Converted from skill
    USER_DEFINED = "user"       # Created by user
    DERIVED = "derived"         # Combination of other genes


@dataclass
class Gene:
    """A gene - core capability encoded into the agent."""
    gene_id: str
    name: str
    description: str
    source_skill: Optional[str] = None  # If converted from skill
    source_genes: List[str] = field(default_factory=list)  # If derived
    origin: str = GeneOrigin.BUILTIN.value
    state: str = GeneState.CANDIDATE.value
    code_path: Optional[str] = None  # Module/class implementing this gene
    parameters: Dict[str, Any] = field(default_factory=dict)  # Gene parameters
    usage_count: int = 0
    success_rate: float = 0.0
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    activated_at: Optional[str] = None
    deprecated_at: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SkillGeneCandidate:
    """A skill that has been identified as a gene candidate."""
    skill_name: str
    skill_path: str
    gene_name: str
    gene_description: str
    conversion_score: float  # How good a gene candidate (0-1)
    quality_score: float     # Skill quality (0-1)
    usage_count: int
    success_evidence: List[str] = field(default_factory=list)
    failure_evidence: List[str] = field(default_factory=list)
    conversion_notes: str = ""
    proposed_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    reviewed: bool = False
    approved: bool = False


# ============================================================================
# Gene Registry
# ============================================================================

_genes_lock = threading.Lock()
_genes: Dict[str, Gene] = {}
_gene_candidates: Dict[str, SkillGeneCandidate] = {}
_GENOME_DIR = "~/.hermes/genes"


def _get_genome_dir() -> 'Path':
    """Get the genome storage directory."""
    from pathlib import Path
    import os
    path = Path(os.path.expanduser(_GENOME_DIR))
    path.mkdir(parents=True, exist_ok=True)
    (path / "candidates").mkdir(parents=True, exist_ok=True)
    (path / "active").mkdir(parents=True, exist_ok=True)
    return path


def _gene_id_from_name(name: str) -> str:
    """Generate a stable gene ID from name."""
    return hashlib.sha256(name.lower().encode()).hexdigest()[:16]


# ============================================================================
# Skill to Gene Conversion
# ============================================================================

SKILL_TO_GENE_PROMPT = """You are evaluating whether a skill should be converted into a GENE.

Genes are CORE capabilities that are encoded directly into the agent's behavior,
not just stored as reference documents. A skill becomes a gene when:
1. It represents a frequently-used, high-value capability
2. Its instructions can be operationalized (not just informational)
3. It would improve agent performance if "baked in" rather than looked up

Analyze this skill:

Skill: {skill_name}
Content: {skill_content}

Evaluation criteria:
1. OPERATIONALIZABLE (40%): Can this be converted into executable behavior?
   - Does it have concrete steps, not just descriptions?
   - Can the logic be encoded programmatically?

2. HIGH VALUE (30%): Would this significantly improve agent capability?
   - Is this a common, important task?
   - Does the skill represent expertise that should be innate?

3. STABILITY (20%): Is this skill proven over many uses?
   - Has it been used successfully multiple times?
   - Are failure cases well understood?

4. CONVERSION FEASIBILITY (10%): Can we actually convert this?
   - Is the skill well-structured enough to convert?
   - Are dependencies minimal?

Output:
```json
{{
  "should_convert": true/false,
  "conversion_score": 0.0-1.0,
  "gene_name": "proposed_name_for_the_gene",
  "gene_description": "what_this_gene_does",
  "conversion_rationale": "why this should become a gene",
  "operationalization_notes": "how to encode this as code",
  "concerns": ["any concerns about conversion"]
}}
```
"""


def evaluate_skill_to_gene_candidate(
    skill_name: str,
    skill_content: str,
    usage_count: int,
    llm_callback: Optional[Callable[[str], str]] = None,
) -> SkillGeneCandidate:
    """
    Evaluate whether a skill should become a gene candidate.
    
    Args:
        skill_name: Name of the skill
        skill_content: Full SKILL.md content
        usage_count: How many times this skill was used
        llm_callback: Optional LLM callback
    
    Returns:
        SkillGeneCandidate with conversion assessment
    """
    if llm_callback:
        try:
            prompt = SKILL_TO_GENE_PROMPT.format(
                skill_name=skill_name,
                skill_content=skill_content[:6000],
            )
            response = llm_callback(prompt)
            assessment = _parse_conversion_assessment(response)
            
            return SkillGeneCandidate(
                skill_name=skill_name,
                skill_path=f"~/.hermes/skills/{skill_name}",
                gene_name=assessment.get("gene_name", f"gene_{skill_name}"),
                gene_description=assessment.get("gene_description", ""),
                conversion_score=assessment.get("conversion_score", 0.0),
                quality_score=0.5,  # Would come from quality audit
                usage_count=usage_count,
                success_evidence=[],
                failure_evidence=[],
                conversion_notes=assessment.get("conversion_rationale", ""),
            )
        except Exception:
            pass
    
    # Heuristic fallback
    return _heuristic_skill_evaluation(skill_name, usage_count)


def _parse_conversion_assessment(response: str) -> Dict[str, Any]:
    """Parse LLM conversion assessment response."""
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


def _heuristic_skill_evaluation(skill_name: str, usage_count: int) -> SkillGeneCandidate:
    """Heuristic skill-to-gene evaluation."""
    # Simple heuristics
    conversion_score = 0.3
    
    # Higher usage = higher score
    if usage_count >= 10:
        conversion_score += 0.3
    elif usage_count >= 5:
        conversion_score += 0.2
    elif usage_count >= 1:
        conversion_score += 0.1
    
    # Known operationalizable patterns
    operational_keywords = ["run", "execute", "build", "create", "deploy", "install", "setup"]
    if any(kw in skill_name.lower() for kw in operational_keywords):
        conversion_score += 0.2
    
    # Code-related skills are good candidates
    if any(kw in skill_name.lower() for kw in ["code", "debug", "test", "lint", "format"]):
        conversion_score += 0.2
    
    return SkillGeneCandidate(
        skill_name=skill_name,
        skill_path=f"~/.hermes/skills/{skill_name}",
        gene_name=f"gene_{skill_name}",
        gene_description=f"Capability derived from skill: {skill_name}",
        conversion_score=min(1.0, conversion_score),
        quality_score=0.5,
        usage_count=usage_count,
    )


# ============================================================================
# Gene Lifecycle Management
# ============================================================================

def propose_gene_candidate(candidate: SkillGeneCandidate) -> str:
    """
    Propose a new gene candidate from a skill.
    
    Returns the gene_id.
    """
    gene_id = _gene_id_from_name(candidate.gene_name)
    
    with _genes_lock:
        if gene_id in _gene_candidates:
            return gene_id
        
        _gene_candidates[gene_id] = candidate
        
        # Persist to disk
        _save_candidate(gene_id, candidate)
    
    return gene_id


def approve_gene_candidate(
    gene_id: str,
    reviewer_notes: str = "",
) -> Optional[Gene]:
    """
    Approve a gene candidate, creating an active gene.
    
    Args:
        gene_id: The candidate gene ID
        reviewer_notes: Notes from the review process
    
    Returns:
        The approved Gene, or None if not found
    """
    with _genes_lock:
        if gene_id not in _gene_candidates:
            return None
        
        candidate = _gene_candidates[gene_id]
        candidate.reviewed = True
        candidate.approved = True
        
        # Create gene
        gene = Gene(
            gene_id=gene_id,
            name=candidate.gene_name,
            description=candidate.gene_description,
            source_skill=candidate.skill_name,
            origin=GeneOrigin.SKILL_CANDIDATE.value,
            state=GeneState.APPROVED.value,
            metadata={
                "reviewer_notes": reviewer_notes,
                "conversion_notes": candidate.conversion_notes,
                "conversion_score": candidate.conversion_score,
            },
        )
        
        _genes[gene_id] = gene
        _save_gene(gene)
        
        return gene


def activate_gene(gene_id: str) -> Optional[Gene]:
    """
    Activate a gene so it's expressed in agent behavior.
    
    Returns the activated Gene, or None if not found.
    """
    with _genes_lock:
        if gene_id not in _genes:
            return None
        
        gene = _genes[gene_id]
        gene.state = GeneState.ACTIVE.value
        gene.activated_at = datetime.now(timezone.utc).isoformat()
        
        _save_gene(gene)
        
        return gene


def deprecate_gene(gene_id: str, replacement_id: Optional[str] = None) -> Optional[Gene]:
    """
    Deprecate an active gene.
    
    Args:
        gene_id: Gene to deprecate
        replacement_id: Optional replacement gene ID
    
    Returns:
        The deprecated Gene, or None if not found
    """
    with _genes_lock:
        if gene_id not in _genes:
            return None
        
        gene = _genes[gene_id]
        gene.state = GeneState.DEPRECATED.value
        gene.deprecated_at = datetime.now(timezone.utc).isoformat()
        
        if replacement_id:
            gene.metadata["replaced_by"] = replacement_id
        
        _save_gene(gene)
        
        return gene


# ============================================================================
# Gene Expression (Agent Behavior Integration)
# ============================================================================

def express_genes() -> List[Gene]:
    """
    Get all active genes that should be expressed in agent behavior.
    
    This is called during agent initialization to load genes.
    """
    with _genes_lock:
        return [
            g for g in _genes.values()
            if g.state == GeneState.ACTIVE.value
        ]


def get_gene_behavior_modifications() -> Dict[str, Any]:
    """
    Get behavior modifications from active genes.
    
    Returns a dict suitable for merging into agent config.
    """
    active_genes = express_genes()
    
    modifications = {
        "prompt_modifications": [],
        "tool_preferences": {},
        "behavior_rules": [],
        "capability_overrides": {},
    }
    
    for gene in active_genes:
        # Merge gene parameters into behavior
        if "prompt_addition" in gene.metadata:
            modifications["prompt_modifications"].append({
                "gene_id": gene.gene_id,
                "addition": gene.metadata["prompt_addition"],
            })
        
        if "tool_preference" in gene.metadata:
            modifications["tool_preferences"][gene.gene_id] = gene.metadata["tool_preference"]
        
        if "behavior_rule" in gene.metadata:
            modifications["behavior_rules"].append({
                "gene_id": gene.gene_id,
                "rule": gene.metadata["behavior_rule"],
            })
    
    return modifications


# ============================================================================
# Evolution Loop
# ============================================================================

def run_evolution_cycle(
    min_usage_threshold: int = 5,
    min_quality_threshold: float = 0.7,
    llm_callback: Optional[Callable[[str], str]] = None,
) -> Dict[str, Any]:
    """
    Run a complete evolution cycle.
    
    This:
    1. Scans skills for conversion candidates
    2. Evaluates candidates
    3. Auto-approves high-confidence candidates
    4. Returns evolution results
    
    Args:
        min_usage_threshold: Minimum usage count to consider
        min_quality_threshold: Minimum quality score to auto-approve
        llm_callback: Optional LLM for evaluation
    
    Returns:
        Evolution cycle results
    """
    from tools import skill_usage
    
    results = {
        "cycle_at": datetime.now(timezone.utc).isoformat(),
        "candidates_evaluated": 0,
        "candidates_proposed": 0,
        "candidates_auto_approved": 0,
        "genes_activated": 0,
        "details": [],
    }
    
    # Get skill usage data
    try:
        report = skill_usage.agent_created_report()
    except Exception:
        return {**results, "error": "Could not load skill report"}
    
    # Find high-usage, high-quality skills
    for row in report:
        skill_name = row.get("name", "")
        usage_count = row.get("use_count", 0) + row.get("view_count", 0)
        
        if usage_count < min_usage_threshold:
            continue
        
        # Load skill content
        try:
            skill_path = skill_usage._skills_dir() / skill_name / "SKILL.md"
            if not skill_path.exists():
                continue
            content = skill_path.read_text(encoding='utf-8')
        except Exception:
            continue
        
        # Evaluate conversion potential
        candidate = evaluate_skill_to_gene_candidate(
            skill_name=skill_name,
            skill_content=content,
            usage_count=usage_count,
            llm_callback=llm_callback,
        )
        
        results["candidates_evaluated"] += 1
        
        # Auto-approve high-confidence candidates
        if (candidate.conversion_score >= 0.8 and 
            candidate.quality_score >= min_quality_threshold):
            
            gene_id = propose_gene_candidate(candidate)
            
            # Auto-approve and activate
            gene = approve_gene_candidate(gene_id, "Auto-approved by evolution cycle")
            if gene:
                gene = activate_gene(gene_id)
                results["candidates_auto_approved"] += 1
                results["genes_activated"] += 1
                results["details"].append({
                    "skill": skill_name,
                    "gene_id": gene_id,
                    "gene_name": candidate.gene_name,
                    "conversion_score": candidate.conversion_score,
                    "action": "activated",
                })
        else:
            # Just propose for review
            if candidate.conversion_score >= 0.5:
                propose_gene_candidate(candidate)
                results["candidates_proposed"] += 1
    
    return results


# ============================================================================
# Persistence
# ============================================================================

def _save_candidate(gene_id: str, candidate: SkillGeneCandidate) -> None:
    """Persist candidate to disk."""
    try:
        genome_dir = _get_genome_dir()
        path = genome_dir / "candidates" / f"{gene_id}.json"
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(asdict(candidate), f, indent=2)
    except Exception:
        pass


def _save_gene(gene: Gene) -> None:
    """Persist gene to disk."""
    try:
        genome_dir = _get_genome_dir()
        if gene.state == GeneState.ACTIVE.value:
            path = genome_dir / "active" / f"{gene.gene_id}.json"
        else:
            path = genome_dir / "genes" / f"{gene.gene_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(gene.to_dict(), f, indent=2)
    except Exception:
        pass


def load_all_genes() -> None:
    """Load all genes from disk into memory."""
    global _genes, _gene_candidates
    
    genome_dir = _get_genome_dir()
    
    # Load candidates
    candidates_dir = genome_dir / "candidates"
    if candidates_dir.exists():
        for path in candidates_dir.glob("*.json"):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    gene_id = data.get("gene_name", "")
                    if gene_id:
                        _gene_candidates[_gene_id_from_name(gene_id)] = SkillGeneCandidate(**data)
            except Exception:
                pass
    
    # Load active genes
    active_dir = genome_dir / "active"
    if active_dir.exists():
        for path in active_dir.glob("*.json"):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    gene = Gene(**data)
                    _genes[gene.gene_id] = gene
            except Exception:
                pass


# ============================================================================
# INTEGRATION PATCHES
# ============================================================================
"""
To integrate these fixes, add to the agent system:

1. Add gene system to curator:
   - In curator.py, after consolidation passes, call run_evolution_cycle()
   - This converts high-value skills into gene candidates

2. Add gene management tools:
   - gene_list: List all genes and their states
   - gene_propose: Manually propose a skill as gene candidate
   - gene_approve: Review and approve candidates
   - gene_activate: Activate approved genes
   - gene_evolve: Run full evolution cycle

3. Modify agent initialization:
   - On startup, call load_all_genes()
   - Call express_genes() to get active gene behaviors
   - Merge gene behavior modifications into agent config

4. Add gene tracking to skill usage:
   - When skill is used successfully, increment usage_count
   - When skill usage leads to gene candidate, mark skill as "evolving"

5. Create gene expression system:
   - Genes add modifications to system prompt
   - Genes can modify tool selection behavior
   - Genes can add behavioral rules
"""
