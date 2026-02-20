#!/usr/bin/env python3
"""
Skills Tool Module

This module provides tools for listing and viewing skill documents.
Skills are organized as directories containing a SKILL.md file (the main instructions)
and optional supporting files like references, templates, and examples.

Inspired by Anthropic's Claude Skills system with progressive disclosure architecture:
- Metadata (name ≤64 chars, description ≤1024 chars) - shown in skills_list
- Full Instructions - loaded via skill_view when needed
- Linked Files (references, templates) - loaded on demand

Directory Structure:
    skills/
    ├── my-skill/
    │   ├── SKILL.md           # Main instructions (required)
    │   ├── references/        # Supporting documentation
    │   │   ├── api.md
    │   │   └── examples.md
    │   ├── templates/         # Templates for output
    │   │   └── template.md
    │   └── assets/            # Supplementary files (agentskills.io standard)
    └── category/              # Category folder for organization
        └── another-skill/
            └── SKILL.md

SKILL.md Format (YAML Frontmatter, agentskills.io compatible):
    ---
    name: skill-name              # Required, max 64 chars
    description: Brief description # Required, max 1024 chars
    version: 1.0.0                # Optional
    license: MIT                  # Optional (agentskills.io)
    compatibility: Requires X     # Optional (agentskills.io)
    metadata:                     # Optional, arbitrary key-value (agentskills.io)
      hermes:
        tags: [fine-tuning, llm]
        related_skills: [peft, lora]
    ---
    
    # Skill Title
    
    Full instructions and content here...

Available tools:
- skills_list: List skills with metadata (progressive disclosure tier 1)
- skill_view: Load full skill content (progressive disclosure tier 2-3)

Usage:
    from tools.skills_tool import skills_list, skill_view, check_skills_requirements
    
    # List all skills (returns metadata only - token efficient)
    result = skills_list()
    
    # View a skill's main content (loads full instructions)
    content = skill_view("axolotl")
    
    # View a reference file within a skill (loads linked file)
    content = skill_view("axolotl", "references/dataset-formats.md")
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import yaml


# All skills live in ~/.hermes/skills/ (seeded from bundled skills/ on install).
# This is the single source of truth -- agent edits, hub installs, and bundled
# skills all coexist here without polluting the git repo.
HERMES_HOME = Path(os.getenv("HERMES_HOME", Path.home() / ".hermes"))
SKILLS_DIR = HERMES_HOME / "skills"

# Anthropic-recommended limits for progressive disclosure efficiency
MAX_NAME_LENGTH = 64
MAX_DESCRIPTION_LENGTH = 1024


def check_skills_requirements() -> bool:
    """Skills are always available -- the directory is created on first use if needed."""
    return True


def _parse_frontmatter(content: str) -> Tuple[Dict[str, Any], str]:
    """
    Parse YAML frontmatter from markdown content.
    
    Uses yaml.safe_load for full YAML support (nested metadata, lists, etc.)
    with a fallback to simple key:value splitting for robustness.
    
    Args:
        content: Full markdown file content
        
    Returns:
        Tuple of (frontmatter dict, remaining content)
    """
    frontmatter = {}
    body = content
    
    if content.startswith("---"):
        end_match = re.search(r'\n---\s*\n', content[3:])
        if end_match:
            yaml_content = content[3:end_match.start() + 3]
            body = content[end_match.end() + 3:]
            
            try:
                parsed = yaml.safe_load(yaml_content)
                if isinstance(parsed, dict):
                    frontmatter = parsed
                # yaml.safe_load returns None for empty frontmatter
            except yaml.YAMLError:
                # Fallback: simple key:value parsing for malformed YAML
                for line in yaml_content.strip().split('\n'):
                    if ':' in line:
                        key, value = line.split(':', 1)
                        frontmatter[key.strip()] = value.strip()
    
    return frontmatter, body


def _get_category_from_path(skill_path: Path) -> Optional[str]:
    """
    Extract category from skill path based on directory structure.
    
    For paths like: ~/.hermes/skills/mlops/axolotl/SKILL.md -> "mlops"
    """
    try:
        rel_path = skill_path.relative_to(SKILLS_DIR)
        parts = rel_path.parts
        if len(parts) >= 3:
            return parts[0]
        return None
    except ValueError:
        return None


def _estimate_tokens(content: str) -> int:
    """
    Rough token estimate (4 chars per token average).
    
    Args:
        content: Text content
        
    Returns:
        Estimated token count
    """
    return len(content) // 4


def _parse_tags(tags_value) -> List[str]:
    """
    Parse tags from frontmatter value.
    
    Handles:
    - Already-parsed list (from yaml.safe_load): [tag1, tag2]
    - String with brackets: "[tag1, tag2]"
    - Comma-separated string: "tag1, tag2"
    
    Args:
        tags_value: Raw tags value — may be a list or string
        
    Returns:
        List of tag strings
    """
    if not tags_value:
        return []
    
    # yaml.safe_load already returns a list for [tag1, tag2]
    if isinstance(tags_value, list):
        return [str(t).strip() for t in tags_value if t]
    
    # String fallback — handle bracket-wrapped or comma-separated
    tags_value = str(tags_value).strip()
    if tags_value.startswith('[') and tags_value.endswith(']'):
        tags_value = tags_value[1:-1]
    
    return [t.strip().strip('"\'') for t in tags_value.split(',') if t.strip()]


def _find_all_skills() -> List[Dict[str, Any]]:
    """
    Recursively find all skills in ~/.hermes/skills/.
    
    Returns metadata for progressive disclosure (tier 1):
    - name, description, category
    
    Returns:
        List of skill metadata dicts
    """
    skills = []
    
    if not SKILLS_DIR.exists():
        return skills
    
    for skill_md in SKILLS_DIR.rglob("SKILL.md"):
        path_str = str(skill_md)
        if '/.git/' in path_str or '/.github/' in path_str or '/.hub/' in path_str:
            continue
            
        skill_dir = skill_md.parent
        
        try:
            content = skill_md.read_text(encoding='utf-8')
            frontmatter, body = _parse_frontmatter(content)
            
            name = frontmatter.get('name', skill_dir.name)[:MAX_NAME_LENGTH]
            
            description = frontmatter.get('description', '')
            if not description:
                for line in body.strip().split('\n'):
                    line = line.strip()
                    if line and not line.startswith('#'):
                        description = line
                        break
            
            if len(description) > MAX_DESCRIPTION_LENGTH:
                description = description[:MAX_DESCRIPTION_LENGTH - 3] + "..."
            
            category = _get_category_from_path(skill_md)
            
            skills.append({
                "name": name,
                "description": description,
                "category": category,
            })
            
        except Exception:
            continue
    
    return skills


def _load_category_description(category_dir: Path) -> Optional[str]:
    """
    Load category description from DESCRIPTION.md if it exists.
    
    Args:
        category_dir: Path to the category directory
        
    Returns:
        Description string or None if not found
    """
    desc_file = category_dir / "DESCRIPTION.md"
    if not desc_file.exists():
        return None
    
    try:
        content = desc_file.read_text(encoding='utf-8')
        # Parse frontmatter if present
        frontmatter, body = _parse_frontmatter(content)
        
        # Prefer frontmatter description, fall back to first non-header line
        description = frontmatter.get('description', '')
        if not description:
            for line in body.strip().split('\n'):
                line = line.strip()
                if line and not line.startswith('#'):
                    description = line
                    break
        
        # Truncate to reasonable length
        if len(description) > MAX_DESCRIPTION_LENGTH:
            description = description[:MAX_DESCRIPTION_LENGTH - 3] + "..."
        
        return description if description else None
    except Exception:
        return None


def skills_categories(verbose: bool = False, task_id: str = None) -> str:
    """
    List available skill categories with descriptions (progressive disclosure tier 0).
    
    Returns category names and descriptions for efficient discovery before drilling down.
    Categories can have a DESCRIPTION.md file with a description frontmatter field
    or first paragraph to explain what skills are in that category.
    
    Args:
        verbose: If True, include skill counts per category (default: False, but currently always included)
        task_id: Optional task identifier (unused, for API consistency)
        
    Returns:
        JSON string with list of categories and their descriptions
    """
    try:
        if not SKILLS_DIR.exists():
            return json.dumps({
                "success": True,
                "categories": [],
                "message": "No skills directory found."
            }, ensure_ascii=False)
        
        category_dirs = {}
        for skill_md in SKILLS_DIR.rglob("SKILL.md"):
            category = _get_category_from_path(skill_md)
            if category:
                category_dir = SKILLS_DIR / category
                if category not in category_dirs:
                    category_dirs[category] = category_dir
        
        categories = []
        for name in sorted(category_dirs.keys()):
            category_dir = category_dirs[name]
            description = _load_category_description(category_dir)
            skill_count = sum(1 for _ in category_dir.rglob("SKILL.md"))
            
            cat_entry = {"name": name, "skill_count": skill_count}
            if description:
                cat_entry["description"] = description
            categories.append(cat_entry)
        
        return json.dumps({
            "success": True,
            "categories": categories,
            "hint": "If a category is relevant to your task, use skills_list with that category to see available skills"
        }, ensure_ascii=False)
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e)
        }, ensure_ascii=False)


def skills_list(category: str = None, task_id: str = None) -> str:
    """
    List all available skills (progressive disclosure tier 1 - minimal metadata).
    
    Returns only name + description to minimize token usage. Use skill_view() to 
    load full content, tags, related files, etc.
    
    Args:
        category: Optional category filter (e.g., "mlops")
        task_id: Optional task identifier (unused, for API consistency)
        
    Returns:
        JSON string with minimal skill info: name, description, category
    """
    try:
        if not SKILLS_DIR.exists():
            SKILLS_DIR.mkdir(parents=True, exist_ok=True)
            return json.dumps({
                "success": True,
                "skills": [],
                "categories": [],
                "message": "No skills found. Skills directory created at ~/.hermes/skills/"
            }, ensure_ascii=False)
        
        # Find all skills
        all_skills = _find_all_skills()
        
        if not all_skills:
            return json.dumps({
                "success": True,
                "skills": [],
                "categories": [],
                "message": "No skills found in skills/ directory."
            }, ensure_ascii=False)
        
        # Filter by category if specified
        if category:
            all_skills = [s for s in all_skills if s.get("category") == category]
        
        # Sort by category then name
        all_skills.sort(key=lambda s: (s.get("category") or "", s["name"]))
        
        # Extract unique categories
        categories = sorted(set(s.get("category") for s in all_skills if s.get("category")))
        
        return json.dumps({
            "success": True,
            "skills": all_skills,
            "categories": categories,
            "count": len(all_skills),
            "hint": "Use skill_view(name) to see full content, tags, and linked files"
        }, ensure_ascii=False)
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e)
        }, ensure_ascii=False)


def skill_view(name: str, file_path: str = None, task_id: str = None) -> str:
    """
    View the content of a skill or a specific file within a skill directory.
    
    Args:
        name: Name or path of the skill (e.g., "axolotl" or "03-fine-tuning/axolotl")
        file_path: Optional path to a specific file within the skill (e.g., "references/api.md")
        task_id: Optional task identifier (unused, for API consistency)
        
    Returns:
        JSON string with skill content or error message
    """
    try:
        if not SKILLS_DIR.exists():
            return json.dumps({
                "success": False,
                "error": "Skills directory does not exist yet. It will be created on first install."
            }, ensure_ascii=False)
        
        skill_dir = None
        skill_md = None
        
        # Try direct path first (e.g., "mlops/axolotl")
        direct_path = SKILLS_DIR / name
        if direct_path.is_dir() and (direct_path / "SKILL.md").exists():
            skill_dir = direct_path
            skill_md = direct_path / "SKILL.md"
        elif direct_path.with_suffix('.md').exists():
            skill_md = direct_path.with_suffix('.md')
        
        # Search by directory name
        if not skill_md:
            for found_skill_md in SKILLS_DIR.rglob("SKILL.md"):
                if found_skill_md.parent.name == name:
                    skill_dir = found_skill_md.parent
                    skill_md = found_skill_md
                    break
        
        # Legacy: flat .md files
        if not skill_md:
            for found_md in SKILLS_DIR.rglob(f"{name}.md"):
                if found_md.name != "SKILL.md":
                    skill_md = found_md
                    break
        
        if not skill_md or not skill_md.exists():
            # List available skills in error message
            all_skills = _find_all_skills()
            available = [s["name"] for s in all_skills[:20]]  # Limit to 20
            return json.dumps({
                "success": False,
                "error": f"Skill '{name}' not found.",
                "available_skills": available,
                "hint": "Use skills_list to see all available skills"
            }, ensure_ascii=False)
        
        # If a specific file path is requested, read that instead
        if file_path and skill_dir:
            target_file = skill_dir / file_path
            if not target_file.exists():
                # List available files in the skill directory, organized by type
                available_files = {
                    "references": [],
                    "templates": [],
                    "assets": [],
                    "scripts": [],
                    "other": []
                }
                
                # Scan for all readable files
                for f in skill_dir.rglob("*"):
                    if f.is_file() and f.name != "SKILL.md":
                        rel = str(f.relative_to(skill_dir))
                        if rel.startswith("references/"):
                            available_files["references"].append(rel)
                        elif rel.startswith("templates/"):
                            available_files["templates"].append(rel)
                        elif rel.startswith("assets/"):
                            available_files["assets"].append(rel)
                        elif rel.startswith("scripts/"):
                            available_files["scripts"].append(rel)
                        elif f.suffix in ['.md', '.py', '.yaml', '.yml', '.json', '.tex', '.sh']:
                            available_files["other"].append(rel)
                
                # Remove empty categories
                available_files = {k: v for k, v in available_files.items() if v}
                
                return json.dumps({
                    "success": False,
                    "error": f"File '{file_path}' not found in skill '{name}'.",
                    "available_files": available_files,
                    "hint": "Use one of the available file paths listed above"
                }, ensure_ascii=False)
            
            # Read the file content
            try:
                content = target_file.read_text(encoding='utf-8')
            except UnicodeDecodeError:
                # Binary file - return info about it instead
                return json.dumps({
                    "success": True,
                    "name": name,
                    "file": file_path,
                    "content": f"[Binary file: {target_file.name}, size: {target_file.stat().st_size} bytes]",
                    "is_binary": True
                }, ensure_ascii=False)
            
            return json.dumps({
                "success": True,
                "name": name,
                "file": file_path,
                "content": content,
                "file_type": target_file.suffix
            }, ensure_ascii=False)
        
        # Read the main skill content
        content = skill_md.read_text(encoding='utf-8')
        frontmatter, body = _parse_frontmatter(content)
        
        # Get reference, template, asset, and script files if this is a directory-based skill
        reference_files = []
        template_files = []
        asset_files = []
        script_files = []
        
        if skill_dir:
            references_dir = skill_dir / "references"
            if references_dir.exists():
                reference_files = [str(f.relative_to(skill_dir)) for f in references_dir.glob("*.md")]
            
            templates_dir = skill_dir / "templates"
            if templates_dir.exists():
                for ext in ['*.md', '*.py', '*.yaml', '*.yml', '*.json', '*.tex', '*.sh']:
                    template_files.extend([str(f.relative_to(skill_dir)) for f in templates_dir.rglob(ext)])
            
            # assets/ — agentskills.io standard directory for supplementary files
            assets_dir = skill_dir / "assets"
            if assets_dir.exists():
                for f in assets_dir.rglob("*"):
                    if f.is_file():
                        asset_files.append(str(f.relative_to(skill_dir)))
            
            scripts_dir = skill_dir / "scripts"
            if scripts_dir.exists():
                for ext in ['*.py', '*.sh', '*.bash', '*.js', '*.ts', '*.rb']:
                    script_files.extend([str(f.relative_to(skill_dir)) for f in scripts_dir.glob(ext)])
        
        # Read tags/related_skills with backward compat:
        # Check metadata.hermes.* first (agentskills.io convention), fall back to top-level
        hermes_meta = {}
        metadata = frontmatter.get('metadata')
        if isinstance(metadata, dict):
            hermes_meta = metadata.get('hermes', {}) or {}
        
        tags = _parse_tags(hermes_meta.get('tags') or frontmatter.get('tags', ''))
        related_skills = _parse_tags(hermes_meta.get('related_skills') or frontmatter.get('related_skills', ''))
        
        # Build linked files structure for clear discovery
        linked_files = {}
        if reference_files:
            linked_files["references"] = reference_files
        if template_files:
            linked_files["templates"] = template_files
        if asset_files:
            linked_files["assets"] = asset_files
        if script_files:
            linked_files["scripts"] = script_files
        
        rel_path = str(skill_md.relative_to(SKILLS_DIR))
        
        result = {
            "success": True,
            "name": frontmatter.get('name', skill_md.stem if not skill_dir else skill_dir.name),
            "description": frontmatter.get('description', ''),
            "tags": tags,
            "related_skills": related_skills,
            "content": content,
            "path": rel_path,
            "linked_files": linked_files if linked_files else None,
            "usage_hint": "To view linked files, call skill_view(name, file_path) where file_path is e.g. 'references/api.md' or 'assets/config.yaml'" if linked_files else None
        }
        
        # Surface agentskills.io optional fields when present
        if frontmatter.get('compatibility'):
            result["compatibility"] = frontmatter['compatibility']
        if isinstance(metadata, dict):
            result["metadata"] = metadata
        
        return json.dumps(result, ensure_ascii=False)
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e)
        }, ensure_ascii=False)


# Tool description for model_tools.py
SKILLS_TOOL_DESCRIPTION = """Access skill documents providing specialized instructions, guidelines, and executable knowledge.

Progressive disclosure workflow:
1. skills_list() - Returns metadata (name, description, tags, linked_file_count) for all skills
2. skill_view(name) - Loads full SKILL.md content + shows available linked_files
3. skill_view(name, file_path) - Loads specific linked file (e.g., 'references/api.md', 'scripts/train.py')

Skills may include:
- references/: Additional documentation, API specs, examples
- templates/: Output formats, config files, boilerplate code
- assets/: Supplementary files (agentskills.io standard)
- scripts/: Executable helpers (Python, shell scripts)"""


if __name__ == "__main__":
    """Test the skills tool"""
    print("🎯 Skills Tool Test")
    print("=" * 60)
    
    # Test listing skills
    print("\n📋 Listing all skills:")
    result = json.loads(skills_list())
    if result["success"]:
        print(f"Found {result['count']} skills in {len(result.get('categories', []))} categories")
        print(f"Categories: {result.get('categories', [])}")
        print("\nFirst 10 skills:")
        for skill in result["skills"][:10]:
            cat = f"[{skill['category']}] " if skill.get('category') else ""
            refs = f" (+{len(skill['reference_files'])} refs)" if skill.get('reference_files') else ""
            print(f"  • {cat}{skill['name']}: {skill['description'][:60]}...{refs}")
    else:
        print(f"Error: {result['error']}")
    
    # Test viewing a skill
    print("\n📖 Viewing skill 'axolotl':")
    result = json.loads(skill_view("axolotl"))
    if result["success"]:
        print(f"Name: {result['name']}")
        print(f"Description: {result.get('description', 'N/A')[:100]}...")
        print(f"Content length: {len(result['content'])} chars")
        if result.get('reference_files'):
            print(f"Reference files: {result['reference_files']}")
    else:
        print(f"Error: {result['error']}")
    
    # Test viewing a reference file
    print("\n📄 Viewing reference file 'axolotl/references/dataset-formats.md':")
    result = json.loads(skill_view("axolotl", "references/dataset-formats.md"))
    if result["success"]:
        print(f"File: {result['file']}")
        print(f"Content length: {len(result['content'])} chars")
        print(f"Preview: {result['content'][:150]}...")
    else:
        print(f"Error: {result['error']}")
