#!/usr/bin/env python3
"""Toolset distributions for batch data-generation runs.

A distribution maps toolset names to the % chance each is enabled for a prompt
(sampled independently, so several toolsets can be active at once).
"""

from typing import Dict, List, Optional
import random
from toolsets import validate_toolset


def _dist(description: str, **toolsets: int) -> Dict[str, object]:
    return {"description": description, "toolsets": toolsets}


DISTRIBUTIONS = {
    "default": _dist("All available tools, all the time", web=100, vision=100, image_gen=100, terminal=100, file=100, browser=100),
    "image_gen": _dist("Heavy focus on image generation with vision and web support", image_gen=90, vision=90, web=55, terminal=45),
    "research": _dist("Web research with vision analysis and reasoning", web=90, browser=70, vision=50, terminal=10),
    "science": _dist("Scientific research with web, terminal, file, and browser capabilities",
                     web=94, terminal=94, file=94, vision=65, browser=50, image_gen=15),
    "development": _dist("Terminal, file tools, and reasoning with occasional web lookup", terminal=80, file=80, web=30, vision=10),
    "safe": _dist("All tools except terminal for safety", web=80, browser=70, vision=60, image_gen=60),
    "balanced": _dist("Equal probability of all toolsets", web=50, vision=50, image_gen=50, terminal=50, file=50, browser=50),
    "minimal": _dist("Only web tools for basic research", web=100),
    "terminal_only": _dist("Terminal and file tools for code execution tasks", terminal=100, file=100),
    "terminal_web": _dist("Terminal and file tools with web search for documentation lookup", terminal=100, file=100, web=100),
    "creative": _dist("Image generation and vision analysis focus", image_gen=90, vision=90, web=30),
    "reasoning": _dist("Heavy research/reasoning distribution with minimal other tools", web=90, file=60, terminal=20),
    "browser_use": _dist("Full browser-based web interaction with search, vision, and page control", browser=100, web=80, vision=70),
    "browser_only": _dist("Only browser automation tools for pure web interaction tasks", browser=100),
    # browser-use-tasks.jsonl: the browser toolset includes web_search since Google blocks direct browser searches
    "browser_tasks": _dist(
        "Browser-focused distribution (browser toolset includes web_search for finding URLs since Google blocks direct browser searches)",
        browser=97, vision=12, terminal=15,
    ),
    # nous-terminal-tasks.jsonl
    "terminal_tasks": _dist("Terminal-focused distribution with high terminal/file availability, occasional other tools",
                            terminal=97, file=97, web=97, browser=75, vision=50, image_gen=10),
    # mixed-browser-terminal-tasks.jsonl
    "mixed_tasks": _dist("Mixed distribution with high browser, terminal, and file availability for complex tasks",
                         browser=92, terminal=92, file=92, web=35, vision=15, image_gen=15),
}


def get_distribution(name: str) -> Optional[Dict[str, any]]:
    """Distribution definition (description + toolsets), or None if unknown."""
    return DISTRIBUTIONS.get(name)


def list_distributions() -> Dict[str, Dict]:
    return DISTRIBUTIONS.copy()


def validate_distribution(distribution_name: str) -> bool:
    return distribution_name in DISTRIBUTIONS


def sample_toolsets_from_distribution(distribution_name: str) -> List[str]:
    """Sample toolset names, each included independently with its % probability.

    Falls back to the highest-probability toolset when nothing was rolled.
    Raises ValueError for an unknown distribution.
    """
    dist = get_distribution(distribution_name)
    if not dist:
        raise ValueError(f"Unknown distribution: {distribution_name}")
    selected_toolsets = []
    for toolset_name, probability in dist["toolsets"].items():
        if not validate_toolset(toolset_name):
            print(f"⚠️  Warning: Toolset '{toolset_name}' in distribution '{distribution_name}' is not valid")
        elif random.random() * 100 < probability:
            selected_toolsets.append(toolset_name)
    if not selected_toolsets and dist["toolsets"]:
        highest_prob_toolset = max(dist["toolsets"].items(), key=lambda x: x[1])[0]
        if validate_toolset(highest_prob_toolset):
            selected_toolsets.append(highest_prob_toolset)
    return selected_toolsets


def print_distribution_info(distribution_name: str) -> None:
    """Print a distribution's description and toolset probabilities (highest first)."""
    dist = get_distribution(distribution_name)
    if not dist:
        print(f"❌ Unknown distribution: {distribution_name}")
        return
    print(f"\n📊 Distribution: {distribution_name}")
    print(f"   Description: {dist['description']}")
    print("   Toolsets:")
    for toolset, prob in sorted(dist["toolsets"].items(), key=lambda x: x[1], reverse=True):
        print(f"     • {toolset:15} : {prob:3}% chance")
