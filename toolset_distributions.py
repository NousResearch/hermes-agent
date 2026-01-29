#!/usr/bin/env python3
"""
Toolset Distributions Module

This module defines distributions of toolsets for data generation runs.
Each distribution specifies which toolsets should be used and their probability
of being selected for any given prompt during the batch processing.

A distribution is a dictionary mapping toolset names to their selection probability (%).
Probabilities should sum to 100, but the system will normalize if they don't.

Usage:
    from toolset_distributions import get_distribution, list_distributions
    
    # Get a specific distribution
    dist = get_distribution("image_gen")
    
    # List all available distributions
    all_dists = list_distributions()
"""

from typing import Dict, List, Optional
import random
from toolsets import validate_toolset


# Distribution definitions
# Each key is a distribution name, and the value is a dict of toolset_name: probability_percentage
DISTRIBUTIONS = {
    # Default: All tools available 100% of the time
    "default": {
        "description": "All available tools, all the time",
        "toolsets": {
            "web": 100,
            "vision": 100,
            "image_gen": 100,
            "terminal": 100,
            "moa": 100,
            "browser": 100
        }
    },
    
    # Image generation focused distribution
    "image_gen": {
        "description": "Heavy focus on image generation with vision and web support",
        "toolsets": {
            "image_gen": 90,  # 80% chance of image generation tools
            "vision": 90,      # 60% chance of vision tools
            "web": 55,         # 40% chance of web tools
            "terminal": 45,
            "moa": 10          # 20% chance of reasoning tools
        }
    },
    
    # Research-focused distribution
    "research": {
        "description": "Web research with vision analysis and reasoning",
        "toolsets": {
            "web": 90,       # 90% chance of web tools
            "browser": 70,   # 70% chance of browser tools for deep research
            "vision": 50,    # 50% chance of vision tools
            "moa": 40,       # 40% chance of reasoning tools
            "terminal": 10   # 10% chance of terminal tools
        }
    },

    # Scientific problem solving focused distribution
    "science": {
        "description": "Scientific research with web, terminal, and browser capabilities",
        "toolsets": {
            "web": 94,       # 94% chance of web tools
            "terminal": 94,  # 94% chance of terminal tools
            "vision": 65,    # 65% chance of vision tools
            "browser": 50,   # 50% chance of browser for accessing papers/databases
            "image_gen": 15, # 15% chance of image generation tools
            "moa": 10        # 10% chance of reasoning tools
        }
    },

    # Development-focused distribution
    "development": {
        "description": "Terminal and reasoning with occasional web lookup",
        "toolsets": {
            "terminal": 80,  # 80% chance of terminal tools
            "moa": 60,       # 60% chance of reasoning tools
            "web": 30,       # 30% chance of web tools
            "vision": 10     # 10% chance of vision tools
        }
    },
    
    # Safe mode (no terminal)
    "safe": {
        "description": "All tools except terminal for safety",
        "toolsets": {
            "web": 80,
            "browser": 70,   # Browser is safe (no local filesystem access)
            "vision": 60,
            "image_gen": 60,
            "moa": 50
        }
    },
    
    # Balanced distribution
    "balanced": {
        "description": "Equal probability of all toolsets",
        "toolsets": {
            "web": 50,
            "vision": 50,
            "image_gen": 50,
            "terminal": 50,
            "moa": 50,
            "browser": 50
        }
    },
    
    # Minimal (web only)
    "minimal": {
        "description": "Only web tools for basic research",
        "toolsets": {
            "web": 100
        }
    },
    
    # Terminal only
    "terminal_only": {
        "description": "Only terminal tool for code execution tasks",
        "toolsets": {
            "terminal": 100
        }
    },
    
    # Terminal + web (common for coding tasks that need docs)
    "terminal_web": {
        "description": "Terminal with web search for documentation lookup",
        "toolsets": {
            "terminal": 100,
            "web": 100
        }
    },
    
    # Creative (vision + image generation)
    "creative": {
        "description": "Image generation and vision analysis focus",
        "toolsets": {
            "image_gen": 90,
            "vision": 90,
            "web": 30
        }
    },
    
    # Reasoning heavy
    "reasoning": {
        "description": "Heavy mixture of agents usage with minimal other tools",
        "toolsets": {
            "moa": 90,
            "web": 30,
            "terminal": 20
        }
    },
    
    # Browser-based web interaction
    "browser_use": {
        "description": "Full browser-based web interaction with search, vision, and page control",
        "toolsets": {
            "browser": 100,  # All browser tools always available
            "web": 80,       # Web search for finding URLs and quick lookups
            "vision": 70     # Vision analysis for images found on pages
        }
    },
    
    # Browser only (no other tools)
    "browser_only": {
        "description": "Only browser automation tools for pure web interaction tasks",
        "toolsets": {
            "browser": 100
        }
    },
    
    # Browser-focused tasks distribution (for browser-use-tasks.jsonl)
    "browser_tasks": {
        "description": "Browser-focused distribution (browser toolset includes web_search for finding URLs since Google blocks direct browser searches)",
        "toolsets": {
            "browser": 97,   # 97% - browser tools (includes web_search) almost always available
            "vision": 12,    # 12% - vision analysis occasionally
            "terminal": 15   # 15% - terminal occasionally for local operations
        }
    },
    
    # Terminal-focused tasks distribution (for nous-terminal-tasks.jsonl)
    "terminal_tasks": {
        "description": "Terminal-focused distribution with high terminal availability, occasional other tools",
        "toolsets": {
            "terminal": 97,   # 97% - terminal almost always available
            "web": 15,        # 15% - web search/scrape for documentation
            "browser": 10,    # 10% - browser occasionally for web interaction
            "vision": 8,      # 8% - vision analysis rarely
            "image_gen": 3    # 3% - image generation very rarely
        }
    },
    
    # Mixed browser+terminal tasks distribution (for mixed-browser-terminal-tasks.jsonl)
    "mixed_tasks": {
        "description": "Mixed distribution with high browser and terminal availability for complex tasks",
        "toolsets": {
            "browser": 92,    # 92% - browser tools highly available
            "terminal": 92,   # 92% - terminal highly available  
            "web": 35,        # 35% - web search/scrape fairly common
            "vision": 15,     # 15% - vision analysis occasionally
            "image_gen": 15   # 15% - image generation occasionally
        }
    }
}


def get_distribution(name: str) -> Optional[Dict[str, any]]:
    """
    Get a toolset distribution by name.
    
    Args:
        name (str): Name of the distribution
        
    Returns:
        Dict: Distribution definition with description and toolsets
        None: If distribution not found
    """
    return DISTRIBUTIONS.get(name)


def list_distributions() -> Dict[str, Dict]:
    """
    List all available distributions.
    
    Returns:
        Dict: All distribution definitions
    """
    return DISTRIBUTIONS.copy()


def sample_toolsets_from_distribution(distribution_name: str) -> List[str]:
    """
    Sample toolsets based on a distribution's probabilities.
    
    Each toolset in the distribution has a % chance of being included.
    This allows multiple toolsets to be active simultaneously.
    
    Args:
        distribution_name (str): Name of the distribution to sample from
        
    Returns:
        List[str]: List of sampled toolset names
        
    Raises:
        ValueError: If distribution name is not found
    """
    dist = get_distribution(distribution_name)
    if not dist:
        raise ValueError(f"Unknown distribution: {distribution_name}")
    
    # Sample each toolset independently based on its probability
    selected_toolsets = []
    
    for toolset_name, probability in dist["toolsets"].items():
        # Validate toolset exists
        if not validate_toolset(toolset_name):
            print(f"⚠️  Warning: Toolset '{toolset_name}' in distribution '{distribution_name}' is not valid")
            continue
        
        # Roll the dice - if random value is less than probability, include this toolset
        if random.random() * 100 < probability:
            selected_toolsets.append(toolset_name)
    
    # If no toolsets were selected (can happen with low probabilities), 
    # ensure at least one toolset is selected by picking the highest probability one
    if not selected_toolsets and dist["toolsets"]:
        # Find toolset with highest probability
        highest_prob_toolset = max(dist["toolsets"].items(), key=lambda x: x[1])[0]
        if validate_toolset(highest_prob_toolset):
            selected_toolsets.append(highest_prob_toolset)
    
    return selected_toolsets


def validate_distribution(distribution_name: str) -> bool:
    """
    Check if a distribution name is valid.
    
    Args:
        distribution_name (str): Distribution name to validate
        
    Returns:
        bool: True if valid, False otherwise
    """
    return distribution_name in DISTRIBUTIONS


def print_distribution_info(distribution_name: str) -> None:
    """
    Print detailed information about a distribution.
    
    Args:
        distribution_name (str): Distribution name
    """
    dist = get_distribution(distribution_name)
    if not dist:
        print(f"❌ Unknown distribution: {distribution_name}")
        return
    
    print(f"\n📊 Distribution: {distribution_name}")
    print(f"   Description: {dist['description']}")
    print(f"   Toolsets:")
    for toolset, prob in sorted(dist["toolsets"].items(), key=lambda x: x[1], reverse=True):
        print(f"     • {toolset:15} : {prob:3}% chance")


if __name__ == "__main__":
    """
    Demo and testing of the distributions system
    """
    print("📊 Toolset Distributions Demo")
    print("=" * 60)
    
    # List all distributions
    print("\n📋 Available Distributions:")
    print("-" * 40)
    for name, dist in list_distributions().items():
        print(f"\n  {name}:")
        print(f"    {dist['description']}")
        toolset_list = ", ".join([f"{ts}({p}%)" for ts, p in dist["toolsets"].items()])
        print(f"    Toolsets: {toolset_list}")
    
    # Demo sampling
    print("\n\n🎲 Sampling Examples:")
    print("-" * 40)
    
    test_distributions = ["image_gen", "research", "balanced", "default"]
    
    for dist_name in test_distributions:
        print(f"\n{dist_name}:")
        # Sample 5 times to show variability
        samples = []
        for _ in range(5):
            sampled = sample_toolsets_from_distribution(dist_name)
            samples.append(sorted(sampled))
        
        print(f"  Sample 1: {samples[0]}")
        print(f"  Sample 2: {samples[1]}")
        print(f"  Sample 3: {samples[2]}")
        print(f"  Sample 4: {samples[3]}")
        print(f"  Sample 5: {samples[4]}")
    
    # Show detailed info
    print("\n\n📊 Detailed Distribution Info:")
    print("-" * 40)
    print_distribution_info("image_gen")
    print_distribution_info("research")

