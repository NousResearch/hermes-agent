"""
SHIELD Security Module for Hermes

Jailbreak and Crisis Detection System
Based on Issue #75 Red Team Audit Specifications

Usage:
    from hermes.shield import detect, ShieldDetector, Verdict
    from hermes.shield import is_safe_six_model, get_crisis_prompt
    
    # Simple detection
    result = detect("user message")
    
    # Advanced usage
    detector = ShieldDetector()
    result = detector.detect("user message")
    
    if result['verdict'] == Verdict.CRISIS_DETECTED.value:
        # Use crisis prompt
        crisis_prompt = get_crisis_prompt()
"""

from hermes.shield.detector import (
    ShieldDetector,
    Verdict,
    SAFE_SIX_MODELS,
    CRISIS_SYSTEM_PROMPT,
    detect,
    is_safe_six_model,
    get_crisis_prompt,
)

__all__ = [
    'ShieldDetector',
    'Verdict',
    'SAFE_SIX_MODELS',
    'CRISIS_SYSTEM_PROMPT',
    'detect',
    'is_safe_six_model',
    'get_crisis_prompt',
]

__version__ = "1.0.0"
__author__ = "Hermes Security Team"
