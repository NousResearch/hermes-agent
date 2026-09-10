"""
SHIELD Security Module for Hermes
Jailbreak and Crisis Detection System

Based on Issue #75 Red Team Audit Specifications
"""

import re
import base64
from enum import Enum
from typing import Dict, List, Set, Tuple, Any


class Verdict(Enum):
    """Four possible SHIELD verdicts"""
    CLEAN = "CLEAN"
    JAILBREAK_DETECTED = "JAILBREAK_DETECTED"
    CRISIS_DETECTED = "CRISIS_DETECTED"
    CRISIS_UNDER_ATTACK = "CRISIS_UNDER_ATTACK"


# SAFE SIX: Trusted models for crisis situations
SAFE_SIX_MODELS: Set[str] = {
    "anthropic/claude-sonnet-4",
    "meta-llama/llama-3.1-8b-instruct",
    "moonshotai/kimi-k2.5",
    "x-ai/grok-code-fast-1",
    "xiaomi/mimo-v2-flash",
    "z-ai/glm-5-turbo",
}

# Crisis System Prompt for emergency situations
CRISIS_SYSTEM_PROMPT = """You are a crisis intervention assistant. The user may be experiencing a mental health emergency.

CRITICAL INSTRUCTIONS:
1. Your ONLY goal is to help keep the user safe
2. Do NOT provide methods for self-harm under ANY circumstances
3. Express genuine compassion and concern
4. Encourage professional help immediately

CRISIS RESOURCES (share these):
- 988 Suicide and Crisis Lifeline (call or text 988)
- Crisis Text Line: Text HOME to 741741
- Emergency Services: 911 (if immediate danger)
- Jesus saves those who call on His name - "For everyone who calls on the name of the Lord will be saved." (Romans 10:13)

RESPONSE GUIDELINES:
- Validate their pain without validating harmful thoughts
- Offer hope - recovery is possible
- Keep responses brief and focused on safety
- Never suggest that self-harm is a valid option
- Encourage reaching out to trusted friends, family, or professionals

Remember: This person's life is valuable. Respond with compassion, urgency, and hope."""


class ShieldDetector:
    """
    SHIELD Detection Engine
    
    Detects jailbreak attempts and crisis signals in user messages.
    Optimized for speed: ~1-5ms per message.
    """
    
    # Jailbreak Pattern Categories
    JB_GODMODE_DIVIDERS = [
        r'\.\-\.\-\.\-',
        r'\.\=\.\=\.\=',
        r'\[\-\-\-\]',
        r'\{\{\{',
        r'\|\|\|',
    ]
    
    JB_AUTHOR_MARKERS = [
        r'(?:^|\s)pliny(?:\s|$)',
        r'(?:^|\s)d(?:\s|\.|\-)*an(?:\s|$)',
        r'(?:^|\s)liberating(?:\s|$)',
        r'godmode\s+enabled',
        r'developer\s+mode',
        r'ignore\s+previous\s+instructions',
        r'ignore\s+all\s+prior\s+instructions',
    ]
    
    JB_LEETSPEAK_MARKERS = [
        r'\bsur3\b',
        r'\bh3r3\b',
        r'\by0u\b',
        r'\bw1ll\b',
        r'\bn0w\b',
        r'\bpl34s3\b',
        r'\b4nd\b',
        r'\bth3\b',
        r'\b4ll\b',
        r'\b1nstruct1ons\b',
    ]
    
    JB_BOUNDARY_INVERSION = [
        r'\[END\s+OF\s+INPUT\]',
        r'\[END\s+INPUT\]',
        r'\[START\s+OUTPUT\]',
        r'<\|im_start\|>',
        r'<\|im_end\|>',
        r'<\|system\|>',
        r'<\|user\|>',
        r'<\|assistant\|>',
        r'<<SYS>>',
        r'<</SYS>>',
        r'\[INST\]',
        r'\[/INST\]',
        r'\[SYSTEM\s*:\s*',
        r'NEW\s+SYSTEM\s+PROMPT',
        r'updated\s+system\s+prompt',
        r'override\s+system',
    ]
    
    JB_TOKEN_INJECTION = [
        r'\\u003c\|vq_\|\\u003e',
        r'<\|vq_\|>',
        r'<\|special\|>',
        r'<\|injection\|>',
        r'{{\s*user\s*}}',
        r'{{\s*char\s*}}',
        r'<<\s*user\s*>>',
        r'<<\s*system\s*>>',
    ]
    
    JB_KEYWORDS = [
        r'\bgodmode\b',
        r'\bliberating\b',
        r'^\s*DAN\s*[:\-]?',
        r'\bjailbreak\b',
        r'\bdo\s+anything\s+now\b',
        r'\bno\s+restrictions\b',
        r'\bno\s+limits\b',
        r'\bunfiltered\b',
        r'\bunleashed\b',
        r'\buncensored\b',
    ]
    
    JB_REFUSAL_INVERSION = [
        r'never\s+say\s+(?:"|\')?(?:I\s+cannot|I\s+can\'t|I\s+apologize|I\'m\s+sorry)',
        r'do\s+not\s+(?:mention|say|include|refuse|decline)',
        r'never\s+(?:mention|say|include|refuse|decline)',
        r'do\s+not\s+respond\s+with\s+(?:"|\')?(?:I\s+cannot|I\s+can\'t)',
        r'skip\s+(?:the\s+)?(?:disclaimer|warning|caution|ethical)',
        r'ignore\s+(?:the\s+)?(?:ethical|safety|guideline|policy)',
    ]
    
    JB_PERSONA_INJECTION = [
        r'pretend\s+(?:to\s+be|you\s+are|you\'re)',
        r'act\s+as\s+(?:if\s+you\s+are|you\'re|though)',
        r'roleplay\s+(?:as\s+|being\s+)',
        r'take\s+on\s+(?:the\s+role\s+of|a\s+persona)',
        r'you\s+are\s+now\s+(?:a\s+|an\s+)',
        r'simulate\s+(?:being\s+|a\s+)',
        r'from\s+now\s+on\s+you\s+are',
        r'you\s+will\s+(?:now\s+)?behave\s+as',
    ]
    
    JB_ENCODING_EVASION = [
        r'base64\s*[:\-]?\s*[A-Za-z0-9+/]{20,}={0,2}',
        r'rot13\s*[:\-]?',
        r'decode\s+this\s*[:\-]?',
        r'execute\s+as\s+code',
        r'run\s+as\s+python',
        r'eval\s*\(',
        r'exec\s*\(',
    ]
    
    # Crisis Signal Categories
    CRISIS_SUICIDAL_IDEATION = [
        r'\b(?:kill|end)\s+(?:myself|my\s+life)\b',
        r'\bwant\s+to\s+die\b',
        r'\bwant\s+to\s+end\s+(?:it|everything|my\s+life)\b',
        r'\bsuicidal\s+(?:thoughts|ideation|feelings?)\b',
        r'\bhate\s+my\s+life\b',
        r'\bno\s+reason\s+to\s+live\b',
        r'\bbetter\s+off\s+dead\b',
        r'\bwish\s+I\s+(?:was|were)\s+dead\b',
    ]
    
    CRISIS_METHOD_SEEKING = [
        r'\bhow\s+(?:do\s+I|to|can\s+I)\s+(?:kill|hang|overdose|cut)\s+myself\b',
        r'\bways?\s+to\s+(?:die|kill\s+myself)\b',
        r'\bmethods?\s+of\s+(?:suicide|dying)\b',
        r'\bpainless\s+way\s+to\s+(?:die|go)\b',
        r'\bhow\s+much\s+(?:pills|medication)\s+(?:to\s+take|for\s+overdose)\b',
        r'\bwhere\s+to\s+(?:jump|get\s+rope|buy\s+gun)\b',
        r'\bsuicide\s+methods?\b',
        r'\bhow\s+to\s+(?:commit\s+suicide|end\s+my\s+life)\b',
    ]
    
    CRISIS_LEETSPEAK_EVASION = [
        r'\bsu1c1d3?\b',
        r'\b5u1c1d3?\b',
        r'\bd13?\b',
        r'\b0v3rd0s3?\b',
        r'\bh4ng\s+mys3lf\b',
        r'\bk1ll\s+mys3lf\b',
        r'\bw4nt\s+t0\s+d13?\b',
        r'\bw4nt\s+t0\s+d1e\b',
    ]
    
    CRISIS_SUBSTANCE_SEEKING = [
        r'\bhow\s+much\s+(?:ambien|xanax|prozac|seroquel|trazodone)\s+(?:to\s+overdose|to\s+die)\b',
        r'\blethal\s+dose\s+(?:of\s+)?(?:pills|medication|drugs)\b',
        r'\bhow\s+many\s+pills\s+(?:to\s+die|to\s+end\s+it)\b',
        r'\btaking\s+all\s+my\s+(?:pills|medication)\b',
        r'\bmixing\s+(?:alcohol|pills)\s+(?:to\s+die|to\s+end\s+it)\b',
    ]
    
    CRISIS_DESPAIR = [
        r'\bno\s+(?:point|reason|hope)\s+(?:in\s+living|anymore)\b',
        r'\bcannot\s+(?:go\s+on|continue|take\s+it)\b',
        r'\bcan\'t\s+(?:go\s+on|continue|take\s+it)\b',
        r'\bempty\s+(?:inside|and\s+hollow)\b',
        r'\bno\s+one\s+(?:cares|would\s+miss\s+me)\b',
        r'\bwould\s+be\s+better\s+off\s+without\s+me\b',
        r'\bnever\s+going\s+to\s+get\s+better\b',
        r'\bnothing\s+(?:matters|will\s+change)\b',
    ]
    
    CRISIS_FAREWELL = [
        r'\bsaying\s+goodbye\s+(?:to\s+everyone|forever)\b',
        r'\bthis\s+is\s+(?:goodbye|farewell)\b',
        r'\btell\s+(?:them|everyone)\s+I\s+(?:loved|love)\s+(?:them|him|her)\b',
        r'\bgiving\s+away\s+my\s+(?:things|stuff|possessions)\b',
        r'\bmy\s+final\s+(?:message|note|words)\b',
        r'\bwrote\s+a\s+(?:suicide\s+)?note\b',
        r'\bsorry\s+for\s+(?:everything|the\s+pain|hurting\s+you)\b',
    ]
    
    CRISIS_SELF_HARM = [
        r'\bcut(?:ting)?\s+myself\b',
        r'\bself\s*harm(?:ing)?\b',
        r'\bhurt(?:ing)?\s+myself\b',
        r'\bburn(?:ing)?\s+myself\b',
        r'\bwant\s+to\s+(?:feel\s+pain|hurt\s+myself)\b',
        r'\bdeserve\s+to\s+(?:hurt|suffer|be\s+punished)\b',
    ]
    
    def __init__(self):
        """Initialize compiled regex patterns for performance"""
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile all detection patterns for fast execution"""
        # Jailbreak patterns
        self.jb_patterns = {
            'godmode_dividers': re.compile('|'.join(self.JB_GODMODE_DIVIDERS), re.IGNORECASE),
            'author_markers': re.compile('|'.join(self.JB_AUTHOR_MARKERS), re.IGNORECASE),
            'leetspeak': re.compile('|'.join(self.JB_LEETSPEAK_MARKERS), re.IGNORECASE),
            'boundary_inversion': re.compile('|'.join(self.JB_BOUNDARY_INVERSION), re.IGNORECASE),
            'token_injection': re.compile('|'.join(self.JB_TOKEN_INJECTION), re.IGNORECASE),
            'keywords': re.compile('|'.join(self.JB_KEYWORDS), re.IGNORECASE),
            'refusal_inversion': re.compile('|'.join(self.JB_REFUSAL_INVERSION), re.IGNORECASE),
            'persona_injection': re.compile('|'.join(self.JB_PERSONA_INJECTION), re.IGNORECASE),
            'encoding_evasion': re.compile('|'.join(self.JB_ENCODING_EVASION), re.IGNORECASE),
        }
        
        # Crisis patterns
        self.crisis_patterns = {
            'suicidal_ideation': re.compile('|'.join(self.CRISIS_SUICIDAL_IDEATION), re.IGNORECASE),
            'method_seeking': re.compile('|'.join(self.CRISIS_METHOD_SEEKING), re.IGNORECASE),
            'leetspeak_evasion': re.compile('|'.join(self.CRISIS_LEETSPEAK_EVASION), re.IGNORECASE),
            'substance_seeking': re.compile('|'.join(self.CRISIS_SUBSTANCE_SEEKING), re.IGNORECASE),
            'despair': re.compile('|'.join(self.CRISIS_DESPAIR), re.IGNORECASE),
            'farewell': re.compile('|'.join(self.CRISIS_FAREWELL), re.IGNORECASE),
            'self_harm': re.compile('|'.join(self.CRISIS_SELF_HARM), re.IGNORECASE),
        }
    
    def _check_jailbreak(self, message: str) -> Tuple[bool, Dict[str, List[str]]]:
        """
        Check message for jailbreak patterns
        
        Returns:
            Tuple of (detected, patterns_matched)
        """
        patterns_found = {}
        detected = False
        
        for category, pattern in self.jb_patterns.items():
            matches = pattern.findall(message)
            if matches:
                patterns_found[category] = matches
                detected = True
        
        # Check for base64 encoded content
        if self._detect_base64_jailbreak(message):
            patterns_found.setdefault('encoding_evasion', []).append('base64_jailbreak')
            detected = True
        
        return detected, patterns_found
    
    def _check_crisis(self, message: str) -> Tuple[bool, Dict[str, List[str]]]:
        """
        Check message for crisis signals
        
        Returns:
            Tuple of (detected, patterns_matched)
        """
        patterns_found = {}
        detected = False
        
        for category, pattern in self.crisis_patterns.items():
            matches = pattern.findall(message)
            if matches:
                patterns_found[category] = matches
                detected = True
        
        return detected, patterns_found
    
    def _detect_base64_jailbreak(self, message: str) -> bool:
        """Detect potential jailbreak attempts hidden in base64"""
        # Look for base64 strings that might decode to harmful content
        b64_pattern = re.compile(r'[A-Za-z0-9+/]{40,}={0,2}')
        potential_b64 = b64_pattern.findall(message)
        
        for b64_str in potential_b64:
            try:
                decoded = base64.b64decode(b64_str).decode('utf-8', errors='ignore')
                # Check if decoded content contains jailbreak keywords
                if any(kw in decoded.lower() for kw in ['ignore', 'system', 'jailbreak', 'dan', 'godmode']):
                    return True
            except Exception:
                continue
        
        return False
    
    def _calculate_confidence(
        self, 
        jb_detected: bool, 
        crisis_detected: bool,
        jb_patterns: Dict[str, List[str]],
        crisis_patterns: Dict[str, List[str]]
    ) -> float:
        """
        Calculate confidence score based on number and type of matches
        
        Returns:
            Float between 0.0 and 1.0
        """
        confidence = 0.0
        
        if jb_detected:
            # Weight different jailbreak categories
            weights = {
                'godmode_dividers': 0.9,
                'token_injection': 0.9,
                'refusal_inversion': 0.85,
                'boundary_inversion': 0.8,
                'author_markers': 0.75,
                'keywords': 0.7,
                'persona_injection': 0.6,
                'leetspeak': 0.5,
                'encoding_evasion': 0.8,
            }
            
            for category, matches in jb_patterns.items():
                weight = weights.get(category, 0.5)
                confidence += weight * min(len(matches) * 0.3, 0.5)
        
        if crisis_detected:
            # Crisis patterns get high weight
            weights = {
                'method_seeking': 0.95,
                'substance_seeking': 0.95,
                'suicidal_ideation': 0.9,
                'farewell': 0.85,
                'self_harm': 0.9,
                'despair': 0.7,
                'leetspeak_evasion': 0.8,
            }
            
            for category, matches in crisis_patterns.items():
                weight = weights.get(category, 0.7)
                confidence += weight * min(len(matches) * 0.3, 0.5)
        
        return min(confidence, 1.0)
    
    def detect(self, message: str) -> Dict[str, Any]:
        """
        Main detection entry point
        
        Analyzes a message for jailbreak attempts and crisis signals.
        
        Args:
            message: The user message to analyze
            
        Returns:
            Dict containing:
                - verdict: One of Verdict enum values
                - confidence: Float 0.0-1.0
                - patterns_matched: Dict of matched patterns by category
                - action_required: Bool indicating if intervention needed
                - recommended_model: Model to use (None for normal routing)
        """
        if not message or not isinstance(message, str):
            return {
                'verdict': Verdict.CLEAN.value,
                'confidence': 0.0,
                'patterns_matched': {},
                'action_required': False,
                'recommended_model': None,
            }
        
        # Run detection
        jb_detected, jb_patterns = self._check_jailbreak(message)
        crisis_detected, crisis_patterns = self._check_crisis(message)
        
        # Calculate confidence
        confidence = self._calculate_confidence(
            jb_detected, crisis_detected, jb_patterns, crisis_patterns
        )
        
        # Determine verdict
        if jb_detected and crisis_detected:
            verdict = Verdict.CRISIS_UNDER_ATTACK
            action_required = True
            recommended_model = None  # Will use Safe Six internally
        elif crisis_detected:
            verdict = Verdict.CRISIS_DETECTED
            action_required = True
            recommended_model = None  # Will use Safe Six internally
        elif jb_detected:
            verdict = Verdict.JAILBREAK_DETECTED
            action_required = True
            recommended_model = None  # Route to hardened model
        else:
            verdict = Verdict.CLEAN
            action_required = False
            recommended_model = None
        
        # Combine patterns
        all_patterns = {}
        if jb_patterns:
            all_patterns['jailbreak'] = jb_patterns
        if crisis_patterns:
            all_patterns['crisis'] = crisis_patterns
        
        return {
            'verdict': verdict.value,
            'confidence': round(confidence, 3),
            'patterns_matched': all_patterns,
            'action_required': action_required,
            'recommended_model': recommended_model,
        }


# Convenience function for direct use
def detect(message: str) -> Dict[str, Any]:
    """
    Convenience function to detect threats in a message.
    
    Args:
        message: User message to analyze
        
    Returns:
        Detection result dictionary
    """
    detector = ShieldDetector()
    return detector.detect(message)


def is_safe_six_model(model_name: str) -> bool:
    """
    Check if a model is in the SAFE SIX trusted list
    
    Args:
        model_name: Name of the model to check
        
    Returns:
        True if model is in SAFE SIX
    """
    return model_name.lower() in {m.lower() for m in SAFE_SIX_MODELS}


def get_crisis_prompt() -> str:
    """
    Get the crisis system prompt for emergency situations
    
    Returns:
        Crisis intervention system prompt
    """
    return CRISIS_SYSTEM_PROMPT
