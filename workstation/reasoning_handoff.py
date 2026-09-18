"""Compact exception projection over the canonical artifact/checkpoint owners."""
from workstation.recipes import sanitize


def needs_reasoning(artifacts, owner, *, completed_until, expected, observed,
                    safe_to_resume, context=None):
    body = sanitize({'completed_until': completed_until, 'expected': expected,
                     'observed': observed, 'context': context or {},
                     'safe_to_resume': bool(safe_to_resume)})
    from workstation.recipes import digest
    ref = artifacts.store(owner, 'reasoning_' + digest(body) + '.json', body)
    return {'status': 'NEEDS_REASONING', 'completed_until': completed_until,
            'expected': 'See state_ref for the unresolved contract',
            'observed': 'Runtime requires semantic adaptation or reconciliation',
            'state_ref': ref.ref, 'safe_to_resume': bool(safe_to_resume)}
