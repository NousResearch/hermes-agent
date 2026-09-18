"""Writes the setup facts agreed during onboarding into the default profile's user memory."""
import json

from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from hermes_cli.profiles import get_profile_dir
from tools.memory_tool import load_on_disk_store, memory_tool


def remember_onboarding(answers: dict) -> dict:
    if not isinstance(answers, dict):
        raise ValueError('Onboarding answers must be an object')
    facts = []
    for key, label in (('name', 'User prefers to be called'), ('context', 'Working on'),
                       ('theme', 'Desktop theme'), ('accent', 'Desktop accent'), ('layout', 'Desktop layout')):
        value = answers.get(key)
        if value is not None and not isinstance(value, str):
            raise ValueError(f'{key} must be text')
        if value and value.strip():
            facts.append(f'{label}: {value.strip()}')
    for key, label in (('focus', 'Focus areas'), ('connectors', 'Tools the user uses (not connection status)')):
        values = answers.get(key, [])
        if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
            raise ValueError(f'{key} must be a list of text')
        if values := list(dict.fromkeys(value.strip() for value in values if value.strip())):
            facts.append(f'{label}: {", ".join(values)}')
    if not facts:
        return {'saved': True, 'profile': 'default', 'target': 'user'}
    content = 'Agreed during onboarding:\n' + '\n'.join(facts)
    if len(content) > 2000:
        raise ValueError('Onboarding facts are too long to remember')

    # The entry must land in the 'default' profile directory even when this RPC arrives on the guide's
    # backend or under a custom Hermes home.
    token = set_hermes_home_override(get_profile_dir('default'))
    try:
        result = json.loads(memory_tool(action='add', target='user', content=content, store=load_on_disk_store()))
        if not result.get('success') or result.get('staged'):
            raise ValueError(result.get('error') or result.get('message') or 'Memory was not saved')
        # memory_tool can report success without the entry reaching disk, so read it back from a fresh store.
        if content not in load_on_disk_store().user_entries:
            raise ValueError('Could not verify saved onboarding facts')
        return {'saved': True, 'profile': 'default', 'target': 'user'}
    finally:
        reset_hermes_home_override(token)
