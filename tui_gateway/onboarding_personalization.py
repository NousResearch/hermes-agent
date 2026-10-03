import json
import logging

from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from hermes_cli.profiles import get_profile_dir
from tools.memory_tool import ENTRY_DELIMITER, load_on_disk_store, memory_tool

logger = logging.getLogger(__name__)

_HEADER = 'Agreed during onboarding:'


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
    for key, label in (('focus', 'Focus areas'), ('connectors', 'Tools the user uses (not connection status)'),
                       ('plugins', 'Hermes plugins the user picked during onboarding (not install status)')):
        values = answers.get(key, [])
        if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
            raise ValueError(f'{key} must be a list of text')
        if values := list(dict.fromkeys(value.strip() for value in values if value.strip())):
            facts.append(f'{label}: {", ".join(values)}')
    if not facts:
        return {'saved': True, 'profile': 'default', 'target': 'user'}

    token = set_hermes_home_override(get_profile_dir('default'))
    try:
        store = load_on_disk_store()
        previous = [entry for entry in store.user_entries if entry.startswith(_HEADER)]
        others = [entry for entry in store.user_entries if not entry.startswith(_HEADER)]
        # USER.md is capped by user_char_limit and shared with what the agent already saved there:
        # keep the facts that fit instead of failing the first build with the model-facing "Consolidate now".
        fitting = []
        for fact in facts:
            if len(ENTRY_DELIMITER.join([*others, '\n'.join([_HEADER, *fitting, fact])])) <= store.user_char_limit:
                fitting.append(fact)
        if not fitting:
            raise ValueError(f'USER.md is full ({len(ENTRY_DELIMITER.join(others)):,}/{store.user_char_limit:,} '
                             'chars), so your onboarding answers could not be saved. Remove an entry from it, '
                             'or raise memory.user_char_limit, then retry.')
        if len(fitting) < len(facts):
            logger.warning('USER.md has room for %d of %d onboarding facts; the rest were not saved',
                           len(fitting), len(facts))
        content = '\n'.join([_HEADER, *fitting])
        if previous == [content]:  # same answers already saved, e.g. "Retry first build": nothing to write
            return {'saved': True, 'profile': 'default', 'target': 'user'}
        if previous:
            # Re-running onboarding replaces the earlier answers instead of adding a second, conflicting entry.
            operations = [{'action': 'remove', 'old_text': entry} for entry in previous]
            raw = memory_tool(target='user', operations=[*operations, {'action': 'add', 'content': content}],
                              store=store)
        else:
            raw = memory_tool(action='add', target='user', content=content, store=store)
        result = json.loads(raw)
        if not result.get('success') or result.get('staged'):
            raise ValueError(result.get('error') or result.get('message') or 'Memory was not saved')
        if content not in load_on_disk_store().user_entries:
            raise ValueError('Could not verify saved onboarding facts')
        return {'saved': True, 'profile': 'default', 'target': 'user'}
    finally:
        reset_hermes_home_override(token)
