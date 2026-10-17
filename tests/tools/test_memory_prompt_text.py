"""Native formatting, full source recovery, and safe UTF-16 segmentation."""
from tools.memory_prompt_text import render_cards, split_for_platform, units, word_diff


def entity_text(message, entity):
    raw = message.text.encode('utf-16-le')
    return raw[entity['offset'] * 2:(entity['offset'] + entity['length']) * 2].decode('utf-16-le')


def test_short_card_full_text_and_every_changed_span(tmp_path):
    old = 'Backups: all profiles; health checks nightly.'
    new = 'Backups: retain 7 days; cap logs at 100 MB.'
    target_path = str(tmp_path / 'memories/USER.md')
    record = {'target_path': target_path, 'payload': {'action': 'replace', 'target': 'user', 'matched_entry': old, 'content': new}}
    cards = render_cards(record)
    assert len(cards) == 1
    card = cards[0]
    assert card.text.startswith(f'default wants to modify {target_path}\n\nCurrent\n')
    assert card.text.count('Backups:') == 2
    assert '\nBackups\n' not in card.text
    assert 'These actions affect this memory only.' not in card.text
    assert 'Nothing changes until you approve.' not in card.text
    assert old in card.text and new in card.text
    quotes = [entity_text(card, e) for e in card.entities if e['type'] == 'blockquote']
    assert quotes == [old, new]
    assert 'all profiles' in card.text
    removed, added = word_diff(old, new)
    assert all(old[a:b] in [entity_text(card, e) for e in card.entities if e['type'] == 'strikethrough'] for a,b,_ in removed)
    assert all(new[a:b] in [entity_text(card, e) for e in card.entities if e['type'] == 'bold'] for a,b,_ in added)


def test_rewording_is_one_readable_comparison_without_invented_excerpts():
    old = (
        'Use the canonical, documented, battle-tested answer for familiar tasks and deliver it verbatim. '
        'Do not reinvent, research again, or offer option menus when a known path exists. '
        'The documented skill already exists; stop thrashing and use the established answer.'
    )
    new = (
        'Use the canonical, documented answer for familiar tasks and deliver it verbatim. '
        'Do not reinvent or offer option menus when a known path exists. '
        'The documented skill already exists; use it.'
    )
    cards = render_cards({'payload': {'action': 'replace', 'target': 'user',
                                     'matched_entry': old, 'content': new}})
    assert len(cards) == 1
    card = cards[0]
    # Only complete source passages, once each. A rephrased policy must not acquire
    # a misleading third rendition assembled from unrelated deleted/added fragments.
    assert card.text.count(old) == card.text.count(new) == 1
    assert 'Removed:' not in card.text and 'Added:' not in card.text
    assert '…' not in card.text
    assert any(e['type'] == 'strikethrough' for e in card.entities)
    assert 'highlights' not in card.text and 'complete text' not in card.text
    assert card.text.startswith('default wants to modify USER.md\n\nCurrent\n')


def test_full_versions_that_fit_are_not_displaced_by_a_duplicate_summary():
    old = ('Fleet restart ordering: restart the hosting gateway last. '
           + 'Historical failure and implementation detail. ' * 42)
    new = ('Fleet restart ordering: restart the hosting gateway last. '
           'Reapply the local patch after updates. Verify every gateway version. ')
    cards = render_cards({'payload': {'action': 'replace', 'matched_entry': old, 'content': new}}, limit=2500)
    assert len(cards) == 1
    quotes = [entity_text(cards[0], e) for e in cards[0].entities
              if e['type'] in {'blockquote', 'expandable_blockquote'}]
    assert quotes == [old, new]


def test_long_unicode_commands_are_lossless_and_entities_valid():
    old = 'macOS TCC durability: launchers bind ~/.hermes/bin; 😀\n' * 160
    new = 'Fleet update + restart ordering: `hermes gateway restart`; <literal>&😀\n' * 160
    cards = render_cards({'payload': {'action': 'replace', 'matched_entry': old, 'content': new}}, limit=4096)
    assert len(cards) > 3
    source = []
    for card in cards:
        assert units(card.text) <= 4096
        for entity in card.entities:
            assert entity['offset'] + entity['length'] <= units(card.text)
            value = entity_text(card, entity) # no split surrogate pair
            if entity['type'] in {'blockquote', 'expandable_blockquote'}:
                source.append(value)
    assert ''.join(source) == old + new
    assert not any(e['type'] == 'expandable_blockquote' for c in cards for e in c.entities)
    assert 'Current · 1/' in cards[0].text
    assert 'directly above' in cards[-1].text


def test_add_and_delete_show_only_the_actual_entry():
    source = 'Keep the actual entry visible. ' * 25
    for action, field, heading in [('add', 'content', 'New memory'), ('remove', 'matched_entry', 'Delete')]:
        card, = render_cards({'payload': {'action': action, field: source}})
        assert card.text.startswith('default wants to modify MEMORY.md\n')
        assert card.text.count(source) == 1
        assert heading + '\n' in card.text
        assert 'complete text' not in card.text and 'Replacement' not in card.text
        assert [entity_text(card, e) for e in card.entities if e['type'] == 'blockquote'] == [source]
        assert not any(e['type'] in {'strikethrough', 'expandable_blockquote'} for e in card.entities)


def test_semantic_split_is_lossless_and_safe_for_giant_token():
    original = 'A paragraph. 😀\n\n' * 900 + '😀' * 4000
    pieces = split_for_platform([original], 4096)
    assert ''.join(pieces) == original
    assert all(units(piece) <= 4096 for piece in pieces)
    assert pieces[0].endswith('\n\n')
