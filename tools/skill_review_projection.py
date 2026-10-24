"""Read-only, complete per-file projections using the existing skill resolver/matcher."""
import hashlib
import json
from pathlib import Path


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def operations(payload):
    return payload['operations'] if payload.get('action') == 'batch' else [payload]


def source(path):
    if not path.exists():
        return {'hash': None, 'text': None, 'bytes': 0}
    data = path.read_bytes()
    try:
        text = data.decode('utf-8-sig')
    except UnicodeDecodeError:
        text = None
    return {'hash': hashlib.sha256(data).hexdigest(), 'text': text, 'bytes': len(data)}


def project(payload):
    """Freeze the final result of patch chains, never mutate a skill to preview it."""
    from tools import skill_manager_tool as sm
    from tools.skill_manager_batch import _validate_batch_ops
    from tools.registry import tool_error
    ops = operations(payload)
    if not isinstance(ops, list) or not ops:
        raise ValueError('A skill proposal must contain operations.')
    if payload.get('action') == 'batch' and not (len(ops) == 1 and ops[0].get('action') == 'delete'):
        _, error = _validate_batch_ops(ops, payload.get('name'), tool_error)
        if error:
            raise ValueError(json.loads(error)['error'])
    roots, files, packages, creates, create_categories = {}, {}, {}, set(), {}
    for index, op in enumerate(ops):
        name = op.get('name') or payload.get('name')
        action = op.get('action')
        error = sm._validate_name(name if action == 'create' else Path(name or '').name)
        if error or not name:
            raise ValueError(error or 'The skill name is missing.')
        if error := sm._op_shape_error(action, op):
            raise ValueError(error)
        if name not in roots:
            found = sm._find_skill(name)
            if action == 'create':
                if found:
                    raise ValueError(f"Skill '{name}' already exists.")
                if error := sm._validate_category(op.get('category')):
                    raise ValueError(error)
                root = sm._resolve_skill_dir(name, op.get('category'))
                if root.exists() and (root.is_symlink() or not root.is_dir() or any(root.iterdir())):
                    raise ValueError(f'Creation target already exists: {root}')
                creates.add(name)
                create_categories[name] = op.get('category')
            else:
                if not found:
                    raise ValueError(f"Skill '{name}' no longer exists.")
                root = Path(found['path'])
            roots[name] = str(root.absolute())
        root = Path(roots[name])
        if action == 'delete':
            if len(ops) != 1:
                raise ValueError('Whole-skill deletion must be the only operation.')
            _project_deleted_package(root, name, index, files, packages)
            continue
        _project_file(root, action, op, name, index, files)
    if not files:
        raise ValueError('This proposal contains no files to review.')
    return {'payload_hash': fingerprint(payload), 'roots': roots, 'creates': sorted(creates),
            'create_categories': create_categories, 'packages': packages, 'files': list(files.values()), 'approved': [], 'edits': {}}



def _project_deleted_package(root, name, index, files, packages):
    manifest = {}
    for path in sorted(root.rglob('*')):
        if path.is_symlink():
            raise ValueError(f'Cannot preview deletion through a symbolic link: {path}')
        if path.is_file():
            value = source(path)
            rel = path.relative_to(root).as_posix()
            manifest[rel] = value['hash']
            files[str(path)] = dict(name=name, root=str(root), relative_path=rel,
                path=str(path), resolved_path=str(path.resolve()), before=value['text'], after=None,
                base_hash=value['hash'], bytes=value['bytes'], binary=value['text'] is None,
                indices=[index], delete_package=True)
    packages[name] = manifest


def _project_file(root, action, op, name, index, files):
    from tools import skill_manager_tool as sm
    rewrite = action in {'create', 'edit'} or (action == 'patch' and op.get('content'))
    rel = 'SKILL.md' if rewrite else op.get('file_path') or 'SKILL.md'
    path, error = sm._resolve_supporting_file(root, rel)
    if error:
        raise ValueError(error['error'])
    key = str(path.absolute())
    if key not in files:
        value = source(path)
        if value['hash'] and value['text'] is None:
            raise ValueError(f'This skill operation targets a non-text file: {path}')
        files[key] = dict(name=name, root=str(root), relative_path=rel, path=key,
            resolved_path=str(path.resolve()), before=value['text'], after=value['text'],
            base_hash=value['hash'], bytes=value['bytes'], binary=False, indices=[])
    file = files[key]
    if rewrite:
        after = op['content']
    elif action == 'write_file':
        after = op['file_content']
    elif action == 'remove_file':
        if file['after'] is None:
            raise ValueError(f'File no longer exists: {path}')
        after = None
    elif action == 'patch':
        if file['after'] is None:
            raise ValueError(f'Patch target does not exist: {path}')
        from tools.write_approval import _fold_patch
        after, error = _fold_patch(file['after'], op.get('old_string'), op.get('new_string'), op.get('replace_all'))
        if error:
            raise ValueError(error)
    else:
        raise ValueError(f'Unsupported skill review action: {action}')
    validate_text(file, after, new_skill=action == 'create')
    file['after'] = after
    file['indices'].append(index)


def validate_text(file, text, *, new_skill=False):
    from tools import skill_manager_tool as sm
    if text is None:
        return
    if not isinstance(text, str):
        raise ValueError('Provide the full replacement text.')
    if file['relative_path'] == 'SKILL.md':
        if error := sm._validate_frontmatter(text, new_skill=new_skill):
            raise ValueError(error)
    if error := sm._validate_content_size(text, label=file['relative_path']):
        raise ValueError(error)
    if len(text.encode('utf-8')) > sm.MAX_SKILL_FILE_BYTES:
        raise ValueError('The replacement exceeds the skill file size limit.')


def check_sources(review):
    """Refuse changed files, resolver targets, creation collisions and deletion additions."""
    from tools import skill_manager_tool as sm
    for name, root in review['roots'].items():
        found = sm._find_skill(name)
        if name in review['creates']:
            path = Path(root)
            if str(sm._resolve_skill_dir(name, review.get('create_categories', {}).get(name)).absolute()) != root:
                raise ValueError(f'Creation destination changed: {root}. Recreate this proposal.')
            if found or (path.exists() and (not path.is_dir() or path.is_symlink() or any(path.iterdir()))):
                raise ValueError(f'Creation target changed: {root}')
        elif not found or str(Path(found['path']).absolute()) != root:
            raise ValueError(f"The location of skill '{name}' changed. Recreate this proposal.")
    for file in review['files']:
        path = Path(file['path'])
        if str(path.resolve()) != file['resolved_path'] or source(path)['hash'] != file['base_hash']:
            raise ValueError(f"The original file changed: {path}. Recreate this proposal.")
    for name, manifest in review['packages'].items():
        root = Path(review['roots'][name])
        paths = list(root.rglob('*'))
        if any(p.is_symlink() for p in paths):
            raise ValueError(f"The files in skill '{name}' changed.")
        current = {p.relative_to(root).as_posix(): source(p)['hash'] for p in paths if p.is_file()}
        if current != manifest:
            raise ValueError(f"The files in skill '{name}' changed. Recreate this proposal.")


def selected_payload(record):
    """Fold edits into the original operation ordering; preserve the atomic apply engine."""
    import copy
    review = record['skill_review']
    payload = copy.deepcopy(record['payload'])
    ops = operations(payload)
    replacements, skip = {}, set()
    for key, text in review['edits'].items():
        file = review['files'][int(key)]
        first, *later = file['indices']
        original = ops[first]
        name, rel = file['name'], file['relative_path']
        if original['action'] == 'create':
            replacement = {**original, 'content': text}
        elif rel == 'SKILL.md':
            replacement = {'action': 'patch', 'name': name, 'content': text}
        else:
            replacement = {'action': 'write_file', 'name': name, 'file_path': rel, 'file_content': text}
        replacements[first] = replacement
        skip.update(later)
    revised = [replacements.get(i, op) for i, op in enumerate(ops) if i not in skip]
    if payload.get('action') == 'batch':
        payload['operations'] = revised
        return payload
    return revised[0]
