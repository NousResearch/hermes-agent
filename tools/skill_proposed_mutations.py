"""Ordered, read-only skill mutations; native lookup and discovery retain their own semantics."""

from dataclasses import dataclass, field
from pathlib import Path

from agent import skill_utils as su


class ProposedSkillFS:
    """Overlay changed file bytes and lexical unlinks, never materialize trial writes."""

    def __init__(self):
        self.entries: dict[Path, str | None] = {}
        self.manifests: dict[Path, bool] = {}
        self.directories: set[Path] = set()
        self.removed_directories: set[Path] = set()

    @staticmethod
    def lexical(path):
        path = Path(path)
        return path.parent.resolve() / path.name

    def resolve(self, path):
        leaf = self.lexical(path)
        return leaf if leaf in self.entries else Path(path).resolve()

    def exists(self, path):
        real = self.resolve(path)
        if real in self.entries:
            return self.entries[real] is not None
        return real not in self.removed_directories and (real in self.directories or real.exists())

    def is_dir(self, path):
        real = self.resolve(path)
        return real not in self.entries and real not in self.removed_directories and (real in self.directories or real.is_dir())

    def lexists(self, path):
        path = self.lexical(path)
        if path in self.entries:
            return self.entries[path] is not None
        return path not in self.removed_directories and (path in self.directories or path.is_symlink() or path.exists())

    def is_symlink(self, path):
        path = self.lexical(path)
        return path not in self.entries and path.is_symlink()

    def children(self, directory):
        directory = self.resolve(directory)
        children = set(directory.iterdir()) if directory.is_dir() else set()
        children.update(p for p in self.directories | self.entries.keys() if p.parent == directory)
        return [p for p in children if self.lexists(p)]

    def read_text(self, path, **kwargs):
        real = self.resolve(path)
        if real not in self.entries:
            return real.read_text(encoding=kwargs.pop("encoding", "utf-8"), **kwargs)
        text = self.entries[real]
        if text is None:
            raise FileNotFoundError(str(path))
        return text[1:] if kwargs.get("encoding") == "utf-8-sig" and text.startswith("\ufeff") else text

    def write(self, path, text):
        real = self.resolve(path)
        self.entries[real] = text
        self.directories.update(real.parents)
        self.removed_directories.difference_update(real.parents)
        if real.name == "SKILL.md":
            self.manifests[real] = True
        return real

    def unlink(self, path, skill_dir):
        real = self.lexical(path)
        self.entries[real] = None
        if real.name == "SKILL.md":
            self.manifests[real] = False
        parent, stop = real.parent, self.resolve(skill_dir)
        while parent != stop and self.is_dir(parent) and not self.children(parent):
            self.directories.discard(parent)
            self.removed_directories.add(parent)
            parent = parent.parent
        return real

    def roots(self, *, project=False):
        return [p for _, p in su.get_skill_search_roots(include_project=project, is_dir=self.is_dir)]

    def iter_skill_dirs(self, root):
        # Management's rglob does not follow directory aliases; discovery DOES.
        from agent.skill_manifest_overlay import walk_proposed_manifests
        for node, dirs, files in walk_proposed_manifests(root, self.manifests, followlinks=False, is_dir=self.is_dir):
            if "SKILL.md" in files or "SKILL.md" in dirs:
                path = Path(node) / "SKILL.md"
                if not su.is_excluded_skill_path(path, exists=self.exists):
                    yield path.parent

    def find(self, name):
        from tools import skill_manager_tool as smt
        return smt._find_skill(name, roots=self.roots(), iter_dirs=self.iter_skill_dirs,
                               frontmatter_name=lambda p: smt._read_frontmatter_name(p, read_text=self.read_text),
                               exists=self.exists, resolve=self.resolve, lexists=self.lexists)

    def discoverable(self):
        return {p for root in self.roots(project=True)
                for p in su.iter_skill_index_files(root, "SKILL.md", manifest_changes=self.manifests, is_dir=self.is_dir)}


@dataclass
class MutationPlan:
    requires_approval: bool = False
    resources: set[Path] = field(default_factory=set)
    publish_targets: set[Path] = field(default_factory=set)
    replacement_targets: set[Path] = field(default_factory=set)
    clobber_error: str | None = None


def _step(op, fs):
    """Native validators and fuzzy matching; None means this step cannot write."""
    from tools import skill_manager_tool as smt
    action, name = op.get("action"), op.get("name")
    if not isinstance(name, str) or not name:
        return None
    if action == "create":
        text = op.get("content")
        if (smt._validate_name(name) or smt._validate_category(op.get("category"))
                or not isinstance(text, str) or smt._validate_content_size(text)
                or smt._validate_frontmatter(text)):
            return None
        directory = smt._resolve_skill_dir(name, op.get("category"))
        if fs.exists(directory / "SKILL.md"):
            return None
        return directory, directory / "SKILL.md", text
    skill = fs.find(name)
    if skill is None:
        return None
    directory = skill["path"]
    rewrite = action == "edit" or (action == "patch" and bool(op.get("content")))
    supporting = action in ("write_file", "remove_file") or (action == "patch" and op.get("file_path") and not rewrite)
    if supporting:
        target, error = smt._resolve_supporting_file(directory, op.get("file_path") or "", resolve=fs.resolve)
        if error:
            return None
    else:
        target = directory / "SKILL.md"
    if fs.is_dir(target) and (action != "remove_file" or not fs.is_symlink(target)):
        return None  # The native read/unlink/publish rejects a directory leaf.
    if action == "remove_file":
        return (directory, target, None) if fs.exists(target) else None
    if rewrite:
        text = op.get("content")
        if not isinstance(text, str) or smt._validate_frontmatter(text) or smt._validate_content_size(text):
            return None
    elif action == "write_file":
        text = op.get("file_content")
        if (not isinstance(text, str) or len(text.encode("utf-8")) > smt.MAX_SKILL_FILE_BYTES
                or smt._validate_content_size(text, label=op["file_path"])):
            return None
    elif action == "patch":
        from tools.fuzzy_match import fuzzy_find_and_replace
        if not op.get("old_string") or op.get("new_string") is None or not fs.exists(target):
            return None
        text, _, _, error = fuzzy_find_and_replace(fs.read_text(target, encoding="utf-8-sig"),
                                                   op["old_string"], op["new_string"], op.get("replace_all", False))
        if error or smt._validate_content_size(text) or (not supporting and smt._validate_frontmatter(text)):
            return None
    else:
        return None
    return directory, target, text


def _destructive_overlap(index, op, target, touched):
    destructive = op.get("action") in ("create", "edit", "write_file", "remove_file") or bool(op.get("content"))
    if destructive and target in touched:
        return (f"operations[{index}]: {op['action']} on '{target}' — an earlier op in this "
                "batch already touched that file, and this op would silently discard its work. "
                "One destructive op (write_file/remove_file/full rewrite) per file per batch; put "
                "it first, or fold the change in. Patch chains are fine.")
    touched.add(target)
    return None


def plan_mutations(operations, *, scan_creation=True, check_clobber=False):
    """Resolve each operation AFTER its predecessors; collect fences before any live write."""
    fs, plan = ProposedSkillFS(), MutationPlan()
    touched = set()
    # Protect missing configured roots too, without creating them during lock acquisition.
    plan.resources.update(p.resolve() for _, p in su.get_skill_search_roots(is_dir=lambda p: not p.exists() or p.is_dir()))
    baseline = fs.discoverable() if scan_creation else set()
    for index, op in enumerate(operations):
        if not isinstance(op, dict):
            break
        if op.get("action") == "create":
            plan.requires_approval = True
        if op.get("action") == "delete":
            skill = fs.find(op.get("name") or "")
            if skill:
                plan.resources.add(skill["path"].resolve())
            break  # A delete is sole and cannot activate a new skill within its removed subtree.
        step = _step(op, fs)
        if step is None:
            break  # The executor does not run later siblings after a rejected step.
        directory, target, text = step
        if check_clobber:
            plan.clobber_error = _destructive_overlap(index, op, fs.resolve(target), touched)
            if plan.clobber_error:
                break
        plan.resources.add(directory.resolve())
        if text is not None and any(fs.exists(parent) and not fs.is_dir(parent) for parent in fs.resolve(target).parents):
            break  # Native mkdir cannot pass through a file created by an earlier step.
        physical = fs.unlink(target, directory) if text is None else fs.write(target, text)
        plan.resources.add(physical)
        if text is not None:
            plan.publish_targets.add(physical)
            if op.get("action") in ("edit", "write_file") or op.get("content"):
                plan.replacement_targets.add(physical)
        if scan_creation and fs.discoverable() - baseline:
            plan.requires_approval = True
    return plan
