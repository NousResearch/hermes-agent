"""Skills Hub GitHub adapter: API auth, tap providers, and the Contents/Trees source."""

import json
import logging
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from urllib.parse import quote

import httpx

from hermes_cli._subprocess_compat import windows_hide_flags
from tools.skills_guard import TRUSTED_REPOS
from tools.skills_hub_models import (
    SkillBundle, SkillMeta, SkillSource, _cache_metas, _cached_metas, _dedupe_by_trust,
    _hermes_tags, _matches_query, _parse_frontmatter, _referenced_support_paths,
    _validate_bundle_rel_path,
)

logger = logging.getLogger("tools.skills_hub")

# GitHub tap repo (owner/repo) -> provider label used by the docs-site catalog
# (website/scripts/extract-skills.py::GITHUB_TAP_LABELS). The runtime index collapses every tap into
# source="github"; ``extra.provider`` keeps per-tap identity searchable/filterable without disturbing
# dedup / floor / index-skip logic keyed on the bare source id.
GITHUB_TAP_PROVIDERS = {
    "openai/skills": "OpenAI", "anthropics/skills": "Anthropic", "huggingface/skills": "HuggingFace",
    "nvidia/skills": "NVIDIA", "voltagent/awesome-agent-skills": "VoltAgent", "garrytan/gstack": "gstack",
    "minimax-ai/cli": "MiniMax",
}

# Accepted ``--source`` provider filters (lowercased). Not real source ids —
# they narrow merged results to GitHub-tap skills carrying that ``extra.provider``.
_PROVIDER_FILTER_VALUES = frozenset(v.lower() for v in GITHUB_TAP_PROVIDERS.values())

_API = "https://api.github.com/repos"
_ACCEPT_JSON = "application/vnd.github.v3+json"


def github_provider_for(repo: str) -> Optional[str]:
    """Provider label for an ``owner/repo`` tap (case-insensitive), or None."""
    return GITHUB_TAP_PROVIDERS.get(repo.strip().lower()) if repo else None


def _filter_results_by_provider(results: List[SkillMeta], provider: str) -> List[SkillMeta]:
    """Keep only results whose ``extra.provider`` matches ``provider``. An explicit provider filter
    (``--source nvidia``) narrows to exactly that provider — the official catalog is NOT injected the
    way unfiltered browse does."""
    want = provider.strip().lower()
    return [r for r in results if str((r.extra or {}).get("provider", "")).lower() == want]


def _is_rate_limit_response(resp: httpx.Response) -> bool:
    """403 with exhausted quota, or any 429."""
    return resp.status_code == 429 or (
        resp.status_code == 403 and resp.headers.get("X-RateLimit-Remaining", "") == "0"
    )


class GitHubAuth:
    """GitHub API authentication, tried in priority order: GITHUB_TOKEN / GH_TOKEN (PAT), `gh auth token`
    (gh CLI), GitHub App JWT + installation token, then unauthenticated (60 req/hr, public repos only)."""

    def __init__(self):
        self._cached_token: Optional[str] = None
        self._cached_method: Optional[str] = None
        self._app_token_expiry: float = 0

    def get_headers(self) -> Dict[str, str]:
        token = self._resolve_token()
        return {"Accept": _ACCEPT_JSON, **({"Authorization": f"token {token}"} if token else {})}

    def is_authenticated(self) -> bool:
        return self._resolve_token() is not None

    def auth_method(self) -> str:
        """'pat', 'gh-cli', 'github-app', or 'anonymous'."""
        self._resolve_token()
        return self._cached_method or "anonymous"

    def _resolve_token(self) -> Optional[str]:
        if self._cached_token and (self._cached_method != "github-app" or time.time() < self._app_token_expiry):
            return self._cached_token
        for method, resolve in (
            ("pat", self._try_pat), ("gh-cli", self._try_gh_cli), ("github-app", self._try_github_app),
        ):
            token = resolve()
            if token:
                self._cached_token, self._cached_method = token, method
                if method == "github-app":
                    self._app_token_expiry = time.time() + 3500  # ~58 min (tokens last 1 hour)
                return token
        self._cached_method = "anonymous"
        return None

    @staticmethod
    def _try_pat() -> Optional[str]:
        # Profile-scoped secret lookup (multiplexed gateway safe).
        from agent.secret_scope import get_secret
        return get_secret("GITHUB_TOKEN") or get_secret("GH_TOKEN")

    def _try_gh_cli(self) -> Optional[str]:
        try:
            result = subprocess.run(
                ["gh", "auth", "token"], capture_output=True, text=True, encoding='utf-8', errors='replace',
                timeout=5, stdin=subprocess.DEVNULL, creationflags=windows_hide_flags(),
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except (FileNotFoundError, subprocess.TimeoutExpired) as e:
            logger.debug("gh CLI token lookup failed: %s", e)
        return None

    def _try_github_app(self) -> Optional[str]:
        from agent.secret_scope import get_secret
        app_id, key_path = get_secret("GITHUB_APP_ID"), get_secret("GITHUB_APP_PRIVATE_KEY_PATH")
        installation_id = get_secret("GITHUB_APP_INSTALLATION_ID")
        if not all([app_id, key_path, installation_id]):
            return None
        try:
            import jwt  # PyJWT
        except ImportError:
            logger.debug("PyJWT not installed, skipping GitHub App auth")
            return None
        try:
            key_file = Path(key_path)
            if not key_file.exists():
                return None
            now = int(time.time())
            encoded_jwt = jwt.encode(
                {"iat": now - 60, "exp": now + (10 * 60), "iss": app_id},
                key_file.read_text(encoding="utf-8"), algorithm="RS256",
            )
            resp = httpx.post(
                f"https://api.github.com/app/installations/{installation_id}/access_tokens",
                headers={"Authorization": f"Bearer {encoded_jwt}", "Accept": _ACCEPT_JSON}, timeout=10,
            )
            if resp.status_code == 201:
                return resp.json().get("token")
        except Exception as e:
            logger.debug("GitHub App auth failed: %s", e)
        return None


def _split_repo_id(identifier: str) -> Optional[Tuple[str, str]]:
    """``owner/repo/path/to/skill`` -> ``(owner/repo, path/to/skill)``; None when too short."""
    parts = identifier.split("/", 2)
    return (f"{parts[0]}/{parts[1]}", parts[2]) if len(parts) >= 3 else None


def _skip_bundle_file(rel_path: str) -> bool:
    """Dotfiles, bytecode and __pycache__ never ship in a bundle."""
    base = rel_path.rsplit("/", 1)[-1]
    return base.startswith(".") or base.endswith(".pyc") or "__pycache__" in rel_path.split("/")


def _tree_members(entries: List[dict], prefix: str):
    """``(rel_path, item_path, is_regular_blob)`` for every git-tree entry under ``prefix``. Symlinks
    (mode 120000) and non-blobs report ``is_regular_blob=False`` so callers can reject a SKILL.md-linked
    symlink instead of silently following it."""
    for item in entries:
        item_path = item.get("path", "")
        if item_path.startswith(prefix):
            yield item_path[len(prefix):], item_path, item.get("type") == "blob" and item.get("mode") != "120000"


class GitHubSource(SkillSource):
    """Fetch skills from GitHub repos via the Contents API."""

    DEFAULT_TAPS = [
        # openai/skills keeps content under skills/.curated/ + skills/.system/; _list_skills_in_repo
        # skips "."/"_" directories, so both entries point at the inner paths.
        {"repo": "openai/skills", "path": "skills/.curated/"},
        {"repo": "openai/skills", "path": "skills/.system/"},
        {"repo": "anthropics/skills", "path": "skills/"},
        {"repo": "huggingface/skills", "path": "skills/"},
        # NVIDIA-verified skills (CUDA-X, NeMo, cuOpt, ...), each with a signed skill.oms.sig
        # + governance card; `trusted` via tools/skills_guard.py::TRUSTED_REPOS.
        {"repo": "NVIDIA/skills", "path": "skills/"},
        {"repo": "garrytan/gstack", "path": ""},
    ]

    SOURCE_ID = "github"
    _parse_frontmatter_quick = staticmethod(_parse_frontmatter)

    def __init__(self, auth: GitHubAuth, extra_taps: Optional[List[Dict]] = None):
        self.auth = auth
        self.taps = list(self.DEFAULT_TAPS) + list(extra_taps or [])
        # Per-instance repo -> (default_branch, tree_entries); lives for one
        # search/install flow so repeated tree lookups cost no API calls.
        self._tree_cache: Dict[str, Tuple[str, List[dict]]] = {}
        self._tree_revisions: Dict[str, str] = {}
        # repo -> skills.sh.json grouping map; None = fetched, no sidecar.
        self._skillsh_groupings: Dict[str, Optional[Dict[str, str]]] = {}
        self._rate_limited: bool = False

    @property
    def is_rate_limited(self) -> bool:  # whether the GitHub API rate limit was hit during operations
        return self._rate_limited

    def trust_level_for(self, identifier: str) -> str:
        # identifier format: "owner/repo/path/to/skill"
        parts = identifier.split("/", 2)
        return "trusted" if len(parts) >= 2 and f"{parts[0]}/{parts[1]}" in TRUSTED_REPOS else "community"

    def search(self, query: str, limit: int = 10) -> List[SkillMeta]:
        """Substring-match all taps; dedupe by identifier preferring higher trust."""
        results: List[SkillMeta] = []
        query_lower = query.lower()
        for tap in self.taps:
            try:
                for skill in self._list_skills_in_repo(tap["repo"], tap.get("path", "")):
                    if _matches_query(query_lower, skill.name, skill.description, skill.tags):
                        results.append(skill)
            except Exception as e:
                logger.debug("Failed to search %s: %s", tap['repo'], e)
        return _dedupe_by_trust(results)[:limit]

    def fetch(self, identifier: str) -> Optional[SkillBundle]:
        """Download a skill; identifier format: "owner/repo/path/to/skill-dir"."""
        if (split := _split_repo_id(identifier)) is None:
            return None
        repo, skill_path = split
        skill_dir = skill_path.rstrip("/")
        # Resolve the tree FIRST so every byte fetch — SKILL.md included — is pinned to the
        # same revision; an unpinned /contents fetch floats to HEAD and can serve bytes newer
        # than the tree the paths were validated against (TOCTOU). Idempotent + cached.
        tree = self._get_repo_tree(repo)
        pinned_ref = self._tree_revisions.get(repo)
        skill_md = self._fetch_file_content(repo, f"{skill_dir}/SKILL.md", ref=pinned_ref)
        if skill_md is None:
            return None
        referenced = _referenced_support_paths(skill_md)
        if referenced is None:
            return None
        files: Dict[str, Union[str, bytes]] = {"SKILL.md": skill_md}
        if tree is not None:
            if not self._collect_tree_files(repo, skill_dir, tree[1], pinned_ref, referenced, files):
                return None
            revision = pinned_ref or tree[0]
        else:
            for rel_path in referenced:
                self._add_support_file(repo, f"{skill_dir}/{rel_path}", rel_path, files, rel_path)
            revision = ""
        url = f"https://github.com/{repo}/" + (f"tree/{revision}/{skill_path}" if revision else skill_path)
        return SkillBundle(
            name=skill_dir.split("/")[-1], files=files, source="github", identifier=identifier,
            trust_level=self.trust_level_for(identifier), metadata={"source_url": url, "source_revision": revision},
        )

    def _add_support_file(self, repo: str, item_path: str, rel_path: str, files: dict, shown: str, **kw) -> None:
        """Fetch one support file into ``files``; a failed fetch warns (naming ``shown``) and is skipped."""
        content = self._fetch_file_bytes(repo, item_path, **kw)
        if content is None:
            logger.warning("Failed to fetch referenced skill support file; continuing without it: %s", shown)
        else:
            files[rel_path] = content

    def _collect_tree_files(
        self, repo: str, skill_path: str, entries: List[dict], ref: Optional[str], referenced: set,
        files: Dict[str, Union[str, bytes]],
    ) -> bool:
        """Download the FULL skill directory from the pinned tree into ``files``. Link-driven fetching
        silently dropped support files under non-canonical dirs (``reference/``, ``agents/``, root
        LICENSE); everything still goes through quarantine + scan, and the scanner sees MORE this way.
        Returns False (bundle rejected) on an unsafe path or a SKILL.md-linked path that exists in the
        tree as a symlink/non-blob — that shape is an escape attempt. A linked path that is simply absent
        is a dangling link (repo-only dev tool, prose over-match): warn and install without it."""
        prefix = f"{skill_path}/"
        symlinked: set = set()
        for rel_path, item_path, regular in _tree_members(entries, prefix):
            if not regular:
                symlinked.add(rel_path)
                continue
            if rel_path == "SKILL.md" or _skip_bundle_file(rel_path):
                continue
            try:
                rel_path = _validate_bundle_rel_path(rel_path)
            except ValueError:
                logger.warning("Rejected unsafe file path in skill bundle: %s", item_path)
                return False
            self._add_support_file(repo, item_path, rel_path, files, item_path, ref=ref)
        for rel_path in sorted(referenced):
            # A SKILL.md-linked support path that isn't in the tree is a dangling link — a repo-only dev
            # tool, prose over-match, or a file the author forgot to push. Warn and install without it
            # rather than aborting the whole install (#66760/#90081): the skill body still works, and the
            # gap is visible in the log. A referenced path that IS in the tree but as a symlink (or any
            # non-regular entry) stays a hard rejection — that shape is an escape attempt, not a forgotten
            # file.
            if rel_path in symlinked:
                logger.warning("Rejected non-regular referenced file in skill bundle: %s%s", prefix, rel_path)
                return False
            if rel_path not in files:
                logger.warning(
                    "Referenced skill support file is missing; continuing without it: %s%s", prefix, rel_path)
        return True

    def inspect(self, identifier: str) -> Optional[SkillMeta]:
        """Fetch just the SKILL.md metadata for preview."""
        if (split := _split_repo_id(identifier)) is None:
            return None
        repo, skill_path = split[0], split[1].rstrip("/")
        content = self._fetch_file_content(repo, f"{skill_path}/SKILL.md")
        if not content:
            return None
        fm = _parse_frontmatter(content)
        tags = _hermes_tags(fm) or (fm["tags"] if isinstance(fm.get("tags"), list) else [])
        provider = github_provider_for(repo)
        return SkillMeta(
            name=fm.get("name", skill_path.split("/")[-1]), description=str(fm.get("description", "")),
            source="github", identifier=identifier, trust_level=self.trust_level_for(identifier),
            repo=repo, path=skill_path, tags=[str(t) for t in tags],
            extra={"provider": provider} if provider else {},
        )

    # -- Internal helpers --

    def _list_skills_in_repo(self, repo: str, path: str) -> List[SkillMeta]:
        """List skill directories in a GitHub repo path, using cached index."""
        cache_key = f"{repo}_{path}".replace("/", "_").replace(" ", "_")
        cached = _cached_metas(cache_key)
        if cached is not None:
            return cached
        resp = self._github_get(f"{_API}/{repo}/contents/{path.rstrip('/')}")
        if resp is None or resp.status_code != 200:
            return []
        entries = resp.json()
        if not isinstance(entries, list):
            return []
        skills: List[SkillMeta] = []
        groupings = self._get_skillsh_groupings(repo)
        prefix = path.rstrip("/")
        for entry in entries:
            if entry.get("type") != "dir" or entry["name"].startswith((".", "_")):
                continue
            dir_name = entry["name"]
            meta = self.inspect(f"{repo}/{prefix}/{dir_name}" if prefix else f"{repo}/{dir_name}")
            if meta:
                category = groupings and (groupings.get(meta.name) or groupings.get(dir_name))
                if category:
                    meta.extra["category"] = category
                skills.append(meta)
        _cache_metas(cache_key, skills)
        return skills

    def _get_repo_tree(self, repo: str) -> Optional[Tuple[str, List[dict]]]:
        """Cached ``(default_branch, tree_entries)`` for a repo, or None. One install may need the tree
        several times; caching saves the ``GET /repos/{repo}`` + ``GET .../git/trees/{branch}`` pair each
        time (~12 of the 60/hr unauthenticated budget before)."""
        if repo in self._tree_cache:
            return self._tree_cache[repo]
        repo_data = self._github_json(f"{_API}/{repo}")
        if repo_data is None:
            return None
        default_branch = repo_data.get("default_branch", "main")
        tree_data = self._github_json(
            f"{_API}/{repo}/git/trees/{default_branch}", params={"recursive": "1"}, timeout=30.0,
        )
        if tree_data is None:
            return None
        if tree_data.get("truncated"):
            logger.debug("Git tree truncated for %s, cannot cache", repo)
            return None
        if isinstance(tree_data.get("sha"), str) and tree_data["sha"]:
            self._tree_revisions[repo] = tree_data["sha"]
        self._tree_cache[repo] = tree = (default_branch, tree_data.get("tree", []))
        return tree

    def _github_json(self, url: str, **kwargs) -> Optional[dict]:
        """Decoded JSON body of a 200 ``_github_get`` (which flags rate-limit exhaustion), else None."""
        resp = self._github_get(url, **kwargs)
        try:
            return resp.json() if resp is not None and resp.status_code == 200 else None
        except ValueError:
            return None

    def _github_get(
        self, url: str, *, params: Optional[Dict] = None, headers: Optional[Dict] = None,
        timeout: float = 15.0, max_retries: int = 3,
    ) -> Optional[httpx.Response]:
        """GET against the GitHub API with retry/backoff on transient failures. Returns the final
        response (caller inspects status) or None when every attempt raised a transport error.
        Retries rate-limit 403/429 (waiting until ``Retry-After`` / ``X-RateLimit-Reset`` when present,
        capped 60s — one shared limit zeroes every GitHub tap at once during an index build), 5xx, and
        transport errors with exponential backoff. Terminal rate-limit exhaustion flags the instance so
        an index build fails loud instead of silently shipping zero GitHub skills."""
        hdrs = headers if headers is not None else self.auth.get_headers()
        backoff = 1.0
        last_resp: Optional[httpx.Response] = None
        for attempt in range(max_retries):
            last_attempt = attempt >= max_retries - 1
            wait = backoff
            try:
                resp = httpx.get(url, params=params, headers=hdrs, timeout=timeout, follow_redirects=True)
            except httpx.HTTPError as e:
                logger.debug("GitHub GET %s failed (attempt %d/%d): %s", url, attempt + 1, max_retries, e)
                if last_attempt:
                    return None
            else:
                last_resp = resp
                if resp.status_code == 200:
                    return resp
                if resp.status_code in (403, 429):
                    limited = _is_rate_limit_response(resp)
                    if not limited or last_attempt:
                        if limited:  # terminal exhaustion: flag the instance so callers fail loud
                            self._rate_limited = True
                            logger.warning("GitHub API rate limit exhausted (unauthenticated: 60 req/hr). "
                                           "Set GITHUB_TOKEN or install the gh CLI to raise the limit to 5,000/hr.")
                        return resp
                    reset = resp.headers.get("X-RateLimit-Reset", "")
                    retry_after = resp.headers.get("Retry-After", "")
                    if retry_after.isdigit():
                        wait = min(float(retry_after), 60.0)
                    elif reset.isdigit():
                        delta = float(reset) - time.time()
                        if 0 < delta <= 60.0:
                            wait = delta
                    logger.debug("GitHub rate limited on %s, waiting %.1fs (attempt %d/%d)",
                                 url, wait, attempt + 1, max_retries)
                elif not (500 <= resp.status_code < 600) or last_attempt:
                    return resp
            time.sleep(wait)
            backoff = min(backoff * 2, 30.0)
        return last_resp

    def _find_skill_in_repo_tree(self, repo: str, skill_name: str) -> Optional[str]:
        """Locate ``<skill_name>/SKILL.md`` anywhere in the repo tree (one API call); full identifier or None."""
        if (cached := self._get_repo_tree(repo)) is None:
            return None
        skill_md_suffix = f"/{skill_name}/SKILL.md"
        for entry in cached[1]:
            path = entry.get("path", "")
            if entry.get("type") == "blob" and (path.endswith(skill_md_suffix) or path == skill_md_suffix[1:]):
                return f"{repo}/{path[: -len('/SKILL.md')]}"
        return None

    def _fetch_file_content(self, repo: str, path: str, ref: Optional[str] = None) -> Optional[str]:
        """Fetch a single text file from GitHub (None on miss or non-UTF-8)."""
        content = self._fetch_file_bytes(repo, path, ref=ref)
        try:
            return None if content is None else content.decode("utf-8")
        except UnicodeDecodeError:
            return None

    def _fetch_file_bytes(self, repo: str, path: str, ref: Optional[str] = None) -> Optional[bytes]:
        """Fetch exact file bytes. ``ref`` pins to a tree SHA (see ``fetch`` on
        the TOCTOU); None keeps the legacy unpinned behavior."""
        resp = self._github_get(
            f"{_API}/{repo}/contents/{quote(path, safe='/')}", params={"ref": ref} if ref else None,
            headers={**self.auth.get_headers(), "Accept": "application/vnd.github.v3.raw"},
        )
        return resp.content if resp is not None and resp.status_code == 200 else None

    def _get_skillsh_groupings(self, repo: str) -> Optional[Dict[str, str]]:
        """Repo-root ``skills.sh.json`` groupings flattened to ``{skill_name: title}``. ``skills.sh.json``
        is a cross-ecosystem standard (``$schema: https://skills.sh/schemas/skills.sh.schema.json``); any
        tap shipping it gets category pills for free. None when absent/unparsable; cached per repo."""
        if repo not in self._skillsh_groupings:
            content = self._fetch_file_content(repo, "skills.sh.json")
            self._skillsh_groupings[repo] = self._parse_skillsh_groupings(content) if content else None
        return self._skillsh_groupings[repo]

    @staticmethod
    def _parse_skillsh_groupings(content: str) -> Optional[Dict[str, str]]:
        """Flatten ``{"groupings": [{"title", "skills": [...]}]}``; None if not usable."""
        try:
            data = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            return None
        groupings = data.get("groupings") if isinstance(data, dict) else None
        if not isinstance(groupings, list):
            return None
        mapping: Dict[str, str] = {}
        for group in groupings:
            if not isinstance(group, dict):
                continue
            title, members = group.get("title"), group.get("skills")
            if not isinstance(title, str) or not isinstance(members, list):
                continue
            for member in members:
                if isinstance(member, str) and member:
                    mapping.setdefault(member, title)  # first grouping wins
        return mapping
