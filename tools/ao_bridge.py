"""Small Python wrapper around the local Agent Orchestrator Node bridge."""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any, Dict, Optional

from gateway.status import _pid_exists, terminate_pid


DEFAULT_AO_CONFIG_PATH = "/Users/felipelamartine/projects/Oryn/agent-orchestrator.yaml"
DEFAULT_AO_HOME = "/Users/felipelamartine"
DEFAULT_CODEX_BIN = "/opt/homebrew/bin/codex"
DEFAULT_AGENT = "codex"
TERMINAL_STATUSES = {"done", "merged", "killed", "errored", "terminated"}
FAILED_STATUSES = {"killed", "errored", "terminated"}


@dataclass
class AOSession:
    id: str
    project_id: Optional[str] = None
    status: Optional[str] = None
    activity: Optional[str] = None
    branch: Optional[str] = None
    issue_id: Optional[str] = None
    workspace_path: Optional[str] = None
    tmux_name: Optional[str] = None
    agent: Optional[str] = None
    model: Optional[str] = None
    reasoning_effort: Optional[str] = None
    pr: Any = None
    summary: Optional[str] = None
    open_command: Optional[str] = None

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "AOSession":
        return cls(
            id=str(payload.get("id") or ""),
            project_id=payload.get("project_id"),
            status=payload.get("status"),
            activity=payload.get("activity"),
            branch=payload.get("branch"),
            issue_id=payload.get("issue_id"),
            workspace_path=payload.get("workspace_path"),
            tmux_name=payload.get("tmux_name"),
            agent=payload.get("agent"),
            model=payload.get("model"),
            reasoning_effort=payload.get("reasoning_effort") or payload.get("reasoningEffort"),
            pr=payload.get("pr"),
            summary=payload.get("summary"),
            open_command=payload.get("open_command"),
        )

    @property
    def is_terminal(self) -> bool:
        return (self.status or "").lower() in TERMINAL_STATUSES

    @property
    def display_status(self) -> str:
        status = (self.status or "").lower()
        if status in FAILED_STATUSES:
            return "failed"
        if status in {"done", "merged"}:
            return "completed"
        return "running"

    def event_fields(self) -> Dict[str, Any]:
        return {
            "runtime": "ao",
            "runtime_session_id": self.id,
            "runtime_project_id": self.project_id,
            "ao_session_id": self.id,
            "ao_project_id": self.project_id,
            "workspace_path": self.workspace_path,
            "branch": self.branch,
            "issue_id": self.issue_id,
            "tmux_name": self.tmux_name,
            "open_command": self.open_command,
            "agent": self.agent,
            "model": self.model,
            "reasoning_effort": self.reasoning_effort,
            "status": self.display_status,
        }


class AOBridgeError(RuntimeError):
    pass


class AOBridge:
    def __init__(
        self,
        config_path: str = DEFAULT_AO_CONFIG_PATH,
        home: str = DEFAULT_AO_HOME,
        node_bin: str = "node",
        bridge_script: Optional[Path] = None,
        codex_shim_dir: Optional[Path] = None,
        codex_real_bin: Optional[str] = None,
    ):
        self.config_path = config_path
        self.home = home
        self.node_bin = node_bin
        self.bridge_script = bridge_script or Path(__file__).with_name("ao_bridge.mjs")
        self.codex_shim_dir = codex_shim_dir or Path(__file__).with_name("ao_shims")
        self.codex_shim_path = self.codex_shim_dir / "codex"
        self.user_bin_dir = Path(self.home) / "bin"
        self.codex_real_bin = self._resolve_codex_real_bin(codex_real_bin)

    def spawn(
        self,
        *,
        project_id: str,
        prompt: str,
        issue_id: Optional[str] = None,
        branch: Optional[str] = None,
        agent: Optional[str] = None,
        model: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        minimal_worker_prompt: bool = False,
    ) -> AOSession:
        defaults = self._project_agent_defaults.get(project_id) or {}
        resolved_agent = agent or defaults.get("agent") or DEFAULT_AGENT
        resolved_model = model or defaults.get("model")
        resolved_reasoning_effort = reasoning_effort or defaults.get("reasoning_effort")
        payload = self._call(
            "spawn",
            {
                "project_id": project_id,
                "prompt": prompt,
                "issue_id": issue_id,
                "branch": branch,
                "agent": resolved_agent,
                "minimal_worker_prompt": bool(minimal_worker_prompt),
            },
            timeout=180,
        )
        session = self._with_project_defaults(AOSession.from_payload(payload["session"]))
        session.agent = agent or session.agent or resolved_agent
        session.model = model or session.model or resolved_model
        session.reasoning_effort = reasoning_effort or session.reasoning_effort or resolved_reasoning_effort
        return session

    def status(self, session_id: str) -> Optional[AOSession]:
        payload = self._call("status", {"session_id": session_id}, timeout=30)
        session = payload.get("session")
        return self._with_project_defaults(AOSession.from_payload(session)) if session else None

    def list(self, project_id: Optional[str] = None) -> list[AOSession]:
        payload = self._call("list", {"project_id": project_id}, timeout=60)
        sessions = [self._with_project_defaults(AOSession.from_payload(item)) for item in payload.get("sessions") or []]
        seen = {session.id for session in sessions}
        for session in self._archived_sessions(project_id=project_id):
            if session.id not in seen:
                sessions.append(self._with_project_defaults(session))
                seen.add(session.id)
        return sessions

    def send(self, session_id: str, message: str) -> Optional[AOSession]:
        payload = self._call("send", {"session_id": session_id, "message": message}, timeout=60)
        session = payload.get("session")
        return self._with_project_defaults(AOSession.from_payload(session)) if session else None

    def run_codex_exec_benchmark(
        self,
        *,
        project_id: str,
        prompt: str,
        model: Optional[str] = None,
        timeout_seconds: int = 180,
    ) -> Dict[str, Any]:
        """Run a benchmark prompt through Codex exec without AO/tmux interaction."""

        defaults = self._project_agent_defaults.get(project_id) or {}
        resolved_model = model or defaults.get("model")
        project_path = self._project_paths.get(project_id) or os.getcwd()
        session_id = f"codex-exec-{uuid.uuid4().hex[:10]}"
        with tempfile.NamedTemporaryFile(prefix=f"{session_id}-", suffix=".txt", delete=False) as handle:
            output_path = Path(handle.name)
        args = [
            self.codex_real_bin,
            "exec",
            "--sandbox",
            "read-only",
            "--skip-git-repo-check",
            "--cd",
            str(project_path),
            "--output-last-message",
            str(output_path),
        ]
        if resolved_model:
            args.extend(["--model", str(resolved_model)])
        args.append("-")
        started_at = time.time()
        try:
            proc = subprocess.run(
                args,
                input=prompt,
                text=True,
                capture_output=True,
                timeout=max(1, int(timeout_seconds or 180)),
                env=self._bridge_env(),
                check=False,
            )
            timed_out = False
        except subprocess.TimeoutExpired as exc:
            proc = subprocess.CompletedProcess(args, returncode=124, stdout=exc.stdout or "", stderr=exc.stderr or "")
            timed_out = True
        duration = time.time() - started_at
        try:
            final_message = output_path.read_text(encoding="utf-8", errors="replace").strip()
        except Exception:
            final_message = ""
        try:
            output_path.unlink(missing_ok=True)
        except Exception:
            pass
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        combined = "\n".join(part for part in (stdout.strip(), stderr.strip()) if part)
        return {
            "session_id": session_id,
            "project_id": project_id,
            "status": "timeout" if timed_out else "completed" if proc.returncode == 0 else "failed",
            "returncode": proc.returncode,
            "summary": final_message or stdout.strip(),
            "output_tail": combined[-12000:],
            "duration_seconds": duration,
            "workspace_path": str(project_path),
            "agent": "codex",
            "model": resolved_model,
            "reasoning_effort": defaults.get("reasoning_effort"),
            "token_total": self._parse_codex_token_total(combined),
        }

    def kill(self, session_id: str, *, session: Optional[AOSession] = None, force: bool = True) -> None:
        self._call("kill", {"session_id": session_id}, timeout=60)
        if force:
            self.force_cleanup(session_id, session=session)

    def capture_output(self, session: AOSession, lines: int = 40) -> str:
        if not session.tmux_name:
            return ""
        try:
            proc = subprocess.run(
                ["tmux", "capture-pane", "-p", "-t", session.tmux_name, "-S", f"-{int(lines)}"],
                text=True,
                capture_output=True,
                timeout=10,
                check=False,
            )
        except Exception:
            return ""
        if proc.returncode != 0:
            return ""
        return proc.stdout.strip()

    def force_cleanup(self, session_id: str, *, session: Optional[AOSession] = None) -> Dict[str, Any]:
        """Best-effort cleanup for AO workers that survive the AO kill call."""
        tmux_name = (session.tmux_name if session else None) or ""
        killed_tmux = False
        killed_process_patterns: list[str] = []

        if tmux_name:
            try:
                proc = subprocess.run(
                    ["tmux", "kill-session", "-t", tmux_name],
                    text=True,
                    capture_output=True,
                    timeout=10,
                    check=False,
                )
                killed_tmux = proc.returncode == 0
            except Exception:
                killed_tmux = False

        for pattern in (session_id, tmux_name):
            pattern = (pattern or "").strip()
            if not pattern:
                continue
            if self._terminate_processes_matching(pattern):
                killed_process_patterns.append(pattern)

        return {
            "killed_tmux": killed_tmux,
            "killed_process_patterns": killed_process_patterns,
        }

    def runtime_health(self, session: AOSession) -> Dict[str, Any]:
        """Return whether AO's stored session state still has a live runtime."""
        tmux_alive = self._tmux_session_exists(session.tmux_name)
        process_alive = bool(
            self._matching_pids(session.id)
            or self._matching_pids(session.tmux_name or "")
        )
        is_stale = (
            not session.is_terminal
            and bool(session.tmux_name)
            and tmux_alive is False
            and not process_alive
        )
        return {
            "runtime_health": "stale" if is_stale else "ok",
            "runtime_warning": "AO reports this worker as running, but its tmux/process runtime is gone." if is_stale else None,
            "tmux_alive": tmux_alive,
            "process_alive": process_alive,
        }

    def runtime_health_many(self, sessions: list[AOSession]) -> Dict[str, Dict[str, Any]]:
        """Batch runtime health checks for AO session read models."""

        alive_tmux_names = self._tmux_session_names()
        process_patterns = {
            value
            for session in sessions
            for value in (session.id, session.tmux_name or "")
            if value
        }
        alive_process_patterns = self._matching_patterns(process_patterns)

        result: Dict[str, Dict[str, Any]] = {}
        for session in sessions:
            if session.tmux_name:
                tmux_alive = session.tmux_name in alive_tmux_names if alive_tmux_names is not None else None
            else:
                tmux_alive = None
            process_alive = bool(
                session.id in alive_process_patterns
                or bool(session.tmux_name and session.tmux_name in alive_process_patterns)
            )
            is_stale = (
                not session.is_terminal
                and bool(session.tmux_name)
                and tmux_alive is False
                and not process_alive
            )
            result[session.id] = {
                "runtime_health": "stale" if is_stale else "ok",
                "runtime_warning": "AO reports this worker as running, but its tmux/process runtime is gone." if is_stale else None,
                "tmux_alive": tmux_alive,
                "process_alive": process_alive,
            }
        return result

    @staticmethod
    def _tmux_session_exists(tmux_name: Optional[str]) -> Optional[bool]:
        if not tmux_name:
            return None
        try:
            proc = subprocess.run(
                ["tmux", "has-session", "-t", tmux_name],
                text=True,
                capture_output=True,
                timeout=5,
                check=False,
            )
        except Exception:
            return None
        return proc.returncode == 0

    @staticmethod
    def _matching_pids(pattern: str) -> list[int]:
        pattern = (pattern or "").strip()
        if not pattern:
            return []
        try:
            proc = subprocess.run(
                ["pgrep", "-f", pattern],
                text=True,
                capture_output=True,
                timeout=5,
                check=False,
            )
        except Exception:
            return []
        if proc.returncode != 0:
            return []

        current_pid = os.getpid()
        pids: list[int] = []
        for raw in proc.stdout.splitlines():
            try:
                pid = int(raw.strip())
            except ValueError:
                continue
            if pid != current_pid:
                pids.append(pid)
        return pids

    @staticmethod
    def _terminate_processes_matching(pattern: str) -> bool:
        matched = False
        pids = AOBridge._matching_pids(pattern)

        for pid in pids:
            try:
                os.kill(pid, signal.SIGTERM)
                matched = True
            except ProcessLookupError:
                pass
            except Exception:
                pass

        if matched:
            time.sleep(0.25)

        for pid in pids:
            if not _pid_exists(pid):
                continue
            try:
                terminate_pid(pid, force=True)
            except ProcessLookupError:
                pass
            except Exception:
                pass

        return matched

    @staticmethod
    def _tmux_session_names() -> Optional[set[str]]:
        try:
            proc = subprocess.run(
                ["tmux", "list-sessions", "-F", "#{session_name}"],
                text=True,
                capture_output=True,
                timeout=5,
                check=False,
            )
        except Exception:
            return None
        if proc.returncode != 0:
            return set()
        return {line.strip() for line in proc.stdout.splitlines() if line.strip()}

    @staticmethod
    def _matching_patterns(patterns: set[str]) -> set[str]:
        if not patterns:
            return set()
        try:
            proc = subprocess.run(
                ["ps", "-axo", "pid=,command="],
                text=True,
                capture_output=True,
                timeout=5,
                check=False,
            )
        except Exception:
            return set()
        if proc.returncode != 0:
            return set()

        current_pid = os.getpid()
        matched: set[str] = set()
        for line in proc.stdout.splitlines():
            try:
                raw_pid, command = line.strip().split(None, 1)
                pid = int(raw_pid)
            except ValueError:
                continue
            if pid == current_pid:
                continue
            for pattern in patterns:
                if pattern in command:
                    matched.add(pattern)
        return matched

    def open_session(self, session_id: str) -> Dict[str, Any]:
        session = self.status(session_id)
        if not session:
            raise AOBridgeError(f"AO session not found: {session_id}")
        opened = False
        if session.workspace_path:
            try:
                subprocess.Popen(["open", session.workspace_path])
                opened = True
            except Exception:
                opened = False
        return {
            "ok": True,
            "opened": opened,
            "session": {
                **session.event_fields(),
                "id": session.id,
                "status": session.display_status,
                "activity": session.activity,
                "agent": session.agent,
                "pr": session.pr,
                "summary": session.summary,
            },
        }

    def _call(self, command: str, payload: Dict[str, Any], timeout: int) -> Dict[str, Any]:
        request = {
            "config_path": self.config_path,
            **{k: v for k, v in payload.items() if v is not None},
        }
        env = self._bridge_env()
        if command == "spawn":
            self._ensure_codex_shim_on_user_path()
            self._prepare_tmux_environment(env)
        proc = subprocess.run(
            [self.node_bin, str(self.bridge_script), command],
            input=json.dumps(request),
            text=True,
            capture_output=True,
            timeout=timeout,
            env=env,
            check=False,
        )
        stdout = (proc.stdout or "").strip()
        stderr = (proc.stderr or "").strip()
        try:
            data = json.loads(self._last_json_line(stdout or stderr))
        except json.JSONDecodeError as exc:
            raise AOBridgeError(f"AO bridge returned invalid JSON: {stdout or stderr}") from exc
        if proc.returncode != 0 or data.get("ok") is False:
            raise AOBridgeError(str(data.get("error") or stderr or "AO bridge failed"))
        return data

    def _bridge_env(self) -> Dict[str, str]:
        env = os.environ.copy()
        env["HOME"] = self.home
        env["AO_CONFIG_PATH"] = self.config_path
        env["CODEX_REAL_BIN"] = self.codex_real_bin
        env["HERMES_AO_VERIFICATION_CODEX_HOME"] = (
            os.environ.get("HERMES_AO_VERIFICATION_CODEX_HOME")
            or str(Path(self.home) / ".codex-verification")
        )
        env["HERMES_AO_CODEX_AUTH_HOME"] = (
            os.environ.get("HERMES_AO_CODEX_AUTH_HOME")
            or os.environ.get("CODEX_HOME")
            or str(Path.home() / ".codex")
        )
        shim_path = str(self.codex_shim_dir)
        user_bin = str(self.user_bin_dir)
        current_path = env.get("PATH", "")
        path_parts = current_path.split(os.pathsep) if current_path else []
        prepend = [part for part in (user_bin, shim_path) if part and part not in path_parts]
        if prepend:
            env["PATH"] = os.pathsep.join([*prepend, current_path] if current_path else prepend)
        return env

    def _resolve_codex_real_bin(self, explicit: Optional[str]) -> str:
        candidate = explicit or os.environ.get("CODEX_REAL_BIN") or shutil.which("codex")
        if candidate:
            try:
                resolved_candidate = Path(candidate).resolve()
                resolved_shim = self.codex_shim_path.resolve()
                user_shim = (self.user_bin_dir / "codex").resolve()
                if resolved_candidate not in {resolved_shim, user_shim}:
                    return str(candidate)
            except Exception:
                return str(candidate)
        return DEFAULT_CODEX_BIN

    def _prepare_tmux_environment(self, env: Dict[str, str]) -> None:
        """Make AO-created tmux sessions resolve the Codex compatibility shim."""
        for key in ("PATH", "CODEX_REAL_BIN", "HERMES_AO_VERIFICATION_CODEX_HOME", "HERMES_AO_CODEX_AUTH_HOME"):
            value = env.get(key)
            if not value:
                continue
            try:
                subprocess.run(
                    ["tmux", "set-environment", "-g", key, value],
                    text=True,
                    capture_output=True,
                    timeout=5,
                    check=False,
                )
            except Exception:
                pass

    def _ensure_codex_shim_on_user_path(self) -> None:
        """Install a non-destructive ~/bin/codex shim for AO tmux shells."""
        target = self.user_bin_dir / "codex"
        try:
            if target.exists() or target.is_symlink():
                return
            self.user_bin_dir.mkdir(parents=True, exist_ok=True)
            target.symlink_to(self.codex_shim_path)
        except Exception:
            pass

    def _archived_sessions(self, project_id: Optional[str] = None) -> list[AOSession]:
        root = Path(self.home) / ".agent-orchestrator"
        if not root.exists():
            return []
        candidates = sorted(
            root.glob("*/sessions/archive/*"),
            key=lambda path: path.stat().st_mtime if path.exists() else 0,
            reverse=True,
        )
        sessions: list[AOSession] = []
        for path in candidates:
            if not path.is_file():
                continue
            try:
                data = self._read_flat_session_file(path)
            except Exception:
                continue
            if project_id and data.get("project") != project_id:
                continue
            session_id = path.name.split("_", 1)[0]
            runtime_handle = data.get("runtimeHandle")
            if isinstance(runtime_handle, str):
                try:
                    runtime_handle = json.loads(runtime_handle)
                except json.JSONDecodeError:
                    runtime_handle = {}
            runtime_data = runtime_handle.get("data") if isinstance(runtime_handle, dict) else {}
            tmux_name = data.get("tmuxName") or (runtime_handle or {}).get("id")
            status = data.get("status") or "terminated"
            if status in {"spawning", "working", "running"}:
                status = "terminated"
            sessions.append(AOSession(
                id=session_id,
                project_id=data.get("project"),
                status=status,
                branch=data.get("branch"),
                workspace_path=data.get("worktree") or (runtime_data or {}).get("workspacePath"),
                tmux_name=tmux_name,
                agent=data.get("agent"),
                model=data.get("model"),
                reasoning_effort=data.get("reasoningEffort") or data.get("reasoning_effort"),
                open_command=f"tmux attach -t {tmux_name}" if tmux_name else None,
            ))
        return sessions

    def _with_project_defaults(self, session: AOSession) -> AOSession:
        defaults = self._project_agent_defaults.get(session.project_id or "") if session.project_id else None
        if defaults:
            session.agent = session.agent or defaults.get("agent")
            session.model = session.model or defaults.get("model")
            session.reasoning_effort = session.reasoning_effort or defaults.get("reasoning_effort")
        return session

    @cached_property
    def _project_agent_defaults(self) -> Dict[str, Dict[str, Optional[str]]]:
        try:
            import yaml

            raw = yaml.safe_load(Path(self.config_path).read_text(encoding="utf-8")) or {}
        except Exception:
            return {}

        defaults = raw.get("defaults") or {}
        default_agent = defaults.get("agent") or DEFAULT_AGENT
        projects = raw.get("projects") or {}
        result: Dict[str, Dict[str, Optional[str]]] = {}
        for project_id, project in projects.items():
            if not isinstance(project, dict):
                continue
            agent_config = project.get("agentConfig") or {}
            result[str(project_id)] = {
                "agent": project.get("agent") or default_agent,
                "model": agent_config.get("model"),
                "reasoning_effort": (
                    agent_config.get("reasoningEffort")
                    or agent_config.get("reasoning_effort")
                    or agent_config.get("modelReasoningEffort")
                    or agent_config.get("model_reasoning_effort")
                ),
            }
        return result

    @cached_property
    def _project_paths(self) -> Dict[str, str]:
        try:
            import yaml

            raw = yaml.safe_load(Path(self.config_path).read_text(encoding="utf-8")) or {}
        except Exception:
            return {}
        result: Dict[str, str] = {}
        for project_id, project in (raw.get("projects") or {}).items():
            if isinstance(project, dict) and project.get("path"):
                result[str(project_id)] = str(Path(str(project.get("path"))).expanduser())
        return result

    @staticmethod
    def _parse_codex_token_total(output: str) -> Optional[int]:
        match = re.search(r"tokens used\s*\n\s*([0-9,]+)", output or "", re.IGNORECASE)
        if not match:
            return None
        try:
            return int(match.group(1).replace(",", ""))
        except ValueError:
            return None

    @staticmethod
    def _read_flat_session_file(path: Path) -> Dict[str, Any]:
        data: Dict[str, Any] = {}
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            data[key.strip()] = value.strip()
        return data

    @staticmethod
    def _last_json_line(output: str) -> str:
        for line in reversed((output or "").splitlines()):
            candidate = line.strip()
            if candidate.startswith("{") and candidate.endswith("}"):
                return candidate
        return output or "{}"
