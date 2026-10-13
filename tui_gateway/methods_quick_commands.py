"""User-defined ``quick_commands`` (exec / alias / prompt) for commands.catalog, command.dispatch
and slash.exec.

Bodies are rebound onto server.py's globals at install time (see method_ctx.bind_module), so they
reference server.py globals bare (``_ok``, ``_load_cfg``, ``_tools_mod``, ...).
"""

import subprocess

from .method_ctx import HandlerRegistry, bind_module

_registry = HandlerRegistry()


def _catalog_quick_commands(cat) -> None:
    qcmds = _load_cfg().get("quick_commands", {}) or {}
    if not (isinstance(qcmds, dict) and qcmds):
        return
    cat.cat_map.setdefault("User commands", [])  # category exists even when every entry is malformed
    for qname, qc in sorted(qcmds.items()):
        if not isinstance(qc, dict):
            continue
        qtype = qc.get("type", "")
        default_desc = {"exec": f"exec: {qc.get('command', '')}", "alias": f"alias → {qc.get('target', '')}",
                        "prompt": "prompt"}
        desc = str(qc.get("description") or default_desc.get(qtype, qtype or "quick command"))
        cat.add(f"/{qname}", desc, "User commands")
        if qtype == "prompt":
            # Free-text args ({args}); ``review_button`` surfaces it as a Desktop Review pane action.
            meta: dict = {"argument_mode": "text", "desktop": None}
            if label := _tools_mod("hermes_cli.quick_commands").review_button_label(qname, qc):
                meta["review_button"] = label
            cat.commands[f"/{qname}"] = meta


def _dispatch_quick(rid, params, session, name, arg):
    qc = _load_cfg().get("quick_commands", {}).get(name)
    if qc is None:
        return None
    if qc.get("type") == "prompt":
        msg = _tools_mod("hermes_cli.quick_commands").render_prompt_quick_command(qc, arg)
        return _ok(rid, {"type": "send", "message": msg}) if msg else _err(
            rid, 4018, f"quick command /{name} has no prompt defined")
    if qc.get("type") == "exec":
        # Sanitized env: the TUI server process holds every API key in os.environ.
        env = _tools_mod("tools.environments.local").build_subprocess_env()
        kw = _capture_run_kwargs(30)
        r = subprocess.run(qc.get("command", ""), shell=True, env=env, check=False, timeout=kw.pop("timeout"),
                           stdin=kw.pop("stdin"), **kw)
        output = _joined_output(r)[:4000]
        output = _tools_mod("agent.redact").redact_sensitive_text(output) if output else output
        if r.returncode != 0:
            return _err(rid, 4018, output or f"quick command failed with exit code {r.returncode}")
        return _exec_out(rid, output)
    return _ok(rid, {"type": "alias", "target": qc.get("target", "")}) if qc.get("type") == "alias" else None


def _is_prompt_quick_command(name: str) -> bool:
    """``type: prompt`` quick commands answer a ``send`` directive, so slash.exec reroutes them to
    command.dispatch — the slash worker would queue the prompt on its own, never-drained CLI input."""
    qc = (_load_cfg().get("quick_commands") or {}).get(name)
    return isinstance(qc, dict) and qc.get("type") == "prompt"


def register(server) -> None:
    """Rebind this module's helpers onto ``server``."""
    bind_module(globals(), server)
