"""Shell execution RPC, rebound onto the gateway server namespace."""

from .method_ctx import HandlerRegistry, bind_module

_registry = HandlerRegistry()
method = _registry.method


@method("shell.exec")
def _(rid, params: dict) -> dict:
    cmd = params.get("command", "")
    if not cmd:
        return _err(rid, 4004, "empty command")
    try:
        approval = _tools_mod("tools.approval_detection")
        is_hardline, hardline_desc = approval.detect_hardline_command(cmd)
        if is_hardline:
            return _err(rid, 4005, f"blocked (hardline): {hardline_desc}. Use the agent for dangerous commands.")
        is_dangerous, _, desc = approval.detect_dangerous_command(cmd)
        if is_dangerous:
            return _err(rid, 4005, f"blocked: {desc}. Use the agent for dangerous commands.")
    except ImportError:
        return _err(rid, 5001, "shell.exec unavailable: approval safety module not importable")

    def done(result):
        redact = _tools_mod("agent.redact").redact_sensitive_text
        # Unlike the interactive CLI, this output crosses the RPC boundary and can be persisted
        # in the transcript. Redact before tailing so a credential crossing the slice boundary
        # cannot survive as two unmatched fragments.
        stdout = redact(result.stdout or "", force=True, redact_url_credentials=True)[-4000:]
        stderr = redact(result.stderr or "", force=True, redact_url_credentials=True)[-2000:]
        return _ok(rid, {"stdout": stdout, "stderr": stderr, "code": result.returncode})

    # shell=True preserves the user-facing !cmd grammar (pipes, redirects and interpolation).
    # The child must not inherit credentials held by the long-lived gateway process.
    env = _tools_mod("tools.environments.local").build_subprocess_env()
    return _captured_exec(
        rid, cmd, 30, shell=True, env=env, fail_code=5003,
        timeout_err=(5002, "command timed out (30s)"), on_result=done)


def register(server) -> None:
    bind_module(globals(), server, skip=("_",))
