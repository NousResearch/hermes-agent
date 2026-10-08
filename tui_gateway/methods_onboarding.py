from .method_ctx import HandlerRegistry, bind_module

_registry = HandlerRegistry()
method = _registry.method


# onboarding.state and onboarding.set_run are unscoped on purpose: onboarding.run and the first-chat
# flag live in the root profile's config.yaml whichever profile this backend was launched under.
@method("onboarding.state")
def _(rid, params: dict) -> dict:
    return _ok(rid, _run_state())


@method("onboarding.set_run")
def _(rid, params: dict) -> dict:
    from hermes_cli.onboarding_run import set_run
    if not isinstance(params.get("run"), bool):
        return _err(rid, 4002, "onboarding.set_run requires a boolean 'run'")
    try:
        set_run(params["run"], mark_profile_offered=bool(params.get("mark_profile_offered")))
    except Exception as e:
        logger.exception("onboarding.set_run failed")
        return _err(rid, 5077, str(e))
    return _ok(rid, _run_state())


def _run_state() -> dict:
    from hermes_cli.anon_auth import guest_enabled
    from hermes_cli.onboarding_run import should_run
    return {"run": should_run(), "eligible": guest_enabled()}


def register(server) -> None:
    bind_module(globals(), server, skip=("_",))
