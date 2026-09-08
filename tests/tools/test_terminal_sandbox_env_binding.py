"""Regression tests for #105414.

The sandbox env builders are ``functools.partial``-bound closures over
``_build_sandbox_env``, and ``_create_environment`` invokes every builder with
``env_type=<type>`` by keyword. When the partial bound ``env_type`` positionally
the keyword collided, raising::

    TypeError: _build_sandbox_env() got multiple values for argument 'env_type'

and breaking the Singularity / Daytona / Vercel terminal backends on every
terminal and file-tool call.
"""

import functools

import tools.terminal_tool_backends as ttb


def test_sandbox_env_builders_bind_env_type_by_keyword():
    """The partial builders must bind ``env_type`` by keyword so the caller's
    ``env_type=`` keyword does not collide (#105414)."""
    cases = [
        (ttb._build_singularity_env, "singularity"),
        (ttb._build_daytona_env, "daytona"),
        (ttb._build_vercel_env, "vercel_sandbox"),
    ]
    for builder, expected in cases:
        assert isinstance(builder, functools.partial), f"{builder!r} is not a functools.partial"
        assert builder.args == (), (
            f"_build_sandbox_env partial must not bind env_type positionally "
            f"(args={builder.args!r}); positional binding collides with the "
            f"env_type= keyword passed by _create_environment (#105414)."
        )
        assert builder.keywords == {"env_type": expected}, (
            f"{builder!r} must bind env_type by keyword "
            f"(got keywords={builder.keywords!r}); only env_type should be bound."
        )


def test_create_environment_singularity_does_not_raise_typeerror(monkeypatch):
    """_create_environment(env_type='singularity') must invoke the sandbox builder
    without ``TypeError: multiple values for argument 'env_type'`` (#105414)."""

    class _FakeEnv:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    # Replace the singularity row so no real SingularityEnvironment is needed.
    monkeypatch.setitem(
        ttb._SANDBOX_ROWS,
        "singularity",
        (lambda: _FakeEnv, True, lambda cc, kw: {}),
    )
    monkeypatch.setattr(ttb, "_resources", lambda cc: {})

    env = ttb._create_environment(
        env_type="singularity",
        image="img",
        cwd="/tmp",
        timeout=30,
        container_config={},
        task_id="t",
    )

    assert isinstance(env, _FakeEnv)
    # singularity row has with_image=True, so image must reach the environment.
    assert env.kwargs.get("image") == "img"
    assert env.kwargs.get("cwd") == "/tmp"


def test_create_environment_daytona_and_vercel_do_not_raise_typeerror(monkeypatch):
    """Same partial-binding regression for Daytona and Vercel sandbox (#105414)."""

    class _FakeEnv:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(ttb, "_resources", lambda cc: {})
    monkeypatch.setitem(
        ttb._SANDBOX_ROWS,
        "daytona",
        (lambda: _FakeEnv, True, lambda cc, kw: {"cpu": int(kw["cpu"])}),
    )
    monkeypatch.setitem(
        ttb._SANDBOX_ROWS,
        "vercel_sandbox",
        (lambda: _FakeEnv, False, lambda cc, kw: {"runtime": cc.get("vercel_runtime") or None}),
    )

    daytona = ttb._create_environment(
        env_type="daytona", image="img", cwd="/tmp", timeout=30,
        container_config={"container_cpu": 2}, task_id="t",
    )
    assert isinstance(daytona, _FakeEnv)
    assert daytona.kwargs.get("cpu") == 2

    vercel = ttb._create_environment(
        env_type="vercel_sandbox", image="ignored", cwd="/tmp", timeout=30,
        container_config={"vercel_runtime": "nodejs20"}, task_id="t",
    )
    assert isinstance(vercel, _FakeEnv)
    assert vercel.kwargs.get("runtime") == "nodejs20"
