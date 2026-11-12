"""Parity between the approval layer and the in-gateway lifecycle hard block (launchctl verbs)."""

from tools.approval import detect_dangerous_command


# launchd verbs that stop, unload or deregister a running gateway. `disable`
# does not stop a live job on its own, but it is what makes an unload survive
# a reboot, so it belongs to the same family.
GATEWAY_LIFECYCLE_LAUNCHCTL = (
    "launchctl kickstart -k gui/501/ai.hermes.gateway",
    "launchctl unload ~/Library/LaunchAgents/ai.hermes.gateway.plist",
    "launchctl load ~/Library/LaunchAgents/ai.hermes.gateway.plist",
    "launchctl stop ai.hermes.gateway",
    "launchctl restart ai.hermes.gateway",
    "launchctl bootout gui/501/ai.hermes.gateway",
    "launchctl remove ai.hermes.gateway",
    "launchctl disable gui/501/ai.hermes.gateway",
)


class TestLifecycleGuardLaunchctlParity:
    """The in-gateway hard block must cover every launchd verb the approval
    layer already treats as gateway lifecycle.

    These two layers are not interchangeable. In ``tools/terminal_tool.py``
    under ``_HERMES_GATEWAY == "1"``, the ``cron.lifecycle_guard`` block is
    documented as applying unconditionally ("force=True cannot help here"),
    while ``detect_dangerous_command`` below it is explicitly skipped when
    ``force=True``. A verb covered only by the approval layer is therefore
    reachable from inside the gateway, where SIGTERM propagates to the child
    before the command completes and the service may never come back (#74973).

    ``bootout`` was missing exactly this way: it is the modern replacement for
    the ``unload`` the guard already listed. See #80260.
    """

    def test_hard_block_covers_every_lifecycle_verb(self):
        from cron.lifecycle_guard import contains_gateway_lifecycle_command

        for cmd in GATEWAY_LIFECYCLE_LAUNCHCTL:
            assert contains_gateway_lifecycle_command(cmd) is True, cmd

    def test_bypassable_layer_is_never_stricter(self):
        """One-directional invariant: anything ``detect_dangerous_command``
        flags as gateway lifecycle, the hard block must also catch.

        Not equality — the hard block is legitimately stricter (it also covers
        ``load``/``restart``, which the approval layer leaves alone). What must
        never happen is the reverse: a command stopped only by the layer that
        ``force=True`` skips, leaving no cover inside the gateway."""
        from cron.lifecycle_guard import contains_gateway_lifecycle_command

        for cmd in GATEWAY_LIFECYCLE_LAUNCHCTL:
            dangerous, _, _ = detect_dangerous_command(cmd)
            if not dangerous:
                continue
            assert contains_gateway_lifecycle_command(cmd) is True, (
                f"approval layer flags this but the unbypassable hard block "
                f"does not: {cmd}"
            )

    def test_unrelated_labels_are_not_blocked(self):
        """The label anchor must still scope this to the gateway — unrelated
        services, including other Hermes ones, stay runnable."""
        from cron.lifecycle_guard import contains_gateway_lifecycle_command

        for cmd in (
            "launchctl bootout gui/501/com.example.unrelated",
            "launchctl remove ai.hermes.update-checker",
            "launchctl disable gui/501/com.apple.WindowServer",
            "launchctl print system/com.apple.WindowServer",
        ):
            assert contains_gateway_lifecycle_command(cmd) is False, cmd
