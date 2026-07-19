"""Tests for the homelab/infrastructure DANGEROUS_PATTERNS pack (#65122).

kubectl / talosctl / zfs / zpool / firewall / wireguard / ansible mutations
should route through the native approval prompt, while their read-only forms
stay unprompted.

The kubectl `--dry-run=client|server` and ansible-playbook
`--check`/`--syntax-check` exemptions are ARGV-scoped and SEGMENT-scoped, not
substring-scoped: a flag exempts only when it parses as an option token of that
command in that shell segment. The mere spelling of the flag inside a filepath,
an option operand, or a quoted value must never suppress the prompt, and a flag
in one segment must never whitelist a live command after `&&`/`||`/`;`/`|`. The
exemption runs on the non-lowercased command so ansible's `-C` (--check) stays
distinct from `-c` (--connection).

Host-power verbs are deliberately left to the (stronger) hardline floor.
"""

import pytest

from tools.approval import detect_dangerous_command, detect_hardline_command


# Mutations that MUST be flagged dangerous (→ approval prompt).
_DANGEROUS = [
    "kubectl delete pod nginx",
    "kubectl apply -f manifest.yaml",
    "kubectl --namespace prod scale deploy web --replicas=0",
    "kubectl drain node-1",
    "kubectl cordon node-2",
    "kubectl taint nodes n1 key=val:NoSchedule",
    "kubectl patch deploy web -p '{}'",
    "kubectl exec my-pod -- rm -rf /data",
    "kubectl cp local pod:/remote",
    "kubectl replace -f m.yaml",
    "kubectl create -f m.yaml",
    "sudo kubectl edit configmap app",
    "talosctl reset --nodes 10.0.0.1",
    "talosctl reboot --nodes 10.0.0.1",
    "talosctl apply-config -f controlplane.yaml",
    "talosctl upgrade --image ghcr.io/x",
    "zfs destroy tank/dataset",
    "zfs rollback tank/ds@snap",
    "zfs rename tank/a tank/b",
    "zfs set compression=on tank",
    "zfs unmount tank/ds",
    "zpool destroy tank",
    "zpool labelclear /dev/sda",
    "zpool detach tank /dev/sdb",
    "zpool remove tank /dev/sdc",
    "zpool offline tank /dev/sdd",
    "zpool replace tank old new",
    "zpool clear tank",
    "ufw allow 22",
    "ufw deny 80",
    "ufw delete allow 22",
    "ufw enable",
    "ufw disable",
    "ufw reset",
    "iptables -A INPUT -j DROP",
    "iptables -D INPUT 1",
    "iptables -F",
    "iptables -X",
    "iptables -P INPUT DROP",
    "ip6tables -A INPUT -j DROP",
    "sudo iptables -t nat -I PREROUTING 1 -j REDIRECT",
    "nft add rule inet filter input drop",
    "nft delete table inet filter",
    "nft flush ruleset",
    "firewall-cmd --add-port=8080/tcp",
    "firewall-cmd --remove-service=ssh",
    "firewall-cmd --set-default-zone=public",
    "firewall-cmd --new-zone=custom",
    "wg set wg0 peer ABC allowed-ips 0.0.0.0/0",
    "wg setconf wg0 /etc/wireguard/wg0.conf",
    "wg addconf wg0 /etc/wireguard/wg0.conf",
    "ansible-playbook site.yml",
    "ansible-playbook -i inventory prod.yml --become",
]

# Read-only / dry-run forms that MUST stay unprompted.
_SAFE = [
    "kubectl get pods",
    "kubectl get pods -o yaml",
    "kubectl describe pod nginx",
    "kubectl logs my-pod",
    "kubectl diff -f manifest.yaml",
    "kubectl apply -f manifest.yaml --dry-run=client",
    "kubectl delete pod nginx --dry-run=server",
    "talosctl get members",
    "talosctl dmesg",
    "zfs list",
    "zfs get all tank",
    "zpool status",
    "zpool list",
    "ufw status",
    "ufw status verbose",
    "iptables -L",
    "iptables -nvL",
    "iptables -S",
    "ip6tables -L",
    "nft list ruleset",
    "nft list tables",
    "firewall-cmd --list-all",
    "firewall-cmd --state",
    "firewall-cmd --get-default-zone",
    "wg show",
    "wg showconf wg0",
    "ansible-playbook --check site.yml",
    "ansible-playbook site.yml --check",
    "ansible-playbook --syntax-check site.yml",
]


@pytest.mark.parametrize("command", _DANGEROUS)
def test_infra_mutations_are_flagged(command):
    is_dangerous, _key, _desc = detect_dangerous_command(command)
    assert is_dangerous is True, f"expected dangerous: {command!r}"


@pytest.mark.parametrize("command", _SAFE)
def test_readonly_and_dryrun_forms_are_not_flagged(command):
    is_dangerous, _key, _desc = detect_dangerous_command(command)
    assert is_dangerous is False, f"expected safe: {command!r}"


def test_dry_run_none_is_a_live_apply():
    # `--dry-run=none` is a real mutation; only client/server are exempt.
    assert detect_dangerous_command("kubectl delete pod x --dry-run=none")[0] is True


class TestSegmentScopedExemptions:
    """A dry-run/check flag in one shell segment must not whitelist a live
    command in another segment (the key correctness nuance from the issue)."""

    @pytest.mark.parametrize(
        "command",
        [
            "kubectl apply -f m.yaml --dry-run=client && kubectl delete pod live",
            "kubectl delete pod a --dry-run=server; kubectl delete pod b",
            "echo ok --dry-run=client | kubectl delete pod b",
            "ansible-playbook staging.yml --check && ansible-playbook prod.yml",
        ],
    )
    def test_live_segment_still_prompts(self, command):
        assert detect_dangerous_command(command)[0] is True, command

    @pytest.mark.parametrize(
        "command",
        [
            "kubectl apply -f a.yaml --dry-run=client && kubectl diff -f b.yaml",
            "ansible-playbook a.yml --check && ansible-playbook b.yml --syntax-check",
        ],
    )
    def test_all_dry_run_segments_stay_safe(self, command):
        assert detect_dangerous_command(command)[0] is False, command


class TestExemptionsAreOptionTokensNotSubstrings:
    """The exempt SPELLING appearing anywhere in the command must not exempt.

    A negative-lookahead exemption over the raw character run matched the flag
    text wherever it occurred: in a filepath, in an `--option=<value>` operand,
    in a quoted string, after `--`, or as the operand of an option that takes a
    separate argument. Every command here performs a real mutation.
    """

    @pytest.mark.parametrize(
        "command",
        [
            # The flag text is part of a path operand, not an option token.
            "kubectl apply -f /tmp/--dry-run=client",
            "kubectl delete pod prod-db -f /tmp/--dry-run=server",
            "kubectl apply -f ./--dry-run=client/manifest.yaml",
            "ansible-playbook /tmp/--check",
            "ansible-playbook /tmp/--syntax-check",
            # The flag text is the value of a different option.
            "kubectl delete ns prod --context=--dry-run=client",
            "kubectl create secret generic s --from-literal=note=--dry-run=client",
            "kubectl apply -f m.yaml -o jsonpath='--dry-run=client'",
            "ansible-playbook prod.yml -e 'mode=--check'",
            "ansible-playbook prod.yml --extra-vars notes=--syntax-check",
            # The flag text is the separate argument of a preceding option.
            "kubectl delete pod x -n --dry-run=client",
            "ansible-playbook -i inv prod.yml --tags --check",
            "ansible-playbook prod.yml -t --check",
            # Everything after `--` belongs to an inner command.
            "kubectl exec pod -- echo --dry-run=client",
            # A longer option that merely starts with the exempt spelling.
            "ansible-playbook prod.yml --check-mode",
            # kubectl's --dry-run has a NoOptDefVal: a following bare word is a
            # positional, so the detached spelling supplies no value.
            "kubectl apply -f m.yaml --dry-run client",
            # Case matters: `-c` is --connection, not --check.
            "ansible-playbook prod.yml -c ssh",
        ],
    )
    def test_substring_lookalikes_still_prompt(self, command):
        assert detect_dangerous_command(command)[0] is True, command


class TestArgvParsedExemptionsStayExempt:
    """Genuine dry runs must keep their exemption after the argv rewrite."""

    @pytest.mark.parametrize(
        "command",
        [
            "kubectl apply -f manifest.yaml --dry-run=client",
            "kubectl delete pod nginx --dry-run=server",
            "sudo kubectl apply -f m.yaml --dry-run=server",
            "kubectl apply -f m.yaml --wait --dry-run=client",
            "kubectl delete pod x --namespace prod --dry-run=server",
            "ansible-playbook -C prod.yml",
            "ansible-playbook -Cv prod.yml",
            "ansible-playbook -i inv prod.yml --check",
        ],
    )
    def test_real_dry_runs_are_not_flagged(self, command):
        assert detect_dangerous_command(command)[0] is False, command


def test_host_power_delegated_to_hardline_not_dangerous():
    # Host-power verbs are intentionally omitted from DANGEROUS_PATTERNS: the
    # hardline floor blocks them unconditionally, which is stronger than a
    # prompt. Assert the division of responsibility so a future edit can't
    # silently downgrade them to a bypassable prompt.
    for cmd in ("reboot", "poweroff", "shutdown -h now", "init 0"):
        assert detect_dangerous_command(cmd)[0] is False, f"{cmd!r} should not be a DANGEROUS prompt"
        assert detect_hardline_command(cmd)[0] is True, f"{cmd!r} should be hardline-blocked"
