"""Safety policy for native Home Assistant actions."""

from dataclasses import dataclass
from typing import Any, Collection, Mapping, Optional


BLOCKED_DOMAINS = frozenset(
    {"shell_command", "command_line", "python_script", "pyscript", "hassio", "rest_command"}
)
IMMEDIATE_DOMAINS = frozenset({"light", "fan", "media_player"})
TRUSTABLE_DOMAINS = frozenset({"switch", "scene"})


@dataclass(frozen=True)
class ServicePolicyDecision:
    action: str
    reason: str
    targets: tuple[str, ...] = ()


def _targets(entity_id: Optional[str], data: Mapping[str, Any]) -> tuple[str, ...]:
    target = entity_id if entity_id is not None else data.get("entity_id")
    if not isinstance(target, str) or target == "all":
        return ()
    return (target,)


def classify_service_action(
    domain: str,
    entity_id: Optional[str],
    data: Optional[Mapping[str, Any]],
    *,
    mode: str = "legacy",
    trusted_entities: Collection[str] = (),
) -> ServicePolicyDecision:
    """Classify a service call as allow, approve, or block."""
    payload = data or {}
    targets = _targets(entity_id, payload)
    if domain in BLOCKED_DOMAINS:
        return ServicePolicyDecision("block", f"Service domain '{domain}' is blocked")
    if mode != "safe":
        return ServicePolicyDecision("allow", "Legacy Home Assistant policy", targets)
    if any(key in payload for key in ("area_id", "device_id", "target")) or not targets:
        return ServicePolicyDecision("approve", "Broad or ambiguous Home Assistant target", targets)
    if domain in IMMEDIATE_DOMAINS:
        return ServicePolicyDecision("allow", "Routine exact-target action", targets)
    if domain in TRUSTABLE_DOMAINS and all(target in trusted_entities for target in targets):
        return ServicePolicyDecision("allow", "Explicitly trusted Home Assistant entity", targets)
    return ServicePolicyDecision("approve", "Sensitive Home Assistant action", targets)
