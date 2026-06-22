#!/usr/bin/env python3
"""Outil « demander un outil manquant » (capacité B — box Jean-Billie).

Quand il MANQUE un outil pour accomplir une demande (facturer, encaisser, suivre des candidatures…),
l'agent appelle cet outil avec une INTENTION en langage naturel. L'outil POSTe l'intention au control
daemon sur le loopback (127.0.0.1, sans mTLS — moindre privilège : le workload ne détient aucun cert de
flotte), qui la relaie au control-plane. Celui-ci traduit l'intention en outil ALLOWLISTÉ (jamais inventé)
et renvoie un message white-label + un éventuel lien de branchement, que l'agent présente au client.

PUREMENT INTERNE (loopback vers notre propre daemon) → non intercepté par jb_outbound (classify → PASS) :
on génère un lien que le CLIENT cliquera lui-même, rien ne part vers un tiers.

Gated sur JB_DECISION_PUSH_URL (posé par le bundle Jean-Billie) : invisible hors box (cf. check_fn).
"""

from __future__ import annotations

import json
import os
import urllib.request


def _request_connection_url() -> str:
    """URL loopback du daemon où POSTer la demande (RouteRequestConnection)."""
    addr = (os.getenv("JB_DRAFT_ADDR", "127.0.0.1:8442").strip() or "127.0.0.1:8442")
    return f"http://{addr}/v1/request-connection"


def request_tool_connection(capability: str) -> str:
    """Demande au control-plane comment brancher l'outil correspondant à `capability`.

    Renvoie un JSON `{status, message, connectUrl}` : `message` est à présenter au client (l'agent le
    reformule dans son ton), `connectUrl` est le lien de branchement (présent si `link-ready`).
    """
    capability = (capability or "").strip()
    if not capability:
        return json.dumps({"error": "intention requise"}, ensure_ascii=False)

    payload = json.dumps({"capability": capability}).encode("utf-8")
    req = urllib.request.Request(
        _request_connection_url(),
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10.0) as resp:  # noqa: S310 (loopback only)
            body = resp.read()
        out = json.loads(body.decode("utf-8")) if body else {}
    except Exception:
        # Pas de bluff : on ne sait pas faire ça pour l'instant (daemon/control-plane injoignable).
        return json.dumps(
            {
                "status": "unavailable",
                "message": "Je ne sais pas encore faire ça pour vous. Je vous le proposerai dès que ce sera possible.",
            },
            ensure_ascii=False,
        )

    return json.dumps(
        {
            "status": out.get("status", "no-match"),
            "message": out.get("message", ""),
            "connectUrl": out.get("connectUrl", ""),
        },
        ensure_ascii=False,
    )


def check_request_tool_connection_requirements() -> bool:
    """Disponible UNIQUEMENT sur la box Jean-Billie (le daemon loopback + le control-plane sont câblés).

    Même garde que le plugin jb_outbound : `JB_DECISION_PUSH_URL` est posé par le bundle Jean-Billie. En
    CLI/dev standard sans cette variable, l'outil n'apparaît pas (il n'aurait personne à qui parler).
    """
    return bool(os.getenv("JB_DECISION_PUSH_URL"))


# =============================================================================
# OpenAI Function-Calling Schema
# =============================================================================

REQUEST_TOOL_CONNECTION_SCHEMA = {
    "name": "request_tool_connection",
    "description": (
        "Quand il te MANQUE un outil pour accomplir une demande (par exemple facturer un client, "
        "encaisser un paiement, suivre des candidatures), appelle cet outil avec l'INTENTION en langage "
        "naturel. Tu recevras un message à présenter au client — souvent un lien pour brancher l'outil "
        "depuis son espace. N'invente JAMAIS de lien : utilise uniquement ce que cet outil te renvoie, "
        "et ne le propose qu'UNE fois par demande. Si le statut est « no-match » ou « unavailable », "
        "dis simplement et franchement que tu ne sais pas encore faire ça."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "capability": {
                "type": "string",
                "description": (
                    "L'intention en langage naturel (« facturer un client », « encaisser un paiement », "
                    "« suivre des candidatures »)."
                ),
            },
        },
        "required": ["capability"],
    },
}


# --- Registry ---
from tools.registry import registry

registry.register(
    name="request_tool_connection",
    # Toolset déclaré dans toolsets.py (et présent dans l'allowlist de durcissement, cf. SAFE list +
    # « messaging »). L'exposition réelle vient surtout de _HERMES_CORE_TOOLS (CLI + plateformes).
    toolset="messaging",
    schema=REQUEST_TOOL_CONNECTION_SCHEMA,
    handler=lambda args, **kw: request_tool_connection(capability=args.get("capability", "")),
    check_fn=check_request_tool_connection_requirements,
    emoji="🔌",
)
