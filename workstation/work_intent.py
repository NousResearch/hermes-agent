"""One transient operational decision, reusing canonical classifiers."""
from dataclasses import dataclass, field

from workstation.contracts import AcceptanceContract, MessageEnvelope, RiskLevel


@dataclass(slots=True)
class WorkIntent:
    execution_class: str
    requires_task: bool
    requires_browser: bool
    requires_worker: bool
    durability: str
    risk: RiskLevel
    acceptance_policy: AcceptanceContract
    handoff_policy: str = "explicit"
    repeatability_hint: bool = False
    constraints: dict = field(default_factory=dict)


def work_intent(envelope: MessageEnvelope, structured: dict | None = None) -> WorkIntent:
    from workstation.kanban import is_multistep_request
    from workstation.task_compiler import batch_intent, classify
    kind = classify(structured or {}).value
    steps = (structured or {}).get("steps", [])
    browser = any(str(s.get("tool", "")).startswith("browser_") for s in steps)
    durable = structured is not None or batch_intent(envelope.content)
    requires_task = envelope.can_create_work and (durable or is_multistep_request(envelope.content))
    return WorkIntent(kind, requires_task, browser, False, "durable" if durable else "turn",
                      RiskLevel.MEDIUM if durable else RiskLevel.LOW,
                      AcceptanceContract(), repeatability_hint=durable,
                      constraints=(structured or {}).get("constraints", {}))


def prepare_turn_work(agent, content, envelope: MessageEnvelope | None = None) -> WorkIntent:
    """Bind trusted ingress to canonical work without changing the prompt or schema."""
    from workstation.contracts import MessageOrigin, IntentAuthority
    from workstation.task_compiler import batch_intent
    root = getattr(agent, "_conversation_root_id", lambda: None)()
    session_id = str(root or getattr(agent, "session_id", None) or "")
    if envelope is None:
        envelope = MessageEnvelope(MessageOrigin.RUNTIME, IntentAuthority.OBSERVATION_ONLY,
                                   session_id, content if isinstance(content, str) else "")
    if not isinstance(envelope, MessageEnvelope) or envelope.session_id != session_id:
        raise ValueError("Message envelope must belong to the owning conversation")
    intent = work_intent(envelope)
    agent._message_envelope = envelope
    agent._work_intent = intent
    agent._work_repeatability_hint = batch_intent(envelope.content)
    if intent.requires_task:
        from workstation.kanban import WorkstationKanbanBridge
        from hermes_cli import kanban_db
        bridge = WorkstationKanbanBridge()
        current_id = getattr(agent, "_canonical_work_task_id", None)
        current_run_id = getattr(agent, "_canonical_work_run_id", None)
        if current_id:
            conn = bridge.get_connection()
            try:
                current = kanban_db.get_task(conn, current_id)
                if (current is None or current.session_id != session_id or current.status in {"done", "cancelled"}
                        or current.body != envelope.content):
                    current_id = None
                    current_run_id = None
                else:
                    current_run_id = current.current_run_id
            finally:
                conn.close()
        if current_id is None:
            current_id = bridge.promote_request_if_multistep(envelope.content, session_id=session_id, envelope=envelope,
                                                           acceptance_contract=intent.acceptance_policy)
            if current_id:
                conn = bridge.get_connection()
                try:
                    current = kanban_db.get_task(conn, current_id)
                    if current:
                        current_run_id = current.current_run_id
                finally:
                    conn.close()
        agent._canonical_work_task_id = current_id
        agent._canonical_work_run_id = current_run_id
    return intent
