---
sidebar_position: 4
title: "Execution Router Plugin API"
description: "Contract 1.0 for policy-neutral, pre-credential execution routing"
---

# Execution Router Plugin API

`agent.execution_router` defines contract `1.0` for a plugin to propose the route of one new execution attempt. Hermes remains responsible for eligibility, consent, credentials, executor construction, lifecycle, fallback authority, and events. A router never dispatches work itself.

## Contract and discovery

`discover_execution_router_capabilities()` returns the host release/build identity, supported and operational execution kinds, contract versions, limits, and event/read availability. Discovery is read-only: it neither registers nor activates a provider.

Contract `1.0` has exactly three execution kinds:

- `main_turn`
- `native_child`
- `kanban_worker`

The provider callback is synchronous:

`resolve_execution_route(request, cancellation) -> ExecutionRouteDecisionV1 | None`

The provider returns `route`, `pass_through`, or `stop`. `None` is normalized to `pass_through`. `router_error` is host-owned and is not a provider decision. An unsupported kind, incompatible version or schema, malformed decision, conflict, exception, or host deadline expiry fails before executor creation.

## Registration and consent

A native plugin declares the narrow `execution.routing` capability and calls `PluginContext.register_execution_router(provider)`. Hermes admits one active provider per execution scope. Registration validates the frozen descriptor, contract version, all three supported kinds, plugin identity/version, and the live capability grant.

Installation or a generic capability grant does not activate routing. The first valid registration records a pending five-field consent subject:

- plugin ID
- plugin version
- provider ID
- execution-router contract version
- capability-set hash

An interactive `hermes plugins enable <plugin-id>` confirmation records exact consent. Activation occurs only on a subsequent normal plugin load. `hermes plugins capabilities <plugin-id>` reports pending, consented, or stale state without changing it. Any subject change requires consent again; unload or disable removes future routing authority without rewriting existing attempt history.

## Request, decision, and event lifecycle

Hermes creates an immutable, credential-free `ExecutionRouteRequestV1` after execution authorization and before credential binding or executor construction. It contains opaque IDs, execution/surface kind, bounded instruction projection, explicit pins, the native candidate ID when present, a host-issued eligibility projection, one bounded previous-attempt projection, and digest/revision bindings.

A `route` decision names only one `candidate_id` from `request.eligible_candidates`. Hermes validates correlation, eligibility revision, explicit pins, and the selected target before binding credentials. `pass_through` preserves the native route. `stop` prevents executor creation and carries only bounded reason fields.

The host records these distinct states through the existing execution owner:

1. `route_requested`
2. `route_accepted`
3. `route_started`
4. `route_not_started`
5. `route_finished`

Requested, accepted, and actual routes are separate facts. Events use host-issued correlation IDs and expose read-only observation; they provide no replay, dispatch, retry, mutation, or completion authority.

## Redaction and bounds

The host applies execution-kind authorization and existing redaction before disclosure. The provider does not receive credentials, approval tokens, system prompts, conversation history, tools, workspace handles, writable config/tasks, live agents, or dispatch handles.

The instruction projection is at most 16,384 UTF-8 bytes after redaction. Its digest binds the exact delivered bytes; truncation and original byte count are explicit. The complete serialized request is at most 65,536 bytes, contains at most 128 eligible candidates, and carries at most one previous-attempt link. Reason text is at most 512 UTF-8 bytes. Provider execution has a 250 ms host deadline; late output is discarded. Contract constants in `agent.execution_router` are authoritative for the remaining field and event-page bounds.

## Native behavior and fallback

With no registered and consented provider, Hermes uses the existing path without a callback or synthetic route events. Explicit `pass_through` also preserves each execution kind's native route and native failure handling, identified as native pass-through rather than router-selected execution.

A router-selected route is immutable within its attempt. After a routed failure:

- `main_turn`: existing surface authority may consume its existing budget and create a fresh attempt with a fresh router decision.
- `kanban_worker`: existing worker failure/retry/requeue authority may later create a fresh attempt with a fresh router decision.
- `native_child`: contract `1.0` returns a bounded terminal failure when recovery would require a route/model change; it does not construct or resubmit a replacement inside the same `delegate_task` unit. A later ordinary parent delegation is a new execution.

The router never decides whether a retry is allowed and never expands retry, depth, cost, permission, tool, or workspace ceilings.

## Reference fixture

`agent.execution_router_reference.FirstEligibleExecutionRouterProviderV1` is an inert, deterministic, credential-free fixture. It inspects only the public request projection, selects the first host-issued eligible `candidate_id`, and otherwise returns `pass_through`. It has no registration side effect; a plugin must construct it with that plugin's exact frozen descriptor and explicitly register it through the supported `PluginContext` mechanism.

The fixture has no credentials, network access, model/tool execution, classifier, catalog, chains, retry, storage, command, auto-registration, or runtime activation.

## Non-goals

Contract `1.0` does not provide or define:

- model policy, complexity classification, model catalogs, tiers, or routing chains;
- credentials, endpoints with secrets, clients, executors, dispatchers, queues, stores, schedulers, gateways, or retry engines;
- new commands, UI, config keys, permission expansion, task selection, completion authority, or consumer-specific behavior;
- auxiliary LLM, cron, plugin-owned `ctx.llm`, aggregation, or provider-internal retry routing;
- forced cancellation, descendant cleanup, process isolation, or a sandbox for trusted Python plugins;
- installation, deployment, migration, publication, pilot, or LIVE activation.
