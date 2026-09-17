"""Bounded caller contract and actionable compiler corrections."""
CONTRACT = {
    "actions": {"discover": "up to 8 read/discovery preflight probes; preparation refs only, no frozen plan or mutation authority",
                "execute": "operation_key + items/items_ref + steps, or recipe_key + items/items_ref",
                "resume": "plan_id; confirmed steps are skipped; uncertain mutations require review",
                "status": "plan_id; reconstruct state without dispatch", "contract": "this guide; no dispatch"},
    "bindings": {"$item.field": "current structured record", "$setup.id.field": "shared setup result",
                 "$steps.id.field": "prior result in the same phase", "$items_ref": "finalize-only artifact manifest"},
    "graph": "setup_steps once, steps per item, finalize_steps after all items complete. IDs unique; depends_on forms a DAG.",
    "verification": "Mutation steps need expect. For multi-item mutations add a read/discovery step with verifies=[mutation_id] and expect proving persisted state. success=true or click completion alone is insufficient. browser_console is arbitrary JS, so cannot be a read verifier.",
    "canary": "Unknown mutable workflow uses item 1 as canary; no extra item. Unverified/uncertain canary blocks remaining items. A VERIFIED recipe with passing read-only preflight can reuse its stored graph.",
    "recipes": "recipe_key optionally saves externally verified graph. recipe_scope describes route/host/path_family. preflight is at most 8 read/discovery steps with expect. STALE recipes need corrected graph and new canary; no silent bulk execution.",
    "constraints": "allowed_routes/forbidden_routes are enforced before dispatch; cannot relax the user's constraints",
    "discovery": "Discovery may prepare a mutation plan, but discovery may never become an untracked mutation channel. Use browser_snapshot and browser_extract_items inspection; never label browser_console or terminal read-only.",
    "capabilities": "Successful read capability does not establish write capability. Check provider + channel + method separately. A PUT 403 rejects PUT, never GET. UI-only write policies permit independent API readback.",
    "persistence": "Fan-out is authorized by verified persistence of a representative canary, not by successful execution of a preparation step.",
    "outputs": "SUMMARY default: refs + bounded ledger. FULL explicit: load output content. Raw outputs remain artifacts.",
    "examples": [
        {"operation_key": "read-records-v1", "items": [{"path": "a.txt"}],
         "steps": [{"id": "read", "tool": "read_file", "args": {"path": "$item.path"}}]},
        {"operation_key": "cards-v1", "recipe_key": "cards.native.v1", "items": [{"title": "A"}, {"title": "B"}],
         "steps": [{"id": "create", "tool": "connector_create", "args": {"title": "$item.title"}, "expect": {"ok": True}},
                   {"id": "verify", "tool": "connector_get", "verifies": ["create"], "args": {"id": "$steps.create.id"}, "expect": {"persisted": True}}]},
        {"recipe_key": "cards.native.v1", "items": [{"title": "C"}, {"title": "D"}]}],
}


def correction(error):
    text = str(error)
    if "decided operation" in text:
        from workstation.task_compiler import discovery_guidance
        return {"error": text[:400], "code": "PREFLIGHT_REQUIRED", **discovery_guidance(),
                "fix": {"discover_before_compile": True}}
    if "recipe_fingerprint" in text:
        code, fix = "recipe_fingerprint_mismatch", {"provide_corrected_graph": True, "run_new_canary": True}
    elif "recipe_stale" in text:
        code, fix = "recipe_stale", {"provide_corrected_graph": True, "run_new_canary": True}
    elif "recipe_scope" in text or "Browser recipe requires" in text:
        code, fix = "recipe_scope_mismatch", {"match_verified_host_path_family": True, "use_read_preflight_reporting_url": True}
    elif "verifier" in text or "verification" in text:
        code, fix = "mutation_verifier_required", {"add_expect": True, "add_read_verifier": True, "example": CONTRACT["examples"][1]["steps"][1]}
    elif "route" in text or "constraint" in text.lower():
        code, fix = "constraint_violation", {"choose_permitted_route": True, "never_relax_user_constraints": True}
    elif "binding" in text or "Cross-item" in text or "Shared step" in text:
        code, fix = "invalid_binding", {"allowed_bindings": list(CONTRACT["bindings"])}
    else:
        code, fix = "invalid_graph", {"check_unique_ids_dependencies_bindings": True, "contract_action": "contract"}
    return {"error": text[:400], "code": code, "message": text[:400], "fix": fix}
