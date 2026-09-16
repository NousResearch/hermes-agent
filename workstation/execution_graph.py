"""Small deterministic DAG validation over the existing WorkItem checkpoints."""
import re

PHASES = ("setup", "fan_out", "finalize")


def prepare_graph(request):
    groups = [request.get("setup_steps", []), request["steps"], request.get("finalize_steps", [])]
    nodes, ordered = {}, {}
    for phase, steps in zip(PHASES, groups):
        if not isinstance(steps, list) or len(steps) > 64:
            raise ValueError("Invalid bounded graph phase")
        for index, original in enumerate(steps):
            step = dict(original)
            step_id = step.get("id", f"{phase}_{index}")
            if not isinstance(step_id, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_-]{0,63}", step_id) or step_id in nodes:
                raise ValueError("Graph IDs must be valid and unique")
            step["id"] = step_id
            deps = step.get("depends_on", [])
            if not isinstance(deps, list) or any(not isinstance(d, str) for d in deps):
                raise ValueError("Invalid graph dependencies")
            # References carry implicit dependencies too; detect before dispatch.
            def references(value):
                if isinstance(value, str):
                    if value.startswith("$item.") and phase != "fan_out":
                        raise ValueError("Shared step cannot depend on a single item")
                    if value == "$items_ref" and phase != "finalize":
                        raise ValueError("Item results exist only after fan-in")
                    match = re.match(r"\$(setup|steps|finalize)\.([\w-]+)(?:\.|$)", value)
                    if match:
                        return [(match[1], match[2])]
                if isinstance(value, dict):
                    return sum((references(v) for v in value.values()), [])
                if isinstance(value, list):
                    return sum((references(v) for v in value), [])
                return []
            refs = references(step.get("args", {})) + references(step.get("expect", {}))
            step["depends_on"] = sorted(set(deps + [r[1] for r in refs]))
            nodes[step_id] = (phase, step, refs)
    for step_id, (phase, step, refs) in nodes.items():
        for dep in step["depends_on"]:
            if dep not in nodes or PHASES.index(nodes[dep][0]) > PHASES.index(phase):
                raise ValueError("Missing or impossible graph dependency")
        for namespace, ref in refs:
            expected = {"setup": "setup", "steps": phase, "finalize": "finalize"}[namespace]
            if nodes[ref][0] != expected or (phase == "finalize" and nodes[ref][0] == "fan_out"):
                raise ValueError("Cross-item outputs require $items_ref, not a single $steps reference")
    for phase in PHASES:
        pending = {key for key, (p, _, _) in nodes.items() if p == phase}
        done = {key for key, (p, _, _) in nodes.items() if PHASES.index(p) < PHASES.index(phase)}
        ordered[phase] = []
        while pending:
            ready = sorted(key for key in pending if set(nodes[key][1]["depends_on"]) <= done)
            if not ready:
                raise ValueError("Graph cycle detected")
            for key in ready:
                ordered[phase].append(nodes[key][1])
                done.add(key)
                pending.remove(key)
    return ordered
