"""Deterministic Operational Kernel for executing validated Operational Capabilities.

Executes browser, filesystem, and composite operational primitives without intermediate
LLM calls. Performs variable interpolation, precondition checks, postcondition verification,
and drift quarantine with handoff to reasoning when unexpected variance occurs.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import time
from typing import Any, Callable

from workstation.artifacts import ArtifactStore
from workstation.operational_capabilities import (
    CapabilityDependency,
    CapabilityDriftError,
    CapabilityLifecycle,
    CapabilityNotFoundError,
    CapabilityResolver,
    CapabilityValidationError,
    OperationalCapability,
    OperationalCapabilityRegistry,
)
from workstation.recipes import sanitize, digest


def _resolve_path(path: str, base_dir: str | Path | None = None) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p.resolve()
    base = Path(base_dir) if base_dir else Path.cwd()
    return (base / p).resolve()


def interpolate_variables(value: Any, context: dict[str, Any]) -> Any:
    """Recursively replace $inputs.<var>, $deps.<id>.<var>, $prev.<var> in values."""
    if isinstance(value, dict):
        return {k: interpolate_variables(v, context) for k, v in value.items()}
    if isinstance(value, list):
        return [interpolate_variables(v, context) for v in value]
    if isinstance(value, str):
        # Exact match single token replacement (preserves types, e.g. int, dict, list)
        token_match = re.fullmatch(r"\$([a-zA-Z0-9_]+(?:\.[a-zA-Z0-9_]+)*)", value.strip())
        if token_match:
            resolved = _lookup_dotted_path(token_match.group(1), context)
            if resolved is not None:
                return resolved

        # In-string substring template replacement (both $var and ${var})
        def _replace(match: re.Match) -> str:
            var_name = match.group(1) or match.group(2)
            resolved = _lookup_dotted_path(var_name, context)
            return str(resolved) if resolved is not None else match.group(0)

        return re.sub(r"(?:\$\{([a-zA-Z0-9_]+(?:\.[a-zA-Z0-9_]+)*)\}|\$([a-zA-Z0-9_]+(?:\.[a-zA-Z0-9_]+)*))", _replace, value)
    return value


def _lookup_dotted_path(path_str: str, context: dict[str, Any]) -> Any:
    parts = path_str.split(".")
    # Map input -> inputs alias
    if parts[0] == "input":
        parts[0] = "inputs"
    current = context
    for part in parts:
        if isinstance(current, dict):
            if part in current:
                current = current[part]
            else:
                return None
        elif isinstance(current, list):
            try:
                idx = int(part)
                current = current[idx]
            except (ValueError, IndexError):
                return None
        elif hasattr(current, part):
            current = getattr(current, part)
        else:
            return None
    return current


class OperationalKernel:
    """Engine for deterministic capability execution and composition."""

    def __init__(
        self,
        registry: OperationalCapabilityRegistry | None = None,
        resolver: CapabilityResolver | None = None,
        artifacts: ArtifactStore | None = None,
    ):
        self.artifacts = artifacts or ArtifactStore()
        self.registry = registry or OperationalCapabilityRegistry(artifacts=self.artifacts)
        self.resolver = resolver or CapabilityResolver(registry=self.registry)

    # -------------------------------------------------------------------------
    # Filesystem Primitives
    # -------------------------------------------------------------------------

    def fs_stat(self, path: str, base_dir: str | Path | None = None) -> dict[str, Any]:
        target = _resolve_path(path, base_dir)
        if not target.exists():
            return {"exists": False, "path": str(target)}
        st = target.stat()
        return {
            "exists": True,
            "path": str(target),
            "is_file": target.is_file(),
            "is_dir": target.is_dir(),
            "size": st.st_size,
            "mtime": st.st_mtime,
        }

    def fs_read(self, path: str, offset: int = 0, limit: int | None = None, base_dir: str | Path | None = None) -> str:
        target = _resolve_path(path, base_dir)
        if not target.is_file():
            raise FileNotFoundError(f"File not found: {target}")
        content = target.read_text(encoding="utf-8")
        if offset:
            content = content[offset:]
        if limit is not None:
            content = content[:limit]
        return content

    def fs_write(self, path: str, content: str, overwrite: bool = True, base_dir: str | Path | None = None) -> dict[str, Any]:
        target = _resolve_path(path, base_dir)
        if target.exists() and not overwrite:
            raise FileExistsError(f"File already exists: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write
        temp_file = target.with_name(f"{target.name}.tmp.{os.getpid()}")
        try:
            temp_file.write_text(content, encoding="utf-8")
            temp_file.replace(target)
        except Exception:
            if temp_file.exists():
                temp_file.unlink(missing_ok=True)
            raise
        return {"success": True, "path": str(target), "size": len(content), "sha256": hashlib.sha256(content.encode()).hexdigest()}

    def fs_patch(self, path: str, target_content: str, replacement_content: str, base_dir: str | Path | None = None) -> dict[str, Any]:
        target = _resolve_path(path, base_dir)
        content = target.read_text(encoding="utf-8")
        if target_content not in content:
            raise ValueError(f"Target content not found in {target}")
        new_content = content.replace(target_content, replacement_content, 1)
        self.fs_write(str(target), new_content, overwrite=True)
        return {"success": True, "path": str(target), "modified": True}

    def fs_copy(self, src: str, dst: str, base_dir: str | Path | None = None) -> dict[str, Any]:
        src_p = _resolve_path(src, base_dir)
        dst_p = _resolve_path(dst, base_dir)
        dst_p.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_p, dst_p)
        return {"success": True, "src": str(src_p), "dst": str(dst_p)}

    def fs_move(self, src: str, dst: str, base_dir: str | Path | None = None) -> dict[str, Any]:
        src_p = _resolve_path(src, base_dir)
        dst_p = _resolve_path(dst, base_dir)
        dst_p.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(src_p, dst_p)
        return {"success": True, "src": str(src_p), "dst": str(dst_p)}

    def fs_hash_file(self, path: str, base_dir: str | Path | None = None) -> str:
        target = _resolve_path(path, base_dir)
        return hashlib.sha256(target.read_bytes()).hexdigest()

    def fs_list_dir(self, path: str, recursive: bool = False, base_dir: str | Path | None = None) -> list[dict[str, Any]]:
        target = _resolve_path(path, base_dir)
        if not target.is_dir():
            raise NotADirectoryError(f"Not a directory: {target}")
        results = []
        pattern = "**/*" if recursive else "*"
        for item in target.glob(pattern):
            results.append({
                "name": item.name,
                "path": str(item),
                "is_dir": item.is_dir(),
                "size": item.stat().st_size if item.is_file() else 0,
            })
        return results

    def fs_mkdir(self, path: str, parents: bool = True, base_dir: str | Path | None = None) -> dict[str, Any]:
        target = _resolve_path(path, base_dir)
        target.mkdir(parents=parents, exist_ok=True)
        return {"success": True, "path": str(target)}

    # -------------------------------------------------------------------------
    # Primitive Dispatch
    # -------------------------------------------------------------------------

    def execute_primitive(
        self,
        primitive: str,
        inputs: dict[str, Any],
        *,
        dispatch: Callable | None = None,
        context: dict[str, Any] | None = None,
    ) -> Any:
        ctx = context or {}
        inputs = interpolate_variables(inputs, ctx)

        # 1. Filesystem primitives
        if primitive in {"fs_stat", "stat"}:
            return self.fs_stat(inputs["path"], base_dir=inputs.get("base_dir"))
        if primitive in {"fs_read", "read_file", "read"}:
            return self.fs_read(inputs["path"], offset=inputs.get("offset", 0), limit=inputs.get("limit"), base_dir=inputs.get("base_dir"))
        if primitive in {"fs_write", "write_file", "write"}:
            return self.fs_write(inputs["path"], inputs["content"], overwrite=inputs.get("overwrite", True), base_dir=inputs.get("base_dir"))
        if primitive in {"fs_patch", "patch"}:
            return self.fs_patch(inputs["path"], inputs["target_content"], inputs["replacement_content"], base_dir=inputs.get("base_dir"))
        if primitive in {"fs_copy", "copy"}:
            return self.fs_copy(inputs["src"], inputs["dst"], base_dir=inputs.get("base_dir"))
        if primitive in {"fs_move", "move"}:
            return self.fs_move(inputs["src"], inputs["dst"], base_dir=inputs.get("base_dir"))
        if primitive in {"fs_hash", "hash_file"}:
            return self.fs_hash_file(inputs["path"], base_dir=inputs.get("base_dir"))
        if primitive in {"fs_list_dir", "list_dir"}:
            return self.fs_list_dir(inputs["path"], recursive=inputs.get("recursive", False), base_dir=inputs.get("base_dir"))
        if primitive in {"fs_mkdir", "mkdir"}:
            return self.fs_mkdir(inputs["path"], parents=inputs.get("parents", True), base_dir=inputs.get("base_dir"))

        # 2. Browser primitives (dispatched through agent scoped dispatcher or loopback)
        if primitive in {"browser_navigate", "navigate"}:
            if not dispatch:
                raise RuntimeError("Browser primitive requires dispatch function")
            return dispatch("browser_navigate", {"url": inputs["url"]})
        if primitive in {"browser_snapshot", "snapshot"}:
            if not dispatch:
                raise RuntimeError("Browser primitive requires dispatch function")
            return dispatch("browser_snapshot", {"full": inputs.get("full", False)})
        if primitive in {"browser_click", "click"}:
            if not dispatch:
                raise RuntimeError("Browser primitive requires dispatch function")
            ref = inputs.get("ref")
            if not ref and inputs.get("anchor"):
                snap = dispatch("browser_snapshot", {"full": False})
                elements = snap.get("elements", []) if isinstance(snap, dict) else []
                ref = self._find_ref_by_anchor(elements, inputs["anchor"])
            if not ref:
                raise CapabilityDriftError(f"Target ref could not be resolved for click: {inputs}")
            return dispatch("browser_click", {"ref": ref})
        if primitive in {"browser_type", "type", "fill"}:
            if not dispatch:
                raise RuntimeError("Browser primitive requires dispatch function")
            ref = inputs.get("ref")
            if not ref and inputs.get("anchor"):
                snap = dispatch("browser_snapshot", {"full": False})
                elements = snap.get("elements", []) if isinstance(snap, dict) else []
                ref = self._find_ref_by_anchor(elements, inputs["anchor"])
            if not ref:
                raise CapabilityDriftError(f"Target ref could not be resolved for type: {inputs}")
            return dispatch("browser_type", {"ref": ref, "text": inputs.get("text", ""), "clear": inputs.get("clear", True)})
        if primitive in {"browser_press", "press"}:
            if not dispatch:
                raise RuntimeError("Browser primitive requires dispatch function")
            return dispatch("browser_press", {"key": inputs["key"]})
        if primitive in {"browser_scroll", "scroll"}:
            if not dispatch:
                raise RuntimeError("Browser primitive requires dispatch function")
            return dispatch("browser_scroll", {"direction": inputs.get("direction", "down")})
        if primitive in {"browser_extract_items", "extract_items"}:
            if not dispatch:
                raise RuntimeError("Browser primitive requires dispatch function")
            return dispatch("browser_extract_items", inputs)

        # 3. Control primitives
        if primitive in {"wait", "sleep"}:
            duration = float(inputs.get("duration", inputs.get("seconds", 0.5)))
            time.sleep(min(10.0, max(0.01, duration)))
            return {"waited_seconds": duration}

        # 4. Fallback dispatch
        if dispatch:
            return dispatch(primitive, inputs)
        raise ValueError(f"Unknown primitive: {primitive}")

    def _find_ref_by_anchor(self, elements: list[dict[str, Any]], anchor: dict[str, Any]) -> str | None:
        anchor_type = anchor.get("type")
        anchor_val = str(anchor.get("value", "")).lower()
        if anchor_type == "testid":
            for el in elements:
                tid = str(el.get("testid", el.get("attributes", {}).get("data-testid", ""))).lower()
                if tid and tid == anchor_val:
                    return el.get("ref")
        elif anchor_type == "name":
            for el in elements:
                n = str(el.get("name", "")).lower()
                if n and n == anchor_val:
                    return el.get("ref")
        elif anchor_type == "role_name":
            role_req, _, label_req = anchor_val.partition(":")
            for el in elements:
                r = str(el.get("role", "")).lower()
                lbl = str(el.get("label", el.get("name", ""))).lower()
                if role_req in r and label_req in lbl:
                    return el.get("ref")
        return None

    # -------------------------------------------------------------------------
    # Condition Verification (Preconditions / Postconditions)
    # -------------------------------------------------------------------------

    def verify_condition(self, condition: str | dict[str, Any], context: dict[str, Any]) -> bool:
        if isinstance(condition, str):
            try:
                cond_dict = json.loads(condition)
            except ValueError:
                cond_dict = {"type": "expression", "expr": condition}
        else:
            cond_dict = condition

        cond_dict = interpolate_variables(cond_dict, context)
        c_type = cond_dict.get("type")

        if c_type == "file_exists":
            target = _resolve_path(cond_dict["path"], cond_dict.get("base_dir"))
            return target.exists()
        if c_type == "file_hash":
            target = _resolve_path(cond_dict["path"], cond_dict.get("base_dir"))
            if not target.exists():
                return False
            actual_hash = hashlib.sha256(target.read_bytes()).hexdigest()
            return actual_hash.lower() == str(cond_dict["expected"]).lower()
        if c_type == "file_contains":
            target = _resolve_path(cond_dict["path"], cond_dict.get("base_dir"))
            if not target.exists():
                return False
            return str(cond_dict["text"]) in target.read_text(encoding="utf-8")
        if c_type == "field_equals":
            actual = _lookup_dotted_path(cond_dict["field"], context)
            return actual == cond_dict["expected"]
        if c_type == "not_empty":
            actual = _lookup_dotted_path(cond_dict["field"], context)
            return bool(actual)
        return True

    def check_conditions(self, conditions: list[str | dict[str, Any]], context: dict[str, Any], stage: str) -> None:
        for cond in conditions:
            if not self.verify_condition(cond, context):
                raise CapabilityDriftError(f"{stage.capitalize()} condition failed: {cond}")

    # -------------------------------------------------------------------------
    # Execution & Composition
    # -------------------------------------------------------------------------

    def execute_capability(
        self,
        capability_or_id: str | OperationalCapability,
        inputs: dict[str, Any],
        *,
        version: str | None = None,
        dispatch: Callable | None = None,
        context: dict[str, Any] | None = None,
        owner: str | None = None,
    ) -> dict[str, Any]:
        """Execute a capability, resolving its dependencies and running deterministic steps."""
        if isinstance(capability_or_id, str):
            cap = self.resolver.resolve(capability_or_id, version or "*")
        else:
            cap = capability_or_id

        if cap.drift_state == "quarantined":
            raise CapabilityDriftError(f"Capability '{cap.id}' is quarantined due to operational drift")

        exec_context: dict[str, Any] = {
            "inputs": dict(inputs),
            "deps": {},
            "steps": {},
            "prev": None,
        }
        if context:
            exec_context.update({k: v for k, v in context.items() if k not in exec_context})

        try:
            # 1. Execute dependencies in topological order
            for dep in cap.dependencies:
                dep_inputs = {}
                for target_field, src_path in dep.input_mappings.items():
                    val = _lookup_dotted_path(src_path, exec_context)
                    if val is not None:
                        dep_inputs[target_field] = val
                    else:
                        dep_inputs[target_field] = interpolate_variables(src_path, exec_context)

                dep_result = self.execute_capability(
                    dep.capability_id,
                    dep_inputs,
                    version=dep.version_constraint,
                    dispatch=dispatch,
                    context=exec_context,
                    owner=owner,
                )
                dep_key = dep.output_alias or dep.capability_id
                exec_context["deps"][dep_key] = dep_result

            # 2. Verify preconditions
            self.check_conditions(cap.preconditions, exec_context, stage="precondition")

            # 3. Execute implementation steps
            steps = cap.implementation.get("steps", [])
            output = None
            for idx, step in enumerate(steps):
                step_id = step.get("id", f"step_{idx}")
                primitive = step.get("primitive") or step.get("tool") or step.get("action")
                step_args = step.get("args") or step.get("inputs") or {}
                step_result = self.execute_primitive(primitive, step_args, dispatch=dispatch, context=exec_context)
                exec_context["steps"][step_id] = step_result
                exec_context["prev"] = step_result
                output = step_result

            # If implementation declares explicit return mapping or output
            if "output" in cap.implementation:
                output = interpolate_variables(cap.implementation["output"], exec_context)
            elif output is None:
                output = {"success": True, "id": cap.id, "version": cap.version}

            exec_context["output"] = output

            # 4. Verify postconditions
            self.check_conditions(cap.postconditions, exec_context, stage="postcondition")

            # 5. Record validation evidence & savings
            evidence = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "status": "success",
                "inputs_sample": sanitize(inputs),
            }
            cap.success_count += 1
            cap.savings["llm_calls_saved"] = cap.savings.get("llm_calls_saved", 0) + 1
            cap.savings["tokens_saved"] = cap.savings.get("tokens_saved", 0) + 1500
            self.registry.register(cap)

            return {
                "success": True,
                "capability_id": cap.id,
                "capability_version": cap.version,
                "output": output,
                "savings": cap.savings,
            }

        except CapabilityDriftError as drift_err:
            self.registry.record_drift(cap.id, str(drift_err), quarantine=True)
            if owner:
                from workstation.reasoning_handoff import needs_reasoning
                handoff = needs_reasoning(
                    self.artifacts,
                    owner,
                    completed_until=exec_context.get("prev"),
                    expected=f"Capability '{cap.id}' postconditions satisfied",
                    observed=str(drift_err),
                    safe_to_resume=False,
                    context={"capability_id": cap.id, "phase": "operational_kernel"},
                )
                return handoff
            raise
