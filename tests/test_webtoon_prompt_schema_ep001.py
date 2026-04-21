from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import yaml

EP001_DIR = Path("docs/plans/orbi-trend-webnovel-webtoon-20260417/webtoon/ep001")
PANEL_PROMPTS = EP001_DIR / "panel_prompts.yaml"
SCROLL_PLAN = EP001_DIR / "scroll_plan.yaml"
REVIEW_PATH = EP001_DIR / "character_consistency_review.yaml"
RENDERER_PATH = EP001_DIR / "render_webtoon_fal_v3.py"
EXPECTED_REFERENCE_SUMMARIES = {
    "p06": {
        "used_labels_prefix": [
            "location_master",
            "mother.doorway_fullbody",
            "mother.portrait_front",
        ],
        "required_used_labels": {
            "protagonist.desk_halfbody",
            "protagonist.last_strong_ref.seated_medium",
        },
        "required_skips": {
            ("prev_panel", "different_visible_character_set"),
            ("mother.last_strong_ref.standing_full", "missing_strong_ref"),
            ("mother.last_strong_ref.portrait", "missing_strong_ref"),
        },
    },
    "p08": {
        "used_labels_prefix": [
            "location_master",
            "mother.doorway_fullbody",
            "mother.portrait_front",
        ],
        "required_used_labels": {
            "protagonist.portrait_3q",
            "protagonist.desk_halfbody",
        },
        "allowed_mother_strong_labels": {
            "mother.last_strong_ref.portrait",
            "mother.last_strong_ref.standing_full",
        },
        "required_skips": {
            ("prev_panel", "continuity_reset_shot"),
            ("scene_ref:mother+protagonist", "continuity_reset_shot"),
        },
    },
}


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _load_renderer_module():
    spec = importlib.util.spec_from_file_location("render_webtoon_fal_v3_ep001", RENDERER_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _panel_map(panel_data: dict) -> dict[str, dict]:
    return {panel["panel_id"]: panel for panel in panel_data["panels"]}


def test_schema_structure_and_character_sheets() -> None:
    panel_data = _load_yaml(PANEL_PROMPTS)
    assert panel_data["prompt_schema_version"] == 2
    assert "positive" in panel_data["style_anchor"]
    assert "negative" in panel_data["style_anchor"]

    defaults = panel_data["reference_policy_defaults"]
    assert {
        "always_include",
        "use_previous_panel_as_primary",
        "continuity_reset_shots",
        "eligible_strong_ref_face_visibility",
        "strong_ref_view_classes",
    } <= set(defaults)

    for character in ("protagonist", "mother"):
        sheet = panel_data["character_sheets"][character]
        assert {
            "identity_core",
            "visual_invariants",
            "expression_baseline",
            "negative_drift_cues",
            "anchor_views",
        } <= set(sheet)
        assert len(sheet["anchor_views"]) >= 2
        for anchor_view in sheet["anchor_views"]:
            assert {"view_class", "suitable_face_visibility", "suitable_shots"} <= set(anchor_view)


def test_per_panel_continuity_contract_and_reset_shots() -> None:
    panel_data = _load_yaml(PANEL_PROMPTS)
    panels = _panel_map(panel_data)
    for panel in panels.values():
        assert panel["visible_characters"]
        if panel["visible_characters"]:
            assert "identity_focus" in panel
            assert "face_visibility" in panel
            assert "continuity_mode" in panel
            assert "must_keep" in panel
            assert "reference_injection" in panel

    for panel_id in ("p03", "p05", "p07"):
        assert panels[panel_id]["continuity_mode"] == "continuity_reset"

    assert panels["p06"]["identity_focus"]["mother"] == "high"
    assert panels["p08"]["identity_focus"]["mother"] == "high"


def test_prompt_fragment_builders_expose_structured_parts() -> None:
    renderer = _load_renderer_module()
    panel_data, scroll = renderer.load_episode_data()
    panels = renderer.panel_map_from_data(panel_data)
    blocks = renderer.scroll_block_map(scroll)

    p06_parts = renderer.build_prompt_parts(panel_data, panels["p06"], blocks[panels["p06"]["block_id"]])
    assert set(p06_parts) == {"style", "scene", "identity_lock", "negative_lock"}
    assert "beige homewear" in p06_parts["identity_lock"]
    assert "restrained sharp eyes" in p06_parts["identity_lock"]
    assert "black zip hoodie" in p06_parts["identity_lock"]
    assert "hoodie silhouette" in p06_parts["identity_lock"]
    assert "mouth openness" not in p06_parts["identity_lock"]
    assert "no bob cut" in p06_parts["negative_lock"]


def test_reference_routing_is_shot_aware_for_p06_and_p08() -> None:
    renderer = _load_renderer_module()
    manifest = renderer.render_episode(dry_run=True, manifest_only=True)
    panels = {panel["panel_id"]: panel for panel in manifest["panels"]}

    p06_refs = panels["p06"]["reference_strategy"]
    p08_refs = panels["p08"]["reference_strategy"]
    p06_used_labels = [ref["label"] for ref in p06_refs["used_refs"]]
    p08_used_labels = [ref["label"] for ref in p08_refs["used_refs"]]

    p06_expected = EXPECTED_REFERENCE_SUMMARIES["p06"]
    assert p06_used_labels[:3] == p06_expected["used_labels_prefix"]
    assert p06_expected["required_used_labels"] <= set(p06_used_labels)
    assert p06_expected["required_skips"] <= {
        (ref["label"], ref["reason"]) for ref in p06_refs["skipped_refs"]
    }

    p08_expected = EXPECTED_REFERENCE_SUMMARIES["p08"]
    assert p08_used_labels[:3] == p08_expected["used_labels_prefix"]
    assert p08_expected["required_used_labels"] <= set(p08_used_labels)
    assert p08_expected["required_skips"] <= {
        (ref["label"], ref["reason"]) for ref in p08_refs["skipped_refs"]
    }
    assert not any(ref["label"] == "prev_panel" for ref in p08_refs["used_refs"])
    assert any(
        ref["label"] in p08_expected["allowed_mother_strong_labels"] and ref.get("source_panel") == "p06"
        for ref in p08_refs["used_refs"]
    ) or any(
        ref["label"] in p08_expected["allowed_mother_strong_labels"]
        and ref["reason"] == "duplicate_url_preserved_by_precedence"
        for ref in p08_refs["skipped_refs"]
    )


def test_manifest_observability_and_strong_ref_updates() -> None:
    renderer = _load_renderer_module()
    manifest = renderer.render_episode(dry_run=True, manifest_only=True)
    p08 = next(panel for panel in manifest["panels"] if panel["panel_id"] == "p08")
    strategy = p08["reference_strategy"]

    assert p08["visible_characters"] == ["protagonist", "mother"]
    assert {"style", "scene", "identity_lock", "negative_lock"} == set(p08["prompt_parts"])
    assert {"continuity_mode", "used_refs", "skipped_refs", "strong_ref_updates"} == set(strategy)
    assert any(update["character"] == "mother" and update["view_class"] == "portrait" and update["accepted"] for update in strategy["strong_ref_updates"])
    assert any(update["character"] == "mother" and update["view_class"] == "standing_full" and not update["accepted"] for update in strategy["strong_ref_updates"])
    assert manifest["dry_run"] is True
    assert manifest["manifest_only"] is True
    assert manifest["longscroll"] is None


def test_review_yaml_has_one_entry_per_panel_and_required_links() -> None:
    panel_data = _load_yaml(PANEL_PROMPTS)
    review_data = _load_yaml(REVIEW_PATH)
    review_items = {item["panel_id"]: item for item in review_data["review_items"]}

    assert len(review_items) == len(panel_data["panels"])
    assert {"face_match", "hair_match", "wardrobe_match", "role_readability", "notes"} <= set(review_data["review_rubric"])
    assert "mother.anchor:doorway_fullbody" in review_items["p06"]["continuity_links"]["compare_against"]["mother"]
    assert "p06" in review_items["p08"]["continuity_links"]["compare_against"]["mother"]
    assert "mother.anchor:portrait_front" in review_items["p08"]["continuity_links"]["compare_against"]["mother"]


def test_manifest_file_written_by_dry_run_matches_runtime_result() -> None:
    renderer = _load_renderer_module()
    manifest = renderer.render_episode(dry_run=True, manifest_only=True)
    written = json.loads((EP001_DIR / "generated_fal_manifest_v3.json").read_text(encoding="utf-8"))
    assert written["mode"] == manifest["mode"]
    assert written["panels"][-1]["panel_id"] == "p08"
