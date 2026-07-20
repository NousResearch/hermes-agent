"""Tests for shared group control schema helpers."""

from tools.group_control_schema import build_group_control_properties


def test_build_group_control_properties_includes_shared_fields():
    properties = build_group_control_properties(
        platform_label="QQ",
        target_description="QQ target",
        target_group_description="QQ target group",
        file_path_description="QQ attachment path",
    )

    assert properties["target"]["description"] == "QQ target"
    assert properties["target_group"]["description"] == "QQ target group"
    assert properties["file_path"]["description"] == "QQ attachment path"
    assert properties["user_id"]["description"] == "QQ user id for profile lookup or moderation."
    assert properties["group_name"]["description"] == "Optional human-friendly QQ group label."
    assert properties["delivery_target"]["description"] == "Explicit delivery target for deliver_report."


def test_build_group_control_properties_merges_extra_fields():
    properties = build_group_control_properties(
        platform_label="QQ",
        target_description="QQ target",
        target_group_description="QQ target group",
        file_path_description="QQ attachment path",
        extra_properties={"busid": {"type": "integer", "description": "NapCat busid"}},
    )

    assert properties["busid"] == {"type": "integer", "description": "NapCat busid"}
