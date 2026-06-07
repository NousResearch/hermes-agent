"""Tests for draft skill status management."""

import pytest
import tempfile
from pathlib import Path
from datetime import datetime
import yaml


class TestDraftProvenance:
    """Test draft skill provenance metadata."""

    def test_inject_provenance_metadata_draft(self):
        """Test that _inject_provenance_metadata adds draft metadata correctly."""
        from tools.skill_manager_tool import _inject_provenance_metadata
        
        content = """---
name: test-skill
description: A test skill
---

## Instructions

Do something."""

        result = _inject_provenance_metadata(content, is_draft=True)
        
        # Parse the result YAML
        assert result.startswith("---")
        end_match = result.find('\n---\n', 3)
        yaml_str = result[3:end_match]
        fm = yaml.safe_load(yaml_str)
        
        assert fm['author'] == 'agent'
        assert fm['status'] == 'draft'
        assert fm['confirmed_at'] is None
        assert fm['name'] == 'test-skill'  # Original field preserved
        assert fm['description'] == 'A test skill'  # Original field preserved

    def test_inject_provenance_metadata_user(self):
        """Test that user-created skills get author=user without status."""
        from tools.skill_manager_tool import _inject_provenance_metadata
        
        content = """---
name: test-skill
description: A test skill
---

## Instructions

Do something."""

        result = _inject_provenance_metadata(content, is_draft=False)
        
        # Parse the result YAML
        end_match = result.find('\n---\n', 3)
        yaml_str = result[3:end_match]
        fm = yaml.safe_load(yaml_str)
        
        assert fm['author'] == 'user'
        assert 'status' not in fm  # User skills don't get status field
        assert fm['name'] == 'test-skill'

    def test_inject_provenance_preserves_body(self):
        """Test that body content is preserved after provenance injection."""
        from tools.skill_manager_tool import _inject_provenance_metadata
        
        content = """---
name: test
description: Test
---

## Instructions

Line 1
Line 2
Line 3"""

        result = _inject_provenance_metadata(content, is_draft=True)
        
        # Extract body
        end_match = result.find('\n---\n', 3)
        body = result[end_match + 5:]
        
        assert "Line 1" in body
        assert "Line 2" in body
        assert "Line 3" in body


class TestDraftSuppressionContextVar:
    """Test draft suppression context variable."""

    def test_suppress_drafts_default_true(self):
        """Test that draft suppression is enabled by default."""
        from tools.skill_provenance import should_suppress_drafts
        
        assert should_suppress_drafts() is True

    def test_suppress_drafts_can_be_disabled(self):
        """Test that draft suppression can be disabled."""
        from tools.skill_provenance import set_suppress_drafts, reset_suppress_drafts, should_suppress_drafts
        
        token = set_suppress_drafts(False)
        try:
            assert should_suppress_drafts() is False
        finally:
            reset_suppress_drafts(token)
        
        # Should be restored to default
        assert should_suppress_drafts() is True


class TestSkillStatusParsing:
    """Test extraction of skill status from SKILL.md."""

    def test_get_skill_status_draft(self):
        """Test reading draft status from skill file."""
        from tools.skill_provenance import get_skill_status
        
        with tempfile.TemporaryDirectory() as tmpdir:
            skill_file = Path(tmpdir) / "SKILL.md"
            skill_file.write_text("""---
name: test
description: Test
status: draft
author: agent
confirmed_at: null
---

Body""")
            
            assert get_skill_status(skill_file) == 'draft'

    def test_get_skill_status_confirmed(self):
        """Test reading confirmed status from skill file."""
        from tools.skill_provenance import get_skill_status
        
        with tempfile.TemporaryDirectory() as tmpdir:
            skill_file = Path(tmpdir) / "SKILL.md"
            skill_file.write_text("""---
name: test
description: Test
status: confirmed
author: user
confirmed_at: "2024-01-01T00:00:00Z"
---

Body""")
            
            assert get_skill_status(skill_file) == 'confirmed'

    def test_get_skill_status_pending(self):
        """Test reading pending status from skill file."""
        from tools.skill_provenance import get_skill_status
        
        with tempfile.TemporaryDirectory() as tmpdir:
            skill_file = Path(tmpdir) / "SKILL.md"
            skill_file.write_text("""---
name: test
description: Test
status: pending
author: agent
---

Body""")
            
            assert get_skill_status(skill_file) == 'pending'

    def test_get_skill_status_legacy(self):
        """Test that skills without status field return None."""
        from tools.skill_provenance import get_skill_status
        
        with tempfile.TemporaryDirectory() as tmpdir:
            skill_file = Path(tmpdir) / "SKILL.md"
            skill_file.write_text("""---
name: test
description: Test
---

Body""")
            
            assert get_skill_status(skill_file) is None

    def test_get_skill_author(self):
        """Test reading author field from skill file."""
        from tools.skill_provenance import get_skill_author
        
        with tempfile.TemporaryDirectory() as tmpdir:
            skill_file = Path(tmpdir) / "SKILL.md"
            skill_file.write_text("""---
name: test
description: Test
author: agent
---

Body""")
            
            assert get_skill_author(skill_file) == 'agent'

    def test_is_draft_skill(self):
        """Test is_draft_skill predicate."""
        from tools.skill_provenance import is_draft_skill
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Draft skill
            draft_file = Path(tmpdir) / "draft.md"
            draft_file.write_text("""---
name: test
description: Test
status: draft
---

Body""")
            assert is_draft_skill(draft_file) is True
            
            # Non-draft skill
            confirmed_file = Path(tmpdir) / "confirmed.md"
            confirmed_file.write_text("""---
name: test
description: Test
status: confirmed
---

Body""")
            assert is_draft_skill(confirmed_file) is False

    def test_is_confirmed_or_legacy(self):
        """Test is_confirmed_or_legacy predicate."""
        from tools.skill_provenance import is_confirmed_or_legacy
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Confirmed skill
            confirmed_file = Path(tmpdir) / "confirmed.md"
            confirmed_file.write_text("""---
name: test
description: Test
status: confirmed
---

Body""")
            assert is_confirmed_or_legacy(confirmed_file) is True
            
            # Legacy skill (no status)
            legacy_file = Path(tmpdir) / "legacy.md"
            legacy_file.write_text("""---
name: test
description: Test
---

Body""")
            assert is_confirmed_or_legacy(legacy_file) is True
            
            # Draft skill
            draft_file = Path(tmpdir) / "draft.md"
            draft_file.write_text("""---
name: test
description: Test
status: draft
---

Body""")
            assert is_confirmed_or_legacy(draft_file) is False


class TestCreateDraftSkillFromBackgroundReview:
    """Test that skills created from background_review get draft status."""

    def test_create_skill_from_background_review_is_draft(self):
        """Test that _create_skill marks skills as draft when from background_review."""
        from tools.skill_manager_tool import skill_manage
        from tools.skill_provenance import set_current_write_origin, reset_current_write_origin, BACKGROUND_REVIEW, get_skill_status
        from hermes_constants import get_hermes_home
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Temporarily override HERMES_HOME for this test
            import tools.skill_manager_tool as smt
            original_skills_dir = smt.SKILLS_DIR
            smt.SKILLS_DIR = Path(tmpdir) / "skills"
            smt.SKILLS_DIR.mkdir(exist_ok=True)
            
            try:
                token = set_current_write_origin(BACKGROUND_REVIEW)
                try:
                    result = skill_manage(
                        action="create",
                        name="bg-skill",
                        content="""---
name: bg-skill
description: Background created skill
---

Instructions."""
                    )
                finally:
                    reset_current_write_origin(token)
                
                assert "success" in result
                
                # Check the created skill has draft status
                skill_file = smt.SKILLS_DIR / "bg-skill" / "SKILL.md"
                assert skill_file.exists()
                
                from tools.skill_provenance import get_skill_status, get_skill_author
                assert get_skill_status(skill_file) == 'draft'
                assert get_skill_author(skill_file) == 'agent'
            finally:
                smt.SKILLS_DIR = original_skills_dir

    def test_create_skill_from_foreground_is_user(self):
        """Test that user-created skills don't get draft status."""
        from tools.skill_manager_tool import skill_manage
        from tools.skill_provenance import set_current_write_origin, reset_current_write_origin, get_skill_status, get_skill_author
        
        with tempfile.TemporaryDirectory() as tmpdir:
            import tools.skill_manager_tool as smt
            original_skills_dir = smt.SKILLS_DIR
            smt.SKILLS_DIR = Path(tmpdir) / "skills"
            smt.SKILLS_DIR.mkdir(exist_ok=True)
            
            try:
                # Set to foreground (default)
                token = set_current_write_origin("foreground")
                try:
                    result = skill_manage(
                        action="create",
                        name="user-skill",
                        content="""---
name: user-skill
description: User created skill
---

Instructions."""
                    )
                finally:
                    reset_current_write_origin(token)
                
                assert "success" in result
                
                # Check the skill has author=user and no status
                skill_file = smt.SKILLS_DIR / "user-skill" / "SKILL.md"
                assert skill_file.exists()
                
                from tools.skill_provenance import get_skill_status, get_skill_author
                assert get_skill_status(skill_file) is None  # No status for user skills
                assert get_skill_author(skill_file) == 'user'
            finally:
                smt.SKILLS_DIR = original_skills_dir
