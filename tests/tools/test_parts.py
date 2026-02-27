"""
Tests for Dynamic Parts feature (Issue #90)
Tests for Part creation, storage, retrieval, prediction evaluation, and archiving.
"""

import pytest
import json
import tempfile
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Import parts modules directly to avoid fal_client dependency in tools/__init__.py
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import importlib.util

# Load models.py directly
models_path = Path(__file__).parent.parent.parent / "tools" / "parts" / "models.py"
spec = importlib.util.spec_from_file_location("parts_models", models_path)
parts_models = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parts_models)

Part = parts_models.Part
SuggestionResult = parts_models.SuggestionResult
OriginatingEvent = parts_models.OriginatingEvent
PartsStore = parts_models.PartsStore

# Load runtime.py directly
runtime_path = Path(__file__).parent.parent.parent / "tools" / "parts" / "runtime.py"
spec = importlib.util.spec_from_file_location("parts_runtime", runtime_path)
parts_runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parts_runtime)

PartsRuntime = parts_runtime.PartsRuntime
PartsContext = parts_runtime.PartsContext


class TestSuggestionResult:
    """Test SuggestionResult data class."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        suggestion = SuggestionResult(
            predicted_result="User will be happy",
            predicted_result_confidence="80%",
            predicted_result_timeframe_seconds=3600,
            your_suggestion="Send a friendly message",
            timestamp=datetime.now().isoformat()
        )
        result = suggestion.to_dict()
        assert result["predicted_result"] == "User will be happy"
        assert result["predicted_result_confidence"] == "80%"

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "predicted_result": "User will be happy",
            "predicted_result_confidence": "80%",
            "predicted_result_timeframe_seconds": 3600,
            "your_suggestion": "Send a friendly message",
            "timestamp": datetime.now().isoformat(),
            "result": None
        }
        suggestion = SuggestionResult.from_dict(data)
        assert suggestion.predicted_result == "User will be happy"
        assert suggestion.your_suggestion == "Send a friendly message"


class TestOriginatingEvent:
    """Test OriginatingEvent data class."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        event = OriginatingEvent(
            timestamp=datetime.now().isoformat(),
            result="User complained about the response"
        )
        result = event.to_dict()
        assert "timestamp" in result
        assert result["result"] == "User complained about the response"


class TestPart:
    """Test Part data class."""

    def test_create_part_minimal(self):
        """Test creating a part with minimal required fields."""
        part = Part(
            name="Test Part",
            description="A test part for unit testing"
        )
        assert part.name == "Test Part"
        assert part.description == "A test part for unit testing"
        assert part.id is not None
        assert part.archived is False
        assert len(part.triggers) == 0

    def test_create_part_full(self):
        """Test creating a part with all fields."""
        part = Part(
            name="Fear of Dogs",
            description="A protective part that holds trauma associated with dogs.",
            emotion="Terror",
            intensity="High",
            personality="Stubborn and alarmist",
            triggers=["Seeing a dog", "Hearing barking"],
            phrases=["It's not worth the risk!", "What if the leash breaks?"],
            core_part=True
        )
        assert part.emotion == "Terror"
        assert len(part.triggers) == 2
        assert part.core_part is True

    def test_part_to_dict(self):
        """Test conversion to dictionary."""
        part = Part(
            name="Test Part",
            description="Test description",
            triggers=["trigger1", "trigger2"]
        )
        result = part.to_dict()
        assert result["name"] == "Test Part"
        assert result["description"] == "Test description"
        assert result["triggers"] == ["trigger1", "trigger2"]
        assert "id" in result
        assert "created_at" in result

    def test_part_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "id": "test-id",
            "name": "Test Part",
            "description": "Test description",
            "triggers": ["trigger1"],
            "suggestions_and_results": [],
            "archived": False,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        part = Part.from_dict(data)
        assert part.id == "test-id"
        assert part.name == "Test Part"

    def test_part_update(self):
        """Test updating part fields."""
        part = Part(
            name="Original Name",
            description="Original description"
        )
        original_updated_at = part.updated_at
        part.update(description="Updated description", emotion="Happy")
        assert part.description == "Updated description"
        assert part.emotion == "Happy"
        assert part.updated_at != original_updated_at

    def test_add_suggestion_result(self):
        """Test adding a suggestion result."""
        part = Part(name="Test", description="Test")
        suggestion = SuggestionResult(
            predicted_result="Something good",
            predicted_result_confidence="70%",
            predicted_result_timeframe_seconds=60,
            your_suggestion="Do something",
            timestamp=datetime.now().isoformat()
        )
        part.add_suggestion_result(suggestion)
        assert len(part.suggestions_and_results) == 1
        assert part.suggestions_and_results[0].predicted_result == "Something good"

    def test_evaluate_prediction(self):
        """Test evaluating a pending prediction."""
        part = Part(name="Test", description="Test")
        suggestion = SuggestionResult(
            predicted_result="User will reply",
            predicted_result_confidence="80%",
            predicted_result_timeframe_seconds=60,
            your_suggestion="Wait for reply",
            timestamp=datetime.now().isoformat(),
            result=None
        )
        part.add_suggestion_result(suggestion)
        assert part.needs_evaluation is False

        # Evaluate the prediction
        result = part.evaluate_prediction("User replied positively")
        assert result is True
        assert part.suggestions_and_results[0].result == "User replied positively"
        assert part.needs_evaluation is False

    def test_evaluate_prediction_no_pending(self):
        """Test evaluating when no pending prediction exists."""
        part = Part(name="Test", description="Test")
        result = part.evaluate_prediction("Some result")
        assert result is False


class TestPartsStore:
    """Test PartsStore functionality."""

    @pytest.fixture
    def temp_storage(self):
        """Create a temporary storage path."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            path = Path(f.name)
        yield path
        if path.exists():
            path.unlink()

    @pytest.fixture
    def empty_store(self, temp_storage):
        """Create an empty store."""
        return PartsStore(storage_path=temp_storage)

    @pytest.fixture
    def populated_store(self, temp_storage):
        """Create a store with some parts."""
        store = PartsStore(storage_path=temp_storage)

        # Add some test parts
        part1 = Part(
            name="Part 1",
            description="First test part",
            triggers=["test", "example"]
        )
        part2 = Part(
            name="Part 2",
            description="Second test part",
            emotion="Neutral"
        )
        part3 = Part(
            name="Part 3",
            description="Archived part",
            archived=True
        )

        store.create(part1)
        store.create(part2)
        store.create(part3)

        return store

    def test_create_part(self, empty_store):
        """Test creating a new part."""
        part = Part(name="New Part", description="A new part")
        created = empty_store.create(part)
        assert created.name == "New Part"
        assert created.id in empty_store._parts

    def test_get_part(self, populated_store):
        """Test retrieving a part by ID."""
        parts = populated_store.list_all()
        part = parts[0]
        retrieved = populated_store.get(part.id)
        assert retrieved is not None
        assert retrieved.name == part.name

    def test_get_by_name(self, populated_store):
        """Test retrieving a part by name."""
        retrieved = populated_store.get_by_name("Part 1")
        assert retrieved is not None
        assert retrieved.description == "First test part"

        # Test case-insensitive search
        retrieved = populated_store.get_by_name("part 1")
        assert retrieved is not None

    def test_list_all_active(self, populated_store):
        """Test listing all active (non-archived) parts."""
        parts = populated_store.list_all(include_archived=False)
        assert len(parts) == 2
        assert all(not p.archived for p in parts)

    def test_list_all_including_archived(self, populated_store):
        """Test listing all parts including archived."""
        parts = populated_store.list_all(include_archived=True)
        assert len(parts) == 3

    def test_update_part(self, populated_store):
        """Test updating a part."""
        parts = populated_store.list_all()
        part = parts[0]
        updated = populated_store.update(part.id, description="Updated description")
        assert updated is not None
        assert updated.description == "Updated description"

    def test_archive_part(self, populated_store):
        """Test archiving a part."""
        parts = populated_store.list_all()
        part = parts[0]
        archived = populated_store.archive(part.id)
        assert archived is not None
        assert archived.archived is True

        # Should not appear in active list
        active_parts = populated_store.list_all(include_archived=False)
        assert part.id not in [p.id for p in active_parts]

    def test_unarchive_part(self, populated_store):
        """Test unarchiving a part."""
        # First get an archived part
        all_parts = populated_store.list_all(include_archived=True)
        archived_part = [p for p in all_parts if p.archived][0]

        unarchived = populated_store.unarchive(archived_part.id)
        assert unarchived is not None
        assert unarchived.archived is False

    def test_delete_part(self, populated_store):
        """Test deleting a part."""
        parts = populated_store.list_all()
        part = parts[0]
        result = populated_store.delete(part.id)
        assert result is True
        assert populated_store.get(part.id) is None

    def test_get_due_evaluations(self, temp_storage):
        """Test retrieving parts with due evaluations."""
        store = PartsStore(storage_path=temp_storage)

        # Create a part with an old suggestion
        old_time = (datetime.now() - timedelta(seconds=100)).isoformat()
        part = Part(name="Test", description="Test")
        suggestion = SuggestionResult(
            predicted_result="Something",
            predicted_result_confidence="50%",
            predicted_result_timeframe_seconds=60,  # 60 seconds
            your_suggestion="Do something",
            timestamp=old_time,
            result=None
        )
        part.add_suggestion_result(suggestion)
        store.create(part)

        # Get due evaluations
        due_parts = store.get_due_evaluations()
        assert len(due_parts) == 1
        assert due_parts[0].id == part.id

    def test_search_by_trigger(self, populated_store):
        """Test searching parts by trigger."""
        results = populated_store.search_by_trigger("test")
        assert len(results) > 0
        assert any("test" in str(t).lower() for p in results for t in p.triggers)

    def test_get_stats(self, populated_store):
        """Test getting store statistics."""
        stats = populated_store.get_stats()
        assert stats["total_parts"] == 3
        assert stats["active_parts"] == 2
        assert stats["archived_parts"] == 1

    def test_persistence(self, temp_storage):
        """Test that parts persist across store instances."""
        # Create and save a part
        store1 = PartsStore(storage_path=temp_storage)
        part = Part(name="Persistent Part", description="Should persist")
        store1.create(part)

        # Create a new store instance and verify the part exists
        store2 = PartsStore(storage_path=temp_storage)
        retrieved = store2.get_by_name("Persistent Part")
        assert retrieved is not None
        assert retrieved.description == "Should persist"


class TestPartsRuntime:
    """Test PartsRuntime functionality."""

    @pytest.fixture
    def temp_storage(self):
        """Create a temporary storage path."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            path = Path(f.name)
        yield path
        if path.exists():
            path.unlink()

    @pytest.fixture
    def runtime(self, temp_storage):
        """Create a PartsRuntime instance."""
        store = PartsStore(storage_path=temp_storage)
        return PartsRuntime(storage=store)

    def test_process_turn_when_disabled(self, runtime):
        """Test that disabled runtime returns empty context."""
        runtime.enabled = False
        context = runtime.process_turn("test context", "hello")
        assert isinstance(context, PartsContext)
        assert len(context.active_bids) == 0
        assert len(context.due_evaluations) == 0

    def test_process_turn_basic(self, runtime, temp_storage):
        """Test basic turn processing."""
        # Add a part with a trigger
        store = PartsStore(storage_path=temp_storage)
        part = Part(
            name="Test Part",
            description="Test",
            triggers=["hello"]
        )
        store.create(part)

        context = runtime.process_turn("test context", "hello world")
        assert isinstance(context, PartsContext)

    def test_get_system_prompt_addition_empty(self, runtime):
        """Test system prompt addition with no active parts."""
        empty_context = PartsContext(
            active_bids=[],
            due_evaluations=[],
            relevant_parts=[],
            persona_decision=None
        )
        addition = runtime.get_system_prompt_addition(empty_context)
        assert addition == ""

    def test_get_system_prompt_addition_with_bids(self, runtime):
        """Test system prompt addition with active bids."""
        context = PartsContext(
            active_bids=[{
                "part_name": "Test Part",
                "recommendation": "Do something",
                "triggers": ["test"],
                "urgency": 0.8
            }],
            due_evaluations=[],
            relevant_parts=[],
            persona_decision=None
        )
        addition = runtime.get_system_prompt_addition(context)
        assert "Test Part" in addition
        assert "Do something" in addition
