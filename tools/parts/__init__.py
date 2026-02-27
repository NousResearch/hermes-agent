import importlib.util
from pathlib import Path

# Avoid triggering tools/__init__.py by loading directly
_parts_models_path = Path(__file__).parent / "models.py"
spec = importlib.util.spec_from_file_location("parts_models", _parts_models_path)
parts_models = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parts_models)

PartsStore = parts_models.PartsStore
Part = parts_models.Part
SuggestionResult = parts_models.SuggestionResult

_parts_storage_path = Path(__file__).parent / "storage.py"
spec2 = importlib.util.spec_from_file_location("parts_storage", _parts_storage_path)
parts_storage = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(parts_storage)

PartsStorage = parts_storage.PartsStorage

__all__ = ["Part", "SuggestionResult", "PartsStore", "PartsStorage"]
