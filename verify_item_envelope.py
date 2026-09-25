"""Check: el fix item-envelope acepta las formas reales del bug; el control falla."""
import sys
sys.path.insert(0, '/Users/braingram/.hermes/hermes-agent')

import model_tools          # noqa: F401  (popula el registro)
import tools.todo_tool      # noqa: F401
from tools.tool_search_validation import validate_deferred_call_args

CASES = [
    ("array simple",
     {"todos": [{"id": "a", "content": "primera", "status": "pending"},
                {"id": "b", "content": "segunda", "status": "in_progress"}], "merge": False}),
    ("array anidado en dict",
     {"todos": {"item": {"id": "p1", "content": "probe", "status": "pending"}}, "merge": "false"}),
    ("merge string + item anidado",
     {"merge": "false", "todos": {"item": [{"id": "1", "content": "x", "status": "pending"}]}}),
    ("no-regresion (payload sano)",
     {"todos": [{"id": "x", "content": "sana", "status": "pending"}], "merge": False}),
]

ok = True
for label, args in CASES:
    err = validate_deferred_call_args("todo_list", args)
    ok &= err is None
    print(f"[{'ACEPTADO' if err is None else 'RECHAZADO':8}] {label}")
    if err:
        print(f"           {str(err).splitlines()[0][:100]}")

print()
print("RESULTADO:", "TODAS OK" if ok else "HAY FALLOS")
sys.exit(0 if ok else 1)
