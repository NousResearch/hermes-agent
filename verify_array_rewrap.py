"""Verifica el fix contra validate_deferred_call_args (el camino real) + no-regresion."""
import sys
sys.path.insert(0, "/Users/braingram/.hermes/hermes-agent")
import model_tools  # dispara discovery: sin esto el registry esta vacio y todo falla abierto
from tools.tool_search_validation import validate_deferred_call_args as V

TODO = {"id": "t1", "content": "x", "status": "in_progress"}
DEBERIA_PASAR = {
    "correcto":          {"todos": [TODO]},
    "env en cada item":  {"todos": [{"item": TODO}]},
    "env en la lista":   {"todos": {"item": [TODO]}},
    "env doble":         {"todos": {"item": {"item": [TODO]}}},
    "env single dict":   {"todos": {"item": TODO}},   # el que fallaba en vivo
    "lista doble":       {"todos": [[TODO]]},
    "sin key merge":     {"todos": [TODO], "merge": True},
}
NO_DEBE_PASAR = {
    "falta id":          {"todos": [{"content": "x", "status": "in_progress"}]},
    "status invalido":   {"todos": [{"id": "t1", "content": "x", "status": "nope"}]},
    "todos no-array":    {"todos": "basura"},
}
fails = []
for label, args in DEBERIA_PASAR.items():
    err = V("todo_list", args)
    ok = err is None
    print(f"{'PASA ' if ok else 'FALLA'} {label:18} {'' if ok else err[:110]}")
    if not ok: fails.append(label)
for label, args in NO_DEBE_PASAR.items():
    err = V("todo_list", args)
    ok = err is not None
    print(f"{'OK   ' if ok else 'FUGA '} {label:18} {'' if ok else 'ACEPTO MALFORMADO'}")
    if not ok: fails.append(label)
print()
print("RESULTADO:", "TODO OK" if not fails else f"FALLOS: {fails}")
sys.exit(1 if fails else 0)
