"""Build the handler-frame shim for the acceptance tests: a real module file whose
path ends in gateway/slash_commands.py, exposing _handle_approvals_command that
stamps the grant and performs the policy write. Returns the module so tests can
call its handler directly — the writer's frame check sees the REAL chain."""
import importlib.util
import os
import tempfile


def build_handler_module(action: str = "enroll", value: str = '["nagatha"]'):
    tmpmod = tempfile.mkdtemp(prefix="gateway-sim-") + "/gateway"
    os.makedirs(tmpmod, exist_ok=True)
    handler_path = os.path.join(tmpmod, "slash_commands.py")
    with open(handler_path, "w") as f:
        f.write(f'''
def _handle_approvals_command():
    """Handler-shaped frame: stamp the grant, perform the policy write."""
    from hermes_cli.config import set_config_value
    from tools.approval_context import grant_operator_policy_write, reset_operator_policy_write
    token = grant_operator_policy_write()
    try:
        set_config_value("approvals.trusted_execute_code_profiles", {value!r})
    finally:
        reset_operator_policy_write(token)
    return True
''')
    # Unique module name; its __file__ ends with gateway/slash_commands.py so the
    # writer's endswith check matches — exactly the real handler's signature.
    name = f"gateway_sim_handler_{abs(hash(handler_path))}"
    spec = importlib.util.spec_from_file_location(name, handler_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
