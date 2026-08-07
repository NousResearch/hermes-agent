from tools.environments.modal_utils import BaseModalExecutionEnvironment, ModalExecStart


class _DummyModalEnv(BaseModalExecutionEnvironment):
    def __init__(self, cwd: str = "/root", timeout: int = 60):
        super().__init__(cwd=cwd, timeout=timeout)

    def _prepare_command(self, command: str):
        return command, None

    def _start_modal_exec(self, prepared):
        _ = prepared
        return ModalExecStart(immediate_result={"output": "", "returncode": 0})

    def _poll_modal_exec(self, handle):
        _ = handle
        return None

    def _cancel_modal_exec(self, handle):
        _ = handle

    def cleanup(self):
        pass


def test_prepare_modal_exec_falls_back_for_blank_cwd():
    env = _DummyModalEnv(cwd="/root")
    prepared = env._prepare_modal_exec("echo hi", cwd="   ", timeout=5)
    assert prepared.cwd == "/root"


def test_prepare_modal_exec_preserves_nonblank_cwd_whitespace():
    env = _DummyModalEnv(cwd="/root")
    prepared = env._prepare_modal_exec("echo hi", cwd=" /tmp/spaced ", timeout=5)
    assert prepared.cwd == " /tmp/spaced "


def test_prepare_modal_exec_uses_default_timeout_when_non_positive():
    env = _DummyModalEnv(timeout=60)
    prepared_zero = env._prepare_modal_exec("echo hi", timeout=0)
    prepared_negative = env._prepare_modal_exec("echo hi", timeout=-3)
    assert prepared_zero.timeout == 60
    assert prepared_negative.timeout == 60


def test_prepare_modal_exec_coerces_string_timeout():
    env = _DummyModalEnv(timeout=60)
    prepared = env._prepare_modal_exec("echo hi", timeout="7")  # type: ignore[arg-type]
    assert prepared.timeout == 7


def test_prepare_modal_exec_invalid_timeout_falls_back():
    env = _DummyModalEnv(timeout=45)
    prepared = env._prepare_modal_exec("echo hi", timeout="bad")  # type: ignore[arg-type]
    assert prepared.timeout == 45
