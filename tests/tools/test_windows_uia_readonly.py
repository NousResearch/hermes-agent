"""Tests for Windows UIA read-only live enumeration."""

from __future__ import annotations


class FakeInfo:
    def __init__(
        self,
        *,
        name="",
        automation_id="",
        control_type="",
        class_name="",
        process_id=100,
        handle=0,
        rich_text="",
    ) -> None:
        self.name = name
        self.automation_id = automation_id
        self.control_type = control_type
        self.class_name = class_name
        self.process_id = process_id
        self.handle = handle
        self.rich_text = rich_text


class FakeRect:
    def __init__(self, left=1, top=2, right=101, bottom=52):
        self.left = left
        self.top = top
        self.right = right
        self.bottom = bottom


class FakeElement:
    def __init__(self, info, *, password=False, enabled=True, visible=True, children=None, patterns=None):
        self.element_info = info
        self._password = password
        self._enabled = enabled
        self._visible = visible
        self._children = children or []
        self._patterns = patterns or []

    def rectangle(self):
        return FakeRect()

    def is_enabled(self):
        return self._enabled

    def is_visible(self):
        return self._visible

    def children(self):
        return list(self._children)

    def descendants(self):
        out = []
        for child in self._children:
            out.append(child)
            out.extend(child.descendants())
        return out

    def friendly_class_name(self):
        return self.element_info.control_type

    def get_supported_patterns(self):
        return tuple(self._patterns)


class FakeDesktop:
    def __init__(self, windows):
        self._windows = windows

    def windows(self):
        return list(self._windows)


def test_uia_readonly_list_windows_uses_injected_desktop_without_mutation():
    from tools.computer_use.windows_uia_readonly import WindowsUiaReadOnlyBackend

    root = FakeElement(FakeInfo(name="Demo", automation_id="root", control_type="Window", class_name="DemoWin", handle=123))
    backend = WindowsUiaReadOnlyBackend(platform="win32", desktop_factory=lambda: FakeDesktop([root]))

    result = backend.list_windows()

    assert result["available"] is True
    assert result["read_only"] is True
    assert result["mutation_allowed"] is False
    assert result["would_execute_native_input"] is False
    assert result["windows"][0]["name"] == "Demo"
    assert result["windows"][0]["hwnd"] == 123


def test_uia_readonly_snapshot_tree_redacts_password_and_limits_depth():
    from tools.computer_use.windows_uia_readonly import WindowsUiaReadOnlyBackend

    password = FakeElement(
        FakeInfo(name="super-secret", automation_id="pwd", control_type="Edit", class_name="PasswordBox"),
        password=True,
    )
    button = FakeElement(
        FakeInfo(name="OK", automation_id="ok", control_type="Button", class_name="Button"),
        patterns=("InvokePattern",),
    )
    root = FakeElement(
        FakeInfo(name="Login", automation_id="root", control_type="Window", class_name="LoginWin", handle=456),
        children=[password, button],
    )
    backend = WindowsUiaReadOnlyBackend(platform="win32", desktop_factory=lambda: FakeDesktop([root]))

    result = backend.snapshot_tree(max_depth=2, max_elements=10)

    assert result["available"] is True
    assert result["read_only"] is True
    names = [item["name"] for item in result["elements"]]
    assert "Login" in names
    assert "OK" in names
    assert "super-secret" not in names
    password_items = [item for item in result["elements"] if item["automation_id"] == "pwd"]
    assert password_items[0]["is_password"] is True
    assert password_items[0]["name"] == "<redacted password field>"


def test_uia_readonly_element_capabilities_matches_selector():
    from tools.computer_use.windows_uia_readonly import WindowsUiaReadOnlyBackend

    button = FakeElement(
        FakeInfo(name="Submit", automation_id="submit", control_type="Button", class_name="Button"),
        patterns=("InvokePattern", "LegacyIAccessiblePattern"),
    )
    root = FakeElement(FakeInfo(name="Form", control_type="Window"), children=[button])
    backend = WindowsUiaReadOnlyBackend(platform="win32", desktop_factory=lambda: FakeDesktop([root]))

    result = backend.element_capabilities(selector={"automation_id": "submit"})

    assert result["available"] is True
    assert result["matched"] is True
    assert result["capabilities"]["automation_id"] == "submit"
    assert "InvokePattern" in result["capabilities"]["supported_patterns"]
    assert result["would_execute_native_input"] is False


def test_uia_readonly_status_reports_available_with_injected_factory_only():
    from tools.computer_use.windows_uia_readonly import WindowsUiaReadOnlyBackend

    backend = WindowsUiaReadOnlyBackend(platform="win32", desktop_factory=lambda: FakeDesktop([]))
    caps = backend.capabilities()

    assert caps.available is True
    assert caps.read_only is True
    assert caps.mutation_allowed is False
    assert caps.input_fallback_enabled is False
