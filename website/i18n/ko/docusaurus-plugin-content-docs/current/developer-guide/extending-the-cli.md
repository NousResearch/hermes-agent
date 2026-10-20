---
sidebar_position: 8
title: "CLI 확장하기"
description: "사용자 지정 위젯, 키 바인딩, 레이아웃 변경으로 Hermes TUI를 확장하는 래퍼 CLI 빌드"
---

# CLI 확장하기

Hermes는 `HermesCLI`에 보호된 확장 훅을 제공하므로, 래퍼 CLI가 1000줄이 넘는 `run()` 메서드를 재정의하지 않고도 위젯, 키 바인딩, 레이아웃 사용자 지정을 추가할 수 있습니다. 이를 통해 확장 기능을 내부 변경 사항과 분리할 수 있습니다.

## 확장 지점

사용할 수 있는 확장 지점은 다섯 가지입니다.

| 훅 | 용도 | 다음과 같은 경우 재정의... |
|------|---------|------------------|
| `_get_extra_tui_widgets()` | 레이아웃에 위젯 삽입 | 지속적으로 표시되는 UI 요소(패널, 상태 줄, 미니 플레이어)가 필요한 경우 |
| `_register_extra_tui_keybindings(kb, *, input_area)` | 키보드 단축키 추가 | 단축키(패널 토글, 전송 제어, 모달 단축키)가 필요한 경우 |
| `_build_tui_layout_children(**widgets)` | 위젯 순서를 완전히 제어 | 기존 위젯의 순서를 바꾸거나 감싸야 하는 경우(드묾) |
| `process_command()` | 사용자 지정 슬래시 명령 추가 | `/mycommand` 처리가 필요한 경우(기존 훅) |
| `_build_tui_style_dict()` | 사용자 지정 prompt_toolkit 스타일 | 사용자 지정 색상이나 스타일이 필요한 경우(기존 훅) |

처음 세 가지는 새 보호 훅입니다. 마지막 두 가지는 이미 존재했습니다.

## 빠른 시작: 래퍼 CLI

```python
#!/usr/bin/env python3
"""my_cli.py — Example wrapper CLI that extends Hermes."""

from cli import HermesCLI
from prompt_toolkit.layout import FormattedTextControl, Window
from prompt_toolkit.filters import Condition


class MyCLI(HermesCLI):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._panel_visible = False

    def _get_extra_tui_widgets(self):
        """Add a toggleable info panel above the status bar."""
        cli_ref = self
        return [
            Window(
                FormattedTextControl(lambda: "📊 My custom panel content"),
                height=1,
                filter=Condition(lambda: cli_ref._panel_visible),
            ),
        ]

    def _register_extra_tui_keybindings(self, kb, *, input_area):
        """F2 toggles the custom panel."""
        cli_ref = self

        @kb.add("f2")
        def _toggle_panel(event):
            cli_ref._panel_visible = not cli_ref._panel_visible

    def process_command(self, cmd: str) -> bool:
        """Add a /panel slash command."""
        if cmd.strip().lower() == "/panel":
            self._panel_visible = not self._panel_visible
            state = "visible" if self._panel_visible else "hidden"
            print(f"Panel is now {state}")
            return True
        return super().process_command(cmd)


if __name__ == "__main__":
    cli = MyCLI()
    cli.run()
```

실행 방법:

```bash
cd ~/.hermes/hermes-agent
source .venv/bin/activate
python my_cli.py
```

## 훅 레퍼런스

### `_get_extra_tui_widgets()`

TUI 레이아웃에 삽입할 prompt_toolkit 위젯 목록을 반환합니다. 위젯은 **스페이서와 상태 표시줄 사이**에 표시되며, 입력 영역 위이자 기본 출력 아래에 위치합니다.

```python
def _get_extra_tui_widgets(self) -> list:
    return []  # default: no extra widgets
```

각 위젯은 prompt_toolkit 컨테이너(예: `Window`, `ConditionalContainer`, `HSplit`)여야 합니다. 위젯을 토글할 수 있게 하려면 `ConditionalContainer` 또는 `filter=Condition(...)`을 사용합니다.

```python
from prompt_toolkit.layout import ConditionalContainer, Window, FormattedTextControl
from prompt_toolkit.filters import Condition

def _get_extra_tui_widgets(self):
    return [
        ConditionalContainer(
            Window(FormattedTextControl("Status: connected"), height=1),
            filter=Condition(lambda: self._show_status),
        ),
    ]
```

### `_register_extra_tui_keybindings(kb, *, input_area)`

Hermes가 자체 키 바인딩을 등록한 후, 레이아웃이 빌드되기 전에 호출됩니다. 키 바인딩을 `kb`에 추가합니다.

```python
def _register_extra_tui_keybindings(self, kb, *, input_area):
    pass  # default: no extra keybindings
```

매개변수:
- **`kb`** — prompt_toolkit 애플리케이션의 `KeyBindings` 인스턴스
- **`input_area`** — 사용자의 입력을 읽거나 조작해야 할 때 사용하는 기본 `TextArea` 위젯

```python
def _register_extra_tui_keybindings(self, kb, *, input_area):
    cli_ref = self

    @kb.add("f3")
    def _clear_input(event):
        input_area.text = ""

    @kb.add("f4")
    def _insert_template(event):
        input_area.text = "/search "
```

기본 제공 키 바인딩과의 **충돌을 피하세요**: `Enter`(제출), `Escape Enter`(줄 바꿈), `Ctrl-C`(중단), `Ctrl-D`(종료), `Tab`(자동 제안 수락). 일반적으로 F2 이상 기능 키와 Ctrl 조합은 안전합니다.

### `_build_tui_layout_children(**widgets)`

위젯 순서를 완전히 제어해야 할 때만 재정의하세요. 대부분의 확장은 대신 `_get_extra_tui_widgets()`를 사용해야 합니다.

```python
def _build_tui_layout_children(self, *, sudo_widget, secret_widget,
    approval_widget, clarify_widget, model_picker_widget=None,
    spinner_widget=None, spacer, status_bar, input_rule_top,
    image_bar, input_area, input_rule_bot, voice_status_bar,
    completions_menu) -> list:
```

기본 구현은 다음을 반환합니다(`None` 위젯은 필터링됨).

```python
[
    Window(height=0),       # anchor
    sudo_widget,            # sudo password prompt (conditional)
    secret_widget,          # secret input prompt (conditional)
    approval_widget,        # dangerous command approval (conditional)
    clarify_widget,         # clarify question UI (conditional)
    model_picker_widget,    # model picker overlay (conditional)
    spinner_widget,         # thinking spinner (conditional)
    spacer,                 # fills remaining vertical space
    *self._get_extra_tui_widgets(),  # YOUR WIDGETS GO HERE
    status_bar,             # model/token/context status line
    input_rule_top,         # ─── border above input
    image_bar,              # attached images indicator
    input_area,             # user text input
    input_rule_bot,         # ─── border below input
    voice_status_bar,       # voice mode status (conditional)
    completions_menu,       # autocomplete dropdown
]
```

## 레이아웃 다이어그램

기본 레이아웃은 위에서 아래 순서로 구성됩니다.

1. **출력 영역** — 스크롤 가능한 대화 기록
2. **스페이서**
3. **추가 위젯** — `_get_extra_tui_widgets()`에서 제공
4. **상태 표시줄** — 모델, 컨텍스트 %, 경과 시간
5. **이미지 표시줄** — 첨부된 이미지 수
6. **입력 영역** — 사용자 프롬프트
7. **음성 상태** — 녹음 표시
8. **완성 메뉴** — 자동 완성 제안

## 팁

- 상태를 변경한 후에는 **표시를 무효화**하세요. `self._invalidate()`를 호출하면 prompt_toolkit이 다시 그려집니다.
- **에이전트 상태에 접근**: `self.agent`, `self.model`, `self.conversation_history`를 모두 사용할 수 있습니다.
- **사용자 지정 스타일**: `_build_tui_style_dict()`를 재정의하고 사용자 지정 스타일 클래스에 대한 항목을 추가하세요.
- **슬래시 명령**: `process_command()`를 재정의하여 명령을 처리하고, 나머지는 `super().process_command(cmd)`를 호출하세요.
- **정말 필요한 경우가 아니면 `run()`을 재정의하지 마세요** — 확장 훅은 바로 이러한 결합을 피하기 위해 존재합니다.
