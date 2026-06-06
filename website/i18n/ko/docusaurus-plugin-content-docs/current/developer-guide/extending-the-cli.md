---
sidebar_position: 8
title: "CLI 확장하기 (Extending the CLI)"
description: "1000줄이 넘는 `run()` 메서드를 재정의하지 않고도 사용자 지정 위젯, 키 바인딩, 레이아웃 변경 등을 통해 Hermes TUI를 확장하는 래퍼(wrapper) CLI를 빌드하는 방법"
---

# CLI 확장하기 (Extending the CLI)

Hermes는 `HermesCLI`에 보호된 확장 훅(hook)들을 노출하여, 래퍼 CLI가 1000줄이 넘는 `run()` 메서드를 오버라이드할 필요 없이 위젯, 키 바인딩, 레이아웃 커스터마이징을 추가할 수 있게 합니다. 이를 통해 여러분의 확장 기능이 내부 구현 변경에 영향을 받지 않고 분리(decoupled)될 수 있습니다.

## 확장 포인트 (Extension points)

다섯 가지 확장 지점(seams)을 사용할 수 있습니다:

| 훅(Hook) | 목적 | 이럴 때 오버라이드하세요... |
|------|---------|------------------|
| `_get_extra_tui_widgets()` | 레이아웃에 위젯 주입 | 지속적인 UI 요소(패널, 상태 줄, 미니 플레이어 등)가 필요할 때 |
| `_register_extra_tui_keybindings(kb, *, input_area)` | 키보드 단축키 추가 | 단축키(패널 토글, 전송 제어, 모달 단축키 등)가 필요할 때 |
| `_build_tui_layout_children(**widgets)` | 위젯 순서에 대한 완전한 제어 | 기존 위젯의 순서를 변경하거나 감싸야 할 때 (드묾) |
| `process_command()` | 맞춤형 슬래시 명령어 추가 | `/mycommand`와 같은 명령 처리가 필요할 때 (기존 훅) |
| `_build_tui_style_dict()` | 사용자 지정 prompt_toolkit 스타일 | 커스텀 색상이나 스타일링이 필요할 때 (기존 훅) |

처음 세 가지는 새로운 보호된 훅이며, 마지막 두 개는 기존에 존재하던 훅입니다.

## 빠른 시작: 래퍼 CLI

```python
#!/usr/bin/env python3
"""my_cli.py — Hermes를 확장하는 예제 래퍼 CLI"""

from cli import HermesCLI
from prompt_toolkit.layout import FormattedTextControl, Window
from prompt_toolkit.filters import Condition


class MyCLI(HermesCLI):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._panel_visible = False

    def _get_extra_tui_widgets(self):
        """상태 표시줄 위에 전환 가능한 정보 패널을 추가합니다."""
        cli_ref = self
        return [
            Window(
                FormattedTextControl(lambda: "📊 My custom panel content"),
                height=1,
                filter=Condition(lambda: cli_ref._panel_visible),
            ),
        ]

    def _register_extra_tui_keybindings(self, kb, *, input_area):
        """F2 키를 눌러 사용자 지정 패널을 토글합니다."""
        cli_ref = self

        @kb.add("f2")
        def _toggle_panel(event):
            cli_ref._panel_visible = not cli_ref._panel_visible

    def process_command(self, cmd: str) -> bool:
        """/panel 슬래시 명령어를 추가합니다."""
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

실행:

```bash
cd ~/.hermes/hermes-agent
source .venv/bin/activate
python my_cli.py
```

## 훅 레퍼런스 (Hook reference)

### `_get_extra_tui_widgets()`

TUI 레이아웃에 삽입할 prompt_toolkit 위젯들의 리스트를 반환합니다. 위젯들은 **여백(spacer)과 상태 표시줄(status bar) 사이** — 즉, 기본 출력 영역 아래이면서 입력 영역보다는 위에 나타납니다.

```python
def _get_extra_tui_widgets(self) -> list:
    return []  # 기본값: 추가 위젯 없음
```

각 위젯은 prompt_toolkit 컨테이너(예: `Window`, `ConditionalContainer`, `HSplit`)여야 합니다. 위젯을 켜고 끌 수 있게 하려면 `ConditionalContainer`나 `filter=Condition(...)`을 사용하세요.

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

Hermes가 자체 키 바인딩을 등록한 후, 레이아웃이 빌드되기 전에 호출됩니다. 여러분의 키 바인딩을 `kb`에 추가하세요.

```python
def _register_extra_tui_keybindings(self, kb, *, input_area):
    pass  # 기본값: 추가 키 바인딩 없음
```

파라미터:
- **`kb`** — prompt_toolkit 애플리케이션의 `KeyBindings` 인스턴스
- **`input_area`** — 메인 `TextArea` 위젯 (사용자 입력을 읽거나 조작해야 할 경우 사용)

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

내장 키 바인딩과 **충돌하지 않도록 주의하세요**: `Enter` (제출), `Escape Enter` (줄 바꿈), `Ctrl-C` (중단), `Ctrl-D` (종료), `Tab` (자동 완성 수락). 기능 키(F2 이상)와 Ctrl 조합은 일반적으로 안전합니다.

### `_build_tui_layout_children(**widgets)`

위젯 순서에 대한 완전한 제어가 필요할 때만 이 메서드를 오버라이드하세요. 대부분의 확장은 대신 `_get_extra_tui_widgets()`를 사용해야 합니다.

```python
def _build_tui_layout_children(self, *, sudo_widget, secret_widget,
    approval_widget, clarify_widget, model_picker_widget=None,
    spinner_widget=None, spacer, status_bar, input_rule_top,
    image_bar, input_area, input_rule_bot, voice_status_bar,
    completions_menu) -> list:
```

기본 구현은 다음을 반환합니다 (`None`인 위젯은 필터링되어 제외됨):

```python
[
    Window(height=0),       # 앵커(anchor)
    sudo_widget,            # sudo 비밀번호 프롬프트 (조건부)
    secret_widget,          # 비밀 입력 프롬프트 (조건부)
    approval_widget,        # 위험한 명령어 승인 (조건부)
    clarify_widget,         # 질문 구체화 UI (조건부)
    model_picker_widget,    # 모델 선택기 오버레이 (조건부)
    spinner_widget,         # 생각 중 스피너 (조건부)
    spacer,                 # 남은 수직 공간 채움
    *self._get_extra_tui_widgets(),  # 여러분의 위젯이 여기에 들어갑니다
    status_bar,             # 모델/토큰/컨텍스트 상태 줄
    input_rule_top,         # ─── 입력창 위쪽 테두리
    image_bar,              # 첨부된 이미지 표시기
    input_area,             # 사용자 텍스트 입력창
    input_rule_bot,         # ─── 입력창 아래쪽 테두리
    voice_status_bar,       # 음성 모드 상태 (조건부)
    completions_menu,       # 자동 완성 드롭다운 메뉴
]
```

## 레이아웃 다이어그램

위에서 아래로 나열된 기본 레이아웃:

1. **출력 영역 (Output area)** — 스크롤되는 대화 기록
2. **여백 (Spacer)**
3. **추가 위젯 (Extra widgets)** — `_get_extra_tui_widgets()`에서 제공됨
4. **상태 표시줄 (Status bar)** — 모델, 컨텍스트 사용량(%), 경과 시간
5. **이미지 표시줄 (Image bar)** — 첨부된 이미지 수
6. **입력 영역 (Input area)** — 사용자 프롬프트 입력창
7. **음성 상태 (Voice status)** — 녹음 표시기
8. **자동 완성 메뉴 (Completions menu)** — 추천 자동 완성

## 팁 (Tips)

- 상태가 변경된 후 **디스플레이를 무효화(Invalidate)하세요**: `self._invalidate()`를 호출하여 prompt_toolkit이 화면을 다시 그리도록(redraw) 트리거합니다.
- **에이전트 상태 접근**: `self.agent`, `self.model`, `self.conversation_history` 모두 접근 가능합니다.
- **사용자 지정 스타일**: `_build_tui_style_dict()`를 오버라이드하고 커스텀 스타일 클래스에 대한 항목을 추가하세요.
- **슬래시 명령어**: `process_command()`를 오버라이드하여 여러분의 명령어를 처리하고, 그 외의 다른 모든 것에 대해서는 `super().process_command(cmd)`를 호출하세요.
- 꼭 필요한 경우가 아니라면 **`run()`을 오버라이드하지 마세요** — 확장 훅은 바로 그 결합(coupling)을 피하기 위해 특별히 존재하는 것입니다.
