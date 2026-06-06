---
sidebar_position: 16
title: "LSP — 시맨틱 진단"
description: "실제 언어 서버(pyright, gopls, rust-analyzer, ...)가 write_file 및 patch에서 사용되는 쓰기 후 린트 검사에 연결됩니다."
---

# 언어 서버 프로토콜 (Language Server Protocol, LSP)

Hermes는 pyright, gopls, rust-analyzer, typescript-language-server, clangd 및 ~20개 이상의 전체 언어 서버를 백그라운드 하위 프로세스로 실행하고, 이들의 시맨틱 진단(semantic diagnostics)을 `write_file` 및 `patch`에서 사용되는 쓰기 후 린트(lint) 검사에 공급합니다. 에이전트가 파일을 편집할 때, 구문 오류뿐만 아니라 **유형 오류, 정의되지 않은 이름, 누락된 가져오기(imports) 및 프로젝트 전체의 시맨틱 문제** 등 언어 서버가 감지하는, 편집으로 인해 발생한 정확한 오류를 보게 됩니다.

이것은 최상위 코딩 에이전트들이 사용하는 것과 동일한 아키텍처입니다. Hermes는 이를 독립적으로 제공합니다: 편집기 호스트가 필요 없고, 설치할 플러그인도 없으며, 관리해야 할 별도의 데몬도 없습니다.

## LSP가 실행되는 경우

LSP는 **git 작업 공간(workspace) 감지**를 기반으로 작동합니다. 에이전트의 작업 디렉토리(또는 편집 중인 파일)가 git 저장소 내부에 있을 때, LSP는 해당 작업 공간에 대해 실행됩니다. 둘 다 git 저장소에 없는 경우 LSP는 휴면 상태로 유지됩니다. 이는 cwd가 사용자의 홈 디렉토리이고 진단할 프로젝트가 없는 메시징 게이트웨이에 유용합니다.

검사는 계층화되어 있습니다: 프로세스 내 구문 검사(마이크로초 소요)가 먼저 진행되고, 구문이 깨끗할 때 두 번째로 LSP 진단이 수행됩니다. 불안정하거나 누락된 언어 서버가 파일 쓰기를 손상시킬 수는 없습니다 — 모든 LSP 실패 경로는 구문 전용 결과로 조용히 대체됩니다.

구체적으로, `write_file` 또는 `patch`가 성공할 때마다:

1. Hermes는 파일에 대한 현재 진단의 기준선(baseline)을 캡처합니다.
2. 쓰기 작업을 수행합니다.
3. 언어 서버에 다시 쿼리하여 기준선에 이미 있던 진단은 필터링하고 새로운 진단만 표시합니다.

에이전트는 다음과 같은 출력을 보게 됩니다:

```
{
  "bytes_written": 42,
  "dirs_created": false,
  "lint": {"status": "ok", "output": ""},
  "lsp_diagnostics": "LSP diagnostics introduced by this edit:\n<diagnostics file=\"/path/to/foo.py\">\nERROR [42:5] Cannot find name 'foo' [reportUndefinedVariable] (Pyright)\nERROR [50:1] Argument of type \"str\" is not assignable to \"int\" [reportArgumentType] (Pyright)\n</diagnostics>"
}
```

`lint` 필드는 구문 검사 결과( `ast.parse`, `json.loads` 등을 통한 프로세스 내 구문 분석)를 전달하고, `lsp_diagnostics` 필드는 실제 언어 서버에서 온 시맨틱 진단을 전달합니다. 이는 독립적인 두 채널입니다 — 에이전트는 시맨틱 문제가 있는 구문적으로 깨끗한 파일을 ``lint: ok``와 내용이 채워진 ``lsp_diagnostics``로 인식합니다.

## 지원되는 언어

| 언어 | 서버 | 자동 설치 |
|----------|--------|--------------|
| Python | `pyright-langserver` | npm |
| TypeScript / JavaScript / JSX / TSX | `typescript-language-server` | npm |
| Vue | `@vue/language-server` | npm |
| Svelte | `svelte-language-server` | npm |
| Astro | `@astrojs/language-server` | npm |
| Go | `gopls` | `go install` |
| Rust | `rust-analyzer` | 수동 (rustup) |
| C / C++ | `clangd` | 수동 (LLVM) |
| Bash / Zsh | `bash-language-server` | npm |
| YAML | `yaml-language-server` | npm |
| Lua | `lua-language-server` | 수동 (GitHub 릴리스) |
| PHP | `intelephense` | npm |
| OCaml | `ocaml-lsp` | 수동 (opam) |
| Dockerfile | `dockerfile-language-server-nodejs` | npm |
| Terraform | `terraform-ls` | 수동 |
| Dart | `dart language-server` | 수동 (dart sdk) |
| Haskell | `haskell-language-server` | 수동 (ghcup) |
| Julia | `julia` + LanguageServer.jl | 수동 |
| Clojure | `clojure-lsp` | 수동 |
| Nix | `nixd` | 수동 |
| Zig | `zls` | 수동 |
| Gleam | `gleam lsp` | 수동 (gleam install) |
| Elixir | `elixir-ls` | 수동 |
| Prisma | `prisma language-server` | 수동 |
| Kotlin | `kotlin-language-server` | 수동 |
| Java | `jdtls` | 수동 |

"수동(manual)" 항목의 경우 해당 언어에 적합한 툴체인 관리자(rustup, ghcup, opam, brew 등)를 통해 서버를 설치합니다. Hermes는 PATH 또는 `<HERMES_HOME>/lsp/bin/`에 있는 바이너리를 자동 감지합니다.

일부 서버는 npm이 자동으로 가져오지 않는 피어(peer) 종속성과 함께 설치됩니다. 현재의 예로는 `typescript-language-server`가 있는데, 이는 동일한 `node_modules` 트리에서 가져올 수 있는 `typescript` SDK를 필요로 합니다 — Hermes는 사용자가 `hermes lsp install typescript`를 실행하거나 처음 사용할 때 자동 설치가 시작될 때 두 패키지를 함께 설치합니다.

## CLI

```
hermes lsp status          # 서비스 상태 + 서버별 설치 상태
hermes lsp list            # 레지스트리, 선택적으로 --installed-only
hermes lsp install <id>    # 하나의 서버를 사전에 설치
hermes lsp install-all     # 알려진 레시피가 있는 모든 서버 시도
hermes lsp restart         # 실행 중인 클라이언트 종료
hermes lsp which <id>      # 확인된 바이너리 경로 인쇄
```

`hermes lsp status`가 가장 좋은 시작점입니다 — 오늘 어떤 언어에 시맨틱 진단이 제공되는지와 어떤 바이너리를 설치해야 하는지 보여줍니다.

## 구성 (Configuration)

기본값은 일반적인 설정에서 잘 작동합니다. 바이너리가 PATH에 있으면 설정할 것이 없습니다.

```yaml
# config.yaml
lsp:
  # 마스터 토글. 비활성화하면 전체 하위 시스템을 건너뜁니다 — 서버가 생성되지 않고,
  # 백그라운드 이벤트 루프가 실행되지 않습니다.
  enabled: true

  # 각 쓰기 후 진단을 기다리는 시간.
  wait_mode: document      # "document" 또는 "full"
  wait_timeout: 5.0

  # 누락된 서버 바이너리 처리 방법.
  #   auto    — npm/pip/go install을 통해 <HERMES_HOME>/lsp/bin 에 설치
  #   manual  — 이미 PATH에 있는 바이너리만 사용
  install_strategy: auto

  # 서버별 재정의 (모두 선택 사항).
  servers:
    pyright:
      disabled: false
      command: ["/abs/path/to/pyright-langserver", "--stdio"]
      env: { PYRIGHT_LOG_LEVEL: "info" }
      initialization_options:
        python:
          analysis:
            typeCheckingMode: "strict"
    typescript:
      disabled: true       # TS 확장이 일치하더라도 TS 건너뛰기
```

### 서버별 키 (Per-server keys)

* `disabled: true` — 확장이 파일과 일치하더라도 이 서버를 완전히 건너뜁니다.
* `command: [bin, ...args]` — 사용자 지정 바이너리 경로를 고정합니다. 자동 설치를 우회합니다.
* `env: {KEY: value}` — 생성된 프로세스에 전달되는 추가 환경 변수.
* `initialization_options: {...}` — `initialize` 핸드셰이크에서 전송되는 LSP `initializationOptions` 페이로드에 병합됩니다. 서버마다 다르므로 해당 언어 서버의 문서를 참조하세요.

## 설치 위치

`install_strategy: auto`일 때 Hermes는 바이너리를 `<HERMES_HOME>/lsp/bin/`에 설치합니다. NPM 패키지는 `<HERMES_HOME>/lsp/node_modules/`에 위치하며 한 단계 위로 bin 심볼릭 링크가 생성됩니다. Go 바이너리는 `GOBIN`이 스테이징 디렉토리를 가리키도록 설정된 상태에서 `go install`을 통해 제공됩니다.

`/usr/local/`, `~/.local/` 또는 기타 공유 위치에는 아무것도 설치되지 않습니다 — 스테이징 디렉토리는 완전히 Hermes의 소유이며 프로필을 재설정하면 제거됩니다.

## 성능 특성

LSP 서버는 처음 사용할 때 **지연 생성(lazy-spawned)**됩니다. `.py` 트래픽을 본 적 없는 프로젝트에서 Python 파일을 편집하면 pyright가 생성됩니다. 대부분의 서버는 생성에 1-3초가 걸립니다(rust-analyzer는 콜드 프로젝트에서 10초 이상 걸릴 수 있습니다). 동일한 작업 공간에서의 후속 편집은 실행 중인 서버를 재사용합니다.

LSP 계층은 진단이 방출되지 않을 때 클린 쓰기 작업에 몇 밀리초를 추가합니다. 진단이 방출될 때의 대기 예산은 `wait_timeout` 초입니다 — 일반적으로 서버는 pyright/tsserver의 경우 수십 밀리초, rust-analyzer의 인덱싱 중에는 몇 초 안에 응답합니다.

서버는 Hermes 프로세스가 살아있는 동안 계속 실행됩니다. 유휴 시간 초과 수확기(idle-timeout reaper)는 없습니다 — 모든 쓰기 작업마다 서버 인덱스를 다시 시작하는 비용이 데몬을 유지하는 것보다 훨씬 크기 때문입니다.

## 비활성화

전체 하위 시스템을 비활성화하려면 `config.yaml`에서 `lsp.enabled: false`로 설정합니다. 쓰기 후 검사는 이전 버전에서 변경되지 않고 제공되는 프로세스 내 구문 검사(Python의 경우 `ast.parse`, JSON의 경우 `json.loads` 등)로 대체됩니다.

전체 계층을 비활성화하지 않고 단일 언어만 비활성화하려면:

```yaml
lsp:
  servers:
    rust-analyzer:
      disabled: true
```

## 문제 해결 (Troubleshooting)

**`hermes lsp status`에 서버가 "missing"으로 표시됨**

바이너리가 PATH에 없고 `<HERMES_HOME>/lsp/bin/`에도 없습니다. `hermes lsp install <server_id>`를 실행하여 자동 설치를 시도하거나, 해당 언어의 일반적인 툴체인을 통해 바이너리를 수동으로 설치하십시오.

**`hermes lsp status`의 `Backend warnings` 섹션**

일부 서버는 실제 진단을 위해 외부 CLI를 감싸는 얇은 래퍼(wrapper)로 제공됩니다 — 이들은 깨끗하게 생성되고 요청을 수락하지만 외부(sidecar) 바이너리가 없으면 오류를 방출하지 않습니다. 가장 일반적인 경우는 진단을 `shellcheck`에 위임하는 `bash-language-server`입니다. `hermes lsp status`에 `Backend warnings` 섹션이 표시되면, OS 패키지 관리자를 통해 명시된 도구를 설치하십시오:

```
apt install shellcheck      # Debian / Ubuntu
brew install shellcheck     # macOS
scoop install shellcheck    # Windows
```

동일한 경고가 서버 생성 시 `~/.hermes/logs/agent.log`에 한 번 기록됩니다.

**서버가 시작되지만 진단을 반환하지 않음**

`~/.hermes/logs/agent.log`에서 `[agent.lsp.client]` 항목을 확인하십시오 — 언어 서버의 stderr 및 프로토콜 오류가 모두 여기에 기록됩니다. 일부 서버(특히 rust-analyzer)는 파일별 진단을 방출하기 전에 프로젝트 전체 인덱싱을 완료해야 합니다. 서버 시작 후 첫 번째 편집은 진단 없이 완료될 수 있으며, 후속 편집에서 이를 픽업할 수 있습니다.

**서버가 충돌함 (Server crashed)**

충돌한 서버는 고장난(broken) 세트에 추가되며 나머지 세션 동안 재시도되지 않습니다. `hermes lsp restart`를 실행하여 세트를 지우십시오. 다음 편집 시 다시 생성됩니다.

**git 저장소 외부의 파일 편집**

의도적으로 LSP는 git 저장소 내부에서만 실행됩니다. 프로젝트가 아직 초기화되지 않은 경우 `git init`을 실행하여 LSP 진단을 활성화하십시오. 그렇지 않으면 프로세스 내의 구문 전용(syntax-only) 폴백이 적용됩니다.
