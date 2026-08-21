---
sidebar_position: 16
title: "LSP — 의미 진단"
description: "실제 언어 서버(pyright, gopls, rust-analyzer, …)를 write_file과 patch에서 사용하는 쓰기 후 린트 검사에 연결합니다."
---

# Language Server Protocol (LSP)

Hermes는 pyright, gopls, rust-analyzer,
typescript-language-server, clangd 및 약 20개 이상의 언어 서버를
백그라운드 서브프로세스로 실행하고, 이들의 의미 진단을
`write_file`과 `patch`에서 사용하는 쓰기 후 린트 검사에 전달합니다. 에이전트가
파일을 편집하면 단순한 구문 오류뿐 아니라 언어 서버가 감지하는 **형식 오류, 정의되지 않은 이름, 누락된 임포트,
프로젝트 전반의 의미 문제**까지 해당 편집으로 발생한 오류를 정확히 확인할 수 있습니다.

이는 최고 수준의 코딩 에이전트가 사용하는 것과 같은 아키텍처입니다. Hermes는 이를 자체적으로 제공합니다. 편집기 호스트, 설치할 플러그인, 별도로 관리할 데몬이 필요하지 않습니다.

## LSP가 실행되는 경우

LSP는 **git 작업 공간 감지**를 기준으로 활성화됩니다. 에이전트의 작업 디렉터리(또는 편집 중인 파일)가 git 저장소 안에 있으면 LSP가 해당 작업 공간을 기준으로 실행됩니다. 어느 위치도 git 저장소에 속하지 않으면 LSP는 대기 상태로 유지됩니다. cwd가 사용자의 홈 디렉터리이고 진단할 프로젝트가 없는 메시징 게이트웨이에서 유용합니다.

검사는 계층적으로 이루어집니다. 먼저 인프로세스 구문 검사를 수행하고(마이크로초), 구문이 올바른 경우에만 LSP 진단을 수행합니다. 언어 서버가 불안정하거나 없더라도 쓰기가 실패할 수는 없습니다. 모든 LSP 실패 경로는 조용히 구문 전용 결과로 대체됩니다.

구체적으로, 성공한 `write_file` 또는 `patch`마다 다음을 수행합니다.

1. Hermes가 해당 파일의 현재 진단 기준선을 캡처합니다.
2. 쓰기를 수행합니다.
3. 언어 서버를 다시 조회하고 기준선에 이미 있던 진단을 필터링해 새 진단만 표시합니다.

에이전트에는 다음과 같은 출력이 표시됩니다.

```
{
  "bytes_written": 42,
  "dirs_created": false,
  "lint": {"status": "ok", "output": ""},
  "lsp_diagnostics": "LSP diagnostics introduced by this edit:\n<diagnostics file=\"/path/to/foo.py\">\nERROR [42:5] Cannot find name 'foo' [reportUndefinedVariable] (Pyright)\nERROR [50:1] Argument of type \"str\" is not assignable to \"int\" [reportArgumentType] (Pyright)\n</diagnostics>"
}
```

`lint` 필드에는 구문 검사 결과가 담깁니다(`ast.parse`, `json.loads` 등을 통한 마이크로초 단위의 인프로세스 파싱). `lsp_diagnostics` 필드에는 실제 언어 서버의 의미 진단이 담깁니다. 서로 독립적인 두 신호 채널이므로, 에이전트는 구문상 문제가 없는 파일에 의미 문제가 있는 경우에도
``lint: ok``와 내용이 채워진 ``lsp_diagnostics``를 함께 확인합니다.

## 지원 언어

| 언어 | 서버 | 자동 설치 |
|----------|--------|--------------|
| Python | `pyright-langserver` | npm |
| TypeScript / JavaScript / JSX / TSX | `typescript-language-server` | npm |
| Vue | `@vue/language-server` | npm |
| Svelte | `svelte-language-server` | npm |
| Astro | `@astrojs/language-server` | npm |
| Go | `gopls` | `go install` |
| Rust | `rust-analyzer` | 수동(rustup) |
| C / C++ | `clangd` | 수동(LLVM) |
| Bash / Zsh | `bash-language-server` | npm |
| YAML | `yaml-language-server` | npm |
| Lua | `lua-language-server` | 수동(GitHub releases) |
| PHP | `intelephense` | npm |
| OCaml | `ocaml-lsp` | 수동(opam) |
| Dockerfile | `dockerfile-language-server-nodejs` | npm |
| Terraform | `terraform-ls` | 수동 |
| Dart | `dart language-server` | 수동(dart sdk) |
| Haskell | `haskell-language-server` | 수동(ghcup) |
| Julia | `julia` + LanguageServer.jl | 수동 |
| Clojure | `clojure-lsp` | 수동 |
| Nix | `nixd` | 수동 |
| Zig | `zls` | 수동 |
| Gleam | `gleam lsp` | 수동(gleam install) |
| Elixir | `elixir-ls` | 수동 |
| Prisma | `prisma language-server` | 수동 |
| Kotlin | `kotlin-language-server` | 수동 |
| Java | `jdtls` | 수동 |
| PowerShell | `PowerShellEditorServices` (`pwsh` host) | 수동(release zip) |

"수동" 항목은 해당 언어에 적합한 도구 체인 관리자(rustup, ghcup, opam, brew 등)를 통해 서버를 설치합니다. Hermes는 PATH 또는 `<HERMES_HOME>/lsp/bin/`에 있는 바이너리를 자동으로 감지합니다.

### PowerShell

PowerShellEditorServices는 단일 바이너리가 아니라 `pwsh`(PowerShell 7 이상) 또는 `powershell` 호스트로 실행되는 PowerShell 모듈 번들입니다. 설정 방법은 다음과 같습니다.

1. `pwsh`(또는 Windows의 `powershell`)가 PATH에 있도록 [PowerShell](https://github.com/PowerShell/PowerShell)을 설치합니다.
2. [PowerShellEditorServices releases](https://github.com/PowerShell/PowerShellEditorServices/releases)에서 최신 릴리스 zip을 다운로드하고 압축을 풉니다.
3. 압축을 푼 번들을 가리키도록 Hermes를 설정합니다. `PowerShellEditorServices/Start-EditorServices.ps1`을 포함하는 디렉터리입니다. 다음 중 하나를 선택합니다.
   - `config.yaml`에서 `lsp.servers.powershell.command: ["/path/to/bundle"]`를 설정하거나
   - `<HERMES_HOME>/lsp/PowerShellEditorServices`에 압축을 풀거나
   - `PSES_BUNDLE_PATH=/path/to/bundle`을 내보냅니다.

`hermes lsp status`에서 `pwsh`를 찾으면 `installed`로 표시됩니다. 번들이 없으면 다운로드 링크와 함께 로그에 한 번 경고가 표시됩니다.

일부 서버는 npm이 자동으로 가져오지 않는 피어 종속성과 함께 설치됩니다. 현재 해당되는 경우는 `typescript-language-server`이며, 같은 `node_modules` 트리에서 가져올 수 있는 `typescript` SDK가 필요합니다. `hermes lsp install typescript`를 실행하거나 처음 사용할 때 자동 설치가 실행되면 Hermes가 두 패키지를 함께 설치합니다.

## CLI

```
hermes lsp status          # service state + per-server install status
hermes lsp list            # registry, optionally --installed-only
hermes lsp install <id>    # eagerly install one server
hermes lsp install-all     # try every server with a known recipe
hermes lsp restart         # tear down running clients
hermes lsp which <id>      # print resolved binary path
```

`hermes lsp status`가 가장 좋은 시작점입니다. 오늘 의미 진단을 제공할 언어와 바이너리 설치가 필요한 언어를 보여줍니다.

## 구성

기본값은 일반적인 설정에서 작동하므로 바이너리가 PATH에 있다면 설정할 것이 없습니다.

```yaml
# config.yaml
lsp:
  # Master toggle. Disabling skips the entire subsystem — no servers
  # spawn, no background event loop runs.
  enabled: true

  # How long to wait for diagnostics after each write.
  wait_mode: document      # "document" or "full"
  # Max seconds to wait for the server to re-check the file after an
  # edit. Only *fresh* diagnostics (produced for the post-edit
  # content) are ever reported; if the server doesn't finish within
  # this budget, the edit reports "no LSP data" rather than stale
  # errors from before the edit. Raise this for slow servers on big
  # projects (tsserver, rust-analyzer mid-indexing).
  wait_timeout: 5.0

  # How to handle missing server binaries.
  #   auto    — install via npm/pip/go install into <HERMES_HOME>/lsp/bin
  #   manual  — only use binaries already on PATH
  install_strategy: auto

  # How long an unused language-server client stays alive (seconds).
  # Idle servers are shut down automatically and respawned on the next
  # relevant file operation. Set to 0 to disable idle reaping and keep
  # servers alive for the life of the process. Values below 30s are
  # clamped to 30 so a sweep can never reap a client mid-operation.
  idle_timeout: 600

  # Per-server overrides (all optional).
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
      disabled: true       # skip TS even when its extensions match
```

### 서버별 키

* `disabled: true` — 파일 확장자가 일치하더라도 이 서버를 완전히 건너뜁니다.
* `command: [bin, ...args]` — 사용자 지정 바이너리 경로를 고정합니다. 자동 설치를 우회합니다.
* `env: {KEY: value}` — 생성된 프로세스에 전달할 추가 환경 변수입니다.
* `initialization_options: {...}` — `initialize` 핸드셰이크에서 전송하는 LSP `initializationOptions` 페이로드에 병합됩니다. 서버별 설정이므로 언어 서버 문서를 참고하세요.

## 설치 위치

`install_strategy: auto`인 경우 Hermes는 바이너리를 `<HERMES_HOME>/lsp/bin/`에 설치합니다. NPM 패키지는 `<HERMES_HOME>/lsp/node_modules/`에 설치되고, bin 심볼릭 링크는 한 단계 위에 생성됩니다. Go 바이너리는 스테이징 디렉터리를 가리키는 `GOBIN`과 함께 `go install`에서 가져옵니다.

`/usr/local/`, `~/.local/` 또는 다른 공유 위치에는 아무것도 설치되지 않습니다. 스테이징 디렉터리는 Hermes가 전적으로 소유하며 프로필을 초기화할 때 제거됩니다.

## 성능 특성

LSP 서버는 **처음 사용할 때 지연 생성**됩니다. `.py` 파일을 한 번도 처리하지 않은 프로젝트에서 Python 파일을 편집하면 pyright가 생성됩니다. 대부분의 서버는 생성에 1~3초가 걸리고(rust-analyzer는 콜드 프로젝트에서 10초 이상 걸릴 수 있음), 같은 작업 공간에서 이어지는 편집은 실행 중인 서버를 재사용합니다.

진단이 출력되지 않는 깨끗한 쓰기에서는 LSP 계층이 몇 밀리초를 추가합니다. 진단이 출력되면 대기 예산은 `wait_timeout`초입니다. 일반적으로 pyright/tsserver는 수십 밀리초 안에, rust-analyzer는 인덱싱 중인 프로젝트에서 몇 초 안에 응답합니다.

진단은 **최신성 기준**을 적용합니다. 서버가 현재 편집의 내용에 대해 생성한 결과일 때만 결과로 인정됩니다(변경 시점 이후의 `publishDiagnostics` 푸시 또는 변경 후에 응답한 풀 요청). 아직 다시 검사하지 않은 느린 서버는 해당 편집에 대해 "데이터 없음"을 반환하며, 어제의 오류를 현재 오류로 다시 보고하지 않습니다.

서버는 사용 중인 동안 유지되며 파일 활동이 `lsp.idle_timeout`초(기본값 600) 동안 없으면 종료됩니다. 파일을 다루는 장기 실행 게이트웨이도 이제 작업 공간마다 언어 서버 프로세스가 영구적으로 쌓이지 않습니다. 종료된 서버는 다음 관련 파일 작업에서 자동으로 다시 생성됩니다. `idle_timeout: 0`으로 설정하면 종료를 비활성화하고 프로세스 수명 동안 모든 서버의 인덱스를 유지할 수 있습니다.

## 비활성화

`config.yaml`에서 `lsp.enabled: false`로 설정하면 전체 하위 시스템이 비활성화됩니다. 쓰기 후 검사는 인프로세스 구문 검사(`ast.parse`는 Python, `json.loads`는 JSON 등)로 대체되며, 이전 버전부터 제공되던 기능은 변경되지 않습니다.

전체 계층을 비활성화하지 않고 하나의 언어만 비활성화하려면 다음과 같이 설정합니다.

```yaml
lsp:
  servers:
    rust-analyzer:
      disabled: true
```

## 문제 해결

**`hermes lsp status`에서 서버가 "missing"으로 표시됨**

바이너리가 PATH에도 없고 `<HERMES_HOME>/lsp/bin/`에도 없습니다. `hermes lsp install <server_id>`를 실행해 자동 설치를 시도하거나, 해당 언어의 일반적인 도구 체인을 통해 바이너리를 수동으로 설치합니다.

**`hermes lsp status`의 `Backend warnings` 섹션**

일부 서버는 실제 진단을 외부 CLI에 위임하는 얇은 래퍼로 제공됩니다. 이러한 서버는 정상적으로 생성되고 요청도 수락하지만, 보조 바이너리가 없으면 오류를 전혀 출력하지 않습니다. 가장 일반적인 경우는 진단을 `shellcheck`에 위임하는 `bash-language-server`입니다. `hermes lsp status`에 `Backend warnings` 섹션이 표시되면 운영 체제의 패키지 관리자를 통해 표시된 도구를 설치합니다.

```
apt install shellcheck      # Debian / Ubuntu
brew install shellcheck     # macOS
scoop install shellcheck    # Windows
```

같은 경고가 서버 생성 시 `~/.hermes/logs/agent.log`에 한 번 기록됩니다.

**서버는 시작되지만 진단을 반환하지 않음**

`[agent.lsp.client]` 항목이 있는지 `~/.hermes/logs/agent.log`를 확인합니다. 언어 서버의 표준 오류와 프로토콜 오류가 모두 해당 로그에 기록됩니다. 일부 서버, 특히 rust-analyzer는 파일별 진단을 출력하기 전에 프로젝트 전체 인덱스를 완료해야 합니다. 서버 시작 후 첫 번째 편집은 진단 없이 완료될 수 있으며, 이후 편집에서 진단이 수집됩니다.

**서버가 비정상 종료됨**

비정상 종료된 서버는 문제가 있는 집합에 추가되며 세션이 끝날 때까지 다시 시도되지 않습니다. `hermes lsp restart`를 실행해 해당 집합을 초기화하면 다음 편집에서 서버가 다시 생성됩니다.

**git 저장소 외부의 파일 편집**

의도한 동작에 따라 LSP는 git 저장소 안에서만 실행됩니다. 프로젝트가 아직 초기화되지 않았다면 `git init`을 실행해 LSP 진단을 활성화합니다. 그렇지 않으면 인프로세스 구문 전용 대체 경로가 적용됩니다.
