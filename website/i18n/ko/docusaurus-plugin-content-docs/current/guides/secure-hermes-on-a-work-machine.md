---
sidebar_position: 26
title: "개인 또는 업무용 컴퓨터에서 Hermes 실행하기"
description: "내가 사용하는 개인 노트북이나 회사에서 관리하는 워크스테이션에서 Hermes Agent를 실행하기 위한 보안 상태 안내 — 기본 설정이 보호하는 항목, 보안을 더 강화하는 방법, 실수가 발생했을 때 되돌리는 방법"
---

# 개인 또는 업무용 컴퓨터에서 Hermes 실행하기

내가 사용하는 컴퓨터, 즉 개인 노트북이나 회사에서 관리하는 워크스테이션에서 에이전트를 실행하려 합니다. 안전한 보안 상태는 어떤 모습일까요?

짧게 답하면, 기본 설정만으로도 대부분의 보호 기능이 이미 작동합니다. Hermes는 명령 승인, 파일 쓰기 안전성, 자격 증명 처리를 아우르는 심층 방어 모델을 바탕으로 기본적으로 안전하게 제공됩니다. 이 페이지에서는 기본 제공되는 기능, 공유 컴퓨터나 업무용 컴퓨터에서 강화할 수 있는 설정, 문제가 발생했을 때 되돌리는 방법을 설명합니다. 여기에서 다루는 모든 제어 기능은 [보안](/user-guide/security) 가이드에서 자세히 설명합니다.

## 기본 설정이 이미 보호하는 항목

새로 설치하고 별도 설정을 하지 않아도 다음 보호 기능이 활성화되어 있습니다.

**위험한 명령에는 승인이 필요합니다.** 명령을 실행하기 전에 Hermes는 재귀 삭제, `/etc/`에 대한 쓰기, 디스크 작업, 셸로 파이프하기 등 위험한 패턴을 엄선한 목록과 대조합니다. 기본 `approvals.mode: smart`는 보조 LLM을 사용해 위험을 평가합니다. 위험이 낮은 명령은 해당 명령에 한해서 자동 승인하고, 실제로 위험한 명령은 자동 거부하며, 판단이 불확실한 경우에는 수동 프롬프트로 올립니다.

**승인 프롬프트는 안전하게 실패합니다.** 승인 프롬프트의 제한 시간(기본 300초) 안에 응답하지 않으면 명령은 **거부됩니다**. 자리를 비워도 아무것도 조용히 승인되지 않습니다.

**강력한 차단 목록은 항상 적용되는 최저선입니다.** `rm -rf /`, 포크 폭탄, 물리 디스크를 0으로 만드는 작업 같은 일부 명령은 승인 모드, `--yolo`, 명시적인 "항상 허용" 설정과 **관계없이** 거부됩니다. 차단 목록은 승인 계층이 명령을 확인하기도 전에 작동하며, 이를 무시할 플래그는 없습니다.

**민감한 경로에 대한 파일 쓰기는 차단됩니다.** `write_file` 및 `patch` 도구는 OS 자격 증명 저장소(`~/.ssh/`, `~/.aws/`, `~/.kube/`, `/etc/sudoers`, `~/.netrc`), Hermes 자격 증명 저장소(`auth.json`, `.env`, 페어링 데이터), 프로젝트 비밀 파일(`.env`, `.env.local`, `.envrc`)을 디스크 어디에도 기록할 수 없습니다. 차단된 쓰기는 즉시 오류를 반환하며, 승인 프롬프트가 표시되지 않고 채팅 UI에서 우회할 방법도 없습니다.

**비밀 정보는 출력에서 가려집니다.** `security.redact_secrets`는 기본적으로 켜져 있습니다. 도구 출력에서 API 키, 토큰, 비밀번호처럼 보이는 패턴은 대화 컨텍스트와 로그에 들어가기 전에 가려집니다.

**데이터는 지정한 곳으로만 전송됩니다.** API 호출은 **사용자가 설정한 LLM 제공자에게만** 전송됩니다. Hermes Agent는 텔레메트리, 사용 데이터, 분석 정보를 수집하지 않습니다. 대화, 메모리, 스킬은 `~/.hermes/`에 로컬로 저장됩니다. [FAQ](/reference/faq#is-my-data-sent-anywhere)를 참조하세요.

:::info
표면 아래에는 더 많은 보호 기능이 있습니다. URL을 처리할 수 있는 모든 도구의 SSRF 보호, MCP 하위 프로세스를 위한 필터링된 환경, 컨텍스트 파일의 프롬프트 인젝션 검사 등이 그 예입니다. [보안](/user-guide/security) 페이지에서 모든 계층을 설명합니다.
:::

## 공유 컴퓨터 또는 업무용 컴퓨터를 위한 강화

회사 데이터, 운영 자격 증명 또는 다른 사람의 파일이 있는 컴퓨터라면 기본 설정에 다음 설정을 추가하세요.

### 승인을 수동으로 전환

`smart` 모드는 위험이 낮은 명령을 자동 승인합니다. 플래그가 지정된 모든 명령을 직접 확인하려면 다음과 같이 하세요.

```yaml
approvals:
  mode: manual
```

수동 모드에서는 플래그가 지정된 명령을 실행하기 전에 항상 프롬프트를 표시합니다.

### 직접 거부 규칙 추가

`approvals.deny`는 일치하는 터미널 명령을 무조건 차단하는 glob 패턴 목록입니다. `--yolo`, `/yolo`, `mode: off`에서도 적용됩니다. 기본 제공되는 강력한 차단 목록을 사용자가 편집할 수 있게 만든 대응 항목입니다. 이 컴퓨터에서 절대 실행되어서는 안 되는 작업을 선언할 때 사용하세요.

```yaml
approvals:
  deny:
    - "git push --force*"
    - "*curl*|*sh*"
    - "dd if=* of=/dev/*"
```

패턴은 대소문자를 구분하지 않는 [fnmatch](https://docs.python.org/3/library/fnmatch.html) glob이며 전체 명령 텍스트에 대조됩니다. 대조는 위험 패턴 탐지기가 사용하는 것과 동일한 정규화 및 난독화 해제 변형에 대해 수행되므로, 단순한 인용 부호 트릭으로 규칙을 빠져나갈 수 없습니다. 패턴은 항상 따옴표로 감싸세요. 맨 앞의 `*`를 따옴표 없이 쓰면 YAML 파싱 오류가 발생합니다. 변경 사항은 재시작 없이 즉시 적용됩니다. 자세한 내용은 [사용자 정의 거부 규칙](/user-guide/security#user-defined-deny-rules-approvalsdeny)을 참조하세요.

### 파일 쓰기 샌드박스 설정

`HERMES_WRITE_SAFE_ROOT`는 `write_file` 및 `patch`를 지정한 디렉터리 접두사로 제한합니다. 목록 밖의 모든 경로는 강제로 차단됩니다. 여러 루트는 Unix에서 `:`로 구분합니다.

```bash
export HERMES_WRITE_SAFE_ROOT=/path/to/project:/home/you/.hermes
```

안전한 루트 안에 있는 민감한 경로도 여전히 차단됩니다. `$HOME`을 지정해도 `~/.ssh/id_rsa`에 쓸 수 없습니다.

:::caution
이 설정을 무심코 `~/.hermes/.env`에 추가하지 마세요. 프로젝트 디렉터리만 지정하면 에이전트는 해당 접두사 밖에 있는 `~/.hermes/cron/jobs.json`, 프로필 스킬 또는 기타 Hermes 상태에 쓸 수 없습니다. 위 예시처럼 Hermes 홈을 두 번째 루트로 포함하세요.
:::

### 호스트에서 명령 실행을 분리

가장 강력한 격리는 컴퓨터에서 아예 명령을 실행하지 않는 것입니다. 터미널 도구는 여러 [백엔드](/user-guide/features/tools#terminal-backends)를 지원합니다.

| 백엔드 | 격리 |
|---------|-----------|
| `local` | 없음 — 호스트에서 실행(위험한 명령 검사는 적용됨) |
| `docker` | 컨테이너 — 컨테이너 자체가 보안 경계 |
| `ssh` | 원격 컴퓨터 — 별도 서버에서 실행 |

```yaml
terminal:
  backend: docker
  docker_image: "nikolaik/python-nodejs:python3.11-nodejs20"
  docker_forward_env: []  # Explicit allowlist only; empty keeps secrets out of the container
```

모든 Docker 컨테이너는 강화된 설정으로 실행됩니다. Linux 기능을 모두 제거하고 최소한의 기능만 다시 추가하며, `no-new-privileges`, 프로세스 수 제한, 크기가 제한된 tmpfs 마운트를 적용합니다. 컨테이너 백엔드를 사용하면 컨테이너 내부의 파괴적인 명령이 호스트에 영향을 줄 수 없으므로 위험한 명령 검사를 건너뜁니다.

`ssh`의 경우 `config.yaml`에서 `terminal.backend: ssh`를 설정하고 `~/.hermes/.env`에 `TERMINAL_SSH_HOST`, `TERMINAL_SSH_USER`, `TERMINAL_SSH_KEY`로 호스트 정보를 제공하세요. [네트워크 격리](/user-guide/security#network-isolation)를 참조하세요.

### 메시징을 사용한다면: 허용 목록과 페어링

이 컴퓨터에서 [게이트웨이](/user-guide/security#user-authorization-gateway)를 실행하고 있나요? 허용 목록이 설정되지 않았고 `GATEWAY_ALLOW_ALL_USERS`도 설정되지 않았다면 기본값은 거부이므로 **모든 사용자가 거부됩니다**. 명시적인 설정을 유지하세요.

```bash
# ~/.hermes/.env
TELEGRAM_ALLOWED_USERS=123456789
GATEWAY_ALLOWED_USERS=123456789
```

ID를 하드코딩하는 대신 DM 페어링을 사용할 수도 있습니다. 알 수 없는 사용자에게 일회성 페어링 코드가 전송되며, CLI에서 `hermes pairing approve <platform> <code>`로 승인할 수 있습니다. 신경 쓰이는 컴퓨터에서는 절대로 `GATEWAY_ALLOW_ALL_USERS=true`를 설정하지 마세요.

## 되돌리기 계층: 체크포인트와 `/rollback`

승인 게이트는 피해를 방지하고, [체크포인트](/user-guide/checkpoints-and-rollback)는 피해를 되돌립니다. 활성화하면 Hermes는 파괴적인 작업 전에 프로젝트의 스냅샷을 자동으로 만듭니다. `write_file`, `patch`, `rm`, `mv`, `sed -i`, `git reset` 같은 파괴적인 터미널 명령이 대상이며, 스냅샷은 `~/.hermes/checkpoints/store/` 아래의 별도 git 저장소에 저장됩니다. 실제 프로젝트의 `.git`은 절대 변경되지 않습니다.

체크포인트는 선택 사항입니다. 세션별로 활성화하세요.

```bash
hermes chat --checkpoints
```

또는 전역으로 활성화하세요.

```yaml
checkpoints:
  enabled: true
```

그런 다음 세션에서 다음 명령을 사용합니다.

| 명령 | 설명 |
|---------|-------------|
| `/rollback` | 변경 통계와 함께 모든 체크포인트 나열 |
| `/rollback diff <N>` | 체크포인트 N 이후 변경 사항 미리 보기 |
| `/rollback <N>` | 체크포인트 N으로 복원(마지막 채팅 턴도 되돌림) |
| `/rollback <N> <file>` | 체크포인트 N에서 파일 하나 복원 |

:::tip
복원하기 전에 `/rollback diff <N>`로 미리 보고, 최대한 안전하게 사용하려면 체크포인트를 git worktree와 함께 사용하세요. 각 Hermes 세션을 별도의 worktree에서 실행하고 체크포인트를 추가 계층으로 두면 됩니다.
:::

## 이 위협 모델의 범위와 한계

이 제어 기능이 무엇을 방어하는지 명확히 이해해야 합니다. [보안](/user-guide/security#user-defined-deny-rules-approvalsdeny) 가이드에서 설명하듯이:

> 거부 규칙은 정직하지만 잘못된 에이전트에 대한 안전장치이며, 위험 패턴 탐지기와 동일한 위협 모델을 사용합니다. 의도적으로 적대적인 프로세스를 위한 샌드박스는 아닙니다. 그런 경우에는 격리된 백엔드(Docker, Modal)나 송신이 제한된 환경을 사용하세요.

파일 쓰기 보호에도 같은 원칙이 적용됩니다. 보호 기능은 `write_file`과 `patch`에만 적용되고, `terminal` 도구는 동일한 OS 사용자로 실행됩니다. 거부 목록은 실수로 인한 피해를 줄이고 모델에 명확한 중지 신호를 제공하지만, 적대적이거나 침해된 에이전트를 샌드박스에 가두지는 않습니다. 요구 사항이 안전장치가 아니라 격리라면, 답은 격리된 터미널 백엔드입니다. 바로 그 경계를 위해 설계되었습니다.

## 신중하게 시작하기 위한 설정

위의 내용을 하나로 모은 설정입니다. `~/.hermes/config.yaml`에서 필요에 맞게 조정하세요.

```yaml
approvals:
  mode: manual                  # See every flagged command yourself
  timeout: 300                  # Unanswered prompts are denied (fail-closed)
  deny:                         # Never-run list — survives even /yolo
    - "git push --force*"
    - "*curl*|*sh*"
    - "dd if=* of=/dev/*"

security:
  redact_secrets: true          # Already the default; stated here for clarity

checkpoints:
  enabled: true                 # Snapshot before destructive operations

terminal:
  backend: docker               # Or ssh — keep execution off the host
  docker_forward_env: []        # No host secrets inside the container
```

쓰기 샌드박스를 사용하려면 `~/.hermes/.env`에도 다음을 추가하세요.

```bash
HERMES_WRITE_SAFE_ROOT=/path/to/project:/home/you/.hermes
```

## 함께 보기

- **[보안](/user-guide/security)** — 심층 방어 전체 참고 자료: 모든 승인 패턴, 컨테이너 강화 플래그, 게이트웨이 인증, MCP 자격 증명 필터링
- **[체크포인트 및 롤백](/user-guide/checkpoints-and-rollback)** — 설정, 저장소 유지 관리, 복원 작업 흐름
- **[도구 및 도구 세트](/user-guide/features/tools)** — 모든 터미널 백엔드와 설정
- **[설정](/user-guide/configuration)** — 전체 `config.yaml` 참고 자료
