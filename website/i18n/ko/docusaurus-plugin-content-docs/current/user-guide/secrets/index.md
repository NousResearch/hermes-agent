# 비밀키 (Secrets)

Hermes는 `~/.hermes/.env`에 API 키를 직접 저장하는 대신 프로세스가 시작될 때 외부 비밀키 관리자에서 가져올 수 있습니다. 비밀키 관리자에 접근하기 위한 부트스트랩 토큰만 `.env` 파일에 보관하고, 그 외의 다른 제공자 키(OpenAI, Anthropic, OpenRouter 등)는 비밀키 관리자 내부에 보관하여 중앙에서 교체(rotate) 및 관리할 수 있습니다.

지원 목록:

- [Bitwarden 비밀 관리자 (Bitwarden Secrets Manager)](./bitwarden) — `bws` CLI, 지연 설치(lazy-install), 무료 티어로 사용 가능.

동일한 인터페이스 뒤에 더 많은 백엔드(Vault, AWS Secrets Manager, 1Password CLI)를 쉽게 추가할 수 있습니다 — `agent/secret_sources/`의 모듈 하나와 CLI 핸들러 하나만 있으면 됩니다. 원하시는 백엔드가 있다면 요청을 남겨주세요.
