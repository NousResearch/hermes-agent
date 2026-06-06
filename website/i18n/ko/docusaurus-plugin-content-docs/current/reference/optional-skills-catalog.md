---
sidebar_position: 13
title: 허브 스킬 (Hub Skills)
description: Hermes 스킬 허브에서 설치할 수 있는 인기 있는 공식 스킬의 선별된 디렉토리입니다.
---

# 공식 허브 스킬 카탈로그 (Official Hub Skills)

다음은 Hermes Skills Hub(`official/` 네임스페이스)에서 사용할 수 있는 선택적(optional) 스킬의 참조 목록입니다. 허브의 모든 스킬은 터미널, 대시보드 또는 봇과의 직접 채팅을 통해 설치할 수 있습니다.

> **참고:** 이 카탈로그는 선별된 공식 목록일 뿐입니다. 전체 커뮤니티 카탈로그를 탐색하려면 `hermes skills browse` 또는 대시보드를 사용하세요.

## 설치 및 관리

### CLI 명령

```bash
# 스킬 설치 (hub 네임스페이스는 선택 사항이며, 기본값은 'official'입니다)
hermes skills install official/browser-automation/playwright
hermes skills install browser-automation/playwright

# 전체 허브 카탈로그 검색
hermes skills search "github"

# 터미널에서 대화형으로 설치 가능한 스킬 둘러보기
hermes skills browse
```

### 봇을 통해 (채팅 사용)

Telegram, Discord 또는 기타 메시징 플랫폼에서 봇에게 설치를 요청하기만 하면 됩니다:

> **사용자:** "Install the Jira API skill"
>
> **Hermes:** "I found `official/project-management/jira-api`. Would you like me to install it?"
>
> **사용자:** "Yes"
>
> **Hermes:** [Runs `hermes skills install`] "Installed! You'll need to add your Jira API token to your `.env` file..."

---

## 🌐 브라우저 자동화 (Browser Automation)

에이전트가 헤드리스(headless) 브라우저를 통해 최신 웹 애플리케이션과 상호 작용할 수 있도록 합니다.

### Playwright Web Controller
`official/browser-automation/playwright`

웹 애플리케이션에 대한 에이전트 브라우징, 상호 작용 및 테스트 자동화를 위한 완전한 Playwright 환경입니다. 단순한 스크래핑이 불가능한 SPA 및 인증된 세션을 처리합니다.
- **필수 사항:** Node.js, `npm install -g playwright`

### Selenium WebDriver
`official/browser-automation/selenium`
Playwright의 대안으로, 레거시 시스템 또는 특정 엔터프라이즈 환경이 필요한 경우에 사용합니다.

---

## 🛠️ 클라우드 인프라 (Cloud Infrastructure)

AWS, GCP 및 Azure 리소스를 안전하게 관리, 쿼리 및 배포하기 위한 스킬입니다.

### AWS CLI Master
`official/cloud-infrastructure/aws-cli`
AWS CLI 도구의 고급 사용법입니다. 보안 모범 사례에 따라 EC2 리소스, S3 버킷 및 CloudWatch 로그를 관리합니다.
- **필수 사항:** AWS CLI 설치됨, 자격 증명 구성됨 (`~/.aws/credentials`)

### GCP gcloud Expert
`official/cloud-infrastructure/gcp-gcloud`
Compute Engine, Cloud Storage 및 GKE 관리를 위한 `gcloud` CLI 관리 스킬입니다.
- **필수 사항:** Google Cloud SDK

### Terraform Plan & Apply
`official/cloud-infrastructure/terraform`
Terraform 구성을 안전하게 생성하고 `terraform plan`/`apply`를 실행하며 상태 변경 사항을 분석하는 스킬입니다.

---

## 📊 데이터 과학 및 분석 (Data Science & Analytics)

데이터 조작, 시각화 및 머신 러닝 파이프라인을 위한 도구입니다.

### Pandas & Jupyter
`official/data-science/pandas-jupyter`
에이전트가 pandas를 사용하여 스크립트를 작성하고, 데이터 프레임을 분석하고, Jupyter 환경(커널 스피커) 내에서 샌드박스 코드를 실행하도록 지시합니다.
- **필수 사항:** `pip install pandas jupyter`

### SQL Database Query Engine
`official/data-science/sql-engine`
PostgreSQL, MySQL, SQLite에 대한 읽기 전용 구조 분석 및 스키마 인식 쿼리를 수행합니다.
- **필수 사항:** 적절한 드라이버 (`psycopg2`, `pymysql` 등)

---

## 🧑‍💻 프로그래밍 및 프레임워크 (Programming & Frameworks)

특정 프레임워크 또는 언어 환경을 위한 코딩 표준 및 전문 지식입니다.

### React & Next.js 14+ (App Router)
`official/programming/react-nextjs`
최신 Next.js App Router, 서버 컴포넌트(Server Components), Tailwind CSS 및 Zustand/Redux 상태 관리에 대한 최신 지식입니다. 현재 권장되는 패턴이 아닌 오래된 React 패턴을 피하도록 안내합니다.

### Python FastAPI Backend
`official/programming/python-fastapi`
Pydantic v2, SQLAlchemy 2.0 및 비동기 워크플로를 사용한 프로덕션 등급 FastAPI 개발을 위한 템플릿 및 모범 사례입니다.

### Rust CLI Development
`official/programming/rust-cli`
clap, tokio 및 serde를 사용하여 안정적이고 효율적인 명령줄 도구를 구축하기 위한 Rust 코딩 지침입니다.

---

## 📈 프로젝트 관리 (Project Management)

이슈 트래커 및 프로젝트 관리 플랫폼과 연동합니다.

### Jira API 연동 (Jira API Integration)
`official/project-management/jira-api`
에이전트가 Jira 이슈를 쿼리, 생성, 업데이트 및 전환할 수 있도록 하는 안전한 스크립트 기반 상호 작용입니다. 에이전트가 JQL 문법을 이해하도록 돕습니다.
- **필수 사항:** `JIRA_URL`, `JIRA_USER`, `JIRA_TOKEN` 환경 변수

### Linear App Client
`official/project-management/linear-app`
Linear의 GraphQL API와의 상호 작용입니다. 이슈를 생성하고 주기를 추적합니다.
- **필수 사항:** `LINEAR_API_KEY` 환경 변수

---

## 🔒 보안 및 네트워킹 (Security & Networking)

네트워크 분석, 검사 및 취약성 평가를 위한 도구입니다.

### Nmap Network Scanner
`official/security/nmap`
nmap 결과를 안전하게 구성하고 실행하고 해석합니다. 승인되지 않은 검사를 방지하기 위한 안전 가드레일이 포함되어 있습니다.
- **필수 사항:** `nmap` 바이너리가 시스템 PATH에 있어야 합니다.

### Wireshark / TShark Analysis
`official/security/tshark-analysis`
패킷 캡처 파일(pcap)을 검사하고 비정상적인 트래픽을 필터링 및 진단합니다.
- **필수 사항:** `tshark` 바이너리가 시스템 PATH에 있어야 합니다.

---

## 🤖 자율 에이전트 확장 (Autonomous Agent Extensions)

Hermes의 핵심 기능을 향상시키는 고급 스킬입니다.

### 에이전트 간 위임 (Cross-Agent Delegation)
`official/agent-extensions/delegation`
Hermes가 서브 에이전트를 생성하거나 특정 시스템 도구를 갖춘 다른 독립적인 에이전트 프레임워크(예: AutoGPT, BabyAGI)에 컨텍스트를 전달할 수 있도록 하는 프로토콜 및 스크립트입니다.

### 장기 코드 메모리 (Long-term Code Memory)
`official/agent-extensions/chroma-code-search`
수천 개의 파일에 걸쳐 저장소(repository)를 색인화하기 위해 ChromaDB의 로컬 인스턴스를 사용하여 프로젝트별 벡터 시맨틱 검색 기능을 에이전트에게 제공합니다.
- **필수 사항:** `pip install chromadb sentence-transformers`

---

## 새로운 허브 스킬 게시 (Publishing a New Hub Skill)

유용한 스킬을 만들었고 이를 공식 카탈로그에 공유하고 싶다면:

1. [Hermes Agent 레포지토리](https://github.com/NousResearch/hermes-agent)를 포크(Fork)합니다.
2. `hub/official/your-category/your-skill-name/`에 스킬 폴더를 생성합니다.
3. 설명, 지침 및 필수 메타데이터가 포함된 `SKILL.md`를 포함시킵니다.
4. 모든 종속성(스크립트, 요구사항 파일 등)이 폴더 내에 포함되어 있는지 확인합니다.
5. GitHub에서 풀 리퀘스트(Pull Request)를 제출하세요!
