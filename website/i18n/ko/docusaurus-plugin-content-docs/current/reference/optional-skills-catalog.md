---
sidebar_position: 9
title: "선택적 스킬 카탈로그"
description: "hermes-agent와 함께 제공되는 공식 선택적 스킬 — hermes skills install official/<category>/<skill>로 설치"
---

# 선택적 스킬 카탈로그

선택적 스킬은 `optional-skills/` 아래에서 hermes-agent와 함께 제공되지만 **기본적으로 활성화되지 않습니다**. 다음 명령으로 명시적으로 설치하세요.

```bash
hermes skills install official/<category>/<skill>
```

예시:

```bash
hermes skills install official/blockchain/solana
hermes skills install official/mlops/flash-attention
```

아래의 각 스킬은 전체 정의, 설정, 사용법이 담긴 전용 페이지로 연결됩니다.

제거하려면:

```bash
hermes skills uninstall <skill-name>
```

## autonomous-ai-agents

| 스킬 | 설명 |
|-------|-------------|
| [**antigravity-cli**](/docs/user-guide/skills/optional/autonomous-ai-agents/autonomous-ai-agents-antigravity-cli) | Antigravity CLI(agy)를 운영합니다: 플러그인, 인증, 샌드박스. |
| [**blackbox**](/docs/user-guide/skills/optional/autonomous-ai-agents/autonomous-ai-agents-blackbox) | Blackbox AI 멀티 모델 CLI에 코딩 작업을 위임합니다. |
| [**grok**](/docs/user-guide/skills/optional/autonomous-ai-agents/autonomous-ai-agents-grok) | xAI Grok Build CLI에 코딩을 위임합니다(기능, PR). |
| [**honcho**](/docs/user-guide/skills/optional/autonomous-ai-agents/autonomous-ai-agents-honcho) | Hermes용 Honcho 메모리를 설정하고 문제를 해결합니다. |
| [**openhands**](/docs/user-guide/skills/optional/autonomous-ai-agents/autonomous-ai-agents-openhands) | OpenHands CLI에 코딩을 위임합니다(모델에 구애받지 않음, LiteLLM). |

## blockchain

| 스킬 | 설명 |
|-------|-------------|
| [**evm**](/docs/user-guide/skills/optional/blockchain/blockchain-evm) | 읽기 전용 EVM 클라이언트: 8개 체인의 지갑, 토큰, 가스. |
| [**hyperliquid**](/docs/user-guide/skills/optional/blockchain/blockchain-hyperliquid) | Hyperliquid 시장 데이터, 계정 기록, 거래 검토. |
| [**solana**](/docs/user-guide/skills/optional/blockchain/blockchain-solana) | USD 기준으로 Solana 지갑, 토큰, 트랜잭션, NFT를 조회합니다. |

## communication

| 스킬 | 설명 |
|-------|-------------|
| [**one-three-one-rule**](/docs/user-guide/skills/optional/communication/communication-one-three-one-rule) | 1-3-1 의사결정 브리프: 문제, 세 가지 선택지, 하나의 추천안. |

## creative

| 스킬 | 설명 |
|-------|-------------|
| [**audiocraft-audio-generation**](/docs/user-guide/skills/optional/creative/creative-audiocraft-audio-generation) | AudioCraft: 텍스트를 음악으로 변환하는 MusicGen, 텍스트를 소리로 변환하는 AudioGen. |
| [**baoyu-article-illustrator**](/docs/user-guide/skills/optional/creative/creative-baoyu-article-illustrator) | 기사 일러스트: 유형 × 스타일 × 팔레트 일관성. |
| [**baoyu-comic**](/docs/user-guide/skills/optional/creative/creative-baoyu-comic) | 지식 만화: 교육, 전기, 튜토리얼. |
| [**concept-diagrams**](/docs/user-guide/skills/optional/creative/creative-concept-diagrams) | HTML로 평면적이고 간결한 교육용 SVG 시각 자료를 생성합니다. |
| [**creative-ideation**](/docs/user-guide/skills/optional/creative/creative-creative-ideation) | 창의적 실천에서 사용하는 이름 있는 방법으로 아이디어를 생성합니다. |
| [**heartmula**](/docs/user-guide/skills/optional/creative/creative-heartmula) | HeartMuLa: 가사 + 태그를 사용한 Suno 스타일 노래 생성. |
| [**hyperframes**](/docs/user-guide/skills/optional/creative/creative-hyperframes) | HTML 컴포지션에서 MP4/WebM 동영상을 렌더링합니다. |
| [**kanban-video-orchestrator**](/docs/user-guide/skills/optional/creative/creative-kanban-video-orchestrator) | 멀티 에이전트 동영상 제작 파이프라인을 계획하고 실행합니다. |
| [**meme-generation**](/docs/user-guide/skills/optional/creative/creative-meme-generation) | Pillow 텍스트 오버레이를 사용해 템플릿에서 밈 PNG를 만듭니다. |
| [**pixel-art**](/docs/user-guide/skills/optional/creative/creative-pixel-art) | 시대별 팔레트를 사용한 픽셀 아트(NES, Game Boy, PICO-8). |
| [**social-media-content-calendar**](/docs/user-guide/skills/optional/creative/creative-social-media-content-calendar) | 멀티 플랫폼 소셜 캠페인을 계획합니다: 브리프부터 게시까지. |
| [**tldraw-offline**](/docs/user-guide/skills/optional/creative/creative-tldraw-offline) | 오프라인 tldraw 캔버스를 에이전트로 조작하고 스크립팅합니다. |
| [**unreal-mcp**](/docs/user-guide/skills/optional/creative/creative-unreal-mcp) | Unreal Engine 에디터의 장면, 액터, 렌더링을 자동화합니다. |

## data-science

| 스킬 | 설명 |
|-------|-------------|
| [**jupyter-notebook**](/docs/user-guide/skills/optional/data-science/data-science-jupyter-notebook) | 실시간 Jupyter 커널(hamelnb)을 통한 반복적 Python 실행. |

## devops

| 스킬 | 설명 |
|-------|-------------|
| [**actual-setup**](/docs/user-guide/skills/optional/devops/devops-actual-setup) | Hermes에서 Actual Computer(actual.inc) 추론을 설정합니다. |
| [**docker-management**](/docs/user-guide/skills/optional/devops/devops-docker-management) | Docker 컨테이너, 이미지, 볼륨, Compose를 관리합니다. |
| [**hermes-s6-container-supervision**](/docs/user-guide/skills/optional/devops/devops-hermes-s6-container-supervision) | Hermes Docker 이미지의 s6 서비스를 수정하거나 디버깅합니다. |
| [**inference-sh-cli**](/docs/user-guide/skills/optional/devops/devops-inference-sh-cli) | inference.sh CLI로 150개 이상의 AI 앱(이미지, 동영상, LLM)을 실행합니다. |
| [**pinggy-tunnel**](/docs/user-guide/skills/optional/devops/devops-pinggy-tunnel) | Pinggy를 통해 SSH로 설치 없이 localhost 터널을 엽니다. |
| [**watchers**](/docs/user-guide/skills/optional/devops/devops-watchers) | 워터마크 중복 제거를 사용해 RSS, JSON API, GitHub를 폴링합니다. |

## dogfood

| 스킬 | 설명 |
|-------|-------------|
| [**adversarial-ux-test**](/docs/user-guide/skills/optional/dogfood/dogfood-adversarial-ux-test) | 적대적인 사용자를 연기해 UX 문제를 찾고 분류합니다. |

## email

| 스킬 | 설명 |
|-------|-------------|
| [**agentmail**](/docs/user-guide/skills/optional/email/email-agentmail) | 에이전트 전용 받은 편지함을 제공합니다: 이메일 송수신. |

## finance

| 스킬 | 설명 |
|-------|-------------|
| [**3-statement-model**](/docs/user-guide/skills/optional/finance/finance-3-statement-model) | Excel에서 통합 IS/BS/CF 재무 워크북을 작성합니다. |
| [**comps-analysis**](/docs/user-guide/skills/optional/finance/finance-comps-analysis) | Excel에서 비교기업 가치평가 워크북을 작성합니다. |
| [**dcf-model**](/docs/user-guide/skills/optional/finance/finance-dcf-model) | 할인현금흐름 가치평가 워크북을 작성합니다. |
| [**excel-author**](/docs/user-guide/skills/optional/finance/finance-excel-author) | openpyxl을 사용해 헤드리스 방식으로 감사 가능한 재무 워크북을 작성합니다. |
| [**lbo-model**](/docs/user-guide/skills/optional/finance/finance-lbo-model) | Excel에서 IRR/MOIC가 포함된 차입매수 워크북을 작성합니다. |
| [**merger-model**](/docs/user-guide/skills/optional/finance/finance-merger-model) | M&A 증액/희석 분석 워크북을 작성합니다. |
| [**polymarket**](/docs/user-guide/skills/optional/finance/finance-polymarket) | Polymarket을 조회합니다: 시장, 가격, 오더북, 기록. |
| [**pptx-author**](/docs/user-guide/skills/optional/finance/finance-pptx-author) | python-pptx를 사용해 헤드리스 방식으로 PowerPoint 프레젠테이션을 작성합니다. |
| [**stocks**](/docs/user-guide/skills/optional/finance/finance-stocks) | Yahoo를 통한 주가, 기록, 검색, 비교, 암호화폐. |

## gaming

| 스킬 | 설명 |
|-------|-------------|
| [**minecraft-modpack-server**](/docs/user-guide/skills/optional/gaming/gaming-minecraft-modpack-server) | 모드가 적용된 Minecraft 서버를 호스팅합니다(CurseForge, Modrinth). |
| [**pokemon-player**](/docs/user-guide/skills/optional/gaming/gaming-pokemon-player) | 헤드리스 에뮬레이터와 RAM 읽기를 사용해 Pokemon을 플레이합니다. |

## health

| 스킬 | 설명 |
|-------|-------------|
| [**fitness-nutrition**](/docs/user-guide/skills/optional/health/health-fitness-nutrition) | wger/USDA를 활용한 운동 계획, 매크로, 신체 지표. |
| [**neuroskill-bci**](/docs/user-guide/skills/optional/health/health-neuroskill-bci) | NeuroSkill의 실시간 BCI 인지 및 기분 상태를 사용합니다. |

## mcp

| 스킬 | 설명 |
|-------|-------------|
| [**fastmcp**](/docs/user-guide/skills/optional/mcp/mcp-fastmcp) | Python MCP 서버를 빌드, 테스트, 배포합니다. |
| [**mcp-oauth-remote-gateway**](/docs/user-guide/skills/optional/mcp/mcp-mcp-oauth-remote-gateway) | 헤드리스 게이트웨이에서 원격 MCP 서버를 위한 수동 OAuth를 수행합니다. |
| [**mcporter**](/docs/user-guide/skills/optional/mcp/mcp-mcporter) | 터미널에서 MCP 서버/도구를 나열하고, 인증하고, 호출합니다. |

## migration

| 스킬 | 설명 |
|-------|-------------|
| [**openclaw-migration**](/docs/user-guide/skills/optional/migration/migration-openclaw-migration) | OpenClaw 설정(메모리, 스킬)을 Hermes로 가져옵니다. |

## mlops

| 스킬 | 설명 |
|-------|-------------|
| [**accelerate**](/docs/user-guide/skills/optional/mlops/mlops-accelerate) | 최소한의 변경으로 여러 GPU에서 PyTorch 학습을 실행합니다. |
| [**axolotl**](/docs/user-guide/skills/optional/mlops/mlops-training-axolotl) | Axolotl: YAML LLM 파인튜닝(LoRA, DPO, GRPO). |
| [**chroma**](/docs/user-guide/skills/optional/mlops/mlops-chroma) | RAG 및 의미 검색을 위한 임베딩 데이터베이스. |
| [**clip**](/docs/user-guide/skills/optional/mlops/mlops-clip) | 제로샷 이미지 분류 및 이미지-텍스트 검색. |
| [**dspy**](/docs/user-guide/skills/optional/mlops/mlops-research-dspy) | DSPy: 선언적 LM 프로그램, 프롬프트 자동 최적화, RAG. |
| [**faiss**](/docs/user-guide/skills/optional/mlops/mlops-faiss) | 10억 개 규모의 빠른 벡터 유사도 검색. |
| [**flash-attention**](/docs/user-guide/skills/optional/mlops/mlops-flash-attention) | 긴 시퀀스 트랜스포머 학습 및 추론을 가속합니다. |
| [**guidance**](/docs/user-guide/skills/optional/mlops/mlops-guidance) | 문법으로 LLM 출력을 제한해 유효한 JSON을 보장합니다. |
| [**huggingface-tokenizers**](/docs/user-guide/skills/optional/mlops/mlops-huggingface-tokenizers) | 빠른 BPE/WordPiece 토큰화 및 사용자 지정 어휘 학습. |
| [**instructor**](/docs/user-guide/skills/optional/mlops/mlops-instructor) | Pydantic으로 검증하는 구조화된 LLM 출력. |
| [**lambda-labs**](/docs/user-guide/skills/optional/mlops/mlops-lambda-labs) | ML 학습을 위한 주문형 GPU 클라우드 인스턴스. |
| [**llava**](/docs/user-guide/skills/optional/mlops/mlops-llava) | 비전-언어 채팅: VQA, 캡션, 이미지 대화. |
| [**modal**](/docs/user-guide/skills/optional/mlops/mlops-modal) | ML 작업 및 모델 API를 위한 서버리스 GPU 클라우드. |
| [**nemo-curator**](/docs/user-guide/skills/optional/mlops/mlops-nemo-curator) | LLM 학습 데이터를 선별합니다: 중복 제거, 필터링, PII 삭제. |
| [**obliteratus**](/docs/user-guide/skills/optional/mlops/mlops-obliteratus) | OBLITERATUS: LLM 거부를 제거합니다(diff-in-means). |
| [**outlines**](/docs/user-guide/skills/optional/mlops/mlops-inference-outlines) | Outlines: 구조화된 JSON/정규식/Pydantic LLM 생성. |
| [**peft**](/docs/user-guide/skills/optional/mlops/mlops-peft) | 제한된 VRAM에서 LoRA로 대규모 모델을 파인튜닝합니다. |
| [**pinecone**](/docs/user-guide/skills/optional/mlops/mlops-pinecone) | 프로덕션 RAG 및 검색을 위한 관리형 벡터 DB. |
| [**pytorch-fsdp**](/docs/user-guide/skills/optional/mlops/mlops-pytorch-fsdp) | 대규모 모델을 위한 완전 샤딩 데이터 병렬 처리. |
| [**pytorch-lightning**](/docs/user-guide/skills/optional/mlops/mlops-pytorch-lightning) | 분산 지원이 내장된 깔끔한 학습 루프. |
| [**qdrant**](/docs/user-guide/skills/optional/mlops/mlops-qdrant) | 프로덕션 RAG를 위한 벡터 검색 엔진. |
| [**saelens**](/docs/user-guide/skills/optional/mlops/mlops-saelens) | 모델 특성을 해석하기 위한 희소 오토인코더 학습. |
| [**segment-anything-model**](/docs/user-guide/skills/optional/mlops/mlops-models-segment-anything-model) | SAM: 점, 상자, 마스크를 활용한 제로샷 이미지 분할. |
| [**simpo**](/docs/user-guide/skills/optional/mlops/mlops-simpo) | 참조가 필요 없는 선호도 정렬로 DPO보다 간단합니다. |
| [**slime**](/docs/user-guide/skills/optional/mlops/mlops-slime) | Megatron 및 SGLang을 활용한 LLM RL 후속 학습. |
| [**stable-diffusion**](/docs/user-guide/skills/optional/mlops/mlops-stable-diffusion) | 텍스트-이미지 생성, 인페인팅, img2img. |
| [**tensorrt-llm**](/docs/user-guide/skills/optional/mlops/mlops-tensorrt-llm) | NVIDIA GPU에서 고처리량 LLM 추론. |
| [**torchtitan**](/docs/user-guide/skills/optional/mlops/mlops-torchtitan) | PyTorch 4D 병렬 처리로 대규모 LLM을 사전 학습합니다. |
| [**trl-fine-tuning**](/docs/user-guide/skills/optional/mlops/mlops-training-trl-fine-tuning) | TRL: LLM RLHF 보상 모델링을 위한 SFT, DPO, GRPO, RLOO. |
| [**unsloth**](/docs/user-guide/skills/optional/mlops/mlops-training-unsloth) | VRAM을 적게 사용하면서 LoRA/QLoRA 파인튜닝을 2~5배 빠르게 수행합니다. |
| [**whisper**](/docs/user-guide/skills/optional/mlops/mlops-whisper) | 99개 언어로 음성을 전사하고 번역합니다. |

## payments

| 스킬 | 설명 |
|-------|-------------|
| [**mpp-agent**](/docs/user-guide/skills/optional/payments/payments-mpp-agent) | Machine Payments Protocol(MPP)을 통해 HTTP 402 API에 결제합니다. |
| [**stripe-link-cli**](/docs/user-guide/skills/optional/payments/payments-stripe-link-cli) | Stripe Link를 통한 에이전트 결제 — 카드, SPT, 승인. |
| [**stripe-projects**](/docs/user-guide/skills/optional/payments/payments-stripe-projects) | SaaS 서비스를 프로비저닝하고 Stripe Projects를 통해 인증 정보를 동기화합니다. |

## productivity

| 스킬 | 설명 |
|-------|-------------|
| [**canvas**](/docs/user-guide/skills/optional/productivity/productivity-canvas) | API 토큰으로 Canvas LMS 강좌와 과제를 가져옵니다. |
| [**here-now**](/docs/user-guide/skills/optional/productivity/productivity-here-now) | &#123;slug&#125;.here.now에 사이트를 게시하고 Drives에 파일을 저장합니다. |
| [**memento-flashcards**](/docs/user-guide/skills/optional/productivity/productivity-memento-flashcards) | 간격 반복 플래시카드: 생성, 복습, 퀴즈, 내보내기. |
| [**shop**](/docs/user-guide/skills/optional/productivity/productivity-shop) | 상점 카탈로그 검색, 결제, 주문 추적, 반품. |
| [**shopify**](/docs/user-guide/skills/optional/productivity/productivity-shopify) | curl을 통해 Shopify Admin/Storefront GraphQL API를 조회합니다. |
| [**siyuan**](/docs/user-guide/skills/optional/productivity/productivity-siyuan) | API를 통해 SiYuan 지식 기반을 조회하고 편집합니다. |
| [**telephony**](/docs/user-guide/skills/optional/productivity/productivity-telephony) | Twilio 번호, SMS/MMS, AI 아웃바운드 통화를 프로비저닝합니다. |

## research

| 스킬 | 설명 |
|-------|-------------|
| [**bioinformatics**](/docs/user-guide/skills/optional/research/research-bioinformatics) | 400개 이상의 유전체학 및 계산생물학 스킬로 연결되는 관문. |
| [**darwinian-evolver**](/docs/user-guide/skills/optional/research/research-darwinian-evolver) | Imbue의 진화 루프로 프롬프트/정규식/SQL/코드를 진화시킵니다. |
| [**domain-intel**](/docs/user-guide/skills/optional/research/research-domain-intel) | 서브도메인, SSL 인증서, WHOIS, DNS를 수동으로 정찰합니다. |
| [**drug-discovery**](/docs/user-guide/skills/optional/research/research-drug-discovery) | 신약 개발: ChEMBL 검색, 약물 유사성, 상호작용. |
| [**duckduckgo-search**](/docs/user-guide/skills/optional/research/research-duckduckgo-search) | ddgs를 통한 무료 무키 웹, 뉴스, 이미지 검색. |
| [**gitnexus-explorer**](/docs/user-guide/skills/optional/research/research-gitnexus-explorer) | 대화형 코드베이스 지식 그래프 웹 UI를 제공합니다. |
| [**osint-investigation**](/docs/user-guide/skills/optional/research/research-osint-investigation) | 공개 기록과 제재 데이터를 통해 자금 흐름을 추적합니다. |
| [**parallel-cli**](/docs/user-guide/skills/optional/research/research-parallel-cli) | 에이전트 기반 웹 검색, 심층 연구, 데이터 보강. |
| [**pinecone-research**](/docs/user-guide/skills/optional/research/research-pinecone-research) | Pinecone을 활용한 에이전트 RAG 및 장기 메모리. |
| [**qmd**](/docs/user-guide/skills/optional/research/research-qmd) | 노트, 문서, 트랜스크립트에 대한 하이브리드 로컬 검색. |
| [**scrapling**](/docs/user-guide/skills/optional/research/research-scrapling) | 스텔스 브라우징 및 Cloudflare 우회로 사이트를 스크랩합니다. |
| [**searxng-search**](/docs/user-guide/skills/optional/research/research-searxng-search) | 70개 이상의 엔진을 통합하는 무료 무키 메타 검색. |

## security

| 스킬 | 설명 |
|-------|-------------|
| [**1password**](/docs/user-guide/skills/optional/security/security-1password) | op CLI를 설정하고 로그인한 뒤 비밀을 읽거나 주입합니다. |
| [**godmode**](/docs/user-guide/skills/optional/security/security-godmode) | LLM을 탈옥합니다: Parseltongue, GODMODE, ULTRAPLINIAN. |
| [**oss-forensics**](/docs/user-guide/skills/optional/security/security-oss-forensics) | GitHub 공급망 포렌식: 복구, IOC, 보고. |
| [**sherlock**](/docs/user-guide/skills/optional/security/security-sherlock) | 400개 이상의 플랫폼에서 사용자 이름에 해당하는 계정을 찾습니다. |
| [**unbroker**](/docs/user-guide/skills/optional/security/security-unbroker) | 데이터 브로커 사이트에서 자신의 정보를 자동으로 삭제합니다. |
| [**web-pentest**](/docs/user-guide/skills/optional/security/security-web-pentest) | 승인된 웹 침투 테스트: 정찰, 증거 기반 익스플로잇, 보고서. |

## software-development

| 스킬 | 설명 |
|-------|-------------|
| [**code-wiki**](/docs/user-guide/skills/optional/software-development/software-development-code-wiki) | 모든 코드베이스를 위한 위키 문서와 Mermaid 다이어그램을 생성합니다. |
| [**rest-graphql-debug**](/docs/user-guide/skills/optional/software-development/software-development-rest-graphql-debug) | REST/GraphQL API를 디버깅합니다: 상태 코드, 인증, 스키마, 재현. |
| [**subagent-driven-development**](/docs/user-guide/skills/optional/software-development/software-development-subagent-driven-development) | delegate_task 서브에이전트로 2단계 검토를 거쳐 계획을 실행합니다. |

## web-development

| 스킬 | 설명 |
|-------|-------------|
| [**cloudflare-temporary-deploy**](/docs/user-guide/skills/optional/web-development/web-development-cloudflare-temporary-deploy) | 계정 없이 wrangler --temporary로 Worker를 실시간 배포합니다. |
| [**page-agent**](/docs/user-guide/skills/optional/web-development/web-development-page-agent) | 웹 앱에 페이지 내 자연어 GUI 코파일럿을 삽입합니다. |

## yuanbao

| 스킬 | 설명 |
|-------|-------------|
| [**yuanbao**](/docs/user-guide/skills/optional/yuanbao/yuanbao-yuanbao) | Yuanbao 그룹: @mention으로 사용자를 언급하고 정보/멤버를 조회합니다. |

---

## 선택적 스킬 기여

새 선택적 스킬을 저장소에 추가하려면:

1. `optional-skills/<category>/<skill-name>/` 아래에 디렉터리를 만듭니다.
2. 표준 프런트매터(name, description, version, author)가 포함된 `SKILL.md`를 추가합니다.
3. `references/`, `templates/`, `scripts/` 하위 디렉터리에 필요한 지원 파일을 포함합니다.
4. 풀 리퀘스트를 제출합니다 — 스킬은 이 카탈로그에 표시되고 병합되면 자체 문서 페이지도 제공됩니다.
