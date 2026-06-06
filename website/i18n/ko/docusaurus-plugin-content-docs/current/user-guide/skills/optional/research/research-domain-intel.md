---
title: "Domain Intel — Python 표준 라이브러리를 사용한 패시브 도메인 정찰"
sidebar_label: "Domain Intel"
description: "Python 표준 라이브러리를 사용한 패시브 도메인 정찰"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Domain Intel

Python 표준 라이브러리만을 사용하여 패시브(수동적) 도메인 정찰을 수행합니다. 서브도메인 발견, SSL 인증서 검사, WHOIS 조회, DNS 레코드, 도메인 사용 가능성 확인, 다중 도메인 대량 분석 기능을 지원합니다. API 키가 필요하지 않습니다.

## 스킬 메타데이터 (Skill metadata)

| | |
|---|---|
| Source | Optional — `hermes skills install official/research/domain-intel` 명령으로 설치 |
| Path | `optional-skills/research/domain-intel` |
| Platforms | linux, macos, windows |

## 참조: 전체 SKILL.md (Reference: full SKILL.md)

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 이것은 스킬이 활성화되었을 때 에이전트가 지시사항으로 보는 내용입니다.
:::

# Domain Intelligence — Passive OSINT

Python 표준 라이브러리만을 사용한 패시브 도메인 정찰.
**의존성 제로(Zero dependencies). API 키 제로. Linux, macOS, Windows에서 모두 작동합니다.**

## 헬퍼 스크립트 (Helper script)

이 스킬은 모든 도메인 인텔리전스 작업을 수행할 수 있는 완벽한 CLI 도구인 `scripts/domain_intel.py`를 포함합니다.

```bash
# Certificate Transparency 로그를 통한 서브도메인 발견
python3 SKILL_DIR/scripts/domain_intel.py subdomains example.com

# SSL 인증서 검사 (만료일, 암호, SANs, 발급자)
python3 SKILL_DIR/scripts/domain_intel.py ssl example.com

# WHOIS 조회 (등록기관, 등록일, 네임서버 — 100개 이상의 TLD 지원)
python3 SKILL_DIR/scripts/domain_intel.py whois example.com

# DNS 레코드 (A, AAAA, MX, NS, TXT, CNAME)
python3 SKILL_DIR/scripts/domain_intel.py dns example.com

# 도메인 사용 가능성 확인 (패시브: DNS + WHOIS + SSL 신호 활용)
python3 SKILL_DIR/scripts/domain_intel.py available coolstartup.io

# 대량 분석 — 다중 도메인, 병렬 다중 검사
python3 SKILL_DIR/scripts/domain_intel.py bulk example.com github.com google.com
python3 SKILL_DIR/scripts/domain_intel.py bulk example.com github.com --checks ssl,dns
```

`SKILL_DIR`은 이 SKILL.md 파일이 포함된 디렉토리입니다. 모든 출력은 구조화된 JSON 형태입니다.

## 사용 가능한 명령어 (Available commands)

| Command | What it does | Data source |
|---------|-------------|-------------|
| `subdomains` | 인증서 로그에서 서브도메인 찾기 | crt.sh (HTTPS) |
| `ssl` | TLS 인증서 세부정보 검사 | 타겟 서버로의 직접 TCP:443 연결 |
| `whois` | 등록 정보, 등록기관, 날짜 | WHOIS 서버 (TCP:43) |
| `dns` | A, AAAA, MX, NS, TXT, CNAME 레코드 | 시스템 DNS + Google DoH |
| `available` | 도메인이 등록되어 있는지 확인 | DNS + WHOIS + SSL 신호 |
| `bulk` | 다중 도메인에 대해 다중 검사 실행 | 위의 모든 데이터 소스 |

## 내장 도구와 비교 — 언제 무엇을 사용할 것인가

- 인프라 질문의 경우 **이 스킬을 사용하세요**: 서브도메인, SSL 인증서, WHOIS, DNS 레코드, 사용 가능성 여부
- 특정 도메인/회사가 무엇을 하는지에 대한 일반적인 연구의 경우 **`web_search`를 사용하세요**
- 웹페이지의 실제 콘텐츠를 가져오려면 **`web_extract`를 사용하세요**
- URL이 도달 가능한지(reachable) 확인하는 간단한 검사에는 **`curl -I`와 함께 `terminal`을 사용하세요**

| Task | Better tool | Why |
|------|-------------|-----|
| "example.com은 어떤 사이트인가요?" | `web_extract` | DNS/WHOIS 데이터가 아닌 웹페이지 내용을 가져옴 |
| "회사에 대한 정보 찾기" | `web_search` | 도메인 한정이 아닌 일반적인 리서치 |
| "이 웹사이트는 안전한가요?" | `web_search` | 평판 확인에는 웹 문맥이 필요함 |
| "URL에 접속 가능한지 확인하기" | `terminal` with `curl -I` | 간단한 HTTP 상태 확인 |
| "X의 서브도메인 찾기" | **This skill** | 이를 위한 유일한 패시브(수동적) 소스 |
| "SSL 인증서가 언제 만료되나요?" | **This skill** | 내장 도구는 TLS를 검사할 수 없음 |
| "누가 이 도메인을 등록했나요?" | **This skill** | 웹 검색에서는 WHOIS 데이터가 잘 나오지 않음 |
| "coolstartup.io를 사용할 수 있나요?" | **This skill** | DNS+WHOIS+SSL을 통한 수동적인 가용성 확인 |

## 플랫폼 호환성 (Platform compatibility)

순수 Python 표준 라이브러리(`socket`, `ssl`, `urllib`, `json`, `concurrent.futures`)만 사용합니다.
외부 종속성 없이 Linux, macOS, Windows에서 동일하게 작동합니다.

- **crt.sh 쿼리**는 HTTPS(포트 443)를 사용합니다 — 대부분의 방화벽 뒤에서 작동합니다.
- **WHOIS 쿼리**는 TCP 포트 43을 사용합니다 — 제한된 네트워크에서는 차단될 수 있습니다.
- **DNS 쿼리**는 MX/NS/TXT 레코드 조회 시 Google DoH(HTTPS)를 사용합니다 — 방화벽 친화적입니다.
- **SSL 확인**은 대상 포트 443에 연결합니다 — 유일한 "액티브(active)" 작업입니다.

## 데이터 소스 (Data sources)

모든 쿼리는 **패시브(수동적)**이며 포트 스캔이나 취약성 테스트를 수행하지 않습니다:

- **crt.sh** — Certificate Transparency 로그 (서브도메인 발견, HTTPS 전용)
- **WHOIS servers** — 100개 이상의 공인 TLD 등록기관으로 향하는 직접적인 TCP 통신
- **Google DNS-over-HTTPS** — MX, NS, TXT, CNAME 확인 (방화벽 친화적)
- **System DNS** — A/AAAA 레코드 확인
- **SSL check**은 유일한 "액티브(active)" 작업입니다 (대상 포트 443으로의 TCP 연결)

## 참고 (Notes)

- WHOIS 쿼리는 TCP 포트 43을 사용하므로, 제한된 네트워크에서는 막힐 수 있습니다.
- 일부 WHOIS 서버는 (GDPR 등으로 인해) 등록자 정보를 가립니다 — 이에 대해 사용자에게 언급해주세요.
- crt.sh는 (수천 개의 인증서가 있는) 아주 인기 있는 도메인의 경우 느려질 수 있습니다 — 적절한 기대치를 설정하세요.
- 도메인 사용 가능성 확인(availability check)은 경험적(heuristic) 방법(3가지 패시브 신호)에 기반하며 등록기관 API처럼 공신력이 있지는 않습니다.

---

*기여자: [@FurkanL0](https://github.com/FurkanL0)*
