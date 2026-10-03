---
title: "도메인 인텔리전스 — 서브도메인, SSL 인증서, WHOIS 및 DNS에 대한 수동 정찰"
sidebar_label: "도메인 인텔리전스"
description: "서브도메인, SSL 인증서, WHOIS 및 DNS에 대한 수동 정찰"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# 도메인 인텔리전스

서브도메인, SSL 인증서, WHOIS 및 DNS에 대한 수동 정찰입니다.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | 선택 사항 — `hermes skills install official/research/domain-intel`로 설치 |
| 경로 | `optional-skills/research/domain-intel` |
| 버전 | `1.0.0` |
| 작성자 | FurkanL0, Hermes Agent |
| 라이선스 | MIT |
| 플랫폼 | linux, macos, windows |
| 태그 | `Domains`, `OSINT`, `DNS`, `Research` |

## 참고: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 스킬이 활성화되면 에이전트가 지침으로 확인하는 내용입니다.
:::

# 도메인 인텔리전스 — 수동 OSINT

Python 표준 라이브러리만 사용하는 수동 도메인 정찰입니다.
**종속성 0개. API 키 0개. Linux, macOS, Windows에서 작동합니다.**

## 도우미 스크립트

이 스킬에는 모든 도메인 인텔리전스 작업을 위한 완전한 CLI 도구인 `scripts/domain_intel.py`가 포함되어 있습니다.

```bash
# Subdomain discovery via Certificate Transparency logs
python3 SKILL_DIR/scripts/domain_intel.py subdomains example.com

# SSL certificate inspection (expiry, cipher, SANs, issuer)
python3 SKILL_DIR/scripts/domain_intel.py ssl example.com

# WHOIS lookup (registrar, dates, name servers — 100+ TLDs)
python3 SKILL_DIR/scripts/domain_intel.py whois example.com

# DNS records (A, AAAA, MX, NS, TXT, CNAME)
python3 SKILL_DIR/scripts/domain_intel.py dns example.com

# Domain availability check (passive: DNS + WHOIS + SSL signals)
python3 SKILL_DIR/scripts/domain_intel.py available coolstartup.io

# Bulk analysis — multiple domains, multiple checks in parallel
python3 SKILL_DIR/scripts/domain_intel.py bulk example.com github.com google.com
python3 SKILL_DIR/scripts/domain_intel.py bulk example.com github.com --checks ssl,dns
```

`SKILL_DIR`은 이 SKILL.md 파일이 들어 있는 디렉터리입니다. 모든 출력은 구조화된 JSON입니다.

## 사용 가능한 명령

| 명령 | 기능 | 데이터 출처 |
|---------|-------------|-------------|
| `subdomains` | 인증서 로그에서 서브도메인 찾기 | crt.sh (HTTPS) |
| `ssl` | TLS 인증서 세부 정보 검사 | 대상에 직접 TCP:443 연결 |
| `whois` | 등록 정보, 등록기관, 날짜 | WHOIS 서버 (TCP:43) |
| `dns` | A, AAAA, MX, NS, TXT, CNAME 레코드 | 시스템 DNS + Google DoH |
| `available` | 도메인 등록 여부 확인 | DNS + WHOIS + SSL 신호 |
| `bulk` | 여러 도메인에서 여러 검사 실행 | 위 모든 출처 |

## 이 도구와 기본 제공 도구 중 선택

- **인프라 관련 질문에는 이 스킬을 사용합니다:** 서브도메인, SSL 인증서, WHOIS, DNS 레코드, 사용 가능 여부
- **도메인/회사의 일반적인 조사에는 `web_search`를 사용합니다**
- **웹페이지의 실제 내용을 가져오려면 `web_extract`를 사용합니다**
- **URL에 간단히 연결 가능한지 확인하려면 `curl -I`와 함께 `terminal`을 사용합니다**

| 작업 | 더 적합한 도구 | 이유 |
|------|-------------|-----|
| "example.com은 무엇을 하나요?" | `web_extract` | DNS/WHOIS 데이터가 아니라 페이지 내용을 가져옵니다 |
| "회사에 대한 정보를 찾아줘" | `web_search` | 도메인에 국한되지 않은 일반 조사입니다 |
| "이 웹사이트는 안전한가요?" | `web_search` | 평판 확인에는 웹 맥락이 필요합니다 |
| "URL에 연결 가능한지 확인해줘" | `curl -I`와 함께 `terminal` | 간단한 HTTP 확인입니다 |
| "X의 서브도메인을 찾아줘" | **이 스킬** | 이를 위한 유일한 수동 출처입니다 |
| "SSL 인증서는 언제 만료되나요?" | **이 스킬** | 기본 제공 도구로는 TLS를 검사할 수 없습니다 |
| "이 도메인을 등록한 사람은 누구인가요?" | **이 스킬** | WHOIS 데이터는 웹 검색에 포함되지 않습니다 |
| "coolstartup.io를 사용할 수 있나요?" | **이 스킬** | DNS+WHOIS+SSL을 통한 수동 사용 가능 여부 확인입니다 |

## 플랫폼 호환성

순수 Python 표준 라이브러리(`socket`, `ssl`, `urllib`, `json`, `concurrent.futures`)만 사용합니다.
종속성 없이 Linux, macOS, Windows에서 동일하게 작동합니다.

- **crt.sh 쿼리**는 HTTPS(포트 443)를 사용하므로 대부분의 방화벽 뒤에서 작동합니다
- **WHOIS 쿼리**는 TCP 포트 43을 사용하므로 제한적인 네트워크에서 차단될 수 있습니다
- **DNS 쿼리**는 MX/NS/TXT에 Google DoH(HTTPS)를 사용하므로 방화벽 친화적입니다
- **SSL 검사**는 대상의 포트 443에 연결하는 유일한 "능동" 작업입니다

## 데이터 출처

모든 쿼리는 **수동**입니다 — 포트 스캔이나 취약점 테스트를 하지 않습니다.

- **crt.sh** — Certificate Transparency 로그(서브도메인 검색, HTTPS 전용)
- **WHOIS 서버** — 100개 이상의 권위 있는 TLD 등록기관에 직접 TCP 연결
- **Google DNS-over-HTTPS** — MX, NS, TXT, CNAME 확인(방화벽 친화적)
- **시스템 DNS** — A/AAAA 레코드 확인
- **SSL 검사**는 유일한 "능동" 작업입니다(대상:443에 대한 TCP 연결)

## 참고 사항

- WHOIS 쿼리는 TCP 포트 43을 사용하므로 제한적인 네트워크에서 차단될 수 있습니다
- 일부 WHOIS 서버는 등록자 정보를 GDPR에 따라 삭제합니다 — 사용자에게 이를 알려야 합니다
- 매우 인기 있는 도메인(수천 개의 인증서)의 경우 crt.sh가 느릴 수 있습니다 — 합리적인 기대치를 안내하세요
- 사용 가능 여부 검사는 3개의 수동 신호를 기반으로 한 휴리스틱이며 등록기관 API처럼 권위 있는 결과가 아닙니다

---

*[@FurkanL0](https://github.com/FurkanL0)가 기여했습니다*
