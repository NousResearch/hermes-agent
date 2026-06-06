---
title: "Docker Management"
sidebar_label: "Docker Management"
description: "Docker 컨테이너, 이미지, 볼륨, 네트워크 및 Compose 스택 관리 — 수명 주기 운영, 디버깅, 정리 및 Dockerfile 최적화"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Docker Management

Docker 컨테이너, 이미지, 볼륨, 네트워크 및 Compose 스택을 관리합니다 — 수명 주기 작업(lifecycle ops), 디버깅, 정리 및 Dockerfile 최적화를 다룹니다.

## Skill metadata

| | |
|---|---|
| Source | Optional — install with `hermes skills install official/devops/docker-management` |
| Path | `optional-skills/devops/docker-management` |
| Version | `1.0.0` |
| Author | sprmn24 |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `docker`, `containers`, `devops`, `infrastructure`, `compose`, `images`, `volumes`, `networks`, `debugging` |

## Reference: full SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 이것은 스킬이 활성화되었을 때 에이전트가 지시 사항으로 보는 내용입니다.
:::

# Docker Management

표준 Docker CLI 명령을 사용하여 Docker 컨테이너, 이미지, 볼륨, 네트워크 및 Compose 스택을 관리합니다. Docker 자체 이외의 추가 종속성은 없습니다.

## When to Use

- 컨테이너 실행, 중지, 다시 시작, 제거 또는 검사
- Docker 이미지 빌드, 풀(pull), 푸시(push), 태그 지정 또는 정리
- Docker Compose(다중 서비스 스택) 작업
- 볼륨 또는 네트워크 관리
- 충돌하는 컨테이너 디버깅 또는 로그 분석
- Docker 디스크 사용량 확인 또는 공간 확보
- Dockerfile 검토 또는 최적화

## Prerequisites

- Docker Engine 설치 및 실행 중
- 사용자가 `docker` 그룹에 추가됨 (또는 `sudo` 사용)
- Docker Compose v2 (최신 Docker 설치에 포함됨)

빠른 확인:

```bash
docker --version && docker compose version
```

## Quick Reference

| Task | Command |
|------|---------|
| 컨테이너 실행 (백그라운드) | `docker run -d --name NAME IMAGE` |
| 중지 + 제거 | `docker stop NAME && docker rm NAME` |
| 로그 보기 (팔로우) | `docker logs --tail 50 -f NAME` |
| 컨테이너 내부 셸 접속 | `docker exec -it NAME /bin/sh` |
| 모든 컨테이너 목록 | `docker ps -a` |
| 이미지 빌드 | `docker build -t TAG .` |
| Compose 시작 | `docker compose up -d` |
| Compose 중지 | `docker compose down` |
| 디스크 사용량 | `docker system df` |
| 매달린(dangling) 항목 정리 | `docker image prune && docker container prune` |

## Procedure

### 1. Identify the domain

요청이 속하는 영역을 파악합니다:

- **컨테이너 수명 주기(lifecycle)** → run, stop, start, restart, rm, pause/unpause
- **컨테이너 상호 작용** → exec, cp, logs, inspect, stats
- **이미지 관리** → build, pull, push, tag, rmi, save/load
- **Docker Compose** → up, down, ps, logs, exec, build, config
- **볼륨 및 네트워크** → create, inspect, rm, prune, connect
- **문제 해결(Troubleshooting)** → 로그 분석, 종료 코드, 리소스 문제

### 2. Container operations

**새 컨테이너 실행:**

```bash
# 포트 매핑을 사용한 분리(detached) 서비스
docker run -d --name web -p 8080:80 nginx

# 환경 변수와 함께
docker run -d -e POSTGRES_PASSWORD=secret -e POSTGRES_DB=mydb --name db postgres:16

# 영구 데이터 포함 (기명 볼륨 - named volume)
docker run -d -v pgdata:/var/lib/postgresql/data --name db postgres:16

# 개발용 (소스 코드 바인드 마운트)
docker run -d -v $(pwd)/src:/app/src -p 3000:3000 --name dev my-app

# 대화형 디버깅 (종료 시 자동 제거)
docker run -it --rm ubuntu:22.04 /bin/bash

# 리소스 제한 및 재시작 정책과 함께
docker run -d --memory=512m --cpus=1.5 --restart=unless-stopped --name app my-app
```

주요 플래그: `-d` 분리 모드, `-it` 대화형+tty, `--rm` 자동 제거, `-p` 포트 (호스트:컨테이너), `-e` 환경 변수, `-v` 볼륨, `--name` 이름, `--restart` 재시작 정책.

**실행 중인 컨테이너 관리:**

```bash
docker ps                        # 실행 중인 컨테이너
docker ps -a                     # 모든 컨테이너 (중지된 것 포함)
docker stop NAME                 # 정상(graceful) 중지
docker start NAME                # 중지된 컨테이너 시작
docker restart NAME              # 중지 + 시작
docker rm NAME                   # 중지된 컨테이너 제거
docker rm -f NAME                # 실행 중인 컨테이너 강제 제거
docker container prune           # 모든 중지된 컨테이너 제거
```

**컨테이너와 상호 작용:**

```bash
docker exec -it NAME /bin/sh          # 셸 액세스 (가능한 경우 /bin/bash 사용)
docker exec NAME env                   # 환경 변수 보기
docker exec -u root NAME apt update    # 특정 사용자로 실행
docker logs --tail 100 -f NAME         # 마지막 100줄 팔로우
docker logs --since 2h NAME            # 최근 2시간 동안의 로그
docker cp NAME:/path/file ./local      # 컨테이너에서 파일 복사
docker cp ./file NAME:/path/           # 컨테이너로 파일 복사
docker inspect NAME                    # 컨테이너 전체 상세 정보 (JSON)
docker stats --no-stream               # 리소스 사용량 스냅샷
docker top NAME                        # 실행 중인 프로세스
```

### 3. Image management

```bash
# 빌드
docker build -t my-app:latest .
docker build -t my-app:prod -f Dockerfile.prod .
docker build --no-cache -t my-app .              # 캐시 없이 새로 빌드
DOCKER_BUILDKIT=1 docker build -t my-app .       # BuildKit으로 더 빠르게 빌드

# 풀(Pull) 및 푸시(Push)
docker pull node:20-alpine
docker login ghcr.io
docker tag my-app:latest registry/my-app:v1.0
docker push registry/my-app:v1.0

# 검사
docker images                          # 로컬 이미지 목록
docker history IMAGE                   # 레이어 보기
docker inspect IMAGE                   # 전체 상세 정보

# 정리
docker image prune                     # 매달린(dangling, 태그 없는) 이미지 제거
docker image prune -a                  # 사용하지 않는 모든 이미지 제거 (주의!)
docker image prune -a --filter "until=168h"   # 7일 이상 된 사용하지 않는 이미지
```

### 4. Docker Compose

```bash
# 시작/중지
docker compose up -d                   # 모든 서비스를 분리 모드로 시작
docker compose up -d --build           # 시작하기 전에 이미지 재빌드
docker compose down                    # 컨테이너 중지 및 제거
docker compose down -v                 # 볼륨도 제거 (데이터가 파괴됨)

# 모니터링
docker compose ps                      # 서비스 목록
docker compose logs -f api             # 특정 서비스의 로그 팔로우
docker compose logs --tail 50          # 모든 서비스의 마지막 50줄

# 상호 작용
docker compose exec api /bin/sh        # 실행 중인 서비스 내부 셸 접속
docker compose run --rm api npm test   # 일회성 명령 (새 컨테이너)
docker compose restart api             # 특정 서비스 다시 시작

# 검증
docker compose config                  # 구성(config)을 검증하고 해석된 내용 확인
```

**최소 compose.yml 예제:**

```yaml
services:
  api:
    build: .
    ports:
      - "3000:3000"
    environment:
      - DATABASE_URL=postgres://user:pass@db:5432/mydb
    depends_on:
      db:
        condition: service_healthy

  db:
    image: postgres:16-alpine
    environment:
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
      POSTGRES_DB: mydb
    volumes:
      - pgdata:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U user"]
      interval: 10s
      timeout: 5s
      retries: 5

volumes:
  pgdata:
```

### 5. Volumes and networks

```bash
# 볼륨
docker volume ls                       # 볼륨 목록
docker volume create mydata            # 기명 볼륨 생성
docker volume inspect mydata           # 상세 정보 (마운트 지점 등)
docker volume rm mydata                # 제거 (사용 중인 경우 실패)
docker volume prune                    # 사용하지 않는 볼륨 제거

# 네트워크
docker network ls                      # 네트워크 목록
docker network create mynet            # 브리지 네트워크 생성
docker network inspect mynet           # 상세 정보 (연결된 컨테이너 등)
docker network connect mynet NAME      # 컨테이너를 네트워크에 연결
docker network disconnect mynet NAME   # 컨테이너 분리
docker network rm mynet                # 네트워크 제거
docker network prune                   # 사용하지 않는 네트워크 제거
```

### 6. Disk usage and cleanup

정리하기 전에 항상 진단부터 시작하세요:

```bash
# 공간을 차지하는 요소 확인
docker system df                       # 요약
docker system df -v                    # 세부 내역

# 선별적 정리 (안전)
docker container prune                 # 중지된 컨테이너
docker image prune                     # 매달린(dangling) 이미지
docker volume prune                    # 사용하지 않는 볼륨
docker network prune                   # 사용하지 않는 네트워크

# 공격적 정리 (반드시 사용자에게 먼저 확인하세요!)
docker system prune                    # 컨테이너 + 이미지 + 네트워크
docker system prune -a                 # 사용하지 않는 이미지까지 포함
docker system prune -a --volumes       # 모든 것 — 기명 볼륨까지 포함
```

**경고:** 사용자에게 확인하지 않고 `docker system prune -a --volumes`를 실행하지 마세요. 이렇게 하면 중요한 데이터가 있을 수 있는 기명 볼륨이 제거됩니다.

## Pitfalls

| Problem | Cause | Fix |
|---------|-------|-----|
| 컨테이너가 즉시 종료됨 | 기본 프로세스가 완료되었거나 충돌함 | `docker logs NAME` 확인, `docker run -it --entrypoint /bin/sh IMAGE` 시도 |
| "port is already allocated" (포트가 이미 할당됨) | 다른 프로세스가 해당 포트를 사용 중임 | `docker ps` 또는 `lsof -i :PORT`로 찾기 |
| "no space left on device" (장치에 남은 공간 없음) | Docker 디스크가 가득 참 | `docker system df` 후 대상 지정 정리 |
| 컨테이너에 연결할 수 없음 | 앱이 컨테이너 내부의 127.0.0.1에 바인딩됨 | 앱은 `0.0.0.0`에 바인딩되어야 함, `-p` 매핑 확인 |
| 볼륨에서 권한 거부됨 | 호스트와 컨테이너 간의 UID/GID 불일치 | `--user $(id -u):$(id -g)` 사용 또는 권한 수정 |
| Compose 서비스가 서로 통신할 수 없음 | 잘못된 네트워크 또는 서비스 이름 | 서비스는 서비스 이름을 호스트 이름으로 사용함, `docker compose config` 확인 |
| 빌드 캐시가 작동하지 않음 | Dockerfile의 레이어 순서가 잘못됨 | 거의 변경되지 않는 레이어를 먼저 배치 (소스 코드 전에 의존성 배치) |
| 이미지가 너무 큼 | 다단계 빌드 없음, .dockerignore 없음 | 다단계 빌드(multi-stage builds) 사용, `.dockerignore` 추가 |

## Verification

모든 Docker 작업 후에는 결과를 확인하세요:

- **컨테이너가 시작되었나요?** → `docker ps` (상태가 "Up"인지 확인)
- **로그가 깨끗한가요?** → `docker logs --tail 20 NAME` (오류 없음)
- **포트에 접근 가능한가요?** → `curl -s http://localhost:PORT` 또는 `docker port NAME`
- **이미지가 빌드되었나요?** → `docker images | grep TAG`
- **Compose 스택이 정상인가요?** → `docker compose ps` (모든 서비스가 "running" 또는 "healthy")
- **디스크 공간이 확보되었나요?** → `docker system df` (이전/이후 비교)

## Dockerfile Optimization Tips

Dockerfile을 검토하거나 생성할 때 다음과 같은 개선 사항을 제안하세요:

1. **다단계 빌드(Multi-stage builds)** — 런타임에서 빌드 환경을 분리하여 최종 이미지 크기 줄이기
2. **레이어 순서(Layer ordering)** — 변경 사항이 캐시된 레이어를 무효화하지 않도록 소스 코드보다 의존성을 먼저 배치
3. **RUN 명령 결합(Combine RUN commands)** — 레이어 수 감소, 이미지 크기 감소
4. **.dockerignore 사용** — `node_modules`, `.git`, `__pycache__` 등을 제외
5. **기본 이미지 버전 고정** — `node:latest`가 아닌 `node:20-alpine` 사용
6. **루트 권한 없이 실행(Run as non-root)** — 보안을 위해 `USER` 명령어 추가
7. **슬림/알파인(slim/alpine) 기본 이미지 사용** — `python:3.12`가 아닌 `python:3.12-slim` 사용
