---
title: "Docker 관리 — Docker 컨테이너, 이미지, 볼륨 및 Compose 관리"
sidebar_label: "Docker 관리"
description: "Docker 컨테이너, 이미지, 볼륨 및 Compose 관리"
---

{/* 이 페이지는 website/scripts/generate-skill-docs.py가 스킬의 SKILL.md에서 자동으로 생성합니다. 이 페이지가 아니라 원본 SKILL.md를 편집하세요. */}

# Docker 관리

Docker 컨테이너, 이미지, 볼륨 및 Compose를 관리합니다.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | 선택 사항 — `hermes skills install official/devops/docker-management`로 설치 |
| 경로 | `optional-skills/devops/docker-management` |
| 버전 | `1.0.0` |
| 작성자 | sprmn24 |
| 라이선스 | MIT |
| 플랫폼 | linux, macos, windows |
| 태그 | `docker`, `containers`, `devops`, `infrastructure`, `compose`, `images`, `volumes`, `networks`, `debugging` |

## 참조: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 완전한 스킬 정의입니다. 스킬이 활성화되었을 때 에이전트가 보는 지침입니다.
:::

# Docker 관리

표준 Docker CLI 명령을 사용하여 Docker 컨테이너, 이미지, 볼륨, 네트워크 및 Compose 스택을 관리합니다. Docker 자체 외에 추가 종속성은 없습니다.

## 사용 시점

- 컨테이너 실행, 중지, 재시작, 제거 또는 검사
- Docker 이미지 빌드, 가져오기, 푸시, 태그 지정 또는 정리
- Docker Compose(다중 서비스 스택) 작업
- 볼륨 또는 네트워크 관리
- 충돌하는 컨테이너 디버깅 또는 로그 분석
- Docker 디스크 사용량 확인 또는 공간 확보
- Dockerfile 검토 또는 최적화

## 사전 요구 사항

- Docker Engine 설치 및 실행
- `docker` 그룹에 사용자 추가(또는 `sudo` 사용)
- Docker Compose v2(최신 Docker 설치에 포함)

빠른 확인:

```bash
docker --version && docker compose version
```

## 빠른 참조

| 작업 | 명령 |
|------|---------|
| 컨테이너 실행(백그라운드) | `docker run -d --name NAME IMAGE` |
| 중지 + 제거 | `docker stop NAME && docker rm NAME` |
| 로그 보기(팔로우) | `docker logs --tail 50 -f NAME` |
| 컨테이너 셸 접속 | `docker exec -it NAME /bin/sh` |
| 모든 컨테이너 나열 | `docker ps -a` |
| 이미지 빌드 | `docker build -t TAG .` |
| Compose 시작 | `docker compose up -d` |
| Compose 종료 | `docker compose down` |
| 디스크 사용량 | `docker system df` |
| 댕글링 리소스 정리 | `docker image prune && docker container prune` |

## 절차

### 1. 영역 식별

요청이 어느 영역에 해당하는지 파악합니다.

- **컨테이너 수명 주기** → run, stop, start, restart, rm, pause/unpause
- **컨테이너 상호작용** → exec, cp, logs, inspect, stats
- **이미지 관리** → build, pull, push, tag, rmi, save/load
- **Docker Compose** → up, down, ps, logs, exec, build, config
- **볼륨 및 네트워크** → create, inspect, rm, prune, connect
- **문제 해결** → 로그 분석, 종료 코드, 리소스 문제

### 2. 컨테이너 작업

**새 컨테이너 실행:**

```bash
# Detached service with port mapping
docker run -d --name web -p 8080:80 nginx

# With environment variables
docker run -d -e POSTGRES_PASSWORD=secret -e POSTGRES_DB=mydb --name db postgres:16

# With persistent data (named volume)
docker run -d -v pgdata:/var/lib/postgresql/data --name db postgres:16

# For development (bind mount source code)
docker run -d -v $(pwd)/src:/app/src -p 3000:3000 --name dev my-app

# Interactive debugging (auto-remove on exit)
docker run -it --rm ubuntu:22.04 /bin/bash

# With resource limits and restart policy
docker run -d --memory=512m --cpus=1.5 --restart=unless-stopped --name app my-app
```

주요 플래그: `-d` 분리 실행, `-it` 대화형+tty, `--rm` 자동 제거, `-p` 포트(호스트:컨테이너), `-e` 환경 변수, `-v` 볼륨, `--name` 이름, `--restart` 재시작 정책.

**실행 중인 컨테이너 관리:**

```bash
docker ps                        # running containers
docker ps -a                     # all (including stopped)
docker stop NAME                 # graceful stop
docker start NAME                # start stopped container
docker restart NAME              # stop + start
docker rm NAME                   # remove stopped container
docker rm -f NAME                # force remove running container
docker container prune           # remove ALL stopped containers
```

**컨테이너와 상호작용:**

```bash
docker exec -it NAME /bin/sh          # shell access (use /bin/bash if available)
docker exec NAME env                   # view environment variables
docker exec -u root NAME apt update    # run as specific user
docker logs --tail 100 -f NAME         # follow last 100 lines
docker logs --since 2h NAME            # logs from last 2 hours
docker cp NAME:/path/file ./local      # copy file from container
docker cp ./file NAME:/path/           # copy file to container
docker inspect NAME                    # full container details (JSON)
docker stats --no-stream               # resource usage snapshot
docker top NAME                        # running processes
```

### 3. 이미지 관리

```bash
# Build
docker build -t my-app:latest .
docker build -t my-app:prod -f Dockerfile.prod .
docker build --no-cache -t my-app .              # clean rebuild
DOCKER_BUILDKIT=1 docker build -t my-app .       # faster with BuildKit

# Pull and push
docker pull node:20-alpine
docker login ghcr.io
docker tag my-app:latest registry/my-app:v1.0
docker push registry/my-app:v1.0

# Inspect
docker images                          # list local images
docker history IMAGE                   # see layers
docker inspect IMAGE                   # full details

# Cleanup
docker image prune                     # remove dangling (untagged) images
docker image prune -a                  # remove ALL unused images (careful!)
docker image prune -a --filter "until=168h"   # unused images older than 7 days
```

### 4. Docker Compose

```bash
# Start/stop
docker compose up -d                   # start all services detached
docker compose up -d --build           # rebuild images before starting
docker compose down                    # stop and remove containers
docker compose down -v                 # also remove volumes (DESTROYS DATA)

# Monitoring
docker compose ps                      # list services
docker compose logs -f api             # follow logs for specific service
docker compose logs --tail 50          # last 50 lines all services

# Interaction
docker compose exec api /bin/sh        # shell into running service
docker compose run --rm api npm test   # one-off command (new container)
docker compose restart api             # restart specific service

# Validation
docker compose config                  # validate and view resolved config
```

**최소 compose.yml 예시:**

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

### 5. 볼륨 및 네트워크

```bash
# Volumes
docker volume ls                       # list volumes
docker volume create mydata            # create named volume
docker volume inspect mydata           # details (mount point, etc.)
docker volume rm mydata                # remove (fails if in use)
docker volume prune                    # remove unused volumes

# Networks
docker network ls                      # list networks
docker network create mynet            # create bridge network
docker network inspect mynet           # details (connected containers)
docker network connect mynet NAME      # attach container to network
docker network disconnect mynet NAME   # detach container
docker network rm mynet                # remove network
docker network prune                   # remove unused networks
```

### 6. 디스크 사용량 및 정리

정리하기 전에 항상 진단부터 시작합니다.

```bash
# Check what's using space
docker system df                       # summary
docker system df -v                    # detailed breakdown

# Targeted cleanup (safe)
docker container prune                 # stopped containers
docker image prune                     # dangling images
docker volume prune                    # unused volumes
docker network prune                   # unused networks

# Aggressive cleanup (confirm with user first!)
docker system prune                    # containers + images + networks
docker system prune -a                 # also unused images
docker system prune -a --volumes       # EVERYTHING — named volumes too
```

**경고:** 사용자에게 확인받지 않고 `docker system prune -a --volumes`를 실행하지 마세요. 이 명령은 중요할 수 있는 데이터를 포함한 이름 있는 볼륨을 제거합니다.

## 문제점

| 문제 | 원인 | 해결 방법 |
|---------|-------|-----|
| 컨테이너가 즉시 종료됨 | 주 프로세스가 완료되었거나 충돌함 | `docker logs NAME`을 확인하고 `docker run -it --entrypoint /bin/sh IMAGE`를 시도 |
| "port is already allocated" | 해당 포트를 사용하는 다른 프로세스가 있음 | `docker ps` 또는 `lsof -i :PORT`로 확인 |
| "no space left on device" | Docker 디스크가 가득 참 | `docker system df`를 실행한 후 필요한 항목만 정리 |
| 컨테이너에 연결할 수 없음 | 앱이 컨테이너 내부에서 127.0.0.1에 바인딩됨 | 앱은 `0.0.0.0`에 바인딩해야 하며 `-p` 매핑을 확인 |
| 볼륨 권한 거부 | 호스트와 컨테이너 간 UID/GID 불일치 | `--user $(id -u):$(id -g)`를 사용하거나 권한 수정 |
| Compose 서비스가 서로 연결되지 않음 | 잘못된 네트워크 또는 서비스 이름 | 서비스 이름을 호스트 이름으로 사용하고 `docker compose config` 확인 |
| 빌드 캐시가 작동하지 않음 | Dockerfile의 레이어 순서가 잘못됨 | 거의 변경되지 않는 레이어를 먼저 배치(소스 코드보다 종속성 우선) |
| 이미지가 너무 큼 | 멀티 스테이지 빌드 또는 .dockerignore가 없음 | 멀티 스테이지 빌드를 사용하고 `.dockerignore` 추가 |

## 검증

Docker 작업 후에는 결과를 검증합니다.

- **컨테이너가 시작되었나요?** → `docker ps`(상태가 "Up"인지 확인)
- **로그가 깨끗한가요?** → `docker logs --tail 20 NAME`(오류 없음)
- **포트에 접근할 수 있나요?** → `curl -s http://localhost:PORT` 또는 `docker port NAME`
- **이미지가 빌드되었나요?** → `docker images | grep TAG`
- **Compose 스택이 정상인가요?** → `docker compose ps`(모든 서비스가 "running" 또는 "healthy")
- **디스크 공간이 확보되었나요?** → `docker system df`(전후 비교)

## Dockerfile 최적화 팁

Dockerfile을 검토하거나 만들 때 다음 개선 사항을 제안합니다.

1. **멀티 스테이지 빌드** — 최종 이미지 크기를 줄이기 위해 빌드 환경과 런타임을 분리
2. **레이어 순서** — 변경 사항이 캐시된 레이어를 무효화하지 않도록 소스 코드보다 종속성을 먼저 배치
3. **RUN 명령 결합** — 레이어 수를 줄여 더 작은 이미지 생성
4. **.dockerignore 사용** — `node_modules`, `.git`, `__pycache__` 등을 제외
5. **기본 이미지 버전 고정** — `node:latest`가 아닌 `node:20-alpine`
6. **비루트 사용자로 실행** — 보안을 위해 `USER` 지시문 추가
7. **slim/alpine 기본 이미지 사용** — `python:3.12`가 아닌 `python:3.12-slim`
