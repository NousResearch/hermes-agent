---
title: Egress 프록시
sidebar_position: 1
---

# Egress 프록시

원격 터미널 샌드박스를 위한 선택적 아웃바운드 자격 증명 주입 방화벽입니다. 샌드박스에는 불투명한 프록시 토큰만 저장되며, 실제 API 키는 호스트 외부로 나가지 않습니다.

- [iron-proxy](./iron-proxy) — [ironsh/iron-proxy](https://github.com/ironsh/iron-proxy)의 단일 바이너리 TLS 가로채기 프록시로, `hermes egress`를 통해 지연 설치되고 관리됩니다.
