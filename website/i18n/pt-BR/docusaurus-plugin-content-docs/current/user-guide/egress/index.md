---
title: Proxy de egresso
sidebar_position: 1
---

# Proxy de egresso

Firewall opcional de injeção de credenciais para tráfego de saída em sandboxes de terminal remotos. O sandbox só mantém tokens de proxy opacos; as chaves de API reais nunca saem do host.

- [iron-proxy](./iron-proxy) — proxy de interceptação TLS em binário único, de [ironsh/iron-proxy](https://github.com/ironsh/iron-proxy), instalado sob demanda e gerenciado por `hermes egress`.
