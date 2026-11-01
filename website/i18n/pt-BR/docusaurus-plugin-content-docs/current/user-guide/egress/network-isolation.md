---
title: "Isolamento de saída de rede (Docker)"
description: "Segmente redes Docker para que o sandbox do agente alcance apenas hosts permitidos"
---

# Isolamento de saída de rede para implantações Docker

Ao executar o Hermes dentro do Docker, o `network_mode: host` padrão concede ao
processo do agente acesso irrestrito à rede de saída. Este guia mostra como
segmentar o tráfego para que o núcleo do agente alcance apenas os serviços de
que precisa, bloqueando conexões arbitrárias.

Isso é principalmente uma defesa contra ataques de injeção de prompt que tentam
exfiltrar dados por `curl`, `wget` ou HTTP bruto em comandos shell gerados por
ferramentas.

## Modelo de ameaça

A seção 2 do [SECURITY.md](https://github.com/NousResearch/hermes-agent/blob/main/SECURITY.md)
define o modelo de confiança. O backend de terminal é a principal fronteira de
execução. Porém, com `network_mode: host`, qualquer comando executado pelo
agente pode alcançar qualquer endpoint da rede, inclusive externos.

O isolamento de saída adiciona uma segunda camada: mesmo que um comando
malicioso seja executado dentro do contêiner, ele não poderá alcançar endpoints
fora do conjunto explicitamente permitido.

## Arquitetura

```text
┌─────────────────────────────────────────────┐
│  Rede Docker: internal (sem internet)       │
│                                             │
│   ┌──────────────┐   ┌──────────────────┐   │
│   │ hermes-agent │   │ hermes-dashboard │   │
│   └──────┬───────┘   └────────┬─────────┘   │
│          │                    │              │
│          ▼                    │              │
│   ┌──────────────┐            │              │
│   │ hermes-gtw   │◄───────────┘              │
│   └──────┬───────┘                           │
│          │                                   │
└──────────┼───────────────────────────────────┘
           │
┌──────────┼───────────────────────────────────┐
│  Rede Docker: egress (com acesso à internet) │
│          │                                   │
│          ▼                                   │
│   ┌─────────────────┐                        │
│   │ egress-proxy     │──► hosts permitidos   │
│   │ (squid / envoy)  │                       │
│   └─────────────────┘                        │
└──────────────────────────────────────────────┘
```

Duas redes Docker:

- **`internal`** — sem rota padrão e sem acesso à internet. O agente, o
  dashboard e o gateway são executados aqui.
- **`egress`** — tem acesso à internet. Apenas serviços que precisam alcançar
  APIs externas são conectados a esta rede.

O serviço do gateway tem duas interfaces (conectado às duas redes): recebe
mensagens de entrada do Telegram/Slack/etc. e as encaminha ao agente pela rede
interna.

## Configuração do Compose

Substitua o `docker-compose.yml` padrão com um
`docker-compose.override.yml`:

```yaml
# docker-compose.override.yml
# Isolamento de saída para implantações de produção.
#
# Uso:
#   HERMES_UID=$(id -u) HERMES_GID=$(id -g) docker compose up -d
#
# Isso substitui network_mode: host por redes Docker isoladas.

networks:
  internal:
    driver: bridge
    internal: true          # sem rota padrão, sem internet
  egress:
    driver: bridge

services:
  gateway:
    network_mode: ""        # limpa o padrão host-mode
    networks:
      - internal
      - egress              # precisa de saída para Telegram e APIs de LLM
    ports:
      - "127.0.0.1:9119:9119"   # proxy do dashboard, somente localhost

  dashboard:
    network_mode: ""
    networks:
      - internal            # somente interna, sem necessidade de egress
```

### Com um proxy de saída (recomendado)

Para um controle mais rigoroso, encaminhe todo o tráfego de saída por um proxy
HTTP com uma allowlist explícita:

```yaml
# docker-compose.override.yml (com proxy de saída)

networks:
  internal:
    driver: bridge
    internal: true
  egress:
    driver: bridge

services:
  gateway:
    network_mode: ""
    networks:
      - internal
      - egress
    environment:
      - HTTP_PROXY=http://egress-proxy:3128
      - HTTPS_PROXY=http://egress-proxy:3128
      - NO_PROXY=hermes,hermes-dashboard,localhost

  dashboard:
    network_mode: ""
    networks:
      - internal

  egress-proxy:
    image: ubuntu/squid:6.10-24.04_edge
    networks:
      - egress
    volumes:
      - ./config/squid-allowlist.conf:/etc/squid/conf.d/allowlist.conf:ro
    restart: unless-stopped
```

Exemplo de `config/squid-allowlist.conf`:

```
# Permitir CONNECT HTTPS somente para estes hosts
acl allowed_hosts dstdomain api.openai.com
acl allowed_hosts dstdomain api.anthropic.com
acl allowed_hosts dstdomain openrouter.ai
acl allowed_hosts dstdomain generativelanguage.googleapis.com
acl allowed_hosts dstdomain api.telegram.org
acl allowed_hosts dstdomain api.github.com
acl allowed_hosts dstdomain discord.com

http_access allow CONNECT allowed_hosts
http_access deny all
```

Ajuste a allowlist de acordo com seu provedor de LLM e sua plataforma de
mensagens.

## Validando a configuração

Depois de iniciar a pilha, verifique o isolamento:

```bash
# Do contêiner do agente: deve FALHAR (sem egress)
docker compose exec gateway \
  curl -sf --max-time 5 https://example.com && echo "FALHA: egress não bloqueado" || echo "OK: egress bloqueado"

# Do contêiner do agente: deve FUNCIONAR (rede interna)
docker compose exec gateway \
  curl -sf --max-time 5 http://hermes-dashboard:9119/health && echo "OK: interno acessível" || echo "FALHA"

# Com proxy de saída: deve FUNCIONAR (host permitido)
docker compose exec gateway \
  curl -sf --max-time 5 --proxy http://egress-proxy:3128 https://api.openai.com/v1/models && echo "OK" || echo "FALHA"
```

## Limitações

- **Resolução DNS:** a rede `internal` ainda pode resolver nomes DNS externos,
  a menos que você também execute um resolvedor DNS local que bloqueie consultas
  externas. Para a maioria dos modelos de ameaça, isso é aceitável, pois a
  resolução DNS sozinha não exfiltra dados significativos.
- **Não substitui backends de sandbox:** este guia isola a rede do *contêiner*
  do agente. Com o backend de terminal local padrão, os comandos de ferramentas
  são executados no mesmo contêiner. Para isolamento maior, combine a
  segmentação de rede com um backend de terminal em sandbox (Docker, Modal,
  Daytona).
- **Adaptadores de plataforma precisam de egress:** o gateway precisa de saída
  para alcançar as APIs das plataformas de mensagens. Ao adicionar um novo
  adaptador, inclua seus endpoints na allowlist do proxy.

## Relacionado

- [SECURITY.md](https://github.com/NousResearch/hermes-agent/blob/main/SECURITY.md) — modelo de confiança e relatório de vulnerabilidades
- [Docker](/user-guide/docker) — executar o Hermes em um contêiner
- [Proxy de saída](iron-proxy.md) — firewall de injeção de credenciais para o sandbox
- [docker-compose.yml](https://github.com/NousResearch/hermes-agent/blob/main/docker-compose.yml) — configuração padrão do Compose
