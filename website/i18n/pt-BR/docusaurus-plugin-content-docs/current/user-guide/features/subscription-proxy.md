---
sidebar_position: 15
title: "Proxy de assinatura"
description: "Use sua assinatura Nous Portal (ou outro provider OAuth) como endpoint OpenAI-compatível para apps externos"
---

# Proxy de assinatura {#subscription-proxy}

O proxy de assinatura é um servidor HTTP local que permite que apps externos —
OpenViking, Karakeep, Open WebUI, qualquer coisa que fale chat completions compatível com OpenAI —
usem sua assinatura de provider gerenciada pelo Hermes como endpoint LLM. O proxy anexa as credenciais corretas (renovando automaticamente) para que o app nunca precise de uma API key estática.

Isso é diferente do [API server](./api-server.md):

| | API server | Proxy de assinatura |
|---|---|---|
| O que serve | Seu agente (toolset completo, memória, skills) | Inferência de modelo bruta |
| Caso de uso | "Usar Hermes como backend de chat" | "Usar minha assinatura Portal de outro app" |
| Auth | Sua `API_SERVER_KEY` | Qualquer bearer (proxy anexa a real) |
| Tool calls | Sim — o agente executa tools | Não — só passthrough |

Use o API server quando quiser o **agente** como backend. Use o
proxy quando quiser só **o modelo** via sua assinatura.

## Início rápido {#quick-start}

### 1. Faça login no seu provider (uma vez) {#1-log-into-your-provider-one-time}

```bash
hermes portal
```

Isso abre o browser para o fluxo OAuth do Nous Portal. O Hermes armazena
o refresh token em `~/.hermes/auth.json` — o mesmo lugar de todos os
logins de provider do Hermes.

### 2. Inicie o proxy {#2-start-the-proxy}

```bash
hermes proxy start
```

```
Starting Hermes proxy for Nous Portal
  Listening on:  http://127.0.0.1:8645/v1
  Forwarding to: (resolved per-request from your subscription)
  Use any bearer token in the client — the proxy attaches your real credential.
```

Deixe rodando em foreground. Use `tmux`, `nohup` ou uma unit systemd
se quiser que sobreviva ao logout.

### 3. Aponte seu app para ele {#3-point-your-app-at-it}

Qualquer config de app compatível com OpenAI usa o mesmo trio:

```
Base URL:   http://127.0.0.1:8645/v1
API key:    anything (e.g. "sk-unused")
Model:      Hermes-4-70B    # or Hermes-4.3-36B, Hermes-4-405B
```

O proxy ignora o header `Authorization` do seu app e anexa
sua credencial real do Portal na requisição upstream. Renovações acontecem
automaticamente quando o bearer se aproxima da expiração.

## Providers disponíveis {#available-providers}

```bash
hermes proxy providers
```

Atualmente disponíveis: `nous` (Nous Portal) e `xai` (xAI / Grok). Mais
providers OAuth podem ser adicionados implementando a interface `UpstreamAdapter`
em `hermes_cli/proxy/adapters/`.

## Verificar status {#check-status}

```bash
hermes proxy status
```

```
Hermes proxy upstream adapters

  [nous    ] Nous Portal — ready (bearer expires 2026-05-15T06:43:21Z)
```

Se vir `not logged in`, execute `hermes portal`. Se vir
`credentials need attention`, seu refresh token foi revogado (raro —
acontece se você saiu da UI web do Portal) — basta rodar
`hermes portal` de novo.

## Caminhos permitidos {#allowed-paths}

O proxy só encaminha caminhos que o upstream realmente serve. Para Nous
Portal:

| Caminho | Propósito |
|------|---------|
| `/v1/chat/completions` | Chat completions (streaming + não streaming) |
| `/v1/completions` | Text completions legado |
| `/v1/embeddings` | Embeddings |
| `/v1/models` | Lista de modelos |

Outros caminhos (`/v1/images/generations`, `/v1/audio/speech`, etc.) retornam
404 com erro claro apontando os caminhos permitidos. Isso impede que clientes
estranhos vazem requisições estranhas para o upstream.

## Configurando OpenViking para usar o Portal {#configuring-openviking-to-use-portal}

[OpenViking](https://github.com/volcengine/OpenViking) é um banco de contexto
que precisa de um provider LLM para seu VLM (modelo visão/linguagem
usado para extrair memórias) e modelo de embedding. Com o proxy, você pode
apontar o `vlm.api_base` dele para seu proxy local:

Edite `~/.openviking/ov.conf`:

```json
{
  "vlm": {
    "provider": "openai",
    "model": "Hermes-4-70B",
    "api_base": "http://127.0.0.1:8645/v1",
    "api_key": "unused-proxy-attaches-real-creds"
  }
}
```

Depois inicie seu proxy em um terminal junto com `openviking-server`:

```bash
# Terminal 1
hermes proxy start

# Terminal 2
openviking-server
```

As chamadas VLM do OpenViking agora passam pela sua assinatura Portal. O
lado do modelo de embedding ainda precisa de provider próprio — o Portal serve
`/v1/embeddings`, mas a seleção de modelo depende do que seu tier
suporta; confira `portal.nousresearch.com/models`.

## Configurando Karakeep (ou qualquer app de bookmark/resumo) {#configuring-karakeep-or-any-bookmarksummarizer-app}

[Karakeep](https://karakeep.app/) aceita API compatível com OpenAI para
resumo de bookmarks. Na config dele:

```bash
# Karakeep .env
OPENAI_API_BASE_URL=http://127.0.0.1:8645/v1
OPENAI_API_KEY=any-non-empty-string
INFERENCE_TEXT_MODEL=Hermes-4-70B
```

O mesmo padrão funciona para Open WebUI, LobeChat, NextChat ou qualquer outro
cliente compatível com OpenAI.

## Expor na LAN {#exposing-on-lan}

Por padrão o proxy escuta em `127.0.0.1` (só localhost). Para outras
máquinas na sua rede usarem:

```bash
hermes proxy start --host 0.0.0.0 --port 8645
```

⚠ **Atenção:** qualquer um na sua rede pode usar sua assinatura
Portal agora. O proxy não tem auth própria — aceita qualquer bearer.
Use firewall, VPN ou reverse proxy com auth adequada se expuser
além da sua rede confiável.

## Rate limits {#rate-limits}

Os limites RPM/TPM do seu tier Portal se aplicam ao proxy inteiro. O
proxy não faz fan-out nem pool — é um único bearer com sua cota completa
de assinatura. Monitore uso em
[portal.nousresearch.com](https://portal.nousresearch.com).

## Arquitetura {#architecture}

O proxy é intencionalmente mínimo. Por requisição:

1. Recebe `POST /v1/chat/completions` do seu app
2. Busca a credencial atual do adapter (refresh se expirando)
3. Encaminha o body da requisição verbatim, com `Authorization: Bearer <minted-key>`
4. Transmite a resposta de volta inalterada (SSE preservado)

Sem transformação. Sem log de bodies de requisição. Sem loop de agente. O
proxy é um pass-through que anexa credenciais.

## Futuro: mais providers OAuth {#future-more-oauth-providers}

O sistema de adapters é plugável. Adicionar um provider novo (por exemplo
HuggingFace, endpoint de chat do GitHub Copilot, Anthropic via OAuth)
exige implementar `UpstreamAdapter` em
`hermes_cli/proxy/adapters/<provider>.py` e registrá-lo em
`adapters/__init__.py`. Providers que não são OpenAI-compatíveis no nível
de protocolo (Messages API da Anthropic, por exemplo) precisariam de uma
camada de transformação, fora do escopo do formato atual.
