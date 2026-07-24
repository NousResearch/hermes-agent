---
sidebar_position: 12
title: "Envie a Saída de Scripts para Plataformas de Mensageria"
description: "Envie texto de qualquer script shell, tarefa cron, hook de CI ou daemon de monitoramento para Telegram, Discord, Slack, Signal e outras plataformas usando `hermes send`."
---

# Envie a Saída de Scripts para Plataformas de Mensageria {#pipe-script-output-to-messaging-platforms}

`hermes send` é uma CLI pequena e roteirizável que envia uma mensagem para qualquer
plataforma de mensageria para a qual o Hermes já esteja configurado. Pense nela como um
`curl` multiplataforma para notificações — você não precisa de um gateway
em execução, não precisa de um LLM, e não precisa colar tokens de bot novamente
em cada um dos seus scripts.

Use-a para:

- Monitoramento de sistema (memória, disco, temperatura da GPU, tarefa de longa duração finalizada)
- Notificações de CI/CD (deploy concluído, falha de teste)
- Scripts cron que precisam te avisar com resultados
- Mensagens rápidas e únicas a partir de um terminal
- Canalizar a saída de qualquer ferramenta para qualquer lugar (`make | hermes send --to slack:#builds`)

O comando reutiliza as mesmas credenciais e adaptadores de plataforma que o `hermes
gateway` já usa, então não há uma segunda superfície de configuração para
manter.

---

## Início Rápido {#quick-start}

```bash
# Plain text to the home channel for a platform
hermes send --to telegram "deploy finished"

# Pipe in stdout from anything
echo "RAM 92%" | hermes send --to telegram:-1001234567890

# Send a file
hermes send --to discord:#ops --file /tmp/report.md

# Attach a subject/header line
hermes send --to slack:#eng --subject "[CI] build.log" --file build.log

# Thread target (Telegram topic, Discord thread)
hermes send --to telegram:-1001234567890:17585 "threaded reply"

# List every configured target
hermes send --list

# Filter by platform
hermes send --list telegram
```

---

## Referência de Argumentos {#argument-reference}

| Flag | Descrição |
|------|-------------|
| `-t, --to TARGET` | Destino. Veja [formatos de destino](#target-formats). |
| `message` (posicional) | Texto da mensagem. Omita para ler de `--file` ou stdin. |
| `-f, --file PATH` | Lê o corpo de um arquivo. `--file -` força a leitura de stdin. |
| `-s, --subject LINE` | Adiciona uma linha de cabeçalho/assunto antes do corpo. |
| `-l, --list` | Lista os destinos disponíveis. Filtro de plataforma posicional opcional. |
| `-q, --quiet` | Sem saída padrão em caso de sucesso (apenas código de saída — ideal para scripts). |
| `--json` | Emite o resultado JSON bruto do envio. |
| `-h, --help` | Mostra o texto de ajuda integrado. |

### Formatos de Destino {#target-formats}

| Formato | Exemplo | Significado |
|--------|---------|---------|
| `platform` | `telegram` | Envia para o canal principal configurado da plataforma |
| `platform:chat_id` | `telegram:-1001234567890` | Chat / grupo / usuário numérico específico |
| `platform:chat_id:thread_id` | `telegram:-1001234567890:17585` | Thread específica ou tópico de fórum do Telegram |
| `platform:#channel` | `discord:#ops` | Nome de canal amigável (resolvido contra o diretório de canais) |
| `platform:+E164` | `signal:+15551234567` | Plataformas endereçadas por telefone: Signal, SMS, WhatsApp |

Qualquer plataforma para a qual o Hermes envie adaptadores funciona como destino:
`telegram`, `discord`, `slack`, `signal`, `sms`, `whatsapp`, `matrix`,
`mattermost`, `feishu`, `dingtalk`, `wecom`, `weixin`, `email` e
outras.

### Códigos de Saída {#exit-codes}

| Código | Significado |
|------|---------|
| `0` | Envio (ou listagem) bem-sucedido |
| `1` | Falha de entrega no nível da plataforma (autenticação, permissões, rede) |
| `2` | Erro de uso / argumento / configuração |

Os códigos de saída seguem a convenção padrão do Unix, então seus scripts podem
ramificar sobre eles da mesma forma que fariam com `curl` ou `grep`.

---

## Resolução do Corpo da Mensagem {#message-body-resolution}

`hermes send` resolve o corpo da mensagem nesta ordem:

1. **Argumento posicional** — `hermes send --to telegram "hi"`
2. **`--file PATH`** — `hermes send --to telegram --file msg.txt`
3. **stdin em pipe** — `echo hi | hermes send --to telegram`

Quando stdin é um TTY (sem pipe), o Hermes **não** espera por entrada — você vai
receber um erro de uso claro em vez disso. Isso evita que scripts fiquem presos se
esquecerem acidentalmente de omitir o corpo.

---

## Exemplos do Mundo Real {#real-world-examples}

### Monitoramento: Alertas de Memória / Disco {#monitoring-memory--disk-alerts}

Substitua chamadas ad-hoc de `curl https://api.telegram.org/...` nos seus watchdogs
por uma única linha portátil:

```bash
#!/usr/bin/env bash
ram_pct=$(free | awk '/^Mem:/ {printf "%d", $3 * 100 / $2}')
if [ "$ram_pct" -ge 85 ]; then
  hermes send --to telegram --subject "⚠ MEMORY WARNING" \
    "RAM ${ram_pct}% on $(hostname)"
fi
```

Como `hermes send` reutiliza sua configuração do Hermes, o mesmo script funciona em
qualquer host onde o Hermes esteja instalado — sem necessidade de exportar tokens de bot para
o ambiente de cada máquina manualmente.

:::tip Não alerte o gateway sobre ele mesmo
Para watchdogs que podem disparar quando o próprio gateway estiver com problemas (alertas
de OOM, alertas de disco cheio), continue usando uma chamada `curl` mínima em vez de
`hermes send`. Se o interpretador Python não conseguir carregar porque a máquina está
sobrecarregada, você ainda quer que aquele alerta seja enviado.
:::

### CI / CD: Resultados de Build e Teste {#ci--cd-build-and-test-results}

```bash
# In .github/workflows/deploy.yml or any CI script
if ./scripts/deploy.sh; then
  hermes send --to slack:#deploys "✅ ${CI_COMMIT_SHA:0:7} deployed"
else
  tail -n 100 deploy.log | hermes send \
    --to slack:#deploys --subject "❌ deploy failed"
  exit 1
fi
```

### Cron: Relatório Diário {#cron-daily-report}

```bash
# Crontab entry
0 9 * * * /usr/local/bin/generate-metrics.sh \
  | /home/me/.hermes/bin/hermes send \
      --to telegram --subject "Daily metrics $(date +%Y-%m-%d)"
```

### Tarefas de Longa Duração: Avise Quando Terminar {#long-running-tasks-ping-when-done}

```bash
./train.py --epochs 200 && \
  hermes send --to telegram "training done" || \
  hermes send --to telegram "training failed (exit $?)"
```

### Scripting com `--json` e `--quiet` {#scripting-with-json-and-quiet}

```bash
# Hard-fail a script if delivery fails; don't clutter logs on success
hermes send --to telegram --quiet "keepalive" || {
  echo "Telegram delivery failed" >&2
  exit 1
}

# Capture the message ID for later editing / threading
msg_id=$(hermes send --to discord:#ops --json "build started" \
  | jq -r .message_id)
```

---

## O `hermes send` Precisa do Gateway em Execução? {#does-hermes-send-need-the-gateway-running}

**Geralmente não.** Para qualquer plataforma de token de bot — Telegram, Discord, Slack,
Signal, SMS, WhatsApp Cloud API e a maioria das outras — o `hermes send` chama
o endpoint REST da plataforma diretamente usando as credenciais de
`~/.hermes/.env` e `~/.hermes/config.yaml`. É um subprocesso independente
que finaliza assim que a mensagem é entregue.

Um gateway ativo só é necessário para **plataformas de plugin** que dependem de uma
conexão de adaptador persistente (por exemplo, um plugin personalizado que mantém
um WebSocket de longa duração aberto). Nesse caso, você receberá um erro claro
apontando para o gateway; inicie-o com `hermes gateway start` e tente de novo.

---

## Listando e Descobrindo Destinos {#listing-and-discovering-targets}

Antes de enviar para um canal específico, você pode inspecionar o que está disponível:

```bash
# Every target across every configured platform
hermes send --list

# Just Telegram targets
hermes send --list telegram

# Machine-readable
hermes send --list --json
```

A listagem é construída a partir de `~/.hermes/channel_directory.json`, que o
gateway atualiza a cada poucos minutos enquanto está em execução. Se você ver
"no channels discovered yet", inicie o gateway uma vez (`hermes gateway
start`) para que ele possa popular o cache.

Nomes amigáveis (`discord:#ops`, `slack:#engineering`) são resolvidos
contra esse cache no momento do envio, então você não precisa memorizar IDs
numéricos.

---

## Comparação com Outras Abordagens {#comparison-with-other-approaches}

| Abordagem | Multiplataforma | Reutiliza credenciais do Hermes | Precisa do gateway | Melhor para |
|----------|----------------|---------------------|---------------|----------|
| `hermes send` | ✅ | ✅ | Não (token de bot) | Tudo abaixo |
| `curl` bruto para cada plataforma | Roteirizado separadamente | Manual | Não | Watchdogs críticos |
| Tarefa `cron` com `--deliver` | ✅ | ✅ | Não | Tarefas de agente agendadas |

`hermes send` é intencionalmente a superfície mais simples possível. Se você precisa de
um agente para decidir o que dizer, agende uma tarefa cron — a resposta final do agente
é entregue automaticamente ao destino `deliver:` configurado (o agente
não dispara mais mensagens sozinho). Se você precisa de uma execução agendada com conteúdo gerado por LLM,
use `cronjob(action='create', prompt=...)` com `deliver='telegram:...'`.
Se você só precisa canalizar uma string bruta, use o `hermes send`.

---

## Relacionados {#related}

- [Automatize Qualquer Coisa com Cron](/guides/automate-with-cron) —
  tarefas agendadas cuja saída é entregue automaticamente a qualquer plataforma.
- [Internos do Gateway](/developer-guide/gateway-internals) —
  o roteador de entrega que o `hermes send` compartilha com a entrega do cron.
- [Configuração de Plataformas de Mensageria](/user-guide/messaging/) —
  configuração única para cada plataforma.
