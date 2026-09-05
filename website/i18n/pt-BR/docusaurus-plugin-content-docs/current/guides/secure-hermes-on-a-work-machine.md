---
sidebar_position: 26
title: "Executando o Hermes em uma máquina pessoal ou de trabalho"
description: "Um passo a passo de postura de segurança para executar o Hermes Agent na máquina em que você trabalha — o que os padrões protegem, como reforçar ainda mais e como desfazer erros"
---

# Executando o Hermes em uma máquina pessoal ou de trabalho {#running-hermes-on-a-personal-or-work-machine}

Você está prestes a executar um agente na máquina em que trabalha — um laptop pessoal ou uma estação gerenciada pelo empregador. Qual é a postura segura?

Resposta curta: os padrões já fazem a maior parte do trabalho. O Hermes vem seguro por padrão, com um modelo de defesa em profundidade que cobre aprovação de comandos, segurança de escrita de arquivos e tratamento de credenciais. Esta página percorre o que já vem ativo, quais ajustes reforçar em uma máquina compartilhada ou de trabalho e como desfazer erros quando acontecem. Cada controle aqui é coberto em profundidade no guia de [Segurança](/user-guide/security).

## O que os padrões já protegem {#what-the-defaults-already-protect}

Instalação nova, sem configuração — estas proteções estão ativas:

**Comandos perigosos exigem aprovação.** Antes de executar qualquer comando, o Hermes o compara com uma lista curada de padrões perigosos — exclusões recursivas, gravações em `/etc/`, operações de disco, pipe-to-shell e mais. O padrão `approvals.mode: smart` usa um LLM auxiliar para avaliar o risco: comandos de baixo risco são autoaprovados apenas para aquele comando, comandos genuinamente perigosos são negados automaticamente e casos incertos escalam para um prompt manual.

**Prompts de aprovação falham fechado.** Se você não responder a um prompt de aprovação dentro do timeout (padrão 300 segundos), o comando é **negado**. Sair da mesa nunca aprova nada silenciosamente.

**Uma blocklist inflexível é o piso sempre ativo.** Alguns comandos — `rm -rf /`, fork bombs, zerar um disco físico — são recusados **independentemente** do modo de aprovação, `--yolo` ou de um "permitir sempre" explícito. A blocklist dispara antes mesmo da camada de aprovação ver o comando, e não há flag de override.

**Gravações em caminhos sensíveis são bloqueadas.** As ferramentas `write_file` e `patch` não podem tocar em repositórios de credenciais do SO (`~/.ssh/`, `~/.aws/`, `~/.kube/`, `/etc/sudoers`, `~/.netrc`), repositórios de credenciais do Hermes (`auth.json`, `.env`, dados de pairing) nem arquivos secretos de projeto (`.env`, `.env.local`, `.envrc`) em qualquer lugar do disco. Gravações bloqueadas retornam erro imediatamente — não há prompt de aprovação e não há como contornar pela UI do chat.

**Secrets são redigidos da saída.** `security.redact_secrets` vem ativo por padrão: padrões que parecem chaves de API, tokens e senhas na saída de ferramentas são redigidos antes de entrarem no contexto da conversa e nos logs.

**Seus dados vão apenas para onde você apontar.** Chamadas de API vão **somente** para o provedor de LLM que você configurar. O Hermes Agent não coleta telemetria, dados de uso ou analytics. Suas conversas, memória e skills ficam armazenadas localmente em `~/.hermes/`. Veja o [FAQ](/reference/faq#is-my-data-sent-anywhere).

:::info
Há mais abaixo da superfície — proteção SSRF em todas as ferramentas com URL, ambientes filtrados para subprocessos MCP, varredura de prompt injection em arquivos de contexto. A página de [Segurança](/user-guide/security) documenta cada camada.
:::

## Reforçando para uma máquina compartilhada ou de trabalho {#tightening-for-a-shared-or-work-machine}

Em uma máquina com dados do empregador, credenciais de produção ou arquivos de outras pessoas, adicione estas camadas sobre os padrões.

### Mude aprovações para manual {#switch-approvals-to-manual}

O modo `smart` autoaprova comandos de baixo risco. Se você quiser ver cada comando sinalizado:

```yaml
approvals:
  mode: manual
```

O modo manual sempre pede confirmação antes de executar um comando sinalizado.

### Adicione suas próprias regras de negação {#add-your-own-deny-rules}

`approvals.deny` é uma lista de padrões glob que bloqueiam comandos de terminal correspondentes incondicionalmente — mesmo com `--yolo`, `/yolo` ou `mode: off`. É a contraparte editável pelo usuário da blocklist inflexível integrada. Use para declarar o que nunca deve rodar nesta máquina:

```yaml
approvals:
  deny:
    - "git push --force*"
    - "*curl*|*sh*"
    - "dd if=* of=/dev/*"
```

Os padrões são globs [fnmatch](https://docs.python.org/3/library/fnmatch.html) case-insensitive comparados com o texto completo do comando, e a correspondência roda sobre as mesmas variantes normalizadas/desofuscadas que o detector de padrões perigosos usa, então truques simples de aspas não passam por uma regra. Sempre coloque padrões entre aspas — um `*` inicial sem aspas é erro de parse YAML. Mudanças entram em vigor imediatamente, sem reiniciar. Detalhes: [Regras de negação definidas pelo usuário](/user-guide/security#user-defined-deny-rules-approvalsdeny).

### Isole gravações de arquivos {#sandbox-file-writes}

`HERMES_WRITE_SAFE_ROOT` restringe `write_file` e `patch` aos prefixos de diretório que você listar — qualquer coisa fora é bloqueada de forma rígida. Múltiplas raízes são separadas por `:` no Unix:

```bash
export HERMES_WRITE_SAFE_ROOT=/path/to/project:/home/you/.hermes
```

Caminhos sensíveis dentro da raiz segura ainda são bloqueados — apontar para `$HOME` não permite gravar `~/.ssh/id_rsa`.

:::caution
Não adicione isso a `~/.hermes/.env` de forma casual. Se você definir apenas um diretório de projeto, o agente não poderá gravar em `~/.hermes/cron/jobs.json`, skills de perfil ou outro estado do Hermes fora desse prefixo. Inclua o home do Hermes como segunda raiz, como acima.
:::

### Mova a execução de comandos para fora do host {#move-command-execution-off-the-host}

O isolamento mais forte é não executar comandos na sua máquina. A ferramenta de terminal suporta vários [backends](/user-guide/features/tools#terminal-backends):

| Backend | Isolamento |
|---------|-----------|
| `local` | Nenhum — roda no host (verificações de comandos perigosos se aplicam) |
| `docker` | Container — o próprio container é o limite de segurança |
| `ssh` | Máquina remota — mantém a execução em um servidor separado |

```yaml
terminal:
  backend: docker
  docker_image: "nikolaik/python-nodejs:python3.11-nodejs20"
  docker_forward_env: []  # Explicit allowlist only; empty keeps secrets out of the container
```

Todo container Docker roda com configurações reforçadas — todas as capabilities Linux removidas (com um conjunto mínimo readicionado), `no-new-privileges`, limite de contagem de processos e mounts tmpfs com tamanho limitado. Com backend de container, comandos destrutivos dentro do container não prejudicam o host, por isso as verificações de comandos perigosos são ignoradas lá.

Para `ssh`, defina `terminal.backend: ssh` em `config.yaml` e forneça detalhes do host via `TERMINAL_SSH_HOST`, `TERMINAL_SSH_USER` e `TERMINAL_SSH_KEY` em `~/.hermes/.env`. Veja [Isolamento de rede](/user-guide/security#network-isolation).

### Se a mensageria estiver ativa: allowlists e pairing {#if-messaging-is-on-allowlists-and-pairing}

Executando o [gateway](/user-guide/security#user-authorization-gateway) nesta máquina? O padrão já nega: se nenhuma allowlist estiver configurada e `GATEWAY_ALLOW_ALL_USERS` não estiver definido, **todos os usuários são negados**. Mantenha explícito:

```bash
# ~/.hermes/.env
TELEGRAM_ALLOWED_USERS=123456789
GATEWAY_ALLOWED_USERS=123456789
```

Ou use pairing de DM em vez de fixar IDs: usuários desconhecidos recebem um código de pairing único e você os aprova pelo CLI com `hermes pairing approve <platform> <code>`. Nunca defina `GATEWAY_ALLOW_ALL_USERS=true` em uma máquina que importa.

## A camada de desfazer: checkpoints e `/rollback` {#the-undo-layer-checkpoints-and-rollback}

Portões de aprovação evitam danos; [checkpoints](/user-guide/checkpoints-and-rollback) revertem. Quando ativados, o Hermes faz snapshot do seu projeto automaticamente antes de operações destrutivas — `write_file`, `patch` e comandos de terminal destrutivos como `rm`, `mv`, `sed -i` e `git reset` — em um repositório git sombra em `~/.hermes/checkpoints/store/`. O `.git` real do projeto nunca é tocado.

Checkpoints são opt-in. Ative por sessão:

```bash
hermes chat --checkpoints
```

Ou globalmente:

```yaml
checkpoints:
  enabled: true
```

Depois, em uma sessão:

| Comando | Descrição |
|---------|-------------|
| `/rollback` | Lista todos os checkpoints com estatísticas de mudança |
| `/rollback diff <N>` | Pré-visualiza o que mudou desde o checkpoint N |
| `/rollback <N>` | Restaura para o checkpoint N (também desfaz o último turno do chat) |
| `/rollback <N> <file>` | Restaura um único arquivo do checkpoint N |

:::tip
Pré-visualize com `/rollback diff <N>` antes de restaurar e combine checkpoints com git worktrees para máxima segurança — cada sessão do Hermes em seu próprio worktree, com checkpoints como camada extra.
:::

## O que este modelo de ameaça é — e o que não é {#what-this-threat-model-is--and-isnt}

Seja claro sobre o que esses controles defendem. Como o guia de [Segurança](/user-guide/security#user-defined-deny-rules-approvalsdeny) coloca:

> Regras de negação são um guardrail contra um agente honesto porém errado, o mesmo modelo de ameaça do detector de padrões perigosos. Elas não são um sandbox contra um processo deliberadamente adversarial — para isso, use um backend isolado (Docker, Modal) ou um ambiente com egress restrito.

O mesmo vale para as proteções de escrita de arquivos: elas se aplicam apenas a `write_file` e `patch`, enquanto a ferramenta `terminal` roda como o mesmo usuário do SO. A denylist reduz danos acidentais e dá aos modelos um sinal claro de parada; não faz sandbox de um agente hostil ou comprometido. Se sua exigência é contenção em vez de guardrails, a resposta é um backend de terminal isolado — esse é o limite projetado para isso.

## Uma config inicial cautelosa {#a-cautious-starting-config}

Tudo acima, reunido. Ajuste ao gosto em `~/.hermes/config.yaml`:

```yaml
approvals:
  mode: manual                  # See every flagged command yourself
  timeout: 300                  # Unanswered prompts are denied (fail-closed)
  deny:                         # Never-run list — survives even /yolo
    - "git push --force*"
    - "*curl*|*sh*"
    - "dd if=* of=/dev/*"

security:
  redact_secrets: true          # Already the default; stated here for clarity

checkpoints:
  enabled: true                 # Snapshot before destructive operations

terminal:
  backend: docker               # Or ssh — keep execution off the host
  docker_forward_env: []        # No host secrets inside the container
```

E em `~/.hermes/.env`, se quiser o sandbox de escrita:

```bash
HERMES_WRITE_SAFE_ROOT=/path/to/project:/home/you/.hermes
```

## Veja também {#see-also}

- **[Segurança](/user-guide/security)** — referência completa de defesa em profundidade: cada padrão de aprovação, flags de hardening de container, autorização do gateway, filtragem de credenciais MCP
- **[Checkpoints e Rollback](/user-guide/checkpoints-and-rollback)** — configuração, manutenção do store e fluxos de restauração
- **[Ferramentas e Toolsets](/user-guide/features/tools)** — todos os backends de terminal e sua configuração
- **[Configuração](/user-guide/configuration)** — referência completa do `config.yaml`
