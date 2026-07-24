---
sidebar_position: 3
title: "Curator"
description: "Manutenção em segundo plano de skills criadas pelo agente — rastreamento de uso, obsolescência, arquivamento e revisão orientada por LLM"
---

# Curator

O curator é uma passagem de manutenção em segundo plano para **skills criadas pelo agente**. Ele rastreia com que frequência cada skill é visualizada, usada e corrigida, move skills não usadas por muito tempo pelos estados `active → stale → archived` e, periodicamente, dispara uma revisão curta com um modelo auxiliar que propõe consolidações ou corrige desvios.

Ele existe para que skills criadas via o [loop de autoaperfeiçoamento](/user-guide/features/skills#agent-managed-skills-skill_manage-tool) não se acumulem para sempre. Toda vez que o agente resolve um problema novo e salva uma skill, ela vai parar em `~/.hermes/skills/`. Sem manutenção, você acaba com dezenas de quase-duplicatas estreitas que poluem o catálogo e desperdiçam tokens.

Por padrão (`prune_builtins: true`), o curator pode arquivar **skills built-in empacotadas não usadas** (enviadas com o repositório) após `archive_after_days` sem uso, junto com as skills criadas pelo agente que ele gerencia principalmente. Skills instaladas via hub (de [agentskills.io](https://agentskills.io)) estão sempre fora dos limites. Defina `curator.prune_builtins: false` para restaurar o comportamento antigo só para skills criadas pelo agente, em que skills empacotadas nunca são tocadas. O curator também **nunca exclui automaticamente** — o pior resultado é o arquivamento em `~/.hermes/skills/.archive/`, que é recuperável.

Acompanha a [issue #7816](https://github.com/NousResearch/hermes-agent/issues/7816).

## Como funciona {#how-it-runs}

O curator é acionado por uma verificação de inatividade, não por um daemon cron. Na inicialização da sessão CLI e em um tick recorrente dentro da thread cron-ticker do gateway, o Hermes verifica se:

1. Passou tempo suficiente desde a última execução do curator (`interval_hours`, padrão **7 dias**), e
2. O agente ficou ocioso tempo suficiente (`min_idle_hours`, padrão **2 horas**).

Se ambos forem verdadeiros, ele dispara um fork em segundo plano de `AIAgent` — o mesmo padrão usado pelos nudges de autoaperfeiçoamento de memória/skill. O fork roda em seu próprio cache de prompt e nunca toca a conversa ativa.

:::info Comportamento na primeira execução
Em uma instalação nova (ou na primeira vez que uma instalação pré-curator dispara após `hermes update`), o curator **não roda imediatamente**. A primeira observação define `last_run_at` como "agora" e adia a primeira passagem real por um `interval_hours` completo. Isso dá a você um intervalo inteiro para revisar sua biblioteca de skills, fixar o que for importante ou desativar totalmente antes de o curator tocar em qualquer coisa.

Se quiser ver o que o curator *faria* antes de rodar de verdade, execute `hermes curator run --dry-run` — ele produz o mesmo relatório de revisão sem mutar a biblioteca.
:::

Uma execução tem duas fases:

1. **Transições automáticas** (determinísticas, sem LLM). Skills não usadas por `stale_after_days` (30) viram `stale`; skills não usadas por `archive_after_days` (90) são movidas para `~/.hermes/skills/.archive/`. Esse é o comportamento de poda sempre ativo — roda sempre que o curator está habilitado, sem custo de modelo auxiliar.
2. **Consolidação por LLM** (passagem única com modelo auxiliar, `max_iterations=8`) — **DESLIGADA por padrão**. Quando `curator.consolidate: true`, o agente forkado examina as skills criadas pelo agente, pode ler qualquer uma delas com `skill_view` e decide por skill se mantém, corrige (via `skill_manage`), consolida sobrepostas em guarda-chuvas de nível de classe ou arquiva via a ferramenta terminal. A consolidação trata uma skill como pacote completo: se uma skill tem `references/`, `templates/`, `scripts/`, `assets/` ou links relativos para esses caminhos, o curator deve ou mantê-la standalone, realocar os arquivos de suporte necessários e reescrever caminhos, ou arquivar o pacote inteiro inalterado — não achatar só `SKILL.md` em um arquivo `references/` de outra skill.

:::info Consolidação é opt-in
Por padrão o curator só **poda** — a passagem determinística de inatividade marca skills como stale e arquiva as não usadas há muito tempo. A passagem de **consolidação** por LLM (construção de guarda-chuvas, fusão de skills sobrepostas) fica desligada por padrão porque consome tokens de modelo auxiliar a cada execução e faz mudanças estruturais amplas na sua biblioteca. Ative com `curator.consolidate: true`, ou rode uma vez sob demanda com `hermes curator run --consolidate`.
:::

Skills fixadas estão fora dos limites tanto das auto-transições do curator quanto da própria ferramenta `skill_manage` do agente. Veja [Fixar uma skill](#pinning-a-skill) abaixo.

## Configuração {#configuration}

Todas as configurações ficam em `config.yaml` sob `curator:` (não em `.env` — isso não é segredo). Padrões:

```yaml
curator:
  enabled: true
  interval_hours: 168          # 7 days
  min_idle_hours: 2
  stale_after_days: 30
  archive_after_days: 90
  consolidate: false           # LLM umbrella-building pass — opt-in (prune-only by default)
  prune_builtins: true         # archive unused bundled built-in skills too (hub skills always exempt)
```

Para desativar totalmente, defina `curator.enabled: false`. Para manter a poda sempre ativa mas optar pela consolidação por LLM, defina `curator.consolidate: true`.

### Rodar a revisão em um modelo auxiliar mais barato

A passagem de revisão por LLM do curator é um slot regular de tarefa auxiliar — `auxiliary.curator` — junto com Vision, Compression, Session Search, etc. "Auto" significa "usar meu modelo principal de chat"; sobrescreva o slot para fixar um provider + modelo específicos para a passagem de revisão.

**Mais fácil — `hermes model`:**

```bash
hermes model                   # → "Auxiliary models — side-task routing"
                               # → pick "Curator" → pick provider → pick model
```

O mesmo seletor está disponível no dashboard web na aba **Models**.

**Direto em config.yaml (equivalente):**

```yaml
auxiliary:
  curator:
    provider: openrouter
    model: google/gemini-3-flash-preview
    timeout: 600               # generous — reviews can take several minutes
```

Deixar `provider: auto` (o padrão) encaminha a passagem de revisão pelo mesmo modelo principal de chat, igual ao comportamento de toda outra tarefa auxiliar.

:::note Config legada
Releases anteriores usavam um bloco avulso `curator.auxiliary.{provider,model}`. Esse caminho ainda funciona, mas emite uma linha de log de depreciação — migre para `auxiliary.curator` acima para o curator compartilhar a mesma infraestrutura (`hermes model`, aba Models do dashboard, `base_url`, `api_key`, `timeout`, `extra_body`) de toda outra tarefa auxiliar.
:::

## CLI {#cli}

```bash
hermes curator status         # last run, counts, pinned list, LRU top 5
hermes curator run            # trigger a run now (blocks until done). Prune-only unless curator.consolidate: true
hermes curator run --consolidate # força a passagem de consolidação LLM nesta execução, sobrescrevendo o padrão do config
hermes curator run --background  # fire-and-forget: inicia a execução em thread em segundo plano
hermes curator run --dry-run  # só preview — relatório sem mutações
hermes curator backup         # tira snapshot manual de ~/.hermes/skills/
hermes curator rollback       # restaura do snapshot mais recente
hermes curator rollback --list     # lista snapshots disponíveis
hermes curator rollback --id <ts>  # restaura um snapshot específico
hermes curator rollback -y         # pula o prompt de confirmação
hermes curator pause          # stop runs until resumed
hermes curator resume
hermes curator pin <skill>    # never auto-transition this skill
hermes curator unpin <skill>
hermes curator restore <skill>  # move an archived skill back to active
hermes curator list-archived    # list skills currently in ~/.hermes/skills/.archive/
hermes curator archive <skill>  # manually archive a single skill now
hermes curator prune [--days N] # bulk-archive agent-created skills idle >= N days (default 90)
```

## Backups e rollback {#backups-and-rollback}

Antes de cada passagem real do curator, o Hermes tira um snapshot tar.gz de `~/.hermes/skills/` em `~/.hermes/skills/.curator_backups/<utc-iso>/skills.tar.gz`. Se uma passagem arquivar ou consolidar algo que você não queria tocado, você pode desfazer a execução inteira com um comando:

```bash
hermes curator rollback        # restaura snapshot mais recente (com confirmação)
hermes curator rollback -y     # pula o prompt
hermes curator rollback --list # ver todos os snapshots com motivo + tamanho
```

O próprio rollback é reversível: antes de substituir a árvore de skills, o Hermes tira outro snapshot marcado como `pre-rollback to <target-id>`, então um rollback errado pode ser desfeito avançando para esse com `--id`.

Você também pode tirar snapshots manuais a qualquer momento com `hermes curator backup --reason "before-refactor"`. A string `--reason` vai para o `manifest.json` do snapshot e aparece em `--list`.

Snapshots são podados para `curator.backup.keep` (padrão 5) para manter o uso de disco limitado:

```yaml
curator:
  backup:
    enabled: true
    keep: 5
```

Defina `curator.backup.enabled: false` para desativar snapshots automáticos. O comando manual `hermes curator backup` ainda funciona quando backups estão desativados só se você definir `enabled: true` primeiro — a flag controla ambos os caminhos simetricamente, então não há como pular acidentalmente o snapshot pré-execução em runs mutáveis.

`hermes curator status` também lista as cinco skills menos usadas recentemente — uma forma rápida de ver o que provavelmente ficará stale em seguida.

Os mesmos subcomandos estão disponíveis como o slash command `/curator` dentro de uma sessão em execução (CLI ou plataformas de gateway).

## O que significa "agent-created" {#what-agent-created-means}

O curator só gerencia skills explicitamente marcadas como **agent-created** em
`~/.hermes/skills/.usage.json`. Uma skill se qualifica quando TODAS as condições
a seguir são verdadeiras:

1. Seu nome **não** está em `~/.hermes/skills/.bundled_manifest` (skills empacotadas enviadas com o repositório).
2. Seu nome **não** está em `~/.hermes/skills/.hub/lock.json` (skills instaladas via hub).
3. Sua entrada em `.usage.json` tem `"created_by": "agent"` ou `"agent_created": true`.

Atualmente, só o **fork de revisão de autoaperfeiçoamento em segundo plano** define esse marcador
— quando cria uma nova skill guarda-chuva durante sua passagem periódica de revisão (~a cada 10
turnos do agente). O fork em segundo plano roda com origem de escrita `"background_review"`
(via `tools/skill_provenance.py`), que é o único caminho que dispara a chamada
`mark_agent_created()` em `skill_manage`.

Skills que o agente em primeiro plano cria via `skill_manage(action="create")` durante uma
conversa **não** são marcadas como agent-created — são consideradas
dirigidas pelo usuário e o curator intencionalmente não as toca.

:::warning Suas skills escritas à mão NÃO são curadas
Se você criou manualmente um `SKILL.md` ou apontou o Hermes para um diretório
de skill externo, essa skill terá uma entrada em `.usage.json` com `created_by: null`
(ou o campo ausente). O curator não a toca. O mesmo vale para
skills que o agente em primeiro plano criou a seu pedido.

**Para ver quais skills o curator realmente gerencia**, execute `hermes curator status`.
Se a contagem de agent-created for 0, nenhuma skill está atualmente na jurisdição
do curator — a passagem de revisão por LLM é pulada e o relatório mostrará
`Model: (not resolved) via (not resolved)` com `Duration: 0s`.
:::

Skills que SÃO agent-created seguem o ciclo de vida completo:

- `active` → (30d sem uso) `stale` → (90d sem uso) `archived`
- Skills fixadas ignoram todas as auto-transições
- Arquivos são recuperáveis via `hermes curator restore <name>`

Se quiser proteger uma skill específica de ser tocada — por exemplo uma
skill escrita à mão da qual você depende — use `hermes curator pin <name>`. Veja a próxima
seção.

## Fixar uma skill {#pinning-a-skill}

Fixar protege uma skill contra exclusão — tanto as passagens automáticas de arquivamento do curator quanto a chamada da ferramenta `skill_manage(action="delete")` do agente. Uma vez fixada:

- O **curator** a ignora durante auto-transições (`active → stale → archived`), e sua passagem de revisão por LLM é instruída a deixá-la em paz.
- A **ferramenta `skill_manage` do agente** recusa `delete` nela, apontando o usuário para `hermes curator unpin <name>`. Patches e edições ainda passam, então o agente pode melhorar o conteúdo de uma skill fixada conforme armadilhas surgem sem dança de pin/unpin/re-pin.

Fixe e desfixe com:

```bash
hermes curator pin <skill>
hermes curator unpin <skill>
```

A flag fica armazenada como `"pinned": true` na entrada da skill em `~/.hermes/skills/.usage.json`, então sobrevive entre sessões.

Só skills **agent-created** podem ser fixadas — `hermes curator pin` recusa em skills empacotadas e instaladas via hub com uma mensagem explicativa se você tentar. Skills instaladas via hub nunca estão sujeitas a mutação do curator. Skills built-in empacotadas só são tocadas quando `curator.prune_builtins: true` (o padrão), e mesmo assim só arquivadas após `archive_after_days` de não uso — nunca corrigidas, consolidadas ou excluídas. Defina `curator.prune_builtins: false` para isentar skills empacotadas totalmente.

Um pequeno conjunto de **built-ins protegidos** está hardcoded como nunca arquiváveis e nunca consolidáveis, independentemente de `curator.prune_builtins`, estado de pin ou julgamento do LLM. Eles sustentam UX crítica — por exemplo, `plan` alimenta o fluxo do slash command `/plan` — então arquivar um silenciosamente transformaria seu slash command em erro "Unknown command" sem sinal para você. Built-ins protegidos são filtrados da lista de candidatos do curator, então a passagem de consolidação nunca os vê.

Se quiser uma garantia mais forte que "sem exclusão" — por exemplo, congelar o conteúdo de uma skill inteiramente enquanto o agente ainda a lê — edite `~/.hermes/skills/<name>/SKILL.md` diretamente com seu editor. O pin protege exclusão via ferramenta, não seu próprio acesso ao filesystem.

## Telemetria de uso {#usage-telemetry}

O curator mantém um sidecar em `~/.hermes/skills/.usage.json` com uma entrada por skill:

```json
{
  "my-skill": {
    "use_count": 12,
    "view_count": 34,
    "last_used_at": "2026-04-24T18:12:03Z",
    "last_viewed_at": "2026-04-23T09:44:17Z",
    "patch_count": 3,
    "last_patched_at": "2026-04-20T22:01:55Z",
    "created_at": "2026-03-01T14:20:00Z",
    "state": "active",
    "pinned": false,
    "archived_at": null
  }
}
```

Contadores incrementam quando:

- `view_count`: o agente chama `skill_view` na skill.
- `use_count`: a skill é carregada no prompt de uma conversa.
- `patch_count`: `skill_manage patch/edit/write_file/remove_file` roda na skill.

Skills empacotadas e instaladas via hub estão explicitamente excluídas de escritas de telemetria.

## Relatórios por execução {#per-run-reports}

Cada execução do curator grava um diretório com timestamp em `~/.hermes/logs/curator/`:

```
~/.hermes/logs/curator/
└── 20260429-111512/
    ├── run.json      # machine-readable: full fidelity, stats, LLM output
    └── REPORT.md     # human-readable summary
```

`REPORT.md` é uma forma rápida de ver o que uma execução fez — quais skills transitaram, o que o revisor LLM disse, quais skills ele corrigiu. Bom para auditoria sem precisar fazer grep em `agent.log`.

:::note Sem candidatos? Relatório mostra `(not resolved)`
Quando o curator **não tem skills agent-created** para revisar, a passagem de revisão por LLM
é pulada inteiramente. O cabeçalho do relatório mostrará
`Model: (not resolved) via (not resolved)` com `Duration: 0s` — isso **não**
indica erro de configuração ou falha de resolução de modelo. Significa simplesmente que não
havia candidatos, então nenhum modelo foi invocado. A fase de auto-transição ainda
roda e reporta suas contagens normalmente.
:::

### Mapa de renomeação no resumo

Se uma execução consolidou várias skills sob um guarda-chuva (ou fundiu quase-duplicatas), o resumo visível ao usuário impresso no final da execução inclui um mapa explícito de renomeação mostrando cada par `old-name → new-name` que o curator aplicou. Isso está além das linhas de transição por skill, então quando uma onda de renomeações chega você as vê de relance sem diffar o relatório JSON. A dica também aparece sob `hermes curator pin` para você fixar o nome do guarda-chuva imediatamente se quiser travar o novo rótulo.

## Restaurar uma skill arquivada {#restoring-an-archived-skill}

Se o curator arquivou algo que você ainda quer:

```bash
hermes curator restore <skill-name>
```

Isso move a skill de volta de `~/.hermes/skills/.archive/` para a árvore ativa e redefine seu estado para `active`. A restauração recusa se uma skill empacotada ou instalada via hub foi instalada desde então com o mesmo nome (sombrearia a upstream).

## Desativar por ambiente {#disabling-per-environment}

O curator está ligado por padrão. Para desligar:

- **Só para um perfil:** edite `~/.hermes/config.yaml` (ou o config do perfil ativo) e defina `curator.enabled: false`.
- **Só para uma execução:** `hermes curator pause` — a pausa persiste entre sessões; use `resume` para reativar.

O curator também recusa rodar se `min_idle_hours` não passou, então em uma máquina de dev ativa ele naturalmente só roda durante intervalos quietos.

## Veja também {#see-also}

- [Sistema de Skills](/user-guide/features/skills) — como skills funcionam em geral e o loop de autoaperfeiçoamento que as cria
- [Memória](/user-guide/features/memory) — uma revisão em segundo plano paralela que mantém memória de longo prazo
- [Catálogo de Skills Empacotadas](/reference/skills-catalog)
- [Issue #7816](https://github.com/NousResearch/hermes-agent/issues/7816) — proposta original e discussão de design
