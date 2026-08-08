---
sidebar_position: 3
title: "Curator"
description: "Manutenção em segundo plano para skills criadas pelo agente — rastreamento de uso, obsolescência, arquivamento e revisão orientada por LLM"
---

# Curator

O curator é uma passagem de manutenção em segundo plano para **skills criadas pelo agente**. Ele rastreia com que frequência cada skill é visualizada, usada e corrigida, move skills muito tempo sem uso pelos estados `active → stale → archived`, e periodicamente dispara uma revisão curta feita por um modelo auxiliar que propõe consolidações ou corrige desvios.

Ele existe para que as skills criadas via [loop de autoaperfeiçoamento](/user-guide/features/skills#agent-managed-skills-skill_manage-tool) não se acumulem para sempre. Toda vez que o agente resolve um problema novo e salva uma skill, essa skill vai parar em `~/.hermes/skills/`. Sem manutenção, você acaba com dezenas de quase duplicatas estreitas que poluem o catálogo e desperdiçam tokens.

Por padrão (`prune_builtins: true`) o curator pode arquivar **skills embutidas (built-in) não utilizadas** (fornecidas com o repositório) após `archive_after_days` de não uso, junto com as skills criadas pelo agente que ele principalmente gerencia. Skills instaladas via hub (do [agentskills.io](https://agentskills.io)) estão sempre fora de alcance. Defina `curator.prune_builtins: false` para restaurar o comportamento antigo, apenas para skills criadas pelo agente, em que as skills embutidas nunca são tocadas. O curator também **nunca exclui automaticamente** — o pior resultado possível é o arquivamento em `~/.hermes/skills/.archive/`, que é recuperável.

Acompanha a [issue #7816](https://github.com/NousResearch/hermes-agent/issues/7816).

## Como ele funciona {#how-it-runs}

O curator é acionado por uma verificação de inatividade, não por um daemon de cron. No início de uma sessão de CLI, e em um tick recorrente dentro da thread de cron-ticker do gateway, o Hermes verifica se:

1. Passou tempo suficiente desde a última execução do curator (`interval_hours`, padrão de **7 dias**), e
2. O agente esteve ocioso por tempo suficiente (`min_idle_hours`, padrão de **2 horas**).

Se ambas forem verdadeiras, ele dispara um fork em segundo plano do `AIAgent` — o mesmo padrão usado pelos lembretes de autoaperfeiçoamento de memória/skills. O fork roda em seu próprio cache de prompt e nunca toca na conversa ativa.

:::info Comportamento na primeira execução
Em uma instalação totalmente nova (ou na primeira vez que uma instalação pré-curator faz um tick após `hermes update`), o curator **não roda imediatamente**. A primeira observação define `last_run_at` como "agora" e adia a primeira passagem real por um `interval_hours` completo. Isso dá a você um intervalo inteiro para revisar sua biblioteca de skills, fixar (pin) o que for importante, ou desativar totalmente antes que o curator sequer a toque.

Se você quiser ver o que o curator *faria* antes dele rodar de verdade, execute `hermes curator run --dry-run` — isso produz o mesmo relatório de revisão sem alterar a biblioteca.
:::

Uma execução tem duas fases:

1. **Transições automáticas** (determinísticas, sem LLM). Skills sem uso por `stale_after_days` (30) se tornam `stale`; skills sem uso por `archive_after_days` (90) são movidas para `~/.hermes/skills/.archive/`. Esse é o comportamento de limpeza sempre ativo — ele roda sempre que o curator está habilitado, sem custo de modelo auxiliar.
   - **Skills fixadas (pinned)** e **skills referenciadas por qualquer job de cron** (incluindo jobs pausados/desativados) são inteiramente ignoradas — tratadas como fixadas para as transições automáticas, para que uma agenda lenta ou pausada não consiga arquivar uma skill debaixo de um job. A consolidação também reescreve referências de skills em cron ao mesclar guarda-chuvas.
   - **Skills nunca usadas** (`use_count == 0`) recebem um piso de tolerância: elas não são arquivadas até terem pelo menos `stale_after_days` de idade. Zero usos é ausência de evidência, não prova de que a skill é descartável.
2. **Consolidação por LLM** (uma única passagem de modelo auxiliar com um teto alto de iterações — uma varredura completa de curadoria tipicamente leva de 50 a 100 chamadas de API) — **DESATIVADA por padrão**. Quando `curator.consolidate: true`, o agente bifurcado analisa as skills criadas pelo agente, pode ler qualquer uma delas com `skill_view`, e decide por skill se deve manter, corrigir (via `skill_manage`), consolidar skills sobrepostas em guarda-chuvas de nível de classe, ou arquivar via a ferramenta de terminal. A consolidação trata uma skill como um pacote completo: se uma skill tem `references/`, `templates/`, `scripts/`, `assets/`, ou links relativos para esses caminhos, o curator precisa ou mantê-la independente, ou realocar os arquivos de suporte necessários e reescrever os caminhos, ou arquivar o pacote inteiro sem alterações — nunca achatar apenas o `SKILL.md` dentro do arquivo `references/` de outra skill.

:::info A consolidação é opcional (opt-in)
Por padrão o curator apenas **poda (prune)** — a passagem determinística de inatividade marca skills como obsoletas e arquiva as não usadas há muito tempo. A passagem opinativa de **consolidação** por LLM (construção de guarda-chuvas, mesclagem de skills sobrepostas) fica desativada por padrão porque consome tokens de modelo auxiliar em cada execução e faz mudanças estruturais amplas na sua biblioteca. Ative-a com `curator.consolidate: true`, ou execute-a uma vez sob demanda com `hermes curator run --consolidate`.
:::

Skills fixadas ficam fora do alcance tanto das transições automáticas do curator quanto da própria ferramenta `skill_manage` do agente. Veja [Fixando uma skill](#pinning-a-skill) abaixo.

## Configuração {#configuration}

Todas as configurações ficam em `config.yaml` sob `curator:` (não em `.env` — isso não é um segredo). Padrões:

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

Para desativar por completo, defina `curator.enabled: false`. Para manter a poda sempre ativa mas ativar a consolidação por LLM, defina `curator.consolidate: true`.

### Executando a revisão em um modelo auxiliar mais barato {#running-the-review-on-a-cheaper-aux-model}

A passagem de revisão por LLM do curator é um slot de tarefa auxiliar comum — `auxiliary.curator` — ao lado de Vision, Compression, Session Search, etc. "Auto" significa "usar meu modelo de chat principal"; substitua o slot para fixar um provedor + modelo específico para a passagem de revisão.

**Mais fácil — `hermes model`:**

```bash
hermes model                   # → "Auxiliary models — side-task routing"
                               # → pick "Curator" → pick provider → pick model
```

O mesmo seletor está disponível no painel web sob a aba **Models**.

**Direto no config.yaml (equivalente):**

```yaml
auxiliary:
  curator:
    provider: openrouter
    model: google/gemini-3-flash-preview
    timeout: 600               # generous — reviews can take several minutes
```

Deixar `provider: auto` (o padrão) roteia a passagem de revisão pelo seu modelo de chat principal, igual ao comportamento de qualquer outra tarefa auxiliar.

:::note Configuração legada
Versões anteriores usavam um bloco pontual `curator.auxiliary.{provider,model}`. Esse caminho ainda funciona, mas emite uma linha de log de descontinuação — por favor migre para `auxiliary.curator` acima, para que o curator compartilhe a mesma infraestrutura (`hermes model`, aba Models do dashboard, `base_url`, `api_key`, `timeout`, `extra_body`) que qualquer outra tarefa auxiliar.
:::

## CLI {#cli}

```bash
hermes curator status         # last run, counts, pinned list, LRU top 5
hermes curator run            # trigger a run now (blocks until done). Prune-only unless curator.consolidate: true
hermes curator run --consolidate # force the LLM consolidation pass on for this run, overriding the config default
hermes curator run --background  # fire-and-forget: start the run in a background thread
hermes curator run --dry-run  # preview only — report without any mutations
hermes curator backup         # take a manual snapshot of ~/.hermes/skills/
hermes curator rollback       # restore from the newest snapshot
hermes curator rollback --list     # list available snapshots
hermes curator rollback --id <ts>  # restore a specific snapshot
hermes curator rollback -y         # skip the confirmation prompt
hermes curator pause          # stop runs until resumed
hermes curator resume
hermes curator pin <skill>    # never auto-transition this skill
hermes curator unpin <skill>
hermes curator adopt <skill>    # hand an unmanaged skill to the curator
hermes curator adopt --all-unmanaged   # hand over every unmanaged skill
hermes curator list-unmanaged   # itemize skills with no provenance marker
hermes curator restore <skill>  # move an archived skill back to active
hermes curator list-archived    # list skills currently in ~/.hermes/skills/.archive/
hermes curator archive <skill>  # manually archive a single skill now
hermes curator prune [--days N] # bulk-archive agent-created skills idle >= N days (default 90)
```

## Backups e rollback {#backups-and-rollback}

Antes de cada passagem real do curator, o Hermes faz um snapshot tar.gz de `~/.hermes/skills/` em `~/.hermes/skills/.curator_backups/<utc-iso>/skills.tar.gz`. Se uma passagem arquivar ou consolidar algo que você não queria que fosse tocado, você pode desfazer a execução inteira com um único comando:

```bash
hermes curator rollback        # restore newest snapshot (with confirmation)
hermes curator rollback -y     # skip the prompt
hermes curator rollback --list # see all snapshots with reason + size
```

O próprio rollback é reversível: antes de substituir a árvore de skills, o Hermes faz outro snapshot rotulado `pre-rollback to <target-id>`, para que um rollback feito por engano possa ser desfeito avançando para esse snapshot com `--id`.

Você também pode fazer snapshots manuais a qualquer momento com `hermes curator backup --reason "before-refactor"`. A string de `--reason` fica registrada no `manifest.json` do snapshot e é exibida em `--list`.

Os snapshots são podados para `curator.backup.keep` (padrão 5) para manter o uso de disco sob controle:

```yaml
curator:
  backup:
    enabled: true
    keep: 5
```

Defina `curator.backup.enabled: false` para desativar o snapshot automático. O comando manual `hermes curator backup` ainda funciona quando os backups estão desativados apenas se você definir `enabled: true` primeiro — a flag controla os dois caminhos de forma simétrica, então não há como pular acidentalmente o snapshot pré-execução em execuções que alteram dados.

`hermes curator status` também lista as cinco skills usadas menos recentemente — uma forma rápida de ver o que provavelmente ficará obsoleto em seguida.

Os mesmos subcomandos estão disponíveis como o comando de barra `/curator` dentro de uma sessão em execução (CLI ou plataformas do gateway).

## O que significa "criado pelo agente" {#what-agent-created-means}

O curator só gerencia skills explicitamente marcadas como **criadas pelo
agente** em `~/.hermes/skills/.usage.json`. Uma skill se qualifica quando
TODAS as condições a seguir são verdadeiras:

1. Seu nome **não** está em `~/.hermes/skills/.bundled_manifest` (skills embutidas fornecidas com o repositório).
2. Seu nome **não** está em `~/.hermes/skills/.hub/lock.json` (skills instaladas via hub).
3. Sua entrada em `.usage.json` tem `"created_by": "agent"` ou `"agent_created": true`.

Atualmente, só o **fork de revisão de autoaperfeiçoamento em segundo plano**
define esse marcador — quando ele cria uma nova skill guarda-chuva durante
sua passagem de revisão periódica (~a cada 10 turnos do agente). O fork em
segundo plano roda com uma origem de gravação `"background_review"` (via
`tools/skill_provenance.py`), que é o único caminho que aciona a chamada
`mark_agent_created()` dentro de `skill_manage`.

Skills que o agente em primeiro plano cria via
`skill_manage(action="create")` durante uma conversa **não** são marcadas
como criadas pelo agente — elas são consideradas dirigidas pelo usuário e o
curator deliberadamente as deixa em paz.

:::warning Suas skills escritas à mão NÃO são curadas
Se você criou manualmente um `SKILL.md` ou apontou o Hermes para um diretório
de skill externo, essa skill terá uma entrada em `.usage.json` com
`created_by: null` (ou o campo ausente). O curator não vai tocá-la. O mesmo
vale para skills que o agente em primeiro plano criou a seu pedido.

**Para ver quais skills o curator realmente gerencia**, execute
`hermes curator status`. Se a contagem de criadas pelo agente for 0, nenhuma
skill está atualmente na jurisdição do curator — a passagem de revisão
por LLM é pulada e o relatório mostrará
`Model: (not resolved) via (not resolved)` com `Duration: 0s`.
:::

### Adotando skills não gerenciadas {#adopting-unmanaged-skills}

`hermes curator status` reporta uma contagem de **não gerenciadas** ao lado
da contagem gerenciada:

```
curator-managed skills: 43 total  (agent-created=43  bundled=0)
  active     41
  stale       2
  archived    0

unmanaged (no provenance marker): 112 total
  pre-dates marker    34
  foreground-created  78
  never auto-staled or archived — `hermes curator adopt <name>` hands one over
```

Essas 112 são *elegíveis* para curadoria, mas permanentemente invisíveis ao
ciclo de vida, por um de dois motivos:

- **anterior ao marcador** — o registro foi escrito antes de `created_by`
  existir, então não carrega nenhum sinal de proveniência. A autoria é
  genuinamente desconhecida a partir do registro.
- **criada em primeiro plano** — um `skill_manage(create)` em primeiro plano
  deixou o marcador sem definição por design, já que skills que você
  pede pertencem a você.

Uma biblioteca grande pode, portanto, parecer totalmente curada enquanto a
maior parte dela está intocável. O `adopt` fecha essa lacuna por
**declaração**:

```bash
hermes curator list-unmanaged                    # itemize them, with reasons
hermes curator adopt <name> [<name> ...]         # hand specific skills over
hermes curator adopt --all-unmanaged --dry-run   # preview the full list
hermes curator adopt --all-unmanaged             # hand over everything (prompts)
hermes curator adopt --all-unmanaged --yes       # skip the prompt
```

A adoção grava o mesmo marcador `created_by: agent` que o fork de revisão em
segundo plano grava. Ela **não** reinicia o relógio de inatividade — uma
skill adotada mantém seu `last_activity_at` existente, então entregar
uma biblioteca que você já parou de usar não lhe dá uma nova janela de 90
dias. Espere que skills adotadas e há muito ociosas fiquem `stale` (ou
`archived`) na próxima passagem; esse é o objetivo.

A adoção também é o que desbloqueia a *melhoria* autônoma. O fork de revisão
em segundo plano se recusa a corrigir uma skill que não é gerenciada
pelo curator, então se ele perceber que uma das suas skills está
desatualizada ele vai dizer isso e recomendar a adoção em vez de editá-la.
Edições em primeiro plano (dirigidas pelo usuário) nunca são afetadas — você
e o agente sempre podem editar suas próprias skills sob pedido.

:::note `created_by` é uma flag de política, não uma declaração de proveniência
O campo armazenado se chama `created_by`, mas ele é interpretado como "a
curadoria autônoma pode tocar nisto?" — não "quem escreveu este arquivo".
Essas são perguntas diferentes, e para registros anteriores ao marcador a
resposta de autoria é simplesmente irrecuperável. O nome é mantido porque já
está em disco em todo `.usage.json`; leia-o como política. `hermes curator
adopt` muda a política, e não diz nada sobre quem é o autor do arquivo.
:::

:::note A proveniência é declarada, nunca inferida
A adoção é deliberadamente manual. A telemetria não consegue estabelecer
autoria: uma skill com milhares de correções prova que o agente a
**mantém**, não que o agente a **escreveu** — o Hermes edita constantemente
skills escritas por usuários a seu pedido. Uma heurística automática de
"parece feita pelo agente, adote" acabaria arquivando algo que você escreveu
à mão. O `adopt` se recusa a agir sobre skills embutidas, instaladas via
hub, externas e skills embutidas protegidas, que têm um dono diferente
de você.
:::

Skills que SÃO criadas pelo agente seguem o ciclo de vida completo:

- `active` → (30 dias sem uso) `stale` → (90 dias sem uso) `archived`
- Skills fixadas ignoram todas as transições automáticas
- Arquivamentos são recuperáveis via `hermes curator restore <name>`

Se você quiser proteger uma skill específica de ser tocada — por
exemplo, uma skill escrita à mão da qual você depende — use `hermes
curator pin <name>`. Veja a próxima seção.

## Fixando uma skill {#pinning-a-skill}

Fixar (pin) protege uma skill contra exclusão — tanto das passagens de
arquivamento automático do curator quanto da chamada de ferramenta
`skill_manage(action="delete")` do agente. Depois que uma skill é
fixada:

- O **curator** a ignora durante as transições automáticas (`active → stale →
  archived`), e sua passagem de revisão por LLM é instruída a deixá-la em
  paz.
- A **ferramenta `skill_manage` do agente** recusa `delete` sobre ela,
  apontando o usuário para `hermes curator unpin <name>`. Correções e
  edições ainda passam, então o agente pode melhorar o conteúdo de uma
  skill fixada conforme problemas aparecem, sem precisar de uma dança
  de pin/unpin/re-pin.

Fixe e desfixe com:

```bash
hermes curator pin <skill>
hermes curator unpin <skill>
```

A flag é armazenada como `"pinned": true` na entrada da skill em
`~/.hermes/skills/.usage.json`, então ela sobrevive entre sessões.

Skills citadas na lista `skills:` de qualquer job de cron são protegidas
da mesma forma para as **transições automáticas** (o curator nunca as torna
obsoletas nem as arquiva enquanto a referência permanecer), mesmo quando o
job está pausado ou desativado. Prefira uma fixação explícita quando você
também quiser bloquear o `skill_manage delete`.

Só skills **criadas pelo agente** podem ser fixadas — `hermes curator
pin` recusa em skills embutidas e instaladas via hub, com uma mensagem
explicativa se você tentar. Skills instaladas via hub nunca estão
sujeitas a mutação do curator. Skills embutidas só são tocadas quando
`curator.prune_builtins: true` (o padrão), e mesmo assim só são arquivadas
após `archive_after_days` de não uso — nunca corrigidas, consolidadas ou
excluídas. Defina `curator.prune_builtins: false` para isentar completamente
as skills embutidas.

Um pequeno conjunto de **built-ins protegidos** é fixado no código como nunca
arquiváveis e nunca consolidáveis, independentemente de
`curator.prune_builtins`, do estado de pin, ou do julgamento do LLM. Esses
sustentam UX estrutural — por exemplo, `plan` alimenta o fluxo do comando de
barra `/plan` — então arquivar um deles silenciosamente transformaria seu
comando de barra em um erro "Unknown command" sem nenhum aviso para você.
Built-ins protegidos são filtrados por completo da lista de candidatos do
curator, então a passagem de consolidação nunca os enxerga.

Se você quiser uma garantia mais forte do que "sem exclusão" — por exemplo,
congelar o conteúdo de uma skill por completo enquanto o agente ainda a
lê — edite `~/.hermes/skills/<name>/SKILL.md` diretamente com seu editor. O
pin protege contra exclusão feita por ferramentas, não contra seu próprio
acesso ao sistema de arquivos.

## Telemetria de uso {#usage-telemetry}

O curator mantém um sidecar em `~/.hermes/skills/.usage.json` com uma
entrada por skill:

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

Os contadores são incrementados quando:

- `view_count`: o agente chama `skill_view` na skill.
- `use_count`: a skill é carregada no prompt de uma conversa.
- `patch_count`: `skill_manage patch/edit/write_file/remove_file` é executado
  sobre a skill.

Skills embutidas e instaladas via hub são explicitamente excluídas das
gravações de telemetria.

## Relatórios por execução {#per-run-reports}

Cada execução do curator grava um diretório com timestamp em
`~/.hermes/logs/curator/`:

```
~/.hermes/logs/curator/
└── 20260429-111512/
    ├── run.json      # machine-readable: full fidelity, stats, LLM output
    └── REPORT.md     # human-readable summary
```

`REPORT.md` é uma forma rápida de ver o que uma dada execução fez — quais
skills transicionaram, o que o revisor LLM disse, quais skills ele
corrigiu. Bom para auditoria sem precisar fazer grep no `agent.log`.

:::note Sem candidatos? O relatório mostra `(not resolved)`
Quando o curator não tem **nenhuma skill criada pelo agente** para
revisar, a passagem de revisão por LLM é pulada por completo. O cabeçalho do
relatório mostrará `Model: (not resolved) via (not resolved)` com
`Duration: 0s` — isso **não** indica um erro de configuração ou falha na
resolução do modelo. Simplesmente significa que não havia candidatos, então
nenhum modelo chegou a ser invocado. A fase de transição automática ainda
roda e relata suas contagens normalmente.
:::

### Mapa de renomeações no resumo {#rename-map-in-the-summary}

Se uma execução consolidou várias skills sob um guarda-chuva (ou
mesclou quase duplicatas), o resumo visível ao usuário impresso no final da
execução inclui um mapa explícito de renomeações mostrando cada par
`nome-antigo → nome-novo` que o curator aplicou. Isso é além das linhas de
transição por skill, então quando uma leva de renomeações acontece você
consegue identificá-las rapidamente sem comparar o relatório JSON. A dica
também aparece em `hermes curator pin`, para que você possa fixar o nome do
guarda-chuva imediatamente se quiser travar o novo rótulo.

## Restaurando uma skill arquivada {#restoring-an-archived-skill}

Se o curator arquivou algo que você ainda quer:

```bash
hermes curator restore <skill-name>
```

Isso move a skill de volta de `~/.hermes/skills/.archive/` para a
árvore ativa e redefine seu estado para `active`. A restauração se recusa
se uma skill embutida ou instalada via hub já tiver sido instalada sob
o mesmo nome (isso ocultaria a versão upstream).

## Desativando por ambiente {#disabling-per-environment}

O curator vem ativado por padrão. Para desativá-lo:

- **Apenas para um perfil:** edite `~/.hermes/config.yaml` (ou a
  configuração do perfil ativo) e defina `curator.enabled: false`.
- **Apenas para uma execução:** `hermes curator pause` — a pausa persiste
  entre sessões; use `resume` para reativar.

O curator também se recusa a rodar se `min_idle_hours` ainda não decorreu,
então em uma máquina de desenvolvimento ativa ele naturalmente só roda
durante períodos de calmaria.

## Veja também {#see-also}

- [Sistema de Skills](/user-guide/features/skills) — como as skills funcionam em geral e o loop de autoaperfeiçoamento que as cria
- [Memória](/user-guide/features/memory) — uma revisão em segundo plano paralela que mantém a memória de longo prazo
- [Catálogo de Skills Embutidas](/reference/skills-catalog)
- [Issue #7816](https://github.com/NousResearch/hermes-agent/issues/7816) — proposta original e discussão de design
