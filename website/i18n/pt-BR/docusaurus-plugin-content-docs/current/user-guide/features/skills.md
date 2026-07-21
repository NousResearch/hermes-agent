---
sidebar_position: 2
title: "Sistema de skills"
description: "Documentos de conhecimento sob demanda — divulgação progressiva, skills gerenciadas pelo agente e Skills Hub"
---

# Sistema de skills

Skills são documentos de conhecimento sob demanda que o agente pode carregar quando necessário. Elas seguem um padrão de **divulgação progressiva** para minimizar o uso de tokens e são compatíveis com o padrão aberto [agentskills.io](https://agentskills.io/specification).

Todas as skills ficam em **`~/.hermes/skills/`** — o diretório principal e fonte da verdade. Na instalação nova, skills bundled são copiadas do repositório. Skills instaladas pelo hub e criadas pelo agente também vão para lá. O agente pode modificar ou excluir qualquer skill.

Você também pode apontar o Hermes para **diretórios externos de skills** — pastas adicionais escaneadas junto com o diretório local. Veja [Diretórios externos de skills](#external-skill-directories) abaixo.

Veja também:

- [Catálogo de skills bundled](/reference/skills-catalog)
- [Catálogo de skills opcionais oficiais](/reference/optional-skills-catalog)

## Começando do zero {#starting-with-a-blank-slate}

Por padrão, todo perfil é inicializado com o catálogo de skills bundled, e cada `hermes update` adiciona skills bundled recém-lançadas. Se você quiser um perfil **sem skills bundled** — e que permaneça vazio entre atualizações — há três caminhos:

**Na instalação** (aplica-se ao perfil padrão `~/.hermes`):

```bash
curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash -s -- --no-skills
```

**Na criação de perfil** (perfis nomeados):

```bash
hermes profile create research --no-skills
```

**Em um perfil já instalado** (padrão ou nomeado), alterne em runtime:

```bash
hermes skills opt-out            # stop future seeding — nothing on disk is touched
hermes skills opt-out --remove   # also delete UNMODIFIED bundled skills (confirms first)
hermes skills opt-in --sync      # undo: remove the marker and re-seed now
```

Os três caminhos gravam um marcador `.no-bundled-skills` no diretório do perfil. Enquanto o marcador estiver presente, o instalador, `hermes update` e qualquer sync de skills pulam a seed de skills bundled para aquele perfil. Exclua o marcador (ou execute `hermes skills opt-in`) para reabilitar.

:::note Seguro por padrão
`hermes skills opt-out` só interrompe a seed *futura* — nunca exclui nada que já está no disco. A flag opcional `--remove` exclui skills bundled **somente** quando estão inalteradas (byte-idênticas à versão que o Hermes instalou). Skills que você editou, instalou pelo hub ou escreveu você mesmo são sempre mantidas.
:::

## Usando skills {#using-skills}

Toda skill instalada fica automaticamente disponível como slash command:

```bash
# In the CLI or any messaging platform:
/gif-search funny cats
/axolotl help me fine-tune Llama 3 on my dataset
/github-pr-workflow create a PR for the auth refactor
/plan design a rollout for migrating our auth provider

# Just the skill name loads it and lets the agent ask what you need:
/excalidraw
```

### Empilhando várias skills em um comando {#stacking-multiple-skills-in-one-command}

Você pode invocar várias skills em uma única mensagem encadeando slash commands
no início — cada token `/skill` inicial (até 5) é carregado, e o restante
vira sua instrução:

```bash
/github-pr-workflow /test-driven-development fix issue #123 and open a PR
```

O parsing para no primeiro token que não é uma skill instalada, então argumentos
que começam com `/` (como caminhos de arquivo) nunca são engolidos:

```bash
/ocr-and-documents /tmp/scan.pdf extract the tables   # loads one skill; /tmp/scan.pdf is the argument
```

Para combinações que você usa repetidamente, prefira um [skill bundle](#skill-bundles) —
mesmo efeito sob um comando curto.

A skill bundled `plan` é um bom exemplo. Executar `/plan [request]` carrega as instruções da skill, dizendo ao Hermes para inspecionar o contexto se necessário, escrever um plano de implementação em markdown em vez de executar a tarefa, e salvar o resultado em `.hermes/plans/` relativo ao workspace/backend working directory ativo.

Você também pode interagir com skills por conversa natural:

```bash
hermes chat --toolsets skills -q "What skills do you have?"
hermes chat --toolsets skills -q "Show me the axolotl skill"
```

## Aprendendo uma skill a partir de fontes (`/learn`) {#learning-a-skill-from-sources-learn}

`/learn` é a forma rápida de transformar algo que você já sabe — ou um monte de
material de referência — em uma skill reutilizável, sem escrever o
`SKILL.md` manualmente. É aberto: aponte para *qualquer coisa que você consiga descrever* e o
agente reúne o material com as ferramentas que já tem, depois autoria uma skill
que segue os [padrões de autoria da casa](#skillmd-format) (descrição ≤60 chars,
ordem padrão de seções, enquadramento com ferramentas Hermes, sem comandos
inventados).

```bash
# A local SDK or doc directory — read with read_file / search_files
/learn the REST client in ~/projects/acme-sdk, focus on auth + pagination

# An online doc page — fetched with web_extract
/learn https://docs.example.com/api/quickstart

# The workflow you just walked the agent through in this conversation
/learn how I just deployed the staging server

# Pasted notes / a described procedure
/learn filing an expense: open the portal, New > Expense, attach the receipt, submit
```

Como o agente ao vivo faz a coleta, `/learn` funciona igual no CLI,
no gateway de mensagens, na TUI e no dashboard — e em qualquer backend de terminal
(local, Docker, remoto), já que não há motor de ingestão separado. No
**dashboard**, a página Skills tem um botão **Learn a skill** que abre um painel
com campo de diretório, campo de URL e caixa de texto aberta; ele compõe uma
requisição `/learn` e executa no chat.

Não há footprint de model-tool: `/learn` monta um prompt guiado por padrões e
entrega ao agente como turn normal. O agente salva o resultado com a
ferramenta `skill_manage`, então o [gate de aprovação de escrita](#gating-agent-skill-writes-skillswrite_approval)
se aplica se você o tiver ligado.

## Divulgação progressiva {#progressive-disclosure}

Skills usam um padrão de carregamento eficiente em tokens:

```
Level 0: skills_list()           → [{name, description, category}, ...]   (~3k tokens)
Level 1: skill_view(name)        → Full content + metadata       (varies)
Level 2: skill_view(name, path)  → Specific reference file       (varies)
```

O agente só carrega o conteúdo completo da skill quando realmente precisa.

## Formato SKILL.md {#skillmd-format}

```markdown
---
name: my-skill
description: Brief description of what this skill does
version: 1.0.0
platforms: [macos, linux]     # Optional — restrict to specific OS platforms
metadata:
  hermes:
    tags: [python, automation]
    category: devops
    fallback_for_toolsets: [web]    # Optional — conditional activation (see below)
    requires_toolsets: [terminal]   # Optional — conditional activation (see below)
    config:                          # Optional — config.yaml settings
      - key: my.setting
        description: "What this controls"
        default: "value"
        prompt: "Prompt for setup"
---

# Skill Title

## When to Use
Trigger conditions for this skill.

## Procedure
1. Step one
2. Step two

## Pitfalls
- Known failure modes and fixes

## Verification
How to confirm it worked.
```

### Skills específicas de plataforma {#platform-specific-skills}

Skills podem restringir-se a sistemas operacionais específicos usando o campo `platforms`:

| Valor | Corresponde a |
|-------|---------|
| `macos` | macOS (Darwin) |
| `linux` | Linux |
| `windows` | Windows |

```yaml
platforms: [macos]            # macOS only (e.g., iMessage, Apple Reminders, FindMy)
platforms: [macos, linux]     # macOS and Linux
```

Quando definido, a skill fica automaticamente oculta do system prompt, `skills_list()` e slash commands em plataformas incompatíveis. Se omitido, a skill carrega em todas as plataformas.

## Saída de skill e entrega de mídia {#skill-output-and-media-delivery}

Quando uma resposta de skill (ou qualquer resposta do agente) inclui um caminho absoluto simples para um arquivo de mídia — por exemplo `/home/user/screenshots/diagram.png` — o gateway detecta automaticamente, remove do texto visível e entrega o arquivo nativamente no chat do usuário (foto Telegram, anexo Discord, etc.) em vez de deixar o caminho bruto na mensagem.

Para áudio especificamente, a diretiva `[[audio_as_voice]]` promove arquivos de áudio a bolhas nativas de mensagem de voz em plataformas que suportam (Telegram, WhatsApp).

### Forçando entrega estilo documento: `[[as_document]]` {#forcing-document-style-delivery-as_document}

Às vezes você quer o **oposto** de preview inline: quer o arquivo entregue como anexo para download, não uma bolha de imagem recomprimida. O exemplo clássico é screenshot ou gráfico em alta resolução — o `sendPhoto` do Telegram recomprime para ~200 KB em 1280 px, destruindo legibilidade. Um PNG de 1–2 MB enviado via `sendDocument` mantém os bytes originais intactos.

Se uma resposta (ou qualquer texto dentro dela — tipicamente a última linha) contém a diretiva literal `[[as_document]]`, todo caminho de mídia extraído dessa resposta é entregue como anexo documento/arquivo em vez de bolha de imagem:

```
Here is your rendered chart:

/home/user/.hermes/cache/chart-q4-2025.png

[[as_document]]
```

A diretiva é removida antes da entrega, então usuários nunca a veem. A granularidade é intencionalmente tudo-ou-nada por resposta: emita `[[as_document]]` uma vez e todo caminho de imagem na mesma resposta é entregue como documento. Isso espelha o escopo de `[[audio_as_voice]]`.

Use a partir de uma skill quando:

- Você produz screenshots ou gráficos que o usuário precisa como arquivos (para editar em outra ferramenta, arquivar, compartilhar intactos).
- O preview lossy padrão obscureceria detalhe (texto pequeno, diagramas pixel-acurados, renders sensíveis a cor).

Plataformas sem caminho de documento separado (ex.: SMS) caem para o mecanismo de anexo que tiverem.

### Ativação condicional (skills de fallback) {#conditional-activation-fallback-skills}

Skills podem mostrar ou ocultar-se automaticamente com base em quais ferramentas estão disponíveis na sessão atual. Isso é mais útil para **skills de fallback** — alternativas gratuitas ou locais que só devem aparecer quando uma ferramenta premium está indisponível.

```yaml
metadata:
  hermes:
    fallback_for_toolsets: [web]      # Show ONLY when these toolsets are unavailable
    requires_toolsets: [terminal]     # Show ONLY when these toolsets are available
    fallback_for_tools: [web_search]  # Show ONLY when these specific tools are unavailable
    requires_tools: [terminal]        # Show ONLY when these specific tools are available
```

| Campo | Comportamento |
|-------|----------|
| `fallback_for_toolsets` | Skill fica **oculta** quando os toolsets listados estão disponíveis. Mostrada quando faltam. |
| `fallback_for_tools` | Igual, mas verifica ferramentas individuais em vez de toolsets. |
| `requires_toolsets` | Skill fica **oculta** quando os toolsets listados estão indisponíveis. Mostrada quando presentes. |
| `requires_tools` | Igual, mas verifica ferramentas individuais. |

**Exemplo:** A skill built-in `duckduckgo-search` usa `fallback_for_toolsets: [web]`. Quando você tem `FIRECRAWL_API_KEY` definida, o toolset web está disponível e o agente usa `web_search` — a skill DuckDuckGo permanece oculta. Se a API key estiver ausente, o toolset web fica indisponível e a skill DuckDuckGo aparece automaticamente como fallback.

Skills sem campos condicionais se comportam exatamente como antes — sempre são mostradas.

## Setup seguro no carregamento {#secure-setup-on-load}

Skills podem declarar variáveis de ambiente obrigatórias sem desaparecer da descoberta:

```yaml
required_environment_variables:
  - name: TENOR_API_KEY
    prompt: Tenor API key
    help: Get a key from https://developers.google.com/tenor
    required_for: full functionality
```

Quando um valor ausente é encontrado, o Hermes pede de forma segura apenas quando a skill é realmente carregada no CLI local. Você pode pular o setup e continuar usando a skill. Superfícies de mensagens nunca pedem segredos no chat — elas orientam você a usar `hermes setup` ou `~/.hermes/.env` localmente.

Uma vez definidas, env vars declaradas são **automaticamente repassadas** aos sandboxes `execute_code` e `terminal` — os scripts da skill podem usar `$TENOR_API_KEY` diretamente. Para env vars que não são de skill, use a opção de config `terminal.env_passthrough`. Veja [Repasse de variáveis de ambiente](/user-guide/security#environment-variable-passthrough) para detalhes.

### Configurações de skill {#skill-config-settings}

Skills também podem declarar configurações não secretas (caminhos, preferências) armazenadas em `config.yaml`:

```yaml
metadata:
  hermes:
    config:
      - key: myplugin.path
        description: Path to the plugin data directory
        default: "~/myplugin-data"
        prompt: Plugin data directory path
```

As configurações ficam em `skills.config` no seu config.yaml. `hermes config migrate` solicita configurações não configuradas, e `hermes config show` as exibe. Quando uma skill carrega, seus valores de config resolvidos são injetados no contexto para o agente saber automaticamente os valores configurados.

Veja [Configurações de skill](/user-guide/configuration#skill-settings) e [Criando skills — Config settings](/developer-guide/creating-skills#config-settings-configyaml) para detalhes.

## Estrutura de diretório de skills {#skill-directory-structure}

```text
~/.hermes/skills/                  # Single source of truth
├── mlops/                         # Category directory
│   ├── axolotl/
│   │   ├── SKILL.md               # Main instructions (required)
│   │   ├── references/            # Additional docs
│   │   ├── templates/             # Output formats
│   │   ├── scripts/               # Helper scripts callable from the skill
│   │   ├── examples/              # Referenced example outputs
│   │   └── assets/                # Supplementary files
│   └── vllm/
│       └── SKILL.md
├── devops/
│   └── deploy-k8s/                # Agent-created skill
│       ├── SKILL.md
│       └── references/
├── .hub/                          # Skills Hub state
│   ├── lock.json
│   ├── quarantine/
│   └── audit.log
└── .bundled_manifest              # Tracks seeded bundled skills
```

Instalações de URL e GitHub de terceiros incluem `SKILL.md` mais os arquivos locais exatos
referenciados em `references/`, `templates/`, `scripts/`, `assets/`
e `examples/`. Arquivos do repositório não referenciados não são copiados. O Hermes escaneia o
bundle completo em quarentena e registra a URL de origem, hash exato de conteúdo,
versão do scanner, achados, timestamp e status fresh-or-cached em
`skills/.hub/lock.json`.

## Diretórios externos de skills {#external-skill-directories}

Se você mantém skills fora do Hermes — por exemplo, um diretório compartilhado `~/.agents/skills/` usado por várias ferramentas de IA — você pode dizer ao Hermes para escanear esses diretórios também.

Adicione `external_dirs` na seção `skills` em `~/.hermes/config.yaml`:

```yaml
skills:
  external_dirs:
    - ~/.agents/skills
    - /home/shared/team-skills
    - ${SKILLS_REPO}/skills
```

Paths support `~` expansion and `${VAR}` environment variable substitution.

### Como funciona {#how-it-works}

- **Criar localmente, atualizar no lugar**: Novas skills criadas pelo agente são escritas em `~/.hermes/skills/`. Skills existentes são modificadas onde são encontradas, incluindo skills em `external_dirs`, quando o agente usa ações `skill_manage` como `patch`, `edit`, `write_file`, `remove_file` ou `delete`.
- **Dirs externos não são limite de proteção de escrita**: Se um diretório externo de skills for gravável pelo processo Hermes, atualizações de skill gerenciadas pelo agente podem alterar arquivos nesse diretório. Use permissões de filesystem ou setup separado de perfil/toolset se skills externas compartilhadas devem permanecer somente leitura.
- **Precedência local**: Se o mesmo nome de skill existir no dir local e em um dir externo, a versão local vence.
- **Integração completa**: Skills externas aparecem no índice do system prompt, `skills_list`, `skill_view` e como slash commands `/skill-name` — igual às skills locais.
- **Caminhos inexistentes são ignorados silenciosamente**: Se um diretório configurado não existir, o Hermes ignora sem erros. Útil para diretórios compartilhados opcionais que podem não estar presentes em toda máquina.

### Exemplo {#example}

```text
~/.hermes/skills/               # Local (primary, read-write)
├── devops/deploy-k8s/
│   └── SKILL.md
└── mlops/axolotl/
    └── SKILL.md

~/.agents/skills/               # External (shared, mutable if writable)
├── my-custom-workflow/
│   └── SKILL.md
└── team-conventions/
    └── SKILL.md
```

As quatro skills aparecem no seu índice de skills. Se você criar localmente uma skill chamada `my-custom-workflow`, ela faz shadow da versão externa.

## Skill bundles {#skill-bundles}

Skill bundles are tiny YAML files that group several skills under a single slash command. When you run `/<bundle-name>`, every skill listed in the bundle loads at once — useful when a particular task always benefits from the same set of skills together.

### Exemplo rápido {#quick-example}

```bash
# Create a bundle for backend feature work
hermes bundles create backend-dev \
  --skill github-code-review \
  --skill test-driven-development \
  --skill github-pr-workflow \
  -d "Backend feature work — review, test, PR workflow"
```

Depois no CLI ou em qualquer plataforma gateway:

```
/backend-dev refactor the auth middleware
```

O agente recebe as três skills carregadas em uma mensagem de usuário, com qualquer texto após o slash command anexado como instrução do usuário.

### Schema YAML {#yaml-schema}

Bundles ficam em **`~/.hermes/skill-bundles/<slug>.yaml`** e têm esta forma:

```yaml
name: backend-dev
description: Backend feature work — review, test, PR workflow.
skills:
  - github-code-review
  - test-driven-development
  - github-pr-workflow
instruction: |
  Always start by writing failing tests, then implement.
  Open the PR through the standard workflow with co-author tags.
```

Campos:
- `name` (opcional — padrão é o stem do filename) — nome de exibição do bundle. Normalizado para slug com hífen no slash command (`Backend Dev` → `/backend-dev`).
- `description` (opcional) — texto curto mostrado em `/bundles` e `hermes bundles list`.
- `skills` (lista obrigatória não vazia) — nomes de skills ou caminhos relativos ao seu diretório de skills. Use o mesmo identificador que passaria para `/<skill-name>`.
- `instruction` (opcional) — orientação extra prepended ao conteúdo de skill carregado. Útil para codificar "como sempre usamos estas juntas".

### Gerenciando bundles {#managing-bundles}

```bash
# List all installed bundles
hermes bundles list

# Inspect one bundle
hermes bundles show backend-dev

# Create a bundle interactively (omit --skill flags to enter them one per line)
hermes bundles create research

# Overwrite an existing bundle
hermes bundles create backend-dev --skill ... --force

# Delete a bundle
hermes bundles delete backend-dev

# Re-scan ~/.hermes/skill-bundles/ and report changes
hermes bundles reload
```

Dentro de uma sessão de chat, `/bundles` lista todo bundle instalado e suas skills.

### Comportamento {#behavior}

- **Bundles têm precedência sobre skills individuais** quando slugs colidem. Se você nomear um bundle `research` e também tiver uma skill chamada `research`, `/research` invoca o bundle. Isso é intencional — você optou pelo bundle ao nomeá-lo.
- **Skills ausentes são ignoradas, não fatais.** Se um bundle lista `skill-foo` e você não instalou, o bundle ainda carrega as skills que resolvem, e o agente recebe uma nota listando o que foi ignorado.
- **Bundles funcionam em toda superfície** — CLI interativo, TUI, chat do dashboard e toda plataforma gateway (Telegram, Discord, Slack, …) — porque o dispatch é centralizado no mesmo lugar dos comandos de skill individuais.
- **Bundles não invalidam o prompt cache.** Eles geram uma mensagem de usuário nova no momento da invocação, da mesma forma que `/<skill-name>` — sem mutação de system prompt.

### Quando bundles vencem instalar cada skill manualmente {#when-bundles-beat-installing-each-skill-manually}

Use um bundle quando:
- Você sempre emparelha as mesmas skills para uma tarefa recorrente (`/backend-dev`, `/release-prep`, `/incident-response`).
- Você quer um modelo mental um caractere mais curto do que digitar várias invocações `/skill` seguidas.
- Você quer enviar um "task profile" de equipe verificando o YAML do bundle em um repo dotfiles compartilhado e symlinkando em `~/.hermes/skill-bundles/`.

Um bundle é apenas um alias YAML — não instala skills para você. As skills em si já devem estar presentes (em `~/.hermes/skills/` ou diretório externo de skills). Caso contrário, a invocação do bundle só ignora as ausentes.

## Skills gerenciadas pelo agente (ferramenta skill_manage) {#agent-managed-skills-skill_manage-tool}

O agente pode criar, atualizar e excluir suas próprias skills via a ferramenta `skill_manage`. Esta é a **memória procedural** do agente — quando descobre um fluxo de trabalho não trivial, salva a abordagem como skill para reuso futuro.

Skills e memória trabalham juntos no loop de autoaperfeiçoamento: memória armazena
fatos duráveis pequenos que devem estar sempre no contexto, enquanto skills armazenam
procedimentos mais longos que devem carregar só quando relevantes. A revisão em background pode
sugerir ou staged mudanças de skill após uma sessão, mas o gate write-approval
abaixo permite exigir revisão humana antes dessas mudanças entrarem em vigor.

### Quando o agente cria skills {#when-the-agent-creates-skills}

- Após completar uma tarefa complexa (5+ chamadas de ferramenta) com sucesso
- Quando encontrou erros ou becos sem saída e achou o caminho que funciona
- Quando o usuário corrigiu sua abordagem
- Quando descobriu um fluxo de trabalho não trivial

### Ações {#actions}

| Action | Use for | Key params |
|--------|---------|------------|
| `create` | Nova skill do zero | `name`, `content` (SKILL.md completo), optional `category` |
| `patch` | Correções pontuais (preferido) | `name`, `old_string`, `new_string` |
| `edit` | Reescritas estruturais maiores | `name`, `content` (substituição completa do SKILL.md) |
| `delete` | Remove uma skill por completo | `name` |
| `write_file` | Adiciona/atualiza arquivos de suporte | `name`, `file_path`, `file_content` |
| `remove_file` | Remove um arquivo de suporte | `name`, `file_path` |

:::tip
A ação `patch` é preferida para atualizações — é mais eficiente em tokens que `edit` porque só o texto alterado aparece na chamada de ferramenta.
:::

### Gate de escritas de skill pelo agente (`skills.write_approval`) {#gating-agent-skill-writes-skillswrite_approval}

Por padrão o agente escreve skills livremente — inclusive pela [revisão de
autoaperfeiçoamento em background](/user-guide/features/memory#controlling-memory-writes-write_approval)
que roda após um turno. Se preferir aprovar toda escrita de skill primeiro
(modelos pequenos que julgam mal o que aprenderam, ambientes seguros, ou apenas
querer olhos no loop de autoaperfeiçoamento), ligue o gate de write-approval:

```yaml
skills:
  write_approval: false     # false = write freely (default) | true = require approval
```

Quando `write_approval: true`, toda escrita `skill_manage` (create / edit /
patch / delete / write_file / remove_file) é **staged** em vez de committed —
um SKILL.md é grande demais para revisar inline, então staging se aplica independentemente de
a escrita ter vindo de um turn em foreground ou da revisão em background.
Escritas staged sobrevivem reinícios em `~/.hermes/pending/skills/` e são
revisadas com o mesmo fluxo familiar approve/deny de comandos perigosos:

```
/skills pending             # list staged skill writes + a one-line gist each
/skills diff <id>           # full unified diff (best viewed in CLI or dashboard)
/skills approve <id>        # apply it (or 'all')
/skills reject <id>         # drop it (or 'all')
/skills approval on         # turn the gate on (or 'off') and persist it
```

A superfície de revisão funciona no CLI interativo e em plataformas de mensagens
(saída de diff truncada para bolhas de chat — leia o diff completo no CLI ou
no arquivo JSON pending). Escritas de memória têm o mesmo gate em
`memory.write_approval` — veja [Controlando escritas de memória](/user-guide/features/memory#controlling-memory-writes-write_approval).

> A configuração separada `skills.guard_agent_created` é um scanner de conteúdo
> (heurísticas de padrões perigosos), não um gate de aprovação — os dois são
> independentes. Veja [Guard em escritas de skill criadas pelo agente](/user-guide/configuration#guard-on-agent-created-skill-writes).

## Skills Hub {#skills-hub}

Navegue, busque, instale e gerencie skills de registries online, `skills.sh`, endpoints well-known diretos e skills opcionais oficiais.

### Comandos comuns {#common-commands}

```bash
hermes skills browse                              # Browse all hub skills (official first)
hermes skills browse --source official            # Browse only official optional skills
hermes skills search kubernetes                   # Search all sources
hermes skills search react --source skills-sh     # Search the skills.sh directory
hermes skills search https://mintlify.com/docs --source well-known
hermes skills inspect openai/skills/k8s           # Preview before installing
hermes skills install openai/skills/k8s           # Install with security scan
hermes skills install official/security/1password
hermes skills install skills-sh/vercel-labs/json-render/json-render-react --force
hermes skills install well-known:https://mintlify.com/docs/.well-known/skills/mintlify
hermes skills install https://sharethis.chat/SKILL.md              # Direct URL (+ referenced support files)
hermes skills install https://example.com/SKILL.md --name my-skill # Override name when frontmatter has none
hermes skills list --source hub                   # List hub-installed skills
hermes skills check                               # Check installed hub skills for upstream updates
hermes skills update                              # Reinstall hub skills with upstream changes when needed
hermes skills audit                               # Re-scan all hub skills for security
hermes skills uninstall k8s                       # Remove a hub skill
hermes skills reset google-workspace              # Un-stick a bundled skill from "user-modified" (see below)
hermes skills reset google-workspace --restore    # Also restore the bundled version, deleting your local edits
hermes skills publish skills/my-skill --to github --repo owner/repo
hermes skills snapshot export setup.json          # Export skill config
hermes skills tap add myorg/skills-repo           # Add a custom GitHub source
```

### Fontes de hub suportadas {#supported-hub-sources}

| Source | Example | Notes |
|--------|---------|-------|
| `official` | `official/security/1password` | Skills opcionais incluídas com o Hermes. |
| `skills-sh` | `skills-sh/vercel-labs/agent-skills/vercel-react-best-practices` | Pesquisável via `hermes skills search <query> --source skills-sh`. O Hermes resolve skills estilo alias quando o slug skills.sh difere da pasta do repo. |
| `well-known` | `well-known:https://mintlify.com/docs/.well-known/skills/mintlify` | Skills servidas diretamente de `/.well-known/skills/index.json` em um site. Pesquise usando a URL do site ou docs. |
| `url` | `https://sharethis.chat/SKILL.md` | URL HTTP(S) direta para `SKILL.md` mais arquivos de suporte explicitamente referenciados. Resolução de nome: frontmatter → slug da URL → prompt interativo → flag `--name`. |
| `github` | `openai/skills/k8s` | Instalações diretas de repo/caminho GitHub e taps customizados. |
| `clawhub`, `lobehub`, `browse-sh` | Identificadores específicos da fonte | Integrações de comunidade ou marketplace. |

### Hubs e registries integrados {#integrated-hubs-and-registries}

O Hermes integra atualmente estes ecossistemas de skills e fontes de descoberta:

#### 1. Skills opcionais oficiais (`official`)

Estas são mantidas no próprio repositório Hermes e instalam com confiança built-in.

- Catalog: [Official Optional Skills Catalog](../../reference/optional-skills-catalog)
- Source in repo: `optional-skills/`
- Example:

```bash
hermes skills browse --source official
hermes skills install official/security/1password
```

#### 2. skills.sh (`skills-sh`)

Este é o diretório público de skills da Vercel. O Hermes pode pesquisá-lo diretamente, inspecionar páginas de detalhe de skills, resolver slugs estilo alias e instalar do repo de origem subjacente.

- Directory: [skills.sh](https://skills.sh/)
- CLI/tooling repo: [vercel-labs/skills](https://github.com/vercel-labs/skills)
- Official Vercel skills repo: [vercel-labs/agent-skills](https://github.com/vercel-labs/agent-skills)
- Example:

```bash
hermes skills search react --source skills-sh
hermes skills inspect skills-sh/vercel-labs/json-render/json-render-react
hermes skills install skills-sh/vercel-labs/json-render/json-render-react --force
```

#### 3. Endpoints well-known de skills (`well-known`)

Esta é descoberta baseada em URL de sites que publicam `/.well-known/skills/index.json`. Não é um hub centralizado único — é uma convenção de descoberta web.

- Example live endpoint: [Mintlify docs skills index](https://mintlify.com/docs/.well-known/skills/index.json)
- Reference server implementation: [vercel-labs/skills-handler](https://github.com/vercel-labs/skills-handler)
- Example:

```bash
hermes skills search https://mintlify.com/docs --source well-known
hermes skills inspect well-known:https://mintlify.com/docs/.well-known/skills/mintlify
hermes skills install well-known:https://mintlify.com/docs/.well-known/skills/mintlify
```

#### 4. Skills GitHub diretas (`github`)

O Hermes pode instalar diretamente de repositórios GitHub e taps baseados em GitHub. Isso é útil quando você já conhece o repo/caminho ou quer adicionar seu próprio repo de origem customizado.

Taps padrão (navegáveis sem setup):
- [openai/skills](https://github.com/openai/skills)
- [anthropics/skills](https://github.com/anthropics/skills)
- [huggingface/skills](https://github.com/huggingface/skills)
- [NVIDIA/skills](https://github.com/NVIDIA/skills) — NVIDIA-verified skills (signed `skill.oms.sig` + governance `skill-card.md`)
- [garrytan/gstack](https://github.com/garrytan/gstack)

- Example:

```bash
hermes skills install openai/skills/k8s
hermes skills tap add myorg/skills-repo
```

**Agrupamentos de categoria (`skills.sh.json`).** Um tap GitHub pode incluir um
arquivo `skills.sh.json` na raiz do repo seguindo o
[schema skills.sh](https://skills.sh/schemas/skills.sh.schema.json). Seus
`groupings` (cada um com um `title` e lista de nomes de skills) são lidos na indexação
e viram os rótulos de categoria mostrados na página
[Skills Hub](https://hermes-agent.nousresearch.com/docs) — em vez de um
palpite derivado de tags. Isso é genérico: qualquer tap que inclua o arquivo recebe categorização
real, sem mudanças no lado Hermes.

```json
{
  "$schema": "https://skills.sh/schemas/skills.sh.schema.json",
  "groupings": [
    { "title": "Inference AI", "skills": ["dynamo-recipe-runner", "dynamo-router-sla"] },
    { "title": "Decision Optimization", "skills": ["cuopt-developer", "cuopt-install"] }
  ]
}
```

#### 5. ClawHub (`clawhub`)

Marketplace de skills de terceiros integrado como fonte community.

- Site: [clawhub.ai](https://clawhub.ai/)
- Hermes source id: `clawhub`

#### 6. Repos estilo marketplace Claude (`claude-marketplace`)

O Hermes suporta repos marketplace que publicam manifests de plugin/marketplace compatíveis com Claude.

Fontes integradas conhecidas incluem:
- [anthropics/skills](https://github.com/anthropics/skills)
- [aiskillstore/marketplace](https://github.com/aiskillstore/marketplace)

Hermes source id: `claude-marketplace`

#### 7. LobeHub (`lobehub`)

O Hermes pode pesquisar e converter entradas de agentes do catálogo público LobeHub em skills Hermes instaláveis.

- Site: [LobeHub](https://lobehub.com/)
- Public agents index: [chat-agents.lobehub.com](https://chat-agents.lobehub.com/)
- Backing repo: [lobehub/lobe-chat-agents](https://github.com/lobehub/lobe-chat-agents)
- Hermes source id: `lobehub`

#### 8. browse.sh (`browse-sh`)

O Hermes integra com [browse.sh](https://browse.sh), o catálogo Browserbase de 200+ arquivos SKILL.md de automação de navegador por site (Airbnb, Amazon, arXiv, 12306.cn, Etsy, Xero e muitos outros). Cada skill descreve como conduzir um site end-to-end e é adequada para uso com as ferramentas de navegador do Hermes e skills de automação de navegador que você já tenha instaladas.

- Site: [browse.sh](https://browse.sh/)
- Catalog API: `https://browse.sh/api/skills`
- Hermes source id: `browse-sh`
- Trust level: `community`

```bash
hermes skills search airbnb --source browse-sh
hermes skills inspect browse-sh/airbnb.com/search-listings-ddgioa
hermes skills install browse-sh/airbnb.com/search-listings-ddgioa
```

Identificadores usam a forma `browse-sh/<hostname>/<task-id>` e correspondem ao slug exposto pelo catálogo browse.sh. O conteúdo é resolvido pelo endpoint de detalhe por skill (`/api/skills/<slug>` → `skillMdUrl`), não pelo `sourceUrl` GitHub do catálogo.

#### 9. URL direta (`url`)

Instale `SKILL.md` diretamente de qualquer URL HTTP(S) — útil quando um autor hospeda uma skill no próprio site (sem listagem no hub, sem caminho GitHub para digitar). O Hermes também busca arquivos explicitamente referenciados em `references/`, `templates/`, `scripts/`, `assets/` e `examples/`, depois escaneia e instala o bundle completo.

- Hermes source id: `url`
- Identifier: the URL itself (no prefix needed)
- Scope: `SKILL.md` plus exact referenced support files in the allowlisted directories. Hermes does not enumerate or copy unrelated files from the host.

```bash
hermes skills install https://sharethis.chat/SKILL.md
hermes skills install https://example.com/my-skill/SKILL.md --category productivity
```

Resolução de nome, em ordem:
1. Campo `name:` no YAML frontmatter do SKILL.md (recomendado — toda skill bem formada tem um).
2. Nome do diretório pai do caminho da URL (ex.: `.../my-skill/SKILL.md` → `my-skill`, ou `.../my-skill.md` → `my-skill`), quando for identificador válido (`^[a-z][a-z0-9_-]*$`).
3. Prompt interativo em terminal com TTY.
4. Em superfícies non-interactive (slash command `/skills install` dentro da TUI, plataformas gateway, scripts), erro limpo apontando para o override `--name`.

```bash
# Frontmatter has no name and the URL slug is unhelpful — supply one:
hermes skills install https://example.com/SKILL.md --name sharethis-chat

# Or inside a chat session:
/skills install https://example.com/SKILL.md --name sharethis-chat
```

O nível de confiança é sempre `community` — o mesmo scan de segurança roda como para toda outra fonte. A URL é armazenada como identificador de instalação, então `hermes skills update` re-busca da mesma URL automaticamente quando você quiser atualizar.

### Scan de segurança e `--force` {#security-scanning-and-force}

Todas as skills instaladas pelo hub passam por um **scanner de segurança** que verifica exfiltração de dados, prompt injection, comandos destrutivos, sinais de supply-chain e outras ameaças.

`hermes skills inspect ...` agora também exibe metadata upstream quando disponível:
- repo URL
- skills.sh detail page URL
- install command
- weekly installs
- upstream security audit statuses
- well-known index/endpoint URLs

Use `--force` quando você revisou uma skill de terceiros e quer sobrescrever um block de policy não perigoso:

```bash
hermes skills install skills-sh/anthropics/skills/pdf --force
```

Comportamento importante:
- `--force` pode sobrescrever blocks de policy para achados estilo caution/warn.
- `--force` **não** sobrescreve um veredito de scan `dangerous`.
- Skills opcionais oficiais (`official/...`) são tratadas como confiança built-in e não mostram o painel de aviso de terceiros.

### Níveis de confiança {#trust-levels}

| Level | Source | Policy |
|-------|--------|--------|
| `builtin` | Incluídas com o Hermes | Sempre confiáveis |
| `official` | `optional-skills/` no repo | Confiança built-in, sem aviso de terceiros |
| `trusted` | Registries/repos confiáveis como `openai/skills`, `anthropics/skills`, `huggingface/skills`, `NVIDIA/skills` | Policy mais permissiva que fontes community |
| `community` | Todo o resto (`skills.sh`, endpoints well-known, repos GitHub customizados, a maioria dos marketplaces) | Achados não perigosos podem ser sobrescritos com `--force`; vereditos `dangerous` permanecem bloqueados |

### Ciclo de vida de atualização {#update-lifecycle}

O hub agora rastreia proveniência suficiente para re-verificar cópias upstream de skills instaladas:

```bash
hermes skills check          # Report which installed hub skills changed upstream
hermes skills update         # Reinstall only the skills with updates available
hermes skills update react   # Update one specific installed hub skill
```

Isso usa o identificador de origem armazenado mais o hash atual do bundle upstream para detectar drift.

:::tip Limites de rate do GitHub
Operações do skills hub usam a API GitHub, com limite de 60 requisições/hora para usuários não autenticados. Se vir erros de rate limit durante install ou search, defina `GITHUB_TOKEN` no seu `.env` para aumentar o limite para 5.000 requisições/hora. A mensagem de erro inclui dica acionável quando isso acontece.
:::

### Publicar um tap de skill customizado {#publishing-a-custom-skill-tap}

Se quiser compartilhar um conjunto curado de skills — para sua equipe, org ou publicamente — você pode publicá-las como **tap**: um repositório GitHub que outros usuários Hermes adicionam com `hermes skills tap add <owner/repo>`. Sem servidor, sem cadastro em registry, sem pipeline de release. Apenas um diretório de arquivos `SKILL.md`.

#### Layout do repo {#repo-layout}

Um tap é qualquer repo GitHub (público ou privado — privado precisa `GITHUB_TOKEN`) organizado assim:

```
owner/repo
├── skills/                       # default path; configurable per-tap
│   ├── my-workflow/
│   │   ├── SKILL.md              # required
│   │   ├── references/           # optional supporting files
│   │   ├── templates/
│   │   └── scripts/
│   ├── another-skill/
│   │   └── SKILL.md
│   └── third-skill/
│       └── SKILL.md
└── README.md                     # optional but helpful
```

Regras:
- Cada skill vive em seu próprio diretório sob o caminho raiz do tap (padrão `skills/`).
- O nome do diretório vira o install slug da skill.
- Cada diretório de skill deve conter um `SKILL.md` com [frontmatter SKILL.md](#skillmd-format) padrão (`name`, `description`, mais opcionais `metadata.hermes.tags`, `version`, `author`, `platforms`, `metadata.hermes.config`).
- Subdiretórios como `references/`, `templates/`, `scripts/`, `assets/` são baixados junto com `SKILL.md` na instalação.
- Skills cujo nome de diretório começa com `.` ou `_` são ignoradas.

O Hermes descobre skills listando todo subdiretório do caminho do tap e sondando cada um por `SKILL.md`.

#### Exemplo mínimo de tap {#minimal-tap-example}

```
my-org/hermes-skills
└── skills/
    └── deploy-runbook/
        └── SKILL.md
```

`skills/deploy-runbook/SKILL.md`:

```markdown
---
name: deploy-runbook
description: Our deployment runbook — services, rollback, Slack channels
version: 1.0.0
author: My Org Platform Team
metadata:
  hermes:
    tags: [deployment, runbook, internal]
---

# Deploy Runbook

Step 1: ...
```

Após push para GitHub, qualquer usuário Hermes pode assinar e instalar:

```bash
hermes skills tap add my-org/hermes-skills
hermes skills search deploy
hermes skills install my-org/hermes-skills/deploy-runbook
```

#### Caminhos não padrão {#non-default-paths}

Se suas skills não ficam em `skills/` (comum ao adicionar subtree `skills/` a um projeto existente), edite a entrada do tap em `~/.hermes/.hub/taps.json`:

```json
{
  "taps": [
    {"repo": "my-org/platform-docs", "path": "internal/skills/"}
  ]
}
```

O CLI `hermes skills tap add` usa `path: "skills/"` como padrão para taps novos; edite o arquivo diretamente se precisar de outro caminho. `hermes skills tap list` mostra o caminho efetivo por tap.

#### Instalar skills individuais diretamente (sem adicionar tap) {#installing-individual-skills-directly-without-adding-a-tap}

Usuários também podem instalar uma skill de qualquer repo GitHub público sem adicionar o repo inteiro como tap:

```bash
hermes skills install owner/repo/skills/my-workflow
```

Útil quando quer compartilhar uma skill sem pedir ao usuário que assine todo o registry.

#### Níveis de confiança para taps {#trust-levels-for-taps}

Taps novos recebem confiança `community` por padrão. Skills instaladas deles passam pelo scan de segurança padrão e mostram o painel de aviso de terceiros na primeira instalação. Se sua org ou uma fonte amplamente confiável deve ter confiança maior, adicione o repo a `TRUSTED_REPOS` em `tools/skills_hub.py` (requer PR no core Hermes).

#### Gerenciamento de taps {#tap-management}

```bash
hermes skills tap list                                # show all configured taps
hermes skills tap add myorg/skills-repo               # add (default path: skills/)
hermes skills tap remove myorg/skills-repo            # remove
```

Dentro de uma sessão em execução:

```
/skills tap list
/skills tap add myorg/skills-repo
/skills tap remove myorg/skills-repo
```

Taps ficam em `~/.hermes/.hub/taps.json` (criado sob demanda).

## Atualizações de skills bundled (`hermes skills reset`) {#bundled-skill-updates-hermes-skills-reset}

O Hermes inclui um conjunto de skills bundled em `skills/` dentro do repositório. Na instalação e em cada `hermes update`, uma passagem de sync copia essas para `~/.hermes/skills/` e registra um manifest em `~/.hermes/skills/.bundled_manifest` mapeando cada nome de skill ao hash de conteúdo no momento do sync (o **origin hash**).

Em cada sync, o Hermes recomputa o hash da sua cópia local e compara com o origin hash:

- **Inalterada** → seguro puxar mudanças upstream, copiar a nova versão bundled, registrar o novo origin hash.
- **Alterada** → tratada como **user-modified** e ignorada para sempre, para suas edições nunca serem sobrescritas.

A proteção é boa, mas tem uma aresta. Se você editar uma skill bundled e depois quiser abandonar suas mudanças e voltar à versão bundled apenas copiando de `~/.hermes/hermes-agent/skills/`, o manifest ainda guarda o origin hash *antigo* de quando o último sync bem-sucedido rodou. O conteúdo fresh copy-paste (hash bundled atual) não corresponderá a esse origin hash obsoleto, então o sync continua marcando como user-modified.

`hermes skills reset` é a saída de emergência:

```bash
# Safe: clears the manifest entry for this skill. Your current copy is preserved,
# but the next sync re-baselines against it so future updates work normally.
hermes skills reset google-workspace

# Full restore: also deletes your local copy and re-copies the current bundled
# version. Use this when you want the pristine upstream skill back.
hermes skills reset google-workspace --restore

# Non-interactive (e.g. in scripts or TUI mode) — skip the --restore confirmation.
hermes skills reset google-workspace --restore --yes
```

O mesmo comando funciona no chat como slash command:

```text
/skills reset google-workspace
/skills reset google-workspace --restore
```

:::note Perfis
Cada perfil tem seu próprio `.bundled_manifest` sob seu próprio `HERMES_HOME`, então `hermes -p coder skills reset <name>` afeta apenas aquele perfil.
:::

### Slash commands (dentro do chat) {#slash-commands-inside-chat}

Todos os mesmos comandos funcionam com `/skills`:

```text
/skills browse
/skills search react --source skills-sh
/skills search https://mintlify.com/docs --source well-known
/skills inspect skills-sh/vercel-labs/json-render/json-render-react
/skills install openai/skills/skill-creator --force
/skills check
/skills update
/skills reset google-workspace
/skills list
```

Skills opcionais oficiais ainda usam identificadores como `official/security/1password` e `official/migration/openclaw-migration`.
