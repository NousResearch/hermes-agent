---
sidebar_position: 3
---

# Distribuições de profile: compartilhe um agente completo

Uma **distribuição de profile** empacota um agente Hermes completo — personalidade, skills, jobs cron, conexões MCP, config — como repositório git. Qualquer pessoa com acesso ao repositório pode instalar o agente inteiro com um comando, atualizá-lo no lugar e manter suas próprias memórias, sessões e API keys intactas.

Se um [profile](./profiles.md) é um agente local, uma distribuição é esse agente tornando-se compartilhável.

## Duas formas de compartilhar um profile {#two-ways-to-share-a-profile}

O Hermes tem dois caminhos de compartilhamento, e eles respondem perguntas diferentes. Distribuições são o durável; arquivos de export são o rápido.

| | **Distribuição** (repo git) | **Arquivo de export** (`.tar.gz`) |
|---|---|---|
| Envie por | `hermes profile install <repo>` | Envie um arquivo — chat, AirDrop, USB, email |
| Destinatário precisa | git, e acesso ao repo | O arquivo |
| Updates | `hermes profile update` puxa novas versões | Reenvie o arquivo |
| Versionamento | Tags, branches, commit SHAs | Nenhum — um snapshot no tempo |
| Custo de setup para o autor | `distribution.yaml` + `.gitignore` + um repo | Nenhum — um comando |
| Carrega | SOUL, config, skills, cron, MCP, plugins | O mesmo, **mais** o tema e layout do desktop |
| Feito com | `hermes profile install` / `update` | `/export` e `/import`, ou `hermes profile export` / `import` |

Escolha uma **distribuição** quando o agente é um produto que você vai continuar melhorando e outras pessoas devem acompanhar: o agente interno revisado de uma equipe, um release da comunidade, o mesmo agente implantado em cinco máquinas.

Escolha um **arquivo de export** quando você só quer que alguém tenha seu setup agora, ou está migrando para um laptop novo. Sem repo, sem manifest — rode `/export` no chat, entregue o arquivo, eles rodam `/import`. Veja [Exportar e importar um arquivo de profile](#export-and-import-a-profile-file).

Os dois não são exclusivos. Muitos autores dogfood um profile, `/export` para um colega ter uma segunda opinião, depois publicam como distribuição quando vale versionar.

## O que isso significa

Antes das distribuições, compartilhar um agente Hermes significava enviar a alguém:

1. Seu SOUL.md
2. Uma lista de skills para instalar
3. Seu config.yaml, menos os segredos
4. Uma descrição de quais servidores MCP você conectou
5. Quaisquer jobs cron que você agendou
6. Instruções de quais env vars definir

…e torcer para montarem corretamente. Cada bump de versão ou correção de bug significava repetir a entrega.

Com distribuições, tudo isso vive em um repositório git:

```
my-research-agent/
├── distribution.yaml    # manifest: name, version, env-var requirements
├── SOUL.md              # personalidade / system prompt do agente
├── config.yaml          # model, temperature, reasoning, tool defaults
├── skills/              # skills embutidas que vêm com o agente
├── cron/                # tarefas agendadas que o agente executa
└── mcp.json             # servidores MCP aos quais o agente se conecta
```

Os destinatários executam:

```bash
hermes profile install github.com/you/my-research-agent --alias
```

…e agora têm o agente inteiro. Preenchem suas próprias API keys (`.env.EXAMPLE` → `.env`) e podem executar `my-research-agent chat` ou falar com ele via Telegram / Discord / Slack / qualquer plataforma gateway. Quando você publica uma nova versão, eles executam `hermes profile update my-research-agent` e puxam suas alterações — memórias e sessões permanecem.

## Por que git?

Consideramos tarballs, arquivos HTTP, um formato customizado. Nenhum venceu o git:

- **Zero passo de build para autores.** Push no GitHub; consumidores instalam. Não há loop de "empacotar isso, fazer upload daquilo, atualizar o índice".
- **Tags, branches e commits já são o sistema de versionamento.** Um push de tag faz por nós o que "empacotar + publicar release" faz em outras ferramentas.
- **Updates são um fetch.** Não um re-download do arquivo inteiro.
- **Transparente.** Usuários podem navegar o repositório, ler diffs entre versões, abrir issues, fazer fork para customizar.
- **Repositórios privados funcionam de graça.** Chaves SSH, helpers `git credential`, credenciais armazenadas do GitHub CLI — qualquer auth que seu terminal já tem se aplica transparentemente.
- **Reprodutibilidade é um commit SHA.** A mesma coisa que pip e npm registram.

A troca: destinatários precisam de git instalado. Em qualquer máquina rodando Hermes em 2026, isso já é verdade.

## Quando usar uma distribuição?

Bons casos:

- **Você compartilha um agente especializado** — monitor de compliance, revisor de código, assistente de pesquisa, bot de suporte — com uma equipe ou com a comunidade.
- **Você implanta o mesmo agente em várias máquinas** e não quer copiar arquivos manualmente cada vez.
- **Você itera em um agente** e quer que destinatários peguem novas versões com um comando.
- **Você constrói um agente como produto** — defaults opinativos, skills curadas, prompts afinados — que outras pessoas devem usar como ponto de partida.

Não é adequado:

- **Você quer entregar seu setup a alguém uma vez, agora.** Uma distribuição precisa de um repo, um manifest e um `.gitignore`. `/export` não precisa de nada disso — veja [Exportar e importar um arquivo de profile](#export-and-import-a-profile-file). O mesmo para backup ou mover um profile para uma máquina nova.
- **Você quer compartilhar seu tema e layout do desktop.** Uma distribuição carrega o agente — SOUL, config, skills, cron, MCP, plugins. Um export feito do app desktop também carrega a aparência: skin, modo light/dark, temas custom, cor da rail e layout da janela.
- **Você quer compartilhar API keys junto com o agente.** `auth.json` e `.env` são deliberadamente excluídos das distribuições. Cada instalador traz suas próprias credenciais. (Arquivos de export também os removem.)
- **Você quer compartilhar memórias / sessões / histórico de conversa.** Isso é dado do usuário, não conteúdo de distribuição. Nunca é enviado. (Arquivos de export são diferentes aqui — leia [o que um arquivo de export contém](#what-an-export-file-contains) antes de enviar um.)

:::caution
**O Hermes não controla git.** As exclusões de arquivo descritas nesta página são aplicadas pelo **instalador** quando alguém executa `hermes profile install` ou `hermes profile update`. **Não** são aplicadas quando você executa `git add` ou `git commit`.
:::

## O ciclo de vida: autor → instalador → update

Abaixo está o fluxo end-to-end completo. Escolha o lado que importa para você.

---

## Para autores: publicando uma distribuição {#for-authors-publishing-a-distribution}

### Passo 1 — Comece de um profile funcional

Construa e refine o agente como qualquer outro profile:

```bash
hermes profile create research-bot
research-bot setup                    # configure model, API keys
# Edite ~/.hermes/profiles/research-bot/SOUL.md
# Instale skills, conecte servidores MCP, agende jobs cron, etc.
research-bot chat                     # dogfood até ficar certo
```

### Passo 2 — Adicione um `distribution.yaml`

Crie `~/.hermes/profiles/research-bot/distribution.yaml`:

```yaml
name: research-bot
version: 1.0.0
description: "Autonomous research assistant with arXiv and web tools"
hermes_requires: ">=0.12.0"
author: "Your Name"
license: "MIT"

# Diga aos instaladores quais env vars o agente precisa. São verificadas contra
# o shell do instalador e o .env existente do profile alvo para não incomodá-lo
# com chaves que já tem configuradas.
env_requires:
  - name: OPENAI_API_KEY
    description: "OpenAI API key (for model access)"
    required: true
  - name: SERPAPI_KEY
    description: "SerpAPI key for web search"
    required: false
    default: ""
```

Esse é o manifest inteiro. Todo campo exceto `name` tem default sensato.

### Passo 3 — Crie um `.gitignore` antes do primeiro commit

:::warning
Faça isso **antes** de executar `git init` ou `git add`. Se você já conversou com o profile, executou setup ou usou de outra forma, o diretório agora contém arquivos que você não deve enviar: `.env`, `auth.json`, `memories/`, `sessions/`, `state.db*`, `logs/` e mais.
:::

Crie `~/.hermes/profiles/research-bot/.gitignore` com no mínimo:

```gitignore
# Credentials & secrets — NEVER commit
auth.json
.env
.env.EXAMPLE    # generated by install, not authorship domain

# Runtime databases & state
state.db
state.db-shm
state.db-wal
hermes_state.db
response_store.db
response_store.db-shm
response_store.db-wal
gateway.pid
gateway_state.json
processes.json
auth.lock
active_profile
.update_check

# User data — NEVER commit
memories/
sessions/
logs/
plans/
workspace/
home/

# Caches & generated artifacts
image_cache/
audio_cache/
document_cache/
browser_screenshots/
cache/

# Infrastructure (should not be in profile dir, but safe to exclude)
hermes-agent/
.worktrees/
profiles/
bin/
node_modules/

# User customization namespace — your local overrides
local/

# Checkpoints & backups (can be huge)
checkpoints/
sandboxes/
backups/

# Logs
errors.log
.hermes_history
```

Isso espelha os [caminhos hard-excluded](#whats-not-in-a-distribution-ever) que o instalador remove do lado dele. Qualquer outra coisa que queira manter fora do repositório (arquivos scratch, assets grandes, skills só locais) também deve ir aqui.

### Passo 4 — Push para um repositório git

```bash
cd ~/.hermes/profiles/research-bot
git init
git add .
git commit -m "v1.0.0"
git remote add origin git@github.com:you/research-bot.git
git tag v1.0.0
git push -u origin main --tags
```

O repositório agora é uma distribuição. Qualquer pessoa com acesso pode instalá-la.

:::note
O instalador também removerá os [caminhos hard-excluded](#whats-not-in-a-distribution-ever) mesmo se um autor de alguma forma os enviar — mas isso só protege instaladores, não o autor.
:::

### Passo 5 — Marque releases versionados

Sempre que o agente atinge um ponto estável, incremente a versão e marque:

```bash
# Edite distribution.yaml: version: 1.1.0
git add distribution.yaml SOUL.md skills/
git commit -m "v1.1.0: tighter research SOUL, add arxiv skill"
git tag v1.1.0
git push --tags
```

Destinatários que executam `hermes profile update research-bot` puxarão o mais recente.

### Como fica o repositório

Uma distribuição autoral completa:

```
research-bot/
├── .gitignore                   # exclui secrets & user data (veja Passo 3)
├── distribution.yaml            # obrigatório
├── SOUL.md                      # fortemente recomendado
├── config.yaml                  # model, provider, tool defaults
├── mcp.json                     # conexões de servidores MCP
├── skills/
│   ├── arxiv-search/SKILL.md
│   ├── paper-summarization/SKILL.md
│   └── citation-lookup/SKILL.md
├── cron/
│   └── weekly-digest.json       # tarefas agendadas
└── README.md                    # descrição humana (opcional)
```

### Propriedade da distribuição vs do usuário

Quando um instalador atualiza para nova versão, algumas coisas são substituídas (domínio do autor) e outras permanecem (domínio do instalador). Padrões:

| Category | Paths | On update |
|---|---|---|
| **Distribution-owned** | `SOUL.md`, `config.yaml`, `mcp.json`, `skills/`, `cron/`, `distribution.yaml` | Substituídos do novo clone |
| **Config override** | `config.yaml` | Na prática preservado por padrão — o instalador pode ter afinado model ou provider. Passe `--force-config` no update para resetar. |
| **User-owned** | `memories/`, `sessions/`, `state.db*`, `auth.json`, `.env`, `logs/`, `workspace/`, `plans/`, `home/`, `*_cache/`, `local/` | Nunca tocados |

Você pode sobrescrever a lista distribution-owned no manifest:

```yaml
distribution_owned:
  - SOUL.md
  - skills/research/            # só minhas research skills; outras skills instaladas permanecem
  - cron/digest.json
```

Quando omitido, aplicam-se os padrões acima — o que a maioria das distribuições quer.

---

## Para instaladores: usando uma distribuição

### Instalar

```bash
hermes profile install github.com/you/research-bot --alias
```

O que acontece:

1. Clona o repositório em diretório temporário.
2. Lê `distribution.yaml`, mostra o manifest (name, version, description, author, required env vars).
3. Verifica cada env var obrigatória contra seu ambiente shell e o `.env` existente do profile alvo. Marca cada uma como `✓ set` ou `needs setting` para você saber exatamente o que configurar.
4. Pede confirmação. Passe `-y` / `--yes` para pular.
5. Copia arquivos distribution-owned para `~/.hermes/profiles/research-bot/` (ou onde o `name` do manifest resolver). Os [caminhos hard-excluded](#whats-not-in-a-distribution-ever) são removidos durante a cópia, mesmo se o autor os deixou no repositório por acidente.
6. Grava `.env.EXAMPLE` com as chaves obrigatórias comentadas — copie para `.env` e preencha.
7. Com `--alias`, cria wrapper para executar `research-bot chat` diretamente.

### Tipos de origem

Qualquer URL git funciona:

```bash
# Atalho GitHub
hermes profile install github.com/you/research-bot

# HTTPS completo
hermes profile install https://github.com/you/research-bot.git

# SSH
hermes profile install git@github.com:you/research-bot.git

# Self-hosted, GitLab, Gitea, Forgejo — qualquer host Git
hermes profile install https://git.example.com/team/research-bot.git

# Repositório privado usando sua auth git configurada
hermes profile install git@github.com:your-org/internal-bot.git

# Diretório local durante desenvolvimento (sem git push)
hermes profile install ~/my-profile-in-progress/
```

### Sobrescrever o nome do profile

Dois usuários querendo a mesma distribuição sob nomes de profile diferentes:

```bash
# Alice
hermes profile install github.com/acme/support-bot --name support-us --alias
# Bob (mesma distribuição, nome local diferente)
hermes profile install github.com/acme/support-bot --name support-eu --alias
```

### Preencher env vars

Após instalar, o profile do agente contém `.env.EXAMPLE`:

```
# Environment variables required by this Hermes distribution.
# Copy to `.env` and fill in your own values before running.

# OpenAI API key (for model access)
# (required)
OPENAI_API_KEY=

# SerpAPI key for web search
# (optional)
# SERPAPI_KEY=
```

Copie:

```bash
cp ~/.hermes/profiles/research-bot/.env.EXAMPLE ~/.hermes/profiles/research-bot/.env
# Edite .env, cole suas chaves reais
```

Chaves obrigatórias que já estavam no seu ambiente shell (ex.: `OPENAI_API_KEY` exportada no `~/.zshrc`) são marcadas `✓ set` durante a instalação — você não precisa duplicá-las em `.env`.

### Verificar o que instalou

```bash
hermes profile info research-bot
```

Mostra:

```
Distribution: research-bot
Version:      1.0.0
Description:  Autonomous research assistant with arXiv and web tools
Author:       Your Name
Requires:     Hermes >=0.12.0
Source:       https://github.com/you/research-bot
Installed:    2026-05-08T17:04:32+00:00

Environment variables:
  OPENAI_API_KEY (required) — OpenAI API key (for model access)
  SERPAPI_KEY (optional) — SerpAPI key for web search
```

`hermes profile list` também mostra coluna `Distribution` para ver de relance quais profiles vieram de repositórios e quais você construiu manualmente:

```
 Profile          Model                        Gateway      Alias        Distribution
 ───────────────    ───────────────────────────    ───────────    ───────────    ────────────────────
 ◆default         claude-sonnet-4              stopped      —            —
  coder           gpt-5                        stopped      coder        —
  research-bot    claude-opus-4                stopped      research-bot research-bot@1.0.0
  telemetry       claude-sonnet-4              running      telemetry    telemetry@2.3.1
```

### Atualizar

```bash
hermes profile update research-bot
```

O que acontece:

1. Re-clona o repositório da URL de origem registrada.
2. Substitui arquivos distribution-owned (SOUL, skills, cron, mcp.json).
3. **Preserva** seu `config.yaml` — você pode ter afinado model, temperature ou outras configurações. Passe `--force-config` para sobrescrever.
4. **Nunca toca** dados do usuário: memories, sessions, auth, `.env`, logs, state.

Sem re-download do arquivo inteiro. Sem pisar suas alterações locais de config. Sem apagar seu histórico de conversa.

### Remover

```bash
hermes profile delete research-bot
```

O prompt de delete mostra info da distribuição antes de pedir confirmação:

```
Profile: research-bot
Path:    ~/.hermes/profiles/research-bot
Model:   claude-opus-4 (anthropic)
Skills:  12
Distribution: research-bot@1.0.0
Installed from: https://github.com/you/research-bot

This will permanently delete:
  • All config, API keys, memories, sessions, skills, cron jobs
  • Command alias (~/.local/bin/research-bot)

Type 'research-bot' to confirm:
```

Assim você nunca apaga um agente por acidente sem saber de onde veio ou poder reinstalá-lo.

---

## Casos de uso e padrões

### Pessoal: sincronizar um agente entre máquinas

Você construiu um assistente de pesquisa no laptop. Quer o mesmo agente na workstation.

```bash
# Laptop — crie .gitignore primeiro (veja "Para autores" Passo 3), depois:
cd ~/.hermes/profiles/research-bot
git init && git add . && git status   # confirme que não há secrets staged
git commit -m "initial"
git remote add origin git@github.com:you/research-bot.git
git push -u origin main

# Workstation
hermes profile install github.com/you/research-bot --alias
# Preencha .env. Pronto.
```

Qualquer iteração no laptop (`git commit && push`) puxa na workstation com `hermes profile update research-bot`. Memórias ficam por máquina — o laptop lembra suas conversas, a workstation lembra as dela, não colidem.

### Equipe: entregar um agente interno revisado

Sua equipe de engenharia quer um bot compartilhado de revisão de PR com SOUL específico, skills específicas e cron que passa todo PR por ele.

```bash
# Tech lead — crie .gitignore primeiro (veja "Para autores" Passo 3), depois:
cd ~/.hermes/profiles/pr-reviewer
# ... construa e afinar ...
git init && git add . && git status   # confirme que não há secrets staged
git commit -m "v1.0 PR reviewer"
git tag v1.0.0
git push -u origin main --tags    # push para Git host interno da empresa

# Cada engenheiro
hermes profile install git@github.com:your-org/pr-reviewer.git --alias
# Preencha .env com API key própria (cobrada neles), .env.EXAMPLE aponta o que é obrigatório
pr-reviewer chat
```

Quando o lead publica v1.1 (SOUL melhor, skill nova), engenheiros executam `hermes profile update pr-reviewer` e todos estão na nova versão em minutos.

### Comunidade: publicar um agente público

Você construiu algo novo — talvez um "Polymarket trader" ou um "academic paper summarizer" ou um "Minecraft server ops assistant". Quer compartilhar.

```bash
# Você — crie .gitignore primeiro (veja "Para autores" Passo 3), depois:
cd ~/.hermes/profiles/polymarket-trader
# Escreva um README.md sólido na raiz do repo — GitHub mostra na página do repo
git init && git add . && git status   # confirme que não há secrets staged
git commit -m "v1.0"
git tag v1.0.0
# Publique em repositório GitHub público
git remote add origin https://github.com/you/hermes-polymarket-trader.git
git push -u origin main --tags

# Qualquer pessoa
hermes profile install github.com/you/hermes-polymarket-trader --alias
```

Tweete o comando de instalação. Quem experimentar manda issues e PRs. Se alguém quiser customizar, faz fork — mesmo fluxo git que todo mundo já conhece.

### Produto: entregar um agente opinativo

Você construiu Hermes-on-top — talvez um harness de monitoramento de compliance, stack de suporte ao cliente, plataforma de pesquisa de domínio específico. Quer distribuir como produto.

```yaml
# distribution.yaml
name: telemetry-harness
version: 2.3.1
description: "Compliance telemetry harness — monitors and reviews regulated workflows"
hermes_requires: ">=0.13.0"
author: "Acme Compliance Inc."
license: "Commercial"

env_requires:
  - name: ACME_API_KEY
    description: "Your Acme Compliance license key (email support@acme.com)"
    required: true
  - name: OPENAI_API_KEY
    description: "OpenAI API key for model access"
    required: true
  - name: GRAPHITI_MCP_URL
    description: "URL for your Graphiti knowledge graph instance"
    required: false
    default: "http://127.0.0.1:8000/sse"
```

Seus clientes instalam com um comando; o preview de instalação diz exatamente quais chaves ter prontas; updates saem no momento em que você marca nova release; dados de compliance deles (`memories/`, `sessions/`) nunca saem da máquina deles.

### Efêmero: scripts one-off em infra compartilhada

Você é o líder de ops. Quer um agente temporário que diagnostica um incidente de produção — SOUL pronto com as ferramentas e conexões MCP certas — rodando nos laptops de três engenheiros de plantão na próxima semana.

```bash
# Você — crie .gitignore primeiro (veja "Para autores" Passo 3), depois:
# Construa o profile, commit, push repositório privado
git push -u origin main

# Cada plantonista
hermes profile install git@github.com:your-org/incident-2026-q2.git --alias

# Incidente resolvido — desmonte
hermes profile delete incident-2026-q2
```

O ciclo install-delete é barato o suficiente para ser descartável.

---

## Receitas

### Fixar em versão específica

:::note
Fixação de ref git (`#v1.2.0`) está planejada, mas não está no release inicial — install atualmente segue a branch default. Acompanhe sua versão instalada via `hermes profile info <name>` e segure updates até estar pronto.
:::

### Verificar sua versão vs. a mais recente

```bash
# Sua versão instalada
hermes profile info research-bot | grep Version

# Upstream mais recente (sem instalar)
git ls-remote --tags https://github.com/you/research-bot | tail -5
```

### Manter customizações locais de config através de updates

O comportamento padrão de update já faz isso: `config.yaml` é preservado. Para segurança, escreva seus ajustes locais em arquivo que a distribuição não possui:

```yaml
# ~/.hermes/profiles/research-bot/local/my-overrides.yaml
# (distribution never touches local/)
```

…e referencie de `config.yaml` ou seu SOUL conforme necessário.

### Forçar reinstalação limpa

```bash
# Apagar e reinstalar do zero (perde memories/sessions também)
hermes profile delete research-bot --yes
hermes profile install github.com/you/research-bot --alias

# Update para main atual mas reset config.yaml para default da distribuição
hermes profile update research-bot --force-config --yes
```

### Fork e customizar

O fluxo git padrão — distribuições são só repositórios:

```bash
# Fork no GitHub, depois instale seu fork
hermes profile install github.com/yourname/forked-research-bot --alias

# Itere localmente em ~/.hermes/profiles/forked-research-bot/
# Edite SOUL.md, commit, push para seu fork
# Mudanças upstream: puxe para seu fork do jeito usual
```

### Testar uma distribuição antes de publicar

Da máquina do autor:

```bash
# Instalar de diretório local (sem git push)
hermes profile install ~/.hermes/profiles/research-bot --name research-bot-test --alias

# Ajuste, delete, reinstale até ficar certo
hermes profile delete research-bot-test --yes
hermes profile install ~/.hermes/profiles/research-bot --name research-bot-test
```

---

## Exportar e importar um arquivo de profile {#export-and-import-a-profile-file}

Quando você não precisa de versionamento, pule o repo. `/export` empacota um profile em um único `.tar.gz`; `/import` desempacota como um profile novo no outro lado. Credenciais são removidas na saída.

### Export {#export}

Na CLI, TUI ou chat do desktop:

```
/export                          # the active profile → <name>.tar.gz
/export research-bot             # a named profile
/export research-bot -o ~/Desktop/research-bot.tar.gz
```

Ou de um shell, mesma maquinaria:

```bash
hermes profile export research-bot
hermes profile export research-bot -o ./research-bot.tar.gz
```

No **app desktop** há três portas, todas desembocando em um save dialog nativo:

- **⌘K → Export profile…**
- Clique direito num quadrado de profile na rail da sidebar → **Export profile…**
- O botão de import ao lado do **+** da rail cobre a outra direção

Um export do desktop adiciona um arquivo extra que a CLI não tem: `desktop.json`, carregando sua skin, modo light/dark, quaisquer definições de tema custom que a skin precise, a cor da rail do profile e seu layout de janela. Por isso um profile compartilhado pelo desktop chega *parecendo* o seu, não só se comportando como o seu.

### Import {#import}

```
/import ~/Downloads/research-bot.tar.gz
/import ~/Downloads/research-bot.tar.gz --name research-bot-2
```

```bash
hermes profile import ./research-bot.tar.gz
hermes profile import ./research-bot.tar.gz --name research-bot-2
```

O nome do profile é inferido do archive a menos que você passe `--name`. Importar sobre um profile existente é recusado — renomeie ou delete o antigo primeiro. Um shell wrapper (`research-bot` → `hermes -p research-bot`) é criado quando o nome não colide com um comando existente.

Importar no app desktop também aplica o overlay `desktop.json` e te coloca no profile novo em um chat fresh. Importar um archive feito no desktop pela CLI é ok — o arquivo overlay fica no disco e aplica na próxima vez que você abrir aquele profile no desktop.

:::note
Você não pode importar como `default` — esse nome é o profile root embutido (`~/.hermes`). Passe `--name something-else`.
:::

### O que um arquivo de export contém {#what-an-export-file-contains}

Sempre excluídos, ambos tipos de profile: `auth.json` e `.env`. Suas API keys nunca saem da máquina.

**O profile default** (`~/.hermes`) é exportado via allow-list — só artefatos conhecidos do Hermes, então um arquivo não relacionado sentado no seu home directory não é varrido:

`config.yaml`, `SOUL.md`, `MEMORY.md`, `USER.md`, `todo.json`, `system_prompt.md`, `AGENTS.md`, `CLAUDE.md`, `.cursorrules`, `skills/`, `plugins/`, `cron/`, `scripts/`, `sessions/`, `memories/`, `knowledge/`, `preferences/`, e `desktop.json` quando o desktop staged um.

**Um profile nomeado** (`~/.hermes/profiles/<name>`) copia o diretório inteiro menos `auth.json` / `.env`. Isso é mais amplo — se o profile tem `state.db`, logs ou caches, eles entram no archive também, e o arquivo fica grande.

:::caution Leia seu archive antes de enviar
Um export é um snapshot do seu profile, não um release curado. Diferente de uma distribuição, **pode** incluir `memories/`, `sessions/` e `USER.md` — e nada faz scan de skills, memórias ou da sua persona por qualquer coisa pessoal que você tenha escrito nelas. Credenciais são filtradas por nome de arquivo; conteúdo não.

Antes de compartilhar com outra pessoa, liste o que está dentro:

```bash
tar -tzf research-bot.tar.gz | less
```

Se carrega histórico de conversa que você preferiria não entregar, publique uma [distribuição](#for-authors-publishing-a-distribution) em vez disso — essas nunca enviam memórias ou sessões.
:::

## O que NUNCA está em uma distribuição {#whats-not-in-a-distribution-ever}

O instalador hard-exclui estes caminhos mesmo se um autor enviar por acidente. Nenhuma opção de config permite sobrescrever — a guarda de segurança é invariante testada por regressão:

- `auth.json` — tokens OAuth, credenciais de plataforma
- `.env` — API keys, secrets
- `memories/` — memória de conversa
- `sessions/` — histórico de conversa
- `state.db`, `state.db-shm`, `state.db-wal` — metadados de sessão
- `logs/` — logs do agente e de erro
- `workspace/` — arquivos de trabalho gerados
- `plans/` — planos scratch
- `home/` — mount home do usuário em backends Docker
- `*_cache/` — caches de imagem / áudio / documento
- `local/` — namespace de customização reservado ao usuário

Quando você clona uma distribuição como instalador, estes simplesmente não são copiados para seu diretório de profile. Quando atualiza, suas cópias permanecem. Se instalou a mesma distribuição em cinco máquinas, você tem cinco conjuntos isolados desses dados — um por máquina.

:::caution
Esta exclusão roda em **install / update time na máquina do instalador**. **Não** impede um autor de commitar arquivos sensíveis/desnecessários. Autores devem usar [`.gitignore`](#passo-3--crie-um-gitignore-antes-do-primeiro-commit) para manter secrets fora do repositório.
:::

## Segurança e confiança

Distribuições de profile são unsigned por padrão. Você confia em:

- **O host git** (GitHub / GitLab / onde for) para servir os bytes que o autor publicou.
- **O autor** para não enviar SOUL, skills ou cron jobs maliciosos.

Jobs cron de uma distribuição **não são auto-agendados** — o instalador imprime `hermes -p <name> cron list` e você os habilita explicitamente. SOUL.md e skills ESTÃO ativos assim que você começa a conversar com o profile, então leia antes da primeira execução se instala de alguém que não conhece.

Analogia grosseira: instalar uma distribuição é como instalar extensão de browser ou extensão VS Code. Baixa fricção, alto poder, confie na fonte. Para distribuições internas de empresa, use repositório privado e sua auth git normal — nada novo para configurar.

Versões futuras podem adicionar assinatura, lockfile (`.distribution-lock.yaml`) com commit SHA resolvido e flag `--dry-run` que imprime o diff antes de aplicar update. Nenhum disso está shipping ainda.

## Por baixo dos panos

Para detalhes de implementação, comportamento preciso da CLI e todas as flags, veja a [referência Profile Commands](../reference/profile-commands.md#distribution-commands).

A versão curta:

- `install`, `update`, `info` ficam dentro de `hermes profile` — não uma árvore de comandos paralela.
- O formato do manifest é YAML com schema mínimo obrigatório (`name` apenas).
- O instalador usa seu binary `git` local para clonar, então qualquer auth que seu shell já trata (chaves SSH, credential helpers) funciona transparentemente.
- Após clone, `.git/` é removido — o profile instalado não é checkout git, evitando armadilhas de "ops, commitei meu `.env` no histórico git da distribuição".
- Nomes de profile reservados (`hermes`, `test`, `tmp`, `root`, `sudo`) são rejeitados na instalação para evitar colisões com binários comuns.

## Veja também

- [Profiles: Running Multiple Agents](./profiles.md) — o conceito base
- [Profile Commands reference](../reference/profile-commands.md) — toda flag, toda opção
- [`hermes profile export` / `import`](../reference/profile-commands.md#hermes-profile-export) — a forma CLI de [arquivos de export](#export-and-import-a-profile-file)
- [Slash Commands reference](../reference/slash-commands.md) — `/export`, `/import` e todo outro comando in-chat
- [Using SOUL with Hermes](../guides/use-soul-with-hermes.md) — autoria de personalidades
- [Personality & SOUL](./features/personality.md) — como SOUL se encaixa no agente
- [Skills catalog](../reference/skills-catalog.md) — skills que você pode embutir
