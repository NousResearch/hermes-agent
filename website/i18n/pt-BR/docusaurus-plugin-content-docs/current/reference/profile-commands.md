---
sidebar_position: 7
---

# Referência de Comandos de Perfil

Esta página cobre todos os comandos relacionados a [perfis do Hermes](../user-guide/profiles.md). Para comandos gerais da CLI, veja a [Referência de Comandos da CLI](./cli-commands.md).

## `hermes profile` {#hermes-profile}

```bash
hermes profile <subcommand>
```

Comando de nível superior para gerenciar perfis. Executar `hermes profile` sem um subcomando mostra a ajuda.

| Subcomando | Descrição |
|------------|-------------|
| `list` | Lista todos os perfis. |
| `use` | Define o perfil ativo (padrão). |
| `create` | Cria um novo perfil. |
| `describe` | Lê ou define a descrição de um perfil (usada pelo orquestrador do kanban para roteamento). |
| `delete` | Exclui um perfil. |
| `show` | Mostra detalhes sobre um perfil. |
| `alias` | Regenera o alias de shell de um perfil. |
| `rename` | Renomeia um perfil. |
| `export` | Exporta um perfil para um arquivo tar.gz. |
| `import` | Importa um perfil a partir de um arquivo tar.gz. |
| `install` | Instala uma distribuição de perfil a partir de uma URL git ou diretório local. Veja [Distribuições de Perfil](../user-guide/profile-distributions.md). |
| `update` | Refaz o pull de um perfil gerenciado por distribuição e reaplica seu bundle. |
| `info` | Mostra metadados de distribuição de um perfil (URL de origem, commit, última atualização). |

## `hermes profile list` {#hermes-profile-list}

```bash
hermes profile list
```

Lista todos os perfis. O perfil ativo atualmente é marcado com `*`.

**Exemplo:**

```bash
$ hermes profile list
  default
* work
  dev
  personal
```

Sem opções.

## `hermes profile use` {#hermes-profile-use}

```bash
hermes profile use <name>
```

Define `<name>` como o perfil ativo. Todos os comandos `hermes` subsequentes (sem `-p`) usarão este perfil.

| Argumento | Descrição |
|----------|-------------|
| `<name>` | Nome do perfil a ativar. Use `default` para voltar ao perfil base. |

**Exemplo:**

```bash
hermes profile use work
hermes profile use default
```

## `hermes profile create` {#hermes-profile-create}

```bash
hermes profile create <name> [options]
```

Cria um novo perfil.

| Argumento / Opção | Descrição |
|-------------------|-------------|
| `<name>` | Nome do novo perfil. Deve ser um nome de diretório válido (alfanumérico, hífens, underscores). |
| `--clone` | Copia `config.yaml`, `.env`, `SOUL.md` e skills do perfil atual. |
| `--clone-all` | Copia tudo (config, memórias, skills, cron, plugins) do perfil atual. Exclui histórico por perfil: sessões, `state.db`, backups, state-snapshots, checkpoints. |
| `--clone-from <profile>` | Clona config/skills/SOUL de um perfil específico em vez do atual. Implica `--clone`, exceto se combinado com `--clone-all`. |
| `--no-alias` | Pula a criação do script wrapper. |
| `--description "<text>"` | Descrição de uma ou duas frases sobre para que este perfil é bom. Usada pelo orquestrador do kanban para rotear tarefas com base na função em vez de apenas o nome do perfil. Pule e adicione depois via `hermes profile describe`. Persistida em `<profile_dir>/profile.yaml`. |
| `--no-skills` | Cria um perfil **vazio** com zero skills incluídas ativadas. Escreve um marcador `.no-bundled-skills` no perfil para que futuras execuções de `hermes update` não reintroduzam o conjunto incluído, e se recusa a combinar com `--clone`, `--clone-from` ou `--clone-all` (que copiariam skills de qualquer forma). Útil para perfis orquestradores restritos ou perfis de sandbox que não devem herdar o catálogo completo de skills. Para alternar isso em um perfil já criado (incluindo o padrão `~/.hermes`), use `hermes skills opt-out` / `hermes skills opt-in`. |

Criar um perfil **não** torna o diretório desse perfil o diretório de projeto/workspace padrão para comandos de terminal. Se você quiser que um perfil inicie em um projeto específico, defina `terminal.cwd` no `config.yaml` desse perfil.

**Exemplos:**

```bash
# Perfil vazio — precisa de configuração completa
hermes profile create mybot

# Clona apenas a config do perfil atual
hermes profile create work --clone

# Clona tudo do perfil atual
hermes profile create backup --clone-all

# Clona a config de um perfil específico
hermes profile create work2 --clone-from work

# Clona tudo de um perfil específico
hermes profile create work2-backup --clone-from work --clone-all
```

## `hermes profile describe` {#hermes-profile-describe}

```bash
hermes profile describe [<name>] [options]
```

Lê ou define a descrição de um perfil. A descrição é consumida pelo orquestrador do kanban para rotear tarefas com base no que cada perfil faz bem, em vez de adivinhar apenas pelo nome do perfil. Persistida em `<profile_dir>/profile.yaml` para sobreviver a reinicializações e ser compartilhada com o gateway.

Sem flags, imprime a descrição atual (ou `(no description set for '<name>')` se vazia).

| Argumento / Opção | Descrição |
|-------------------|-------------|
| `<name>` | Perfil a descrever. Obrigatório, exceto quando `--all --auto` é usado. |
| `--text "<text>"` | Define a descrição como este texto exato (escrito pelo usuário). Sobrescreve qualquer descrição existente. |
| `--auto` | Gera automaticamente uma descrição de 1-2 frases via LLM auxiliar, com base nas skills instaladas do perfil, modelo configurado e nome. Configure o modelo em `auxiliary.profile_describer` no `config.yaml`. Descrições geradas automaticamente são marcadas com `description_auto: true` para que o dashboard possa sinalizá-las para revisão. |
| `--overwrite` | Com `--auto`, substitui também descrições escritas pelo usuário (padrão: pula perfis cuja descrição foi definida explicitamente). |
| `--all` | Com `--auto`, varre todo perfil sem descrição. |

**Exemplos:**

```bash
# Lê a descrição atual
hermes profile describe researcher

# Define explicitamente
hermes profile describe researcher --text "Reads source code and writes findings."

# Deixa o LLM gerar uma
hermes profile describe researcher --auto

# Preenche descrições para todo perfil que não tem uma
hermes profile describe --all --auto
```

## `hermes profile delete` {#hermes-profile-delete}

```bash
hermes profile delete <name> [options]
```

Exclui um perfil e remove seu alias de shell.

| Argumento / Opção | Descrição |
|-------------------|-------------|
| `<name>` | Perfil a excluir. |
| `--yes`, `-y` | Pula o prompt de confirmação. |

**Exemplo:**

```bash
hermes profile delete mybot
hermes profile delete mybot --yes
```

:::warning
Isso exclui permanentemente todo o diretório do perfil, incluindo toda a config, memórias, sessões e skills. Não é possível excluir o perfil atualmente ativo.
:::

## `hermes profile show` {#hermes-profile-show}

```bash
hermes profile show <name>
```

Exibe detalhes sobre um perfil, incluindo seu diretório home, modelo configurado, status do gateway, contagem de skills e status do arquivo de configuração.

Isso mostra o diretório home do Hermes do perfil, não o diretório de trabalho do terminal. Comandos de terminal iniciam a partir de `terminal.cwd` (ou o diretório de lançamento no backend local quando `cwd: "."`).

| Argumento | Descrição |
|----------|-------------|
| `<name>` | Perfil a inspecionar. |

**Exemplo:**

```bash
$ hermes profile show work
Profile: work
Path:    ~/.hermes/profiles/work
Model:   anthropic/claude-sonnet-4 (anthropic)
Gateway: stopped
Skills:  12
.env:    exists
SOUL.md: exists
Alias:   ~/.local/bin/work
```

## `hermes profile alias` {#hermes-profile-alias}

```bash
hermes profile alias <name> [options]
```

Regenera o script de alias de shell em `~/.local/bin/<name>`. Útil se o alias foi excluído acidentalmente ou se você precisa atualizá-lo após mover sua instalação do Hermes.

| Argumento / Opção | Descrição |
|-------------------|-------------|
| `<name>` | Perfil para o qual criar/atualizar o alias. |
| `--remove` | Remove o script wrapper em vez de criá-lo. |
| `--name <alias>` | Nome de alias customizado (padrão: nome do perfil). |

**Exemplo:**

```bash
hermes profile alias work
# Cria/atualiza ~/.local/bin/work

hermes profile alias work --name mywork
# Cria ~/.local/bin/mywork

hermes profile alias work --remove
# Remove o script wrapper
```

## `hermes profile rename` {#hermes-profile-rename}

```bash
hermes profile rename <old-name> <new-name>
```

Renomeia um perfil. Atualiza o diretório e o alias de shell.

| Argumento | Descrição |
|----------|-------------|
| `<old-name>` | Nome atual do perfil. |
| `<new-name>` | Novo nome do perfil. |

**Exemplo:**

```bash
hermes profile rename mybot assistant
# ~/.hermes/profiles/mybot → ~/.hermes/profiles/assistant
# ~/.local/bin/mybot → ~/.local/bin/assistant
```

## `hermes profile export` {#hermes-profile-export}

```bash
hermes profile export <name> [options]
```

Exporta um perfil como um arquivo tar.gz compactado.

| Argumento / Opção | Descrição |
|-------------------|-------------|
| `<name>` | Perfil a exportar. |
| `-o`, `--output <path>` | Caminho do arquivo de saída (padrão: `<name>.tar.gz`). |

**Exemplo:**

```bash
hermes profile export work
# Cria work.tar.gz no diretório atual

hermes profile export work -o ./work-2026-03-29.tar.gz
```

## `hermes profile import` {#hermes-profile-import}

```bash
hermes profile import <archive> [options]
```

Importa um perfil a partir de um arquivo tar.gz.

| Argumento / Opção | Descrição |
|-------------------|-------------|
| `<archive>` | Caminho para o arquivo tar.gz a importar. |
| `--name <name>` | Nome para o perfil importado (padrão: inferido do arquivo). |

**Exemplo:**

```bash
hermes profile import ./work-2026-03-29.tar.gz
# Infere o nome do perfil a partir do arquivo

hermes profile import ./work-2026-03-29.tar.gz --name work-restored
```

## Comandos de distribuição {#distribution-commands}

:::tip
**Novo em distribuições?** Comece com o [guia do usuário de Distribuições de Perfil](../user-guide/profile-distributions.md) — ele cobre o porquê, o quando e o como com exemplos completos. As seções abaixo são uma referência seca de CLI para quando você já sabe o que quer.
:::

Distribuições transformam um perfil em um artefato compartilhável e versionado, publicado como um **repositório git**. Um destinatário instala a distribuição com um único comando e pode atualizá-la posteriormente sem tocar em suas memórias, sessões ou credenciais locais.

`auth.json` e `.env` nunca fazem parte de uma distribuição — eles permanecem na máquina do usuário que instala.

Os dados do usuário destinatário (memórias, sessões, auth, suas próprias edições em `.env`) são sempre preservados durante a instalação inicial e atualizações subsequentes.

:::info
`hermes profile export` / `import` continuam sendo os comandos certos para **backup e restauração local** de um perfil na sua própria máquina. Distribuição (`install` / `update` / `info`) é um conceito separado: enviar um perfil via git para que outra pessoa possa instalá-lo.
:::

### `hermes profile install` {#hermes-profile-install}

```bash
hermes profile install <source> [--name <name>] [--alias] [--force] [--yes]
```

Instala uma distribuição de perfil a partir de uma URL git ou de um diretório local.

| Opção | Descrição |
|--------|-------------|
| `<source>` | URL git (`github.com/user/repo`, `https://...`, `git@...`, `ssh://`, `git://`) ou um diretório local contendo `distribution.yaml` na raiz. |
| `--name NAME` | Sobrescreve o nome do perfil a partir do manifesto. |
| `--alias` | Também cria um wrapper de shell (ex.: `telemetry` → `hermes -p telemetry`). |
| `--force` | Sobrescreve um perfil existente com o mesmo nome. Os dados do usuário ainda são preservados. |
| `-y`, `--yes` | Pula o prompt de confirmação da pré-visualização do manifesto. |

O instalador mostra o manifesto, lista as variáveis de ambiente necessárias e avisa sobre jobs de cron antes de pedir confirmação. As variáveis de ambiente necessárias vão para um arquivo `.env.EXAMPLE` que você copia para `.env` e preenche.

**Exemplos:**

```bash
# Instala a partir de um repositório GitHub (forma abreviada)
hermes profile install github.com/kyle/telemetry-distribution --alias

# Instala a partir de uma URL git HTTPS completa
hermes profile install https://github.com/kyle/telemetry-distribution.git

# Instala via SSH
hermes profile install git@github.com:kyle/telemetry-distribution.git

# Instala a partir de um diretório local durante o desenvolvimento
hermes profile install ./telemetry/
```

### `hermes profile update` {#hermes-profile-update}

```bash
hermes profile update <name> [--force-config] [--yes]
```

Reclona a distribuição a partir de sua fonte registrada e aplica atualizações.
Arquivos pertencentes à distribuição (SOUL.md, skills/, cron/, mcp.json) são
sobrescritos; dados do usuário (memórias, sessões, auth, .env) nunca são tocados.

`config.yaml` é preservado por padrão para manter suas sobreposições locais.
Passe `--force-config` para redefini-lo para a config enviada pela distribuição.

### `hermes profile info` {#hermes-profile-info}

```bash
hermes profile info <name>
```

Imprime o manifesto de distribuição do perfil — nome, versão, versão do Hermes
necessária, autor, requisitos de variáveis de ambiente, a URL/caminho de origem, e
o timestamp `Installed:` registrado da última vez que a distribuição foi
`install`-ada ou `update`-ada. Útil para verificar o que um perfil compartilhado
precisa antes de instalá-lo, e para identificar "este perfil foi instalado
há 6 meses e não foi atualizado."

`hermes profile list` também mostra o nome e a versão da distribuição em uma
coluna `Distribution`, e `hermes profile show <name>` / `delete <name>`
mostram a URL de origem para que você possa distinguir rapidamente quais perfis
vieram de um repositório git vs. foram criados localmente.

### Distribuições privadas {#private-distributions}

Um repositório git privado funciona como fonte de distribuição sem
configuração extra — a instalação delega ao seu binário `git` normal, então
qualquer autenticação que seu shell já esteja configurado para usar (chave SSH,
o helper `git credential`, credenciais HTTPS armazenadas do GitHub CLI) se aplica
transparentemente.

```bash
# Usa sua chave SSH, assim como qualquer outro `git clone`
hermes profile install git@github.com:your-org/internal-assistant.git

# Usa seu helper de credenciais git
hermes profile install https://github.com/your-org/internal-assistant.git
```

Se um clone pedir credenciais interativamente no seu terminal durante a
instalação, esse prompt passa normalmente. Configure sua autenticação da forma
que você normalmente usaria com `git clone` contra o mesmo repositório primeiro, e então instale.

### Manifesto de distribuição (`distribution.yaml`) {#distribution-manifest-distributionyaml}

Toda distribuição tem um `distribution.yaml` na raiz do seu repositório:

```yaml
name: telemetry
version: 0.1.0
description: "Compliance monitoring harness"
hermes_requires: ">=0.12.0"
author: "Your Name"
license: "MIT"
env_requires:
  - name: OPENAI_API_KEY
    description: "OpenAI API key"
    required: true
  - name: GRAPHITI_MCP_URL
    description: "Memory graph URL"
    required: false
    default: "http://127.0.0.1:8000/sse"
distribution_owned:   # opcional; padrão é SOUL.md, config.yaml,
                      #   mcp.json, skills/, cron/, distribution.yaml
  - SOUL.md
  - skills/compliance/
  - cron/
```

`hermes_requires` suporta `>=`, `<=`, `==`, `!=`, `>`, `<`, ou uma
versão simples (tratada como `>=`). A instalação falha com um erro claro se a versão
atual do Hermes não satisfizer a especificação.

`distribution_owned` é opcional. Se definido, apenas esses caminhos são substituídos na
atualização; qualquer outra coisa no perfil permanece de propriedade do usuário. Se omitido, os
padrões acima se aplicam.

### Publicando uma distribuição {#publishing-a-distribution}

Criar uma distribuição é apenas um git push:

1. No diretório do seu perfil, crie `distribution.yaml` com pelo menos `name`
   e `version`.
2. Inicialize um repositório git (ou use um existente) e faça push para GitHub /
   GitLab / qualquer host de onde o Hermes possa clonar.
3. Diga aos destinatários para executar `hermes profile install <your-repo-url>`.

Use tags git para releases versionados — destinatários que clonam a `HEAD` recebem seu
estado mais recente, e você sempre pode incrementar `version:` no manifesto.

## `hermes -p` / `hermes --profile` {#hermes--p--hermes---profile}

```bash
hermes -p <name> <command> [options]
hermes --profile <name> <command> [options]
```

Flag global para executar qualquer comando do Hermes sob um perfil específico sem alterar o padrão fixo. Isso sobrescreve o perfil ativo durante a execução do comando.

| Opção | Descrição |
|--------|-------------|
| `-p <name>`, `--profile <name>` | Perfil a usar para este comando. |

**Exemplos:**

```bash
hermes -p work chat -q "Check the server status"
hermes --profile dev gateway start
hermes -p personal skills list
hermes -p work config edit
```

## `hermes completion` {#hermes-completion}

```bash
hermes completion <shell>
```

Gera scripts de autocompletar de shell. Inclui autocompletar para nomes de perfil e subcomandos de perfil.

| Argumento | Descrição |
|----------|-------------|
| `<shell>` | Shell para o qual gerar o autocompletar: `bash`, `zsh` ou `fish`. |

**Exemplos:**

```bash
# Instala o autocompletar
hermes completion bash >> ~/.bashrc
hermes completion zsh >> ~/.zshrc
hermes completion fish > ~/.config/fish/completions/hermes.fish

# Recarrega o shell
source ~/.bashrc
```

Após a instalação, o autocompletar por tab funciona para:
- `hermes profile <TAB>` — subcomandos (list, use, create, etc.)
- `hermes profile use <TAB>` — nomes de perfil
- `hermes -p <TAB>` — nomes de perfil

## Veja também {#see-also}

- [Guia do Usuário de Perfis](../user-guide/profiles.md)
- [Referência de Comandos da CLI](./cli-commands.md)
- [FAQ — Seção de Perfis](./faq.md#profiles)
