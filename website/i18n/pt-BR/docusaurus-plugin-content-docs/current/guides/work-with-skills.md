---
sidebar_position: 12
title: "Trabalhando com Skills"
description: "Encontre, instale, use e crie skills — conhecimento sob demanda que ensina novos fluxos de trabalho ao Hermes"
---

# Trabalhando com Skills {#working-with-skills}

Skills são documentos de conhecimento sob demanda que ensinam ao Hermes como lidar com tarefas específicas — desde gerar arte ASCII até gerenciar PRs do GitHub. Este guia mostra como usá-las no dia a dia.

Para a referência técnica completa, veja [Sistema de Skills](/user-guide/features/skills).

---

## Encontrando Skills {#finding-skills}

Toda instalação do Hermes vem com skills empacotadas. Veja o que está disponível:

```bash
# In any chat session:
/skills

# Or from the CLI:
hermes skills list
```

Isso mostra uma lista compacta com nomes e descrições:

```
ascii-art         Generate ASCII art using pyfiglet, cowsay, boxes...
arxiv             Search and retrieve academic papers from arXiv...
github-pr-workflow Full PR lifecycle — create branches, commit...
plan              Plan mode — inspect context, write a markdown...
excalidraw        Create hand-drawn style diagrams using Excalidraw...
```

### Buscando uma Skill {#searching-for-a-skill}

```bash
# Search by keyword
/skills search docker
/skills search music
```

### O Hub de Skills {#the-skills-hub}

Skills opcionais oficiais (skills mais pesadas ou de nicho, não ativas por padrão) estão disponíveis através do Hub:

```bash
# Browse official optional skills
/skills browse

# Search the hub
/skills search blockchain
```

---

## Usando uma Skill {#using-a-skill}

Toda skill instalada é automaticamente um comando de barra. Basta digitar o nome dela:

```bash
# Load a skill and give it a task
/ascii-art Make a banner that says "HELLO WORLD"
/plan Design a REST API for a todo app
/github-pr-workflow Create a PR for the auth refactor

# Just the skill name (no task) loads it and lets you describe what you need
/excalidraw
```

Você também pode acionar skills através de conversação natural — pergunte ao Hermes para usar uma skill específica, e ele a carregará via a ferramenta `skill_view`.

### Divulgação Progressiva {#progressive-disclosure}

Skills usam um padrão de carregamento eficiente em tokens. O agente não carrega tudo de uma vez:

1. **`skills_list()`** — lista compacta de todas as skills (~3 mil tokens). Carregada no início da sessão.
2. **`skill_view(name)`** — conteúdo completo do SKILL.md para uma skill. Carregado quando o agente decide que precisa dessa skill.
3. **`skill_view(name, file_path)`** — um arquivo de referência específico dentro da skill. Carregado apenas se necessário.

Isso significa que as skills não custam tokens até que sejam realmente usadas.

---

## Instalando a Partir do Hub {#installing-from-the-hub}

Skills opcionais oficiais vêm com o Hermes, mas não estão ativas por padrão. Instale-as explicitamente:

```bash
# Install an official optional skill
hermes skills install official/research/arxiv

# Install from the hub in a chat session
/skills install official/creative/songwriting-and-ai-music

# Install SKILL.md and its referenced support files from an HTTP(S) URL
hermes skills install https://sharethis.chat/SKILL.md
/skills install https://example.com/SKILL.md --name my-skill
```

O que acontece:
1. O diretório da skill é copiado para `~/.hermes/skills/`
2. Ela aparece na sua saída de `skills_list`
3. Ela se torna disponível como um comando de barra

:::tip
Skills instaladas entram em vigor em novas sessões. Se você quiser que ela esteja disponível na sessão atual, use `/reset` para começar do zero, ou adicione `--now` para invalidar o cache do prompt imediatamente (custa mais tokens na próxima resposta).
:::

### Verificando a Instalação {#verifying-installation}

```bash
# Check it's there
hermes skills list | grep arxiv

# Or in chat
/skills search arxiv
```

---

## Skills Fornecidas por Plugins {#plugin-provided-skills}

Plugins podem empacotar suas próprias skills usando nomes com namespace (`plugin:skill`). Isso evita colisões de nomes com skills integradas.

```bash
# Load a plugin skill by its qualified name
skill_view("superpowers:writing-plans")

# Built-in skill with the same base name is unaffected
skill_view("writing-plans")
```

Skills de plugins **não** são listadas no prompt do sistema e não aparecem em `skills_list`. Elas são opt-in — carregue-as explicitamente quando você souber que um plugin fornece uma. Quando carregada, o agente vê um banner listando as skills irmãs do mesmo plugin.

Para saber como enviar skills no seu próprio plugin, veja [Construir um Plugin do Hermes → Empacotar skills](/developer-guide/plugins#bundle-skills).

---

## Configurando Ajustes de Skills {#configuring-skill-settings}

Algumas skills declaram configurações que precisam em seu frontmatter:

```yaml
metadata:
  hermes:
    config:
      - key: tenor.api_key
        description: "Tenor API key for GIF search"
        prompt: "Enter your Tenor API key"
        url: "https://developers.google.com/tenor/guides/quickstart"
```

Quando uma skill com configuração é carregada por primeira vez, o Hermes solicita os valores a você. Eles são armazenados em `config.yaml` sob `skills.config.*`.

Gerencie a configuração de skills a partir da CLI:

```bash
# Interactive config for a specific skill
hermes skills config gif-search

# View all skill config
hermes config get skills.config --json
```

---

## Criando Sua Própria Skill {#creating-your-own-skill}

Skills são apenas arquivos markdown com frontmatter YAML. Criar uma leva menos de cinco minutos.

### 1. Crie o Diretório {#1-create-the-directory}

```bash
mkdir -p ~/.hermes/skills/my-category/my-skill
```

### 2. Escreva o SKILL.md {#2-write-skillmd}

```markdown title="~/.hermes/skills/my-category/my-skill/SKILL.md"
---
name: my-skill
description: Brief description of what this skill does
version: 1.0.0
metadata:
  hermes:
    tags: [my-tag, automation]
    category: my-category
---

# My Skill

## When to Use
Use this skill when the user asks about [specific topic] or needs to [specific task].

## Procedure
1. First, check if [prerequisite] is available
2. Run `command --with-flags`
3. Parse the output and present results

## Pitfalls
- Common failure: [description]. Fix: [solution]
- Watch out for [edge case]

## Verification
Run `check-command` to confirm the result is correct.
```

### 3. Adicione Arquivos de Referência (Opcional) {#3-add-reference-files-optional}

Skills podem incluir arquivos de suporte que o agente carrega sob demanda:

```
my-skill/
├── SKILL.md                    # Main skill document
├── references/
│   ├── api-docs.md             # API reference the agent can consult
│   └── examples.md             # Example inputs/outputs
├── templates/
│   └── config.yaml             # Template files the agent can use
└── scripts/
    └── setup.sh                # Scripts the agent can execute
```

Referencie-os no seu SKILL.md:

```markdown
For API details, load the reference: `skill_view("my-skill", "references/api-docs.md")`
```

### 4. Teste-a {#4-test-it}

Inicie uma nova sessão e teste sua skill:

```bash
hermes chat -q "/my-skill help me with the thing"
```

A skill aparece automaticamente — sem necessidade de registro. Coloque-a em `~/.hermes/skills/` e ela já está ativa.

:::info
O agente também pode criar e atualizar skills por conta própria usando `skill_manage`. Depois de resolver um problema complexo, o Hermes pode se oferecer para salvar a abordagem como uma skill para a próxima vez.
:::

---

## Gerenciamento de Skills por Plataforma {#per-platform-skill-management}

Controle quais skills estão disponíveis em quais plataformas:

```bash
hermes skills
```

Isso abre uma TUI interativa onde você pode habilitar ou desabilitar skills por plataforma (CLI, Telegram, Discord, etc.). Útil quando você quer que certas skills fiquem disponíveis apenas em contextos específicos — por exemplo, mantendo skills de desenvolvimento fora do Telegram.

---

## Skills vs Memória {#skills-vs-memory}

Ambas são persistentes entre sessões, mas servem a propósitos diferentes:

| | Skills | Memória |
|---|---|---|
| **O quê** | Conhecimento procedural — como fazer as coisas | Conhecimento factual — o que as coisas são |
| **Quando** | Carregada sob demanda, apenas quando relevante | Injetada automaticamente em toda sessão |
| **Tamanho** | Pode ser grande (centenas de linhas) | Deve ser compacta (apenas fatos-chave) |
| **Custo** | Zero tokens até ser carregada | Custo pequeno, mas constante, de tokens |
| **Exemplos** | "Como fazer deploy no Kubernetes" | "Usuário prefere modo escuro, vive no PST" |
| **Quem cria** | Você, o agente, ou instalada a partir do Hub | O agente, com base nas conversas |

**Regra geral:** Se você colocaria em um documento de referência, é uma skill. Se você colocaria em um post-it, é memória.

---

## Dicas {#tips}

**Mantenha as skills focadas.** Uma skill que tenta cobrir "todo o DevOps" será longa demais e vaga demais. Uma skill que cobre "fazer deploy de um app Python no Fly.io" é específica o suficiente para ser genuinamente útil.

**Deixe o agente criar skills.** Depois de uma tarefa complexa de múltiplas etapas, o Hermes frequentemente se oferecerá para salvar a abordagem como uma skill. Diga sim — essas skills escritas pelo agente capturam o fluxo de trabalho exato, incluindo armadilhas descobertas ao longo do caminho.

**Use categorias.** Organize skills em subdiretórios (`~/.hermes/skills/devops/`, `~/.hermes/skills/research/`, etc.). Isso mantém a lista gerenciável e ajuda o agente a encontrar skills relevantes mais rapidamente.

**Atualize skills quando ficarem desatualizadas.** Se você usa uma skill e encontra problemas não cobertos por ela, diga ao Hermes para atualizá-la com o que você aprendeu. Skills que não são mantidas se tornam passivos.

---

*Para a referência completa de skills — campos de frontmatter, ativação condicional, diretórios externos e mais — veja [Sistema de Skills](/user-guide/features/skills).*
