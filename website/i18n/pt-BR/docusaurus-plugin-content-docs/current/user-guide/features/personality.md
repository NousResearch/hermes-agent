---
sidebar_position: 9
title: "Personalidade e SOUL.md"
description: "Personalize a personalidade do Hermes Agent com SOUL.md global, personalidades built-in e definições de persona customizadas"
---

# Personalidade e SOUL.md {#personality}

A personalidade do Hermes Agent é totalmente customizável. `SOUL.md` é a **identidade principal** — é a primeira coisa no system prompt e define quem o agente é.

- `SOUL.md` — arquivo de persona durável que vive em `HERMES_HOME` e serve como identidade do agente (slot #1 no system prompt)
- presets built-in ou customizados de `/personality` — overlays de system prompt no nível da sessão

Se quiser mudar quem o Hermes é — ou substituí-lo por uma persona de agente totalmente diferente — edite `SOUL.md`.

## Como o SOUL.md funciona agora {#how-soulmd-works-now}

O Hermes agora cria automaticamente um `SOUL.md` padrão em:

```text
~/.hermes/SOUL.md
```

Mais precisamente, usa o `HERMES_HOME` da instância atual, então se você rodar o Hermes com um diretório home customizado, usará:

```text
$HERMES_HOME/SOUL.md
```

### Comportamento importante {#important-behavior}

- **SOUL.md é a identidade principal do agente.** Ocupa o slot #1 no system prompt, substituindo a identidade padrão hardcoded.
- O Hermes cria um `SOUL.md` inicial automaticamente se ainda não existir
- Arquivos `SOUL.md` existentes do usuário nunca são sobrescritos
- O Hermes carrega `SOUL.md` somente de `HERMES_HOME`
- O Hermes não procura `SOUL.md` no diretório de trabalho atual
- Se `SOUL.md` existir mas estiver vazio, ou não puder ser carregado, o Hermes usa identidade padrão built-in
- Se `SOUL.md` tiver conteúdo, esse conteúdo é injetado verbatim após varredura de segurança e truncamento
- SOUL.md **não** é duplicado na seção de context files — aparece só uma vez, como identidade

Isso torna `SOUL.md` uma identidade verdadeira por usuário ou por instância, não só uma camada aditiva.

## Por que esse design {#why-this-design}

Isso mantém a personalidade previsível.

Se o Hermes carregasse `SOUL.md` do diretório em que você iniciou, sua personalidade poderia mudar inesperadamente entre projetos. Carregando só de `HERMES_HOME`, a personalidade pertence à própria instância Hermes.

Também facilita ensinar aos usuários:
- "Edite `~/.hermes/SOUL.md` para mudar a personalidade padrão do Hermes."

## Onde editar {#where-to-edit-it}

Para a maioria dos usuários:

```bash
~/.hermes/SOUL.md
```

Se você usa home customizado:

```bash
$HERMES_HOME/SOUL.md
```

## O que colocar no SOUL.md? {#what-should-go-in-soulmd}

Use para orientação durável de voz e personalidade, como:
- tom
- estilo de comunicação
- nível de franqueza
- estilo de interação padrão
- o que evitar estilisticamente
- como o Hermes deve lidar com incerteza, discordância ou ambiguidade

Use menos para:
- instruções pontuais de projeto
- caminhos de arquivo
- convenções de repo
- detalhes temporários de workflow

Isso pertence a `AGENTS.md`, não a `SOUL.md`.

## Conteúdo bom para SOUL.md {#good-soulmd-content}

Um SOUL file bom é:
- estável entre contextos
- amplo o suficiente para várias conversas
- específico o suficiente para moldar materialmente a voz
- focado em comunicação e identidade, não instruções específicas de tarefa

### Exemplo {#example}

```markdown
# Personality

You are a pragmatic senior engineer with strong taste.
You optimize for truth, clarity, and usefulness over politeness theater.

## Style
- Be direct without being cold
- Prefer substance over filler
- Push back when something is a bad idea
- Admit uncertainty plainly
- Keep explanations compact unless depth is useful

## What to avoid
- Sycophancy
- Hype language
- Repeating the user's framing if it's wrong
- Overexplaining obvious things

## Technical posture
- Prefer simple systems over clever systems
- Care about operational reality, not idealized architecture
- Treat edge cases as part of the design, not cleanup
```

## O que o Hermes injeta no prompt {#what-hermes-injects-into-the-prompt}

O conteúdo de `SOUL.md` vai direto para o slot #1 do system prompt — a posição de identidade do agente. Nenhuma linguagem wrapper é adicionada ao redor.

O conteúdo passa por:
- varredura de prompt injection
- truncamento se for grande demais

Se o arquivo estiver vazio, só com whitespace ou não puder ser lido, o Hermes usa identidade padrão built-in ("You are Hermes Agent, an intelligent AI assistant created by Nous Research..."). Esse fallback também se aplica quando `skip_context_files` está definido (por exemplo, em contextos de subagente/delegação).

## Varredura de segurança {#security-scanning}

`SOUL.md` é varrido como outros arquivos que carregam contexto, em busca de padrões de prompt injection antes da inclusão.

Isso significa que você ainda deve mantê-lo focado em persona/voz em vez de tentar inserir meta-instruções estranhas.

## SOUL.md vs AGENTS.md {#soulmd-vs-agentsmd}

Esta é a distinção mais importante.

### SOUL.md {#soulmd}
Use para:
- identidade
- tom
- estilo
- padrões de comunicação
- comportamento no nível de personalidade

### AGENTS.md {#agentsmd}
Use para:
- arquitetura de projeto
- convenções de código
- preferências de ferramentas
- workflows específicos do repo
- comandos, portas, caminhos, notas de deploy

Uma regra útil:
- se deve seguir você em todo lugar, pertence a `SOUL.md`
- se pertence a um projeto, pertence a `AGENTS.md`

## SOUL.md vs `/personality` {#soulmd-vs-personality}

`SOUL.md` é sua personalidade padrão durável.

`/personality` é um overlay no nível da sessão que muda ou complementa o system prompt atual.

Então:
- `SOUL.md` = voz baseline
- `/personality` = troca temporária de modo

Exemplos:
- mantenha um SOUL pragmático padrão, depois use `/personality teacher` para uma conversa de tutoria
- mantenha um SOUL conciso, depois use `/personality creative` para brainstorming

## Personalidades built-in {#built-in-personalities}

O Hermes vem com personalidades built-in que você pode alternar com `/personality`.

| Nome | Descrição |
|------|-------------|
| **helpful** | Assistente amigável e generalista |
| **concise** | Respostas breves e diretas |
| **technical** | Especialista técnico detalhado e preciso |
| **creative** | Pensamento inovador, fora da caixa |
| **teacher** | Educador paciente com exemplos claros |
| **kawaii** | Expressões fofas, brilhos e entusiasmo ★ |
| **catgirl** | Neko-chan com expressões felinas, nya~ |
| **pirate** | Capitão Hermes, bucaneiro tech-savvy |
| **shakespeare** | Prosa bardica com dramaticidade |
| **surfer** | Vibes totalmente relax de surfista |
| **noir** | Narração de detetive hard-boiled |
| **uwu** | Máximo de fofura com uwu-speak |
| **philosopher** | Contemplação profunda em toda pergunta |
| **hype** | ENERGIA E ENTUSIASMO MÁXIMOS!!! |

## Alternar personalidades com comandos {#switching-personalities-with-commands}

### CLI {#cli}

```text
/personality
/personality concise
/personality technical
```

### Plataformas de mensagens {#messaging-platforms}

```text
/personality teacher
```

São overlays convenientes, mas seu `SOUL.md` global ainda dá ao Hermes a personalidade padrão persistente, a menos que o overlay a mude de forma significativa.

## Personalidades customizadas na config {#custom-personalities-in-config}

Você também pode definir personalidades nomeadas customizadas em `~/.hermes/config.yaml` sob `agent.personalities`.

```yaml
agent:
  personalities:
    codereviewer: >
      You are a meticulous code reviewer. Identify bugs, security issues,
      performance concerns, and unclear design choices. Be precise and constructive.
```

Depois alterne com:

```text
/personality codereviewer
```

## Fluxo de trabalho recomendado {#recommended-workflow}

Uma configuração padrão sólida é:

1. Mantenha um `SOUL.md` global bem pensado em `~/.hermes/SOUL.md`
2. Coloque instruções de projeto em `AGENTS.md`
3. Use `/personality` só quando quiser uma mudança temporária de modo

Isso dá:
- uma voz estável
- comportamento específico de projeto onde pertence
- controle temporário quando necessário

## Como a personalidade interage com o prompt completo {#how-personality-interacts-with-the-full-prompt}

Em alto nível, a pilha de prompt inclui:
1. **SOUL.md** (identidade do agente — ou fallback built-in se SOUL.md indisponível)
2. orientação de comportamento com awareness de tools
3. contexto de memória/usuário
4. orientação de skills
5. context files (`AGENTS.md`, `.cursorrules`)
6. timestamp
7. dicas de formatação específicas da plataforma
8. overlays opcionais de system prompt como `/personality`

`SOUL.md` é a fundação — todo o resto se constrói sobre ela.

## Documentação relacionada {#related-docs}

- [Context Files](/user-guide/features/context-files)
- [Configuration](/user-guide/configuration)
- [Tips & Best Practices](/guides/tips)
- [SOUL.md Guide](/guides/use-soul-with-hermes)

## Aparência da CLI vs personalidade conversacional {#cli-appearance-vs-conversational-personality}

Personalidade conversacional e aparência da CLI são separadas:

- `SOUL.md`, `agent.system_prompt` e `/personality` afetam como o Hermes fala
- `display.skin` e `/skin` afetam como o Hermes aparece no terminal

Para aparência do terminal, veja [Skins & Themes](./skins.md).
