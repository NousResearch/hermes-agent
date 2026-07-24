---
sidebar_position: 3
title: 'Rota de Aprendizado'
description: 'Escolha sua rota de aprendizado pela documentação do Hermes Agent com base no seu nível de experiência e objetivos.'
---

# Rota de Aprendizado

O Hermes Agent pode fazer muita coisa — assistente CLI, bot no Telegram/Discord, automação de tarefas, treinamento RL e muito mais. Esta página ajuda você a descobrir por onde começar e o que ler com base no seu nível de experiência e no que você está tentando realizar.

:::tip Comece por aqui
Se você ainda não instalou o Hermes Agent, comece com o [Guia de Instalação](/getting-started/installation) e depois siga o [Início Rápido](/getting-started/quickstart). Tudo abaixo assume que você tem uma instalação funcional.
:::

:::tip Configuração inicial de provider
Usuários iniciantes quase sempre querem `hermes setup --portal` — um OAuth cobre um modelo mais as quatro ferramentas do Tool Gateway (pesquisa/imagem/TTS/navegador). Veja [Nous Portal](/integrations/nous-portal).
:::

## Como Usar Esta Página

- **Sabe seu nível?** Vá para a [tabela por nível de experiência](#por-nível-de-experiência) e siga a ordem de leitura para seu nível.
- **Tem um objetivo específico?** Vá direto para [Por Caso de Uso](#por-caso-de-uso) e encontre o cenário que corresponde.
- **Só está explorando?** Confira a tabela [Principais Recursos de Relance](#principais-recursos-de-relance) para uma visão geral rápida de tudo que o Hermes Agent pode fazer.

## Por Nível de Experiência

| Nível          | Objetivo                                                    | Leitura Recomendada                                                                                          | Tempo Estimado |
|----------------|-------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------|----------------|
| **Iniciante**  | Instalar, ter conversas básicas, usar ferramentas integradas | [Instalação](/getting-started/installation) → [Início Rápido](/getting-started/quickstart) → [Uso do CLI](/user-guide/cli) → [Configuração](/user-guide/configuration) | ~1 hora        |
| **Intermediário** | Configurar bots de mensagem, usar recursos avançados como memória, cron jobs e skills | [Sessões](/user-guide/sessions) → [Mensagens](/user-guide/messaging) → [Ferramentas](/user-guide/features/tools) → [Skills](/user-guide/features/skills) → [Memória](/user-guide/features/memory) → [Cron](/user-guide/features/cron) | ~2–3 horas     |
| **Avançado**   | Construir ferramentas customizadas, criar skills, treinar modelos com RL, contribuir com o projeto | [Arquitetura](/developer-guide/architecture) → [Adicionando Ferramentas](/developer-guide/adding-tools) → [Criando Skills](/developer-guide/creating-skills) → [Contribuindo](/developer-guide/contributing) | ~4–6 horas     |

## Por Caso de Uso

Escolha o cenário que corresponde ao que você quer fazer. Cada um vincula você à documentação relevante na ordem em que deve lê-los.

### "Quero um assistente de programação via CLI"

Use o Hermes Agent como um assistente de terminal interativo para escrever, revisar e executar código.

1. [Instalação](/getting-started/installation)
2. [Início Rápido](/getting-started/quickstart)
3. [Uso do CLI](/user-guide/cli)
4. [Execução de Código](/user-guide/features/code-execution)
5. [Arquivos de Contexto](/user-guide/features/context-files)
6. [Dicas e Truques](/guides/tips)

:::tip
Passe arquivos diretamente para sua conversa com arquivos de contexto. O Hermes Agent pode ler, editar e executar código em seus projetos.
:::

### "Quero um bot no Telegram/Discord"

Implante o Hermes Agent como um bot em sua plataforma de mensagens favorita.

1. [Instalação](/getting-started/installation)
2. [Configuração](/user-guide/configuration)
3. [Visão Geral de Mensagens](/user-guide/messaging)
4. [Configuração do Telegram](/user-guide/messaging/telegram)
5. [Configuração do Discord](/user-guide/messaging/discord)
6. [Modo de Voz](/user-guide/features/voice-mode)
7. [Usar Modo de Voz com Hermes](/guides/use-voice-mode-with-hermes)
8. [Segurança](/user-guide/security)

Para exemplos completos de projetos, veja:
- [Daily Briefing Bot](/guides/daily-briefing-bot)
- [Team Telegram Assistant](/guides/team-telegram-assistant)

### "Quero automatizar tarefas"

Agende tarefas recorrentes, execute jobs em lote ou encadeie ações do agente.

1. [Início Rápido](/getting-started/quickstart)
2. [Agendamento Cron](/user-guide/features/cron)
3. [Processamento em Lote](/user-guide/features/batch-processing)
4. [Delegação](/user-guide/features/delegation)
5. [Hooks](/user-guide/features/hooks)

:::tip
Cron jobs permitem que o Hermes Agent execute tarefas em um agendamento — resumos diários, verificações periódicas, relatórios automatizados — sem você estar presente.
:::

### "Quero construir ferramentas/skills customizadas"

Estenda o Hermes Agent com suas próprias ferramentas e pacotes de skills reutilizáveis.

1. [Plugins](/user-guide/features/plugins)
2. [Construir um Plugin Hermes](/developer-guide/plugins)
3. [Visão Geral de Ferramentas](/user-guide/features/tools)
4. [Visão Geral de Skills](/user-guide/features/skills)
5. [MCP (Model Context Protocol)](/user-guide/features/mcp)
6. [Arquitetura](/developer-guide/architecture)
7. [Adicionando Ferramentas](/developer-guide/adding-tools)
8. [Criando Skills](/developer-guide/creating-skills)

:::tip
Para a maioria das criações de ferramentas customizadas, comece com plugins. A página
[Adicionando Ferramentas](/developer-guide/adding-tools) é para o desenvolvimento interno do núcleo do Hermes,
não para o caminho usual de ferramentas do usuário.
:::

### "Quero treinar modelos"

Use aprendizado por reforço para ajustar o comportamento do modelo com o pipeline de treinamento RL do Hermes Agent (alimentado pelo [Atropos](https://github.com/NousResearch/atropos)).

1. [Início Rápido](/getting-started/quickstart)
2. [Configuração](/user-guide/configuration)
3. [Ambientes RL do Atropos](https://github.com/NousResearch/atropos) (externo)
4. [Roteamento de Provider](/user-guide/features/provider-routing)
5. [Arquitetura](/developer-guide/architecture)

:::tip
O treinamento RL funciona melhor quando você já entende os fundamentos de como o Hermes Agent lida com conversas e chamadas de ferramentas. Siga o caminho Iniciante primeiro se você é novo.
:::

### "Quero usar como biblioteca Python"

Integre o Hermes Agent em suas próprias aplicações Python programaticamente.

1. [Instalação](/getting-started/installation)
2. [Início Rápido](/getting-started/quickstart)
3. [Guia da Biblioteca Python](/guides/python-library)
4. [Arquitetura](/developer-guide/architecture)
5. [Ferramentas](/user-guide/features/tools)
6. [Sessões](/user-guide/sessions)

## Principais Recursos de Relance

Não sabe o que está disponível? Aqui está um diretório rápido dos principais recursos:

| Recurso                    | O Que Faz                                                               | Link                                                                     |
|----------------------------|-------------------------------------------------------------------------|--------------------------------------------------------------------------|
| **Ferramentas**            | Ferramentas integradas que o agente pode chamar (I/O de arquivo, busca, shell, etc.) | [Ferramentas](/user-guide/features/tools)                                 |
| **Skills**                 | Pacotes de plugin instaláveis que adicionam novas capacidades           | [Skills](/user-guide/features/skills)                                     |
| **Memória**                | Memória persistente entre sessões                                       | [Memória](/user-guide/features/memory)                                    |
| **Arquivos de Contexto**   | Alimente arquivos e diretórios nas conversas                            | [Arquivos de Contexto](/user-guide/features/context-files)                 |
| **MCP**                    | Conecte-se a servidores de ferramentas externos via Model Context Protocol | [MCP](/user-guide/features/mcp)                                           |
| **Cron**                   | Agende tarefas recorrentes do agente                                    | [Cron](/user-guide/features/cron)                                         |
| **Delegação**              | Crie subagentes para trabalho paralelo                                  | [Delegação](/user-guide/features/delegation)                               |
| **Execução de Código**     | Execute scripts Python que chamam ferramentas do Hermes programaticamente | [Execução de Código](/user-guide/features/code-execution)                  |
| **Navegador**              | Navegação e raspagem web                                                | [Navegador](/user-guide/features/browser)                                 |
| **Hooks**                  | Callbacks e middleware orientados a eventos                             | [Hooks](/user-guide/features/hooks)                                       |
| **Processamento em Lote**  | Processe múltiplas entradas em massa                                    | [Processamento em Lote](/user-guide/features/batch-processing)             |
| **Roteamento de Provider** | Roteie requisições entre múltiplos providers de LLM                     | [Roteamento de Provider](/user-guide/features/provider-routing)            |

## O Que Ler em Seguida

Com base em onde você está agora:

- **Acabou de instalar?** → Vá para o [Início Rápido](/getting-started/quickstart) para executar sua primeira conversa.
- **Completou o Início Rápido?** → Leia [Uso do CLI](/user-guide/cli) e [Configuração](/user-guide/configuration) para personalizar sua instalação.
- **Confortável com o básico?** → Explore [Ferramentas](/user-guide/features/tools), [Skills](/user-guide/features/skills) e [Memória](/user-guide/features/memory) para liberar todo o poder do agente.
- **Configurando para uma equipe?** → Leia [Segurança](/user-guide/security) e [Sessões](/user-guide/sessions) para entender controle de acesso e gerenciamento de conversas.
- **Pronto para construir?** → Mergulhe no [Guia do Desenvolvedor](/developer-guide/architecture) para entender os internos e começar a contribuir.
- **Quer exemplos práticos?** → Confira a seção [Guias](/guides/tips) para projetos do mundo real e dicas.

:::tip
Você não precisa ler tudo. Escolha o caminho que corresponde ao seu objetivo, siga os links em ordem, e você será produtivo rapidamente. Você sempre pode voltar a esta página para encontrar seu próximo passo.
:::
