---
sidebar_position: 4
title: "Qual arquivo faz o quê?"
description: "SOUL.md vs USER.md vs MEMORY.md vs AGENTS.md — um mapa de uma página dos arquivos do agente, quem escreve cada um e quando o agente de fato os vê"
---

# Qual arquivo faz o quê? {#which-file-does-what}

"Eu disse algo ao meu agente e ele esqueceu." "Qual arquivo é o cérebro do meu agente?" "Editei o SOUL.md — por que ele não sabe meu nome?" Essas perguntas se reduzem à mesma coisa: o Hermes Agent é moldado por vários arquivos markdown, e cada um tem um papel diferente. Esta página mapeia todos em um só lugar. Para profundidade em qualquer um deles, siga os links para [Memória persistente](/user-guide/features/memory), [Personalidade e SOUL.md](/user-guide/features/personality) e [Arquivos de contexto](/user-guide/features/context-files).

## A tabela mestra {#the-master-table}

| Arquivo | O que contém | Quem escreve | Quando o agente vê | Onde fica |
|------|---------------|---------------|------------------------|----------------|
| **SOUL.md** | Identidade primária do agente — personalidade, tom, estilo de comunicação, o que evitar estilisticamente | Você. O Hermes cria um arquivo inicial automaticamente se não existir; arquivos existentes nunca são sobrescritos | Slot #1 do system prompt, no início da sessão | `~/.hermes/SOUL.md` (ou `$HERMES_HOME/SOUL.md` com home customizado) — nunca o diretório de trabalho |
| **USER.md** | Perfil do usuário — seu nome, papel, preferências, estilo de comunicação, expectativas | O agente, via ferramenta `memory` (você pode exigir aprovação com `write_approval` ou editar entradas via `hermes journey edit`) | Injetado no system prompt como snapshot congelado no início da sessão | `~/.hermes/memories/` |
| **MEMORY.md** | Notas pessoais do agente — fatos de ambiente, convenções de projeto, peculiaridades de ferramentas, coisas aprendidas | O agente, via ferramenta `memory` (mesmas opções de gating e edição que USER.md) | Injetado no system prompt como snapshot congelado no início da sessão | `~/.hermes/memories/` |
| **AGENTS.md** | Instruções de projeto, convenções, arquitetura — comandos, portas, caminhos, fluxos específicos do repositório | Você (ou quem autora o projeto) | Carregado no system prompt na inicialização a partir do diretório de trabalho; cópias aninhadas são descobertas progressivamente conforme o agente navega subdiretórios | Diretório de trabalho do projeto + subdiretórios |
| **.hermes.md** / **HERMES.md** | Instruções de projeto, como AGENTS.md, mas específicas do Hermes e com maior prioridade | Você | Carregado no system prompt na inicialização (primeira correspondência vence sobre AGENTS.md) | Seu projeto — a descoberta sobe até a raiz git |

:::info Um arquivo de contexto de projeto por sessão
Apenas **um** tipo de contexto de projeto é carregado por sessão, primeira correspondência vence: `.hermes.md` → `AGENTS.md` → `CLAUDE.md` → `.cursorrules`. `SOUL.md` é sempre carregado independentemente como identidade do agente — não faz parte dessa cadeia de prioridade. Veja [Arquivos de contexto](/user-guide/features/context-files) para a lista completa, incluindo compatibilidade com `CLAUDE.md` e `.cursorrules`.
:::

Um atalho útil:

- **SOUL.md** é quem o agente *é* — se deve segui-lo para todo lugar, pertence aqui.
- **USER.md** é quem *você* é — o agente mantém isso para você.
- **MEMORY.md** é o que o agente *aprendeu* — ele também mantém isso sozinho.
- **AGENTS.md** (ou `.hermes.md`) é o que o *projeto* precisa — se pertence a um projeto, pertence aqui.

## "Por que ele esqueceu o que acabei de dizer?" {#why-did-it-forget-what-i-just-said}

Memória (MEMORY.md e USER.md) é injetada no system prompt como um **snapshot congelado** capturado uma vez no início da sessão — quando o agente salva algo no meio da sessão, a mudança persiste no disco imediatamente, mas não aparecerá no system prompt até a próxima sessão começar. Isso é intencional: preserva o prefix cache do LLM por desempenho, e respostas de ferramentas sempre mostram o estado ao vivo, então nada se perde — inicie uma nova sessão e a memória atualizada estará lá. Detalhes completos em [Como a memória aparece no system prompt](/user-guide/features/memory#how-memory-appears-in-the-system-prompt).

## Confusões comuns {#common-mix-ups}

### "Coloquei fatos sobre mim no SOUL.md, mas o USER.md ficou vazio" {#i-put-facts-about-myself-in-soulmd-but-usermd-stayed-empty}

`SOUL.md` e `USER.md` são sistemas separados que nunca se alimentam. `SOUL.md` é um arquivo de personalidade que **você** edita diretamente — molda tom e identidade, e seu conteúdo é injetado verbatim como slot #1 do prompt. `USER.md` faz parte da memória persistente e é escrito pelo **agente** pela ferramenta `memory`. Se quiser fatos sobre você em USER.md, diga ao agente ("remember that I prefer concise answers") e ele salva — editar SOUL.md não popula memória, e entradas de memória não mudam a persona. Use SOUL.md para orientação durável de voz e personalidade; deixe preferências e fatos de perfil para memória. Veja [O que colocar no SOUL.md?](/user-guide/features/personality#what-should-go-in-soulmd) e [Dois alvos explicados](/user-guide/features/memory#two-targets-explained).

### "Disse meu nome no meio da sessão e ele agiu como se nunca tivesse ouvido" {#i-told-it-my-name-mid-session-and-it-acted-like-it-never-heard-it}

Se o agente salvou seu nome na memória, o save funcionou — verifique com as respostas da ferramenta `memory` ou `hermes journey list`. O que você está vendo é a regra do snapshot congelado acima: o system prompt não atualiza no meio da sessão, então o bloco de memória *injetado* ainda mostra o estado do início da sessão. O agente ainda pode usar o que você disse na conversa atual (está no contexto), e a entrada salva estará no system prompt a partir da próxima sessão. O mesmo vale para edições que você faz em `SOUL.md` ou `AGENTS.md` com uma sessão em execução: o contexto é montado no início da sessão, então reinicie a sessão para aplicar mudanças.

:::tip Guia rápido de decisão
- Quer mudar **como o agente fala**? Edite `~/.hermes/SOUL.md` — [Personalidade e SOUL.md](/user-guide/features/personality).
- Quer que o agente **lembre um fato**? Diga — ele salva na memória sozinho. [Memória persistente](/user-guide/features/memory).
- Quer definir **regras do projeto**? Coloque um `AGENTS.md` (ou `.hermes.md`) no projeto — [Arquivos de contexto](/user-guide/features/context-files).
- Precisa de uma mudança de personalidade **temporária**? Use `/personality` — é uma sobreposição no nível da sessão, sem editar arquivos.
:::

## Documentação relacionada {#related-docs}

- [Memória persistente](/user-guide/features/memory) — MEMORY.md, USER.md, ferramenta `memory`, limites de capacidade, `write_approval`
- [Personalidade e SOUL.md](/user-guide/features/personality) — orientação de conteúdo do SOUL.md, presets de `/personality`, a pilha de prompt
- [Arquivos de contexto](/user-guide/features/context-files) — AGENTS.md, `.hermes.md`, descoberta progressiva, varredura de segurança
