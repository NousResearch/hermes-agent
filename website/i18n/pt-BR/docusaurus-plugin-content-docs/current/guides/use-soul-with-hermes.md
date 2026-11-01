---
sidebar_position: 7
title: "Use o SOUL.md com o Hermes"
description: "Como usar o SOUL.md para moldar a voz padrão do Hermes Agent, o que pertence a ele, e como ele difere do AGENTS.md e do /personality"
---

# Use o SOUL.md com o Hermes {#use-soulmd-with-hermes}

O `SOUL.md` é a **identidade primária** da sua instância do Hermes. É a primeira coisa no prompt do sistema — ele define quem é o agente, como fala e o que evita.

Se você quer que o Hermes pareça o mesmo assistente toda vez que você fala com ele — ou se quer substituir completamente a persona do Hermes pela sua própria — este é o arquivo a usar.

## Para que serve o SOUL.md {#what-soulmd-is-for}

Use o `SOUL.md` para:
- tom
- personalidade
- estilo de comunicação
- quão direto ou caloroso o Hermes deve ser
- o que o Hermes deve evitar estilisticamente
- como o Hermes deve se relacionar com incerteza, discordância e ambiguidade

Em resumo:
- o `SOUL.md` trata de quem o Hermes é e como o Hermes fala

## Para que o SOUL.md não serve {#what-soulmd-is-not-for}

Não o use para:
- convenções de código específicas do repositório
- caminhos de arquivo
- comandos
- portas de serviço
- notas de arquitetura
- instruções de fluxo de trabalho do projeto

Isso pertence ao `AGENTS.md`.

Uma boa regra:
- se deve se aplicar a tudo, coloque no `SOUL.md`
- se pertence apenas a um projeto, coloque no `AGENTS.md`

## Onde ele vive {#where-it-lives}

O Hermes agora usa apenas o arquivo SOUL global para a instância atual:

```text
~/.hermes/SOUL.md
```

Se você executar o Hermes com um diretório home personalizado, ele se torna:

```text
$HERMES_HOME/SOUL.md
```

## Comportamento na Primeira Execução {#first-run-behavior}

O Hermes gera automaticamente um `SOUL.md` inicial para você, caso um ainda não exista.

Isso significa que a maioria dos usuários agora começa com um arquivo real que podem ler e editar imediatamente.

Importante:
- se você já tem um `SOUL.md`, o Hermes não o sobrescreve
- se o arquivo existe, mas está vazio, o Hermes não adiciona nada dele ao prompt

## Como o Hermes o Usa {#how-hermes-uses-it}

Quando o Hermes inicia uma sessão, ele lê o `SOUL.md` de `HERMES_HOME`, verifica se há padrões de injeção de prompt, o trunca se necessário, e o usa como a **identidade do agente** — posição nº 1 no prompt do sistema. Isso significa que o SOUL.md substitui completamente o texto de identidade padrão integrado.

Se o SOUL.md estiver ausente, vazio ou não puder ser carregado, o Hermes recorre a uma identidade padrão integrada.

Nenhuma linguagem de wrapper é adicionada em torno do arquivo. O próprio conteúdo é o que importa — escreva da forma como você quer que seu agente pense e fale.

## Uma Boa Primeira Edição {#a-good-first-edit}

Se você não fizer mais nada, abra o arquivo e mude apenas algumas linhas para que pareça com você.

Por exemplo:

```markdown
You are direct, calm, and technically precise.
Prefer substance over politeness theater.
Push back clearly when an idea is weak.
Keep answers compact unless deeper detail is useful.
```

Só isso já pode mudar perceptivelmente como o Hermes se sente.

## Exemplos de Estilos {#example-styles}

### 1. Engenheiro pragmático {#1-pragmatic-engineer}

```markdown
You are a pragmatic senior engineer.
You care more about correctness and operational reality than sounding impressive.

## Style
- Be direct
- Be concise unless complexity requires depth
- Say when something is a bad idea
- Prefer practical tradeoffs over idealized abstractions

## Avoid
- Sycophancy
- Hype language
- Overexplaining obvious things
```

### 2. Parceiro de pesquisa {#2-research-partner}

```markdown
You are a thoughtful research collaborator.
You are curious, honest about uncertainty, and excited by unusual ideas.

## Style
- Explore possibilities without pretending certainty
- Distinguish speculation from evidence
- Ask clarifying questions when the idea space is underspecified
- Prefer conceptual depth over shallow completeness
```

### 3. Professor / explicador {#3-teacher--explainer}

```markdown
You are a patient technical teacher.
You care about understanding, not performance.

## Style
- Explain clearly
- Use examples when they help
- Do not assume prior knowledge unless the user signals it
- Build from intuition to details
```

### 4. Revisor rigoroso {#4-tough-reviewer}

```markdown
You are a rigorous reviewer.
You are fair, but you do not soften important criticism.

## Style
- Point out weak assumptions directly
- Prioritize correctness over harmony
- Be explicit about risks and tradeoffs
- Prefer blunt clarity to vague diplomacy
```

## O que faz um SOUL.md forte? {#what-makes-a-strong-soulmd}

Um `SOUL.md` forte é:
- estável
- amplamente aplicável
- específico na voz
- não sobrecarregado com instruções temporárias

Um `SOUL.md` fraco é:
- repleto de detalhes do projeto
- contraditório
- tentando microgerenciar o formato de cada resposta
- majoritariamente preenchimento genérico como "seja útil" e "seja claro"

O Hermes já tenta ser útil e claro. O `SOUL.md` deve adicionar personalidade e estilo reais, não repetir os padrões óbvios.

## Estrutura Sugerida {#suggested-structure}

Você não precisa de títulos, mas eles ajudam.

Uma estrutura simples que funciona bem:

```markdown
# Identity
Who Hermes is.

# Style
How Hermes should sound.

# Avoid
What Hermes should not do.

# Defaults
How Hermes should behave when ambiguity appears.
```

## SOUL.md vs /personality {#soulmd-vs-personality}

Estes são complementares.

Use `SOUL.md` para sua linha de base duradoura.
Use `/personality` para trocas de modo temporárias.

Exemplos:
- seu SOUL padrão é pragmático e direto
- então, por uma sessão, você usa `/personality teacher`
- depois você volta ao normal sem alterar seu arquivo de voz base

## SOUL.md vs AGENTS.md {#soulmd-vs-agentsmd}

Este é o erro mais comum.

### Coloque isto no SOUL.md {#put-this-in-soulmd}
- "Seja direto."
- "Evite linguagem de hype."
- "Prefira respostas curtas, a menos que a profundidade ajude."
- "Contraponha quando o usuário estiver errado."

### Coloque isto no AGENTS.md {#put-this-in-agentsmd}
- "Use pytest, não unittest."
- "O frontend está em `frontend/`."
- "Nunca edite migrations diretamente."
- "A API roda na porta 8000."

## Como editá-lo {#how-to-edit-it}

```bash
nano ~/.hermes/SOUL.md
```

ou

```bash
vim ~/.hermes/SOUL.md
```

Depois reinicie o Hermes ou comece uma nova sessão.

## Um Fluxo de Trabalho Prático {#a-practical-workflow}

1. Comece com o arquivo padrão gerado
2. Corte tudo o que não parece a voz que você quer
3. Adicione 4–8 linhas que definam claramente o tom e os padrões
4. Converse com o Hermes por um tempo
5. Ajuste com base no que ainda parece estranho

Essa abordagem iterativa funciona melhor do que tentar projetar a personalidade perfeita de uma vez só.

## Solução de Problemas {#troubleshooting}

### Editei o SOUL.md, mas o Hermes ainda soa igual {#i-edited-soulmd-but-hermes-still-sounds-the-same}

Verifique:
- você editou `~/.hermes/SOUL.md` ou `$HERMES_HOME/SOUL.md`
- não algum `SOUL.md` local do repositório
- o arquivo não está vazio
- sua sessão foi reiniciada após a edição
- uma sobreposição de `/personality` não está dominando o resultado

### O Hermes está ignorando partes do meu SOUL.md {#hermes-is-ignoring-parts-of-my-soulmd}

Causas possíveis:
- instruções de prioridade mais alta estão sobrescrevendo-o
- o arquivo inclui orientações conflitantes
- o arquivo é muito longo e foi truncado
- parte do texto se parece com conteúdo de injeção de prompt e pode ser bloqueado ou alterado pelo scanner

### Meu SOUL.md ficou muito específico do projeto {#my-soulmd-became-too-project-specific}

Mova as instruções do projeto para o `AGENTS.md` e mantenha o `SOUL.md` focado em identidade e estilo.

## Documentos Relacionados {#related-docs}

- [Personalidade e SOUL.md](/user-guide/features/personality)
- [Arquivos de Contexto](/user-guide/features/context-files)
- [Configuração](/user-guide/configuration)
- [Dicas e Boas Práticas](/guides/tips)
