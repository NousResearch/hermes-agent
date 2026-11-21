---
sidebar_position: 9
sidebar_label: "Referências de contexto"
title: "Referências de contexto"
description: "Sintaxe inline com @ para anexar arquivos, pastas, diffs git e URLs diretamente às suas mensagens"
---

# Referências de contexto {#context-references}

Digite `@` seguido de uma referência para injetar conteúdo diretamente na sua mensagem. O Hermes expande a referência inline e anexa o conteúdo em uma seção `--- Attached Context ---`.

## Referências suportadas {#supported-references}

| Sintaxe | Descrição |
|--------|-------------|
| `@file:path/to/file.py` | Injeta o conteúdo do arquivo |
| `@file:path/to/file.py:10-25` | Injeta intervalo específico de linhas (indexado em 1, inclusivo) |
| `@folder:path/to/dir` | Injeta listagem da árvore de diretórios com metadados dos arquivos |
| `@diff` | Injeta `git diff` (alterações não staged na working tree) |
| `@staged` | Injeta `git diff --staged` (alterações staged) |
| `@git:5` | Injeta os últimos N commits com patches (máx. 10) |
| `@url:https://example.com` | Busca e injeta o conteúdo da página web |

## Exemplos de uso {#usage-examples}

```text
Review @file:src/main.py and suggest improvements

What changed? @diff

Compare @file:old_config.yaml and @file:new_config.yaml

What's in @folder:src/components?

Summarize this article @url:https://arxiv.org/abs/2301.00001
```

Várias referências funcionam em uma única mensagem:

```text
Check @file:main.py, and also @file:test.py.
```

Pontuação final (`,`, `.`, `;`, `!`, `?`) é removida automaticamente dos valores de referência.

## Autocompletar com Tab na CLI {#cli-tab-completion}

Na CLI interativa, digitar `@` aciona o autocompletar:

- `@` mostra todos os tipos de referência (`@diff`, `@staged`, `@file:`, `@folder:`, `@git:`, `@url:`)
- `@file:` e `@folder:` acionam autocompletar de caminho no filesystem com metadados de tamanho do arquivo
- `@` seguido de texto parcial mostra arquivos e pastas correspondentes do diretório atual

## Intervalos de linhas {#line-ranges}

A referência `@file:` suporta intervalos de linhas para injeção precisa de conteúdo:

```text
@file:src/main.py:42        # Single line 42
@file:src/main.py:10-25     # Lines 10 through 25 (inclusive)
```

As linhas são indexadas em 1. Intervalos inválidos são ignorados silenciosamente (o arquivo completo é retornado).

## Limites de tamanho {#size-limits}

As referências de contexto são limitadas para evitar sobrecarregar a janela de contexto do modelo:

| Limite | Valor | Comportamento |
|-----------|-------|----------|
| Limite soft | 25% do comprimento de contexto | Aviso anexado, expansão prossegue |
| Limite hard | 50% do comprimento de contexto | Expansão recusada, mensagem original retornada inalterada |
| Entradas de pasta | 200 arquivos máx. | Entradas excedentes substituídas por `- ...` |
| Commits git | 10 máx. | `@git:N` limitado ao intervalo [1, 10] |

## Segurança {#security}

### Bloqueio de caminhos sensíveis {#sensitive-path-blocking}

Estes caminhos são sempre bloqueados em referências `@file:` para evitar exposição de credenciais:

- Chaves e config SSH: `~/.ssh/id_rsa`, `~/.ssh/id_ed25519`, `~/.ssh/authorized_keys`, `~/.ssh/config`
- Perfis de shell: `~/.bashrc`, `~/.zshrc`, `~/.profile`, `~/.bash_profile`, `~/.zprofile`
- Arquivos de credenciais: `~/.netrc`, `~/.pgpass`, `~/.npmrc`, `~/.pypirc`
- Env do Hermes: `$HERMES_HOME/.env`

Estes diretórios são totalmente bloqueados (qualquer arquivo dentro):
- `~/.ssh/`, `~/.aws/`, `~/.gnupg/`, `~/.kube/`, `$HERMES_HOME/skills/.hub/`

### Proteção contra path traversal {#path-traversal-protection}

Todos os caminhos são resolvidos relativamente ao diretório de trabalho. Referências que resolvem fora da raiz de workspace permitida são rejeitadas.

### Detecção de arquivos binários {#binary-file-detection}

Arquivos binários são detectados via tipo MIME e varredura de null bytes. Extensões de texto conhecidas (`.py`, `.md`, `.json`, `.yaml`, `.toml`, `.js`, `.ts`, etc.) ignoram a detecção baseada em MIME. Arquivos binários são rejeitados com um aviso.

## Disponibilidade por plataforma {#platform-availability}

Referências de contexto são principalmente um **recurso da CLI**. Funcionam na CLI interativa, onde `@` aciona autocompletar com Tab e as referências são expandidas antes da mensagem ser enviada ao agente.

Em **plataformas de mensagens** (Telegram, Discord, etc.), a sintaxe `@` não é expandida pelo gateway — as mensagens passam como estão. O agente ainda pode referenciar arquivos via as ferramentas `read_file`, `search_files` e `web_extract`.

## Interação com compressão de contexto {#interaction-with-context-compression}

Quando o contexto da conversa é comprimido, o conteúdo expandido da referência é incluído no resumo da compressão. Isso significa:

- Conteúdos grandes de arquivo injetados via `@file:` contribuem para o uso de contexto
- Se a conversa for comprimida depois, o conteúdo do arquivo é resumido (não preservado verbatim)
- Para arquivos muito grandes, considere intervalos de linhas (`@file:main.py:100-200`) para injetar só as seções relevantes

## Padrões comuns {#common-patterns}

```text
# Code review workflow
Review @diff and check for security issues

# Debug with context
This test is failing. Here's the test @file:tests/test_auth.py
and the implementation @file:src/auth.py:50-80

# Project exploration
What does this project do? @folder:src @file:README.md

# Research
Compare the approaches in @url:https://arxiv.org/abs/2301.00001
and @url:https://arxiv.org/abs/2301.00002
```

## Tratamento de erros {#error-handling}

Referências inválidas produzem avisos inline em vez de falhas:

| Condição | Comportamento |
|-----------|----------|
| Arquivo não encontrado | Aviso: "file not found" |
| Arquivo binário | Aviso: "binary files are not supported" |
| Pasta não encontrada | Aviso: "folder not found" |
| Comando git falha | Aviso com stderr do git |
| URL sem conteúdo | Aviso: "no content extracted" |
| Caminho sensível | Aviso: "path is a sensitive credential file" |
| Caminho fora do workspace | Aviso: "path is outside the allowed workspace" |
