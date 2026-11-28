---
sidebar_position: 9
title: "Importar de Outros Agentes"
description: "Importação com um único comando de uma configuração do Claude Code (~/.claude) ou do OpenAI Codex CLI (~/.codex) para o Hermes — instruções, allowlists, servidores MCP, skills e memórias."
---

# Importar de Outros Agentes

O `hermes import-agent` importa sua configuração existente do **Claude Code** ou do **OpenAI Codex CLI** para o Hermes com um único comando. Ele segue o mesmo padrão de preview primeiro do [`hermes claw migrate`](../guides/migrate-from-openclaw.md): você sempre vê um plano item por item antes de qualquer coisa ser gravada, e `--dry-run` nunca toca o disco.

```bash
hermes import-agent                    # auto-detect ~/.claude or ~/.codex
hermes import-agent claude-code        # import from ~/.claude
hermes import-agent codex              # import from ~/.codex
hermes import-agent claude-code --dry-run          # preview only
hermes import-agent codex --source /path/to/.codex # custom location
hermes import-agent claude-code --overwrite --yes  # replace conflicts, skip prompts
```

## O que é importado {#what-gets-imported}

### Claude Code (`~/.claude`) {#claude-code-claude}

| Claude Code | Hermes |
|---|---|
| `CLAUDE.md` (instruções globais) | Entradas de memória em `~/.hermes/memories/MEMORY.md` |
| `settings.json` → `permissions.allow` (regras `Bash(...)`) | `command_allowlist` em `config.yaml` |
| `settings.json` → `permissions.deny` (regras `Bash(...)`) | `approvals.deny` em `config.yaml` |
| `mcpServers` (de `~/.claude.json` e `settings.json`) | `mcp_servers` em `config.yaml` |
| `skills/<name>/` (diretórios com `SKILL.md`) | `~/.hermes/skills/claude-code-imports/<name>/` |
| `commands/*.md` (slash commands) | Ignorado, com uma nota — converta-os em skills |

As regras de prefixo `Bash(npm run test:*)` do Claude viram globs `npm run test*`. Regras de permissão que não são `Bash` (`Read(...)`, `WebFetch`, ...) controlam ferramentas específicas do Claude e são reportadas como não mapeadas em vez de importadas.

### Codex CLI (`~/.codex`) {#codex-cli-codex}

| Codex CLI | Hermes |
|---|---|
| `AGENTS.md` (instruções globais) | Entradas de memória em `~/.hermes/memories/MEMORY.md` |
| `config.toml` → `[mcp_servers.*]` | `mcp_servers` em `config.yaml` |
| `memories/*.md` | Entradas de memória em `~/.hermes/memories/MEMORY.md` |
| `skills/<name>/` (diretórios com `SKILL.md`) | `~/.hermes/skills/codex-imports/<name>/` |

## O que nunca é importado {#what-is-never-imported}

**Chaves de API e credenciais.** Arquivos de credenciais (`~/.claude/.credentials.json`, `~/.codex/auth.json`) nunca são lidos, e variáveis de ambiente ou headers de servidor MCP com nomes que parecem segredos (`*_TOKEN`, `*_API_KEY`, `Authorization`, ...) são removidos e listados no relatório para que você possa readicioná-los deliberadamente. Execute `hermes setup` para configurar provedores, ou adicione segredos em `~/.hermes/.env`.

## Notas de comportamento {#behavior-notes}

- **Preview primeiro, sempre.** O comando imprime o plano completo antes de aplicar; em sessões não interativas, ele para no preview a menos que você passe `--yes`.
- **Faz merge, não substitui.** Entradas de memória são deduplicadas contra seu `MEMORY.md` existente; os padrões de allowlist/denylist fazem merge com o que já está em `config.yaml`.
- **Conflitos são ignorados por padrão.** Um servidor MCP ou skill que já existe no Hermes é reportado como conflito; passe `--overwrite` para substituí-lo.
- **Arquivos malformados não abortam a execução.** Um `settings.json` ou `config.toml` quebrado vira um erro por item no relatório, enquanto tudo o mais ainda é importado.
- Vindo do OpenClaw em vez disso? Use o [`hermes claw migrate`](../guides/migrate-from-openclaw.md).
