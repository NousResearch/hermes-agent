---
sidebar_position: 3
title: "Atualizar e desinstalar"
description: "Como atualizar o Hermes Agent para a versão mais recente ou desinstalá-lo"
---

# Atualizar e desinstalar

## Atualizando

Atualize para a versão mais recente com um único comando:

```bash
hermes update
```

Isso puxa o código mais recente de `main`, atualiza dependências e pergunta se você quer configurar opções novas adicionadas desde o seu último update.

:::tip
O `hermes update` detecta automaticamente opções novas de configuração e pergunta se você quer adicioná-las. Se pulou esse prompt, rode `hermes config check` para ver o que falta e depois `hermes config migrate` para adicionar de forma interativa.
:::

### O que acontece durante um update

Quando você roda `hermes update`, estes passos ocorrem:

1. **Snapshot pré-update** — um snapshot leve de estado é salvo por padrão (cobre pairing, cron jobs, `config.yaml`, `.env`, `auth.json` e outros arquivos de estado modificados em runtime; arquivos individuais acima de 1 GiB são pulados para que um DB de sessões grande não atrase o update). Controlado por `updates.pre_update_backup` (`quick` por padrão, `full` para um zip de todo o `HERMES_HOME`, `off` para desligar). Recuperável via o fluxo de restore de snapshot descrito em [Snapshots and rollback](../user-guide/checkpoints-and-rollback.md).
2. **Git pull** — puxa o código mais recente do branch `main` e atualiza submodules
3. **Validação de sintaxe pós-pull + auto-rollback** — depois do pull, o Hermes compila os oito arquivos críticos que toda invocação de `hermes` importa na inicialização. Se algum falhar no parse (ex.: marcador órfão de conflito de merge, arquivo truncado por acidente), o Hermes roda `git reset --hard <pre-pull-sha>` para reverter a instalação e o shell continuar bootável. Rode `hermes update` de novo quando o fix upstream cair.
4. **Install de dependências** — roda `uv pip install -e ".[all]"` para pegar dependências novas ou alteradas
5. **Migração de config** — detecta opções novas de config desde a sua versão e pergunta se você quer setá-las
6. **Auto-restart do gateway** — gateways em execução são atualizados depois que o update completa para o código novo valer na hora. Gateways gerenciados por serviço (systemd no Linux, launchd no macOS) são reiniciados pelo service manager. Gateways manuais são relançados automaticamente quando o Hermes consegue mapear o PID em execução de volta a um profile.

### Atualizando contra um branch não default: `--branch`

Por padrão o `hermes update` acompanha `origin/main`. Passe `--branch <name>` para atualizar contra outro branch — útil para canais de QA, feature branches ou teste de release candidate:

```bash
hermes update --branch release-candidate
hermes update --check --branch experimental   # preview behindness only
```

Se o checkout local estiver em outro branch, o Hermes faz auto-stash de trabalho não commitado, muda o HEAD para o branch alvo e então puxa. Branches que não existem localmente são auto-tracked a partir de `origin/<name>` (`git checkout -B <name> origin/<name>`). Branches que não existem em lugar nenhum falham limpo — suas mudanças em stash são restauradas antes de sair, então você não fica preso num estado estranho. A lógica de sync fork-upstream só de `main` é pulada automaticamente em branches que não são `main`.

### Mudanças locais em updates não interativos

Quando você roda `hermes update` num terminal, o Hermes faz stash de mudanças não commitadas na árvore de source, puxa e **pergunta** se deve restaurá-las — exatamente como sempre foi. Nada muda para updates interativos.

Quando o update roda **sem terminal** — pelo botão "Update" do app desktop/chat ou por um update disparado pelo gateway — não há prompt para responder. O setting `updates.non_interactive_local_changes` decide o que acontece com as mudanças em stash:

```yaml
# ~/.hermes/config.yaml
updates:
  non_interactive_local_changes: stash   # default: keep + auto-restore
  # non_interactive_local_changes: discard  # throw local source edits away
```

- `stash` (default) — auto-stash, pull, depois auto-restore das suas mudanças em cima do código atualizado. Nada se perde; se o restore bater em conflitos, eles ficam preservados num git stash para recuperação manual.
- `discard` — auto-stash e drop do stash depois do pull, para o update sempre cair numa árvore limpa. Use só em máquinas onde você nunca pretende manter edits locais no source do Hermes. É stash-drop (não `git reset --hard` + `git clean -fd`), então paths ignorados como `node_modules`, `venv` e outputs de build nunca são tocados.

No app desktop isso fica em **Settings → Advanced → In-App Update Local Changes**.

### Só preview: `hermes update --check`

Quer saber se há update disponível antes de puxar? Rode `hermes update --check` — ele faz fetch e compara commits com `origin/main`. Nenhum arquivo é modificado, nenhum gateway é reiniciado. Útil em scripts e jobs de cron que dependem de "tem update?".

### Backup completo pré-update: `--backup`

Para profiles de alto valor (gateways de produção, installs compartilhados de time) você pode optar por um backup completo pré-pull de `HERMES_HOME` (config, auth, sessões, skills, pairing):

```bash
hermes update --backup
```

Ou tornar isso o default em toda execução:

```yaml
# ~/.hermes/config.yaml
updates:
  pre_update_backup: full
```

`updates.pre_update_backup` é um único knob com três modos: `quick` (default — o snapshot leve de estado descrito acima), `full` (o snapshot quick mais um zip completo de `HERMES_HOME`; pode adicionar minutos em homes grandes) e `off` (sem backup pré-update — `--no-backup` faz o mesmo numa execução só). Valores booleanos legados ainda funcionam: `true` significa `full`, `false` significa `off`.

### Windows: outro `hermes.exe` está rodando

No Windows, o `hermes update` recusa rodar se detectar outro processo `hermes.exe` segurando o executável de entry-point do venv aberto — na maioria das vezes o backend spawnado pelo Hermes Desktop, um REPL `hermes` aberto em outro terminal, ou um gateway em execução:

```
$ hermes update
✗ Another hermes.exe is running:
    PID 12345  hermes.exe

  Updating now would fail to overwrite ...\venv\Scripts\hermes.exe because
  Windows blocks REPLACE on a running executable.

  Close Hermes Desktop, exit any open `hermes` REPLs, and
  stop the gateway (`hermes gateway stop`) before retrying.
  Override with `hermes update --force` if you've already
  confirmed those processes will not write to the venv.
```

Feche os processos listados e rode de novo. Se tiver certeza de que o processo concorrente não interfere (raro — em geral só útil quando um shim de antivírus é atribuído errado), passe `--force` para pular a checagem. Nesse caso o updater ainda tenta o rename do `.exe` com backoff exponencial e, em locks teimosos, agenda a substituição para o próximo reboot via `MoveFileEx(MOVEFILE_DELAY_UNTIL_REBOOT)` para o update completar.

Uma segunda guarda, separada, recusa mexer no venv enquanto qualquer processo estiver rodando a partir do interpretador Python dele (backend do Desktop, gateway, REPL Python). Esses processos mantêm arquivos de extensão nativa (`.pyd`) travados, e um sync de dependências que morre no meio com access-denied deixa a instalação entre versões. Essa guarda **não** é bypassada por `--force`; se tiver certeza de que os holders detectados são falso positivo, use o explícito `hermes update --force-venv`.

A saída esperada parece com:

```
$ hermes update
Updating Hermes Agent...
📥 Pulling latest code...
Already up to date.  (or: Updating abc1234..def5678)
📦 Updating dependencies...
✅ Dependencies updated
🔍 Checking for new config options...
✅ Config is up to date  (or: Found 2 new options — running migration...)
🔄 Restarting gateways...
✅ Gateway restarted
✅ Hermes Agent updated successfully!
```

### Validação recomendada pós-update

O `hermes update` cuida do caminho principal de update, mas uma validação rápida confirma que tudo caiu limpo:

1. `git status --short` — se a árvore estiver suja sem expectativa, inspecione antes de continuar
2. `hermes doctor` — checa config, dependências e saúde do serviço
3. `hermes --version` — confirme que a versão subiu como esperado
4. Se usar o gateway: `hermes gateway status`
5. Se o `doctor` apontar issues de npm audit: rode `npm audit fix` no diretório sinalizado

:::warning Working tree suja depois do update
Se `git status --short` mostrar mudanças inesperadas depois de `hermes update`, pare e inspecione antes de continuar. Em geral isso significa que modificações locais foram reaplicadas em cima do código atualizado, ou um passo de dependência refreshou lockfiles.
:::

### Se o terminal desconectar no meio do update

O `hermes update` se protege contra perda acidental de terminal:

- O update ignora `SIGHUP`, então fechar a sessão SSH ou a janela do terminal não mata mais no meio do install. Processos filhos `pip` e `git` herdam essa proteção, então o ambiente Python não fica semi-instalado por conexão caída.
- Toda a saída é espelhada em `~/.hermes/logs/update.log` enquanto o update roda. Se o terminal sumir, reconecte e inspecione o log para ver se o update terminou e se o restart do gateway deu certo:

```bash
tail -f ~/.hermes/logs/update.log
```

- `Ctrl-C` (SIGINT) e shutdown do sistema (SIGTERM) ainda são honrados — esses são cancelamentos deliberados, não acidentes.

Você não precisa mais embrulhar `hermes update` em `screen` ou `tmux` para sobreviver a uma queda de terminal.

### Conferindo a versão atual

```bash
hermes version
```

Compare com o release mais recente na [página de releases do GitHub](https://github.com/NousResearch/hermes-agent/releases).

### Atualizando a partir de plataformas de messaging

Você também pode atualizar direto do Telegram, Discord, Slack, WhatsApp ou Teams mandando:

```
/update
```

Isso puxa o código mais recente, atualiza dependências e reinicia gateways em execução. O bot fica offline por pouco tempo durante o restart (tipicamente 5–15 segundos) e depois volta.

### Update manual

Se você instalou na mão (não pelo instalador rápido):

```bash
cd /path/to/hermes-agent
# Activate the venv you created during install (outside the source tree)
export VIRTUAL_ENV="$HOME/.hermes/venvs/hermes-dev"
export PATH="$VIRTUAL_ENV/bin:$PATH"

# Pull latest code
git pull origin main

# Reinstall (picks up new dependencies)
uv pip install -e ".[all]"

# Check for new config options
hermes config check
hermes config migrate   # Interactively add any missing options
```

### Instruções de rollback

Se um update introduzir um problema, você pode voltar a uma versão anterior:

```bash
cd /path/to/hermes-agent

# List recent versions
git log --oneline -10

# Roll back to a specific commit
git checkout <commit-hash>
uv pip install -e ".[all]"

# Restart the gateway if running
hermes gateway restart
```

Para voltar a uma tag de release específica (substitua pela tag anterior — ex.: um release recente como `v2026.5.16`, ou qualquer tag anterior de `git tag --sort=-version:refname`):

```bash
git checkout vX.Y.Z
uv pip install -e ".[all]"
```

:::warning
Rollback pode causar incompatibilidades de config se opções novas foram adicionadas. Rode `hermes config check` depois do rollback e remova opções não reconhecidas de `config.yaml` se encontrar erros.
:::

### Nota para usuários Nix

O Nix não é mais um caminho de instalação oficialmente suportado (só best-effort) — veja [Nix Setup](./nix-setup.md). Se você instalou via flake Nix, updates são gerenciados pelo package manager Nix:

```bash
# Update the flake input
nix flake update hermes-agent

# Or rebuild with the latest
nix profile upgrade hermes-agent
```

Instalações Nix são imutáveis — rollback é feito pelo sistema de gerações do Nix:

```bash
nix profile rollback
```

Veja [Nix Setup](./nix-setup.md) para mais detalhes.

---

## Desinstalando

```bash
hermes uninstall
```

O desinstalador oferece a opção de manter seus arquivos de configuração (`~/.hermes/`) para uma reinstalação futura.

### Desinstalação manual

```bash
rm -f ~/.local/bin/hermes
rm -rf /path/to/hermes-agent
rm -rf ~/.hermes            # Optional — keep if you plan to reinstall
```

:::info
Se instalou o gateway como serviço do sistema, pare e desabilite primeiro:
```bash
hermes gateway stop
# Linux: systemctl --user disable hermes-gateway
# macOS: launchctl remove ai.hermes.gateway
```
:::
