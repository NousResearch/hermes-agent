---
sidebar_position: 3
title: "Atualização e Desinstalação"
description: "Como atualizar o Hermes Agent para a versão mais recente ou desinstalá-lo"
---

# Atualização e Desinstalação

## Atualização

Atualize para a versão mais recente com um único comando:

```bash
hermes update
```

Isso puxa o código mais recente do `main`, atualiza as dependências e solicita que você configure quaisquer novas opções que foram adicionadas desde sua última atualização.

:::tip
`hermes update` detecta automaticamente novas opções de configuração e solicita que você as adicione. Se você pulou essa solicitação, execute manualmente `hermes config check` para ver as opções faltantes, depois `hermes config migrate` para adicioná-las interativamente.
:::

### O que acontece durante uma atualização

Quando você executa `hermes update`, os seguintes passos ocorrem:

1. **Snapshot pré-atualização** — um snapshot de estado leve é salvo por padrão (cobre dados de pareamento, cron jobs, `config.yaml`, `.env`, `auth.json` e outros arquivos de estado que são modificados em tempo de execução; arquivos individuais acima de 1 GiB são pulados para que um banco de dados grande de sessões nunca atrase a atualização). Controlado por `updates.pre_update_backup` (`quick` por padrão, `full` para um zip de todo o `HERMES_HOME`, `off` para desabilitar). Recuperável via o fluxo de restauração de snapshot descrito em [Snapshots e rollback](../user-guide/checkpoints-and-rollback.md).
2. **Git pull** — puxa o código mais recente do branch `main` e atualiza submódulos
3. **Validação de sintaxe pós-pull + rollback automático** — após o pull, o Hermes compila os oito arquivos críticos que toda invocação do `hermes` importa na inicialização. Se algum falhar ao fazer parse (ex.: um marcador de conflito de merge órfão, um arquivo acidentalmente truncado), o Hermes executa `git reset --hard <pre-pull-sha>` para reverter a instalação para que seu shell permaneça inicializável. Execute `hermes update` novamente quando a correção upstream chegar.
4. **Instalação de dependências** — executa `uv pip install -e ".[all]"` para incorporar dependências novas ou alteradas
5. **Migração de configuração** — detecta novas opções de configuração adicionadas desde sua versão e solicita que você as defina
6. **Reinício automático do gateway** — gateways em execução são atualizados após a conclusão da atualização para que o novo código entre em vigor imediatamente. Gateways gerenciados por serviço (systemd no Linux, launchd no macOS) são reiniciados através do gerenciador de serviços. Gateways manuais são relançados automaticamente quando o Hermes consegue mapear o PID em execução de volta a um profile.

### Atualização contra um branch não padrão: `--branch`

Por padrão `hermes update` rastreia `origin/main`. Passe `--branch <nome>` para atualizar contra um branch diferente — útil para canais de QA, branches de funcionalidade ou teste de candidatos a lançamento:

```bash
hermes update --branch release-candidate
hermes update --check --branch experimental   # apenas pré-visualiza o atraso
```

Se seu checkout local está em um branch diferente, o Hermes faz auto-stash de qualquer trabalho não commitado, muda o HEAD para o branch alvo e então puxa. Branches que não existem localmente são rastreados automaticamente de `origin/<name>` (`git checkout -B <name> origin/<name>`). Branches que não existem em lugar algum falham limparmente — suas alterações stash são restauradas antes da saída para que você nunca fique preso em um estado estranho. A lógica de sincronização fork-upstream exclusiva do `main` é automaticamente pulada em branches que não são `main`.

### Alterações locais em atualizações não interativas

Quando você executa `hermes update` em um terminal, o Hermes faz stash de quaisquer alterações não commitadas na árvore de código, puxa, e então **pergunta** se deve restaurá-las — exatamente como sempre fez. Nada muda para atualizações interativas.

Quando a atualização é executada **sem um terminal** — pelo botão "Atualizar" do aplicativo desktop/chat ou uma atualização disparada pelo gateway — não há solicitação para responder. A configuração `updates.non_interactive_local_changes` decide o que acontece com suas alterações stash:

```yaml
# ~/.hermes/config.yaml
updates:
  non_interactive_local_changes: stash   # padrão: manter + restaurar automaticamente
  # non_interactive_local_changes: discard  # descartar edições locais do código fonte
```

- `stash` (padrão) — auto-stash, puxar, depois restaurar automaticamente suas alterações sobre o código atualizado. Nada é perdido; se uma restauração encontrar conflitos, eles são preservados em um git stash para recuperação manual.
- `discard` — auto-stash e descartar o stash após o pull, para que a atualização sempre termine em uma árvore limpa. Use isto apenas em máquinas onde você nunca pretende manter edições locais no código fonte do Hermes. Ele faz stash-drop (não `git reset --hard` + `git clean -fd`), então caminhos ignorados como `node_modules`, `venv` e saídas de build nunca são tocados.

No aplicativo desktop, isso está em **Settings → Advanced → In-App Update Local Changes**.

### Apenas pré-visualização: `hermes update --check`

Quer saber se uma atualização está disponível antes de puxar? Execute `hermes update --check` — ele busca e compara commits contra `origin/main`. Nenhum arquivo é modificado, nenhum gateway é reiniciado. Útil em scripts e cron jobs que verificam "há atualização".

### Backup completo pré-atualização: `--backup`

Para profiles de alto valor (gateways de produção, instalações de equipe compartilhadas) você pode optar por um backup completo pré-pull do `HERMES_HOME` (config, auth, sessões, skills, pareamento):

```bash
hermes update --backup
```

Ou torne isso o padrão para toda execução:

```yaml
# ~/.hermes/config.yaml
updates:
  pre_update_backup: full
```

`updates.pre_update_backup` é um único controle com três modos: `quick` (padrão — o snapshot de estado leve descrito acima), `full` (o snapshot rápido mais um zip completo do `HERMES_HOME`; pode adicionar minutos em homes grandes) e `off` (nenhum backup pré-atualização — `--no-backup` faz o mesmo para uma única execução). Valores booleanos legados ainda funcionam: `true` significa `full`, `false` significa `off`.

### Windows: outro `hermes.exe` está em execução

No Windows, `hermes update` se recusará a executar se detectar outro processo `hermes.exe` segurando o executável de ponto de entrada do venv aberto — mais comumente o backend do aplicativo Hermes Desktop, um REPL `hermes` aberto em outro terminal, ou um gateway em execução:

```
$ hermes update
✗ Outro hermes.exe está em execução:
    PID 12345  hermes.exe

  Atualizar agora falharia ao sobrescrever ...\venv\Scripts\hermes.exe porque
  o Windows bloqueia REPLACE em um executável em execução.

  Feche o Hermes Desktop, saia de qualquer REPL `hermes` aberto e
  pare o gateway (`hermes gateway stop`) antes de tentar novamente.
  Substitua com `hermes update --force` se você já
  confirmou que esses processos não escreverão no venv.
```

Feche os processos listados e execute novamente. Se você tem certeza de que o processo concorrente não interferirá (raro — geralmente útil apenas quando um shim de antivírus é mal-atribuído), passe `--force` para pular a verificação. Nesse caso, o atualizador ainda tentará a renomeação do `.exe` com backoff exponencial e, em locks teimosos, agendará a substituição para a próxima reinicialização via `MoveFileEx(MOVEFILE_DELAY_UNTIL_REBOOT)` para que a atualização possa ser concluída.

Uma segunda proteção separada recusa-se a tocar no venv enquanto qualquer processo estiver em execução a partir de seu interpretador Python (o backend do aplicativo Desktop, um gateway, um REPL Python). Esses processos mantêm arquivos de extensão nativa (`.pyd`) bloqueados, e uma sincronização de dependências que morre no meio por um erro de acesso negado deixa a instalação entre versões. Esta proteção **não** é contornada por `--force`; se você tem certeza de que os detentores detectados são falsos positivos, use o explícito `hermes update --force-venv`.

A saída esperada se parece com:

```
$ hermes update
Atualizando o Hermes Agent...
📥 Puxando código mais recente...
Already up to date.  (ou: Updating abc1234..def5678)
📦 Atualizando dependências...
✅ Dependências atualizadas
🔍 Verificando novas opções de configuração...
✅ Configuração atualizada  (ou: Found 2 new options — executando migração...)
🔄 Reiniciando gateways...
✅ Gateway reiniciado
✅ Hermes Agent atualizado com sucesso!
```

### Validação Pós-Atualização Recomendada

`hermes update` lida com o caminho principal de atualização, mas uma validação rápida confirma que tudo chegou limpo:

1. `git status --short` — se a árvore estiver inesperadamente suja, inspecione antes de continuar
2. `hermes doctor` — verifica config, dependências e saúde do serviço
3. `hermes --version` — confirme que a versão subiu como esperado
4. Se você usa o gateway: `hermes gateway status`
5. Se `doctor` relatar problemas de npm audit: execute `npm audit fix` no diretório sinalizado

:::warning Árvore de trabalho suja após atualização
Se `git status --short` mostrar alterações inesperadas após `hermes update`, pare e inspecione-as antes de continuar. Isso geralmente significa que modificações locais foram reaplicadas sobre o código atualizado, ou que uma etapa de dependência atualizou lockfiles.
:::

### Se seu terminal desconectar no meio da atualização

`hermes update` se protege contra perda acidental de terminal:

- A atualização ignora `SIGHUP`, então fechar sua sessão SSH ou janela de terminal não a mata no meio da instalação. Processos filhos `pip` e `git` herdam esta proteção, então o ambiente Python não pode ser deixado meio-instalado por uma conexão perdida.
- Toda a saída é espelhada para `~/.hermes/logs/update.log` enquanto a atualização é executada. Se seu terminal desaparecer, reconecte e inspecione o log para ver se a atualização terminou e se o reinício do gateway foi bem-sucedido:

```bash
tail -f ~/.hermes/logs/update.log
```

- `Ctrl-C` (SIGINT) e desligamento do sistema (SIGTERM) ainda são honrados — esses são cancelamentos deliberados, não acidentes.

Você não precisa mais envolver `hermes update` em `screen` ou `tmux` para sobreviver a uma queda de terminal.

### Verificando sua versão atual

```bash
hermes version
```

Compare com o lançamento mais recente na [página de releases do GitHub](https://github.com/NousResearch/hermes-agent/releases).

### Atualização a partir de Plataformas de Mensagem

Você também pode atualizar diretamente do Telegram, Discord, Slack, WhatsApp ou Teams enviando:

```
/update
```

Isso puxa o código mais recente, atualiza dependências e reinicia gateways em execução. O bot ficará offline brevemente durante o reinício (tipicamente 5–15 segundos) e então retomará.

### Atualização Manual

Se você instalou manualmente (não via o instalador rápido):

```bash
cd /caminho/para/hermes-agent
# Ative o venv que você criou durante a instalação (fora da árvore de código)
export VIRTUAL_ENV="$HOME/.hermes/venvs/hermes-dev"
export PATH="$VIRTUAL_ENV/bin:$PATH"

# Puxe o código mais recente
git pull origin main

# Reinstale (incorpora novas dependências)
uv pip install -e ".[all]"

# Verifique novas opções de configuração
hermes config check
hermes config migrate   # Adicione interativamente quaisquer opções faltantes
```

### Instruções de rollback

Se uma atualização introduzir um problema, você pode reverter para uma versão anterior:

```bash
cd /caminho/para/hermes-agent

# Liste versões recentes
git log --oneline -10

# Reverta para um commit específico
git checkout <commit-hash>
uv pip install -e ".[all]"

# Reinicie o gateway se estiver em execução
hermes gateway restart
```

Para reverter para uma tag de release específica (substitua pela sua tag anterior — ex.: um release recente como `v2026.5.16`, ou qualquer tag anterior de `git tag --sort=-version:refname`):

```bash
git checkout vX.Y.Z
uv pip install -e ".[all]"
```

:::warning
Reverter pode causar incompatibilidades de configuração se novas opções foram adicionadas. Execute `hermes config check` após reverter e remova quaisquer opções não reconhecidas de `config.yaml` se você encontrar erros.
:::

### Nota para usuários Nix

Nix não é mais um caminho de instalação explicitamente suportado (apenas melhor esforço) — veja [Nix Setup](./nix-setup.md). Se você instalou via Nix flake, as atualizações são gerenciadas através do gerenciador de pacotes Nix:

```bash
# Atualize a entrada do flake
nix flake update hermes-agent

# Ou reconstrua com o mais recente
nix profile upgrade hermes-agent
```

Instalações Nix são imutáveis — o rollback é tratado pelo sistema de gerações do Nix:

```bash
nix profile rollback
```

Veja [Nix Setup](./nix-setup.md) para mais detalhes.

---

## Desinstalação

```bash
hermes uninstall
```

O desinstalador oferece a opção de manter seus arquivos de configuração (`~/.hermes/`) para uma reinstalação futura.

### Desinstalação Manual

```bash
rm -f ~/.local/bin/hermes
rm -rf /caminho/para/hermes-agent
rm -rf ~/.hermes            # Opcional — mantenha se você planeja reinstalar
```

:::info
Se você instalou o gateway como um serviço do sistema, pare e desabilite-o primeiro:
```bash
hermes gateway stop
# Linux: systemctl --user disable hermes-gateway
# macOS: launchctl remove ai.hermes.gateway
```
:::
