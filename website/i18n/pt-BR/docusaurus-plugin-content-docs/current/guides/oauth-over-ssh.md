---
sidebar_position: 17
title: "OAuth via SSH / Hosts Remotos"
description: "Como concluir o OAuth baseado em navegador (Spotify, servidores MCP) quando o Hermes é executado em uma máquina remota, contêiner ou por trás de um jump box"
---

# OAuth via SSH / Hosts Remotos {#oauth-over-ssh--remote-hosts}

Alguns provedores do Hermes — **Spotify** e **servidores MCP remotos** (Linear, Sentry, Atlassian, Asana, Figma, …) — usam um fluxo OAuth de *redirecionamento loopback*. O servidor de autenticação redireciona seu navegador para `http://127.0.0.1:<port>/callback`, para que um pequeno listener HTTP iniciado pelo Hermes capture o código de autorização.

Isso funciona perfeitamente quando o Hermes e seu navegador estão na mesma máquina. Isso quebra no momento em que não estão: o navegador do seu laptop tenta acessar `127.0.0.1` no **seu laptop**, mas o listener está vinculado a `127.0.0.1` no **servidor remoto**.

A correção é um encaminhamento local de SSH em uma linha. Para servidores MCP em um terminal interativo, muitas vezes você pode simplesmente colar a URL de redirecionamento de volta em vez de usar um túnel.

**O xAI Grok OAuth (`xai-oauth`) usa código de dispositivo OAuth**, não um callback loopback — abra a URL de verificação impressa em qualquer navegador e o Hermes faz polling até a aprovação. Nenhum túnel SSH é necessário. Veja [xAI Grok OAuth](./xai-grok-oauth.md).

## TL;DR {#tldr}

```bash
# Na sua máquina local (laptop), em um terminal separado:
ssh -N -L 43827:127.0.0.1:43827 user@remote-host

# Na sua sessão SSH existente na máquina remota:
hermes auth add spotify --no-browser
# → O Hermes imprime uma URL de autorização. Abra-a em um navegador no seu laptop.
# → Seu navegador redireciona para 127.0.0.1:43827/callback, o túnel encaminha
#   a requisição para o listener remoto, o login é concluído.
```

O Hermes imprime a porta exata em que se vinculou na linha `Waiting for callback on ...` — copie-a de lá. O Spotify usa por padrão a porta `43827`.

## Quais Provedores Precisam Disso {#which-providers-need-this}

| Provedor | Porta loopback | Túnel necessário? |
|----------|---------------|----------------|
| Spotify | `43827` (padrão) | Sim, quando o Hermes está remoto |
| Servidores MCP (`auth: oauth`) | escolhida automaticamente por servidor | Sim, quando o Hermes está remoto (ou cole a URL de redirecionamento) |
| `xai-oauth` (Grok SuperGrok) | n/a | Não — fluxo de código de dispositivo |
| `anthropic` (Claude Pro/Max) | n/a | Não — fluxo de colar o código |
| `openai-codex` (ChatGPT Plus/Pro) | n/a | Não — fluxo de código de dispositivo |
| `minimax`, `nous-portal` | n/a | Não — fluxo de código de dispositivo |

Se seu provedor não estiver na tabela, você não precisa de um túnel.

## Servidores MCP {#mcp-servers}

Servidores MCP remotos (Linear, Sentry, Atlassian, Asana, Figma, etc.) usam o mesmo fluxo de redirecionamento loopback. O Hermes escolhe automaticamente uma porta livre por servidor e imprime a URL de autorização quando o fluxo OAuth é iniciado — seja na inicialização (quando um novo servidor aparece em `mcp_servers:`) ou quando você executa `hermes mcp login <server>`.

Você tem duas formas de concluir isso a partir de uma máquina remota:

**Opção 1 — cole a URL de redirecionamento de volta (sem configuração, funciona em qualquer lugar).** Em um terminal interativo, o Hermes solicita que você cole a URL de redirecionamento junto com a execução do listener local. Depois de aprovar no seu navegador, o redirecionamento para `http://127.0.0.1:<port>/callback` mostrará um erro de conexão — isso é esperado. Copie a **URL completa da barra de endereço do navegador** e cole no prompt do Hermes:

```
  MCP OAuth: authorization required.
  Open this URL in your browser:

    https://mcp.linear.app/authorize?response_type=code&...

  Or paste the redirect URL here (or the ?code=...&state=... portion) and press Enter:
> https://mcp.linear.app/callback?code=abc123&state=xyz
  Got authorization code from paste — completing flow.
```

Uma simples query string `?code=...&state=...` também é aceita. Isso funciona para qualquer servidor MCP com `auth: oauth` e não requer nenhuma alteração de configuração do SSH.

**Opção 2 — encaminhamento de porta SSH (igual ao Spotify).** O Hermes imprime a porta exata em que se vinculou na dica da sessão SSH. Abra um terminal separado no seu laptop:

```bash
ssh -N -L <port>:127.0.0.1:<port> user@remote-host
```

Depois abra a URL de autorização no seu navegador normalmente; o redirecionamento passa pelo túnel e o listener o captura. Use isso quando você precisar que o fluxo seja concluído sem intervenção (por exemplo, uma re-autenticação roteirizada em que você não pode colar interativamente).

**Cuidado — a corrida dos 30s de recarregamento de configuração.** Se você editar `~/.hermes/config.yaml` para adicionar um servidor MCP OAuth de dentro de uma sessão do Hermes em execução, a CLI recarrega automaticamente as conexões MCP com um timeout de 30s. Isso não é tempo suficiente para concluir um fluxo OAuth interativo, e o recarregamento vai desistir. Use `hermes mcp login <server>` de um terminal novo em vez disso — ele não tem esse limite e espera os 5 minutos completos para você colar de volta.

## Por que o listener não pode simplesmente se vincular a 0.0.0.0 {#why-the-listener-cant-just-bind-000}

O Spotify e a maioria dos servidores OAuth do MCP validam o parâmetro `redirect_uri` contra uma lista de permissões. Ambos exigem a forma loopback (`http://127.0.0.1:<exact-port>/callback`). Vincular o listener a `0.0.0.0` ou a uma porta diferente faria o servidor de autenticação rejeitar a requisição como uma incompatibilidade de redirect_uri. O túnel SSH mantém a URI loopback intacta de ponta a ponta.

## Passo a passo: salto único de SSH {#step-by-step-single-ssh-hop}

### 1. Inicie o túnel a partir da sua máquina local {#1-start-the-tunnel-from-your-local-machine}

```bash
# Spotify (porta 43827)
ssh -N -L 43827:127.0.0.1:43827 user@remote-host
```

`-N` significa "não abra um shell remoto, apenas mantenha o túnel aberto." Mantenha esse terminal em execução durante todo o login.

### 2. Em uma sessão SSH separada, execute o comando de autenticação {#2-in-a-separate-ssh-session-run-the-auth-command}

```bash
ssh user@remote-host
hermes auth add spotify --no-browser
```

O Hermes detecta a sessão SSH, pula a abertura automática do navegador e imprime uma URL de autorização, além de uma linha `Waiting for callback on http://127.0.0.1:<port>/callback`.

### 3. Abra a URL no seu navegador local {#3-open-the-url-in-your-local-browser}

Copie a URL de autorização do terminal remoto e cole no navegador do seu laptop. Aprove a tela de consentimento. O servidor de autenticação redireciona para `http://127.0.0.1:<port>/callback`. Seu navegador acessa o túnel, a requisição é encaminhada para o listener remoto, e o Hermes imprime `Login successful!`.

Você pode encerrar o túnel (Ctrl+C no primeiro terminal) depois de ver a linha de sucesso.

## Passo a passo: através de um jump box {#step-by-step-through-a-jump-box}

Se você acessa o Hermes por meio de um bastion / jump host, use o `-J` (ProxyJump) nativo do SSH:

```bash
ssh -N -L 43827:127.0.0.1:43827 -J jump-user@jump-host user@final-host
```

Isso encadeia uma conexão SSH através do jump host sem colocar a porta loopback no próprio jump box. O `127.0.0.1:43827` local no seu laptop faz o túnel diretamente para `127.0.0.1:43827` na máquina remota final.

Para OpenSSH mais antigo que não suporta `-J`, a forma longa é:

```bash
ssh -N \
    -o "ProxyCommand=ssh -W %h:%p jump-user@jump-host" \
    -L 43827:127.0.0.1:43827 \
    user@final-host
```

## Mosh, tmux, ssh ControlMaster {#mosh-tmux-ssh-controlmaster}

O túnel é uma propriedade da conexão SSH subjacente. Se você estiver executando o Hermes dentro do `tmux` sobre uma sessão mosh, o roaming do mosh não leva consigo o encaminhamento `-L`. Abra uma sessão SSH simples *separada* **apenas** para o túnel `-L` — essa é a conexão que precisa ficar viva durante o fluxo de autenticação. Sua sessão interativa mosh/tmux pode continuar executando o Hermes normalmente.

Se você usar `ssh -o ControlMaster=auto`, os encaminhamentos de porta em uma conexão multiplexada compartilham o ciclo de vida do master. Reinicie o master se o túnel não subir:

```bash
ssh -O exit user@remote-host
ssh -N -L 43827:127.0.0.1:43827 user@remote-host
```

## Solução de Problemas {#troubleshooting}

### `bind [127.0.0.1]:43827: Address already in use` {#bind-1270013827-address-already-in-use}

Algo no seu laptop já está usando essa porta. Ou o túnel anterior não foi encerrado corretamente, ou um Hermes local também está escutando nela. Encontre e finalize o processo responsável:

```bash
# macOS / Linux
lsof -iTCP:43827 -sTCP:LISTEN
kill <PID>
```

Depois tente novamente o comando `ssh -L`.

### A autorização expira esperando pelo callback local {#authorization-timed-out-waiting-for-the-local-callback}

O redirecionamento nunca chegou ao listener remoto. Verifique se o túnel ainda está ativo (`ssh -N` não mostra saída, então observe o terminal de onde você o iniciou), confirme que você usou a porta da última linha `Waiting for callback on ...` (o Hermes pode incrementar automaticamente se a porta preferida estiver ocupada), reinicie o túnel se necessário e execute o comando de autenticação novamente.

### Os tokens vão para o `~/.hermes` errado {#tokens-land-in-the-wrong-hermes}

Os tokens são gravados sob o usuário Linux que executou `hermes auth add ...`. Se o seu gateway / serviço systemd for executado como um usuário diferente (por exemplo, `root` ou um usuário `hermes` dedicado), autentique-se como **esse** usuário para que os tokens sejam gravados no `~/.hermes/auth.json` dele. `sudo -u hermes -i` ou equivalente.

## Veja Também {#see-also}

- [xAI Grok OAuth](./xai-grok-oauth.md) — código de dispositivo; sem túnel SSH
- [Spotify (`Running over SSH`)](../user-guide/features/spotify.md#running-over-ssh--in-a-headless-environment)
- [Native MCP client (OAuth section)](../user-guide/features/mcp.md#oauth-authenticated-http-servers)
- [SSH `-J` / ProxyJump (man page)](https://man.openbsd.org/ssh#J)
