---
title: Senhas e Logins
description: O agente entra em sites, paga e preenche endereços por você sem nunca ver uma senha.
---

# Senhas e Logins

Diga **"log into GitHub"** e o agente entra por você. Na primeira vez que
chega a uma página de login para a qual não tem credencial, ele pergunta ali
mesmo, num prompt mascarado. Depois disso, simplesmente funciona. As senhas
ficam criptografadas nesta máquina e são injetadas direto na página; o modelo
nunca as vê.

Não há nada a configurar.

## Como parece {#what-it-looks-like}

**CLI / TUI**

```
🔐 Save login for github.com
   The agent reached a sign-in page with no saved login for this site.
   Type the email / username you sign in with (shown), then Enter.
   ...
   Now the password (hidden). It is encrypted on this machine, bound to
   https://github.com, and filled into the page without the model ever seeing it.
```

**Desktop** — um card "Save your github.com login?" com um campo de
identificador e um campo de senha mascarado. *Save & sign in* guarda e
continua; *Don't save* diz ao agente para parar de perguntar neste turno.

Dali em diante o agente lista seus logins salvos, digita o identificador e
preenche a senha pelo Hermes. O resultado da tool que ele vê é
`{filled_fields: 1, origin: "https://github.com"}`; a senha também é
registrada no redactor para que uma leitura posterior da página não a ecoe
de volta.

## Códigos de dois fatores {#two-factor-codes}

Sites que pedem um código depois da senha são tratados da mesma forma:

- **Chave do autenticador salva com o login** (a "setup key" ou link
  `otpauth://` que um site mostra ao habilitar 2FA; itens do 1Password e
  Bitwarden que guardam uma seed TOTP também contam): o Hermes gera o código
  atual e o digita. Ninguém é perguntado. Adicione a chave em **Settings →
  Passwords & Logins → Add** ou `hermes vault add`; o item mostra um badge
  *2FA auto*.
- **Código enviado para o telefone ou e-mail**: um prompt pequeno aparece na
  sua superfície ("Verification code for github.com"), você digita o código,
  o Hermes o coloca na página. O código também nunca entra na conversa.
- **Passkeys, chaves de hardware, aprovações no app** ("tap Approve in Duo"):
  nada para digitar. O agente pede para você completar no dispositivo e
  espera a página seguir.

## Já usa 1Password ou Bitwarden? {#already-using-1password-or-bitwarden}

Nada a habilitar. Se a ferramenta de linha de comando `op` ou `bw` estiver
instalada e logada, o Hermes a detecta automaticamente e os logins de site
dela passam a ser preenchíveis junto com os locais. Na primeira vez que o
agente precisa de um desses logins, pede para desbloquear o gerenciador com a
master password (prompt mascarado; uma vez por sessão, 30 minutos de idle).
O Hermes entrega a master password ao CLI do gerenciador pelo canal não
interativo (`op signin` no stdin, `bw unlock --passwordenv` no ambiente do
filho) e guarda só o token de sessão na memória. O agente nunca vê a master
password, o token nem nenhum login.

Prefere não usar um gerenciador detectado? `hermes vault sources --disable bitwarden`,
ou o switch em **Settings → Passwords & Logins**.

## Pagando e preenchendo endereços {#paying-and-filling-addresses}

Cartões e endereços funcionam como logins: salvos uma vez (**Settings →
Passwords & Logins → Add**, ou `hermes vault add`), vinculados ao site de
checkout, e preenchidos pelo agente só naquele site. **Todo preenchimento de
cartão pergunta primeiro**, com o mesmo prompt de aprovação de um comando
perigoso; recusar não grava nada. Sessões headless (cron, webhooks, o API
server) não podem confirmar e são recusadas, então uma prompt injection que
chegue a uma página de checkout pode pedir, mas não pode gastar.
Preenchimentos de endereço não precisam de confirmação.

## Gerenciando o que está salvo {#managing-whats-saved}

- **Desktop → Settings → Passwords & Logins**: tudo salvo, os gerenciadores
  de senha detectados com Unlock/Lock, Add, Remove.
- **CLI**: `hermes vault list`, `hermes vault add`, `hermes vault rm <handle>`,
  `hermes vault sources`.

Os itens ficam criptografados em `~/.hermes/vault/` (chave Fernet + arquivo
do vault, ambos `0600`), no escopo do profile. Labels, origins de site e
identificadores de login são metadados visíveis; senhas e valores de cartão
nunca saem do vault exceto para a página.

## Sessões headless {#headless-sessions}

Jobs de cron, webhooks, o API server e `hermes chat -q` não têm ninguém para
responder a um prompt. Logins locais salvos continuam funcionando aí; um
gerenciador de senha travado reporta `unavailable_in_this_session` e um login
ausente reporta `prompt_unavailable`. Desbloqueie ou salve numa sessão
interativa primeiro, ou dê ao 1Password um token de service account
(`OP_SERVICE_ACCOUNT_TOKEN`).

```yaml
vault:
  onepassword:
    enabled: false          # opt OUT of a detected manager (default: on when installed)
    account: ""             # `op --account` shorthand; empty = default
    service_account_token_env: OP_SERVICE_ACCOUNT_TOKEN
  bitwarden:
    enabled: false
```

## O que isto garante e o que não garante {#what-this-does-and-does-not-guarantee}

**Garante:** a senha nunca entra no contexto do modelo pelo Hermes: nem em
resultados de tool, logs, o banco de sessões, nem nos argumentos de CLI de
qualquer processo. Preenchimentos acontecem pelo socket CDP direto da sessão
de browser supervisionada e são recusados a menos que o origin da página
bata exatamente com o origin salvo, verificado de novo dentro da página
imediatamente antes da escrita.

**Não garante:** proteção contra a própria página. Depois que uma senha é
digitada num site, aquele site (e qualquer script que ele rode) a tem,
exatamente como quando você digita. Num backend de cloud browser o browser do
vendor vê a página como qualquer outro. O binding de origin é a guarda contra
preencher no site errado, não contra um site certo comprometido.
