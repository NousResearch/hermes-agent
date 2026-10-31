---
sidebar_position: 18
title: "Login Nativo do Desktop (RFC 8252)"
description: "Como o app Hermes Desktop faz login em um gateway protegido usando o navegador do sistema e PKCE — sem webview embutida, sem cookies de sessão"
---

# Login Nativo do Desktop (RFC 8252)

Quando o app Hermes Desktop se conecta a um **gateway protegido** (um dashboard hospedado ou
auto-hospedado que fica atrás de um provedor OAuth), ele pode fazer login de duas
formas:

1. **Login nativo (RFC 8252)** — o app abre seu **navegador real do sistema**,
   você aprova no navegador em que já confia, e o app recebe tokens que
   armazena como arquivos owner-only no diretório user-data do app (opcionalmente criptografados
   com o keychain do SO — Settings → Gateway). **Sem webview embutida, sem
   cookies de sessão do navegador.** Este é o padrão sempre que o gateway
   suporta.
2. **Login embutido (fallback legado)** — o app abre uma pequena janela de navegador
   embutida no app e captura o cookie de sessão do gateway. Usado automaticamente
   quando o gateway é uma versão mais antiga que não anuncia o login nativo.

Você não escolhe entre esses dois — o app detecta o que o gateway suporta e
escolhe o melhor. Esta página explica o que acontece e por quê.

## Por que login nativo {#why-native-sign-in}

Embutir um navegador dentro de um app nativo para OAuth tem desvantagens bem conhecidas:
a página de login não consegue ver sua sessão de navegador já existente (então você
redigita credenciais e refaz o MFA), gerenciadores de senha e passkeys geralmente não funcionam,
e o app depende de ler um cookie de sessão de dentro de uma webview privada. A RFC
8252 ("OAuth 2.0 for Native Apps") é a melhor prática da indústria que evita
tudo isso: **fazer a autorização no navegador do sistema e entregar ao app seus
próprios tokens.**

Especificamente para o Hermes, o login nativo significa:

- **Sem webview embutida.** A autorização acontece no Safari / Chrome /
  Firefox / Edge — o que você usar — com seus logins, extensões e
  passkeys intactos.
- **Sem cookies de sessão.** O app mantém um **access token** OAuth (de vida curta)
  e um **refresh token**, armazenados como arquivos owner-only — criptografados at-rest via
  keychain do SO (`safeStorage` do Electron) quando o toggle opt-in de keychain em
  Settings → Gateway está ligado. Chamadas REST e tickets de WebSocket são autenticados com um
  header `Authorization: Bearer`, não com um cookie jar.

## Como funciona {#how-it-works}

```
Desktop app                Gateway (/auth/native/*)          Nous Portal (IDP)
   │ 1. open loopback 127.0.0.1:<random port>
   │ 2. system browser ─►  /auth/native/authorize
   │    (PKCE challenge)    (starts the normal PKCE login) ─► /oauth/authorize
   │                        ◄──── code ──── /auth/callback ◄──┘
   │                        3. mint one-time gateway code
   │ ◄─ 302 127.0.0.1/cb?code=… ─┘
   │ 4. POST /auth/native/token (code + PKCE verifier)
   │ ◄─ 5. { access_token, refresh_token, expires_at } ───────┘
   │ 6. store in local token store; use Bearer for REST + WS tickets
```

O gateway **intermedia** o fluxo: ele é o servidor de autorização *para o
app desktop* e um cliente OAuth *para o provedor de identidade upstream* (Nous
Portal). Isso é necessário porque o `client_id` upstream e as URIs de redirecionamento
permitidas estão vinculados à própria origem do gateway — um app desktop não pode ser um
cliente direto do Portal. O desktop ainda obtém a experiência RFC 8252 completa: seu
próprio par PKCE, seu próprio redirecionamento loopback e tokens que ele mesmo possui.

O **PKCE (RFC 7636)** protege o salto pelo loopback: o código de gateway de uso único é
inútil sem o verificador de código, que nunca sai do app. O código é
de uso único e de vida curta.

## Detecção de capacidade e fallback {#capability-detection--fallback}

O desktop lê o endpoint público `/api/status` do gateway, que anuncia
um array `auth_flows`:

| Valor de `auth_flows` | Significado |
|--------------------|---------|
| `["cookie", "native_pkce"]` | O gateway suporta login nativo → o app o usa |
| `["cookie"]` | O gateway suporta apenas o fluxo legado → o app usa a webview embutida |
| *(campo ausente)* | Gateway mais antigo → o app usa a webview embutida |

Se o login nativo é anunciado mas falha por um motivo local — ex.: uma ferramenta
de segurança bloqueia o listener loopback, ou você fecha a aba do navegador — o app
**cai automaticamente de volta para o fluxo embutido** para que você ainda consiga fazer login.

## Ciclo de vida do token {#token-lifecycle}

- **Access token**: de vida curta (minutos). Enviado como `Authorization: Bearer` em
  toda chamada REST e ao emitir um ticket de WebSocket.
- **Refresh token**: de vida mais longa, rotativo. Quando o access token está perto de
  expirar, o app chama `/auth/native/refresh` para rotacionar ambos os tokens, então
  atualiza seu token store.
- **Expiração terminal**: se o refresh token está morto (expirado / revogado / com
  reuso detectado), o app limpa seus tokens armazenados e solicita um novo
  login.
- **Sign out**: limpa tanto os tokens nativos armazenados quanto qualquer cookie de sessão
  legado para aquele gateway.

## Para operadores de gateway {#for-gateway-operators}

O login nativo está disponível automaticamente em qualquer gateway protegido com um
provedor de sessão interativa registrado. Nenhuma configuração é necessária — as
rotas `/auth/native/*` e o anúncio de `auth_flows` fazem parte do
subsistema de autenticação do dashboard. Provedores OAuth (ex.: o provedor
**Nous** incluso) intermediam o redirecionamento do IdP upstream; provedores de senha
(ex.: o plugin **basic-auth** incluso) levam o browser do sistema ao formulário
de credenciais `/login` do gateway — o que permite que gerenciadores de senha
do SO (Senhas do macOS, etc.) preencham o formulário automaticamente, algo que
nenhum webview embarcado do desktop consegue oferecer. Credenciais apenas de
token (ex.: drain) não são logins interativos e não anunciam `native_pkce`.

Os endpoints relevantes (todos públicos, de bootstrap pré-autenticação, assim como as
rotas OAuth `/auth/*` existentes):

- `GET /auth/native/authorize` — inicia o login PKCE intermediado
- `POST /auth/native/token` — troca o código de loopback + verificador por tokens
- `POST /auth/native/refresh` — rotaciona tokens a partir do refresh token do app

## Veja também {#see-also}

- [OAuth via SSH / Hosts Remotos](./oauth-over-ssh.md) — o padrão de
  callback loopback para OAuth de provedor/MCP em máquinas remotas.
- [Rodar o Hermes com o Nous Portal](./run-hermes-with-nous-portal.md)
