---
sidebar_position: 15
title: "MiniMax OAuth"
description: "Faça login no MiniMax via OAuth no navegador e use os modelos MiniMax-M2.7 no Hermes Agent — sem necessidade de chave de API"
---

# MiniMax OAuth {#minimax-oauth}

O Hermes Agent oferece suporte ao **MiniMax** por meio de um fluxo de login OAuth baseado em navegador, usando as mesmas credenciais do [portal MiniMax](https://www.minimax.io). Nenhuma chave de API ou cartão de crédito é necessária — faça login uma vez e o Hermes atualiza automaticamente sua sessão.

O transporte reutiliza o adaptador `anthropic_messages` (o MiniMax expõe um endpoint compatível com Anthropic Messages em `/anthropic`), então todos os recursos existentes de chamada de ferramentas, streaming e contexto funcionam sem quaisquer alterações no adaptador.

## Visão Geral {#overview}

| Item | Valor |
|------|-------|
| ID do provedor | `minimax-oauth` |
| Nome de exibição | MiniMax (OAuth) |
| Tipo de autenticação | OAuth no navegador (fluxo de redirecionamento PKCE) |
| Transporte | Compatível com Anthropic Messages (`anthropic_messages`) |
| Modelos | `MiniMax-M2.7`, `MiniMax-M2.7-highspeed` |
| Endpoint global | `https://api.minimax.io/anthropic` |
| Endpoint China | `https://api.minimaxi.com/anthropic` |
| Requer variável de ambiente | Não (`MINIMAX_API_KEY` **não** é usada por este provedor) |

## Pré-requisitos {#prerequisites}

- Python 3.9+
- Hermes Agent instalado
- Uma conta MiniMax em [minimax.io](https://www.minimax.io) (global) ou [minimaxi.com](https://www.minimaxi.com) (China)
- Um navegador disponível na máquina local (ou use `--no-browser` para sessões remotas)

## Início Rápido {#quick-start}

```bash
# Inicie o provedor e o seletor de modelo
hermes model
# → Selecione "MiniMax (OAuth)" na lista de provedores
# → O Hermes abre seu navegador na página de autorização do MiniMax
# → Aprove o acesso no navegador
# → Selecione um modelo (MiniMax-M2.7 ou MiniMax-M2.7-highspeed)
# → Comece a conversar

hermes
```

Depois do primeiro login, as credenciais são armazenadas em `~/.hermes/auth.json` e atualizadas automaticamente antes de cada sessão.

## Fazendo Login Manualmente {#logging-in-manually}

Você pode disparar um login sem passar pelo seletor de modelo:

```bash
hermes auth add minimax-oauth
```

### Região China {#china-region}

Se sua conta estiver na plataforma da China (`minimaxi.com`), use o provedor `minimax-cn`, baseado em chave de API, em vez disso — o `minimax-cn` é registrado apenas com `auth_type="api_key"` (sem fluxo OAuth). Configure `MINIMAX_CN_API_KEY` (e opcionalmente `MINIMAX_CN_BASE_URL`) diretamente:

```bash
echo 'MINIMAX_CN_API_KEY=your-key' >> ~/.hermes/.env
```

### Sessões remotas / headless {#remote--headless-sessions}

Em servidores ou contêineres onde nenhum navegador está disponível:

```bash
hermes auth add minimax-oauth --no-browser
```

O Hermes vai imprimir a URL de verificação e o código de usuário — abra a URL em qualquer dispositivo e digite o código quando solicitado.

## O Fluxo OAuth {#the-oauth-flow}

O Hermes implementa um fluxo OAuth PKCE no navegador contra os endpoints OAuth do MiniMax:

1. O Hermes gera um par verificador/desafio PKCE e um valor de estado aleatório.
2. Ele envia um POST para `{base_url}/oauth/code` com o desafio e recebe um `user_code` e uma `verification_uri`.
3. Seu navegador abre a `verification_uri`. Se solicitado, digite o `user_code`.
4. O Hermes faz polling em `{base_url}/oauth/token` até que o token chegue (ou o prazo expire).
5. Os tokens (`access_token`, `refresh_token`, expiração) são salvos em `~/.hermes/auth.json` sob a chave `minimax-oauth`.

A atualização do token (concessão padrão OAuth `refresh_token`) é executada automaticamente no início de cada sessão quando o token de acesso está a menos de 60 segundos de expirar.

## Verificando o Status do Login {#checking-login-status}

```bash
hermes doctor
```

A seção `◆ Auth Providers` mostrará:

```
✓ MiniMax OAuth  (logged in, region=global)
```

ou, se não estiver conectado:

```
⚠ MiniMax OAuth  (not logged in)
```

## Trocando de Modelo {#switching-models}

```bash
hermes model
# → Selecione "MiniMax (OAuth)"
# → Escolha na lista de modelos
```

Ou defina o modelo diretamente:

```bash
hermes config set model.default MiniMax-M2.7
hermes config set model.provider minimax-oauth
```

## Referência de Configuração {#configuration-reference}

Após o login, o `~/.hermes/config.yaml` conterá entradas semelhantes a:

```yaml
model:
  default: MiniMax-M2.7
  provider: minimax-oauth
  base_url: https://api.minimax.io/anthropic
```

### Endpoints por região {#region-endpoints}

| ID do provedor | Portal | Endpoint de inferência |
|-------------|--------|-------------------|
| `minimax-oauth` (global) | `https://api.minimax.io` | `https://api.minimax.io/anthropic` |
| `minimax-cn` (China) | `https://api.minimaxi.com` | `https://api.minimaxi.com/anthropic` |

### Aliases de provedor {#provider-aliases}

Todos os seguintes resolvem para `minimax-oauth`:

```bash
hermes --provider minimax-oauth    # canônico
hermes --provider minimax-portal   # alias
hermes --provider minimax-global   # alias
hermes --provider minimax_oauth    # alias (forma com underscore)
```

## Variáveis de Ambiente {#environment-variables}

O provedor `minimax-oauth` **não** usa `MINIMAX_API_KEY` ou `MINIMAX_BASE_URL`. Essas variáveis são apenas para os provedores `minimax` e `minimax-cn`, baseados em chave de API.

| Variável | Efeito |
|----------|--------|
| `MINIMAX_API_KEY` | Usada apenas pelo provedor `minimax` — ignorada para `minimax-oauth` |
| `MINIMAX_CN_API_KEY` | Usada apenas pelo provedor `minimax-cn` — ignorada para `minimax-oauth` |

Para usar o `minimax-oauth` como provedor ativo, defina `model.provider: minimax-oauth` em `config.yaml` (use `hermes setup` para o fluxo guiado), ou passe `--provider minimax-oauth` para uma única invocação:

```bash
hermes --provider minimax-oauth
```

## Modelos {#models}

| Modelo | Melhor para |
|-------|----------|
| `MiniMax-M2.7` | Raciocínio de longo contexto, chamadas de ferramentas complexas |
| `MiniMax-M2.7-highspeed` | Menor latência, tarefas leves, chamadas auxiliares |

Ambos os modelos suportam até 200.000 tokens de contexto.

O `MiniMax-M2.7-highspeed` também é usado automaticamente como modelo auxiliar para tarefas de visão e delegação quando o `minimax-oauth` é o provedor principal.

## Solução de Problemas {#troubleshooting}

### Token expirado — não faz login novamente automaticamente {#token-expired--not-re-logging-in-automatically}

O Hermes atualiza o token no início de cada sessão se ele estiver a menos de 60 segundos de expirar. Se o token de acesso já estiver expirado (por exemplo, após um longo período offline), a atualização acontece automaticamente na próxima requisição. Se a atualização falhar com `refresh_token_reused` ou `invalid_grant`, o Hermes marca a sessão como exigindo novo login.

Quando a falha de atualização é terminal (HTTP 4xx, `invalid_grant`, concessão revogada, etc.), o Hermes marca o refresh token como morto e o coloca em quarentena localmente, para que não continue repetindo a troca condenada. O agente exibe uma única mensagem de "reautenticação necessária" e fica fora do caminho até que você faça login novamente.

**Correção:** execute `hermes auth add minimax-oauth` novamente para iniciar um login novo. A quarentena é liberada na próxima troca bem-sucedida.

### A autorização expirou {#authorization-timed-out}

O fluxo de código de dispositivo tem uma janela de expiração finita. Se você não aprovar o login a tempo, o Hermes gera um erro de timeout.

**Correção:** execute `hermes auth add minimax-oauth` novamente (ou `hermes model`). O fluxo recomeça do zero.

### Incompatibilidade de state (possível CSRF) {#state-mismatch-possible-csrf}

O Hermes detectou que o valor de `state` retornado pelo servidor de autorização não corresponde ao que foi enviado.

**Correção:** execute o login novamente. Se persistir, verifique se há um proxy ou redirecionamento modificando a resposta OAuth.

### Fazendo login a partir de um servidor remoto {#logging-in-from-a-remote-server}

Se o `hermes` não conseguir abrir uma janela de navegador, use `--no-browser`:

```bash
hermes auth add minimax-oauth --no-browser
```

O Hermes imprime a URL e o código. Abra a URL em qualquer dispositivo e conclua o fluxo lá.

### "Not logged into MiniMax OAuth" error at runtime {#not-logged-into-minimax-oauth-error-at-runtime}

O armazenamento de autenticação não tem credenciais para `minimax-oauth`. Você ainda não fez login, ou o arquivo de credenciais foi excluído.

**Correção:** execute `hermes model` e selecione MiniMax (OAuth), ou execute `hermes auth add minimax-oauth`.

## Encerrando a Sessão {#logging-out}

Para remover as credenciais armazenadas do MiniMax OAuth:

```bash
hermes auth logout minimax-oauth
```

## Veja Também {#see-also}

- [Referência de Provedores de IA](../integrations/providers.md)
- [Variáveis de Ambiente](../reference/environment-variables.md)
- [Configuração](../user-guide/configuration.md)
- [hermes doctor](../reference/cli-commands.md)
