---
sidebar_position: 16
title: "xAI Grok OAuth (SuperGrok / X Premium+)"
description: "Entre com sua assinatura SuperGrok ou X Premium+ para usar os modelos Grok no Hermes Agent — sem necessidade de chave de API"
---

# xAI Grok OAuth (SuperGrok / X Premium+) {#xai-grok-oauth-supergrok--x-premium}

O Hermes Agent oferece suporte ao xAI Grok por meio de um fluxo de login OAuth com código de dispositivo baseado em navegador contra [accounts.x.ai](https://accounts.x.ai), usando uma **assinatura SuperGrok** ([grok.com](https://x.ai/grok)) ou uma **assinatura X Premium+** (conta X vinculada). Nenhuma `XAI_API_KEY` é necessária — faça login uma vez e o Hermes atualiza automaticamente sua sessão em segundo plano.

Quando você faz login com uma conta X que tem Premium+, a xAI vincula automaticamente o status da assinatura à sua sessão xAI, então o fluxo OAuth funciona da mesma forma que para assinantes diretos do SuperGrok.

O transporte reutiliza o adaptador `codex_responses` (a xAI expõe um endpoint no estilo Responses), então raciocínio, chamadas de ferramentas, streaming e cache de prompt funcionam sem quaisquer alterações no adaptador.

O mesmo token bearer OAuth também é reutilizado por toda superfície direto-para-xAI no Hermes — TTS, geração de imagem, geração de vídeo e transcrição — então um único login cobre os quatro.

## Visão Geral {#overview}

| Item | Valor |
|------|-------|
| ID do provedor | `xai-oauth` |
| Nome de exibição | xAI Grok OAuth (SuperGrok / X Premium+) |
| Tipo de autenticação | OAuth 2.0 no navegador com código de dispositivo |
| Transporte | xAI Responses API (`codex_responses`) |
| Modelo padrão | `grok-build-0.1` |
| Endpoint | `https://api.x.ai/v1` |
| Servidor de autenticação | `https://accounts.x.ai` |
| Requer variável de ambiente | Não (`XAI_API_KEY` **não** é usada por este provedor) |
| Assinatura | [SuperGrok](https://x.ai/grok) ou [X Premium+](https://x.com/i/premium_sign_up) — veja a observação abaixo |

## Pré-requisitos {#prerequisites}

- Python 3.9+
- Hermes Agent instalado
- Uma assinatura **SuperGrok** ativa na sua conta xAI, **ou** uma assinatura **X Premium+** na conta X com a qual você faz login (a xAI vincula a assinatura automaticamente)
- Um navegador disponível em qualquer lugar onde você possa abrir a URL de verificação impressa

:::warning A xAI pode restringir o acesso OAuth à API por tier
O backend da xAI aplica sua própria lista de permissões na superfície da API OAuth e já foi visto rejeitando assinantes padrão do SuperGrok com `HTTP 403` (veja a issue [#26847](https://github.com/NousResearch/hermes-agent/issues/26847)), mesmo com a assinatura no app ativa. Se o login OAuth for bem-sucedido no navegador, mas a inferência retornar 403, defina `XAI_API_KEY` e mude para o caminho de chave de API (`provider: xai`) — essa superfície não está sujeita à mesma restrição atualmente.
:::

## Início Rápido {#quick-start}

```bash
# Inicie o provedor e o seletor de modelo
hermes model
# → Selecione "xAI Grok OAuth (SuperGrok / X Premium+)" na lista de provedores
# → O Hermes abre ou imprime uma URL de verificação accounts.x.ai
# → Digite o código exibido se solicitado, depois aprove o acesso no navegador
# → Escolha um modelo (grok-build-0.1 está no topo)
# → Comece a conversar

hermes
```

Depois do primeiro login, as credenciais são armazenadas em `~/.hermes/auth.json` e atualizadas automaticamente antes de expirarem.

## Fazendo Login Manualmente {#logging-in-manually}

Você pode disparar um login sem passar pelo seletor de modelo:

```bash
hermes auth add xai-oauth
```

### Sessões remotas / headless {#remote--headless-sessions}

Em servidores, contêineres, consoles somente-navegador (Cloud Shell, Codespaces, EC2 Instance Connect) ou sessões SSH em que o Hermes não pode abrir um navegador localmente, o Hermes imprime a URL de verificação da xAI e o código de usuário. Abra a URL em qualquer navegador no seu laptop ou no console de nuvem, digite o código se solicitado, e o Hermes continuará fazendo polling até que a xAI aprove o login. Nenhum túnel SSH ou listener de callback local é necessário.

```bash
hermes auth add xai-oauth --no-browser
# Abra a URL de verificação impressa no seu navegador.
```

O mesmo fluxo de código de dispositivo se aplica quando você faz login pelo painel web ou pelo aplicativo desktop: o Hermes mostra a URL de verificação e o código de usuário, depois faz polling em segundo plano até que você aprove o acesso.

## Como o Login Funciona {#how-the-login-works}

1. O Hermes solicita um código de dispositivo a `auth.x.ai`.
2. Você abre a URL de verificação, faz login, digita o código exibido se solicitado, e aprova o acesso.
3. O Hermes faz polling na xAI até a aprovação, depois salva os tokens em `~/.hermes/auth.json`.
4. A partir daí, o Hermes atualiza o token de acesso em segundo plano — você permanece conectado até executar `hermes auth logout xai-oauth` ou revogar o acesso nas configurações da sua conta xAI.

## Verificando o Status do Login {#checking-login-status}

```bash
hermes doctor
```

A seção `◆ Auth Providers` mostrará o estado atual de cada provedor, incluindo `xai-oauth`.

## Trocando de Modelo {#switching-models}

```bash
hermes model
# → Selecione "xAI Grok OAuth (SuperGrok / X Premium+)"
# → Escolha na lista de modelos (grok-build-0.1 está fixado no topo)
```

Ou defina o modelo diretamente:

```bash
hermes config set model.default grok-build-0.1
hermes config set model.provider xai-oauth
```

## Referência de Configuração {#configuration-reference}

Após o login, o `~/.hermes/config.yaml` conterá:

```yaml
model:
  default: grok-build-0.1
  provider: xai-oauth
  base_url: https://api.x.ai/v1
```

### Aliases de provedor {#provider-aliases}

Todos os seguintes resolvem para `xai-oauth`:

```bash
hermes --provider xai-oauth        # canônico
hermes --provider grok-oauth       # alias
hermes --provider x-ai-oauth       # alias
hermes --provider xai-grok-oauth   # alias
```

## Ferramentas Direto-para-xAI (TTS / Imagem / Vídeo / Transcrição / Busca no X) {#direct-to-xai-tools-tts--image--video--transcription--x-search}

Depois que você fizer login via OAuth, toda ferramenta direto-para-xAI reutiliza automaticamente o mesmo token bearer — não há **configuração separada**, a menos que você prefira usar uma chave de API.

Para escolher um backend para cada ferramenta:

```bash
hermes tools
# → Text-to-Speech       → "xAI TTS"
# → Image Generation     → "xAI Grok Imagine (image)"
# → Video Generation     → "xAI Grok Imagine"
# → X (Twitter) Search   → "xAI Grok OAuth (SuperGrok / X Premium+)"
```

Se os tokens OAuth já estiverem armazenados, o seletor confirma isso e pula o prompt de credencial. Se nem OAuth nem `XAI_API_KEY` estiverem definidos, o seletor oferece um menu de 3 opções: login OAuth, colar chave de API, ou pular.

:::note A geração de vídeo está desativada por padrão
O toolset `video_gen` está desativado por padrão. Ative-o em `hermes tools` → `🎬 Video Generation` (pressione espaço) antes que o agente possa chamar `video_generate`. Caso contrário, o agente pode recorrer à skill do ComfyUI incluída, que também é marcada para geração de vídeo.
:::

:::note A busca no X é ativada automaticamente quando há credenciais da xAI
O toolset `x_search` é ativado automaticamente sempre que credenciais da xAI (um token OAuth SuperGrok / X Premium+ ou `XAI_API_KEY`) estão configuradas. Desative explicitamente via `hermes tools` → `🐦 X (Twitter) Search` (pressione espaço) se você não quiser isso. A ferramenta é roteada pela API `x_search` embutida da xAI na Responses API — ela funciona com **qualquer um** dos dois: seu login OAuth SuperGrok / X Premium+ ou uma `XAI_API_KEY` paga, e prefere OAuth quando ambos estão configurados (usa a cota da sua assinatura em vez de gasto de API). O esquema da ferramenta é ocultado do modelo quando nenhuma credencial da xAI está configurada, independentemente de o toolset estar ativado.
:::

### Modelos {#models}

| Ferramenta | Modelo | Notas |
|------|-------|-------|
| Chat | `grok-build-0.1` | Padrão; selecionado automaticamente ao fazer login via OAuth |
| Chat | `grok-4.3` | Padrão anterior |
| Chat | `grok-4.20-0309-reasoning` | Variante de raciocínio |
| Chat | `grok-4.20-0309-non-reasoning` | Variante sem raciocínio |
| Chat | `grok-4.20-multi-agent-0309` | Variante multiagente |
| Image | `grok-imagine-image` | Padrão; ~5–10 s |
| Image | `grok-imagine-image-quality` | Fidelidade mais alta; ~10–20 s |
| Video | `grok-imagine-video` | Texto para vídeo |
| Video | `grok-imagine-video-1.5-preview` | Imagem para vídeo; alias datado `grok-imagine-video-1.5-2026-05-30` |
| TTS | (voz padrão) | Endpoint `/v1/tts` da xAI |

O catálogo de chat é derivado ao vivo do cache local do `models.dev`; novos lançamentos da xAI aparecem automaticamente quando esse cache é atualizado. O `grok-build-0.1` fica sempre fixado no topo da lista.

## Variáveis de Ambiente {#environment-variables}

| Variável | Efeito |
|----------|--------|
| `XAI_BASE_URL` | Substitui o endpoint padrão `https://api.x.ai/v1` (raramente necessário). |

Para selecionar a xAI como provedor ativo, defina `model.provider: xai-oauth` em `config.yaml` (use `hermes setup` para o fluxo guiado) ou passe `--provider xai-oauth` para uma única invocação.

## Solução de Problemas {#troubleshooting}

### Token expirado — não faz login novamente automaticamente {#token-expired--not-re-logging-in-automatically}

O Hermes atualiza o token antes de cada sessão e novamente reativamente em um 401. Se a atualização falhar com `invalid_grant` (o refresh token foi revogado, ou a conta foi rotacionada), o Hermes exibe uma mensagem de reautenticação tipada em vez de travar.

Quando a falha de atualização é terminal (HTTP 4xx, `invalid_grant`, concessão revogada, etc.), o Hermes marca o refresh token como morto e o coloca em quarentena localmente — chamadas subsequentes pulam a tentativa de atualização condenada em vez de repetir o mesmo 401 continuamente. O agente exibe uma única mensagem de "reautenticação necessária" e fica fora do caminho até que você faça login novamente.

**Correção:** execute `hermes auth add xai-oauth` novamente para iniciar um login novo. A quarentena é liberada na próxima troca bem-sucedida.

### A autorização expirou {#authorization-timed-out}

A aprovação do código de dispositivo tem uma janela de expiração finita (a xAI define `expires_in` na resposta do código de dispositivo, tipicamente da ordem de dezenas de minutos). Se você não aprovar o login a tempo, o Hermes gera um erro de timeout.

**Correção:** execute `hermes auth add xai-oauth` novamente (ou `hermes model`). O fluxo recomeça do zero.

### Fazendo login a partir de um servidor remoto {#logging-in-from-a-remote-server}

Em sessões SSH ou de contêiner, o Hermes imprime a URL de verificação e o código de usuário em vez de abrir um navegador. Abra essa URL em um navegador no seu laptop ou em um console de nuvem — nenhum encaminhamento de porta SSH é necessário para o xAI Grok OAuth.

```bash
hermes auth add xai-oauth --no-browser
```

Para provedores de redirecionamento loopback (Spotify, servidores MCP), veja [OAuth via SSH / Hosts Remotos](./oauth-over-ssh.md).

### HTTP 403 após um login bem-sucedido (tier / entitlement) {#http-403-after-a-successful-login-tier--entitlement}

O OAuth foi concluído no navegador, os tokens foram salvos, mas a inferência ou a atualização do token retornam `HTTP 403` com uma mensagem parecida com *"The caller does not have permission to execute the specified operation"*.

Isso **não** é um problema de token obsoleto — executar `hermes model` novamente não vai mudar isso. O backend da xAI já foi visto restringindo o acesso OAuth à API a tiers específicos do SuperGrok, mesmo com a assinatura no app ativa (issue [#26847](https://github.com/NousResearch/hermes-agent/issues/26847)).

**Correção:** defina `XAI_API_KEY` e mude para o caminho de chave de API:

```bash
export XAI_API_KEY=xai-...
hermes config set model.provider xai
```

Ou faça upgrade da sua assinatura em [x.ai/grok](https://x.ai/grok) se o caminho OAuth for necessário.

### "No xAI credentials found" error at runtime {#no-xai-credentials-found-error-at-runtime}

O armazenamento de autenticação não tem uma entrada `xai-oauth` e nenhuma `XAI_API_KEY` está definida. Você ainda não fez login, ou o arquivo de credenciais foi excluído.

**Correção:** execute `hermes model` e escolha o provedor xAI Grok OAuth, ou execute `hermes auth add xai-oauth`.

## Encerrando a Sessão {#logging-out}

Para remover todas as credenciais armazenadas do xAI Grok OAuth:

```bash
hermes auth logout xai-oauth
```

Isso limpa tanto a entrada OAuth singleton em `auth.json` quanto qualquer linha do pool de credenciais para `xai-oauth`. Use `hermes auth remove xai-oauth <index|id|label>` se você só quiser remover uma única entrada do pool (execute `hermes auth list xai-oauth` para vê-las).

## Veja Também {#see-also}

- [OAuth via SSH / Hosts Remotos](./oauth-over-ssh.md) — túneis SSH para provedores de redirecionamento loopback (Spotify, MCP); a xAI usa código de dispositivo e não precisa de túnel
- [Referência de Provedores de IA](../integrations/providers.md)
- [Variáveis de Ambiente](../reference/environment-variables.md)
- [Configuração](../user-guide/configuration.md)
- [Voz e TTS](../user-guide/features/tts.md)
