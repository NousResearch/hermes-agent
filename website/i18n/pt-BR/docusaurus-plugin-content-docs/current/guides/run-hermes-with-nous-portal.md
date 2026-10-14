---
sidebar_position: 1
title: "Execute o Hermes Agent com o Nous Portal"
description: "Passo a passo completo: assine, configure, troque de modelos, ative as ferramentas do gateway e verifique o roteamento"
---

# Execute o Hermes Agent com o Nous Portal

Este guia mostra como executar o Hermes Agent com uma assinatura do [Nous Portal](https://portal.nousresearch.com) de ponta a ponta — desde o cadastro até a verificação de que cada ferramenta está roteando corretamente. Se você só quer uma visão geral do que é o Portal e do que está incluído na assinatura, veja a [página de integração do Nous Portal](/integrations/nous-portal). Esta página é o roteiro de execução.

## Pré-requisitos {#prerequisites}

- Hermes Agent instalado ([Início Rápido](/getting-started/quickstart))
- Um navegador web na máquina que você está configurando (ou redirecionamento de porta via SSH — veja [OAuth via SSH](/guides/oauth-over-ssh))
- Cerca de 5 minutos

Você **não** precisa de: uma chave da OpenAI, uma chave da Anthropic, uma conta Firecrawl, uma conta FAL, uma conta Browser Use, ou qualquer outra credencial por fornecedor. Esse é o objetivo principal.

## 1. Obtenha uma assinatura {#1-get-a-subscription}

Abra [portal.nousresearch.com/manage-subscription](https://portal.nousresearch.com/manage-subscription), cadastre-se e escolha um plano.

Já é assinante? Vá direto para o passo 2.

## 2. Execute a configuração única {#2-run-the-one-shot-setup}

```bash
hermes setup --portal
```

Esse único comando faz cinco coisas:

1. Abre seu navegador em portal.nousresearch.com para o login via OAuth
2. Armazena o token de renovação em `~/.hermes/auth.json`
3. Define `model.provider: nous` em `~/.hermes/config.yaml`
4. Escolhe um modelo agêntico padrão (`anthropic/claude-sonnet-4.6` ou similar)
5. Ativa o Tool Gateway para busca na web, geração de imagens, TTS e automação de navegador

Quando terminar, você estará de volta ao terminal, pronto para conversar.

### E se eu estiver via SSH em um servidor? {#what-if-im-sshd-into-a-server}

O OAuth precisa de um navegador, mas o callback de loopback roda na máquina onde o Hermes está sendo executado. Duas opções:

```bash
# Option A: SSH port forwarding (preferred)
ssh -N -L 8642:127.0.0.1:8642 user@remote-host    # in a local terminal
hermes setup --portal                              # on the remote, open the printed URL in your local browser

# Option B: device-code login (works from Cloud Shell, Codespaces, EC2 Instance Connect)
hermes auth add nous --type oauth
# Then re-run `hermes setup --portal` to wire the provider + gateway
```

Veja [OAuth via SSH / Hosts Remotos](/guides/oauth-over-ssh) para o passo a passo completo, incluindo cadeias ProxyJump, mosh/tmux e detalhes do ControlMaster.

## 3. Verifique se funcionou {#3-verify-it-worked}

```bash
hermes portal info
```

Você deve ver:

```
  Nous Portal
  ───────────
  Auth:    ✓ logged in
  Portal:  https://portal.nousresearch.com
  Model:   ✓ using Nous as inference provider

  Tool Gateway
  ────────────
  Web search & extract  via Nous Portal
  Image generation      via Nous Portal
  Text-to-speech        via Nous Portal
  Browser automation    via Nous Portal
```

Se alguma linha mostrar algo diferente de "via Nous Portal" ou a linha de autenticação disser "not logged in", vá para [Solução de Problemas](#troubleshooting) abaixo.

## 4. Execute sua primeira conversa {#4-run-your-first-conversation}

```bash
hermes chat
```

Tente algo que exercite tanto o modelo quanto o Tool Gateway:

```
Hey, search the web for "Hermes Agent release notes" and summarize the top 3 hits.
```

Você deve ver o Hermes chamar `web_search` (baseado no Firecrawl, através do gateway) e responder com um resumo. Se a busca funcionar e a resposta fizer sentido, está tudo pronto — o Portal está conectado de ponta a ponta.

## 5. Escolha o modelo que você realmente quer {#5-pick-the-model-you-actually-want}

`hermes setup --portal` permite escolher um modelo durante a configuração, mas o grande motivo da assinatura é o acesso ao catálogo completo — troque a qualquer momento com `/model` durante a sessão:

```bash
/model anthropic/claude-sonnet-4.6     # best general-purpose agentic
/model openai/gpt-5.4                  # strong reasoning + tool calling
/model google/gemini-2.5-pro           # huge context window
/model deepseek/deepseek-v3.2          # cost-effective coder
/model anthropic/claude-opus-4.6       # heavyweight for hard problems
```

Ou abra o seletor para navegar:

```bash
/model
```

Escolha um padrão diferente permanentemente:

```bash
# in your terminal, outside any session
hermes config set model.default anthropic/claude-sonnet-4.6
```

### Não escolha o Hermes-4 para trabalho de agente {#dont-pick-hermes-4-for-agent-work}

O Hermes-4-70B e o Hermes-4-405B estão disponíveis no Portal com descontos significativos, mas são **modelos de chat/raciocínio**, não ajustados para chamadas de ferramentas. Eles terão dificuldade com loops de agente de múltiplas etapas. Use-os para trabalho de conversa/pesquisa através do [proxy de assinatura](/user-guide/features/subscription-proxy) a partir de ferramentas não agênticas. Para o Hermes Agent em si, fique com os modelos agênticos de ponta mencionados acima.

A própria [página de informações](https://portal.nousresearch.com/info) do Portal traz esse aviso também — é a orientação oficial da Nous, não apenas uma opinião do lado do Hermes.

## 6. (Opcional) Personalize o roteamento do Tool Gateway {#6-optional-customize-tool-gateway-routing}

O gateway é opt-in por ferramenta, não tudo ou nada. Se você já tem uma conta na Browserbase e quer continuar usando-a enquanto roteia a busca na web e a geração de imagens através da Nous, isso é suportado:

```bash
hermes tools
# → Web search       → "Nous Subscription"     (recommended)
# → Image generation → "Nous Subscription"     (recommended)
# → Browser          → "Browserbase"           (your existing key)
# → TTS              → "Nous Subscription"     (recommended)
```

Essas linhas aparecem no `hermes tools` mesmo antes de você entrar no Nous Portal — se você escolher "Nous Subscription" sem uma sessão ativa, o Hermes faz o login no Portal diretamente (sem alterar seu provedor de inferência ou suas outras ferramentas).

Verifique sua combinação com:

```bash
hermes portal tools
```

Você verá o roteamento por ferramenta — `via Nous Portal` para as que são roteadas através da assinatura, e o nome do parceiro (`browserbase`, `firecrawl`, etc.) para as que usam suas próprias chaves.

## 7. (Opcional) Ative o modo de voz {#7-optional-enable-voice-mode}

Como o Tool Gateway inclui o TTS da OpenAI, o [modo de voz](/user-guide/features/voice-mode) funciona sem uma chave separada da OpenAI:

```bash
hermes setup voice
# → pick "Nous Subscription" for TTS
# → pick a speech-to-text backend (local faster-whisper is free, no setup)
```

Depois, em qualquer sessão de plataforma de mensagens (Telegram, Discord, Signal, etc.), envie uma mensagem de voz e o Hermes vai transcrever, responder e devolver em voz sintetizada — tudo na sua assinatura do Portal.

## 8. (Opcional) Tarefas agendadas e workflows sempre ativos {#8-optional-cron--always-on-workflows}

A assinatura do Portal funciona para [tarefas agendadas (cron jobs)](/user-guide/features/cron) e [processamento em lote](/user-guide/features/batch-processing) da mesma forma que funciona para o chat interativo — o token de renovação do OAuth é reutilizado automaticamente. Sem configuração adicional; basta agendar as tarefas e elas serão cobradas na sua assinatura.

```bash
hermes cron create "every day at 9am" \
  "Search the web for top AI news and summarize the 5 most important stories" \
  --name "Daily AI news"
```

A tarefa agendada roda sem supervisão, chamando o modelo + busca na web + resumo, tudo através da sua assinatura do Portal.

## Perfis e configurações multiusuário {#profiles-and-multi-user-setups}

Se você usa [perfis do Hermes](/user-guide/profiles) (por exemplo, uma configuração separada por projeto), o token de renovação do Portal é automaticamente compartilhado entre todos os perfis via um armazenamento de tokens compartilhado. Faça login uma vez em qualquer perfil, e os demais o reconhecem automaticamente.

Para configurações de equipe onde vários humanos compartilham uma máquina, cada humano tem sua própria conta no Portal → cada diretório home mantém seu próprio `~/.hermes/auth.json` → sem compartilhamento de token entre usuários. Esse é o limite correto.

## Solução de Problemas {#troubleshooting}

### `hermes portal info` mostra "not logged in" depois de `hermes setup --portal`

O fluxo de OAuth não foi concluído. Execute novamente:

```bash
hermes portal
```

Se seu navegador não abrir ou o callback falhar, você provavelmente está em um host remoto/headless — veja [OAuth via SSH](/guides/oauth-over-ssh) para as soluções de redirecionamento de porta.

### "Model: currently openrouter" (ou algum outro provedor) em vez de "using Nous as inference provider"

Sua configuração local ficou desatualizada. O OAuth funcionou, mas `model.provider` ainda está apontando para outro provedor. Corrija:

```bash
hermes config set model.provider nous
```

Ou interativamente:

```bash
hermes model
# pick Nous Portal
```

Reverifique com `hermes portal info`.

### Ferramentas do Tool Gateway mostrando nomes de parceiros em vez de "via Nous Portal"

A configuração por ferramenta está sobrepondo o gateway. Execute:

```bash
hermes tools
# pick "Nous Subscription" for any tool you want gateway-routed
```

Alguns usuários misturam intencionalmente — por exemplo, roteando a web através da Nous, mas usando sua própria chave da Browserbase para o navegador. Se isso for intencional, deixe como está. Se não, este comando corrige.

### "Re-authentication required" no meio da sessão

Seu token de renovação do Portal foi invalidado (troca de senha, revogação manual, expiração de sessão). O token agora está em quarentena localmente, para que o Hermes não o reutilize infinitamente. Basta fazer login novamente:

```bash
hermes auth add nous
```

A quarentena é liberada automaticamente após um novo login bem-sucedido.

### O modelo que eu quero não está no seletor `/model`

O catálogo do Portal reflete a lista de modelos da OpenRouter (mais de 300). Se um modelo estiver faltando, tente digitar o identificador no estilo OpenRouter diretamente:

```bash
/model anthropic/claude-opus-4.6
/model openai/o1-2025-12-17
```

Se um modelo realmente não estiver disponível, [abra uma issue](https://github.com/NousResearch/hermes-agent/issues) — a maioria das lacunas é configuração de roteamento que podemos atualizar.

### A cobrança não aparece na minha conta do Portal

`hermes portal info` vai dizer se você realmente está roteando através do Portal ou de outro provedor. Causas comuns:

- `model.provider` definido como `openrouter`/`anthropic`/etc. em vez de `nous`
- Uma falha de renovação do OAuth que recorreu a outro provedor configurado
- Vários perfis do Hermes onde você está usando o errado (verifique `hermes profile list`)

### Quero revogar e começar do zero

```bash
hermes auth logout nous       # wipes the local refresh token
# Then re-run setup or remove the subscription from the Portal web UI
```

## O que isso significa, em números simples {#what-this-gets-you-in-plain-numbers}

| Sem o Portal | Com o Portal |
|----------------|-------------|
| 1× chave OpenRouter / Anthropic / OpenAI no `.env` | 1× token de renovação OAuth, sem chaves no `.env` |
| 1× chave Firecrawl para web | Web roteada através do gateway |
| 1× chave FAL para geração de imagens | Geração de imagens roteada através do gateway |
| 1× chave Browser Use / Browserbase para navegador | Navegador roteado através do gateway |
| 1× chave OpenAI para TTS / modo de voz | TTS roteado através do gateway |
| 5 painéis separados, recargas, faturas | 1 assinatura, 1 fatura |
| Entre máquinas: replicar todas as 5 chaves | Entre máquinas: refazer o OAuth uma vez |

Esse é o negócio. Se você já usa mais de dois desses backends, a assinatura se paga.

## Veja também {#see-also}

- **[Página de integração do Nous Portal](/integrations/nous-portal)** — Visão geral do que está na assinatura
- **[Tool Gateway](/user-guide/features/tool-gateway)** — Detalhes completos de cada ferramenta roteada pelo gateway
- **[Proxy de assinatura](/user-guide/features/subscription-proxy)** — Use sua assinatura do Portal em ferramentas que não são do Hermes
- **[Modo de voz](/user-guide/features/voice-mode)** — Configure conversas por voz na assinatura do Portal
- **[OAuth via SSH](/guides/oauth-over-ssh)** — Padrões de login remoto/headless
- **[Perfis](/user-guide/profiles)** — Compartilhe um único login do Portal entre várias configurações do Hermes
