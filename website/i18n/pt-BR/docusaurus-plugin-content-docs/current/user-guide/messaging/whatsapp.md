---
sidebar_position: 5
title: "WhatsApp"
description: "Configure o Hermes Agent como bot do WhatsApp via a ponte Baileys built-in"
---

# Configuração do WhatsApp {#whatsapp-setup}

O Hermes se conecta ao WhatsApp por uma ponte built-in baseada em **Baileys**. Isso funciona emulando uma sessão WhatsApp Web — **não** pela API oficial WhatsApp Business. Nenhuma conta de desenvolvedor Meta ou verificação Business é necessária.

> Execute `hermes gateway setup` e escolha **WhatsApp** para um passo a passo guiado.

:::tip Duas integrações WhatsApp
Esta página é para a **ponte Baileys** — configuração rápida, contas pessoais, sem URL pública necessária, risco de banimento.

Se você opera um bot comercial real e quer estabilidade, veja o **[guia WhatsApp Business Cloud API](./whatsapp-cloud.md)**. É o caminho oficial suportado pela Meta: sem risco de banimento de conta, mas exige conta Meta Business e uma URL pública de webhook.

Os dois adaptadores também podem rodar em paralelo com números de telefone diferentes, se fizer sentido para você.
:::

:::warning API não oficial — risco de banimento
O WhatsApp **não** suporta oficialmente bots de terceiros fora da Business API. Usar uma ponte de terceiros traz um pequeno risco de restrições de conta. Para minimizar o risco:
- **Use um número de telefone dedicado** para o bot (não seu número pessoal)
- **Não envie mensagens em massa/spam** — mantenha o uso conversacional
- **Não automatize mensagens de saída** para pessoas que não enviaram mensagem primeiro
:::

:::warning Atualizações do protocolo WhatsApp Web
O WhatsApp atualiza periodicamente o protocolo Web, o que pode quebrar temporariamente a compatibilidade com pontes de terceiros. Quando isso acontece, o Hermes atualiza a dependência da ponte. Se o bot parar após uma atualização do WhatsApp, atualize o Hermes e faça o pareamento novamente.
:::

## Dois modos {#two-modes}

| Modo | Como funciona | Melhor para |
|------|-------------|----------|
| **Número de bot separado** (recomendado) | Dedique um número de telefone ao bot. Pessoas enviam mensagens diretamente para esse número. | UX limpa, múltiplos usuários, menor risco de banimento |
| **Self-chat pessoal** | Use seu próprio WhatsApp. Você envia mensagens para si mesmo para falar com o agente. | Configuração rápida, usuário único, testes |

---

## Pré-requisitos {#prerequisites}

- **Node.js v18+** e **npm** — a ponte WhatsApp roda como processo Node.js
- **Um telefone com WhatsApp** instalado (para escanear o QR code)

Diferente de pontes antigas baseadas em navegador, a ponte atual baseada em Baileys **não** exige Chromium local ou stack de dependências Puppeteer.

---

## Passo 1: Execute o assistente de configuração {#step-1-run-the-setup-wizard}

```bash
hermes whatsapp
```

O assistente irá:

1. Perguntar qual modo você quer (**bot** ou **self-chat**)
2. Instalar dependências da ponte se necessário
3. Exibir um **QR code** no seu terminal
4. Aguardar você escaneá-lo

**Para escanear o QR code:**

1. Abra o WhatsApp no seu telefone
2. Vá em **Settings → Linked Devices**
3. Toque em **Link a Device**
4. Aponte a câmera para o QR code no terminal

Depois do pareamento, o assistente confirma a conexão e encerra. Sua sessão é salva automaticamente.

:::tip
Se o QR code parecer distorcido, garanta que seu terminal tenha pelo menos 60 colunas de largura e suporte Unicode. Você também pode tentar outro emulador de terminal.
:::

---

## Passo 2: Obter um segundo número de telefone (modo bot) {#step-2-getting-a-second-phone-number-bot-mode}

Para o modo bot, você precisa de um número que ainda não esteja registrado no WhatsApp. Três opções:

| Opção | Custo | Notas |
|--------|------|-------|
| **Google Voice** | Grátis | Apenas EUA. Obtenha um número em [voice.google.com](https://voice.google.com). Verifique WhatsApp via SMS pelo app Google Voice. |
| **SIM pré-pago** | US$ 5–15 único | Qualquer operadora. Ative, verifique WhatsApp, depois o SIM pode ficar na gaveta. O número deve permanecer ativo (faça uma ligação a cada 90 dias). |
| **Serviços VoIP** | Grátis–US$ 5/mês | TextNow, TextFree ou similar. Alguns números VoIP são bloqueados pelo WhatsApp — tente alguns se o primeiro não funcionar. |

Depois de obter o número:

1. Instale WhatsApp em um telefone (ou use WhatsApp Business com dual-SIM)
2. Registre o novo número no WhatsApp
3. Execute `hermes whatsapp` e escaneie o QR code dessa conta WhatsApp

---

## Passo 3: Configure o Hermes {#step-3-configure-hermes}

Adicione o seguinte ao seu arquivo `~/.hermes/.env`:

```bash
# Required
WHATSAPP_ENABLED=true
WHATSAPP_MODE=bot                          # "bot" or "self-chat"

# Access control — pick ONE of these options:
WHATSAPP_ALLOWED_USERS=15551234567         # Comma-separated phone numbers (with country code, no +)
# WHATSAPP_ALLOWED_USERS=*                 # OR use * to allow everyone
# WHATSAPP_ALLOW_ALL_USERS=true            # OR set this flag instead (same effect as *)
```

:::tip Atalho allow-all
Definir `WHATSAPP_ALLOWED_USERS=*` permite **todos** os remetentes (equivalente a `WHATSAPP_ALLOW_ALL_USERS=true`).
Isso é consistente com [allowlists de grupo Signal](/reference/environment-variables).
Para usar o fluxo de pareamento, remova ambas as variáveis e confie no
[sistema de pareamento por DM](/user-guide/security#dm-pairing-system).
:::

Configurações opcionais de comportamento em `~/.hermes/config.yaml`:

```yaml
unauthorized_dm_behavior: pair

whatsapp:
  unauthorized_dm_behavior: ignore
```

- `unauthorized_dm_behavior: pair` é o padrão global. Remetentes desconhecidos por DM recebem um código de pareamento.
- `whatsapp.unauthorized_dm_behavior: ignore` faz o WhatsApp ficar silencioso para DMs não autorizados, o que geralmente é a melhor escolha para um número privado.

Depois inicie o gateway:

```bash
hermes gateway              # Foreground
hermes gateway install      # Install as a user service
sudo hermes gateway install --system   # Linux only: boot-time system service
```

O gateway inicia a ponte WhatsApp automaticamente usando a sessão salva.

---

## Persistência de sessão {#session-persistence}

A ponte Baileys salva sua sessão em `~/.hermes/platforms/whatsapp/session`. Isso significa:

- **Sessões sobrevivem a reinicializações** — você não precisa escanear o QR code toda vez
- Os dados de sessão incluem chaves de criptografia e credenciais do dispositivo
- **Não compartilhe nem faça commit deste diretório de sessão** — ele concede acesso completo à conta WhatsApp

---

## Re-pareamento {#re-pairing}

Se a sessão quebrar (reset do telefone, atualização do WhatsApp, desvinculação manual), você verá erros de conexão nos logs do gateway. Para corrigir:

```bash
hermes whatsapp
```

Isso gera um QR code novo. Escaneie novamente e a sessão é restabelecida. O gateway trata desconexões **temporárias** (quedas de rede, telefone offline brevemente) automaticamente com lógica de reconexão.

---

## Mensagens de voz {#voice-messages}

O Hermes suporta voz no WhatsApp:

- **Entrada:** Mensagens de voz (`.ogg` opus) são transcritas automaticamente pelo provedor STT configurado: `faster-whisper` local, Groq Whisper (`GROQ_API_KEY`) ou OpenAI Whisper (`VOICE_TOOLS_OPENAI_KEY`)
- **Saída:** Respostas TTS são enviadas como anexos de arquivo MP3
- Respostas do agente são prefixadas com "⚕ **Hermes Agent**" por padrão. Você pode personalizar ou desabilitar em `config.yaml`:

```yaml
# ~/.hermes/config.yaml
whatsapp:
  reply_prefix: ""                          # Empty string disables the header
  # reply_prefix: "🤖 *My Bot*\n──────\n"  # Custom prefix (supports \n for newlines)
```

---

## Formatação e entrega de mensagens {#message-formatting--delivery}

O WhatsApp suporta **respostas em streaming (progressivas)** — o bot edita sua mensagem em tempo real enquanto a IA gera texto, como Discord e Telegram. Internamente, WhatsApp é classificado como plataforma TIER_MEDIUM para capacidades de entrega.

### Fragmentação {#chunking}

Respostas longas são automaticamente divididas em múltiplas mensagens com **4.096 caracteres** por fragmento (limite prático de exibição do WhatsApp). Você não precisa configurar nada — o gateway trata a divisão e envia os fragmentos sequencialmente.

### Markdown compatível com WhatsApp {#whatsapp-compatible-markdown}

Markdown padrão nas respostas da IA é convertido automaticamente para a formatação nativa do WhatsApp:

| Markdown | WhatsApp | Renders as |
|----------|----------|------------|
| `**bold**` | `*bold*` | **bold** |
| `~~strikethrough~~` | `~strikethrough~` | ~~strikethrough~~ |
| `# Heading` | `*Heading*` | Bold text (no native headings) |
| `[link text](url)` | `link text (url)` | Inline URL |

Blocos de código e código inline são preservados, pois o WhatsApp suporta formatação com triple-backtick nativamente.

### Progresso de ferramentas {#tool-progress}

Quando o agente chama ferramentas (web search, operações de arquivo, etc.), o WhatsApp exibe indicadores de progresso em tempo real mostrando qual ferramenta está rodando. Isso vem habilitado por padrão — sem configuração necessária.

### Agrupamento de mensagens (debounce) {#message-batching-debounce}

O WhatsApp entrega cada mensagem individualmente, então uma rajada rápida (lotes encaminhados, colagens divididas, texto multi-linha) dispararia uma invocação separada do agente por fragmento — desperdiçando tokens e produzindo várias respostas desconexas. O adaptador agrupa mensagens de texto sucessivas do mesmo chat e as despacha como uma única requisição combinada após um curto período de quietude (padrão **5s**, estendido para **10s** para fragmentos muito longos). Ajuste via `config.yaml`:

```yaml
# ~/.hermes/config.yaml
gateway:
  platforms:
    whatsapp:
      extra:
        text_batch_delay_seconds: 5.0         # quiet period before flushing a batch
        text_batch_split_delay_seconds: 10.0  # extended delay near the split threshold
```

Defina `text_batch_delay_seconds: 0` para despachar cada mensagem imediatamente (desabilita agrupamento).

---

## Solução de problemas {#troubleshooting}

| Problema | Solução |
|---------|----------|
| **QR code não escaneia** | Garanta terminal com largura suficiente (60+ colunas). Tente outro terminal. Confirme que está escaneando da conta WhatsApp correta (número do bot, não pessoal). |
| **QR code expira** | QR codes renovam a cada ~20 segundos. Se expirar, reinicie `hermes whatsapp`. |
| **Sessão não persiste** | Verifique se `~/.hermes/platforms/whatsapp/session` existe e é gravável. Se containerizado, monte como volume persistente. |
| **Deslogado inesperadamente** | WhatsApp desvincula dispositivos após longa inatividade. Mantenha o telefone ligado e conectado à rede, depois re-pareie com `hermes whatsapp` se necessário. |
| **Ponte trava ou loops de reconexão** | Reinicie o gateway, atualize o Hermes e re-pareie se a sessão foi invalidada por mudança de protocolo WhatsApp. |
| **Bot para após atualização do WhatsApp** | Atualize o Hermes para a versão mais recente da ponte, depois re-pareie. |
| **macOS: "Node.js not installed" mas node funciona no terminal** | serviços launchd não herdam o PATH do shell. Execute `hermes gateway install` para re-capturar seu PATH atual no plist, depois `hermes gateway start`. Veja a [documentação de serviço do Gateway](./index.md#macos-launchd) para detalhes. |
| **Mensagens não recebidas** | Verifique se `WHATSAPP_ALLOWED_USERS` inclui o número do remetente (com código do país, sem `+` ou espaços), ou defina `*` para permitir todos. Defina `WHATSAPP_DEBUG=true` em `.env` e reinicie o gateway para ver eventos brutos de mensagem em `bridge.log`. |
| **Bot responde a desconhecidos com código de pareamento** | Defina `whatsapp.unauthorized_dm_behavior: ignore` em `~/.hermes/config.yaml` se quiser que DMs não autorizados sejam ignorados silenciosamente. |

---

## Segurança {#security}

:::warning
**Configure controle de acesso** antes de ir ao ar. Defina `WHATSAPP_ALLOWED_USERS` com números específicos (incluindo código do país, sem `+`), use `*` para permitir todos, ou defina
`WHATSAPP_ALLOW_ALL_USERS=true`. Sem nenhum desses, o gateway **nega todas as mensagens recebidas** como medida de segurança.
:::

Por padrão, DMs não autorizados ainda recebem resposta com código de pareamento. Se quiser que um número WhatsApp privado fique completamente silencioso para desconhecidos, defina:

```yaml
whatsapp:
  unauthorized_dm_behavior: ignore
```

- O diretório `~/.hermes/platforms/whatsapp/session` contém credenciais completas de sessão — proteja como uma senha
- Defina permissões de arquivo: `chmod 700 ~/.hermes/platforms/whatsapp/session`
- Use um **número de telefone dedicado** para o bot para isolar risco da sua conta pessoal
- Se suspeitar de comprometimento, desvincule o dispositivo em WhatsApp → Settings → Linked Devices
- Números de telefone nos logs são parcialmente redigidos, mas revise sua política de retenção de logs
