---
sidebar_position: 1
title: "Nous Portal"
description: "Uma assinatura, mais de 300 modelos de fronteira e o Tool Gateway — a forma recomendada de executar o Hermes Agent"
---

# Nous Portal {#nous-portal}

O [Nous Portal](https://portal.nousresearch.com) é o gateway de assinatura unificado da Nous Research e **a forma recomendada de executar o Hermes Agent**. Um único login OAuth substitui a ginástica de contas separadas, chaves de API e relações de faturamento em cada laboratório de modelos, API de busca, gerador de imagens e provedor de navegador que você precisaria configurar manualmente.

Se você só tem tempo para configurar uma coisa, configure esta. O caminho mais rápido:

```bash
hermes setup --portal
```

Esse único comando executa o OAuth do Portal, permite que você escolha um modelo da Nous, define a Nous como seu provedor de inferência no `config.yaml` e ativa o Tool Gateway. Você fica pronto para usar `hermes chat` imediatamente depois.

Ainda não tem uma assinatura? [portal.nousresearch.com/manage-subscription](https://portal.nousresearch.com/manage-subscription) — cadastre-se e depois volte para executar o comando acima.

## O que está incluído na assinatura {#whats-in-the-subscription}

### Mais de 300 modelos de fronteira, uma única fatura {#300-frontier-models-one-bill}

O Portal faz proxy de um catálogo curado de modelos agênticos de todo o ecossistema — cobrado contra sua assinatura da Nous em vez de um saldo de crédito por laboratório.

| Família | Modelos |
|--------|--------|
| **Anthropic Claude** | Opus 4.7, Opus 4.6, Sonnet 4.6, Haiku 4.5 |
| **OpenAI** | GPT-5.5, GPT-5.5 Pro, GPT-5.4 Mini, GPT-5.4 Nano, GPT-5.3 Codex |
| **Google Gemini** | Gemini 3 Pro Preview, Gemini 3 Flash Preview, Gemini 3.1 Pro Preview, Gemini 3.1 Flash Lite Preview |
| **DeepSeek** | DeepSeek V4 Pro |
| **Qwen** | Qwen3.7-Max, Qwen3.6-35B-A3B |
| **Kimi / Moonshot** | Kimi K2.6 |
| **GLM / Zhipu** | GLM-5.1 |
| **MiniMax** | MiniMax M2.7 |
| **xAI** | Grok 4.3 |
| **NVIDIA** | Nemotron-3 Super 120B-A12B |
| **Tencent** | Hunyuan 3 Preview |
| **Xiaomi** | MiMo V2.5 Pro |
| **StepFun** | Step 3.5 Flash |
| **Hermes** | Hermes-4-70B, Hermes-4-405B (chat, veja [nota abaixo](#a-note-on-hermes-4)) |
| **+ todo o resto** | mais de 280 modelos adicionais — toda a fronteira agêntica |

O roteamento acontece por meio do OpenRouter internamente, então a disponibilidade de modelos e o comportamento de failover correspondem ao que você teria com uma chave OpenRouter — só que cobrado contra sua assinatura da Nous. Alterne entre Claude Sonnet 4.6 para código e Gemini 3 Pro para contexto longo com `/model` no meio da sessão — sem novas credenciais, sem recargas, sem erros surpresa de saldo zerado.

### O Nous Tool Gateway {#the-nous-tool-gateway}

A mesma assinatura desbloqueia o [Tool Gateway](/user-guide/features/tool-gateway), que roteia as chamadas de ferramentas do Hermes Agent por infraestrutura gerenciada pela Nous. Cinco backends, um único login:

| Ferramenta | Parceiro | O que faz |
|------|---------|--------------|
| **Busca e extração na web** | Firecrawl | Busca e extração de página completa de nível agente. Sem chave de API do Firecrawl, sem preocupação com limites de taxa. |
| **Geração de imagens** | FAL | Nove modelos sob um único endpoint: FLUX 2 Klein 9B, FLUX 2 Pro, Z-Image Turbo, Nano Banana Pro (Gemini 3 Pro Image), GPT Image 1.5, GPT Image 2, Ideogram V3, Recraft V4 Pro, Qwen Image. |
| **Texto para fala** | OpenAI TTS | TTS de alta qualidade sem uma chave OpenAI separada. Habilita o [modo de voz](/user-guide/features/voice-mode) em todas as plataformas de mensagens. |
| **Automação de navegador em nuvem** | Browser Use | Sessões Chromium headless para `browser_navigate`, `browser_click`, `browser_type`, `browser_vision`. Não precisa de conta no Browserbase. |
| **Sandbox de terminal em nuvem** | Modal | Sandboxes de terminal serverless para execução de código (complemento opcional). |

Sem o gateway, configurar cada um desses serviços significa uma conta no Firecrawl, uma conta no FAL, uma conta no Browser Use, uma chave OpenAI e uma conta no Modal — cinco cadastros separados, cinco painéis separados, cinco fluxos de recarga separados. Com o gateway, tudo isso é roteado por uma única assinatura.

Você também pode habilitar apenas ferramentas específicas do gateway (por exemplo, busca na web, mas não geração de imagens) — veja [Combinando o gateway com seus próprios backends](#mixing-the-gateway-with-your-own-backends) abaixo.

### Sem credenciais nos seus dotfiles {#no-credentials-in-your-dotfiles}

Como tudo é roteado por uma única sessão do Portal autenticada via OAuth, você não acumula um arquivo `.env` com uma dezena de chaves de API de longa duração. O token de refresh em `~/.hermes/auth.json` é a única credencial em disco, e o Hermes gera JWTs de curta duração a partir dele em cada requisição — veja [Gerenciamento de tokens](#token-handling) abaixo.

### Paridade entre plataformas {#cross-platform-parity}

No [Windows nativo](/user-guide/windows-native), a configuração de chaves de API por ferramenta é o ponto mais complicado — instalar uma conta no Firecrawl, uma conta no FAL, uma conta no Browser Use, uma chave OpenAI a partir do Windows é a parte de maior fricção para obter um agente útil. Uma assinatura do Portal resolve isso: um único OAuth cobre o modelo e todas as ferramentas do gateway, então usuários do Windows têm a mesma experiência que usuários de macOS/Linux sem configurar manualmente quatro backends.

## Uma nota sobre o Hermes 4 {#a-note-on-hermes-4}

A própria família **Hermes 4** da Nous Research (Hermes-4-70B, Hermes-4-405B) está disponível através do Portal a taxas fortemente reduzidas. Esses são **modelos de chat de fronteira com raciocínio híbrido** — fortes em matemática, ciências, seguimento de instruções, aderência a esquemas, roleplay e escrita longa.

Eles **não são recomendados para uso dentro do Hermes Agent**, no entanto. O Hermes 4 é ajustado para chat e raciocínio, não para o loop de chamadas de ferramentas em alta velocidade do qual o agente depende. Use-os para fluxos de trabalho de pesquisa ou via o [proxy de assinatura](/user-guide/features/subscription-proxy) a partir de outras ferramentas — mas, para trabalho de agente, escolha um modelo agêntico de fronteira do catálogo:

```bash
/model anthropic/claude-sonnet-4.6     # melhor modelo agêntico de uso geral
/model openai/gpt-5.5-pro              # raciocínio forte + chamadas de ferramentas
/model google/gemini-3-pro-preview     # janela de contexto enorme
/model deepseek/deepseek-v4-pro        # codificador com bom custo-benefício
```

A própria [página de informações de modelos](https://portal.nousresearch.com/info) do Portal traz o mesmo aviso, então isso não é uma opinião do lado do Hermes — é a orientação oficial da Nous Research.

## Configuração {#setup}

### Instalação nova — um único comando {#fresh-install--one-command}

```bash
hermes setup --portal
```

Isso executa a configuração completa em uma única etapa:

1. Abre seu navegador em portal.nousresearch.com para o login OAuth
2. Armazena o token de refresh em `~/.hermes/auth.json`
3. Permite que você escolha um modelo da Nous na lista curada (ou pule para manter o atual)
4. Define a Nous como seu provedor de inferência em `~/.hermes/config.yaml` (quando você escolhe um modelo)
5. Ativa o Tool Gateway (roteamento de web, imagem, TTS e navegador)
6. Retorna você ao terminal pronto para usar `hermes chat`

Se você ainda não tem uma assinatura, cadastre-se primeiro em [portal.nousresearch.com/manage-subscription](https://portal.nousresearch.com/manage-subscription).

### Instalação existente — adicionar o Portal junto com outros provedores {#existing-install--add-portal-alongside-other-providers}

Se você já tem o Hermes configurado com OpenRouter, Anthropic ou qualquer outro provedor e quer adicionar o Portal junto com eles:

```bash
hermes model
# escolha "Nous Portal" na lista de provedores
# o navegador abre, faça login, pronto
```

Seus provedores existentes permanecem configurados. Você pode alternar entre eles com `/model` no meio da sessão ou `hermes model` entre sessões — o Portal se torna um dos seus provedores disponíveis, não o único.

### Configuração headless / SSH / remota {#headless--ssh--remote-setup}

O OAuth precisa de um navegador, mas o callback de loopback é executado na máquina onde o Hermes está sendo executado. Para hosts remotos, veja [OAuth via SSH / Hosts Remotos](/guides/oauth-over-ssh) — os mesmos padrões funcionam para o Portal como para qualquer outro provedor baseado em OAuth (encaminhamento de porta com `ssh -L`).

### Configuração de perfis {#profile-setup}

Se você usa [perfis do Hermes](/user-guide/profiles), o token de refresh do Portal é automaticamente compartilhado entre todos os perfis por meio de um armazenamento de tokens compartilhado. Faça login uma vez em qualquer perfil, e os demais o reconhecem automaticamente — sem necessidade de repetir o fluxo OAuth por perfil.

## Usando o Portal no dia a dia {#using-the-portal-day-to-day}

### Inspecionando o que está configurado {#inspecting-whats-wired-up}

```bash
hermes portal            # faça login no Nous Portal + configure-o (onboarding único)
hermes portal info       # status de login, informações da assinatura, roteamento de modelo + gateway
hermes portal status     # alias para `portal info`
hermes portal tools      # catálogo detalhado do Tool Gateway com roteamento por ferramenta
hermes portal open       # abre a página de gerenciamento da assinatura no seu navegador
```

`hermes portal` (sem subcomando) é o alias legível por humanos para `hermes auth add nous --type oauth` — ele faz seu login, permite que você escolha um modelo da Nous, define a Nous como seu provedor de inferência e oferece a opção de ativar o Tool Gateway (idêntico a `hermes setup --portal`, e o mesmo fluxo da Nous usado na configuração rápida inicial).

`hermes portal info` fornece a visão geral de alto nível:

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
  Cloud terminal        not configured
```

### Alternando modelos {#switching-models}

Dentro de uma sessão:

```bash
/model anthropic/claude-sonnet-4.6
/model openai/gpt-5.5-pro
/model google/gemini-3-pro-preview
```

Ou abra o seletor:

```bash
/model
# setas do teclado, enter para selecionar
```

Fora de uma sessão (o assistente de configuração completo, útil ao adicionar um novo provedor):

```bash
hermes model
```

### Combinando o gateway com seus próprios backends {#mixing-the-gateway-with-your-own-backends}

Se você já tem, por exemplo, uma conta no Browserbase e quer continuar usando-a enquanto roteia a busca na web e a geração de imagens pela Nous, isso é suportado. Use `hermes tools` para escolher os backends por ferramenta:

```bash
hermes tools
# → Busca na web       → "Nous Subscription"
# → Geração de imagens → "Nous Subscription"
# → Navegador          → "Browserbase"  (sua chave existente)
# → TTS                → "Nous Subscription"
```

O Tool Gateway é opt-in por ferramenta, não tudo-ou-nada. Os backends gerenciados aparecem em `hermes tools` independentemente de você estar ou não conectado ao Nous Portal — se você escolher "Nous Subscription" antes de se autenticar, o Hermes executa o login do Portal em linha (isso não altera seu provedor de inferência nem afeta suas outras ferramentas). Veja a [documentação do Tool Gateway](/user-guide/features/tool-gateway) para a matriz completa de configuração por ferramenta.

### Gerenciamento da assinatura {#subscription-management}

Gerencie seu plano, veja o uso, ou faça upgrade/cancele a qualquer momento:

- **Web:** [portal.nousresearch.com/manage-subscription](https://portal.nousresearch.com/manage-subscription)
- **Atalho de CLI:** `hermes portal open` (abre a mesma página no seu navegador padrão)

## Referência de configuração {#configuration-reference}

Depois de `hermes setup --portal`, seu `~/.hermes/config.yaml` ficará assim:

```yaml
model:
  provider: nous
  default: anthropic/claude-sonnet-4.6     # ou o modelo que você escolheu
  base_url: https://inference-api.nousresearch.com/v1
```

As configurações do Tool Gateway ficam em suas respectivas seções de ferramentas:

```yaml
web:
  backend: nous       # busca/extração na web roteia pelo Tool Gateway

image_gen:
  provider: nous

tts:
  provider: nous

browser:
  backend: nous
```

O token de refresh do OAuth é armazenado separadamente em `~/.hermes/auth.json` (não no `config.yaml` — credenciais e configuração são mantidas separadas por design).

## Gerenciamento de tokens {#token-handling}

O Hermes gera um JWT de curta duração a partir do seu token de refresh do Portal armazenado em cada chamada de inferência, em vez de reutilizar uma chave de API de longa duração. O ciclo de vida do token é totalmente automático — refresh, geração, nova tentativa em caso de 401 transitório — e você nunca o vê.

Se o Portal invalidar o token de refresh (mudança de senha, revogação manual, expiração de sessão), o token de refresh inválido é **colocado em quarentena localmente** para que o Hermes deixe de reproduzi-lo e você não veja uma sequência de 401 idênticos. A próxima chamada exibe uma mensagem clara de "reautenticação necessária". Execute `hermes auth add nous` para fazer login novamente; a quarentena é liberada no próximo login bem-sucedido.

## Solução de problemas {#troubleshooting}

### `hermes portal info` mostra "not logged in" {#hermes-portal-info-shows-not-logged-in}

Você não concluiu o fluxo OAuth, ou seu token de refresh foi apagado. Execute:

```bash
hermes portal
```

ou use `hermes model` e selecione novamente o Nous Portal.

### Recebi uma mensagem de "reautenticação necessária" no meio da sessão {#got-a-re-authentication-required-message-mid-session}

Seu token de refresh do Portal foi invalidado (mudança de senha, revogação manual ou expiração de sessão). Execute `hermes auth add nous` e sua próxima requisição usará as novas credenciais. Qualquer quarentena no token antigo é liberada automaticamente após o novo login bem-sucedido.

### Quero usar um modelo de um provedor específico que o Portal não expõe {#want-to-use-a-specific-provider-model-that-the-portal-doesnt-expose}

O Portal faz proxy através do OpenRouter, então qualquer modelo suportado pelo OpenRouter geralmente está disponível. Se um modelo específico não aparecer em `/model`, tente o slug no estilo OpenRouter diretamente:

```bash
/model anthropic/claude-opus-4.6
```

Se um modelo realmente estiver ausente, [abra uma issue](https://github.com/NousResearch/hermes-agent/issues) — nós expomos o catálogo do Portal para o Hermes, e lacunas geralmente significam uma configuração de roteamento que podemos atualizar.

### Cobranças não aparecem na minha conta do Portal {#bills-not-appearing-on-my-portal-account}

Verifique primeiro `hermes portal info` — se ele mostrar que você está usando um provedor diferente (`Model: currently openrouter` em vez de `using Nous as inference provider`), sua configuração local está desatualizada. Execute `hermes model`, escolha o Nous Portal, e a próxima requisição será roteada pela sua assinatura.

## Veja também {#see-also}

- **[Tool Gateway](/user-guide/features/tool-gateway)** — Detalhes completos sobre cada ferramenta do gateway, configuração por ferramenta e preços
- **[Proxy de assinatura](/user-guide/features/subscription-proxy)** — Use sua assinatura do Portal a partir de ferramentas que não são o Hermes (outros agentes, scripts, clientes de terceiros)
- **[Modo de voz](/user-guide/features/voice-mode)** — Conversas de voz usando o OpenAI TTS do Portal
- **[Provedores de IA](/integrations/providers)** — Catálogo completo de provedores caso você queira comparar alternativas
- **[OAuth via SSH](/guides/oauth-over-ssh)** — Login a partir de hosts remotos ou ambientes somente-navegador
- **[Perfis](/user-guide/profiles)** — Múltiplas configurações do Hermes compartilhando um único login do Portal
