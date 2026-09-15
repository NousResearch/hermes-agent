---
sidebar_position: 3
title: "Camada gratuita e login"
description: "O que o Hermes oferece antes de você adicionar uma chave ou entrar, como a camada gratuita coexiste com sua própria API key, como entrar e como desativá-la."
---

# Camada gratuita e login

:::note Ainda não liberada
A camada gratuita está sendo liberada gradualmente. Até estar disponível para todos, nada nesta página acontece a menos que o processo tenha sido iniciado com `HERMES_GUEST_ONBOARDING=1` no ambiente; sem isso, uma instalação nova se comporta exatamente como antes (o seletor de provider na primeira execução). Esta nota some quando a liberação terminar.
:::

Uma instalação nova do Hermes funciona antes de você colar uma API key ou entrar em qualquer lugar. Quando o Hermes inicia, ele configura a **camada gratuita da Nous** (alguns segundos, mostrado como "Setting up free inference…") e responde no modelo `nous/welcome`. Nada a configurar, nenhum wizard para clicar. `hermes setup` continua disponível quando você quiser; nunca é forçado.

## O que você ganha de cara {#what-you-get-out-of-the-box}

| | Camada gratuita | Depois de entrar |
|---|---|---|
| Inference | `nous/welcome` (um modelo) | Catálogo completo do Nous Portal |
| Connectors (Gmail, Linear, Notion, ...) | Sim | Sim |
| Ferramentas pagas pelo [Tool Gateway](/user-guide/features/tool-gateway) (web search, geração de imagem, TTS, cloud browser) | Não | Sim, cobradas na sua assinatura |
| Créditos ou saldo | Nenhum | Sim |

"Connectors" são as contas de terceiros que você vincula no portal da Nous para o agente agir nelas. Funcionam na camada gratuita sem nenhum login.

Trabalho em segundo plano (compactação de conversa, títulos de chat, compreensão de imagem e similares) também roda em `nous/welcome`.

Enquanto a camada gratuita carrega a inference, o banner e `hermes auth status` mostram `Nous · free tier · nous/welcome`, e `hermes model` lista uma linha **Nous · free tier** com esse único modelo. Pedir outro modelo na camada gratuita imprime um ponteiro em vez de trocar em silêncio:

```text
gpt-5 needs a Nous account or an API key. Use /login to sign in, or /model to pick another provider.
```

Chamar uma ferramenta paga diz `This needs a Nous account. Use /login to sign in.` dentro de um chat (e nomeia `hermes auth upgrade` no terminal); o turno continua sem ela.

Se `model.default` em `config.yaml` nomeia algo diferente de `nous/welcome` enquanto a camada gratuita está fazendo inference, o Hermes usa `nous/welcome` mesmo assim e avisa em uma linha. A camada gratuita serve exatamente um modelo.

## Usando sua própria API key junto {#using-your-own-api-key-alongside-it}

A camada gratuita é o último recurso, nunca uma preferência. Qualquer provider que você configurar vence:

| Você tem | A inference roda em | Connectors |
|---|---|---|
| Nada | Camada gratuita da Nous (`nous/welcome`) | Camada gratuita |
| Uma API key em `.env` (OpenRouter, OpenAI, Anthropic, ...) | Sua chave | Camada gratuita |
| `model.provider` definido em `config.yaml` | Esse provider | Camada gratuita |
| Login no Nous Portal | Nous Portal | Sua conta |

Em uma instalação que já tem um provider, o Hermes ainda configura a camada gratuita uma vez na inicialização para os connectors terem com o que autenticar; seu provider continua fazendo inference. Um aviso único diz isso:

```text
Free Nous inference and connectors are now available. /model to try them, /login to sign in.
```

Você pode escolher a camada gratuita explicitamente em `hermes model` (ou `/model`) como qualquer outro provider.

## Entrando por um chat ou terminal {#signing-in-from-a-chat-or-terminal}

### Por um chat {#from-a-chat}

Rode `/login` em um DM do Hermes no Telegram, Discord ou outra plataforma de messaging suportada (no Slack use `/hermes login`), ou em uma sessão de chat do CLI. Precisa ser uma mensagem direta pareada: em outro lugar o Hermes responde `Sign in from a direct message with Hermes.` Plataformas no formato broadcast como ntfy são recusadas pelo mesmo motivo.

O DM recebe um acknowledgement, seguido de três mensagens: o link de consentimento, o código de login em linha própria, depois `Do not share this code. Waiting for sign-in, up to N minutes.` Você pode continuar conversando enquanto o Hermes espera, e o resultado é enviado no mesmo DM. Rodar `/login` de novo substitui o primeiro código. Sessões ao vivo ainda em `nous/welcome` passam para o modelo definitivo na próxima mensagem. No Ink TUI o código aparece, mas a confirmação não; confira `/status`.

:::warning Uma conta por instalação
`/login` vincula esta instalação inteira do Hermes à conta que aprova o código: sua inference, seus connectors, cada chat que ela atende. Em um gateway várias pessoas podem mandar DM; defina `allow_admin_from` para a plataforma (veja o [guia de acesso a slash commands](/reference/slash-commands)) para que só um operador possa rodar isso.
:::

### Por um terminal {#from-a-terminal}

```bash
hermes auth upgrade
```

1. O Hermes imprime uma URL e um código curto, e abre o browser a menos que você passe `--no-browser` ou esteja em uma sessão SSH. Nunca compartilhe o código.
2. Entre no Nous Portal no browser e confirme.
3. De volta no terminal: `Signed in as you@example.com.`
   Se seu modelo padrão era `nous/welcome`, uma segunda linha nomeia o modelo que sua conta passa a usar, por exemplo `Default model is now upstage/solar-pro4:free.`

A inference passa para o catálogo de modelos da sua conta, ferramentas pagas são liberadas, e `hermes auth status` mostra sua conta em vez da linha da camada gratuita.
`nous/welcome` fica com a camada gratuita: uma conta que o usava cai no modelo recomendado do plano (o mesmo que uma escolha nova em `hermes model` sugeriria), e um modelo padrão que você escolheu sozinho é deixado em paz. Se nenhuma recomendação estiver disponível naquele momento, nenhum padrão é definido e o Hermes pede para rodar `hermes model`.

`/login` em um chat, ou `hermes auth upgrade` em um terminal, é oferecido onde a camada gratuita estiver presente, inclusive em instalações que rodam inference com a própria API key. Entrar ainda libera ferramentas pagas nessas instalações.

:::note Login simples começa do zero
`hermes auth add nous --type oauth` também faz login, mas substitui a camada gratuita de uma vez e não leva seus connectors. Use `/login`, ou `hermes auth upgrade` no terminal, quando você tiver connectors que quer manter.
:::

## No Hermes Desktop {#on-hermes-desktop}

O app desktop roda na mesma camada gratuita do CLI e a mostra em quatro lugares:

| Onde | O que você vê |
|---|---|
| Primeira abertura | Uma tela pronta: "Hermes is ready." com o modelo padrão `nous/welcome`, um badge Free tier, e **Begin**. "Sign in with a Nous account instead" e "Other providers" ficam abaixo. A tela aparece uma vez. |
| Primeira abertura com sua própria API key já presente | Uma faixa única acima do composer: "Free Nous inference and connectors are now available." com **Open model picker**, **Sign in** e **Dismiss**. |
| Barra de status | Um chip "Nous · free tier · nous/welcome" com um badge **Sign in** enquanto a camada gratuita carrega a inference. Você pode escondê-lo no menu de clique direito da barra. |
| Settings › Billing | "You're on the Nous free tier" com um botão **Sign in**; o resumo mostra Plan "Free tier", Model `nous/welcome`, Connectors "Included". Não há saldo nem nada a pagar, então seções de pagamento ou uso não aparecem. |

Entrar por qualquer um desses lugares abre um diálogo. Ele mostra um código e um link; abra o link (ou o browser que o app abriu), confirme no portal, e o diálogo termina com "Signed in as you@example.com." e o modelo padrão que sua conta passa a usar. Um login que você rejeita no browser, um código que expirou, ou um código substituído por um mais novo, cada um mostra sua própria mensagem e te deixa na camada gratuita. O seletor de modelo lista a camada gratuita como uma linha, "Nous · free tier", com o único modelo `nous/welcome`; não há ação de login dentro do seletor.

O desktop lê tudo isso do mesmo estado local que o CLI grava. A tela pronta e a faixa usam a mesma flag única do aviso do CLI, então ver um no CLI significa que você não verá de novo no desktop para aquela identidade da camada gratuita, e o contrário também.

## Desativando a camada gratuita {#turning-the-free-tier-off}

```bash
hermes config set nous.guest false
```

`nous.guest` é uma configuração normal de `config.yaml` (padrão `true`), não uma variável de ambiente.
Com ela desligada:

| | `nous.guest: true` (padrão) | `nous.guest: false` |
|---|---|---|
| Inference gratuita em `nous/welcome` | Disponível | Desligada |
| Connectors sem login | Disponíveis | Desligados |
| Linha da camada gratuita em `hermes model` | Mostrada | Oculta |
| Instalação nova sem nada configurado | Chats imediatamente | Oferece `hermes setup` |
| Entrar com uma conta Nous | Funciona | Funciona |

Nada mais muda. Uma conta Nous logada, suas próprias API keys e todo outro provider funcionam exatamente como antes. Volte para `true` e a camada gratuita retorna no próximo comando que precisar dela.

## O que `hermes logout` faz {#what-hermes-logout-does}

| Situação | Resultado |
|---|---|
| Só a camada gratuita está presente | Nada é limpo. O Hermes imprime: `You're not signed in. Free inference and connectors are always on. Run hermes auth to sign in with a Nous account.` |
| Logado com uma conta Nous | O login é removido deste profile e do store compartilhado, para que nenhum outro profile nesta máquina o pegue de novo. Com `nous.guest: true` a instalação volta à camada gratuita na próxima inicialização. |
| Outro provider está ativo | Comportamento inalterado: a credencial armazenada daquele provider é limpa. |

Não há comando para resetar ou recriar a camada gratuita. Ela é criada uma vez e se cuida sozinha.

## Solução de problemas {#troubleshooting}

| Sintoma | O que significa | O que fazer |
|---|---|---|
| O primeiro comando imprime `It looks like Hermes isn't configured yet` e oferece `hermes setup` | A camada gratuita não pôde ser configurada em poucos segundos: você está offline, ou a camada gratuita não está aberta no portal para o qual o Hermes aponta, ou está com rate limit. | Volte online e rode o comando de novo, ou rode `hermes setup` e adicione um provider seu. Nada fica meio configurado. |
| `Nous free tier is not open on this portal.` | O portal para o qual o Hermes aponta não está oferecendo a camada gratuita no momento. Se você definiu `HERMES_PORTAL_BASE_URL`, esse portal pode não tê-la de jeito nenhum. | Entre com uma conta, remova um override de portal que não precisa mais, ou adicione sua própria chave com `hermes setup`. |
| `Nous free tier is rate limited; try again shortly.` | O portal está limitando novas configurações da camada gratuita no momento. | Espere alguns minutos e tente de novo, ou adicione sua própria chave com `hermes setup`. |
| `This needs a Nous account.` | Você chamou uma ferramenta paga do Tool Gateway na camada gratuita. | `/login` em um chat, `hermes auth upgrade` no terminal, ou configure essa ferramenta com sua própria chave em `hermes tools`. |
| O seletor de modelo mostra só `nous/welcome` sob Nous | Esperado na camada gratuita. | Entre para o catálogo completo, ou adicione uma API key de outro provider. |
| A camada gratuita parou de funcionar depois de duas semanas longe | A identidade da camada gratuita expirou (veja abaixo) e é substituída na próxima inicialização, ou na próxima vez que um turno ou connector a encontrar aposentada. | Nada; inicie o Hermes de novo. Connectors vinculados antes do intervalo precisam ser vinculados de novo, a menos que você tenha entrado. |

## Privacidade {#privacy}

Para a camada gratuita funcionar, o Hermes cria uma identidade no portal da Nous na primeira vez que precisa de uma e guarda a credencial no seu diretório Hermes, compartilhada entre os profiles sob esse diretório. Essa identidade não tem endereço de e-mail, nome nem outros dados pessoais; existe para que chamadas de inference e de connectors possam ser autenticadas e limitadas por taxa. Expira após 14 dias sem uso, momento em que o Hermes cria uma nova de forma transparente na próxima vez que você rodar um comando. Entrar (`/login`, ou `hermes auth upgrade` no terminal) move o que essa identidade segura (seus connectors vinculados) para sua conta. Desligar a camada gratuita com `nous.guest: false` significa que nenhuma identidade é criada ou usada.
