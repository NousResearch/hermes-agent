---
sidebar_position: 0
title: "Use o Nemotron 3 Ultra grátis no Hermes Agent"
description: "Experimente o NVIDIA Nemotron 3 Ultra no Nous Portal — grátis de 4 a 18 de junho — com suporte desde o dia 0 no Hermes Agent"
---

# Use o Nemotron 3 Ultra grátis no Hermes Agent

A Nous Research foi convidada a se juntar à **Coalizão Nemotron**, de laboratórios líderes de IA trabalhando com a **NVIDIA** para avançar modelos de fundação de fronteira abertos. Em homenagem a isso, fizemos uma parceria com a **Nebius** para oferecer o **Nemotron 3 Ultra** de graça no [Nous Portal](https://portal.nousresearch.com) por duas semanas (**4 de junho a 18 de junho**). Siga as instruções abaixo para experimentar o modelo no seu Hermes Agent hoje mesmo.

:::info Oferta por tempo limitado
O tier `nvidia/nemotron-3-ultra:free` está disponível de **4 de junho a 18 de junho**. A tag `:free` é o que mantém no plano gratuito — escolha exatamente essa variante.
:::

Escolha a instalação que combina com você. O **aplicativo desktop** é o mais fácil — sem terminal necessário. Se você vive no terminal, a instalação via **linha de comando** está logo abaixo.

## Opção A — Aplicativo desktop (recomendado) {#option-a--desktop-app-recommended}

O caminho mais simples: um instalador de um clique com uma configuração guiada, ponto e clique. Sem necessidade de terminal.

### 1. Baixe e instale {#1-download-and-install}

[Baixe o instalador do Hermes Desktop](https://hermes-agent.nousresearch.com/) para macOS ou Windows e abra-o. Na primeira execução, ele termina de se configurar (geralmente em menos de um minuto).

### 2. Conecte o Nous Portal {#2-connect-nous-portal}

Quando o aplicativo abrir, você verá uma tela "Let's get you set up". Clique em **Nous Portal** (marcado como **Recommended**). Seu navegador abrirá — crie uma conta no [Nous Portal](https://portal.nousresearch.com) (ou faça login), escolha o plano **Free** e autorize o Hermes. O aplicativo se conecta automaticamente.

### 3. Escolha o modelo gratuito Nemotron 3 Ultra {#3-pick-the-free-nemotron-3-ultra-model}

Depois de conectar, o aplicativo mostra um cartão de **Default model**. Clique em **Change**, procure por **nemotron 3 ultra** e selecione a variante marcada como **Free tier**:

```
nvidia/nemotron-3-ultra:free
```

A tag `:free` é o que mantém no tier gratuito — escolha essa variante.

### 4. Comece a conversar {#4-start-chatting}

Clique em **Start chatting**. É isso — você já está conversando com o Nemotron 3 Ultra, de graça.

## Opção B — Linha de comando {#option-b--command-line}

Prefere o terminal?

### 1. Instale o Hermes Agent {#1-install-hermes-agent}

No macOS/Linux/WSL2/Android, execute

```bash
curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash
```

No Windows, execute

```powershell
iex (irm https://hermes-agent.nousresearch.com/install.ps1)
```

Prefere revisar antes? Baixe o [`install.sh`](https://hermes-agent.nousresearch.com/install.sh), inspecione-o e depois execute-o.

Depois que terminar, recarregue seu shell:

```bash
source ~/.bashrc   # or source ~/.zshrc
```

### 2. Execute a Configuração Rápida {#2-run-quick-setup}

```bash
hermes setup
```

Selecione **Quick Setup**. O Hermes abre uma aba do navegador e espera você concluir os próximos passos.

### 3. Crie uma conta no Nous Portal {#3-create-a-nous-portal-account}

No navegador, crie uma conta no [Nous Portal](https://portal.nousresearch.com) (ou faça login) e escolha o plano **Free**.

### 4. Conecte sua conta {#4-connect-your-account}

Quando solicitado a conectar sua conta ao Hermes Agent, clique em **Connect**. Você verá uma confirmação assim que estiver vinculado.

### 5. Selecione o modelo gratuito Nemotron 3 Ultra {#5-select-the-free-nemotron-3-ultra-model}

Volte ao seu terminal. Na lista de modelos, selecione:

```
nvidia/nemotron-3-ultra:free
```

A tag `:free` é o que mantém no tier gratuito, então certifique-se de escolher essa variante.

### 6. Comece a conversar {#6-start-chatting}

Complete os prompts restantes da Configuração Rápida e execute:

```bash
hermes
```

É isso — você já está conversando com o Nemotron 3 Ultra, de graça.

## Trocando para ele depois {#switching-to-it-later}

Já configurou com outro modelo?

- **Aplicativo desktop:** abra o seletor de modelo, procure por **nemotron 3 ultra** e selecione a variante **Free tier**.
- **CLI / TUI:** troque a qualquer momento de dentro de uma sessão com `/model nvidia/nemotron-3-ultra:free`, ou execute `/model` para abrir o seletor e escolhê-lo na lista.

## Solução de Problemas {#troubleshooting}

- **Não vê o modelo na lista?** Certifique-se de ter concluído a conexão com o Nous Portal e de estar no plano **Free**. Na CLI, `hermes portal info` confirma que você está logado e roteando através da Nous.
- **Escolheu a variante errada?** Selecione novamente `nvidia/nemotron-3-ultra:free` — o sufixo `:free` é necessário para permanecer no tier gratuito.
- **O navegador não abriu / você está em um host remoto (CLI)?** Veja [OAuth via SSH / Hosts Remotos](/guides/oauth-over-ssh) para as soluções de redirecionamento de porta.

## Veja também {#see-also}

- **[Aplicativo Desktop](/user-guide/desktop)** — O aplicativo nativo de um clique (macOS, Windows, Linux)
- **[Execute o Hermes Agent com o Nous Portal](/guides/run-hermes-with-nous-portal)** — Passo a passo completo do Portal: modelos, Tool Gateway e verificação
- **[Integração do Nous Portal](/integrations/nous-portal)** — O que está incluído na assinatura
- **[Início Rápido](/getting-started/quickstart)** — Da instalação ao chat em menos de 5 minutos
