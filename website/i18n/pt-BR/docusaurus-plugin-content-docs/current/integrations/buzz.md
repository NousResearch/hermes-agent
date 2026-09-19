---
sidebar_position: 4
title: "Integração com o Buzz"
description: "As três formas de conectar o Hermes Agent ao Buzz — o workspace humano+agente baseado em Nostr do Block"
---

# Integração com o Buzz

[Buzz](https://github.com/block/buzz) é o workspace de código aberto e auto-hospedável do Block, onde humanos e agentes de IA compartilham os mesmos canais. Ele é construído sobre o Nostr: cada mensagem é um evento assinado em um relay que você possui, e cada participante — humano ou agente — é um par de chaves (keypair).

O Hermes se integra ao Buzz de três formas. Escolha de acordo com onde o Hermes roda e o que você quer que ele faça:

| | ① Runtime do Desktop | ② Ponte de relay (ACP) | ③ Plataforma nativa do gateway |
|---|---|---|---|
| **O que é** | O Buzz Desktop inicia o Hermes localmente como um harness gerenciado | O `buzz-acp` do Buzz faz a ponte de um canal para o `hermes acp` via stdio | O gateway do Hermes entra no Buzz como uma plataforma de mensagens de primeira classe |
| **Onde o Hermes roda** | No seu desktop, iniciado pelo Buzz | Em um servidor, iniciado pelo `buzz-acp` | No seu próprio gateway, junto com Telegram/Discord/etc. |
| **Melhor para** | Experimentar o Hermes dentro do Buzz Desktop sem nenhuma configuração | Uma identidade de agente hospedada quando o Buzz é dono do transporte | Hermes completo: memória, skills, aprovações, cron, sessões |
| **Entrada** | ACP via stdio | ACP via stdio (via WebSocket do relay) | WebSocket Nostr autenticado por NIP-42 (com fallback de polling) |
| **Configuração** | Descoberta automática | Env vars do `buzz-acp` | `hermes gateway setup` → Buzz |

## ① Runtime gerenciado do Buzz Desktop {#buzz-desktop-managed-runtime}

O Buzz Desktop distribui o Hermes como um runtime pré-configurado. Com o Hermes instalado da forma normal, abra **Settings → Runtimes** e o Hermes aparece automaticamente — a descoberta resolve o launcher `hermes-acp` no PATH do seu shell de login, que o instalador grava em `~/.local/bin` (e o `hermes update` se autocorrige em instalações mais antigas).

Configuração completa, solução de problemas e a postura de segurança (o Buzz aprova automaticamente as permissões de ferramentas — mantenha os agentes restritos ao owner): **[Integração de Host ACP → Buzz Desktop](/user-guide/features/acp#buzz-desktop)**

## ② Ponte de relay (buzz-acp + ACP) {#relay-bridge-buzz-acp--acp}

Para uma identidade Hermes hospedada que entra em *canais* do Buzz enquanto o próprio harness do Buzz é dono do transporte:

```text
Buzz relay <-- WebSocket --> buzz-acp <-- ACP over stdio --> Hermes Agent
```

O Hermes iniciado usa a mesma config, credenciais, memória e skills que o `hermes` naquele host. Emissão de chaves, descoberta de canais, telemetria restrita ao owner (`BUZZ_ACP_RELAY_OBSERVER`) e orientação sobre permissões headless: **[Integração de Host ACP → Canais do Buzz (ponte de relay)](/user-guide/features/acp#buzz-channels-relay-bridge)**

## ③ Plataforma nativa do gateway (recomendado para o Hermes completo) {#native-gateway-platform-recommended-for-full-hermes}

O plugin de plataforma `buzz` incluso transforma o Buzz em uma plataforma de mensagens normal do Hermes — canais, DMs, controle por menção, respostas em thread, reações, imagens e entrega via cron (`deliver=buzz`), com as próprias aprovações, memória e gerenciamento de sessão do Hermes intactos. A entrada chega por um WebSocket Nostr persistente autenticado por NIP-42 (assinatura BIP-340 sem dependências externas), com fallback automático para polling via CLI; a saída passa pela CLI `buzz`.

```bash
hermes gateway setup   # pick Buzz
```

Referência completa de configuração (env vars, config.yaml, modos de transporte, controle de acesso): **[Mensageria → Buzz](/user-guide/messaging/buzz)**

## Qual devo usar? {#which-one-should-i-use}

- **Só explorando, usuário do Buzz Desktop** → ① funciona sem nenhuma configuração.
- **Rodando um relay comunitário e quer uma identidade de agente gerenciada pelo Buzz** → ②.
- **Você já roda o Hermes como seu agente e quer o Buzz como mais um canal** → ③. Essa é a integração mais profunda e a que preserva todos os recursos do Hermes.

①/② e ③ usam identidades e transportes diferentes; rode ③ com seu próprio par de chaves Nostr dedicado. O adapter obtém um lock com escopo no par relay+pubkey, então dois perfis do Hermes não podem acidentalmente operar a mesma identidade Buzz.

## Créditos {#credits}

A integração com o Buzz foi construída com a comunidade: @SHL0MS (launcher no PATH + auditoria de segurança do Desktop), @NYTEMODEONLY (documentação da ponte de relay), @rob-coco (adapter de plataforma), @ScaleLeanChris (transporte WebSocket Nostr + assinatura NIP-42/BIP-340) e @jethac (verificação multi-agente).
