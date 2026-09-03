---
sidebar_position: 4
title: Modelos locais
description: Execute modelos inteiramente na sua máquina — sem conta, sem chave de API, nada sai do seu computador.
---

# Modelos locais

O Hermes pode executar modelos abertos inteiramente na sua própria máquina. Ele baixa e
gerencia o motor de inferência (llama.cpp), escolhe o build certo de cada
modelo para o seu hardware e cuida da memória para que você nunca configure tamanhos de contexto,
camadas de GPU ou quantização. Você escolhe um modelo; o Hermes faz o resto.

Nada sai do seu computador: sem conta, sem chave de API e sem acesso à rede
depois que um modelo é baixado.

## Como começar {#getting-started}

1. Abra **Settings → Providers → Local Models** (ou escolha **Run models
   locally** durante o onboarding).
2. Clique em **Install runtime**. O Hermes baixa o build oficial do llama.cpp
   para o seu hardware (algumas centenas de MB), verifica-o e o mantém
   atualizado.
3. Escolha um modelo no catálogo e clique em **Download**.
4. Clique em **Use**. Novos chats passam a rodar no modelo local.

Esse é o fluxo completo. O servidor inicia e para com o Hermes, reinícios
sobrevivem a reinícios do app, e voltar a um provedor na nuvem é um clique
no seletor de modelo.

## Como o Hermes escolhe o que baixar {#how-hermes-picks-what-to-download}

Cada modelo no catálogo é avaliado em relação à **sua máquina** antes de você
baixar qualquer coisa. Cada linha mostra:

- **Memory fit** — verde (*Fits your GPU*: roda inteiramente na memória da GPU),
  âmbar (*Uses system RAM*: funciona, mas mais lento) ou vermelho (*Too big for this
  machine*).
- **Context** — a janela com que o modelo começa e o máximo até o qual pode
  crescer.
- O tamanho de download do build selecionado para o seu hardware.

Os modelos são oferecidos em vários níveis de qualidade (quantizações). O Hermes escolhe o
build de maior qualidade que rode totalmente na sua GPU; máquinas com menos
memória recebem um build mais compacto do mesmo modelo, com as mesmas
garantias. Abaixo de 4 bits a perda de qualidade é grave demais, então o Hermes nunca
oferece builds menores que isso — uma máquina que não consegue rodar o build de 4 bits
com overflow para a RAM do sistema simplesmente não consegue rodar esse modelo.

Modelos que não cabem permanecem visíveis com o motivo, para que você sempre saiba
o que um upgrade de hardware desbloquearia.

## Como funciona o gerenciamento de memória {#how-memory-management-works}

Modelos locais vivem ou morrem pela colocação na memória, então o Hermes gerencia isso
de ponta a ponta e não expõe nenhum controle:

- **Os modelos começam com uma janela de contexto que cabe totalmente na sua GPU** e crescem
  em direção ao máximo nativo conforme a conversa precisa de mais espaço. Você
  pode ver "Context window grown" no feed de status durante sessões longas
  — isso é a janela expandindo, não um erro.
- **Todo modelo recomendado recebe pelo menos uma janela de contexto de 64K.** Quando um
  modelo é maior que a memória da sua GPU, o Hermes coloca deliberadamente o
  overflow na RAM do sistema na ordem que menos prejudica (pesos de experts
  primeiro, nunca o cache de atenção), trocando um pouco de velocidade para proteger a
  garantia de contexto.
- **A compressão de conversa só entra em ação no máximo
  da janela do modelo** — o crescimento sempre vem primeiro.
- Modelos ociosos são descarregados após 15 minutos para liberar memória da GPU; eles
  recarregam automaticamente na próxima mensagem.

## A barra de status {#the-status-bar}

Clique com o botão direito na barra de status e ative **System resources** para ver utilização
ao vivo da GPU, memória da GPU e RAM enquanto os modelos locais rodam. O medidor de contexto
sempre reflete a janela com a qual o modelo está realmente rodando.

## Encontrando mais modelos {#finding-more-models}

O catálogo é um ponto de partida curado, não um limite. A seção **Find more
models** na mesma página pesquisa todo o Hugging Face:

- Os resultados mostram contagens de download e uma verificação de encaixe por arquivo dimensionada para a sua
  máquina, para que você saiba antes de baixar se um build roda totalmente na
  sua GPU.
- Tudo o que você baixar se comporta exatamente como um modelo do catálogo — o Hermes
  lê o próprio arquivo do modelo para escolher a janela de contexto e a colocação na
  memória. A única diferença: modelos da comunidade não carregam o nosso
  selo de teste "validated".
- Já tem um arquivo `.gguf` no disco? **Add model file** o vincula à
  sua biblioteca sem copiá-lo (o original permanece onde está), e
  fica utilizável imediatamente.

## Usando o seu próprio llama-server {#using-your-own-llama-server}

Se um llama-server já estiver rodando na sua máquina, o Hermes o detecta
e o usa em vez de iniciar o próprio. Aponte um endpoint personalizado a qualquer
servidor compatível com OpenAI para controle manual completo — o runtime gerenciado é
um padrão, não um requisito. Para configurações manuais (Ollama, MLX, builds
personalizados, máquinas CLI headless), veja
[Rodar o Hermes localmente com Ollama](/guides/local-ollama-setup) e
[Rodar LLMs locais no Mac](/guides/local-llm-on-mac).

## Configuração {#configuration}

O runtime gerenciado é controlado pela seção `local_runtime` do
`config.yaml`. A UI do desktop grava esses valores para você; eles estão
documentados para uso em CLI e headless:

```yaml
local_runtime:
  enabled: false     # true = start the managed server with Hermes.
                     # The desktop "Use" button sets this automatically.
  backend: auto      # auto | cuda | metal | vulkan | hip | cpu
  tag: b10362        # pinned llama.cpp release; Hermes updates it with
                     # each release after re-validation
```

Modelos e builds de runtime ficam sob o diretório home do Hermes
(`models/` e `runtimes/llamacpp/`). Selecionar um modelo local como seu
modelo principal usa as configurações padrão `model.provider: llamacpp` +
`model.default` — o mesmo formato de qualquer outro provedor.

## Requisitos e limites {#requirements-and-limits}

- **Windows e Linux:** GPU NVIDIA (CUDA) ou CPU. **macOS:** Apple
  Silicon (Metal). Builds Vulkan atendem GPUs AMD.
- Uma GPU com 8 GB+ de memória roda confortavelmente os modelos pequenos do catálogo;
  16 GB+ roda os modelos de 27–35B em alta qualidade.
- Downloads de modelos têm o tamanho em bytes conferido em relação ao catálogo durante a
  transferência; um download incompleto é apagado e reportado, nunca
  usado pela metade. (Somente os zips do motor de runtime são verificados com SHA-256.)
- Apagar um modelo remove todos os arquivos que ele preparou, incluindo adaptadores de
  visão e companheiros de speculative decoding.
