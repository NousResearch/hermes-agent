---
sidebar_position: 3
title: "Android / Termux"
description: "Execute o Hermes Agent diretamente em um Android com Termux"
---

# Hermes no Android com Termux

:::warning Plataforma Tier 2
Termux (Android) é uma [plataforma Tier 2](./platform-support.md#tier-2). O script de instalação e a documentação aqui são mantidos apenas como melhor esforço. Commits no `main` podem quebrar estes pacotes a qualquer momento.
:::

O Hermes Agent pode ser executado diretamente em um telefone Android através do [Termux](https://termux.dev/).

Isso oferece um CLI local funcional no telefone, além dos extras principais que atualmente são conhecidos por instalar de forma limpa no Android.

## O que é suportado no caminho testado?

O bundle testado do Termux instala:

- o CLI do Hermes
- suporte a cron
- suporte a terminal PTY/background
- suporte a gateway do Telegram (manual / execução em background como melhor esforço)
- suporte a MCP
- suporte a memória Honcho
- suporte a ACP

Concretamente, ele mapeia para:

```bash
python -m pip install -e '.[termux]' -c constraints-termux.txt
```

## O que ainda não faz parte do caminho testado?

Alguns recursos ainda precisam de dependências estilo desktop/servidor que não são publicadas para Android, ou não foram validadas em telefones ainda:

- `.[all]` não é suportado no Android atualmente
- o extra `voice` é bloqueado por `faster-whisper -> ctranslate2`, e `ctranslate2` não publica wheels para Android
- a configuração automática do navegador / Playwright é pulada no instalador Termux
- o isolamento de terminal baseado em Docker não está disponível dentro do Termux
- o Android pode ainda suspender jobs em background do Termux, então a persistência do gateway é melhor esforço em vez de um serviço gerenciado normal

Isso não impede que o Hermes funcione bem como um agente CLI nativo no telefone — significa apenas que a instalação móvel recomendada é intencionalmente mais restrita que a instalação desktop/servidor.

---

## Opção 1: Instalador de uma linha

O Hermes agora oferece um caminho de instalação ciente do Termux:

```bash
curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash
```

No Termux, o instalador automaticamente:

- usa `pkg` para pacotes do sistema
- cria o venv com `python -m venv`
- tenta o extra amplo `.[termux-all]` primeiro e cai para o menor `.[termux]` (depois uma instalação base) — o instalador curl corresponde a esta ordem automaticamente
- vincula `hermes` em `$PREFIX/bin` para que permaneça no seu PATH do Termux
- pula a configuração não testada do navegador / WhatsApp

Se você quiser os comandos explícitos ou precisar depurar uma instalação falha, use o caminho manual abaixo.

---

## Opção 2: Instalação Manual (totalmente explícita)

### 1. Atualize o Termux e instale pacotes do sistema

```bash
pkg update
pkg install -y git python clang rust make pkg-config libffi openssl nodejs ripgrep ffmpeg
```

Por que estes pacotes?

- `python` — runtime + suporte a venv
- `git` — clonar/atualizar o repositório
- `clang`, `rust`, `make`, `pkg-config`, `libffi`, `openssl` — necessários para compilar algumas dependências Python no Android
- `nodejs` — runtime Node opcional para experimentos além do caminho principal testado
- `ripgrep` — busca rápida em arquivos
- `ffmpeg` — conversões de mídia / TTS

### 2. Clone o Hermes

```bash
git clone https://github.com/NousResearch/hermes-agent.git
cd hermes-agent
```

### 3. Crie um ambiente virtual

```bash
python -m venv venv
source venv/bin/activate
export ANDROID_API_LEVEL="$(getprop ro.build.version.sdk)"
python -m pip install --upgrade pip setuptools wheel
```

`ANDROID_API_LEVEL` é importante para pacotes baseados em Rust / maturin como `jiter`.

### 4. Instale o bundle testado do Termux

```bash
python -m pip install -e '.[termux]' -c constraints-termux.txt
```

Se você quiser apenas o agente principal mínimo, isso também funciona:

```bash
python -m pip install -e '.' -c constraints-termux.txt
```

### 5. Coloque `hermes` no seu PATH do Termux

```bash
ln -sf "$PWD/venv/bin/hermes" "$PREFIX/bin/hermes"
```

`$PREFIX/bin` já está no PATH no Termux, então isso faz o comando `hermes` persistir entre novos shells sem reativar o venv toda vez.

### 6. Verifique a instalação

```bash
hermes version
hermes doctor
```

### 7. Inicie o Hermes

```bash
hermes
```

---

## Configuração de acompanhamento recomendada

### Configure um modelo

```bash
hermes model
```

Ou defina chaves diretamente em `~/.hermes/.env`.

### Execute o assistente de configuração interativo completo depois

```bash
hermes setup
```

### Instale dependências Node opcionais manualmente

O caminho testado do Termux pula a configuração do Node/navegador de propósito. Se você quiser experimentar com ferramentas de navegador depois:

```bash
pkg install nodejs-lts
npm install
```

A ferramenta de navegador inclui automaticamente diretórios do Termux (`/data/data/com.termux/files/usr/bin`) em sua busca PATH, então `agent-browser` e `npx` são descobertos sem qualquer configuração extra de PATH.

Trate as ferramentas de navegador / WhatsApp no Android como experimentais até que seja documentado o contrário.

---

## Solução de Problemas

### `No solution found` ao instalar `.[all]`

Use o bundle testado do Termux:

```bash
python -m pip install -e '.[termux]' -c constraints-termux.txt
```

O bloqueador é atualmente o extra `voice`:

- `voice` puxa `faster-whisper`
- `faster-whisper` depende de `ctranslate2`
- `ctranslate2` não publica wheels para Android

### `uv pip install` falha no Android

Use o caminho Termux com o venv da stdlib + `pip`:

```bash
python -m venv venv
source venv/bin/activate
export ANDROID_API_LEVEL="$(getprop ro.build.version.sdk)"
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e '.[termux]' -c constraints-termux.txt
```

### `jiter` / `maturin` reclama sobre `ANDROID_API_LEVEL`

Defina o nível da API explicitamente antes de instalar:

```bash
export ANDROID_API_LEVEL="$(getprop ro.build.version.sdk)"
python -m pip install -e '.[termux]' -c constraints-termux.txt
```

### `hermes doctor` diz que ripgrep ou Node está faltando

Instale-os com pacotes Termux:

```bash
pkg install ripgrep nodejs
```

### Falhas de compilação ao instalar pacotes Python

Certifique-se de que o toolchain de compilação está instalado:

```bash
pkg install clang rust make pkg-config libffi openssl
```

Depois tente novamente:

```bash
python -m pip install -e '.[termux]' -c constraints-termux.txt
```

---

## Limitações conhecidas em telefones

- O backend Docker não está disponível
- A transcrição de voz local via `faster-whisper` não está disponível no caminho testado
- A configuração de automação de navegador é intencionalmente pulada pelo instalador
- Alguns extras opcionais podem funcionar, mas apenas `.[termux]` e `.[termux-all]` são atualmente documentados como os bundles Android testados

Se você encontrar um problema específico do Android, por favor abra uma issue no GitHub com:

- sua versão do Android
- `termux-info`
- `python --version`
- `hermes doctor`
- o comando de instalação exato e a saída de erro completa
