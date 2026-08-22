---
sidebar_position: 3
title: "Android / Termux"
description: "Rode o Hermes Agent direto num celular Android com Termux"
---

# Hermes no Android com Termux

:::warning Plataforma Tier 2
Termux (Android) é uma [plataforma Tier 2](./platform-support.md#tier-2). O script de instalação e a documentação aqui são mantidos só no melhor esforço. Commits em `main` podem quebrar esses pacotes a qualquer momento.
:::

O Hermes Agent pode rodar direto num celular Android pelo [Termux](https://termux.dev/).

Você ganha um CLI local funcionando no telefone, mais os extras essenciais que hoje se sabe instalar limpo no Android.

## O que é suportado no caminho testado?

O bundle Termux testado instala:

- o CLI do Hermes
- suporte a cron
- suporte a terminal PTY/background
- suporte ao gateway Telegram (runs manuais / background best-effort)
- suporte a MCP
- suporte a memória Honcho
- suporte a ACP

Na prática, isso mapeia para:

```bash
python -m pip install -e '.[termux]' -c constraints-termux.txt
```

## O que ainda não faz parte do caminho testado?

Alguns recursos ainda precisam de dependências no estilo desktop/servidor que não são publicadas para Android, ou ainda não foram validados em telefones:

- `.[all]` não é suportado no Android hoje
- o extra `voice` é bloqueado por `faster-whisper -> ctranslate2`, e o `ctranslate2` não publica wheels Android
- o bootstrap automático de browser / Playwright é pulado no instalador Termux
- isolamento de terminal baseado em Docker não está disponível dentro do Termux
- o Android ainda pode suspender jobs em background do Termux, então a persistência do gateway é best-effort, não um serviço gerenciado normal

Isso não impede o Hermes de funcionar bem como agente CLI nativo no telefone — só significa que o install mobile recomendado é propositalmente mais estreito que o install desktop/servidor.

---

## Opção nativa com `pkg` mantida pela comunidade {#community-maintained-native-pkg-option}

:::caution Distribuição operada por contributor
Este repositório APT é **mantido pela comunidade por `@adybag14-cyber` e não é uma distribuição oficial da NousResearch**. A NousResearch não builda, assina, hospeda nem audita esses pacotes. Habilitar o repositório significa confiar no repositório operado pelo contributor e na sua chave de assinatura. O Termux em si continua sendo uma plataforma Tier 2 / best-effort.
:::

Para usuários que preferem um install via package manager nativo em vez de buildar dependências Python/Rust no telefone, há um repositório APT mantido pela comunidade. O bootstrap do repositório e as fontes de packaging estão publicados em [`adybag14-cyber/termux-python`](https://github.com/adybag14-cyber/termux-python), com o build do pacote Hermes em [`adybag14-cyber/termux-hermes`](https://github.com/adybag14-cyber/termux-hermes).

Instale a chave/fonte do repositório e o Hermes com:

```bash
curl -fsSL https://raw.githubusercontent.com/adybag14-cyber/termux-python/main/scripts/setup_apt_repo.sh | bash
pkg install hermes-agent
```

A fingerprint da chave de assinatura do repositório documentada atualmente pela distribuição da comunidade é:

```text
EAD24A2124EFA7393A78B7B14699F966313F7A6B
```

Installs Hermes gerenciados por APT são marcados com o método de install `apt`. Por isso o Hermes não roda o self-updater Git contra arquivos de propriedade do pacote; use o package manager:

```bash
pkg update
pkg upgrade hermes-agent
```

Problemas de packaging/repositório/assinatura dessa opção devem ser reportados aos repositórios de packaging da comunidade acima. Bugs de runtime do Hermes ainda podem ser reportados aqui, lembrando que o suporte Android/Termux é best-effort.

---

## Opção 1: Instalador de uma linha

O Hermes agora tem um caminho de instalador ciente de Termux:

```bash
curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash
```

No Termux, o instalador automaticamente:

- usa `pkg` para pacotes do sistema
- cria o venv com `python -m venv`
- tenta primeiro o extra amplo `.[termux-all]` e faz fallback para o menor `.[termux]` (depois um install base) — o instalador curl segue essa ordem automaticamente
- linka `hermes` em `$PREFIX/bin` para ficar no PATH do Termux
- pula o bootstrap não testado de browser / WhatsApp

Se quiser os comandos explícitos ou precisar debugar um install falho, use o caminho manual abaixo.

---

## Opção 2: Install manual (totalmente explícito)

### 1. Atualize o Termux e instale pacotes do sistema

```bash
pkg update
pkg install -y git python clang rust make pkg-config libffi openssl nodejs ripgrep ffmpeg
```

Por que esses pacotes?

- `python` — runtime + suporte a venv
- `git` — clonar/atualizar o repo
- `clang`, `rust`, `make`, `pkg-config`, `libffi`, `openssl` — necessários para buildar algumas dependências Python no Android
- `nodejs` — runtime Node opcional para experimentos além do caminho core testado
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

### 4. Instale o bundle Termux testado

```bash
python -m pip install -e '.[termux]' -c constraints-termux.txt
```

Se quiser só o agente core mínimo, isto também funciona:

```bash
python -m pip install -e '.' -c constraints-termux.txt
```

### 5. Coloque `hermes` no PATH do Termux

```bash
ln -sf "$PWD/venv/bin/hermes" "$PREFIX/bin/hermes"
```

`$PREFIX/bin` já está no PATH no Termux, então isso faz o comando `hermes` persistir em shells novos sem reativar o venv toda vez.

### 6. Verifique o install

```bash
hermes --version
hermes doctor
```

### 7. Inicie o Hermes

```bash
hermes
```

---

## Setup de follow-up recomendado

### Configure um modelo

```bash
hermes model
```

Ou sete chaves direto em `~/.hermes/.env`.

### Rode de novo o wizard interativo completo depois

```bash
hermes setup
```

### Instale dependências Node opcionais na mão

O caminho Termux testado pula o bootstrap Node/browser de propósito. Se quiser experimentar tooling de browser depois, o que você precisa depende de qual backend usa:

- **Providers de browser na nuvem** (Browserbase, Browser Use, Firecrawl) hospedam o próprio Chromium, então só o Node.js basta — o `agent-browser` resolve lazily via `npx agent-browser` no primeiro uso:

  ```bash
  pkg install nodejs-lts
  ```

- **Automação local de browser** no Termux precisa de uma instalação real de `agent-browser` — o fallback npx nu é rejeitado de propósito em modo local por ser frágil demais para anunciar como pronto:

  ```bash
  pkg install nodejs-lts
  npm install -g agent-browser && agent-browser install
  ```

A tool de browser inclui automaticamente diretórios Termux (`/data/data/com.termux/files/usr/bin`) na busca de PATH, então `agent-browser` e `npx` são descobertos sem configuração extra de PATH.

Trate tooling de browser / WhatsApp no Android como experimental até documentação dizer o contrário.

---

## Troubleshooting

### `No solution found` ao instalar `.[all]`

Use o bundle Termux testado em vez disso:

```bash
python -m pip install -e '.[termux]' -c constraints-termux.txt
```

O bloqueio hoje é o extra `voice`:

- `voice` puxa `faster-whisper`
- `faster-whisper` depende de `ctranslate2`
- `ctranslate2` não publica wheels Android

### `uv pip install` falha no Android

Use o caminho Termux com venv stdlib + `pip`:

```bash
python -m venv venv
source venv/bin/activate
export ANDROID_API_LEVEL="$(getprop ro.build.version.sdk)"
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e '.[termux]' -c constraints-termux.txt
```

### `jiter` / `maturin` reclamam de `ANDROID_API_LEVEL`

Defina o API level explicitamente antes de instalar:

```bash
export ANDROID_API_LEVEL="$(getprop ro.build.version.sdk)"
python -m pip install -e '.[termux]' -c constraints-termux.txt
```

### `hermes doctor` diz que ripgrep ou Node está faltando

Instale com pacotes Termux:

```bash
pkg install ripgrep nodejs
```

### Falhas de build ao instalar pacotes Python

Garanta que a toolchain de build está instalada:

```bash
pkg install clang rust make pkg-config libffi openssl
```

Depois tente de novo:

```bash
python -m pip install -e '.[termux]' -c constraints-termux.txt
```

---

## Limitações conhecidas em telefones {#known-limitations-on-phones}

- Backend Docker indisponível
- Transcrição de voz local via `faster-whisper` indisponível no caminho testado
- Setup de automação de browser é propositalmente pulado pelo instalador
- Alguns extras opcionais podem funcionar, mas só `.[termux]` e `.[termux-all]` estão documentados hoje como bundles Android testados

Se bater num issue novo específico de Android, abra uma issue no GitHub com:

- sua versão do Android
- `termux-info`
- `python --version`
- `hermes doctor`
- o comando exato de install e a saída completa do erro
