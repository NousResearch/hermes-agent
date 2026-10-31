# Command Helper Secret Source

Resolva credenciais executando seu próprio comando helper na inicialização — qualquer cofre de segredos com uma CLI funciona: `keepassxc-cli`, `secret-tool` (GNOME Keyring), `pass`, `gpg`, a CLI do Vaultwarden, ou um script que faz cat em um arquivo env em tmpfs. O helper imprime linhas `KEY=VALUE` no stdout; o Hermes as aplica através do mesmo orquestrador que o [Bitwarden](./bitwarden) e o [1Password](./onepassword), então você pode habilitar qualquer combinação de fontes simultaneamente.

## Como funciona {#how-it-works}

1. Você configura um comando helper em `config.yaml` (nunca em `.env` — o comando é configuração, `.env` guarda valores).
2. Na inicialização, depois que o `.env` é carregado, o Hermes executa o helper UMA VEZ via `/bin/sh -c` e faz parse do seu stdout como um blob dotenv.
3. As chaves parseadas seguem a escada de precedência padrão: `.env`/shell vencem a menos que `override_existing: true`; fontes mapeadas vencem essa fonte em massa em variáveis disputadas; a primeira a reivindicar vence.

```yaml
secrets:
  command:
    enabled: true
    command: "cat /run/user/1000/hermes-secrets.env"
    # or any vault CLI that dumps KEY=VALUE lines:
    # command: "pass show hermes/env"
    # command: "secret-tool lookup service hermes-env"
```

## Config {#config}

| Chave | Padrão | O que faz |
|---|---|---|
| `enabled` | `false` | Interruptor principal. |
| `command` | `""` | Executado via `/bin/sh -c`; deve imprimir linhas `KEY=VALUE` no stdout. |
| `helper_timeout_seconds` | `3` | Timeout rígido para uma execução do helper. Deliberadamente apertado — o helper precisa ser rápido e NÃO interativo (sem prompts de unlock, sem toque/PIN). |
| `override_existing` | `false` | Valores do helper sobrescrevem valores de `.env`/shell. Desativado por padrão (diferente do Bitwarden/1Password) já que um helper local não é uma autoridade central de rotação. |

## Modelo de segurança {#security-model}

- A string do comando helper é SUA configuração — o mesmo nível de confiança do arquivo `.env` que você controla.
- A saída tem um limite rígido de 1 MiB; um helper descontrolado não consegue travar a inicialização (o grupo de processos é morto no timeout).
- O **stderr do helper é descartado** — diagnósticos de CLIs de cofre podem carregar material secreto, então eles nunca chegam à saída do Hermes. Falhas registram apenas campos estruturados (código de saída / sinal / errno), nunca a string do comando.
- Valores contendo apenas espaços em branco são tratados como "sem valor" — uma entrada de placeholder nunca flui para um header Authorization.
- Somente POSIX (precisa de `/bin/sh`). No Windows, a fonte se reporta como não configurada e a inicialização continua.

## Modos de falha {#failure-modes}

A inicialização nunca é bloqueada. Erros imprimem uma linha mais uma dica de correção `→`:

| Sintoma | Causa | Correção |
|---|---|---|
| `secrets.command.command is empty` | Habilitado sem um comando | Defina `secrets.command.command` em config.yaml |
| `helper command failed` | Saída diferente de zero, timeout, falha ao iniciar processo | Execute o helper manualmente em um shell para ver seu erro real (o Hermes descarta seu stderr de propósito) |
| `helper output was not a KEY=VALUE map` | O helper imprimiu um valor bruto ou lixo | Faça o helper emitir linhas no formato dotenv |

## Quando usar isso em vez de um plugin {#when-to-use-this-vs-a-plugin}

A fonte de comando é a válvula de escape para cofres sem uma integração inclusa. Se você se pegar embrulhando uma dança complexa de CLI em um script longo, considere em vez disso um [plugin de fonte de segredo](/developer-guide/secret-source-plugin) de verdade — plugins ganham cache, rótulos de proveniência e config tipada.
