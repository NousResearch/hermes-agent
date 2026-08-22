---
sidebar_position: 3
title: "Escopo gerenciado"
description: "Configuração e segredos fixados pelo administrador e imutáveis para o usuário, via um diretório gerenciado em nível de sistema"
---

# Escopo gerenciado

O **escopo gerenciado** permite que um administrador imponha uma linha de base de configuração e
segredos que um usuário comum (não root) **não pode sobrescrever**. É destinado a
implantações em frota/organização em que a TI precisa fixar, por exemplo, o provedor de modelo, uma
URL base de API compartilhada ou `security.redact_secrets: true` para todos os usuários de uma
máquina.

Quando um escopo gerenciado está presente, os valores que ele especifica prevalecem sobre o
`~/.hermes/config.yaml`, o `~/.hermes/.env` e até o ambiente do shell do usuário — para
exatamente as chaves que ele fixa. Todo o resto permanece totalmente sob controle do usuário.

:::note Diferente de uma instalação bloqueada pelo gerenciador de pacotes
Uma instalação gerenciada por gerenciador de pacotes (distribuição declarativa / formula) bloqueia *toda*
mutação de configuração e orienta você a usar o gerenciador de pacotes. O escopo gerenciado é um
mecanismo separado: ele injeta *valores imutáveis específicos* por chave,
em vez de bloquear toda a configuração. Os dois são independentes e podem coexistir.
:::

## Onde fica

O escopo gerenciado é lido de um diretório em nível de sistema, por padrão `/etc/hermes`:

```text
/etc/hermes/
├── config.yaml     # camada de config gerenciada (prevalece sobre ~/.hermes/config.yaml)
└── .env            # camada de env gerenciada (prevalece sobre ~/.hermes/.env + shell)
```

O diretório e os arquivos pertencem ao `root` (modo do diretório `0755`, arquivos
`0644`): legíveis por todos, graváveis apenas por um administrador. **Essa
permissão de filesystem é o mecanismo de enforcement** — um usuário comum pode ler
os arquivos gerenciados, mas não pode editá-los.

Cada arquivo é opcional. Um diretório gerenciado ausente ou um arquivo ausente simplesmente
significa "sem escopo gerenciado", e a configuração é resolvida exatamente como sem
o recurso.

### Realocando o diretório

A localização pode ser alterada com a variável de ambiente `HERMES_MANAGED_DIR`
(para containers ou implantações fora de `/etc`). Este é um knob de caminho de
bootstrap/implantação — como `HERMES_HOME` — definido pelo mesmo administrador que possui os arquivos
gerenciados. **Nunca é persistido** em nenhum `.env` pelo Hermes.

```bash
# Apontar o escopo gerenciado para um diretório personalizado (definido pela TI / implantação, não pelo usuário)
export HERMES_MANAGED_DIR=/opt/org/hermes-policy
```

:::warning
Um usuário que pode definir `HERMES_MANAGED_DIR` pode redirecionar o escopo gerenciado para um diretório
sob seu controle, anulando-o. Em uma implantação real, essa variável deve ser fixada
pelo administrador (por exemplo, embutida na unit de serviço / imagem de container), não
deixada configurável pelo usuário. `hermes doctor` reporta o diretório gerenciado *resolvido*, para que
um redirecionamento seja visível.
:::

## Precedência

Para as chaves que uma camada gerenciada especifica, a ordem é (maior prevalece):

| Nível | config.yaml | .env |
|---|---|---|
| 1 | `/etc/hermes/config.yaml` (gerenciado) | `/etc/hermes/.env` (gerenciado) |
| 2 | `~/.hermes/config.yaml` (usuário) | `~/.hermes/.env` (usuário) |
| 3 | padrões embutidos | ambiente de shell pré-existente |

A mesclagem é **no nível da folha**: fixar `model.default` não congela o restante de
`model.*`. Um `config.yaml` gerenciado com:

```yaml
model:
  default: org/standard-model
```

força `model.default` para todos os usuários, deixando `model.fallback` (e toda
outra chave) sob controle do usuário.

:::note Nota sobre precedência
Para as chaves que fixa, o escopo gerenciado deliberadamente prevalece também sobre o ambiente de shell
— caso contrário, não seria "gerenciado". Este é o único lugar que inverte a
regra usual de "uma variável de ambiente sobrescreve config.yaml", e se aplica apenas
às chaves específicas que a camada gerenciada especifica.
:::

## Ver o que é gerenciado

```bash
hermes config        # mostra um cabeçalho com a origem gerenciada + as chaves fixadas
hermes doctor        # reporta o dir gerenciado resolvido + contagens de chaves fixadas
```

Se você tentar alterar um valor gerenciado, o Hermes recusa e informa a origem:

```bash
$ hermes config set model.default my/model
Cannot set 'model.default': it is managed by your administrator
(/etc/hermes/config.yaml) and cannot be changed.
```

O mesmo vale para segredos gerenciados — `hermes config set` / setup não gravará
um valor de usuário para uma chave de env fixada pelo `.env` gerenciado.

## Configurando um escopo gerenciado (administradores)

```bash
sudo mkdir -p /etc/hermes

# Fixar alguns valores de config para todos os usuários desta máquina
sudo tee /etc/hermes/config.yaml >/dev/null <<'YAML'
model:
  provider: nous
security:
  redact_secrets: true
YAML

# Opcionalmente fixar um valor de env compartilhado e não sensível
sudo tee /etc/hermes/.env >/dev/null <<'ENV'
OPENAI_API_BASE=https://inference.example.com/v1
ENV

sudo chmod 0755 /etc/hermes
sudo chmod 0644 /etc/hermes/config.yaml /etc/hermes/.env
```

As alterações entram em vigor na próxima inicialização do Hermes (um arquivo gerenciado malformado é registrado
de forma evidente e ignorado — nunca bloqueia a inicialização, mas o admin deve verificar
`hermes doctor` para confirmar que a política está sendo aplicada).

## Modelo de segurança e limitações (v1)

- **O enforcement é apenas permissões de filesystem.** Se um usuário tem acesso de escrita ao
  diretório gerenciado (ou executa o Hermes como `root`), o escopo gerenciado é consultivo.
- **O `.env` gerenciado é legível por todos** (`0644`), então qualquer usuário local pode ler
  segredos enviados por ele. Use-o para valores compartilhados e não sensíveis (uma URL base de API
  da org, padrões de recursos) em vez de segredos de alta sensibilidade.
- **As ferramentas do próprio agente não são bloqueadas de forma rígida contra um valor de *env* gerenciado.** Uma
  variável de ambiente gerenciada é aplicada na inicialização, mas nada impede o
  agente de definir um valor diferente dentro do próprio shell de subprocesso. A v1 é um
  limite de conveniência de gestão contra um usuário comum, não um sandbox inescapável.

Os itens a seguir estão intencionalmente **fora do escopo da v1** e podem vir depois:

- Um limite rígido que o próprio agente não possa escapar.
- Locais gerenciados nativos no macOS e Windows (a v1 é Linux/POSIX-first).
- Diretórios de fragmentos drop-in (`managed.d/`) para política em camadas.
- Arquivos gerenciados assinados / com verificação de integridade.
- Entrega remota / device management (MDM).
- Permissões mais restritas (escopo de grupo) para segredos gerenciados.
