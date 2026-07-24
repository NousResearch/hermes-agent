---
sidebar_position: 3
sidebar_label: "Git Worktrees"
title: "Git Worktrees"
description: "Execute vários agentes Hermes com segurança no mesmo repositório usando git worktrees e checkouts isolados"
---

# Git Worktrees

O Hermes Agent é frequentemente usado em repositórios grandes e de longa duração. Quando você quer:

- Executar **vários agentes em paralelo** no mesmo projeto, ou
- Manter refatorações experimentais isoladas da sua branch principal,

Git **worktrees** são a forma mais segura de dar a cada agente seu próprio checkout sem duplicar o repositório inteiro.

Esta página mostra como combinar worktrees com o Hermes para que cada sessão tenha um diretório de trabalho limpo e isolado.

## Por que usar worktrees com o Hermes?

O Hermes trata o **diretório de trabalho atual** como a raiz do projeto:

- CLI: o diretório onde você executa `hermes` ou `hermes chat`
- Gateways de mensagens: o diretório definido por `terminal.cwd` em `~/.hermes/config.yaml`

Se você executar vários agentes no **mesmo checkout**, as mudanças deles podem interferir uns nos outros:

- Um agente pode excluir ou reescrever arquivos que o outro está usando.
- Fica mais difícil entender quais mudanças pertencem a qual experimento.

Com worktrees, cada agente recebe:

- Seu **próprio branch e diretório de trabalho**
- Seu **próprio histórico do Checkpoint Manager** para `/rollback`

Veja também: [Checkpoints e /rollback](./checkpoints-and-rollback.md).

## Início rápido: criando um worktree

Do seu repositório principal (contendo `.git/`), crie um worktree novo para uma branch de feature:

```bash
# From the main repo root
cd /path/to/your/repo

# Create a new branch and worktree in ../repo-feature
git worktree add ../repo-feature feature/hermes-experiment
```

Isso cria:

- Um diretório novo: `../repo-feature`
- Uma branch nova: `feature/hermes-experiment` checked out naquele diretório

Agora você pode `cd` para o worktree novo e executar o Hermes lá:

```bash
cd ../repo-feature

# Start Hermes in the worktree
hermes
```

O Hermes irá:

- Ver `../repo-feature` como a raiz do projeto.
- Usar aquele diretório para arquivos de contexto, edições de código e ferramentas.
- Usar um **histórico de checkpoint separado** para `/rollback` escopado a este worktree.

## Executando vários agentes em paralelo

Você pode criar vários worktrees, cada um com sua própria branch:

```bash
cd /path/to/your/repo

git worktree add ../repo-experiment-a feature/hermes-a
git worktree add ../repo-experiment-b feature/hermes-b
```

Em terminais separados:

```bash
# Terminal 1
cd ../repo-experiment-a
hermes

# Terminal 2
cd ../repo-experiment-b
hermes
```

Cada processo Hermes:

- Trabalha em sua própria branch (`feature/hermes-a` vs `feature/hermes-b`).
- Grava checkpoints sob um hash de shadow repo diferente (derivado do caminho do worktree).
- Pode usar `/rollback` independentemente sem afetar o outro.

Isso é especialmente útil quando:

- Executa refatorações em lote.
- Tenta abordagens diferentes para a mesma tarefa.
- Combina sessões CLI + gateway contra o mesmo repo upstream.

## Limpando worktrees com segurança

Quando terminar um experimento:

1. Decida se quer manter ou descartar o trabalho.
2. Se quiser manter:
   - Faça merge da branch na sua branch principal como de costume.
3. Remova o worktree:

```bash
cd /path/to/your/repo

# Remove the worktree directory and its reference
git worktree remove ../repo-feature
```

Notas:

- `git worktree remove` recusará remover um worktree com mudanças não commitadas a menos que você force.
- Remover um worktree **não** exclui automaticamente a branch; você pode excluir ou manter a branch com comandos `git branch` normais.
- Dados de checkpoint Hermes em `~/.hermes/checkpoints/` não são podados automaticamente quando você remove um worktree, mas costumam ser muito pequenos.

## Boas práticas

- **Um worktree por experimento Hermes**
  - Crie uma branch/worktree dedicada para cada mudança substancial.
  - Isso mantém diffs focados e PRs pequenos e revisáveis.
- **Nomeie branches pelo experimento**
  - ex.: `feature/hermes-checkpoints-docs`, `feature/hermes-refactor-tests`.
- **Commit com frequência**
  - Use commits git para marcos de alto nível.
  - Use [checkpoints e /rollback](./checkpoints-and-rollback.md) como rede de segurança para edições dirigidas por ferramentas entre eles.
- **Evite executar Hermes da raiz bare do repo quando usa worktrees**
  - Prefira os diretórios worktree, para cada agente ter escopo claro.

## Usando `hermes -w` (modo worktree automático)

O Hermes tem uma flag built-in `-w` que **cria automaticamente um git worktree descartável** com sua própria branch. Você não precisa configurar worktrees manualmente — basta `cd` no seu repo e executar:

```bash
cd /path/to/your/repo
hermes -w
```

O Hermes irá:

- Criar um worktree temporário em `.worktrees/` dentro do seu repo.
- Fazer checkout de uma branch isolada (ex.: `hermes/hermes-<hash>`).
- Executar a sessão CLI completa dentro daquele worktree.

Esta é a forma mais fácil de obter isolamento por worktree. Você também pode combinar com uma única query:

```bash
hermes -w -z "Fix issue #123"
```

Para agentes paralelos, abra vários terminais e execute `hermes -w` em cada um — toda invocação recebe seu próprio worktree e branch automaticamente.

## Juntando tudo

- Use **git worktrees** para dar a cada sessão Hermes seu próprio checkout limpo.
- Use **branches** para capturar o histórico de alto nível dos seus experimentos.
- Use **checkpoints + `/rollback`** para se recuperar de erros dentro de cada worktree.

Esta combinação oferece:

- Garantias fortes de que agentes e experimentos diferentes não pisam uns nos outros.
- Ciclos de iteração rápidos com recuperação fácil de edições ruins.
- Pull requests limpos e revisáveis.

## Desenvolvendo as superfícies de UI a partir de worktrees

As superfícies TypeScript (`ui-tui/`, `apps/desktop/`) precisam cada uma de um `node_modules`, que um `npm ci` fresco por worktree duplica em cada branch. Se você desenvolve o TUI ou app desktop a partir de vários worktrees, veja [TUI & Desktop from Worktrees](../developer-guide/worktree-ui-dev.md) para os helpers `htui` / `hgui` que compartilham uma instalação por symlink.
