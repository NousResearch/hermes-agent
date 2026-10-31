---
title: "Recuperação do armazenamento de sessões"
description: "O que fazer quando o Hermes informa que outro processo mantém uma cópia antiga do log de gravação do banco de dados de sessões, e o que são os arquivos ao lado de state.db"
---

# Recuperação do armazenamento de sessões

O Hermes mantém todas as conversas em um arquivo SQLite por perfil, `state.db`, com dois arquivos auxiliares gerenciados pelo SQLite: `state.db-wal` (o log de gravação antecipada) e `state.db-shm`. Vários processos do Hermes podem compartilhar esse arquivo com segurança — o gateway, o aplicativo Desktop, o dashboard, o cron e os comandos da CLI escrevem usando o próprio mecanismo de bloqueio do SQLite.

Uma coisa não é segura: **reescrever o armazenamento enquanto outro processo está gravando nele**. Quando isso acontece, os processos que ainda mantêm a cópia antiga do log param de gravar de propósito, e cada turno responde com uma mensagem semelhante a:

> outro processo do Hermes ainda mantém uma cópia antiga do log de gravação antecipada do banco de dados de sessões, então o Hermes parou de gravar para manter o arquivo seguro …

Esta página é o guia vinculado por essa mensagem. Nada é perdido quando você a vê; a recusa existe justamente para evitar perdas.

## A correção em três etapas

1. **Feche todos os processos do Hermes nesse perfil.** Desktop, gateway, dashboard e cron:

   ```bash
   hermes gateway stop          # adicione -p <perfil> para um perfil nomeado
   ```

   Em seguida, feche o aplicativo Desktop pelo menu e pare qualquer dashboard (`hermes dashboard --stop`) ou serviço personalizado que você execute. Reiniciar apenas um deles não basta — um único processo que continue mantendo o log antigo fará com que todos os novos processos recusem a gravação.

2. **Peça ao doctor para identificar quem ainda mantém o log.**

   ```bash
   hermes doctor                # adicione -p <perfil> para um perfil nomeado
   ```

   Enquanto qualquer processo ainda mantiver o log antigo, o doctor exibirá cada responsável como `PID N (comando)` com a mesma orientação, e ignorará suas sondagens de saúde e qualquer operação `--fix`, para não se tornar outro gravador. Pare os processos listados e execute o comando novamente até que a linha desapareça.

3. **Inicie o Hermes novamente** (primeiro apenas um processo — o gateway ou o aplicativo Desktop) e envie sua mensagem mais uma vez. A conversa continuará de onde parou.

## Não faça isto

- **Não execute `hermes doctor --fix` enquanto os processos estiverem em execução.** O doctor recusa o checkpoint quando consegue ver um processo mantendo o log antigo; porém, em um host onde não consegue inspecionar os processos, o caminho de correção será exatamente o segundo gravador que causou o problema.
- **Não exclua `state.db-wal` nem `state.db-shm`.** O log contém conversas confirmadas que ainda não foram incorporadas ao `state.db`. Excluí-lo é a ação que transforma uma recusa em perda real de dados.
- **Não copie apenas `state.db`.** Os três arquivos formam uma única imagem. Use um snapshot (`hermes backup`) ou `hermes sessions recover`, nunca `cp state.db algum-lugar/`.
- **Não peça ao agente para corrigir o problema.** A sessão do próprio agente está no mesmo armazenamento e encontrará a mesma recusa.

## Comandos de manutenção recusam a operação enquanto alguém grava

`hermes sessions optimize`, `hermes sessions optimize-storage` e `hermes sessions prune` reescrevem o armazenamento (VACUUM, reconstrução completa do índice de texto e exclusões em massa). Executar um deles com um gateway ativo pode fazer uma frota de agentes responder com o erro de log antigo, por isso eles agora verificam antes e recusam enquanto outro processo mantém o banco de dados:

```text
Recusando `hermes sessions optimize-storage`: outro processo está usando ~/.hermes/state.db.
  PID 41230 (hermes gateway run): state.db, state.db-shm, state.db-wal
  PID 41355 (hermes serve --profile work): state.db-wal
Reescrever o banco de dados enquanto há um gravador ativo é o que faz todos os agentes recusarem turnos com o erro de state.db-wal antigo.
Nada foi perdido.
Pare-os primeiro (`hermes gateway stop`, feche o aplicativo Desktop, pause o cron) e execute novamente.
Use --force apenas se aceitar o risco.
```

Pré-visualizações `--dry-run` nunca são bloqueadas. `--force` executa mesmo assim — use-o apenas quando souber que os processos listados estão ociosos (por exemplo, um leitor que você mesmo iniciou). A mesma verificação é executada quando você digita `sessions optimize` no console do Desktop.

## Arquivos que podem aparecer ao lado de `state.db`

| Arquivo ou diretório | O que é | O que fazer |
|---|---|---|
| `state.db-wal`, `state.db-shm` | O log de gravação antecipada ativo do SQLite e seu índice de memória compartilhada. Um `-wal` grande é normal enquanto o gateway ou o Desktop está em execução. | Não mexa neles. Eles diminuem sozinhos no próximo checkpoint. |
| `state.db.retired-wal-<timestamp>-<pid>/` | Uma captura que o Hermes fez da cópia do log ainda mantida por um processo quando se recusou a gravar, junto com um `manifest.json` descritivo. É evidência forense, não um backup para restaurar cegamente. | Guarde-o. Se conversas imediatamente anteriores ao incidente estiverem faltando após a recuperação, anexe o diretório a um relatório de bug; um mantenedor poderá verificar no `manifest.json` se os frames pertencem ao arquivo atual. |
| `state.db.pre-update-emergency-<timestamp>.bak` | Um snapshot que o atualizador do Desktop cria antes de alterar o armazenamento. | Guarde-o até usar o aplicativo atualizado por algum tempo. Restaure-o somente com todos os processos do Hermes parados: primeiro execute `hermes sessions recover --source <arquivo> --inspect-only`. |
| `state.db.corrupt.<timestamp>.bak`, `*.malformed-backup` | Cópias de um arquivo que o Hermes encontrou danificado antes de repará-lo ou colocá-lo em quarentena. | Não as restaure sobre `state.db` — elas contêm o mesmo dano. Guarde-as para um relatório; exclua-as com segurança quando tudo voltar ao normal. |
| `state-snapshots/` | Snapshots rápidos criados por `hermes update` e `hermes backup`. | Restaure-os com todos os processos do Hermes parados; consulte [`hermes backup`](../reference/cli-commands.md#hermes-backup). |

## Quando as três etapas não funcionarem

Se todos os processos do Hermes estiverem parados, o `hermes doctor` não listar mais nenhum responsável e o gateway ainda recusar gravações ao ser iniciado, o próprio arquivo pode estar danificado. Pare tudo novamente e inspecione sem gravar:

```bash
hermes sessions recover --source ~/.hermes/state.db --inspect-only
```

`--inspect-only` nunca modifica o arquivo. Se ele informar que o armazenamento pode ser recuperado, siga o comando exibido ou restaure o snapshot mais recente de `state-snapshots/`. Os detalhes estão no guia do desenvolvedor:
[Recuperação do banco de dados de estado](../developer-guide/state-db-recovery.md) e [Armazenamento de sessões](../developer-guide/session-storage.md).
