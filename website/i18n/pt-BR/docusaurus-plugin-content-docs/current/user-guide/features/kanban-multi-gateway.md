---
title: "Implantação do Kanban em vários gateways"
description: "Usar um quadro Kanban em vários gateways por perfil: um dispatcher e entrega pertencente ao perfil"
---

# Implantação em vários gateways

O Hermes aceita vários processos de gateway executados simultaneamente — um por
perfil (default, writer, admin, coder, researcher). Cada gateway abre sua
própria conexão com as APIs das plataformas e entrega mensagens aos assinantes
do seu perfil.

As assinaturas de tarefas também abrangem feedback de revisão. Um evento de
revisão `changes_requested` é entregue como uma notificação acionável de
REVISÃO-BLOQUEADA. Assinaturas com `notify+wake` também despertam o chat,
thread ou sessão de origem exatos para que o controlador inspecione o cartão e
a execução atuais; `notify` continua sendo somente passivo e `wake` continua
sendo somente wake. O feedback de revisão nunca cria, desbloqueia, recoloca na
fila nem modifica uma tarefa.

## Modelo de dispatcher único

Apenas um gateway é dono do dispatcher do Kanban. O gateway proprietário mantém
`kanban.dispatch_in_gateway: true` (o padrão); todos os demais definem esse
valor como `false`.

**Por que isso importa:** o dispatch tem um único proprietário, evitando que
vários gateways concorram para iniciar o mesmo trabalho. A entrega de
notificações pertence ao perfil: cada gateway consulta apenas assinaturas dos
perfis cujos adaptadores de plataforma ele hospeda. A reivindicação atômica do
evento evita entregas duplicadas entre processos observadores.

## Configuração

No gateway que possui o dispatch (normalmente o perfil `default`), nenhuma
alteração é necessária. Em cada outro gateway de perfil, adicione a
`~/.hermes/config.yaml`:

```yaml
kanban:
  dispatch_in_gateway: false
```

Ou defina a variável de ambiente:
`HERMES_KANBAN_DISPATCH_IN_GATEWAY=false`

## O que cada gateway faz

| Papel do gateway | dispatch_in_gateway | Abre os bancos dos quadros assinados? | Dispatcher | Notificador |
|---|---|---|---|---|
| default (proprietário confirmado do lock de dispatch) | true (padrão) | sim | sim | perfis próprios + assinaturas legadas sem marcação |
| writer, admin, coder etc. | false | sim, quando o perfil tem assinaturas | não | perfis pertencentes a esse gateway |

Gateways que não fazem dispatch ainda entregam mensagens de seus próprios
adaptadores (Telegram, Discord etc.). Eles não distribuem tarefas e ignoram
quadros que não tenham assinaturas pertencentes aos seus perfis.
