---
title: Conselhos do Collective Wisdom
---

O Collective Wisdom permite que sua conversa ativa do Hermes explique como um
skill de equipe se encaixa no seu setup antes de pedir para instalar, atualizar
ou compartilhar. Resumos escritos pelo agente são o padrão quando o Collective
Wisdom está habilitado. Notificações fixas estão disponíveis como opt-out em
profile local.

## Escolher o texto das notificações {#choose-notification-copy}

Primeiro entre na sua equipe e complete a disclosure existente de
`hermes wisdom setup`. Profiles sem uma configuração de delivery-mode usam
conselhos escritos pelo agente. Para optar por sair, defina isto na configuração
daquele profile:

```yaml
wisdom:
  notifications:
    delivery_mode: fixed
```

Mantenha as outras configurações de Wisdom do profile. Reinicie o messaging
gateway dele e abra uma nova sessão local depois de atualizar o Hermes. Defina
`delivery_mode: agent` para voltar aos conselhos escritos pelo agente.
Configurações `fixed` já salvas são preservadas na atualização; mude esse valor
explicitamente para usar o novo padrão. Esta configuração não habilita
compartilhamento, não muda a política da organização nem as políticas de
atualização dos skills instalados.

## Conselho e consentimento {#advice-and-consent}

O Hermes usa o modelo da conversa selecionada e o contexto limitado da
conversa. Sua avaliação automática não pode rodar comandos de shell, navegar,
ler arquivos arbitrários, delegar ou instalar nada. Texto do publisher não é
confiável; utilidade e sobreposição são sugestões, não fatos de segurança ou
compatibilidade.

Uma conversa privada recente e autorizada recebe o conselho proativo.
Qualificações locais ficam com a conversa de origem. Se nenhuma conversa
elegível estiver ativa, a atividade espera. Outras superfícies mostram o mesmo
conselho de forma passiva; abri-las não roda outra avaliação.

A conversa ativa é uma sessão voltada ao usuário, não um subagente delegado ou
tarefa em background. Agentes delegados, seus subprocessos, jobs agendados e
forks de review não podem pedir cards de consentimento, se registrar como
sessões de entrega ativas, nem reivindicar o conselho enfileirado do pai. Eles
devolvem achados à conversa principal; só essa conversa pode pedir o controle
nativo de confirmação do usuário.

Quando vários eventos de publicação de um skill estão esperando, o Hermes
mantém só a versão mais nova elegível para conselho automático. Eventos
duplicados não produzem outra recomendação. Avisos históricos e recibos de
entrega continuam disponíveis, e reviews solicitados explicitamente não são
descartados. Uma chegada mais nova não pode retratar uma mensagem já enviada.

- Use **Review first** para inspecionar checks canônicos e requisitos.
- Use o controle nativo **Install**, **Update** ou de compartilhamento para
  consentir.
- No CLI nativo ou do Dashboard, use `/wisdom inbox`, depois a ação exata
  `/wisdom consent`. O comando standalone `hermes wisdom consent` exige uma
  confirmação interativa antes de aplicar.
- Um "sim" conversacional pede ao Hermes para apresentar o controle; não
  aplica a operação.
- **Not Now** suprime outras recomendações para este skill inalterado em
  todos os seus clients na organização. Não apaga o skill nem impede review
  manual. Mudanças offline ficam pendentes até sincronizarem com o Gateway.

O revisor de arquivo de pacote proposto também oferece **Not Now**. Mantém o
pacote disponível para review posterior sem fazer upload ou publicar.
Navegação de página, incluindo **Back to first page**, é somente leitura; a
publicação ainda exige o controle separado **Approve exact package**.

Novos pacotes de compartilhamento preparados pelo agente podem incluir
**Publisher usage (client-reported)** na descrição editável do autor. Isto é
um snapshot das invocações registradas do skill local e dos dias de uso
durante as sete datas de calendário do profile terminando no dia da
preparação. Pode abranger revisões locais e não verifica resultados bem-
sucedidos. Sem uso registrado, nenhum resumo é adicionado. Só totais e o
intervalo de datas entram, não caminhos, trechos de conversa, registros por
dia nem a racionalidade privada da qualificação.

O resumo fica local com o pacote preparado até você aprovar o upload. Revise-o
na descrição do autor, e edite ou remova antes de compartilhar. Editar muda o
hash de aprovação e exige uma confirmação nova. Invocações seguintes e
retentativas de packaging não atualizam um review existente. Se mantido, a
descrição aprovada acompanha aquela versão publicada exata para os colegas;
não é telemetria de uso ao vivo nem certificação do Gateway. A criação manual
de draft privado não adiciona automaticamente evidência de uso.

Conselhos proativos no Telegram e Slack incluem **Notification settings**.
Abrir lê o estado atual sem silenciar nada; escolha uma duração separadamente.
Use **Back** para voltar à inbox atual. Notificações locais oferecem
`/wisdom mute` para os mesmos controles. Silenciar não desabilita review
manual, instalação, atualizações ou compartilhamento.

Botões de recomendação mais antigos abrem o review atual ou as configurações
de notificação. Eles não podem iniciar packaging, instalar um skill ou mudar
preferências sozinhos. Revise e confirme com os novos controles; um botão
anterior não carrega o consentimento adiante. O comando de compatibilidade
`hermes wisdom act` igualmente retorna a navegação atual sem aplicar uma
operação.

Pacotes alterados, edições locais, requisitos adicionais e consentimento
expirado exigem review novo. Reabrir um card expirado ou seus checks mostra
**Expired** com **Recheck** em vez de um botão de confirmação. Recheck busca
um plano novo; não o aprova nem revive a confirmação antiga. Atualizações
automáticas já opt-in continuam inalteradas.

## Recuperação e limites {#recovery-and-limits}

Sair da Nous aposenta os conselhos inacabados e os controles de confirmação
pendentes deste profile. Entre e re-verifique sua equipe com
`hermes wisdom setup` antes de continuar no Wisdom; isto não revive o
conselho enfileirado antigo nem seus controles. Sair de um provider de modelo
não relacionado não desconecta o Wisdom. Operações concluídas e evidência de
entrega continuam disponíveis para recuperação. Um envio já despachado ainda
pode chegar depois do logout; o recibo é retido para settlement sem enviar o
card de novo.

O Wisdom também aposenta conselhos pendentes quando o refresh de credencial
registra que a sessão Nous foi revogada. Um client em cache não pode manter
esse conselho ativo. Expiração ordinária de token e falhas temporárias de
conexão/servidor não o cancelam. Avisos de feed em cache são aposentados no
logout sem apagar seu histórico ou recibos de entrega. Uma resposta de feed
iniciada antes do logout não pode restaurar esses avisos nem avançar o cursor
salvo, mesmo depois de re-verificar a mesma equipe. Re-verificar depois do
logout alcança o feed em silêncio antes de tornar o Wisdom ativo.
Anúncios acumulados enquanto desconectado não viram conselho novo; skills
publicados continuam disponíveis para navegar. Um catch-up com falha deixa o
profile não verificado, e rodar setup de novo retoma da página salva. Mudar
de equipe ou conta usa o escopo de feed daquela identidade, não o cursor
anterior. Isto não remove skills instalados nem descarta registros de
recuperação de operação e entrega; o estado atual de instalação e moderação
ainda se reconcilia normalmente.

Claims de avaliação são locais a um profile e organização, com leases de três
minutos e três tentativas limitadas. Sessões ociosas fazem poll no máximo uma
vez por minuto por profile/org, e o roteamento proativo usa uma janela de
atividade recente de dez minutos. O conselho é salvo antes de ser enviado.
Falhas de provider eventualmente produzem um aviso de review determinístico
em vez de uma recomendação inventada.

Se um envio de messaging der timeout depois que possa ter tido sucesso, o
Hermes não reenvia às cegas. O conselho permanece na inbox. Um apply
interrompido é reconciliado contra o journal exato da operação; resultados
ambíguos ficam visíveis para review em vez de serem aplicados de novo. Use os
comandos ordinários de setup/recuperação e review do Wisdom para operações
que ainda precisam de atenção.

A propriedade da avaliação permanece local ao profile. Reservas de entrega e
preferências de notificação são coordenadas pelo Gateway, sem sincronizar
nomes privados de candidatos, uso ou conselhos. Isto não é uma mailbox de
propósito geral. A review semanal do agente usa uso real existente de skills
e a política da organização; delivery fixo não roda esta review. Mudar o
texto das notificações não muda os limiares de qualificação da organização.
