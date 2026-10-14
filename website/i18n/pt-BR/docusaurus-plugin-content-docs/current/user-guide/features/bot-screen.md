---
title: Tela do bot
sidebar_position: 17
---

# Tela do bot

Em um host Linux sem monitor — um servidor, uma VM na nuvem ou o Hermes Cloud — cada bot tem seu **próprio desktop**: uma tela Xfce na qual o `computer_use` e o navegador com interface gráfica do bot atuam, transmitida ao vivo para o Hermes Desktop. Observe o que o bot faz, **assuma o controle** quando ele chegar a um login, desafio 2FA, CAPTCHA ou etapa de pagamento, depois **devolva o controle** e deixe-o continuar na sessão em que você acabou de entrar. O bot continua trabalhando mesmo depois que você fecha o aplicativo ou desliga o notebook; a tela fica no host do gateway, não na sua máquina.

Cada perfil do Hermes (ou “bot”) tem sua própria tela, seu próprio perfil de navegador e seus próprios cookies. As telas são superfícies de trabalho, não limites de segurança: os bots compartilham a conta de usuário, os arquivos e a rede do host.

**Modelo de ameaça.** O socket RFB da tela, o display X, o perfil do navegador e o arquivo de controle pertencem ao usuário do sistema operacional do gateway. Qualquer processo executado por esse usuário — outro bot no mesmo host e até a ferramenta `terminal` do próprio bot — pode acessá-los diretamente, contornando o painel e a concessão de controle. A concessão é uma barreira no nível das ferramentas `computer_use` e do navegador, não no nível do sistema operacional. A porta DevTools do Chromium é ainda mais ampla: o navegador do painel e todo lançamento do agent-browser anunciam uma porta no loopback, acessível por qualquer usuário local do host. Executar cada bot com um usuário de sistema próprio está fora do escopo; se esse isolamento for necessário, ou se houver usuários locais não confiáveis, coloque os bots em hosts separados. A ponte WebSocket pode manter a decisão da concessão em cache por até 250 ms; uma tomada de controle feita por outro processo é aplicada nessa janela, e os resultados das ferramentas do bot são invalidados pela época da concessão. O `display_ticket` de uso único, válido por 30 segundos, viaja propositalmente como parâmetro na URL: o noVNC não consegue negociar subprotocolos WebSocket. Por isso, um proxy reverso pode registrar nos logs um ticket que já foi usado.

## Requisitos

- O host do gateway executa Linux. Hosts macOS e Windows já possuem um display real; o painel não é oferecido neles.
- O TigerVNC (`Xvnc`) e os componentes principais do Xfce estão instalados no host. Nada é instalado silenciosamente: `hermes update` e novas instalações deixam cada máquina como está. Quando algo estiver faltando, o painel Tela do Hermes Desktop mostrará **Instalar no host** — um clique executa o gerenciador de pacotes no host do gateway, solicita a senha `sudo` desse host em um cartão mascarado e nunca a armazena. Quando o Hermes executa como root, como normalmente ocorre em um contêiner, o instalador usa o gerenciador diretamente, sem `sudo` nem cartão de senha. Se o host não for root e não tiver `sudo`, o painel e a CLI exibem o comando exato para execução manual:

  | Distribuição | Pacotes |
  |---|---|
  | Debian / Ubuntu | `tigervnc-standalone-server xfce4-panel xfwm4 xfdesktop4 xfce4-settings xfce4-terminal dbus-x11 x11-xserver-utils x11-utils xauth fonts-dejavu-core` |
  | Fedora | `tigervnc-x11-server xfce4-panel xfwm4 xfdesktop xfce4-settings xfce4-terminal dbus-daemon xsetroot xset xdpyinfo xprop xorg-x11-xauth setxkbmap dejavu-sans-fonts` |
  | Arch | `tigervnc xfce4-panel xfwm4 xfdesktop xfce4-settings xfce4-terminal xorg-xsetroot xorg-xset xorg-xdpyinfo xorg-xprop xorg-xauth xorg-setxkbmap ttf-dejavu` |

  No Fedora, o binário `Xvnc` vem de `tigervnc-x11-server` (não de `tigervnc-server-minimal`) e `dbus-run-session` vem de `dbus-daemon`. O metapacote `xfce4` não é usado de propósito: ele instala o protetor de tela, o gerenciador de energia e o agente polkit, que bloqueiam ou solicitam interação em um desktop sem monitor.

  A imagem oficial Docker (`nousresearch/hermes-agent`, que também alimenta o Hermes Cloud) executa como usuário sem privilégios e não possui `sudo`. Nesse caso, o painel exibe a linha `apt-get` e um operador a executa uma vez como root no contêiner, por exemplo `docker exec -u 0 <container> apt-get install -y …`. Adicione `chromium` à linha se quiser o ícone Browser do painel. Pelo shell, `hermes computer-use screen status` mostra a linha exata e `hermes computer-use screen install` a executa.
- [Computer Use](./computer-use.md) habilitado para o bot, com o cua-driver instalado.
- Memória. Na imagem oficial, o gateway ocioso usa aproximadamente 300 MB, Xvnc + Xfce adicionam cerca de 220 MB, e o Chromium com interface gráfica aberto durante uma tomada de controle adiciona de 0,5 a 1 GB (uma página: cerca de 550 MB). Planeje **aproximadamente 1,1–1,5 GB por tela aberta com navegador**; o desktop sozinho é barato, o navegador é o custo. A CPU não é uma restrição (desktop ocioso ≈ 0,01 core, transmissão ativa ≈ 0,03 core). No Debian 13, os pacotes ocupam cerca de 930 MB de disco.

  Antes de iniciar uma tela, o Hermes verifica se o host — ou o cgroup do contêiner, conforme o limite mais restritivo — possui `bot_desktop.min_free_memory_mb` disponível (padrão 1536; `0` desativa a verificação). Abaixo desse valor, o painel explica o motivo em vez de mostrar **Iniciar tela** e `hermes computer-use screen start` recusa a operação; uma tela já em execução nunca é encerrada por essa verificação. Uma tela sem uso é parada após `bot_desktop.idle_stop_minutes` (padrão 30) e volta no próximo uso. Para instâncias pequenas: 4 GB executam o desktop; 8 GB oferecem uma tomada de controle confortável com navegador.

### Incluindo os pacotes em uma imagem de contêiner

Uma imagem para implantação hospedada ou sem privilégios não pode instalar nada em tempo de execução, então os pacotes precisam ser incluídos durante o build. A CI publica duas variantes de cada versão: as tags sem sufixo (`:latest`, `:v*`), sem esses pacotes, e as tags **`-desktop`** (`:latest-desktop`, `:v*-desktop`), com eles. Uma implantação hospedada (Fly Machines, instâncias de contêiner do Azure) obtém a Tela do bot usando a tag com sufixo; um argumento de build não resolveria, pois ele nunca é executado durante a provisão. O provisionador ainda não seleciona `-desktop`, então uma instância hospedada continua enxuta; usar a tag com sufixo manualmente já funciona.

Se quiser os pacotes em uma imagem personalizada, use o argumento opcional do `Dockerfile` oficial, desativado por padrão para que `docker build .` continue enxuto:

```bash
docker build --build-arg HERMES_BOT_DESKTOP=1 -t hermes-agent:screen .
```

Isso adiciona TigerVNC, os componentes Xfce, um `chromium` com interface gráfica e o Chromium com interface gráfica do Playwright — cerca de **1,4 GB** à imagem (medição: 4,1 GB sem o argumento e 5,5 GB com ele em arm64), dos quais aproximadamente 930 MB são a camada apt. Nada é iniciado no boot; uma imagem construída assim não consome memória até uma tela ser iniciada.

## Como usar

O computador de cada bot fica a um clique de distância em três lugares do Hermes Desktop:

- **Bots → um bot → Tarefas agendadas**: a tela do bot aparece no topo, acima do título e das rotinas, com uma prévia ao vivo do desktop e a indicação de quem possui o controle. Clique na imagem para abrir o acesso ao vivo. Quando a tela está desligada ou não instalada, o mesmo quadro informa a situação e oferece Iniciar / Instalar.
- **Bots → clique direito em um bot → Abrir tela**. O mesmo menu tem **Abrir tela quando o bot usá-la**: quando ativado, a aba Tela vem para frente na primeira chamada `computer_use` ou do navegador durante uma execução. Desativado por padrão e configurado por bot, ele eleva a aba sem roubar o foco do teclado, não dispara para histórico reproduzido, no máximo uma vez a cada 30 segundos e permanece fechada se você fechá-la durante a execução, até a próxima execução do bot.
- **Barra lateral de sessões**, agrupada por gateway / perfil: o mesmo quadro **Tela** aparece sob o cabeçalho de cada perfil, permitindo acessar a máquina do perfil a partir das conversas.

1. Abra a Tela por qualquer uma das opções acima. A tela fica **desligada por padrão** e nada a inicia automaticamente: clique em **Iniciar tela** no painel, execute `hermes computer-use screen start` no host ou defina `bot_desktop.auto_start: true` para iniciar a tela na primeira chamada `computer_use` ou no primeiro uso do navegador com interface gráfica (`browser.headed: true`). Um navegador com interface gráfica abre na tela quando ela está em execução.
2. O painel transmite o desktop do bot. O indicador no cabeçalho mostra quem está no controle: por padrão, **O bot está no controle**.
3. Clique em **Assumir controle**. A borda fica vermelha e teclado e mouse passam a controlar a tela do bot. Faça login, resolva o CAPTCHA e aprove o pagamento.
4. Clique em **Devolver controle**. O bot recupera o controle e captura a tela novamente antes de continuar. Fechar o painel também devolve o controle. Uma conexão perdida é diferente: se a tampa do notebook fechar ou o Wi-Fi cair enquanto você estiver no controle, você continuará mantendo-o — o bot permanecerá bloqueado em uma tela na qual você pode estar no meio de um login — até reconectar e devolver o controle. Se, após recarregar, o painel ainda informar que uma pessoa controla a tela, aparecerá **Devolver controle (forçado)**.

Enquanto você estiver no controle, as ferramentas `computer_use` e do navegador do bot serão recusadas com `human_has_control`, inclusive as capturas. Isso é uma barreira no nível das ferramentas, não do sistema operacional: o bot executa como o mesmo usuário da tela. Não digite segredos em um bot no qual você não confia.

Quando o bot chegar a uma etapa que não deve executar sozinho (login, 2FA, CAPTCHA ou pagamento), ele informará isso na resposta e encerrará o turno; a solicitação chegará pelo chat em uso. Assuma o controle quando estiver pronto, conclua a etapa, devolva o controle e diga ao bot para continuar. Nada fica bloqueado no lado do bot enquanto ele espera: a tomada de controle sempre começa com você, e o bot nunca mantém uma chamada de ferramenta aberta esperando.

Dois visualizadores na mesma tela: a tomada de controle mais recente vence; o controlador anterior volta a assistir.

## Sessões do navegador que sobrevivem à troca de controle

Enquanto a tela está em execução, o navegador do bot e o ícone **Browser** do painel são o mesmo navegador: o agent-browser controla o Chromium, com um diretório de dados persistente por bot (`<HERMES_HOME>/bot-desktop/browser-profile`; defina `AGENT_BROWSER_PROFILE` para fixar o seu próprio — `~` é expandido e um caminho relativo, como `pin`, é resolvido contra o `HERMES_HOME` daquele bot). Clique em Browser durante uma tomada de controle para acessar as janelas e o conjunto de cookies do próprio bot; o login feito ali será usado pelo bot nas sessões seguintes, até o site expirá-lo. Defina `browser.headed: true` para que a navegação do bot também fique visível na tela.

O painel é configurado uma única vez, na primeira inicialização da tela de um perfil. O arquivo de layout é `<HERMES_HOME>/bot-desktop/xdg/xfce4/xfconf/xfce-perchannel-xml/xfce4-panel.xml`: enquanto ele existir, o inicializador não altera o painel. Para reconstruí-lo com o que estiver instalado, exclua o arquivo e execute `screen start` novamente.

Para escolher o Chromium do painel e do bot, `AGENT_BROWSER_EXECUTABLE_PATH` tem prioridade explícita; caso contrário, o Hermes prefere um `chromium` / `google-chrome` do sistema e usa o Chromium incluído pelo Playwright como fallback. Em Ubuntu 23.10 ou posterior, a configuração `kernel.apparmor_restrict_unprivileged_userns=1` pode impedir o sandbox do Chromium incluído pelo Playwright para usuários sem privilégios. Se a escolha não funcionar no host, defina `AGENT_BROWSER_EXECUTABLE_PATH=/usr/bin/chromium` ou o caminho do Chrome. A imagem oficial Docker inclui somente o shell headless do Playwright, que não desenha janelas; instale um navegador com interface gráfica (`apt-get install chromium`) para obter o ícone Browser. O navegador da pessoa e o do bot usam então as mesmas configurações de sandbox e são um só navegador.

## CLI

```bash
hermes computer-use screen status          # instalado? em execução? quem controla?
hermes computer-use screen start           # inicia a tela deste perfil
hermes computer-use screen stop            # para a tela
hermes computer-use screen stop --force    # também libera uma concessão presa
hermes computer-use screen install [-y]    # instala os pacotes com apt/dnf/pacman
hermes -p research computer-use screen start   # tela de outro bot
```

## Configuração

```yaml
bot_desktop:
  geometry: "1440x900"      # tamanho da tela; o visualizador ajusta ao painel
  auto_start: false         # inicia na primeira chamada computer_use ou uso de navegador com interface gráfica
  min_free_memory_mb: 1536  # recusa abaixo desta memória livre (0 = nunca verificar)
  idle_stop_minutes: 30     # para uma tela sem uso por este tempo (0 = mantê-la ativa)
```

`auto_start` fica desativado por padrão. Inicie a tela pelo painel Tela do Desktop (**Iniciar tela**), por `hermes computer-use screen start` ou defina-o como `true` para que um host sem monitor inicie a tela na primeira chamada `computer_use` ou abertura de um navegador com interface gráfica (`browser.headed: true`) quando não houver display disponível.

O estado fica em `<HERMES_HOME>/bot-desktop/` por perfil (socket Unix RFB, Xauthority, log do inicializador e xfconf do perfil).

## Como funciona

- **TigerVNC `Xvnc`** é o servidor X e o servidor RFB em um único processo, por perfil, escutando apenas em um socket Unix com modo `0600`. Não há porta TCP nem senha VNC: somente processos executados pelo usuário do gateway podem acessá-lo, e a ponte WebSocket autenticada do gateway é o caminho normal de entrada.
- **Xfce** inicia seus componentes (`xfsettingsd`, `xfwm4 --compositor=off`, `xfdesktop`, `xfce4-panel`) sob uma sessão D-Bus privada, sem `xfce4-session`, para que nada tente bloquear a tela ou acessar o `logind`.
- **Hermes Desktop** inclui o noVNC. Ele solicita ao gateway um ticket de uso único (`display.observe`) pela conexão autenticada normal e abre um WebSocket irmão em `/api/display/ws`; o gateway encaminha o fluxo RFB. Nada novo é exposto: o painel funciona em conexões locais, SSH, URL + token e Hermes Cloud.
- **Concessão de controle.** O gateway descarta mensagens de teclado, ponteiro e área de transferência de qualquer visualizador que não possua a concessão, no nível dos bytes RFB; o modo somente visualização do noVNC é apenas uma indicação visual. A mesma concessão controla `computer_use` e as ferramentas do navegador. Ela é um arquivo em `<HERMES_HOME>/bot-desktop/`: sem arquivo, o bot controla (perfil novo); se o arquivo existir mas não puder ser lido ou analisado, o sistema falha de forma segura e trata o bot como bloqueado até a próxima troca bem-sucedida. O Xvnc nunca envia a área de transferência da tela aos visualizadores (`-SendCutText=0`), então quem observa não recebe o que a pessoa no controle copia; colar na tela continua funcionando.
- **Vinculação do display.** O inicializador publica `DISPLAY`, `XAUTHORITY` e o endereço D-Bus; todo processo do cua-driver e do navegador com interface gráfica daquele perfil herda esses valores, para que o bot nunca atue em um display diante de uma pessoa.

## Solução de problemas

- **“Pacotes da tela ausentes”** — clique em **Instalar no host** no painel ou execute a linha de instalação exibida no host do gateway, não na máquina que executa o Hermes Desktop. O painel recusa uma segunda instalação enquanto uma estiver em andamento.
- **A tela inicia e para** — leia `<HERMES_HOME>/bot-desktop/launcher.log`.
- **A digitação produz caracteres incorretos durante uma tomada de controle** — a tela usa o mapa de teclado US para que os keysyms RFB e o cua-driver concordem, e o noVNC envia códigos brutos quando o Xvnc os oferece. Em um teclado físico que não seja US, teclas dependentes do layout (Y/Z e símbolos) podem resultar nas equivalentes US enquanto você estiver no controle. Digite senhas levando isso em conta ou altere o layout com `setxkbmap` no `DISPLAY` da tela.
- **O bot informa `human_has_control` depois que você saiu** — clique em **Devolver controle** no painel ou em **Devolver controle (forçado)** após recarregar. Pelo shell, `hermes computer-use screen stop --force` libera a concessão e para a tela; sem `--force`, o comando recusa enquanto uma pessoa controla a tela, para que uma rotina nunca interrompa uma tomada de controle ativa. `hermes computer-use screen start` a inicia novamente com o bot no controle.

### Testes no WSL

O WSL2 conta como um host Linux compatível: `screen status` o informa como tal e o painel é oferecido. Uma particularidade do WSLg interfere na primeira inicialização: o WSLg monta `/tmp/.X11-unix` como somente leitura, então o `Xvnc` não consegue criar o socket do display e termina com `Cannot establish any listening sockets` no `launcher.log`. Substitua a montagem por um diretório gravável antes de iniciar a tela:

```bash
sudo umount /tmp/.X11-unix  # no-tmp: ok — diretório de socket X11, fixo pelo protocolo
sudo mkdir -p /tmp/.X11-unix && sudo chmod 1777 /tmp/.X11-unix  # no-tmp: ok — o mesmo diretório
```

A montagem volta na próxima reinicialização do WSL; repita os dois comandos nessa situação.
