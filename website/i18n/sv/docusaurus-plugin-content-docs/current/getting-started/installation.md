---
sidebar_position: 2
title: "Installation"
description: "Installera Hermes Agent på Linux, macOS, WSL2, Windows eller Android med Termux"
---

# Installation

Kom i gång med Hermes Agent på några minuter!

:::tip Plattformar som stöds
Den fullständiga matrisen över operativsystem, distributionssätt och plattformsberoende funktioner finns i **[Plattformsstöd](/getting-started/platform-support)**.
:::

## Snabbinstallation {#quick-install}

### Med installationsprogrammet för Hermes Desktop på macOS eller Windows (rekommenderas) {#with-the-hermes-desktop-installer-on-macos-or-windows-recommended}

[Ladda ner installationsprogrammet för Hermes Desktop](https://hermes-agent.nousresearch.com/) från vår webbplats och kör det för att installera både kommandoradsprogrammet och skrivbordsappen.

### Utan Hermes Desktop {#without-hermes-desktop}

Om du endast vill installera kommandoradsprogrammet kör du:

#### Linux / macOS / WSL2 / Android (Termux) {#linux--macos--wsl2--android-termux}

```bash
curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash
```

#### Windows (utan WSL) {#windows-native}

Kör i PowerShell:

```powershell
iex (irm https://hermes-agent.nousresearch.com/install.ps1) 
```

Om du senare vill installera och köra Hermes Desktop kör du:

```bash
hermes desktop
```

### Vad installationsprogrammet gör {#what-the-installer-does}

Installationsprogrammet hanterar beroenden (Python, Node.js, ripgrep och ffmpeg), klonar kodförrådet, skapar en virtuell miljö, gör kommandot `hermes` tillgängligt och konfigurerar modellleverantören. Därefter kan du börja chatta.

#### Katalogstruktur {#install-layout}

Placeringen beror på om du installerar som vanlig användare eller som root:

| Installation | Kodens placering | Programmet `hermes` | Datakatalog |
| --- | --- | --- | --- |
| Per användare (Git-installation) | `~/.hermes/hermes-agent/` | `~/.local/bin/hermes` (symbolisk länk) | `~/.hermes/` |
| Som root (`sudo curl … \| sudo bash`) | `/usr/local/lib/hermes-agent/` | `/usr/local/bin/hermes` | `/root/.hermes/` (eller `$HERMES_HOME`) |

Root-installationens **FHS-struktur** (`/usr/local/lib/…`, `/usr/local/bin/hermes`) följer placeringen av andra utvecklarverktyg som installeras för hela Linux-systemet. Den passar delade datorer där samma installation ska betjäna flera användare. Varje användares konfiguration, autentisering, färdigheter och sessioner ligger fortfarande under den egna katalogen `~/.hermes/` eller ett uttryckligen angivet `HERMES_HOME`.

### Efter installationen {#after-installation}

Läs in skalets konfiguration igen och börja chatta:

```bash
source ~/.bashrc   # eller: source ~/.zshrc
hermes             # Börja chatta!
```

Använd respektive kommando för att ändra inställningar senare:

```bash
hermes model          # Välj modellleverantör och modell
hermes tools          # Välj vilka verktyg som ska vara aktiverade
hermes gateway setup  # Konfigurera meddelandeplattformar
hermes config set     # Ange enskilda konfigurationsvärden
hermes config get     # Granska enskilda konfigurationsvärden
hermes setup          # Kör hela installationsguiden för att konfigurera allt
```

:::tip Snabbaste vägen: Nous Portal
Ett abonnemang omfattar över 300 modeller och [Tool Gateway](/user-guide/features/tool-gateway) för webbsökning, bildgenerering, talsyntes och en webbläsare i molnet. Slipp hantera separata nycklar för varje verktyg:

```bash
hermes setup --portal
```

Kommandot loggar in dig, anger Nous som leverantör och aktiverar Tool Gateway.
:::

:::tip Kör du redan Hermes på en annan dator?
Du behöver inte börja om. Återställ en fullständig säkerhetskopia med `hermes import` (se [Exportera Hermes till en annan dator](/reference/faq#exporting-hermes-to-another-machine)) eller flytta en enskild agent med `hermes profile import` (se [Flytta en profil till en annan dator](/reference/faq#moving-a-single-profile-to-another-machine)). En profilexport utesluter avsiktligt autentiseringsuppgifter och är därför inte en fullständig säkerhetskopia. Läs [Skillnaden mellan `hermes backup` och `hermes profile export`](/reference/faq#hermes-backup-vs-hermes-profile-export) för att välja rätt.
:::

## Förutsättningar {#prerequisites}

**Installationsprogrammet:** På andra plattformar än Windows krävs **Git**. På Linux behöver även `curl` och `xz-utils` finnas, eftersom Node.js hämtas som ett `.tar.xz`-arkiv. Skrivbordsappen behöver dessutom `g++` (eller `build-essential` på Debian/Ubuntu) för att kompilera inbyggda moduler. Installationsprogrammet hanterar resten automatiskt:

- **uv** – en snabb pakethanterare för Python.
- **Python 3.11** – installeras via uv utan sudo.
- **Node.js v26** – för webbläsarautomatisering och WhatsApp-bryggan. En befintlig systemversion 22.22+, 24.11+ eller 26+ används direkt.
- **ripgrep** – snabb filsökning.
- **ffmpeg** – ljudkonvertering för talsyntes.

:::info
Du behöver inte installera Python, Node.js, ripgrep eller ffmpeg manuellt. Installationsprogrammet upptäcker vad som saknas. Kontrollera att Git finns med `git --version`. På Debian/Ubuntu installerar du Linux-beroendena med `sudo apt install curl xz-utils`. För skrivbordsappen behövs även `sudo apt install build-essential`.
:::

:::tip För Nix-användare
Nix är **inte längre en uttryckligen stödd installationsmetod**; stödet ges efter förmåga. Om du redan använder Nix på NixOS, macOS eller Linux finns en installationsväg med en Nix-flake, en deklarativ NixOS-modul och ett valfritt containerläge. Se **[Konfiguration av Nix och NixOS](/getting-started/nix-setup)**.
:::

## Manuell installation och utvecklarinstallation {#manual--developer-installation}

Om du vill klona kodförrådet och installera från källkod för att bidra, köra en viss gren eller själv styra den virtuella miljön, följ avsnittet [Utvecklingsmiljö](/developer-guide/contributing#development-setup) i bidragsguiden.

## Installation utan sudo eller för en tjänsteanvändare {#non-sudo--system-service-user-installs}

Hermes kan köras som en särskild användare utan administratörsbehörighet, exempelvis ett systemd-tjänstekonto med namnet `hermes`. Steget som kräver root är Playwrights `--with-deps`, som installerar Chromiums systembibliotek med apt, bland annat `libnss3` och `libxkbcommon`. Om sudo saknas installeras Chromium i tjänsteanvändarens egen Playwright-cache, och installationsprogrammet visar vilket kommando en administratör behöver köra separat.

**Rekommenderad uppdelning på Debian/Ubuntu:**

1. **Som administratör med sudo**, installera en gång de systembibliotek Chromium behöver:

   ```bash
   sudo npx playwright install-deps chromium
   ```

   Kommandot kan köras från valfri katalog; `npx` hämtar Playwright vid behov.

2. **Som tjänsteanvändaren utan administratörsbehörighet**, kör det vanliga installationsprogrammet. Det hoppar över `--with-deps` när sudo saknas och installerar Chromium i användarens lokala Playwright-cache:

   ```bash
   curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash
   ```

   Hoppa över Playwright helt med `--skip-browser`, exempelvis om du inte behöver webbläsarautomatisering:

   ```bash
   curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash -s -- --skip-browser
   ```

   Installationsprogrammet förinstallerar även [`cua-driver`](/user-guide/features/computer-use), så att verktygsuppsättningen Computer Use fungerar när du aktiverar den. Välj bort förinstallationen med `--skip-computer-use`; då installeras den vid behov när verktyget aktiveras.

3. **Gör `hermes` tillgängligt i tjänsteanvändarens skal.** Startprogrammet skrivs till `~/.local/bin/hermes`. Tjänstekonton har ofta en begränsad PATH utan `~/.local/bin`. Lägg till katalogen i användarens miljö eller skapa en symbolisk länk på systemnivå:

   ```bash
   # Alternativ A: lägg till i tjänsteanvändarens profil
   echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc

   # Alternativ B: skapa en systemgemensam länk som administratör
   sudo ln -s /home/hermes/.hermes/hermes-agent/venv/bin/hermes /usr/local/bin/hermes
   ```

4. **Verifiera:** `hermes doctor` ska nu fungera. Felet `ModuleNotFoundError: No module named 'dotenv'` betyder att källfilen `~/.hermes/hermes-agent/hermes` körs med systemets Python i stället för startprogrammet i den virtuella miljön, `~/.hermes/hermes-agent/venv/bin/hermes`. Rätta steg 3.

5. **Ska kontot köra meddelandegatewayen?** En användartjänst stoppas vid utloggning och startar inte vid systemstart förrän du aktiverar fortsatt körning för tjänsteanvändaren:

   ```bash
   sudo loginctl enable-linger <service-user>
   ```

   Se [Meddelandegateway](/user-guide/messaging/) för själva tjänstekonfigurationen.

Samma uppdelning fungerar på Arch, där installationsprogrammet använder pacman och samma kontroll av sudo. Fedora/RHEL och openSUSE stöder inte `--with-deps`, så där installerar en administratör alltid systembiblioteken separat. Installationsprogrammet visar relevanta `dnf`- eller `zypper`-kommandon.

## Felsökning {#troubleshooting}

| Problem | Lösning |
| --- | --- |
| `hermes: command not found` | Läs in skalets konfiguration igen (`source ~/.bashrc`) eller kontrollera PATH. |
| `API key not set` | Konfigurera leverantören med `hermes model` eller `hermes config set OPENROUTER_API_KEY your_key`. |
| Konfiguration saknas efter uppdatering | Kör `hermes config check` och sedan `hermes config migrate`. |

Kör `hermes doctor` för mer diagnostik och anvisningar om hur fel åtgärdas.

### Symboliskt länkade hemkataloger och extern lagring {#symlinked-home-directories-and-external-storage}

Hermes stöder ett symboliskt länkat `HERMES_HOME` och länkade underkataloger, bland annat `hooks`, `skills`, `sessions` och `logs`. Befintliga kataloglänkar bevaras vid initiering. Behörigheterna på länkade kataloger och underkataloger, exempelvis `logs/curator`, lämnas åt ägaren.

Om länkens mål saknas, inte är åtkomligt eller inte är en katalog avbryts initieringen med ett lagringsfel som anger sökväg och länkmål. Hermes ersätter inte länken och skapar inte det saknade målet. Det skulle annars kunna skriva till den lokala disken när en extern volym eller NAS-volym inte är monterad. Kontrollera länken, återställ monteringen eller rätta målet och kontrollera åtkomstbehörigheterna innan du försöker igen. Om du avsiktligt använder ett nytt mål för konfigurationsfiler skapar du det själv först när du har kontrollerat att rätt lagring är tillgänglig.

`hermes doctor` rapporterar detta som lagringsproblem, inte som ogiltig YAML. Behåll din `config.yaml`; `hermes setup` reparerar inte en otillgänglig katalog. Kontrollen gäller katalogens tillgänglighet och övervakar inte monteringar: en befintlig katalog bevisar inte att rätt volym är monterad.

## Automatisk identifiering av installationsmetod {#install-method-auto-detection}

Hermes identifierar Git-installationer, Docker och NixOS automatiskt. `hermes update` visar rätt uppdateringskommando för metoden. Ingen miljövariabel behöver anges: identifieringen bygger på installationens struktur, såsom kodförrådet i `~/.hermes/hermes-agent/`, Docker-avbildningens märkning eller en Nix store-sökväg. `hermes doctor` visar också den identifierade metoden i miljösammanfattningen.
