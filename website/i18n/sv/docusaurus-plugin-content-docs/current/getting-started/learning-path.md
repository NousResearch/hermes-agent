---
sidebar_position: 3
title: 'Läsvägledning'
description: 'Hitta rätt väg genom dokumentationen för Hermes Agent utifrån din erfarenhet och dina mål.'
---

# Läsvägledning {#learning-path}

Hermes Agent kan användas som CLI-assistent, Telegram- eller Discord-bot, för att automatisera uppgifter, träna modeller med förstärkningsinlärning och mycket annat. Här hittar du en läsordning som passar din erfarenhet och det du vill göra.

:::tip Börja här
Har du inte installerat Hermes Agent ännu? Börja med [Installationsguiden](/getting-started/installation) och fortsätt med [Snabbstart](/getting-started/quickstart). Resten av sidan förutsätter en fungerande installation.
:::

:::tip Konfigurera din första leverantör
För nya användare är `hermes setup --portal` vanligtvis den enklaste vägen. En OAuth-inloggning ger tillgång till en modell och Tool Gateways fyra verktyg för sökning, bilder, talsyntes och webbläsare. Se [Nous Portal](/integrations/nous-portal).
:::

:::info Svenska guider
Installationsguiden och denna läsvägledning finns på svenska. Länkar till ännu inte översatta guider visar den engelska texten. Kommandon och konfigurationsnycklar ska skrivas som i exemplen, oavsett gränssnittsspråk.
:::

## Så använder du sidan {#how-to-use-this-page}

- **Vet du din nivå?** Gå till [tabellen över erfarenhetsnivåer](#by-experience-level) och följ läsordningen.
- **Har du ett särskilt mål?** Välj ett scenario under [Användningsområden](#by-use-case).
- **Vill du få en överblick?** Se [Funktioner i korthet](#key-features-at-a-glance).

## Efter erfarenhetsnivå {#by-experience-level}

| Nivå | Mål | Rekommenderad läsordning | Ungefärlig tid |
| --- | --- | --- | --- |
| **Nybörjare** | Komma i gång, föra enkla samtal och använda inbyggda verktyg | [Installation](/getting-started/installation) → [Snabbstart](/getting-started/quickstart) → [CLI-användning](/user-guide/cli) → [Konfiguration](/user-guide/configuration) | 1 timme |
| **Van användare** | Konfigurera meddelandebotar och använda minne, schemalagda uppgifter och färdigheter | [Sessioner](/user-guide/sessions) → [Meddelanden](/user-guide/messaging) → [Verktyg](/user-guide/features/tools) → [Färdigheter](/user-guide/features/skills) → [Minne](/user-guide/features/memory) → [Cron](/user-guide/features/cron) | 2–3 timmar |
| **Avancerad användare** | Bygga egna verktyg, skapa färdigheter, träna modeller med förstärkningsinlärning och bidra till projektet | [Arkitektur](/developer-guide/architecture) → [Lägga till verktyg](/developer-guide/adding-tools) → [Skapa färdigheter](/developer-guide/creating-skills) → [Bidra](/developer-guide/contributing) | 4–6 timmar |

## Användningsområden {#by-use-case}

Välj det scenario som passar ditt mål. Länkarna är ordnade i rekommenderad läsordning.

### Jag vill ha en kodassistent i terminalen {#i-want-a-cli-coding-assistant}

Använd Hermes Agent som interaktiv terminalassistent för att skriva, granska och köra kod.

1. [Installation](/getting-started/installation)
2. [Snabbstart](/getting-started/quickstart)
3. [CLI-användning](/user-guide/cli)
4. [Kodkörning](/user-guide/features/code-execution)
5. [Kontextfiler](/user-guide/features/context-files)
6. [Tips och råd](/guides/tips)

:::tip
Lägg till filer direkt i samtalet som kontextfiler. Hermes Agent kan läsa, redigera och köra kod i dina projekt.
:::

### Jag vill ha en Telegram- eller Discord-bot {#i-want-a-telegramdiscord-bot}

Kör Hermes Agent som bot på din meddelandeplattform.

1. [Installation](/getting-started/installation)
2. [Konfiguration](/user-guide/configuration)
3. [Översikt över meddelandeplattformar](/user-guide/messaging)
4. [Konfigurera Telegram](/user-guide/messaging/telegram)
5. [Konfigurera Discord](/user-guide/messaging/discord)
6. [Röstläge](/user-guide/features/voice-mode)
7. [Använd röstläge med Hermes](/guides/use-voice-mode-with-hermes)
8. [Säkerhet](/user-guide/security)

Fullständiga projektexempel:

- [Bot för dagliga sammanfattningar](/guides/daily-briefing-bot)
- [Telegram-assistent för team](/guides/team-telegram-assistant)

### Jag vill automatisera uppgifter {#i-want-to-automate-tasks}

Schemalägg återkommande uppgifter, kör jobb i omgångar eller koppla ihop agentåtgärder.

1. [Snabbstart](/getting-started/quickstart)
2. [Schemaläggning med cron](/user-guide/features/cron)
3. [Bearbetning i omgångar](/user-guide/features/batch-processing)
4. [Delegering](/user-guide/features/delegation)
5. [Hooks](/user-guide/features/hooks)

:::tip
Med cron-jobb kan Hermes Agent köra dagliga sammanfattningar, återkommande kontroller och automatiska rapporter enligt ett schema, utan att du behöver vara närvarande.
:::

### Jag vill ha ett team av specialiserade botar {#i-want-a-team-of-specialist-bots}

Skapa namngivna botar med egna modeller, minnen, färdigheter, rutiner och chattar. Låt dem samarbeta i gruppchattar eller genom `@mentions`.

1. [Skrivbordsappen](/user-guide/desktop)
2. [Profiler](/user-guide/profiles)
3. [Botläge](/user-guide/bot-mode)
4. [Schemaläggning med cron](/user-guide/features/cron)
5. [Flera anslutningar i skrivbordsappen](/user-guide/multi-connection-desktop)

### Jag vill bygga egna verktyg eller färdigheter {#i-want-to-build-custom-toolsskills}

Utöka Hermes Agent med egna verktyg och återanvändbara färdighetspaket.

1. [Pluginer](/user-guide/features/plugins)
2. [Bygg en Hermes-plugin](/developer-guide/plugins)
3. [Verktygsöversikt](/user-guide/features/tools)
4. [Färdighetsöversikt](/user-guide/features/skills)
5. [MCP (Model Context Protocol)](/user-guide/features/mcp)
6. [Arkitektur](/developer-guide/architecture)
7. [Lägga till verktyg](/developer-guide/adding-tools)
8. [Skapa färdigheter](/developer-guide/creating-skills)

:::tip
Börja vanligen med en plugin när du skapar egna verktyg. Sidan [Lägga till verktyg](/developer-guide/adding-tools) gäller utveckling av Hermes inbyggda kärna, inte den vanliga vägen för egna användarverktyg.
:::

### Jag vill träna modeller {#i-want-to-train-models}

Finjustera modellernas beteende med förstärkningsinlärning genom Hermes Agents träningsflöde, som använder [Atropos](https://github.com/NousResearch/atropos).

1. [Snabbstart](/getting-started/quickstart)
2. [Konfiguration](/user-guide/configuration)
3. [Atropos-miljöer för förstärkningsinlärning](https://github.com/NousResearch/atropos) (extern länk)
4. [Leverantörsroutning](/user-guide/features/provider-routing)
5. [Arkitektur](/developer-guide/architecture)

:::tip
Modellträning fungerar bäst när du redan förstår hur Hermes Agent hanterar samtal och verktygsanrop. Följ nybörjarspåret först om du är ny.
:::

### Jag vill använda Hermes som Python-bibliotek {#i-want-to-use-it-as-a-python-library}

Integrera Hermes Agent i dina egna Python-program.

1. [Installation](/getting-started/installation)
2. [Snabbstart](/getting-started/quickstart)
3. [Guide till Python-biblioteket](/guides/python-library)
4. [Arkitektur](/developer-guide/architecture)
5. [Verktyg](/user-guide/features/tools)
6. [Sessioner](/user-guide/sessions)

## Funktioner i korthet {#key-features-at-a-glance}

| Funktion | Vad den gör | Länk |
| --- | --- | --- |
| **Verktyg** | Inbyggda verktyg som agenten kan anropa, exempelvis filhantering, sökning och skal | [Verktyg](/user-guide/features/tools) |
| **Färdigheter** | Installerbara paket som ger agenten fler förmågor | [Färdigheter](/user-guide/features/skills) |
| **Minne** | Bestående minne mellan sessioner | [Minne](/user-guide/features/memory) |
| **Botläge** | Namngivna specialistbotar med bestående chattar, rutiner, gruppchattar och `@mentions` | [Botläge](/user-guide/bot-mode) |
| **Kontextfiler** | Lägg till filer och kataloger i samtal | [Kontextfiler](/user-guide/features/context-files) |
| **MCP** | Anslut till externa verktygsservrar med Model Context Protocol | [MCP](/user-guide/features/mcp) |
| **Cron** | Schemalägg återkommande agentuppgifter | [Cron](/user-guide/features/cron) |
| **Delegering** | Starta underagenter som arbetar parallellt | [Delegering](/user-guide/features/delegation) |
| **Kodkörning** | Kör Python-skript som anropar Hermes-verktyg | [Kodkörning](/user-guide/features/code-execution) |
| **Webbläsare** | Surfa på webben och hämta innehåll | [Webbläsare](/user-guide/features/browser) |
| **Hooks** | Händelsestyrda återanrop och mellanprogram | [Hooks](/user-guide/features/hooks) |
| **Bearbetning i omgångar** | Bearbeta flera indata i större omgångar | [Bearbetning i omgångar](/user-guide/features/batch-processing) |
| **Leverantörsroutning** | Fördela anrop mellan flera modellleverantörer | [Leverantörsroutning](/user-guide/features/provider-routing) |

## Vad ska jag läsa härnäst? {#what-to-read-next}

- **Nyss installerat?** Följ [Snabbstart](/getting-started/quickstart) för ditt första samtal.
- **Klar med snabbstarten?** Läs [CLI-användning](/user-guide/cli) och [Konfiguration](/user-guide/configuration) för att anpassa installationen.
- **Bekväm med grunderna?** Utforska [Verktyg](/user-guide/features/tools), [Färdigheter](/user-guide/features/skills) och [Minne](/user-guide/features/memory).
- **Konfigurerar du för ett team?** Läs [Säkerhet](/user-guide/security) och [Sessioner](/user-guide/sessions) om åtkomstkontroll och samtalshantering.
- **Redo att bygga?** Börja med [Utvecklarguiden](/developer-guide/architecture) för att förstå hur systemet fungerar och bidra.
- **Vill du se praktiska exempel?** Besök [Guider](/guides/tips) för projekt och tips.

:::tip
Du behöver inte läsa allt. Välj den väg som passar ditt mål och följ länkarna i ordning. Återvänd hit när du behöver hitta nästa steg.
:::
