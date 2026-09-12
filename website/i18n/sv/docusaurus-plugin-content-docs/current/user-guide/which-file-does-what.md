---
sidebar_position: 4
title: "Vilken fil gör vad?"
description: "SOUL.md, USER.md, MEMORY.md och AGENTS.md: filernas uppgifter, vem som skriver dem och när agenten läser dem."
---

# Vilken fil gör vad? {#which-file-does-what}

”Jag berättade något för agenten och den glömde det.” ”Vilken fil är agentens hjärna?” ”Jag ändrade SOUL.md – varför känner den inte till mitt namn?” Hermes Agent använder flera Markdown-filer med olika uppgifter. Här får du en samlad överblick. Fördjupning finns i [Bestående minne](/user-guide/features/memory), [Personlighet och SOUL.md](/user-guide/features/personality) och [Kontextfiler](/user-guide/features/context-files).

## Översikt {#the-master-table}

| Fil | Innehåll | Vem skriver den? | När läser agenten den? | Placering |
| --- | --- | --- | --- | --- |
| **SOUL.md** | Agentens grundläggande identitet: personlighet, ton, kommunikationsstil och stilval att undvika | Du. Hermes skapar automatiskt en startfil om den saknas; befintliga filer skrivs aldrig över | Första delen av systemprompten, när sessionen startar | `~/.hermes/SOUL.md` (eller `$HERMES_HOME/SOUL.md` med en egen hemkatalog), aldrig arbetskatalogen |
| **USER.md** | Din profil: namn, roll, preferenser, kommunikationsstil och förväntningar | Agenten via verktyget `memory`. Du kan kräva godkännande med `write_approval` eller redigera poster med `hermes journey edit` | Infogas i systemprompten som en fryst ögonblicksbild vid sessionsstart | `~/.hermes/memories/` |
| **MEMORY.md** | Agentens egna anteckningar: miljöfakta, projektkonventioner, verktygsegenskaper och lärdomar | Agenten via `memory`, med samma möjligheter till godkännande och redigering som för USER.md | Infogas i systemprompten som en fryst ögonblicksbild vid sessionsstart | `~/.hermes/memories/` |
| **AGENTS.md** | Projektinstruktioner, konventioner och arkitektur: kommandon, portar, sökvägar och arbetsflöden för kodförrådet | Du eller den som ansvarar för projektet | Läses från arbetskatalogen till systemprompten vid start. Filer i underkataloger upptäcks efter hand när agenten arbetar där | Projektets arbetskatalog och underkataloger |
| **.hermes.md** / **HERMES.md** | Hermes-specifika projektinstruktioner, liknande AGENTS.md men med högst prioritet | Du | Läses till systemprompten vid start; den första träffen har företräde framför AGENTS.md | I projektet; sökningen går upp till Git-roten |

:::info En typ av projektkontext per session
Endast **en** typ av projektkontext läses in per session. Den första träffen används i ordningen `.hermes.md` → `AGENTS.md` → `CLAUDE.md` → `.cursorrules`. `SOUL.md` läses alltid in separat som agentens identitet och ingår inte i denna prioritetsordning. Se [Kontextfiler](/user-guide/features/context-files) för hela listan och kompatibiliteten med `CLAUDE.md` och `.cursorrules`.
:::

En enkel minnesregel:

- **SOUL.md** beskriver vem agenten *är*. Lägg personlighetsinstruktioner som ska följa med överallt här.
- **USER.md** beskriver vem *du* är. Agenten underhåller filen åt dig.
- **MEMORY.md** beskriver vad agenten har *lärt sig*. Agenten underhåller även denna fil.
- **AGENTS.md**, eller `.hermes.md`, beskriver vad *projektet* behöver. Projektspecifika instruktioner hör hemma här.

## Varför glömde agenten det jag nyss sa? {#why-did-it-forget-what-i-just-said}

Minnet i MEMORY.md och USER.md infogas i systemprompten som en **fryst ögonblicksbild** vid sessionsstart. När agenten sparar något mitt i sessionen skrivs ändringen direkt till disk, men den infogas inte i systemprompten förrän nästa session börjar. Det bevarar språkmodellens prefixcache och förbättrar prestandan. Verktygssvaren visar det aktuella innehållet, så informationen går inte förlorad. Starta en ny session för att få det uppdaterade minnet i systemprompten. Läs mer i [Så visas minnet i systemprompten](/user-guide/features/memory#how-memory-appears-in-the-system-prompt).

## Vanliga missförstånd {#common-mix-ups}

### Jag skrev om mig själv i SOUL.md, men USER.md är fortfarande tom {#i-put-facts-about-myself-in-soulmd-but-usermd-stayed-empty}

`SOUL.md` och `USER.md` är separata system som inte fyller på varandra. `SOUL.md` är en personlighetsfil som **du** redigerar direkt. Den formar ton och identitet, och innehållet infogas ordagrant först i prompten. `USER.md` tillhör det bestående minnet och skrivs av **agenten** genom `memory`. Vill du spara fakta om dig själv i USER.md, säg exempelvis ”kom ihåg att jag föredrar korta svar”. Agenten kan då spara dem. Ändringar i SOUL.md fyller inte på minnet, och minnesposter ändrar inte personligheten. Använd SOUL.md för bestående anvisningar om ton och personlighet, och minnet för preferenser och profilfakta. Se [Vad ska finnas i SOUL.md?](/user-guide/features/personality#what-should-go-in-soulmd) och [Minnets två mål](/user-guide/features/memory#two-targets-explained).

### Jag berättade mitt namn mitt i sessionen, men agenten verkar inte ha hört det {#i-told-it-my-name-mid-session-and-it-acted-like-it-never-heard-it}

Om agenten sparade ditt namn i minnet finns det kvar. Kontrollera svaret från `memory` eller kör `hermes journey list`. Systemprompten uppdateras däremot inte mitt i en session, så dess infogade minnesblock visar fortfarande läget vid sessionsstart. Agenten kan ändå använda det du har sagt i det aktuella samtalet, eftersom det finns i kontexten. Den sparade posten ingår i systemprompten från nästa session. Samma sak gäller ändringar i `SOUL.md` eller `AGENTS.md` under en pågående session: kontexten sätts samman vid sessionsstart, så starta om sessionen för att läsa in ändringarna.

:::tip Välj rätt fil eller verktyg
- Vill du ändra hur agenten **uttrycker sig**? Redigera `~/.hermes/SOUL.md`. Se [Personlighet och SOUL.md](/user-guide/features/personality).
- Vill du att agenten ska **komma ihåg ett faktum**? Berätta det; agenten sparar det själv i minnet. Se [Bestående minne](/user-guide/features/memory).
- Vill du ange **projektregler**? Lägg en `AGENTS.md` eller `.hermes.md` i projektet. Se [Kontextfiler](/user-guide/features/context-files).
- Behöver du ändra personligheten **tillfälligt**? Använd `/personality`, som lägger till en personlighet för sessionen utan filändringar.
:::

## Relaterad dokumentation {#related-docs}

- [Bestående minne](/user-guide/features/memory) – MEMORY.md, USER.md, verktyget `memory`, kapacitetsgränser och `write_approval`.
- [Personlighet och SOUL.md](/user-guide/features/personality) – innehåll i SOUL.md, förval för `/personality` och promptens delar.
- [Kontextfiler](/user-guide/features/context-files) – AGENTS.md, `.hermes.md`, stegvis upptäckt och säkerhetsgranskning.
