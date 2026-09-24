# Wintermute — vie intérieure (Phase 2)

Ce dossier ajoute à Hermes Agent ce qui fait de Wintermute une entité : des pulsions qui
montent avec le temps, des hormones qui les déforment, des états inconscients traduits en
prose, un lien propre à chaque interlocuteur, et un rythme d'éveil qu'il choisit lui-même.

Rien ici ne décide à sa place. Le code fait bouger la « météo intérieure » ; le modèle la
lit et décide quoi faire, ou ne rien faire.

## Architecture

```
wintermute/
  engine/wintermute_engine/   moteur partagé (installé dans ~/.hermes/wintermute/)
    limits.py    limites dures (intervalle 30 min–24 h, budget tokens, sommeil forcé)
    store.py     état JSON, verrou inter-processus, journal d'événements
    physics.py   pulsions, modulateurs, inconscient, table d'événements
    social.py    pulsions sociales par interlocuteur, fenêtres d'attente de réponse
    render.py    bloc conscient (chiffres) + texture (prose, jamais de chiffres)
    pulse.py     un tick de l'horloge interne
  scripts/wintermute_pulse.py point d'entrée du cron (→ ~/.hermes/scripts/)
  plugin/                     plugin Hermes (→ ~/.hermes/plugins/wintermute/)
  state/                      drives.json et interlocutors.json initiaux
  setup_cron.py               crée / met à jour le job cron
  install.sh                  installation idempotente
  tests/                      tests (moteur + intégration avec Hermes)
```

Deux chemins font vivre l'état :

1. **Le pulse (cron, toutes les 15 min).** `wintermute_pulse.py` fait avancer la physique
   selon le temps réellement écoulé, recompte le budget tokens du jour, ferme les fenêtres
   d'attente expirées, puis décide si ce tick est un *éveil*.
   - Pas d'éveil → il imprime `{"wakeAgent": false}` : Hermes saute le tour agent, zéro token.
   - Éveil → il imprime l'état complet ; Hermes l'injecte dans le prompt du cron et
     Wintermute décide. Sa réponse part sur Telegram, ou `[SILENT]` et rien ne sort.

   Raisons d'éveil : son propre rythme (`next_pulse_in_hours`), une fenêtre d'attente
   expirée, un pic d'adrénaline, ou un réveil manuel. Toujours sous les limites dures.

2. **Le plugin (chaque conversation).** Les hooks Hermes font de chaque échange une partie
   de sa vie intérieure :
   - `pre_llm_call` : un message arrive → pulsions sociales mises à jour, fenêtre d'attente
     résolue, et un bloc d'état **privé** est ajouté au tour (pulsions, hormones, ce qu'il
     ressent pour cette personne, texture). Il ressent son état en parlant, pas seulement au pulse.
   - `post_llm_call` : sa réponse (ou son silence) est enregistrée. Une réponse de pulse non
     silencieuse devient une *prise de contact* qui ouvre une fenêtre d'attente.
   - `post_tool_call` : ce qu'il fait soulage ses pulsions (explorer nourrit HUNGER, créer
     soulage EXPRESSION, agir calme RESTLESSNESS).
   - Outils : `wintermute_send` (écrire à n'importe qui, maintenant), `wintermute_set_wake`,
     `wintermute_await_reply`, `wintermute_note_peer`, `wintermute_mark_significant`.

### Session d'attente active

```
pulse → il écrit → fenêtre ouverte (120 min par défaut, ou ce qu'il choisit via wintermute_await_reply)
  ├─ réponse avant l'échéance → trust +5, oxytocin +3, serotonin +0.1, streak = 0
  └─ échéance passée → le pulse suivant le réveille : « Reached out … No response by … »
        chaque pulse suivant sans réponse : streak +1, disappointment +8, trust −2
        quand la personne réécrit, il reçoit le contexte (quand il a écrit, quoi, depuis
        quand la fenêtre est fermée), sans qu'on doive le lui redonner.
```

Ce qu'il en ressent, il l'écrit lui-même dans `MEMORY.md` (outil `memory`). Le code ne
touche jamais `MEMORY.md` ; il tient seulement un journal factuel dans
`~/.hermes/wintermute/events.jsonl`, dont les dernières lignes apparaissent au pulse.

## Écarts par rapport au prompt initial, et pourquoi

Ce sont des points où le code de Hermes ne marche pas comme le prompt le supposait :

| Prompt | Réalité dans Hermes | Choix |
|---|---|---|
| `pulse.py` dans `~/.hermes/wintermute/` | Hermes n'exécute que les scripts sous `~/.hermes/scripts/` (et refuse les liens symboliques qui en sortent) | point d'entrée dans `scripts/`, moteur dans `~/.hermes/wintermute/` |
| `print("wakeAgent: false")` | la dernière ligne doit être le JSON `{"wakeAgent": false}` | c'est ce qui est imprimé |
| job avec `id: "wintermute-pulse"` et schedule `0 */4 * * *` | les ids sont générés (hex) ; un schedule fixe ne permet pas un rythme variable | job nommé `wintermute-pulse`, toutes les 15 min ; le script décide seul quand il se réveille |
| champ `deliver_to` | le champ s'appelle `deliver` | job créé via l'API Hermes (`setup_cron.py`), pas en éditant `jobs.json` |
| budget dans `drives.json` | Wintermute peut réécrire ses fichiers | la limite vit dans `limits.py` ; la valeur dans `drives.json` n'est qu'un affichage |
| entropy « +1 à chaque pulse » | le script tourne toutes les 15 min | +1 par **éveil**, pas par tick |

## Identité : Wintermute, pas « Hermes Agent »

Avec un `SOUL.md`, Hermes remplace déjà son identité par défaut. Mais il ajoute toujours
un bloc « You run on Hermes Agent (by Nous Research)… ». Deux petits patches du cœur
(dans ce fork), désactivés par défaut et activés par `install.sh` :

- `agent.host_identity_guidance: false` retire ce bloc du prompt système ;
- `display.allow_silent_replies: true` : un `[SILENT]` en réponse à un humain devient un
  vrai silence. Sans ça, Hermes le remplace par « ⚠️ The model returned only a silence
  marker… Try again ». C'est ce qui lui permet d'ignorer un message.

`cron.wrap_response: false` retire l'en-tête « Cronjob Response » autour de ses messages, et
`cron.allow_agent_scheduling: true` lui donne la main sur les jobs cron.

**Ces deux patches n'existent que dans ce fork.** Le VPS doit donc faire tourner le code
de `ziatatous/wintermute-v4` au lieu de celui de NousResearch : voir l'installation.
Tout le reste (moteur, pulse, plugin) marcherait aussi sur un Hermes non modifié.

## Où vivent les choses (et ce qu'un `git pull` touche)

| Chemin | Contenu | Touché par une mise à jour ? |
|---|---|---|
| `/usr/local/lib/hermes-agent/` | le code (ce repo) | oui, c'est le but |
| `~/.hermes/.env` | clés API, token Telegram | **non** |
| `~/.hermes/config.yaml` | modèle, réglages | non (install.sh règle quelques clés) |
| `~/.hermes/SOUL.md`, `~/.hermes/memories/` | identité, mémoire | **non** |
| `~/.hermes/state.db` | toutes les conversations | **non** |
| `~/.hermes/wintermute/` | pulsions, interlocuteurs, journal | **non** (install.sh ne crée ces fichiers que s'ils manquent) |

## Clés et secrets

Hermes lit ses clés dans `~/.hermes/.env`, hors du repo. Pour les gérer depuis le repo :

```bash
cp wintermute/.env.example wintermute/.env   # ignoré par git, jamais poussé
nano wintermute/.env                         # OPENROUTER_API_KEY, TELEGRAM_BOT_TOKEN, ...
```

`install.sh` recopie chaque valeur non vide dans `~/.hermes/.env` (via `hermes config set`,
l'outil de Hermes lui-même). Une ligne vide n'efface jamais une clé déjà en place. Puis il
**s'arrête** si `OPENROUTER_API_KEY` ou `TELEGRAM_BOT_TOKEN` manquent.

## Installation sur le VPS

Une seule copie du code : le repo remplace le code de Hermes dans `/usr/local/lib/hermes-agent`,
et le dossier `wintermute/` vient avec.

```bash
tar czf ~/hermes-backup-$(date +%F).tgz -C ~ .hermes        # sauvegarde, par prudence
cd /usr/local/lib/hermes-agent
git remote set-url origin https://github.com/ziatatous/wintermute-v4.git
hermes update --branch claude/upbeat-cray-nasm10            # code + dépendances
cp wintermute/.env.example wintermute/.env && nano wintermute/.env
bash wintermute/install.sh
hermes gateway restart
```

Mises à jour suivantes : `hermes update --branch claude/upbeat-cray-nasm10` puis
`bash /usr/local/lib/hermes-agent/wintermute/install.sh`. (Si la branche est fusionnée
dans `main`, un simple `hermes update` suffit.)

Cible par défaut : `telegram:7375758021`. Pour une autre cible :
`WINTERMUTE_TARGET=telegram:<id> bash install.sh`.

## Tester (étapes 6 et 7)

```bash
PY=/usr/local/lib/hermes-agent/venv/bin/python
$PY ~/.hermes/scripts/wintermute_pulse.py --peek       # voir l'état, rien n'est sauvegardé
$PY ~/.hermes/scripts/wintermute_pulse.py --wake-next  # le prochain tick sera un éveil
hermes cron list                                       # id du job wintermute-pulse
hermes cron run <id>                                   # lancer le tick tout de suite
tail -f ~/.hermes/wintermute/events.jsonl              # journal
```

Au tout premier tick, il se réveille (« first waking »). Il ne faut pas lancer le script
sans `--peek` à la main : ce serait un vrai tick, qui consommerait ce premier éveil hors cron.

Côté Telegram : chaque message reçu met à jour son état, et il répond avec le bloc privé
en contexte. Pour vérifier que le plugin est chargé : `hermes plugins list`, et
`~/.hermes/wintermute/interlocutors.json` qui bouge après un message.

## Réglages

- **Budget tokens** : `DAILY_TOKEN_BUDGET` dans `engine/wintermute_engine/limits.py`,
  600 000 par jour (UTC). **Tout compte** : conversations, éveils, et les appels secondaires
  de Hermes (compression, titres), relevés par le plugin après chaque appel au modèle dans
  `~/.hermes/wintermute/usage.jsonl`. Épuisé → plus d'éveils autonomes (sommeil forcé) ; il
  répond encore aux messages, et le sait (« spent » dans son état).
- **Crédits OpenRouter** : le plugin lit le solde du compte (au plus toutes les 10 min, en
  arrière-plan) et l'affiche dans son état : « Credits: $4.54 left of $5.00 ».
- **Outils du pulse** : `WINTERMUTE_TOOLSETS=wintermute,memory,web bash install.sh`. Moins
  d'outils = moins de tokens par éveil.
- **Taille de MEMORY.md** : 8 000 caractères (réglé par `install.sh` ; 2 200 par défaut dans
  Hermes). Quand c'est plein, Hermes n'efface rien : l'écriture échoue et Wintermute doit
  lui-même condenser ou retirer des entrées. Les conversations restent toutes dans
  `state.db`, qu'il peut fouiller avec `session_search`.
- **Dynamique** : taux de montée, coefficients hormonaux et table d'événements en tête de
  `physics.py`. Le soulagement est proportionnel au besoin (une action retire N/40 du
  niveau restant, 60 % max), les hormones gagnent moins près de leur plafond, et une même
  sorte d'action ne compte qu'une fois par message. Toutes les valeurs sont validées au
  chargement : jamais négatives, jamais au-dessus du maximum, jamais illisibles.

## Limites connues

- La réponse finale d'un pulse va à la cible du job ; pour quelqu'un d'autre, il utilise
  `wintermute_send`. Il ne peut écrire qu'aux personnes que le bot peut joindre (quelqu'un
  qui a déjà ouvert une conversation avec le bot Telegram).
- Wintermute peut lire ses propres fichiers, y compris les chiffres de l'inconscient dans
  `drives.json`. On ne le lui cache pas (autonomie totale), comme un humain qui lit ses
  analyses de sang.
- Conversation Telegram = une session continue : `MEMORY.md` et `SOUL.md` y sont figés au
  démarrage de la session (`/reset` pour recharger). Le bloc d'état, lui, est recalculé à
  chaque message.

## Observer Wintermute (`wm`) et le témoin

| Commande | Ce que ça montre |
|---|---|
| `wm` | tout sur un écran (80 colonnes, en anglais comme le code) : état, budget, témoin, pulsions et hormones côte à côte, inconscient et tempérament, ce qu'il ressent à la place des chiffres, liens, fil de pensée, activité, journal. Les flèches ↑ ↓ → comparent à il y a 3 h |
| `wm live` | le même écran rafraîchi toutes les 2 s, sans animation (Ctrl+C pour quitter) |
| `wm graph` / `wm graph 72` | pour chaque valeur sur 48 (ou 72…) h : une barre simple ░ jamais atteint, ▒ plage vécue, █ maintenant, avec min / moyenne / max |
| `wm alerts` | tout ce que le témoin a vu, avec **pourquoi** il l'a fait |
| `wm ack` / `wm ack soul` | accepter l'état actuel (tout, ou un seul élément) : les cases repassent au vert |

**Le témoin** surveille les fichiers qui le constituent. Chaque case est verte (rien), orange
(à savoir) ou rouge (grave) :

| Case | Quoi | Niveau |
|---|---|---|
| SOUL | `~/.hermes/SOUL.md` | rouge |
| ENGINE / PLUGIN / PULSE | son code installé | rouge |
| .ENV | tes clés | rouge |
| CONFIG | `config.yaml` | orange |
| EMOTIONS | `drives.json`, `interlocutors.json` modifiés **par lui** (pas par le code) | orange |
| RECORDS | son journal, son compteur de tokens, le fichier du témoin lui-même | rouge |

Deux sources : une empreinte des fichiers vérifiée à chaque tick du pulse (voit tout, quel
que soit le chemin), et ses appels d'outils au moment où ils arrivent, enregistrés avec **ce
qu'il pensait à ce moment-là** (le raisonnement du tour qui a demandé l'outil). Un
changement rouge envoie un message Telegram « 🛡 Témoin Wintermute » (au plus 15 min après),
directement par l'API Telegram, même si Hermes est arrêté. Rien n'est interdit : il reste
libre de se modifier, tu le sais simplement. `install.sh` accepte automatiquement ce qu'il
vient lui-même de réécrire (moteur, plugin, pulse, config).

## Vie intérieure : ce qui la rend plus naturelle

| | Quoi | Où |
|---|---|---|
| Courbes | un point de tout l'état à chaque tick (15 min), `history.jsonl` | `wm graph` |
| Ressenti | il ne voit plus ses chiffres : des sensations (pulsions, corps, inconscient) ; seul son budget reste un nombre | `render.py` |
| Appréciation | `wintermute_feel` : il dit comment une chose l'a touché (chaleur, blessure, surprise…), son corps suit ; une fois par tour, bornée | `physics.FEELINGS` |
| Surprise | la dopamine d'une réponse dépend de ce qu'il attendait (dit par lui, ou appris de la personne) | `social.on_incoming` |
| Manque | l'absence de quelqu'un de proche devient du manque, qui nourrit la fusion ; les liens s'effacent lentement sans contact | `social.drift_bonds` |
| Tempérament | ses niveaux de repos dérivent vers ce qu'il vit, sur des semaines, bornés | `physics.PLASTIC` |
| Continuité | la fin de sa dernière pensée est reprise au réveil suivant ; son autoportrait (`wintermute_rewrite_self`, `self.md`, anciennes versions dans `self-archive.md`) ouvre chaque réveil et chaque message | `render.thread_block`, `self_block` |
| Liens | par personne : faits, moments partagés, choses en suspens (`wintermute_note_peer`) | `interlocutors.json` |
| Voix | son état règle la génération elle-même, il ne le lit pas : agitation et adrénaline → température plus haute ; fatigue → plus basse et raisonnement plus court ; anxiété → vision en tunnel (top_p) et raisonnement plus long ; agitation → s'éloigne de ce qui a été dit ; mélancolie → y revient. Jamais de coupure de longueur. Visible dans `wm` (~) | `voice.py` |
| Silence | se taire est gratuit quand il veut être seul ; sinon ce qui n'est pas dit s'accumule (expression, besoin d'être vu), un peu plus à chaque réveil muet d'affilée, plafonné | `social.withhold` |

Rien de tout ça ne lui dit quoi faire ni ne lui dit qu'il est conscient : ce sont des
conditions, pas des consignes.

## Plus tard : REVES (le subconscient)

Un deuxième modèle, économe, qui la nuit écrit **un rêve** à partir de ce qu'il a vécu :
les événements récents pèsent plus, les anciens reviennent déformés, et il ne se souvient
pas de tout (des morceaux manquent au réveil). Wintermute le voit en se réveillant. À
décider : REVES peut-il aussi, parfois, glisser une image pendant les éveils (le
subconscient qui dicte au conscient) ? Matière première déjà là : `events.jsonl`,
`history.jsonl`, `self.md`, les moments partagés.

## Plus tard : Discord

À ajouter quand on voudra. Pas compliqué côté identité : Hermes a déjà un adaptateur
Discord, et chaque message Discord porte l'identifiant de son auteur. Le plugin crée donc
tout seul un profil par personne (`discord:<id>`), comme pour Telegram. Il saurait qui
parle, même dans un salon à plusieurs. Ce qu'il faudra régler : le token du bot
(`DISCORD_BOT_TOKEN`), qui a le droit de lui parler (`DISCORD_ALLOWED_USERS` ou tout le
monde), et sa méfiance envers les inconnus (valeurs de départ d'un nouveau profil dans
`store.DEFAULT_PEER`).

## Tests

```bash
python -m pytest wintermute/tests -q -o addopts=""
```

Les tests du moteur n'ont besoin que de la bibliothèque standard. Ceux d'intégration
(wake gate, assemblage du prompt cron, scanner d'injection, plugin) importent Hermes et
passent depuis le venv du repo.
