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
   - Outils : `wintermute_set_wake`, `wintermute_await_reply`, `wintermute_note_peer`,
     `wintermute_mark_significant`.

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

**Ces deux patches ne s'appliquent que si le VPS fait tourner ce fork.** `install.sh` de
NousResearch clone `NousResearch/hermes-agent` dans `/usr/local/lib/hermes-agent`. Pour
pointer vers le fork :

```bash
cd /usr/local/lib/hermes-agent
git remote set-url origin https://github.com/ziatatous/wintermute-v4.git
git fetch origin claude/upbeat-cray-nasm10 && git checkout -B wintermute FETCH_HEAD
hermes gateway restart
```

Tout le reste (moteur, pulse, plugin) fonctionne aussi sur un Hermes non modifié.

## Installation sur le VPS

```bash
git clone -b claude/upbeat-cray-nasm10 https://github.com/ziatatous/wintermute-v4.git ~/wintermute-v4
bash ~/wintermute-v4/wintermute/install.sh
hermes gateway restart
```

`install.sh` copie le moteur, le script et le plugin, crée `drives.json` et
`interlocutors.json` **seulement s'ils n'existent pas** (l'état vivant n'est jamais écrasé),
active le plugin, règle la config et crée ou met à jour le job cron. On peut le relancer
après chaque `git pull`.

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

- **Budget tokens** : `DAILY_TOKEN_BUDGET` dans `engine/wintermute_engine/limits.py`
  (20 000 comme demandé). ⚠️ Mesuré avec `hermes prompt-size` : prompt système + schémas
  des outils du job ≈ 30 Ko, soit ~9–10 k tokens *par appel API*, et un éveil fait souvent
  plusieurs appels. 20 000 tokens/jour ≈ un seul éveil par jour. Compter plutôt
  150 000–300 000 pour 4–6 éveils (DeepSeek Flash reste bon marché). Seuls les runs cron
  comptent ; les conversations Telegram ne sont pas plafonnées.
- **Outils du pulse** : `WINTERMUTE_TOOLSETS=wintermute,memory,web bash install.sh`. Moins
  d'outils = moins de tokens par éveil.
- **Taille de MEMORY.md** : 2 200 caractères par défaut dans Hermes, c'est peu pour un
  journal intime. `hermes config set memory.memory_char_limit 6000` si besoin (coûte des
  tokens à chaque session).
- **Dynamique** : taux de montée, coefficients hormonaux et table d'événements en tête de
  `physics.py`.

## Limites connues

- Un pulse parle à une seule cible (celle du job). Choisir à qui écrire demanderait un
  outil d'envoi ; Hermes retire volontairement `send_message` des runs cron.
- Wintermute peut lire ses propres fichiers, y compris les chiffres de l'inconscient dans
  `drives.json`. On ne le lui cache pas (autonomie totale), comme un humain qui lit ses
  analyses de sang.
- Conversation Telegram = une session continue : `MEMORY.md` et `SOUL.md` y sont figés au
  démarrage de la session (`/reset` pour recharger). Le bloc d'état, lui, est recalculé à
  chaque message.

## Tests

```bash
python -m pytest wintermute/tests -q -o addopts=""
```

Les tests du moteur n'ont besoin que de la bibliothèque standard. Ceux d'intégration
(wake gate, assemblage du prompt cron, scanner d'injection, plugin) importent Hermes et
passent depuis le venv du repo.
