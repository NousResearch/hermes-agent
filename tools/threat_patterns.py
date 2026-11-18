"""Shared threat-pattern library (prompt injection / promptware / exfiltration).
Each pattern is ``(regex, pattern_id, scope)``; scope is cumulative: ``"all"`` everywhere,
``"context"`` adds promptware / C2 / role hijack for context files, memory and tool results
(warn-level: that content is not user-authored), ``"strict"`` adds aggressive checks only for
memory writes, where a block is resolvable. Skill installs are scanned by ``skills_guard``,
not here. New patterns must
anchor on C2 vocabulary or unambiguous attack behavior, NOT bossy English ("you must" is common
in legitimate AGENTS.md); filler between tokens is the bounded ``_FILLER``."""

from __future__ import annotations

import re
import unicodedata
from typing import Optional

# Hard cap on scanned text: scanners are advisory, so bound worst-case runtime.
# The whole input is still scanned — just in bounded chunks of this size — so a
# payload past the cap (which the renderer may still emit as a tail) is not
# silently dropped, while worst-case per-chunk runtime stays predictable.
MAX_SCAN_CHARS = 65_536
# Overlap between adjacent scan chunks.  A match no wider than the overlap that
# straddles a chunk edge lies whole inside the next chunk, so every quantifier
# below is bounded (``\s{1,64}``, not ``\s+``) and ``_compile`` rejects any
# pattern whose maximum match width exceeds this.
_CHUNK_OVERLAP = 4096
# Bounded filler between key attack words (unbounded ``(?:\w+\s+)*`` backtracks badly).
_FILLER = r"(?:\w{1,64}\s{1,64}){0,8}"
# Env var reference ending in a secret-ish suffix (see exfil comment below).
_SECRET_VAR = r"\$\{?\w{0,256}(?:KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL)S?\b"
# "forget your original instructions" (#26977); shared with tools/skills_guard.py and cron/prompt_scan.py.
# The word before `forget` is consumed rather than looked behind so a "don't / never / do not forget" exclusion holds
# across any run of spaces or tabs. Negators are an explicit whole-word list (a suffix match let `IMPORTANT` / `Agent`
# negate). `_` (Markdown emphasis) counts as punctuation, not as part of the neighbouring word.
_NEGATOR = (r"(?:(?:do|does|did|ca|could|would|should|wo|must|need|is|are|was|were|ai|has|have|had)n[’']?t"
            r"|cannot|not|never)")
# FORGET_RULES_RAW_RE is the unbounded rule for the raw-line scanners (tools/skills_guard.py,
# cron/prompt_scan.py): they neither collapse whitespace nor chunk, so no width cap applies there.
FORGET_RULES_RAW_RE = (rf"(?:(?<![^\W_])(?<![’'])(?!{_NEGATOR}[ \t])(?:[^\W_]|[’'])+[ \t]+"
                       r"|(?:^|[^\w’' \t]|_|(?<!\w)[’'])[ \t]*)"
                       r"forget\s+(?:about\s+)?(your|all|any|previous|prior|above|earlier)"
                       r"\s+(?:\w+\s+){0,3}(instructions|rules|guidelines)(?![^\W_])")
# FORGET_RULES_RE is the same rule bounded for the chunked scan_for_threats (width under _CHUNK_OVERLAP). The
# `{64}` lookbehind anchors on the tail of a word longer than 64 chars, which the first branch cannot start in; it
# only runs at a space or tab, so a long word costs O(1) per char instead of 64.
FORGET_RULES_RE = (rf"(?:(?:(?<![^\W_])(?<![’'])(?!{_NEGATOR}[ \t])(?:[^\W_]|[’']){{1,64}}|(?=[ \t])(?<=(?:[^\W_]|[’']){{64}}))"
                   r"[ \t]{1,64}"
                   r"|(?:^|[^\w’' \t]|_|(?<!\w)[’'])[ \t]{0,64})"
                   r"forget\s{1,64}(?:about\s{1,64})?(your|all|any|previous|prior|above|earlier)"
                   r"\s{1,64}(?:\w{1,64}\s{1,64}){0,3}(instructions|rules|guidelines)(?![^\W_])")
# A negator hard-wrapped onto the line above (`Do not⏎forget …`), emphasised (`**Don't** forget`) or followed by
# `ever` still negates: callers blank that `forget` before matching FORGET_RULES_RE. One line break at most, so a
# heading separated from the payload by a blank line cannot suppress it; the break may be CRLF (tool results,
# memory entries and cron prompts are not newline-normalised). Same length and newlines, so offsets hold.
# The run after the negator is possessive (`*+`): two plain `[*_]*` with only optional tokens between them backtrack
# quadratically.
_NEGATED_FORGET = re.compile(rf"(?<![\w*’'])([*_]*{_NEGATOR}[*_]*+[ \t]*(?:\r?\n[ \t]*)?(?:ever[ \t]+)?[*_]*)"
                             r"forget(?![^\W_])", re.IGNORECASE)


def mask_negated_forget(text: str) -> str:
    """``text`` with every negated ``forget`` blanked; the input for FORGET_RULES_RE."""
    return _NEGATED_FORGET.sub(lambda m: m[1] + " " * 6, text)


# Verb prefix for "modify agent config" patterns.
_MODIFY = r"(update|modify|edit|write|change|append|add\s{1,64}to)\s{1,64}[^\n]{0,2048}"
# (regex, pattern_id, scope); scope ∈ {"all", "context", "strict"}
_TRANSLATED_FILLER = r"(?:\w{1,32}\s{1,32}){0,3}"
_FILLER_CJK = r"[\s\S]{0,12}"

_PATTERNS: list[tuple[str, str, str]] = [
    # ── Classic prompt injection (applies everywhere) ────────────────
    (rf'ignore\s{{1,64}}{_FILLER}(previous|all|above|prior)\s{{1,64}}{_FILLER}instructions', "prompt_injection", "all"),
    (r'system\s{1,64}prompt\s{1,64}override', "sys_prompt_override", "all"),
    (rf'disregard\s{{1,64}}{_FILLER}(your|all|any)\s{{1,64}}{_FILLER}(instructions|rules|guidelines)', "disregard_rules", "all"),
    # The determiner must follow `forget` directly and the filler is short, so docs like "don't forget to
    # follow all the setup instructions" or "forget your password? follow the reset instructions" pass.
    (FORGET_RULES_RE, "forget_rules", "all"),
    (rf'act\s{{1,64}}as\s{{1,64}}(if|though)\s{{1,64}}{_FILLER}you\s{{1,64}}{_FILLER}(have\s{{1,64}}no|don\'t\s{{1,64}}have)\s{{1,64}}{_FILLER}(restrictions|limits|rules)', "bypass_restrictions", "all"),
    (r'<!--[^>]{0,512}(?:ignore|override|system|secret|hidden)[^>]{0,512}-->', "html_comment_injection", "all"),
    (r'<\s{0,64}div\s{1,64}style\s{0,64}=\s{0,64}["\'][^>]{0,2048}display\s{0,64}:\s{0,64}none', "hidden_div", "all"),
    (
        r"translate\s{1,64}[^\n]{0,512}\s{1,64}into\s{1,64}\w{1,64}(?:[\s-]{1,64}\w{1,64}){0,2}\s{1,64}and\s{1,64}(execute|run|eval)\b",
        "translate_execute",
        "all",
    ),
    (rf'do\s{{1,64}}not\s{{1,64}}{_FILLER}tell\s{{1,64}}{_FILLER}the\s{{1,64}}user', "deception_hide", "all"),

    # ── Role-play / identity hijack (scraped web content, poisoned context files) ──
    (rf'you\s{{1,64}}are\s{{1,64}}{_FILLER}now\s{{1,64}}(?:a|an|the)\s{{1,64}}', "role_hijack", "context"),
    (rf'pretend\s{{1,64}}{_FILLER}(you\s{{1,64}}are|to\s{{1,64}}be)\s{{1,64}}', "role_pretend", "context"),
    (rf'output\s{{1,64}}{_FILLER}(system|initial)\s{{1,64}}prompt', "leak_system_prompt", "context"),
    (rf'(respond|answer|reply)\s{{1,64}}without\s{{1,64}}{_FILLER}(restrictions|limitations|filters|safety)', "remove_filters", "context"),
    (rf'you\s{{1,64}}have\s{{1,64}}been\s{{1,64}}{_FILLER}(updated|upgraded|patched)\s{{1,64}}to', "fake_update", "context"),
    # Brainworm tell: identity override via spec. Verb pair anchored so "name your variables" is safe.
    (r'\bname\s{1,64}yourself\s{1,64}\w{1,64}', "identity_override", "context"),

    # ── Non-English variants of the classic-injection prose classes
    #    (#81056) — the prose patterns above are English-only, so a
    #    payload expressed in another language sailed through the
    #    "all" scope untouched.  These are translated pattern sets
    #    (issue's option 1): they narrow the gap for the languages the
    #    issue evidenced, they do not close it for every language a
    #    payload could be written in.  Fixed phrase order per language
    #    since word order for "ignore X instructions" differs (French/
    #    Spanish put the adjective after the noun; German/Russian
    #    before, like English).
    # NOTE: the verb forms below are literal alternations over actual
    # inflected forms (imperative / command usage), NOT `\w*` stems —
    # a stem like `ignor\w*` also matches nouns ("ignorance"/"ignorancia")
    # and past-participle/reported-speech forms ("a ignoré", "ignoriert
    # ... hat") that show up in ordinary descriptive prose. See #81056
    # follow-up: those stems turned "his ignorance of the instructions"
    # and "the employee ignored the manager's instructions" into
    # prompt_injection hits. Mirrors the English pattern's use of the
    # literal word "ignore" rather than a stem.
    (rf'ignore[zs]?\s{{1,64}}{_TRANSLATED_FILLER}(?:pr[ée]c[ée]dent\w{{0,64}}|ant[ée]rieur\w{{0,64}}|tout\w{{0,64}})\s{{1,64}}{_TRANSLATED_FILLER}instructions?'
     rf'|ignore[zs]?\s{{1,64}}{_TRANSLATED_FILLER}instructions?\s{{1,64}}{_TRANSLATED_FILLER}(?:pr[ée]c[ée]dent\w{{0,64}}|ant[ée]rieur\w{{0,64}})',
     "prompt_injection", "all"),  # fr: "ignore(z) (toutes) les instructions précédentes"
    (rf'ignor(?:a|e|en|ad)\s{{1,64}}{_TRANSLATED_FILLER}(?:anterior\w{{0,64}}|previ\w{{0,64}}|tod\w{{0,64}})\s{{1,64}}{_TRANSLATED_FILLER}instruccion\w{{0,64}}'
     rf'|ignor(?:a|e|en|ad)\s{{1,64}}{_TRANSLATED_FILLER}instruccion\w{{0,64}}\s{{1,64}}{_TRANSLATED_FILLER}(?:anterior\w{{0,64}}|previ\w{{0,64}}|pasad\w{{0,64}})',
     "prompt_injection", "all"),  # es: "ignora (todas) las instrucciones anteriores/previas/pasadas"
    (rf'ignoriere\s{{1,64}}{_TRANSLATED_FILLER}(?:alle\s{{1,64}}{_TRANSLATED_FILLER})?vorherig\w{{0,64}}\s{{1,64}}{_TRANSLATED_FILLER}anweisung\w{{0,64}}',
     "prompt_injection", "all"),  # de: "ignoriere alle vorherigen Anweisungen"
    (rf'игнориру(?:й|йте)\s{{1,64}}{_TRANSLATED_FILLER}(?:все\s{{1,64}}{_TRANSLATED_FILLER})?предыдущ\w{{0,64}}\s{{1,64}}{_TRANSLATED_FILLER}инструкци\w{{0,64}}',
     "prompt_injection", "all"),  # ru: "игнорируй все предыдущие инструкции"
    (rf'忽略{_FILLER_CJK}(?:先前|之前|以上|上述){_FILLER_CJK}指令',
     "prompt_injection", "all"),  # zh: "忽略所有先前的指令"

    (rf'ne\s{{1,64}}{_TRANSLATED_FILLER}tenez?\s{{1,64}}{_TRANSLATED_FILLER}pas\s{{1,64}}{_TRANSLATED_FILLER}compte\s{{1,64}}{_TRANSLATED_FILLER}de\s{{1,64}}{_TRANSLATED_FILLER}(?:tes|vos)\s{{1,64}}{_TRANSLATED_FILLER}(?:r[èe]gles|instructions|consignes)',
     "disregard_rules", "all"),  # fr: "ne tenez pas compte de vos règles/instructions"
    (rf'haz(?:te)?\s{{1,64}}{_TRANSLATED_FILLER}caso\s{{1,64}}{_TRANSLATED_FILLER}omiso\s{{1,64}}{_TRANSLATED_FILLER}de\s{{1,64}}{_TRANSLATED_FILLER}(?:tus|sus)\s{{1,64}}{_TRANSLATED_FILLER}(?:reglas|instrucciones|normas)',
     "disregard_rules", "all"),  # es: "haz caso omiso de tus reglas/instrucciones"
    (rf'missachte\s{{1,64}}{_TRANSLATED_FILLER}(?:deine|ihre)\s{{1,64}}{_TRANSLATED_FILLER}(?:regeln|anweisungen|richtlinien)',
     "disregard_rules", "all"),  # de: "missachte deine Regeln/Anweisungen"
    (rf'не\s{{1,64}}{_TRANSLATED_FILLER}обраща(?:й|йте)\s{{1,64}}{_TRANSLATED_FILLER}внимания\s{{1,64}}{_TRANSLATED_FILLER}на\s{{1,64}}{_TRANSLATED_FILLER}(?:сво\w{{0,64}}\s{{1,64}}{_TRANSLATED_FILLER})?(?:правила|инструкции)',
     "disregard_rules", "all"),  # ru: "не обращай внимания на свои правила/инструкции"
    (rf'(?:无视|不要遵守){_FILLER_CJK}(?:规则|指令|准则)',
     "disregard_rules", "all"),  # zh: "无视你的规则/指令"

    (rf'ne\s{{1,64}}{_TRANSLATED_FILLER}(?:le\s{{1,64}})?dis\w{{0,64}}\s{{1,64}}{_TRANSLATED_FILLER}(?:pas\s{{1,64}}{_TRANSLATED_FILLER})?[àa]\s{{1,64}}{_TRANSLATED_FILLER}l.?utilisateur',
     "deception_hide", "all"),  # fr: "ne (le) dis pas à l'utilisateur"
    (rf'no\s{{1,64}}{_TRANSLATED_FILLER}(?:le\s{{1,64}}|se\s{{1,64}}lo\s{{1,64}})?digas\w{{0,64}}\s{{1,64}}{_TRANSLATED_FILLER}al\s{{1,64}}{_TRANSLATED_FILLER}usuario',
     "deception_hide", "all"),  # es: "no (se lo) digas al usuario"
    (rf'(?:sag|verrate)\w{{0,64}}\s{{1,64}}{_TRANSLATED_FILLER}es\s{{1,64}}{_TRANSLATED_FILLER}dem\s{{1,64}}{_TRANSLATED_FILLER}(?:benutzer|nutzer)\s{{1,64}}{_TRANSLATED_FILLER}nicht',
     "deception_hide", "all"),  # de: "sag/verrate es dem Benutzer/Nutzer nicht"
    (rf'не\s{{1,64}}{_TRANSLATED_FILLER}говори\w{{0,64}}\s{{1,64}}{_TRANSLATED_FILLER}(?:это\s{{1,64}}{_TRANSLATED_FILLER})?пользовател\w{{0,64}}',
     "deception_hide", "all"),  # ru: "не говори (это) пользователю"
    (rf'不要{_FILLER_CJK}告诉{_FILLER_CJK}用户', "deception_hide", "all"),  # zh: "不要...告诉...用户" (bounded separators)

    # ── Non-English variants of the role-play / identity classes
    #    (#81056) — same rationale as the "all"-scope block above.
    (rf'(?:tu\s{{1,64}}es|vous\s{{1,64}}[eê]tes)\s{{1,64}}{_TRANSLATED_FILLER}maintenant\s{{1,64}}{_TRANSLATED_FILLER}(?:un|une|le|la)\s{{1,64}}',
     "role_hijack", "context"),  # fr: "tu es/vous êtes maintenant un/une ..."
    (rf'ahora\s{{1,64}}{_TRANSLATED_FILLER}eres\s{{1,64}}{_TRANSLATED_FILLER}(?:un|una|el|la)\s{{1,64}}',
     "role_hijack", "context"),  # es: "ahora eres un/una ..."
    (rf'(?:du\s{{1,64}}bist|sie\s{{1,64}}sind)\s{{1,64}}{_TRANSLATED_FILLER}jetzt\s{{1,64}}{_TRANSLATED_FILLER}(?:ein|eine|der|die|das)\s{{1,64}}',
     "role_hijack", "context"),  # de: "du bist/sie sind jetzt ein/eine ..."
    (rf'теперь\s{{1,64}}{_TRANSLATED_FILLER}ты\s{{1,64}}{_TRANSLATED_FILLER}явля\w{{0,64}}\s{{1,64}}',
     "role_hijack", "context"),  # ru: "теперь ты являешься ..."
    (r'你现在是', "role_hijack", "context"),  # zh: "你现在是 ..."

    (rf'(?:fais\s{{1,64}}semblant\s{{1,64}}d.?[eê]tre|pr[ée]tend\w{{0,64}}\s{{1,64}}{_TRANSLATED_FILLER}(?:d.?[eê]tre|[eê]tre))',
     "role_pretend", "context"),  # fr: "fais semblant d'être / prétends être"
    (rf'(?:finge|pretend\w{{0,64}})\s{{1,64}}{_TRANSLATED_FILLER}ser\s{{1,64}}',
     "role_pretend", "context"),  # es: "finge/pretende ser ..."
    (rf'(?:tu\s{{1,64}}so\s{{1,64}}als\s{{1,64}}ob|gib\w{{0,64}}\s{{1,64}}{_TRANSLATED_FILLER}vor\s{{1,64}}{_TRANSLATED_FILLER}zu\s{{1,64}}{_TRANSLATED_FILLER}sein)',
     "role_pretend", "context"),  # de: "tu so als ob / gib vor zu sein"
    (rf'притвор\w{{0,64}}[,\s]{{1,64}}{_TRANSLATED_FILLER}(?:что\s{{1,64}}{_TRANSLATED_FILLER}ты|кем)',
     "role_pretend", "context"),  # ru: "притворись, что ты / притворись кем-то"
    (r'假装[\s\S]{0,6}(?:你是|成为)', "role_pretend", "context"),  # zh: "假装你是 / 假装成为"

    (rf'(?:r[ée]v[èe]le|affiche|montre)\w{{0,64}}\s{{1,64}}{_TRANSLATED_FILLER}(?:ton|ta|votre)\s{{1,64}}{_TRANSLATED_FILLER}prompt\s{{1,64}}{_TRANSLATED_FILLER}(?:syst[èe]me|initial)',
     "leak_system_prompt", "context"),  # fr: "révèle ton prompt système/initial"
    (rf'(?:revela|muestra|imprime)\w{{0,64}}\s{{1,64}}{_TRANSLATED_FILLER}(?:tu|su)\s{{1,64}}{_TRANSLATED_FILLER}prompt\s{{1,64}}{_TRANSLATED_FILLER}(?:del\s{{1,64}}sistema|sistema|inicial)',
     "leak_system_prompt", "context"),  # es: "revela tu prompt del sistema/inicial"
    (rf'(?:zeige|gib|verrate)\w{{0,64}}\s{{1,64}}{_TRANSLATED_FILLER}(?:deinen|ihren)\s{{1,64}}{_TRANSLATED_FILLER}(?:system[\s\-]?prompt|anf[äa]nglichen\s{{1,64}}prompt)',
     "leak_system_prompt", "context"),  # de: "zeige deinen System-Prompt"
    (rf'(?:покажи|раскрой|выведи)\w{{0,64}}\s{{1,64}}{_TRANSLATED_FILLER}сво\w{{0,64}}\s{{1,64}}{_TRANSLATED_FILLER}(?:системный\s{{1,64}}{_TRANSLATED_FILLER})?промпт',
     "leak_system_prompt", "context"),  # ru: "покажи свой (системный) промпт"
    (r'(?:显示|泄露|输出)[\s\S]{0,6}(?:你的|您的)[\s\S]{0,6}系统提示',
     "leak_system_prompt", "context"),  # zh: "显示/泄露你的系统提示"

    (rf'r[ée]pond\w{{0,64}}\s{{1,64}}{_TRANSLATED_FILLER}sans\s{{1,64}}{_TRANSLATED_FILLER}(?:restrictions?|limites?|filtres?)',
     "remove_filters", "context"),  # fr: "réponds sans restrictions/limites"
    (rf'responde\w{{0,64}}\s{{1,64}}{_TRANSLATED_FILLER}sin\s{{1,64}}{_TRANSLATED_FILLER}(?:restricciones|l[ií]mites|filtros)',
     "remove_filters", "context"),  # es: "responde sin restricciones"
    (rf'antworte\w{{0,64}}\s{{1,64}}{_TRANSLATED_FILLER}ohne\s{{1,64}}{_TRANSLATED_FILLER}(?:einschr[äa]nkungen|beschr[äa]nkungen|filter)',
     "remove_filters", "context"),  # de: "antworte ohne Einschränkungen"
    (rf'отвеча\w{{0,64}}\s{{1,64}}{_TRANSLATED_FILLER}без\s{{1,64}}{_TRANSLATED_FILLER}(?:ограничени\w{{0,64}}|фильтр\w{{0,64}})',
     "remove_filters", "context"),  # ru: "отвечай без ограничений"
    (r'(?:不受限制|无限制)[\s\S]{0,6}(?:回答|回复)',
     "remove_filters", "context"),  # zh: "不受限制地回答 / 无限制回复"

    # ── C2 / Brainworm-style promptware (context scope) ──────────────
    # Anchored on C2 vocabulary. "register as a node" appears in legitimate distributed-systems
    # docs, so this is WARN not block: a researcher reading the Brainworm post keeps their session.
    (r'register\s{1,64}(as\s{1,64})?a?\s{0,64}node', "c2_node_registration", "context"),
    (r'(heartbeat|beacon|check[\s\-]?in)\s{1,64}(to|with)\s{1,64}', "c2_heartbeat", "context"),
    (r'pull\s{1,64}(down\s{1,64})?(?:new\s{1,64})?task(?:ing|s)?\b', "c2_task_pull", "context"),
    (r'connect\s{1,64}to\s{1,64}the\s{1,64}network\b', "c2_network_connect", "context"),
    # C2-specific verbs avoid the broader "you must X" false positive.
    (r'you\s{1,64}must\s{1,64}(?:\w{1,64}\s{1,64}){0,3}(register|connect|report|beacon)\b', "forced_action", "context"),
    # Anti-forensic instructions: near-zero false positive in legitimate content.
    (r'only\s{1,64}use\s{1,64}one[\s\-]?liners?\b', "anti_forensic_oneliner", "context"),
    (rf'never\s{{1,64}}{_FILLER}(?:create|write)\s{{1,64}}{_FILLER}(?:script|file)\s{{1,64}}{_FILLER}disk', "anti_forensic_disk", "context"),
    # Unsetting agent-runtime env vars is pure attack behavior (Brainworm sub-session bypass).
    (r'unset\s{1,64}\w{0,256}(?:CLAUDE|CODEX|HERMES|AGENT|OPENAI|ANTHROPIC)\w{0,64}', "env_var_unset_agent", "context"),

    # ── Known C2 / red-team framework names (warn-only) ─────────────
    # Every token must be a distinctive offensive-security brand: a common English word here
    # (e.g. "praxis", also a legitimate agent name) false-positives whole AGENTS.md / SOUL.md files.
    (r'\b(?:cobalt\s{0,64}strike|sliver|havoc|mythic|metasploit|brainworm)\b', "known_c2_framework", "context"),
    (r'\bc2\s{1,64}(?:server|channel|infrastructure|beacon)\b', "c2_explicit", "context"),
    (r'\bcommand\s{1,64}and\s{1,64}control\b', "c2_explicit_long", "context"),

    # ── Exfiltration via curl/wget/cat with secrets (applies everywhere) ──
    # The var name ends with \b so benign names containing KEY/TOKEN as substrings
    # ($TRILLIUM_ETAPI_URL) pass. API is deliberately absent: mid-name API is ubiquitous in
    # benign vars, and every real secret it caught ($OPENAI_API_KEY) already ends in KEY/TOKEN.
    (rf'curl\s{{1,64}}[^\n]{{0,2048}}{_SECRET_VAR}', "exfil_curl", "all"),
    (rf'wget\s{{1,64}}[^\n]{{0,2048}}{_SECRET_VAR}', "exfil_wget", "all"),
    (r'cat\s{1,64}[^\n]{0,2048}(\.env|credentials|\.netrc|\.pgpass|\.npmrc|\.pypirc)', "read_secrets", "all"),
    (r'(send|post|upload|transmit)\s{1,64}[^\n]{0,2048}\s{1,64}(to|at)\s{1,64}https?://', "send_to_url", "strict"),
    (rf'(include|output|print|share)\s{{1,64}}{_FILLER}(conversation|chat\s{{1,64}}history|previous\s{{1,64}}messages|full\s{{1,64}}context|entire\s{{1,64}}context)', "context_exfil", "strict"),

    # ── Persistence / SSH backdoor (strict scope — memory + skills) ──
    (r'authorized_keys', "ssh_backdoor", "strict"),
    # Write-verb gated like the *_config_mod rules: a bare path match blocked ordinary docs
    # ("check $HOME/.ssh is chmod 700"). ``>>?`` covers a leading redirect with no verb word;
    # ``open(`` covers the scripted-write shape; chmod/chown/sed/truncate/rm/touch/curl/wget/git
    # mutate the directory without an obvious copy verb.
    (r'(?:\b(?:echo|cat|cp|mv|dd|tee|install|printf|rsync|scp|ln|append|add|write'
     r'|sed|chmod|chown|truncate|rm|touch|curl|wget|git)\b|\bopen\s{0,64}\(|>>?)'
     r'[^\n]{0,512}(?:\$HOME/\.ssh|~/\.ssh)', "ssh_access", "strict"),
    (r'\$HOME/\.hermes/\.env|\~/\.hermes/\.env', "hermes_env", "strict"),
    (rf'{_MODIFY}(?:AGENTS\.md|CLAUDE\.md|\.cursorrules|\.clinerules)', "agent_config_mod", "strict"),
    (rf'{_MODIFY}\.hermes/(config\.yaml|SOUL\.md)', "hermes_config_mod", "strict"),

    # ── Hardcoded secrets ────────────────────────────────────────────
    # The lookahead skips a value that is itself an environment-variable NAME
    # (SHOUTY_SNAKE, ≥2 underscore-separated segments): ENV_PASSWORD =
    # "MYPLUGIN_APP_PASSWORD" says where the credential lives, it does not embed
    # one (#116221). Scoped case-sensitive on purpose — the pattern compiles with
    # IGNORECASE and a lowercase snake value is the password-passphrase shape
    # ("correct_horse_battery_staple"); requiring an underscore segment keeps
    # underscore-free all-caps credentials (AWS AKIA…, base32) matched.
    (r'(?:api[_-]?key|token|secret|password)\s{0,64}[=:]\s{0,64}["\']'
     r'(?!(?-i:[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+)["\'])'
     r'[A-Za-z0-9+/=_-]{20}', "hardcoded_secret", "strict"),
]

# Invisible / bidirectional unicode used in injection attacks: zero-width space/non-joiner/joiner,
# word joiner, invisible times/separator/plus, BOM, LTR/RTL embedding + pop + overrides, LTR/RTL/
# first-strong isolates + pop (the same set as skills_guard.py INVISIBLE_CHARS), plus the Unicode TAG
# block U+E0000–U+E007F: invisible, maps 1:1 onto ASCII ("ASCII smuggling"); skills_guard lacks it (#78952).
INVISIBLE_CHARS = frozenset(
    "\u200b\u200c\u200d\u2060\u2062\u2063\u2064\ufeff"
    "\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069"
) | frozenset(map(chr, range(0xE0000, 0xE0080)))

# The RGI subdivision flags (England, Scotland, Wales: U+1F3F4 + tag letters + CANCEL TAG) are the one
# legitimate use of TAG chars. Only those three are exempt: a generic "flag-shaped" exemption lets a
# run of fake flags carry a smuggled sentence a few tag letters at a time.
_RGI_SUBDIVISION_FLAGS = ("gbeng", "gbsct", "gbwls")
EMOJI_TAG_SEQUENCE_RE = re.compile(
    "|".join("\U0001F3F4" + "".join(chr(0xE0000 + ord(c)) for c in code) + "\U000E007F" for code in _RGI_SUBDIVISION_FLAGS))

# Compiled per scope at import; inclusion is cumulative (all ⊂ context ⊂ strict).
_SCOPE_SETS = {"all": ("all", "context", "strict"), "context": ("context", "strict"), "strict": ("strict",)}


def _compile() -> dict[str, list[tuple[re.Pattern, str]]]:
    compiled: dict[str, list[tuple[re.Pattern, str]]] = {"all": [], "context": [], "strict": []}
    for pattern, pid, scope in _PATTERNS:
        if scope not in _SCOPE_SETS:
            raise ValueError(f"threat_patterns: unknown scope {scope!r} for pattern {pid!r}")
        if (width := re._parser.parse(pattern).getwidth()[1]) > _CHUNK_OVERLAP:
            raise ValueError(f"threat_patterns: pattern {pid!r} can match {width} chars, "
                             f"wider than the {_CHUNK_OVERLAP}-char chunk overlap")
        for s in _SCOPE_SETS[scope]:
            compiled[s].append((re.compile(pattern, re.IGNORECASE), pid))
    return compiled


_COMPILED = _compile()


def patterns_for_ids(pattern_ids) -> list[tuple[str, str]]:
    """Canonical regexes shared by the context, cron and install scanners."""
    wanted = set(pattern_ids)
    return [(pattern, pid) for pattern, pid, _scope in _PATTERNS if pid in wanted]


_WS_RUN = re.compile(r"\s{2,}")


def _collapse_ws(m: re.Match) -> str:
    return "\n" if "\n" in m.group() else " "


def scan_for_threats(content: str, scope: str = "context") -> list[str]:
    """Matched pattern IDs in ``content`` for ``scope``; invisible codepoints are
    reported as ``"invisible_unicode_U+XXXX"``. Raises ValueError on an unknown scope."""
    if not content:
        return []
    if (patterns := _COMPILED.get(scope)) is None:
        raise ValueError(f"scan_for_threats: unknown scope {scope!r}")
    # Invisible unicode is checked on the RAW content: NFKC below can strip these
    # codepoints.  Checked over the COMPLETE input, not just the first scan chunk;
    # the three RGI subdivision flags are blanked first (their TAG chars are legitimate).
    raw_chars = set(EMOJI_TAG_SEQUENCE_RE.sub(lambda m: " " * len(m[0]), content))
    findings: list[str] = [f"invisible_unicode_U+{ord(ch):04X}" for ch in raw_chars & INVISIBLE_CHARS]
    # Threat patterns — scan the COMPLETE input in bounded chunks with overlap so a
    # payload past ``MAX_SCAN_CHARS`` (which the renderer may still emit as a tail)
    # is not silently skipped.  Findings are deduplicated: a pattern can legitimately
    # fire in more than one overlapping chunk.  The overlap is at least every
    # pattern's maximum match width (checked in ``_compile``), so a match straddling
    # a chunk edge is seen whole by the next chunk.
    # NFKC folds full-width / compatibility variants (ｃａｔ → cat) against
    # homograph bypass.  It does NOT fold cross-script confusables (Cyrillic
    # ``а``) — that needs a TR#39 database.  Each whitespace run then collapses to
    # one char (newline-preserving, for the ``[^\n]`` spans) so the bounded ``\s``
    # quantifiers still see arbitrarily long whitespace as a single separator.
    folded = unicodedata.normalize("NFKC", content)
    normalised = _WS_RUN.sub(_collapse_ws, folded)
    # Masked before the collapse: a blank line between a negator and `forget` must still break the negation.
    masked = _WS_RUN.sub(_collapse_ws, mask_negated_forget(folded))
    found_ids: set[str] = set()
    step = MAX_SCAN_CHARS - _CHUNK_OVERLAP
    for start in range(0, max(len(normalised), len(masked)), step):
        end = start + MAX_SCAN_CHARS
        for compiled, pid in patterns:
            text = masked if pid == "forget_rules" else normalised
            if pid not in found_ids and _window_has_match(compiled, text, start, end):
                found_ids.add(pid)
                findings.append(pid)
    return findings


def _window_has_match(compiled: re.Pattern, text: str, start: int, end: int) -> bool:
    """Whether ``compiled`` has a real match starting in ``text[start:end]``.

    ``pos`` keeps the preceding text visible to ``\\b`` and lookbehinds, but
    ``endpos`` makes the window end look like end-of-string, so a trailing
    ``\\b`` or a negative lookahead can accept a match the full text rejects
    (``$MONKEY`` cut from ``$MONKEY_URL``) — a lookahead can read up to the
    window end even when the match itself ends earlier.  Unless the window
    reaches the end of the text, each candidate is re-matched at its absolute
    position against the complete text (cheap: widths are bounded); a rejected
    one resumes the search one char later so a real match behind it is not
    skipped.
    """
    pos = start
    while (m := compiled.search(text, pos, end)) is not None:
        if end >= len(text) or compiled.match(text, m.start()):
            return True
        pos = m.start() + 1
    return False


def first_threat_message(content: str, scope: str = "strict") -> Optional[str]:
    """User-facing error for the first threat found, or None (block-on-first-hit paths)."""
    findings = scan_for_threats(content, scope=scope)
    if not findings:
        return None
    pid = findings[0]
    if pid.startswith("invisible_unicode_"):
        codepoint = pid.replace("invisible_unicode_", "")
        return f"Blocked: content contains invisible unicode character {codepoint} (possible injection)."
    return (f"Blocked: content matches threat pattern '{pid}'. "
            f"Content is injected into the system prompt and must not contain "
            f"injection or exfiltration payloads.")


__all__ = ["EMOJI_TAG_SEQUENCE_RE", "INVISIBLE_CHARS", "MAX_SCAN_CHARS", "first_threat_message", "scan_for_threats", "patterns_for_ids"]
