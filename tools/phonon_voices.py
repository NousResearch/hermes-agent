"""Phonon voice catalogue: the 102 voices of gradium/phonon-voices.

Data module, not logic: one row per voice, in the catalogue's own order (English US, English UK,
the 12 English character voices, then French, German, Spanish and Portuguese). ``ptts`` resolves a
voice by name (case- and accent-insensitive) or by ``voice_id``, and downloads only the embedding
the checkpoint lacks, so the names here are what ``tts.phonon.voice`` accepts.

Generated from the repo's ``voices.jsonl`` (format ``ba7a5a45@300``); regenerate rather than
hand-editing when Gradium adds voices. ``descriptions`` are deliberately dropped — they run to
several sentences per voice and nothing here needs them.
"""

from __future__ import annotations

# (name, voice_id, language, locale, gender, age) — gender/age are None for character voices.
PHONON_VOICES: tuple[tuple[str, str, str, str, str | None, str | None], ...] = (
    ("Zoey", "NbpkqMVS3CJeq2j8", "en", "en-us", 'female', 'Adult'),
    ("Juniper", "NIWIOv5vTi5eRMj8", "en", "en-us", 'female', 'Young Adult'),
    ("Everly", "vjFyp2BcCyTlUmYb", "en", "en-us", 'female', 'Middle-age'),
    ("Harper", "4SZHfMpw-p46Ywgs", "en", "en-us", 'female', 'Adult'),
    ("Hadley", "bXwrkHhURL1VFJdG", "en", "en-us", 'female', 'Middle-age'),
    ("Delia", "nGzMHMPAWGaJFHOo", "en", "en-us", 'female', 'Middle-age'),
    ("Easton", "LMDIYE0Oa377Qh6Z", "en", "en-us", 'male', 'Young Adult'),
    ("Micah", "40QrMVgAxFiUnP9f", "en", "en-us", 'male', 'Young Adult'),
    ("Trevor", "0zLApsZfVLLNaEGl", "en", "en-us", 'male', 'Middle-age'),
    ("Russell", "_6Aslh2DxfmnRLmP", "en", "en-us", 'male', 'Adult'),
    ("Preston", "qIxlZ7Ei8KvWlyQH", "en", "en-us", 'male', 'Middle-age'),
    ("Miles", "d9XCyhjo4JhkHF1d", "en", "en-us", 'male', 'Young Adult'),
    ("Maeve", "6PWnV0Nq4wu7RVBT", "en", "en-gb", 'female', 'Adult'),
    ("Freya", "GgfEkEJtxZR7gnpy", "en", "en-gb", 'female', 'Young'),
    ("Pippa", "uem82D50GRv2Dwma", "en", "en-gb", 'female', 'Adult'),
    ("Elodie-Rose", "y9bQqIDnyxwqK01t", "en", "en-gb", 'female', 'Young'),
    ("Imogen", "gDT1nz7Ie36ZhL-C", "en", "en-gb", 'female', 'Adult'),
    ("Lottie", "m9MxpY6IUGXtWp76", "en", "en-gb", 'female', 'Young Adult'),
    ("Alfie", "X6NceqF285Gr04gb", "en", "en-gb", 'male', 'Young Adult'),
    ("Hugh", "YyKcxPG7iIgS3jsK", "en", "en-gb", 'male', 'Middle-age'),
    ("Edwin", "zUCxzrOrugTSis03", "en", "en-gb", 'male', 'Middle-age'),
    ("Spencer", "JOW0vchyswkQUMDS", "en", "en-gb", 'male', 'Middle-age'),
    ("Toby", "dME3IWyZBvmh1n1q", "en", "en-gb", 'male', 'Adult'),
    ("Teddy", "Zn7g6NxLqXlJb7Qi", "en", "en-gb", 'male', 'Young Adult'),
    ("Noa", "b6FvJAiokjdqIti4", "es", "es-es", 'female', 'Young'),
    ("Lucía-Sol", "3p0eIGbkmny71GlA", "es", "es-es", 'female', 'Young'),
    ("Candela", "c9wqBDmQiBie5q6Y", "es", "es-es", 'female', 'Young'),
    ("Celia", "rmLBOdHrfkFym9MM", "es", "es-es", 'female', 'Young Adult'),
    ("Alba", "u2UscyAnHilpiwmf", "es", "es-es", 'female', 'Young'),
    ("Alicia", "4701YgRXjkBcg6sL", "es", "es-es", 'female', 'Middle-age'),
    ("Marcos", "jvPx8j8zLGQ3utZz", "es", "es-es", 'male', 'Adult'),
    ("Borja", "tIjCgP8v3RY8UpgM", "es", "es-es", 'male', 'Young Adult'),
    ("Iker", "t-_TS1e-0GzDAX02", "es", "es-es", 'male', 'Adult'),
    ("Nuno", "fghJGlv90qu8OJ36", "es", "es-es", 'male', 'Middle-age'),
    ("Izan", "Ejb29qHnSOK0cIHY", "es", "es-es", 'male', 'Young Adult'),
    ("Guillermo", "FaLVVqpFQcDOtq45", "es", "es-es", 'male', 'Middle-age'),
    ("Rebeca", "Sx8m6WdXBG06zwZT", "es", "es-mx", 'female', 'Young Adult'),
    ("Regina", "s58clrg2fe0MO7Y-", "es", "es-mx", 'female', 'Adult'),
    ("Danae", "evfxKSxpDIhmsNoq", "es", "es-mx", 'female', 'Young Adult'),
    ("Azucena", "KJpIQlbYrZPA4aLq", "es", "es-mx", 'female', 'Older Adult'),
    ("Jimena", "qG5BXOJYrKEnXLIX", "es", "es-mx", 'female', 'Young Adult'),
    ("Ximena", "VDwnGxAo68C8U8vC", "es", "es-mx", 'female', 'Adult'),
    ("Lalo", "F05gqkzX7WhKVpSL", "es", "es-mx", 'male', 'Young Adult'),
    ("Aldo", "ax7dW1XB0eiMfHgO", "es", "es-mx", 'male', 'Young Adult'),
    ("Yael", "ALczEazoF4n6PmGl", "es", "es-mx", 'male', 'Young Adult'),
    ("Ulises", "G5HLUsQd659UtRe7", "es", "es-mx", 'male', 'Middle-age'),
    ("Leonel", "BkuASiOJRrJEB5qA", "es", "es-mx", 'male', 'Middle-age'),
    ("Nico", "PwgPEaweQILZdGc4", "es", "es-mx", 'male', 'Young Adult'),
    ("Nele", "2A2fFf8SkDEJ8nzx", "de", "de-de", 'female', 'Middle-age'),
    ("Jette", "4Mn9VfG2wsLLEzi5", "de", "de-de", 'female', 'Young'),
    ("Annika", "p6Uutkyi3j2iNAUu", "de", "de-de", 'female', 'Adult'),
    ("Jule", "MxhgvuHuUER7qjmL", "de", "de-de", 'female', 'Young Adult'),
    ("Lotta", "MOrDPn052KUlnKXk", "de", "de-de", 'female', 'Young Adult'),
    ("Josefine", "yadpmqsHVrYC0NJp", "de", "de-de", 'female', 'Middle-age'),
    ("Anton", "3ZKKapPOvuWFcw9f", "de", "de-de", 'male', 'Adult'),
    ("Mirko", "Xn9xDzqSSr4xGTNg", "de", "de-de", 'male', 'Young Adult'),
    ("Mats", "Kf5m22mROozoMWj3", "de", "de-de", 'male', 'Adult'),
    ("Henrik", "yyS1KYWs6mXoEw7D", "de", "de-de", 'male', 'Adult'),
    ("Leon", "20zdyYrQPzKlCwkk", "de", "de-de", 'male', 'Adult'),
    ("Mika", "R3uU1GTAZJotUhUs", "de", "de-de", 'male', 'Young Adult'),
    ("Noémie", "FXxJ9mANRq6BCTX5", "fr", "fr-fr", 'female', 'Young'),
    ("Coralie", "ZeSg853xFACESHHI", "fr", "fr-fr", 'female', 'Young'),
    ("Tiphaine", "0fvH6XAKvA748v2v", "fr", "fr-fr", 'female', 'Middle-age'),
    ("Maëlys", "s048cR1l2Jmu4k3B", "fr", "fr-fr", 'female', 'Young'),
    ("Orianne", "TUk4X11S6UytB3G6", "fr", "fr-fr", 'female', 'Middle-age'),
    ("Solène", "YhIHaAfQ0cQPDV9R", "fr", "fr-fr", 'female', 'Young'),
    ("Augustin", "Tek4tJXiX6_yvXq7", "fr", "fr-fr", 'male', 'Adult'),
    ("Arnaud", "aXMVwiUhITMrwPzd", "fr", "fr-fr", 'male', 'Middle-age'),
    ("Jules", "YKeBw3OV1RgpdhLh", "fr", "fr-fr", 'male', 'Adult'),
    ("Baptiste", "rqqeErpASHVQXHDd", "fr", "fr-fr", 'male', 'Middle-age'),
    ("Lucien", "LIv4NaG8ay4Mhqtd", "fr", "fr-fr", 'male', 'Middle-age'),
    ("Marius", "biuhvu17TxVKOcyy", "fr", "fr-fr", 'male', 'Adult'),
    ("Manuela-Lu", "KgC2Nqnjj48NUiyV", "pt", "pt-br", 'female', 'Young'),
    ("Isadora", "CtNvKUUMrx1h56ih", "pt", "pt-br", 'female', 'Young'),
    ("Débora", "K8mj96kYHMy8q3Hb", "pt", "pt-br", 'female', 'Middle-age'),
    ("Rafaela", "6k6cRt7QJfm5ChrH", "pt", "pt-br", 'female', 'Young'),
    ("Clarice", "sGCMzFIA53ppIngM", "pt", "pt-br", 'female', 'Older Adult'),
    ("Flávia", "gOc90Rjmv5CobHCj", "pt", "pt-br", 'female', 'Middle-age'),
    ("Davi", "NuUr_x5V90hSHzCJ", "pt", "pt-br", 'male', 'Adult'),
    ("Igor", "l7520SIXi14iEamd", "pt", "pt-br", 'male', 'Middle-age'),
    ("Kauã", "6HftnVU1XqVBJS8t", "pt", "pt-br", 'male', 'Young Adult'),
    ("Vitor", "mWgyMRkFED70QxQv", "pt", "pt-br", 'male', 'Middle-age'),
    ("Otávio", "JYtmeaysoaR0Wybf", "pt", "pt-br", 'male', 'Older Adult'),
    ("Marcelo", "qDeAnE4L4bk0BRz1", "pt", "pt-br", 'male', 'Middle-age'),
    ("Bryony", "gfP44RlHwr2kEwk7", "en", "en-gb", 'female', 'Middle-age'),
    ("Ninon", "qtPlUYAZ4EJDuWMm", "fr", "fr-fr", 'female', 'Middle-age'),
    ("Nerea", "EQcQFGkYfMn5YJgx", "es", "es-es", 'female', 'Middle-age'),
    ("Svea", "Adoh4wA9tp5QXJi9", "de", "de-de", 'female', 'Young Adult'),
    ("Lilou", "4iKxBtHBRzr1YOph", "fr", "fr-fr", 'female', 'Young'),
    ("Hélène", "Vvakv7iWZCUrBooR", "fr", "fr-fr", 'female', 'Older Adult'),
    ("Captain Silas Blackwater", "xPDpsV3F592srkjA", "en", "characters", None, None),
    ("Morwenna of the Deep", "nOykTmcH9ydsGGIc", "en", "characters", None, None),
    ("Old Mortlake", "lX3LavMNdlIbgEHH", "en", "characters", None, None),
    ("Azgorath", "dGyGDKo2osj7h3s7", "en", "characters", None, None),
    ("Fennick the Young Elf", "cXoIXJ2llS8S0WOW", "en", "characters", None, None),
    ("The Thing Under the Stairs", "GXaSqMEmOg12OIYb", "en", "characters", None, None),
    ("Vyrmathrax the Ancient", "xO8xQo3BrB8FuI6J", "en", "characters", None, None),
    ("Grub the Ogre", "cYtsRSxV5q9IMehn", "en", "characters", None, None),
    ("Master Aldric", "gSgG5Tor0GPEnqzC", "en", "characters", None, None),
    ("The Snow Queen", "NIWoEurJHMJjyUTe", "en", "characters", None, None),
    ("Sir Bramwell", "y2eiKxtQo2B67deU", "en", "characters", None, None),
    ("Fairy Godmother", "QXNr0SvkukLB4wPq", "en", "characters", None, None),
)

PHONON_VOICE_NAMES: tuple[str, ...] = tuple(v[0] for v in PHONON_VOICES)

# Locale -> the ``lang`` ptts needs for text normalization.
LOCALE_LANG: dict[str, str] = {
    "en-us": "en", "en-gb": "en", "characters": "en", "fr-fr": "fr", "de-de": "de",
    "es-es": "es", "es-mx": "es", "pt-br": "pt",
}


def _fold(name: str) -> str:
    """Case- and accent-insensitive key, matching how ptts resolves a voice name.

    ``unicodedata`` decomposition then drop the combining marks: "Noémie" -> "noemie", which is
    also how a user is likely to type it on a keyboard without accents.
    """
    import unicodedata
    decomposed = unicodedata.normalize("NFKD", str(name).strip().casefold())
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def phonon_voice(name: str) -> tuple | None:
    """The catalogue row for *name* (case- and accent-insensitive) or a ``voice_id``."""
    if not name:
        return None
    key = _fold(name)
    for row in PHONON_VOICES:
        if _fold(row[0]) == key or row[1].lower() == key:
            return row
    return None


def phonon_lang_for_voice(name: str) -> str | None:
    """ptts ``lang`` for *name*, or None when the voice is not in the catalogue."""
    row = phonon_voice(name)
    return LOCALE_LANG.get(row[3]) if row else None
