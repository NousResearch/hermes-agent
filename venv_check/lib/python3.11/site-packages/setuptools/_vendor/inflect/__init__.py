"""
inflect: english language inflection
 - correctly generate plurals, ordinals, indefinite articles
 - convert numbers to words

Copyright (C) 2010 Paul Dyson

Based upon the Perl module
`Lingua::EN::Inflect <https://metacpan.org/pod/Lingua::EN::Inflect>`_.

methods:
    classical inflect
    plural plural_noun plural_verb plural_adj singular_noun no num a an
    compare compare_nouns compare_verbs compare_adjs
    present_participle
    ordinal
    number_to_words
    join
    defnoun defverb defadj defa defan

INFLECTIONS:
    classical inflect
    plural plural_noun plural_verb plural_adj singular_noun compare
    no num a an present_participle

PLURALS:
    classical inflect
    plural plural_noun plural_verb plural_adj singular_noun no num
    compare compare_nouns compare_verbs compare_adjs

COMPARISONS:
    classical
    compare compare_nouns compare_verbs compare_adjs

ARTICLES:
    classical inflect num a an

NUMERICAL:
    ordinal number_to_words

USER_DEFINED:
    defnoun defverb defadj defa defan

Exceptions:
 UnknownClassicalModeError
 BadNumValueError
 BadChunkingOptionError
 NumOutOfRangeError
 BadUserDefinedPatternError
 BadRcFileError
 BadGenderError

"""

from __future__ import annotations

import ast
import collections
import contextlib
import functools
import itertools
import re
from numbers import Number
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    Match,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
)

from more_itertools import windowed_complete
from typeguard import typechecked

from .compat.py38 import Annotated


class UnknownClassicalModeError(Exception):
    pass


class BadNumValueError(Exception):
    pass


class BadChunkingOptionError(Exception):
    pass


class NumOutOfRangeError(Exception):
    pass


class BadUserDefinedPatternError(Exception):
    pass


class BadRcFileError(Exception):
    pass


class BadGenderError(Exception):
    pass


def enclose(s: str) -> str:
    return f"(?:{s})"


def joinstem(cutpoint: Optional[int] = 0, words: Optional[Iterable[str]] = None) -> str:
    """
    Join stem of each word in words into a string for regex.

    Each word is truncated at cutpoint.

    Cutpoint is usually negative indicating the number of letters to remove
    from the end of each word.

    >>> joinstem(-2, ["ephemeris", "iris", ".*itis"])
    '(?:ephemer|ir|.*it)'

    >>> joinstem(None, ["ephemeris"])
    '(?:ephemeris)'

    >>> joinstem(5, None)
    '(?:)'
    """
    return enclose("|".join(w[:cutpoint] for w in words or []))


def bysize(words: Iterable[str]) -> Dict[int, set]:
    """
    From a list of words, return a dict of sets sorted by word length.

    >>> words = ['ant', 'cat', 'dog', 'pig', 'frog', 'goat', 'horse', 'elephant']
    >>> ret = bysize(words)
    >>> sorted(ret[3])
    ['ant', 'cat', 'dog', 'pig']
    >>> ret[5]
    {'horse'}
    """
    res: Dict[int, set] = collections.defaultdict(set)
    for w in words:
        res[len(w)].add(w)
    return res


def make_pl_si_lists(
    lst: Iterable[str],
    plending: str,
    siendingsize: Optional[int],
    dojoinstem: bool = True,
):
    """
    given a list of singular words: lst

    an ending to append to make the plural: plending

    the number of characters to remove from the singular
    before appending plending: siendingsize

    a flag whether to create a joinstem: dojoinstem

    return:
    a list of pluralised words: si_list (called si because this is what you need to
    look for to make the singular)

    the pluralised words as a dict of sets sorted by word length: si_bysize
    the singular words as a dict of sets sorted by word length: pl_bysize
    if dojoinstem is True: a regular expression that matches any of the stems: stem
    """
    if siendingsize is not None:
        siendingsize = -siendingsize
    si_list = [w[:siendingsize] + plending for w in lst]
    pl_bysize = bysize(lst)
    si_bysize = bysize(si_list)
    if dojoinstem:
        stem = joinstem(siendingsize, lst)
        return si_list, si_bysize, pl_bysize, stem
    else:
        return si_list, si_bysize, pl_bysize


# 1. PLURALS

pl_sb_irregular_s = {
    "corpus": "corpuses|corpora",
    "opus": "opuses|opera",
    "genus": "genera",
    "mythos": "mythoi",
    "penis": "penises|penes",
    "testis": "testes",
    "atlas": "atlases|atlantes",
    "yes": "yeses",
}

pl_sb_irregular = {
    "child": "children",
    "chili": "chilis|chilies",
    "brother": "brothers|brethren",
    "infinity": "infinities|infinity",
    "loaf": "loaves",
    "lore": "lores|lore",
    "hoof": "hoofs|hooves",
    "beef": "beefs|beeves",
    "thief": "thiefs|thieves",
    "money": "monies",
    "mongoose": "mongooses",
    "ox": "oxen",
    "cow": "cows|kine",
    "graffito": "graffiti",
    "octopus": "octopuses|octopodes",
    "genie": "genies|genii",
    "ganglion": "ganglions|ganglia",
    "trilby": "trilbys",
    "turf": "turfs|turves",
    "numen": "numina",
    "atman": "atmas",
    "occiput": "occiputs|occipita",
    "sabretooth": "sabretooths",
    "sabertooth": "sabertooths",
    "lowlife": "lowlifes",
    "flatfoot": "flatfoots",
    "tenderfoot": "tenderfoots",
    "romany": "romanies",
    "jerry": "jerries",
    "mary": "maries",
    "talouse": "talouses",
    "rom": "roma",
    "carmen": "carmina",
}

pl_sb_irregular.update(pl_sb_irregular_s)
# pl_sb_irregular_keys = enclose('|'.join(pl_sb_irregular.keys()))

pl_sb_irregular_caps = {
    "Romany": "Romanies",
    "Jerry": "Jerrys",
    "Mary": "Marys",
    "Rom": "Roma",
}

pl_sb_irregular_compound = {"prima donna": "prima donnas|prime donne"}

si_sb_irregular = {v: k for (k, v) in pl_sb_irregular.items()}
for k in list(si_sb_irregular):
    if "|" in k:
        k1, k2 = k.split("|")
        si_sb_irregular[k1] = si_sb_irregular[k2] = si_sb_irregular[k]
        del si_sb_irregular[k]
si_sb_irregular_caps = {v: k for (k, v) in pl_sb_irregular_caps.items()}
si_sb_irregular_compound = {v: k for (k, v) in pl_sb_irregular_compound.items()}
for k in list(si_sb_irregular_compound):
    if "|" in k:
        k1, k2 = k.split("|")
        si_sb_irregular_compound[k1] = si_sb_irregular_compound[k2] = (
            si_sb_irregular_compound[k]
        )
        del si_sb_irregular_compound[k]

# si_sb_irregular_keys = enclose('|'.join(si_sb_irregular.keys()))

# Z's that don't double

pl_sb_z_zes_list = ("quartz", "topaz")
pl_sb_z_zes_bysize = bysize(pl_sb_z_zes_list)

pl_sb_ze_zes_list = ("snooze",)
pl_sb_ze_zes_bysize = bysize(pl_sb_ze_zes_list)


# CLASSICAL "..is" -> "..ides"

pl_sb_C_is_ides_complete = [
    # GENERAL WORDS...
    "ephemeris",
    "iris",
    "clitoris",
    "chrysalis",
    "epididymis",
]

pl_sb_C_is_ides_endings = [
    # INFLAMATIONS...
    "itis"
]

pl_sb_C_is_ides = joinstem(
    -2, pl_sb_C_is_ides_complete + [f".*{w}" for w in pl_sb_C_is_ides_endings]
)

pl_sb_C_is_ides_list = pl_sb_C_is_ides_complete + pl_sb_C_is_ides_endings

(
    si_sb_C_is_ides_list,
    si_sb_C_is_ides_bysize,
    pl_sb_C_is_ides_bysize,
) = make_pl_si_lists(pl_sb_C_is_ides_list, "ides", 2, dojoinstem=False)


# CLASSICAL "..a" -> "..ata"

pl_sb_C_a_ata_list = (
    "anathema",
    "bema",
    "carcinoma",
    "charisma",
    "diploma",
    "dogma",
    "drama",
    "edema",
    "enema",
    "enigma",
    "lemma",
    "lymphoma",
    "magma",
    "melisma",
    "miasma",
    "oedema",
    "sarcoma",
    "schema",
    "soma",
    "stigma",
    "stoma",
    "trauma",
    "gumma",
    "pragma",
)

(
    si_sb_C_a_ata_list,
    si_sb_C_a_ata_bysize,
    pl_sb_C_a_ata_bysize,
    pl_sb_C_a_ata,
) = make_pl_si_lists(pl_sb_C_a_ata_list, "ata", 1)

# UNCONDITIONAL "..a" -> "..ae"

pl_sb_U_a_ae_list = (
    "alumna",
    "alga",
    "vertebra",
    "persona",
    "vita",
)
(
    si_sb_U_a_ae_list,
    si_sb_U_a_ae_bysize,
    pl_sb_U_a_ae_bysize,
    pl_sb_U_a_ae,
) = make_pl_si_lists(pl_sb_U_a_ae_list, "e", None)

# CLASSICAL "..a" -> "..ae"

pl_sb_C_a_ae_list = (
    "amoeba",
    "antenna",
    "formula",
    "hyperbola",
    "medusa",
    "nebula",
    "parabola",
    "abscissa",
    "hydra",
    "nova",
    "lacuna",
    "aurora",
    "umbra",
    "flora",
    "fauna",
)
(
    si_sb_C_a_ae_list,
    si_sb_C_a_ae_bysize,
    pl_sb_C_a_ae_bysize,
    pl_sb_C_a_ae,
) = make_pl_si_lists(pl_sb_C_a_ae_list, "e", None)


# CLASSICAL "..en" -> "..ina"

pl_sb_C_en_ina_list = ("stamen", "foramen", "lumen")

(
    si_sb_C_en_ina_list,
    si_sb_C_en_ina_bysize,
    pl_sb_C_en_ina_bysize,
    pl_sb_C_en_ina,
) = make_pl_si_lists(pl_sb_C_en_ina_list, "ina", 2)


# UNCONDITIONAL "..um" -> "..a"

pl_sb_U_um_a_list = (
    "bacterium",
    "agendum",
    "desideratum",
    "erratum",
    "stratum",
    "datum",
    "ovum",
    "extremum",
    "candelabrum",
)
(
    si_sb_U_um_a_list,
    si_sb_U_um_a_bysize,
    pl_sb_U_um_a_bysize,
    pl_sb_U_um_a,
) = make_pl_si_lists(pl_sb_U_um_a_list, "a", 2)

# CLASSICAL "..um" -> "..a"

pl_sb_C_um_a_list = (
    "maximum",
    "minimum",
    "momentum",
    "optimum",
    "quantum",
    "cranium",
    "curriculum",
    "dictum",
    "phylum",
    "aquarium",
    "compendium",
    "emporium",
    "encomium",
    "gymnasium",
    "honorarium",
    "interregnum",
    "lustrum",
    "memorandum",
    "millennium",
    "rostrum",
    "spectrum",
    "speculum",
    "stadium",
    "trapezium",
    "ultimatum",
    "medium",
    "vacuum",
    "velum",
    "consortium",
    "arboretum",
)

(
    si_sb_C_um_a_list,
    si_sb_C_um_a_bysize,
    pl_sb_C_um_a_bysize,
    pl_sb_C_um_a,
) = make_pl_si_lists(pl_sb_C_um_a_list, "a", 2)


# UNCONDITIONAL "..us" -> "i"

pl_sb_U_us_i_list = (
    "alumnus",
    "alveolus",
    "bacillus",
    "bronchus",
    "locus",
    "nucleus",
    "stimulus",
    "meniscus",
    "sarcophagus",
)
(
    si_sb_U_us_i_list,
    si_sb_U_us_i_bysize,
    pl_sb_U_us_i_bysize,
    pl_sb_U_us_i,
) = make_pl_si_lists(pl_sb_U_us_i_list, "i", 2)

# CLASSICAL "..us" -> "..i"

pl_sb_C_us_i_list = (
    "focus",
    "radius",
    "genius",
    "incubus",
    "succubus",
    "nimbus",
    "fungus",
    "nucleolus",
    "stylus",
    "torus",
    "umbilicus",
    "uterus",
    "hippopotamus",
    "cactus",
)

(
    si_sb_C_us_i_list,
    si_sb_C_us_i_bysize,
    pl_sb_C_us_i_bysize,
    pl_sb_C_us_i,
) = make_pl_si_lists(pl_sb_C_us_i_list, "i", 2)


# CLASSICAL "..us" -> "..us"  (ASSIMILATED 4TH DECLENSION LATIN NOUNS)

pl_sb_C_us_us = (
    "status",
    "apparatus",
    "prospectus",
    "sinus",
    "hiatus",
    "impetus",
    "plexus",
)
pl_sb_C_us_us_bysize = bysize(pl_sb_C_us_us)

# UNCONDITIONAL "..on" -> "a"

pl_sb_U_on_a_list = (
    "criterion",
    "perihelion",
    "aphelion",
    "phenomenon",
    "prolegomenon",
    "noumenon",
    "organon",
    "asyndeton",
    "hyperbaton",
)
(
    si_sb_U_on_a_list,
    si_sb_U_on_a_bysize,
    pl_sb_U_on_a_bysize,
    pl_sb_U_on_a,
) = make_pl_si_lists(pl_sb_U_on_a_list, "a", 2)

# CLASSICAL "..on" -> "..a"

pl_sb_C_on_a_list = ("oxymoron",)

(
    si_sb_C_on_a_list,
    si_sb_C_on_a_bysize,
    pl_sb_C_on_a_bysize,
    pl_sb_C_on_a,
) = make_pl_si_lists(pl_sb_C_on_a_list, "a", 2)


# CLASSICAL "..o" -> "..i"  (BUT NORMALLY -> "..os")

pl_sb_C_o_i = [
    "solo",
    "soprano",
    "basso",
    "alto",
    "contralto",
    "tempo",
    "piano",
    "virtuoso",
]  # list not tuple so can concat for pl_sb_U_o_os

pl_sb_C_o_i_bysize = bysize(pl_sb_C_o_i)
si_sb_C_o_i_bysize = bysize([f"{w[:-1]}i" for w in pl_sb_C_o_i])

pl_sb_C_o_i_stems = joinstem(-1, pl_sb_C_o_i)

# ALWAYS "..o" -> "..os"

pl_sb_U_o_os_complete = {"ado", "ISO", "NATO", "NCO", "NGO", "oto"}
si_sb_U_o_os_complete = {f"{w}s" for w in pl_sb_U_o_os_complete}


pl_sb_U_o_os_endings = [
    "aficionado",
    "aggro",
    "albino",
    "allegro",
    "ammo",
    "Antananarivo",
    "archipelago",
    "armadillo",
    "auto",
    "avocado",
    "Bamako",
    "Barquisimeto",
    "bimbo",
    "bingo",
    "Biro",
    "bolero",
    "Bolzano",
    "bongo",
    "Boto",
    "burro",
    "Cairo",
    "canto",
    "cappuccino",
    "casino",
    "cello",
    "Chicago",
    "Chimango",
    "cilantro",
    "cochito",
    "coco",
    "Colombo",
    "Colorado",
    "commando",
    "concertino",
    "contango",
    "credo",
    "crescendo",
    "cyano",
    "demo",
    "ditto",
    "Draco",
    "dynamo",
    "embryo",
    "Esperanto",
    "espresso",
    "euro",
    "falsetto",
    "Faro",
    "fiasco",
    "Filipino",
    "flamenco",
    "furioso",
    "generalissimo",
    "Gestapo",
    "ghetto",
    "gigolo",
    "gizmo",
    "Greensboro",
    "gringo",
    "Guaiabero",
    "guano",
    "gumbo",
    "gyro",
    "hairdo",
    "hippo",
    "Idaho",
    "impetigo",
    "inferno",
    "info",
    "intermezzo",
    "intertrigo",
    "Iquico",
    "jumbo",
    "junto",
    "Kakapo",
    "kilo",
    "Kinkimavo",
    "Kokako",
    "Kosovo",
    "Lesotho",
    "libero",
    "libido",
    "libretto",
    "lido",
    "Lilo",
    "limbo",
    "limo",
    "lineno",
    "lingo",
    "lino",
    "livedo",
    "loco",
    "logo",
    "lumbago",
    "macho",
    "macro",
    "mafioso",
    "magneto",
    "magnifico",
    "Majuro",
    "Malabo",
    "manifesto",
    "Maputo",
    "Maracaibo",
    "medico",
    "memo",
    "metro",
    "Mexico",
    "micro",
    "Milano",
    "Monaco",
    "mono",
    "Montenegro",
    "Morocco",
    "Muqdisho",
    "myo",
    "neutrino",
    "Ningbo",
    "octavo",
    "oregano",
    "Orinoco",
    "Orlando",
    "Oslo",
    "panto",
    "Paramaribo",
    "Pardusco",
    "pedalo",
    "photo",
    "pimento",
    "pinto",
    "pleco",
    "Pluto",
    "pogo",
    "polo",
    "poncho",
    "Porto-Novo",
    "Porto",
    "pro",
    "psycho",
    "pueblo",
    "quarto",
    "Quito",
    "repo",
    "rhino",
    "risotto",
    "rococo",
    "rondo",
    "Sacramento",
    "saddo",
    "sago",
    "salvo",
    "Santiago",
    "Sapporo",
    "Sarajevo",
    "scherzando",
    "scherzo",
    "silo",
    "sirocco",
    "sombrero",
    "staccato",
    "sterno",
    "stucco",
    "stylo",
    "sumo",
    "Taiko",
    "techno",
    "terrazzo",
    "testudo",
    "timpano",
    "tiro",
    "tobacco",
    "Togo",
    "Tokyo",
    "torero",
    "Torino",
    "Toronto",
    "torso",
    "tremolo",
    "typo",
    "tyro",
    "ufo",
    "UNESCO",
    "vaquero",
    "vermicello",
    "verso",
    "vibrato",
    "violoncello",
    "Virgo",
    "weirdo",
    "WHO",
    "WTO",
    "Yamoussoukro",
    "yo-yo",
    "zero",
    "Zibo",
] + pl_sb_C_o_i

pl_sb_U_o_os_bysize = bysize(pl_sb_U_o_os_endings)
si_sb_U_o_os_bysize = bysize([f"{w}s" for w in pl_sb_U_o_os_endings])


# UNCONDITIONAL "..ch" -> "..chs"

pl_sb_U_ch_chs_list = ("czech", "eunuch", "stomach")

(
    si_sb_U_ch_chs_list,
    si_sb_U_ch_chs_bysize,
    pl_sb_U_ch_chs_bysize,
    pl_sb_U_ch_chs,
) = make_pl_si_lists(pl_sb_U_ch_chs_list, "s", None)


# UNCONDITIONAL "..[ei]x" -> "..ices"

pl_sb_U_ex_ices_list = ("codex", "murex", "silex")
(
    si_sb_U_ex_ices_list,
    si_sb_U_ex_ices_bysize,
    pl_sb_U_ex_ices_bysize,
    pl_sb_U_ex_ices,
) = make_pl_si_lists(pl_sb_U_ex_ices_list, "ices", 2)

pl_sb_U_ix_ices_list = ("radix", "helix")
(
    si_sb_U_ix_ices_list,
    si_sb_U_ix_ices_bysize,
    pl_sb_U_ix_ices_bysize,
    pl_sb_U_ix_ices,
) = make_pl_si_lists(pl_sb_U_ix_ices_list, "ices", 2)

# CLASSICAL "..[ei]x" -> "..ices"

pl_sb_C_ex_ices_list = (
    "vortex",
    "vertex",
    "cortex",
    "latex",
    "pontifex",
    "apex",
    "index",
    "simplex",
)

(
    si_sb_C_ex_ices_list,
    si_sb_C_ex_ices_bysize,
    pl_sb_C_ex_ices_bysize,
    pl_sb_C_ex_ices,
) = make_pl_si_lists(pl_sb_C_ex_ices_list, "ices", 2)


pl_sb_C_ix_ices_list = ("appendix",)

(
    si_sb_C_ix_ices_list,
    si_sb_C_ix_ices_bysize,
    pl_sb_C_ix_ices_bysize,
    pl_sb_C_ix_ices,
) = make_pl_si_lists(pl_sb_C_ix_ices_list, "ices", 2)


# ARABIC: ".." -> "..i"

pl_sb_C_i_list = ("afrit", "afreet", "efreet")

(si_sb_C_i_list, si_sb_C_i_bysize, pl_sb_C_i_bysize, pl_sb_C_i) = make_pl_si_lists(
    pl_sb_C_i_list, "i", None
)


# HEBREW: ".." -> "..im"

pl_sb_C_im_list = ("goy", "seraph", "cherub")

(si_sb_C_im_list, si_sb_C_im_bysize, pl_sb_C_im_bysize, pl_sb_C_im) = make_pl_si_lists(
    pl_sb_C_im_list, "im", None
)


# UNCONDITIONAL "..man" -> "..mans"

pl_sb_U_man_mans_list = """
    ataman caiman cayman ceriman
    desman dolman farman harman hetman
    human leman ottoman shaman talisman
""".split()
pl_sb_U_man_mans_caps_list = """
    Alabaman Bahaman Burman German
    Hiroshiman Liman Nakayaman Norman Oklahoman
    Panaman Roman Selman Sonaman Tacoman Yakiman
    Yokohaman Yuman
""".split()

(
    si_sb_U_man_mans_list,
    si_sb_U_man_mans_bysize,
    pl_sb_U_man_mans_bysize,
) = make_pl_si_lists(pl_sb_U_man_mans_list, "s", None, dojoinstem=False)
(
    si_sb_U_man_mans_caps_list,
    si_sb_U_man_mans_caps_bysize,
    pl_sb_U_man_mans_caps_bysize,
) = make_pl_si_lists(pl_sb_U_man_mans_caps_list, "s", None, dojoinstem=False)

# UNCONDITIONAL "..louse" -> "..lice"
pl_sb_U_louse_lice_list = ("booklouse", "grapelouse", "louse", "woodlouse")

(
    si_sb_U_louse_lice_list,
    si_sb_U_louse_lice_bysize,
    pl_sb_U_louse_lice_bysize,
) = make_pl_si_lists(pl_sb_U_louse_lice_list, "lice", 5, dojoinstem=False)

pl_sb_uninflected_s_complete = [
    # PAIRS OR GROUPS SUBSUMED TO A SINGULAR...
    "breeches",
    "britches",
    "pajamas",
    "pyjamas",
    "clippers",
    "gallows",
    "hijinks",
    "headquarters",
    "pliers",
    "scissors",
    "testes",
    "herpes",
    "pincers",
    "shears",
    "proceedings",
    "trousers",
    # UNASSIMILATED LATIN 4th DECLENSION
    "cantus",
    "coitus",
    "nexus",
    # RECENT IMPORTS...
    "contretemps",
    "corps",
    "debris",
    "siemens",
    # DISEASES
    "mumps",
    # MISCELLANEOUS OTHERS...
    "diabetes",
    "jackanapes",
    "series",
    "species",
    "subspecies",
    "rabies",
    "chassis",
    "innings",
    "news",
    "mews",
    "haggis",
]

pl_sb_uninflected_s_endings = [
    # RECENT IMPORTS...
    "ois",
    # DISEASES
    "measles",
]

pl_sb_uninflected_s = pl_sb_uninflected_s_complete + [
    f".*{w}" for w in pl_sb_uninflected_s_endings
]

pl_sb_uninflected_herd = (
    # DON'T INFLECT IN CLASSICAL MODE, OTHERWISE NORMAL INFLECTION
    "wildebeest",
    "swine",
    "eland",
    "bison",
    "buffalo",
    "cattle",
    "elk",
    "rhinoceros",
    "zucchini",
    "caribou",
    "dace",
    "grouse",
    "guinea fowl",
    "guinea-fowl",
    "haddock",
    "hake",
    "halibut",
    "herring",
    "mackerel",
    "pickerel",
    "pike",
    "roe",
    "seed",
    "shad",
    "snipe",
    "teal",
    "turbot",
    "water fowl",
    "water-fowl",
)

pl_sb_uninflected_complete = [
    # SOME FISH AND HERD ANIMALS
    "tuna",
    "salmon",
    "mackerel",
    "trout",
    "bream",
    "sea-bass",
    "sea bass",
    "carp",
    "cod",
    "flounder",
    "whiting",
    "moose",
    # OTHER ODDITIES
    "graffiti",
    "djinn",
    "samuri",
    "offspring",
    "pence",
    "quid",
    "hertz",
] + pl_sb_uninflected_s_complete
# SOME WORDS ENDING IN ...s (OFTEN PAIRS TAKEN AS A WHOLE)

pl_sb_uninflected_caps = [
    # ALL NATIONALS ENDING IN -ese
    "Portuguese",
    "Amoyese",
    "Borghese",
    "Congoese",
    "Faroese",
    "Foochowese",
    "Genevese",
    "Genoese",
    "Gilbertese",
    "Hottentotese",
    "Kiplingese",
    "Kongoese",
    "Lucchese",
    "Maltese",
    "Nankingese",
    "Niasese",
    "Pekingese",
    "Piedmontese",
    "Pistoiese",
    "Sarawakese",
    "Shavese",
    "Vermontese",
    "Wenchowese",
    "Yengeese",
]


pl_sb_uninflected_endings = [
    # UNCOUNTABLE NOUNS
    "butter",
    "cash",
    "furniture",
    "information",
    # SOME FISH AND HERD ANIMALS
    "fish",
    "deer",
    "sheep",
    # ALL NATIONALS ENDING IN -ese
    "nese",
    "rese",
    "lese",
    "mese",
    # DISEASES
    "pox",
    # OTHER ODDITIES
    "craft",
] + pl_sb_uninflected_s_endings
# SOME WORDS ENDING IN ...s (OFTEN PAIRS TAKEN AS A WHOLE)


pl_sb_uninflected_bysize = bysize(pl_sb_uninflected_endings)


# SINGULAR WORDS ENDING IN ...s (ALL INFLECT WITH ...es)

pl_sb_singular_s_complete = [
    "acropolis",
    "aegis",
    "alias",
    "asbestos",
    "bathos",
    "bias",
    "bronchitis",
    "bursitis",
    "caddis",
    "cannabis",
    "canvas",
    "chaos",
    "cosmos",
    "dais",
    "digitalis",
    "epidermis",
    "ethos",
    "eyas",
    "gas",
    "glottis",
    "hubris",
    "ibis",
    "lens",
    "mantis",
    "marquis",
    "metropolis",
    "pathos",
    "pelvis",
    "polis",
    "rhinoceros",
    "sassafras",
    "trellis",
] + pl_sb_C_is_ides_complete


pl_sb_singular_s_endings = ["ss", "us"] + pl_sb_C_is_ides_endings

pl_sb_singular_s_bysize = bysize(pl_sb_singular_s_endings)

si_sb_singular_s_complete = [f"{w}es" for w in pl_sb_singular_s_complete]
si_sb_singular_s_endings = [f"{w}es" for w in pl_sb_singular_s_endings]
si_sb_singular_s_bysize = bysize(si_sb_singular_s_endings)

pl_sb_singular_s_es = ["[A-Z].*es"]

pl_sb_singular_s = enclose(
    "|".join(
        pl_sb_singular_s_complete
        + [f".*{w}" for w in pl_sb_singular_s_endings]
        + pl_sb_singular_s_es
    )
)


# PLURALS ENDING IN uses -> use


si_sb_ois_oi_case = ("Bolshois", "Hanois")

si_sb_uses_use_case = ("Betelgeuses", "Duses", "Meuses", "Syracuses", "Toulouses")

si_sb_uses_use = (
    "abuses",
    "applauses",
    "blouses",
    "carouses",
    "causes",
    "chartreuses",
    "clauses",
    "contuses",
    "douses",
    "excuses",
    "fuses",
    "grouses",
    "hypotenuses",
    "masseuses",
    "menopauses",
    "misuses",
    "muses",
    "overuses",
    "pauses",
    "peruses",
    "profuses",
    "recluses",
    "reuses",
    "ruses",
    "souses",
    "spouses",
    "suffuses",
    "transfuses",
    "uses",
)

si_sb_ies_ie_case = (
    "Addies",
    "Aggies",
    "Allies",
    "Amies",
    "Angies",
    "Annies",
    "Annmaries",
    "Archies",
    "Arties",
    "Aussies",
    "Barbies",
    "Barries",
    "Basies",
    "Bennies",
    "Bernies",
    "Berties",
    "Bessies",
    "Betties",
    "Billies",
    "Blondies",
    "Bobbies",
    "Bonnies",
    "Bowies",
    "Brandies",
    "Bries",
    "Brownies",
    "Callies",
    "Carnegies",
    "Carries",
    "Cassies",
    "Charlies",
    "Cheries",
    "Christies",
    "Connies",
    "Curies",
    "Dannies",
    "Debbies",
    "Dixies",
    "Dollies",
    "Donnies",
    "Drambuies",
    "Eddies",
    "Effies",
    "Ellies",
    "Elsies",
    "Eries",
    "Ernies",
    "Essies",
    "Eugenies",
    "Fannies",
    "Flossies",
    "Frankies",
    "Freddies",
    "Gillespies",
    "Goldies",
    "Gracies",
    "Guthries",
    "Hallies",
    "Hatties",
    "Hetties",
    "Hollies",
    "Jackies",
    "Jamies",
    "Janies",
    "Jannies",
    "Jeanies",
    "Jeannies",
    "Jennies",
    "Jessies",
    "Jimmies",
    "Jodies",
    "Johnies",
    "Johnnies",
    "Josies",
    "Julies",
    "Kalgoorlies",
    "Kathies",
    "Katies",
    "Kellies",
    "Kewpies",
    "Kristies",
    "Laramies",
    "Lassies",
    "Lauries",
    "Leslies",
    "Lessies",
    "Lillies",
    "Lizzies",
    "Lonnies",
    "Lories",
    "Lorries",
    "Lotties",
    "Louies",
    "Mackenzies",
    "Maggies",
    "Maisies",
    "Mamies",
    "Marcies",
    "Margies",
    "Maries",
    "Marjories",
    "Matties",
    "McKenzies",
    "Melanies",
    "Mickies",
    "Millies",
    "Minnies",
    "Mollies",
    "Mounties",
    "Nannies",
    "Natalies",
    "Nellies",
    "Netties",
    "Ollies",
    "Ozzies",
    "Pearlies",
    "Pottawatomies",
    "Reggies",
    "Richies",
    "Rickies",
    "Robbies",
    "Ronnies",
    "Rosalies",
    "Rosemaries",
    "Rosies",
    "Roxies",
    "Rushdies",
    "Ruthies",
    "Sadies",
    "Sallies",
    "Sammies",
    "Scotties",
    "Selassies",
    "Sherries",
    "Sophies",
    "Stacies",
    "Stefanies",
    "Stephanies",
    "Stevies",
    "Susies",
    "Sylvies",
    "Tammies",
    "Terries",
    "Tessies",
    "Tommies",
    "Tracies",
    "Trekkies",
    "Valaries",
    "Valeries",
    "Valkyries",
    "Vickies",
    "Virgies",
    "Willies",
    "Winnies",
    "Wylies",
    "Yorkies",
)

si_sb_ies_ie = (
    "aeries",
    "baggies",
    "belies",
    "biggies",
    "birdies",
    "bogies",
    "bonnies",
    "boogies",
    "bookies",
    "bourgeoisies",
    "brownies",
    "budgies",
    "caddies",
    "calories",
    "camaraderies",
    "cockamamies",
    "collies",
    "cookies",
    "coolies",
    "cooties",
    "coteries",
    "crappies",
    "curies",
    "cutesies",
    "dogies",
    "eyries",
    "floozies",
    "footsies",
    "freebies",
    "genies",
    "goalies",
    "groupies",
    "hies",
    "jalousies",
    "junkies",
    "kiddies",
    "laddies",
    "lassies",
    "lies",
    "lingeries",
    "magpies",
    "menageries",
    "mommies",
    "movies",
    "neckties",
    "newbies",
    "nighties",
    "oldies",
    "organdies",
    "overlies",
    "pies",
    "pinkies",
    "pixies",
    "potpies",
    "prairies",
    "quickies",
    "reveries",
    "rookies",
    "rotisseries",
    "softies",
    "sorties",
    "species",
    "stymies",
    "sweeties",
    "ties",
    "underlies",
    "unties",
    "veggies",
    "vies",
    "yuppies",
    "zombies",
)


si_sb_oes_oe_case = (
    "Chloes",
    "Crusoes",
    "Defoes",
    "Faeroes",
    "Ivanhoes",
    "Joes",
    "McEnroes",
    "Moes",
    "Monroes",
    "Noes",
    "Poes",
    "Roscoes",
    "Tahoes",
    "Tippecanoes",
    "Zoes",
)

si_sb_oes_oe = (
    "aloes",
    "backhoes",
    "canoes",
    "does",
    "floes",
    "foes",
    "hoes",
    "mistletoes",
    "oboes",
    "pekoes",
    "roes",
    "sloes",
    "throes",
    "tiptoes",
    "toes",
    "woes",
)

si_sb_z_zes = ("quartzes", "topazes")

si_sb_zzes_zz = ("buzzes", "fizzes", "frizzes", "razzes")

si_sb_ches_che_case = (
    "Andromaches",
    "Apaches",
    "Blanches",
    "Comanches",
    "Nietzsches",
    "Porsches",
    "Roches",
)

si_sb_ches_che = (
    "aches",
    "avalanches",
    "backaches",
    "bellyaches",
    "caches",
    "cloches",
    "creches",
    "douches",
    "earaches",
    "fiches",
    "headaches",
    "heartaches",
    "microfiches",
    "niches",
    "pastiches",
    "psyches",
    "quiches",
    "stomachaches",
    "toothaches",
    "tranches",
)

si_sb_xes_xe = ("annexes", "axes", "deluxes", "pickaxes")

si_sb_sses_sse_case = ("Hesses", "Jesses", "Larousses", "Matisses")
si_sb_sses_sse = (
    "bouillabaisses",
    "crevasses",
    "demitasses",
    "impasses",
    "mousses",
    "posses",
)

si_sb_ves_ve_case = (
    # *[nwl]ives -> [nwl]live
    "Clives",
    "Palmolives",
)
si_sb_ves_ve = (
    # *[^d]eaves -> eave
    "interweaves",
    "weaves",
    # *[nwl]ives -> [nwl]live
    "olives",
    # *[eoa]lves -> [eoa]lve
    "bivalves",
    "dissolves",
    "resolves",
    "salves",
    "twelves",
    "valves",
)


plverb_special_s = enclose(
    "|".join(
        [pl_sb_singular_s]
        + pl_sb_uninflected_s
        + list(pl_sb_irregular_s)
        + ["(.*[csx])is", "(.*)ceps", "[A-Z].*s"]
    )
)

_pl_sb_postfix_adj_defn = (
    ("general", enclose(r"(?!major|lieutenant|brigadier|adjutant|.*star)\S+")),
    ("martial", enclose("court")),
    ("force", enclose("pound")),
)

pl_sb_postfix_adj: Iterable[str] = (
    enclose(val + f"(?=(?:-|\\s+){key})") for key, val in _pl_sb_postfix_adj_defn
)

pl_sb_postfix_adj_stems = f"({'|'.join(pl_sb_postfix_adj)})(.*)"


# PLURAL WORDS ENDING IS es GO TO SINGULAR is

si_sb_es_is = (
    "amanuenses",
    "amniocenteses",
    "analyses",
    "antitheses",
    "apotheoses",
    "arterioscleroses",
    "atheroscleroses",
    "axes",
    # 'bases', # bases -> basis
    "catalyses",
    "catharses",
    "chasses",
    "cirrhoses",
    "cocces",
    "crises",
    "diagnoses",
    "dialyses",
    "diereses",
    "electrolyses",
    "emphases",
    "exegeses",
    "geneses",
    "halitoses",
    "hydrolyses",
    "hypnoses",
    "hypotheses",
    "hystereses",
    "metamorphoses",
    "metastases",
    "misdiagnoses",
    "mitoses",
    "mononucleoses",
    "narcoses",
    "necroses",
    "nemeses",
    "neuroses",
    "oases",
    "osmoses",
    "osteoporoses",
    "paralyses",
    "parentheses",
    "parthenogeneses",
    "periphrases",
    "photosyntheses",
    "probosces",
    "prognoses",
    "prophylaxes",
    "prostheses",
    "preces",
    "psoriases",
    "psychoanalyses",
    "psychokineses",
    "psychoses",
    "scleroses",
    "scolioses",
    "sepses",
    "silicoses",
    "symbioses",
    "synopses",
    "syntheses",
    "taxes",
    "telekineses",
    "theses",
    "thromboses",
    "tuberculoses",
    "urinalyses",
)

pl_prep_list = """
    about above across after among around at athwart before behind
    below beneath beside besides between betwixt beyond but by
    during except for from in into near of off on onto out over
    since till to under until unto upon with""".split()

pl_prep_list_da = pl_prep_list + ["de", "du", "da"]

pl_prep_bysize = bysize(pl_prep_list_da)

pl_prep = enclose("|".join(pl_prep_list_da))

pl_sb_prep_dual_compound = rf"(.*?)((?:-|\s+)(?:{pl_prep})(?:-|\s+))a(?:-|\s+)(.*)"


singular_pronoun_genders = {
    "neuter",
    "feminine",
    "masculine",
    "gender-neutral",
    "feminine or masculine",
    "masculine or feminine",
}

pl_pron_nom = {
    # NOMINATIVE    REFLEXIVE
    "i": "we",
    "myself": "ourselves",
    "you": "you",
    "yourself": "yourselves",
    "she": "they",
    "herself": "themselves",
    "he": "they",
    "himself": "themselves",
    "it": "they",
    "itself": "themselves",
    "they": "they",
    "themself": "themselves",
    #   POSSESSIVE
    "mine": "ours",
    "yours": "yours",
    "hers": "theirs",
    "his": "theirs",
    "its": "theirs",
    "theirs": "theirs",
}

si_pron: Dict[str, Dict[str, Union[str, Dict[str, str]]]] = {
    "nom": {v: k for (k, v) in pl_pron_nom.items()}
}
si_pron["nom"]["we"] = "I"


pl_pron_acc = {
    # ACCUSATIVE    REFLEXIVE
    "me": "us",
    "myself": "ourselves",
    "you": "you",
    "yourself": "yourselves",
    "her": "them",
    "herself": "themselves",
    "him": "them",
    "himself": "themselves",
    "it": "them",
    "itself": "themselves",
    "them": "them",
    "themself": "themselves",
}

pl_pron_acc_keys = enclose("|".join(pl_pron_acc))
pl_pron_acc_keys_bysize = bysize(pl_pron_acc)

si_pron["acc"] = {v: k for (k, v) in pl_pron_acc.items()}

for _thecase, _plur, _gend, _sing in (
    ("nom", "they", "neuter", "it"),
    ("nom", "they", "feminine", "she"),
    ("nom", "they", "masculine", "he"),
    ("nom", "they", "gender-neutral", "they"),
    ("nom", "they", "feminine or masculine", "she or he"),
    ("nom", "they", "masculine or feminine", "he or she"),
    ("nom", "themselves", "neuter", "itself"),
    ("nom", "themselves", "feminine", "herself"),
    ("nom", "themselves", "masculine", "himself"),
    ("nom", "themselves", "gender-neutral", "themself"),
    ("nom", "themselves", "feminine or masculine", "herself or himself"),
    ("nom", "themselves", "masculine or feminine", "himself or herself"),
    ("nom", "theirs", "neuter", "its"),
    ("nom", "theirs", "feminine", "hers"),
    ("nom", "theirs", "masculine", "his"),
    ("nom", "theirs", "gender-neutral", "theirs"),
    ("nom", "theirs", "feminine or masculine", "hers or his"),
    ("nom", "theirs", "masculine or feminine", "his or hers"),
    ("acc", "them", "neuter", "it"),
    ("acc", "them", "feminine", "her"),
    ("acc", "them", "masculine", "him"),
    ("acc", "them", "gender-neutral", "them"),
    ("acc", "them", "feminine or masculine", "her or him"),
    ("acc", "them", "masculine or feminine", "him or her"),
    ("acc", "themselves", "neuter", "itself"),
    ("acc", "themselves", "feminine", "herself"),
    ("acc", "themselves", "masculine", "himself"),
    ("acc", "themselves", "gender-neutral", "themself"),
    ("acc", "themselves", "feminine or masculine", "herself or himself"),
    ("acc", "themselves", "masculine or feminine", "himself or herself"),
):
    try:
        si_pron[_thecase][_plur][_gend] = _sing  # type: ignore
    except TypeError:
        si_pron[_thecase][_plur] = {}
        si_pron[_thecase][_plur][_gend] = _sing  # type: ignore


si_pron_acc_keys = enclose("|".join(si_pron["acc"]))
si_pron_acc_keys_bysize = bysize(si_pron["acc"])


def get_si_pron(thecase, word, gender) -> str:
    try:
        sing = si_pron[thecase][word]
    except KeyError:
        raise  # not a pronoun
    try:
        return sing[gender]  # has several types due to gender
    except TypeError:
        return cast(str, sing)  # answer independent of gender


# These dictionaries group verbs by first, second and third person
# conjugations.

plverb_irregular_pres = {
    "am": "are",
    "are": "are",
    "is": "are",
    "was": "were",
    "were": "were",
    "have": "have",
    "has": "have",
    "do": "do",
    "does": "do",
}

plverb_ambiguous_pres = {
    "act": "act",
    "acts": "act",
    "blame": "blame",
    "blames": "blame",
    "can": "can",
    "must": "must",
    "fly": "fly",
    "flies": "fly",
    "copy": "copy",
    "copies": "copy",
    "drink": "drink",
    "drinks": "drink",
    "fight": "fight",
    "fights": "fight",
    "fire": "fire",
    "fires": "fire",
    "like": "like",
    "likes": "like",
    "look": "look",
    "looks": "look",
    "make": "make",
    "makes": "make",
    "reach": "reach",
    "reaches": "reach",
    "run": "run",
    "runs": "run",
    "sink": "sink",
    "sinks": "sink",
    "sleep": "sleep",
    "sleeps": "sleep",
    "view": "view",
    "views": "view",
}

plverb_ambiguous_pres_keys = re.compile(
    rf"^({enclose('|'.join(plverb_ambiguous_pres))})((\s.*)?)$", re.IGNORECASE
)


plverb_irregular_non_pres = (
    "did",
    "had",
    "ate",
    "made",
    "put",
    "spent",
    "fought",
    "sank",
    "gave",
    "sought",
    "shall",
    "could",
    "ought",
    "should",
)

plverb_ambiguous_non_pres = re.compile(
    r"^((?:thought|saw|bent|will|might|cut))((\s.*)?)$", re.IGNORECASE
)

# "..oes" -> "..oe" (the rest are "..oes" -> "o")

pl_v_oes_oe = ("canoes", "floes", "oboes", "roes", "throes", "woes")
pl_v_oes_oe_endings_size4 = ("hoes", "toes")
pl_v_oes_oe_endings_size5 = ("shoes",)


pl_count_zero = ("0", "no", "zero", "nil")


pl_count_one = ("1", "a", "an", "one", "each", "every", "this", "that")

pl_adj_special = {"a": "some", "an": "some", "this": "these", "that": "those"}

pl_adj_special_keys = re.compile(
    rf"^({enclose('|'.join(pl_adj_special))})$", re.IGNORECASE
)

pl_adj_poss = {
    "my": "our",
    "your": "your",
    "its": "their",
    "her": "their",
    "his": "their",
    "their": "their",
}

pl_adj_poss_keys = re.compile(rf"^({enclose('|'.join(pl_adj_poss))})$", re.IGNORECASE)


# 2. INDEFINITE ARTICLES

# THIS PATTERN MATCHES STRINGS OF CAPITALS STARTING WITH A "VOWEL-SOUND"
# CONSONANT FOLLOWED BY ANOTHER CONSONANT, AND WHICH ARE NOT LIKELY
# TO BE REAL WORDS (OH, ALL RIGHT THEN, IT'S JUST MAGIC!)

A_abbrev = re.compile(
    r"""
^(?! FJO | [HLMNS]Y.  | RY[EO] | SQU
  | ( F[LR]? | [HL] | MN? | N | RH? | S[CHKLMNPTVW]? | X(YL)?) [AEIOU])
[FHLMNRSX][A-Z]
""",
    re.VERBOSE,
)

# THIS PATTERN CODES THE BEGINNINGS OF ALL ENGLISH WORDS BEGINING WITH A
# 'y' FOLLOWED BY A CONSONANT. ANY OTHER Y-CONSONANT PREFIX THEREFORE
# IMPLIES AN ABBREVIATION.

A_y_cons = re.compile(r"^(y(b[lor]|cl[ea]|fere|gg|p[ios]|rou|tt))", re.IGNORECASE)

# EXCEPTIONS TO EXCEPTIONS

A_explicit_a = re.compile(r"^((?:unabomber|unanimous|US))", re.IGNORECASE)

A_explicit_an = re.compile(
    r"^((?:euler|hour(?!i)|heir|honest|hono[ur]|mpeg))", re.IGNORECASE
)

A_ordinal_an = re.compile(r"^([aefhilmnorsx]-?th)", re.IGNORECASE)

A_ordinal_a = re.compile(r"^([bcdgjkpqtuvwyz]-?th)", re.IGNORECASE)


# NUMERICAL INFLECTIONS

nth = {
    0: "th",
    1: "st",
    2: "nd",
    3: "rd",
    4: "th",
    5: "th",
    6: "th",
    7: "th",
    8: "th",
    9: "th",
    11: "th",
    12: "th",
    13: "th",
}
nth_suff = set(nth.values())

ordinal = dict(
    ty="tieth",
    one="first",
    two="second",
    three="third",
    five="fifth",
    eight="eighth",
    nine="ninth",
    twelve="twelfth",
)

ordinal_suff = re.compile(rf"({'|'.join(ordinal)})\Z")


# NUMBERS

unit = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
teen = [
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
]
ten = [
    "",
    "",
    "twenty",
    "thirty",
    "forty",
    "fifty",
    "sixty",
    "seventy",
    "eighty",
    "ninety",
]
mill = [
    " ",
    " thousand",
    " million",
    " billion",
    " trillion",
    " quadrillion",
    " quintillion",
    " sextillion",
    " septillion",
    " octillion",
    " nonillion",
    " decillion",
]


# SUPPORT CLASSICAL PLURALIZATIONS

def_classical = dict(
    all=False, zero=False, herd=False, names=True, persons=False, ancient=False
)

all_classical = {k: True for k in def_classical}
no_classical = {k: False for k in def_classical}


# Maps strings to built-in constant types
string_to_constant = {"True": True, "False": False, "None": None}


# Pre-compiled regular expression objects
DOLLAR_DIGITS = re.compile(r"\$(\d+)")
FUNCTION_CALL = re.compile(r"((\w+)\([^)]*\)*)", re.IGNORECASE)
PARTITION_WORD = re.compile(r"\A(\s*)(.+?)(\s*)\Z")
PL_SB_POSTFIX_ADJ_STEMS_RE = re.compile(
    rf"^(?:{pl_sb_postfix_adj_stems})$", re.IGNORECASE
)
PL_SB_PREP_DUAL_COMPOUND_RE = re.compile(
    rf"^(?:{pl_sb_prep_dual_compound})$", re.IGNORECASE
)
DENOMINATOR = re.compile(r"(?P<denominator>.+)( (per|a) .+)")
PLVERB_SPECIAL_S_RE = re.compile(rf"^({plverb_special_s})$")
WHITESPACE = re.compile(r"\s")
ENDS_WITH_S = re.compile(r"^(.*[^s])s$", re.IGNORECASE)
ENDS_WITH_APOSTROPHE_S = re.compile(r"^(.*)'s?$")
INDEFINITE_ARTICLE_TEST = re.compile(r"\A(\s*)(?:an?\s+)?(.+?)(\s*)\Z", re.IGNORECASE)
SPECIAL_AN = re.compile(r"^[aefhilmnorsx]$", re.IGNORECASE)
SPECIAL_A = re.compile(r"^[bcdgjkpqtuvwyz]$", re.IGNORECASE)
SPECIAL_ABBREV_AN = re.compile(r"^[aefhilmnorsx][.-]", re.IGNORECASE)
SPECIAL_ABBREV_A = re.compile(r"^[a-z][.-]", re.IGNORECASE)
CONSONANTS = re.compile(r"^[^aeiouy]", re.IGNORECASE)
ARTICLE_SPECIAL_EU = re.compile(r"^e[uw]", re.IGNORECASE)
ARTICLE_SPECIAL_ONCE = re.compile(r"^onc?e\b", re.IGNORECASE)
ARTICLE_SPECIAL_ONETIME = re.compile(r"^onetime\b", re.IGNORECASE)
ARTICLE_SPECIAL_UNIT = re.compile(r"^uni([^nmd]|mo)", re.IGNORECASE)
ARTICLE_SPECIAL_UBA = re.compile(r"^u[bcfghjkqrst][aeiou]", re.IGNORECASE)
ARTICLE_SPECIAL_UKR = re.compile(r"^ukr", re.IGNORECASE)
SPECIAL_CAPITALS = re.compile(r"^U[NK][AIEO]?")
VOWELS = re.compile(r"^[aeiou]", re.IGNORECASE)

DIGIT_GROUP = re.compile(r"(\d)")
TWO_DIGITS = re.compile(r"(\d)(\d)")
THREE_DIGITS = re.compile(r"(\d)(\d)(\d)")
THREE_DIGITS_WORD = re.compile(r"(\d)(\d)(\d)(?=\D*\Z)")
TWO_DIGITS_WORD = re.compile(r"(\d)(\d)(?=\D*\Z)")
ONE_DIGIT_WORD = re.compile(r"(\d)(?=\D*\Z)")

FOUR_DIGIT_COMMA = re.compile(r"(\d)(\d{3}(?:,|\Z))")
NON_DIGIT = re.compile(r"\D")
WHITESPACES_COMMA = re.compile(r"\s+,")
COMMA_WORD = re.compile(r", (\S+)\s+\Z")
WHITESPACES = re.compile(r"\s+")


PRESENT_PARTICIPLE_REPLACEMENTS = (
    (re.compile(r"ie$"), r"y"),
    (
        re.compile(r"ue$"),
        r"u",
    ),  # TODO: isn't ue$ -> u encompassed in the following rule?
    (re.compile(r"([auy])e$"), r"\g<1>"),
    (re.compile(r"ski$"), r"ski"),
    (re.compile(r"[^b]i$"), r""),
    (re.compile(r"^(are|were)$"), r"be"),
    (re.compile(r"^(had)$"), r"hav"),
    (re.compile(r"^(hoe)$"), r"\g<1>"),
    (re.compile(r"([^e])e$"), r"\g<1>"),
    (re.compile(r"er$"), r"er"),
    (re.compile(r"([^aeiou][aeiouy]([bdgmnprst]))$"), r"\g<1>\g<2>"),
)

DIGIT = re.compile(r"\d")


class Words(str):
    lowered: str
    split_: List[str]
    first: str
    last: str

    def __init__(self, orig) -> None:
        self.lowered = self.lower()
        self.split_ = self.split()
        self.first = self.split_[0]
        self.last = self.split_[-1]


Falsish = Any  # ideally, falsish would only validate on bool(value) is False


_STATIC_TYPE_CHECKING = TYPE_CHECKING
# ^-- Workaround for typeguard AST manipulation:
#     https://github.com/agronholm/typeguard/issues/353#issuecomment-1556306554

if _STATIC_TYPE_CHECKING:  # pragma: no cover
    Word = Annotated[str, "String with at least 1 character"]
else:

    class _WordMeta(type):  # Too dynamic to be supported by mypy...
        def __instancecheck__(self, instance: Any) -> bool:
            return isinstance(instance, str) and len(instance) >= 1

    class Word(metaclass=_WordMeta):  # type: ignore[no-redef]
        """String with at least 1 character"""


class engine:
    def __init__(self) -> None:
        self.classical_dict = def_classical.copy()
        self.persistent_count: Optional[int] = None
        self.mill_count = 0
        self.pl_sb_user_defined: List[Optional[Word]] = []
        self.pl_v_user_defined: List[Optional[Word]] = []
        self.pl_adj_user_defined: List[Optional[Word]] = []
        self.si_sb_user_defined: List[Optional[Word]] = []
        self.A_a_user_defined: List[Optional[Word]] = []
        self.thegender = "neuter"
        self.__number_args: Optional[Dict[str, str]] = None

    @property
    def _number_args(self):
        return cast(Dict[str, str], self.__number_args)

    @_number_args.setter
    def _number_args(self, val):
        self.__number_args = val

    @typechecked
    def defnoun(self, singular: Optional[Word], plural: Optional[Word]) -> int:
        """
        Set the noun plural of singular to plural.

        """
        self.checkpat(singular)
        self.checkpatplural(plural)
        self.pl_sb_user_defined.extend((singular, plural))
        self.si_sb_user_defined.extend((plural, singular))
        return 1

    @typechecked
    def defverb(
        self,
        s1: Optional[Word],
        p1: Optional[Word],
        s2: Optional[Word],
        p2: Optional[Word],
        s3: Optional[Word],
        p3: Optional[Word],
    ) -> int:
        """
        Set the verb plurals for s1, s2 and s3 to p1, p2 and p3 respectively.

        Where 1, 2 and 3 represent the 1st, 2nd and 3rd person forms of the verb.

        """
        self.checkpat(s1)
        self.checkpat(s2)
        self.checkpat(s3)
        self.checkpatplural(p1)
        self.checkpatplural(p2)
        self.checkpatplural(p3)
        self.pl_v_user_defined.extend((s1, p1, s2, p2, s3, p3))
        return 1

    @typechecked
    def defadj(self, singular: Optional[Word], plural: Optional[Word]) -> int:
        """
        Set the adjective plural of singular to plural.

        """
        self.checkpat(singular)
        self.checkpatplural(plural)
        self.pl_adj_user_defined.extend((singular, plural))
        return 1

    @typechecked
    def defa(self, pattern: Optional[Word]) -> int:
        """
        Define the indefinite article as 'a' for words matching pattern.

        """
        self.checkpat(pattern)
        self.A_a_user_defined.extend((pattern, "a"))
        return 1

    @typechecked
    def defan(self, pattern: Optional[Word]) -> int:
        """
        Define the indefinite article as 'an' for words matching pattern.

        """
        self.checkpat(pattern)
        self.A_a_user_defined.extend((pattern, "an"))
        return 1

    def checkpat(self, pattern: Optional[Word]) -> None:
        """
        check for errors in a regex pattern
        """
        if pattern is None:
            return
        try:
            re.match(pattern, "")
        except re.error as err:
            raise BadUserDefinedPatternError(pattern) from err

    def checkpatplural(self, pattern: Optional[Word]) -> None:
        """
        check for errors in a regex replace pattern
        """
        return

    @typechecked
    def ud_match(self, word: Word, wordlist: Sequence[Optional[Word]]) -> Optional[str]:
        for i in range(len(wordlist) - 2, -2, -2):  # backwards through even elements
            mo = re.search(rf"^{wordlist[i]}$", word, re.IGNORECASE)
            if mo:
                if wordlist[i + 1] is None:
                    return None
                pl = DOLLAR_DIGITS.sub(
                    r"\\1", cast(Word, wordlist[i + 1])
                )  # change $n to \n for expand
                return mo.expand(pl)
        return None

    def classical(self, **kwargs) -> None:
        """
        turn classical mode on and off for various categories

        turn on all classical modes:
        classical()
        classical(all=True)

        turn on or off specific claassical modes:
        e.g.
        classical(herd=True)
        classical(names=False)

        By default all classical modes are off except names.

        unknown value in args or key in kwargs raises
        exception: UnknownClasicalModeError

        """
        if not kwargs:
            self.classical_dict = all_classical.copy()
            return
        if "all" in kwargs:
            if kwargs["all"]:
                self.classical_dict = all_classical.copy()
            else:
                self.classical_dict = no_classical.copy()

        for k, v in kwargs.items():
            if k in def_classical:
                self.classical_dict[k] = v
            else:
                raise UnknownClassicalModeError

    def num(
        self, count: Optional[int] = None, show: Optional[int] = None
    ) -> str:  # (;$count,$show)
        """
        Set the number to be used in other method calls.

        Returns count.

        Set show to False to return '' instead.

        """
        if count is not None:
            try:
                self.persistent_count = int(count)
            except ValueError as err:
                raise BadNumValueError from err
            if (show is None) or show:
                return str(count)
        else:
            self.persistent_count = None
        return ""

    def gender(self, gender: str) -> None:
        """
        set the gender for the singular of plural pronouns

        can be one of:
        'neuter'                ('they' -> 'it')
        'feminine'              ('they' -> 'she')
        'masculine'             ('they' -> 'he')
        'gender-neutral'        ('they' -> 'they')
        'feminine or masculine' ('they' -> 'she or he')
        'masculine or feminine' ('they' -> 'he or she')
        """
        if gender in singular_pronoun_genders:
            self.thegender = gender
        else:
            raise BadGenderError

    def _get_value_from_ast(self, obj):
        """
        Return the value of the ast object.
        """
        if isinstance(obj, ast.Num):
            return obj.n
        elif isinstance(obj, ast.Str):
            return obj.s
        elif isinstance(obj, ast.List):
            return [self._get_value_from_ast(e) for e in obj.elts]
        elif isinstance(obj, ast.Tuple):
            return tuple([self._get_value_from_ast(e) for e in obj.elts])

        # None, True and False are NameConstants in Py3.4 and above.
        elif isinstance(obj, ast.NameConstant):
            return obj.value

        # Probably passed a variable name.
        # Or passed a single word without wrapping it in quotes as an argument
        # ex: p.inflect("I plural(see)") instead of p.inflect("I plural('see')")
        raise NameError(f"name '{obj.id}' is not defined")

    def _string_to_substitute(
        self, mo: Match, methods_dict: Dict[str, Callable]
    ) -> str:
        """
        Return the string to be substituted for the match.
        """
        matched_text, f_name = mo.groups()
        # matched_text is the complete match string. e.g. plural_noun(cat)
        # f_name is the function name. e.g. plural_noun

        # Return matched_text if function name is not in methods_dict
        if f_name not in methods_dict:
            return matched_text

        # Parse the matched text
        a_tree = ast.parse(matched_text)

        # get the args and kwargs from ast objects
        args_list = [
            self._get_value_from_ast(a)
            for a in a_tree.body[0].value.args  # type: ignore[attr-defined]
        ]
        kwargs_list = {
            kw.arg: self._get_value_from_ast(kw.value)
            for kw in a_tree.body[0].value.keywords  # type: ignore[attr-defined]
        }

        # Call the corresponding function
        return methods_dict[f_name](*args_list, **kwargs_list)

    # 0. PERFORM GENERAL INFLECTIONS IN A STRING

    @typechecked
    def inflect(self, text: Word) -> str:
        """
        Perform inflections in a string.

        e.g. inflect('The plural of cat is plural(cat)') returns
        'The plural of cat is cats'

        can use plural, plural_noun, plural_verb, plural_adj,
        singular_noun, a, an, no, ordinal, number_to_words,
        and prespart

        """
        save_persistent_count = self.persistent_count

        # Dictionary of allowed methods
        methods_dict: Dict[str, Callable] = {
            "plural": self.plural,
            "plural_adj": self.plural_adj,
            "plural_noun": self.plural_noun,
            "plural_verb": self.plural_verb,
            "singular_noun": self.singular_noun,
            "a": self.a,
            "an": self.a,
            "no": self.no,
            "ordinal": self.ordinal,
            "number_to_words": self.number_to_words,
            "present_participle": self.present_participle,
            "num": self.num,
        }

        # Regular expression to find Python's function call syntax
        output = FUNCTION_CALL.sub(
            lambda mo: self._string_to_substitute(mo, methods_dict), text
        )
        self.persistent_count = save_persistent_count
        return output

    # ## PLURAL SUBROUTINES

    def postprocess(self, orig: str, inflected) -> str:
        inflected = str(inflected)
        if "|" in inflected:
            word_options = inflected.split("|")
            # When two parts of a noun need to be pluralized
            if len(word_options[0].split(" ")) == len(word_options[1].split(" ")):
                result = inflected.split("|")[self.classical_dict["all"]].split(" ")
            # When only the last part of the noun needs to be pluralized
            else:
                result = inflected.split(" ")
                for index, word in enumerate(result):
                    if "|" in word:
                        result[index] = word.split("|")[self.classical_dict["all"]]
        else:
            result = inflected.split(" ")

        # Try to fix word wise capitalization
        for index, word in enumerate(orig.split(" ")):
            if word == "I":
                # Is this the only word for exceptions like this
                # Where the original is fully capitalized
                # without 'meaning' capitalization?
                # Also this fails to handle a capitalizaion in context
                continue
            if word.capitalize() == word:
                result[index] = result[index].capitalize()
            if word == word.upper():
                result[index] = result[index].upper()
        return " ".join(result)

    def partition_word(self, text: str) -> Tuple[str, str, str]:
        mo = PARTITION_WORD.search(text)
        if mo:
            return mo.group(1), mo.group(2), mo.group(3)
        else:
            return "", "", ""

    @typechecked
    def plural(self, text: Word, count: Optional[Union[str, int, Any]] = None) -> str:
        """
        Return the plural of text.

        If count supplied, then return text if count is one of:
            1, a, an, one, each, every, this, that

        otherwise return the plural.

        Whitespace at the start and end is preserved.

        """
        pre, word, post = self.partition_word(text)
        if not word:
            return text
        plural = self.postprocess(
            word,
            self._pl_special_adjective(word, count)
            or self._pl_special_verb(word, count)
            or self._plnoun(word, count),
        )
        return f"{pre}{plural}{post}"

    @typechecked
    def plural_noun(
        self, text: Word, count: Optional[Union[str, int, Any]] = None
    ) -> str:
        """
        Return the plural of text, where text is a noun.

        If count supplied, then return text if count is one of:
            1, a, an, one, each, every, this, that

        otherwise return the plural.

        Whitespace at the start and end is preserved.

        """
        pre, word, post = self.partition_word(text)
        if not word:
            return text
        plural = self.postprocess(word, self._plnoun(word, count))
        return f"{pre}{plural}{post}"

    @typechecked
    def plural_verb(
        self, text: Word, count: Optional[Union[str, int, Any]] = None
    ) -> str:
        """
        Return the plural of text, where text is a verb.

        If count supplied, then return text if count is one of:
            1, a, an, one, each, every, this, that

        otherwise return the plural.

        Whitespace at the start and end is preserved.

        """
        pre, word, post = self.partition_word(text)
        if not word:
            return text
        plural = self.postprocess(
            word,
            self._pl_special_verb(word, count) or self._pl_general_verb(word, count),
        )
        return f"{pre}{plural}{post}"

    @typechecked
    def plural_adj(
        self, text: Word, count: Optional[Union[str, int, Any]] = None
    ) -> str:
        """
        Return the plural of text, where text is an adjective.

        If count supplied, then return text if count is one of:
            1, a, an, one, each, every, this, that

        otherwise return the plural.

        Whitespace at the start and end is preserved.

        """
        pre, word, post = self.partition_word(text)
        if not word:
            return text
        plural = self.postprocess(word, self._pl_special_adjective(word, count) or word)
        return f"{pre}{plural}{post}"

    @typechecked
    def compare(self, word1: Word, word2: Word) -> Union[str, bool]:
        """
        compare word1 and word2 for equality regardless of plurality

        return values:
        eq - the strings are equal
        p:s - word1 is the plural of word2
        s:p - word2 is the plural of word1
        p:p - word1 and word2 are two different plural forms of the one word
        False - otherwise

        >>> compare = engine().compare
        >>> compare("egg", "eggs")
        's:p'
        >>> compare('egg', 'egg')
        'eq'

        Words should not be empty.

        >>> compare('egg', '')
        Traceback (most recent call last):
        ...
        typeguard.TypeCheckError:...is not an instance of inflect.Word
        """
        norms = self.plural_noun, self.plural_verb, self.plural_adj
        results = (self._plequal(word1, word2, norm) for norm in norms)
        return next(filter(None, results), False)

    @typechecked
    def compare_nouns(self, word1: Word, word2: Word) -> Union[str, bool]:
        """
        compare word1 and word2 for equality regardless of plurality
        word1 and word2 are to be treated as nouns

        return values:
        eq - the strings are equal
        p:s - word1 is the plural of word2
        s:p - word2 is the plural of word1
        p:p - word1 and word2 are two different plural forms of the one word
        False - otherwise

        """
        return self._plequal(word1, word2, self.plural_noun)

    @typechecked
    def compare_verbs(self, word1: Word, word2: Word) -> Union[str, bool]:
        """
        compare word1 and word2 for equality regardless of plurality
        word1 and word2 are to be treated as verbs

        return values:
        eq - the strings are equal
        p:s - word1 is the plural of word2
        s:p - word2 is the plural of word1
        p:p - word1 and word2 are two different plural forms of the one word
        False - otherwise

        """
        return self._plequal(word1, word2, self.plural_verb)

    @typechecked
    def compare_adjs(self, word1: Word, word2: Word) -> Union[str, bool]:
        """
        compare word1 and word2 for equality regardless of plurality
        word1 and word2 are to be treated as adjectives

        return values:
        eq - the strings are equal
        p:s - word1 is the plural of word2
        s:p - word2 is the plural of word1
        p:p - word1 and word2 are two different plural forms of the one word
        False - otherwise

        """
        return self._plequal(word1, word2, self.plural_adj)

    @typechecked
    def singular_noun(
        self,
        text: Word,
        count: Optional[Union[int, str, Any]] = None,
        gender: Optional[str] = None,
    ) -> Union[str, Literal[False]]:
        """
        Return the singular of text, where text is a plural noun.

        If count supplied, then return the singular if count is one of:
            1, a, an, one, each, every, this, that or if count is None

        otherwise return text unchanged.

        Whitespace at the start and end is preserved.

        >>> p = engine()
        >>> p.singular_noun('horses')
        'horse'
        >>> p.singular_noun('knights')
        'knight'

        Returns False when a singular noun is passed.

        >>> p.singular_noun('horse')
        False
        >>> p.singular_noun('knight')
        False
        >>> p.singular_noun('soldier')
        False

        """
        pre, word, post = self.partition_word(text)
        if not word:
            return text
        sing = self._sinoun(word, count=count, gender=gender)
        if sing is not False:
            plural = self.postprocess(word, sing)
            return f"{pre}{plural}{post}"
        return False

    def _plequal(self, word1: str, word2: str, pl) -> Union[str, bool]:  # noqa: C901
        classval = self.classical_dict.copy()
        self.classical_dict = all_classical.copy()
        if word1 == word2:
            return "eq"
        if word1 == pl(word2):
            return "p:s"
        if pl(word1) == word2:
            return "s:p"
        self.classical_dict = no_classical.copy()
        if word1 == pl(word2):
            return "p:s"
        if pl(word1) == word2:
            return "s:p"
        self.classical_dict = classval.copy()

        if pl == self.plural or pl == self.plural_noun:
            if self._pl_check_plurals_N(word1, word2):
                return "p:p"
            if self._pl_check_plurals_N(word2, word1):
                return "p:p"
        if pl == self.plural or pl == self.plural_adj:
            if self._pl_check_plurals_adj(word1, word2):
                return "p:p"
        return False

    def _pl_reg_plurals(self, pair: str, stems: str, end1: str, end2: str) -> bool:
        pattern = rf"({stems})({end1}\|\1{end2}|{end2}\|\1{end1})"
        return bool(re.search(pattern, pair))

    def _pl_check_plurals_N(self, word1: str, word2: str) -> bool:
        stem_endings = (
            (pl_sb_C_a_ata, "as", "ata"),
            (pl_sb_C_is_ides, "is", "ides"),
            (pl_sb_C_a_ae, "s", "e"),
            (pl_sb_C_en_ina, "ens", "ina"),
            (pl_sb_C_um_a, "ums", "a"),
            (pl_sb_C_us_i, "uses", "i"),
            (pl_sb_C_on_a, "ons", "a"),
            (pl_sb_C_o_i_stems, "os", "i"),
            (pl_sb_C_ex_ices, "exes", "ices"),
            (pl_sb_C_ix_ices, "ixes", "ices"),
            (pl_sb_C_i, "s", "i"),
            (pl_sb_C_im, "s", "im"),
            (".*eau", "s", "x"),
            (".*ieu", "s", "x"),
            (".*tri", "xes", "ces"),
            (".{2,}[yia]n", "xes", "ges"),
        )

        words = map(Words, (word1, word2))
        pair = "|".join(word.last for word in words)

        return (
            pair in pl_sb_irregular_s.values()
            or pair in pl_sb_irregular.values()
            or pair in pl_sb_irregular_caps.values()
            or any(
                self._pl_reg_plurals(pair, stems, end1, end2)
                for stems, end1, end2 in stem_endings
            )
        )

    def _pl_check_plurals_adj(self, word1: str, word2: str) -> bool:
        word1a = word1[: word1.rfind("'")] if word1.endswith(("'s", "'")) else ""
        word2a = word2[: word2.rfind("'")] if word2.endswith(("'s", "'")) else ""

        return (
            bool(word1a)
            and bool(word2a)
            and (
                self._pl_check_plurals_N(word1a, word2a)
                or self._pl_check_plurals_N(word2a, word1a)
            )
        )

    def get_count(self, count: Optional[Union[str, int]] = None) -> Union[str, int]:
        if count is None and self.persistent_count is not None:
            count = self.persistent_count

        if count is not None:
            count = (
                1
                if (
                    (str(count) in pl_count_one)
                    or (
                        self.classical_dict["zero"]
                        and str(count).lower() in pl_count_zero
                    )
                )
                else 2
            )
        else:
            count = ""
        return count

    # @profile
    def _plnoun(  # noqa: C901
        self, word: str, count: Optional[Union[str, int]] = None
    ) -> str:
        count = self.get_count(count)

        # DEFAULT TO PLURAL

        if count == 1:
            return word

        # HANDLE USER-DEFINED NOUNS

        value = self.ud_match(word, self.pl_sb_user_defined)
        if value is not None:
            return value

        # HANDLE EMPTY WORD, SINGULAR COUNT AND UNINFLECTED PLURALS

        if word == "":
            return word

        word = Words(word)

        if word.last.lower() in pl_sb_uninflected_complete:
            if len(word.split_) >= 3:
                return self._handle_long_compounds(word, count=2) or word
            return word

        if word in pl_sb_uninflected_caps:
            return word

        for k, v in pl_sb_uninflected_bysize.items():
            if word.lowered[-k:] in v:
                return word

        if self.classical_dict["herd"] and word.last.lower() in pl_sb_uninflected_herd:
            return word

        # HANDLE COMPOUNDS ("Governor General", "mother-in-law", "aide-de-camp", ETC.)

        mo = PL_SB_POSTFIX_ADJ_STEMS_RE.search(word)
        if mo and mo.group(2) != "":
            return f"{self._plnoun(mo.group(1), 2)}{mo.group(2)}"

        if " a " in word.lowered or "-a-" in word.lowered:
            mo = PL_SB_PREP_DUAL_COMPOUND_RE.search(word)
            if mo and mo.group(2) != "" and mo.group(3) != "":
                return (
                    f"{self._plnoun(mo.group(1), 2)}"
                    f"{mo.group(2)}"
                    f"{self._plnoun(mo.group(3))}"
                )

        if len(word.split_) >= 3:
            handled_words = self._handle_long_compounds(word, count=2)
            if handled_words is not None:
                return handled_words

        # only pluralize denominators in units
        mo = DENOMINATOR.search(word.lowered)
        if mo:
            index = len(mo.group("denominator"))
            return f"{self._plnoun(word[:index])}{word[index:]}"

        # handle units given in degrees (only accept if
        # there is no more than one word following)
        # degree Celsius => degrees Celsius but degree
        # fahrenheit hour => degree fahrenheit hours
        if len(word.split_) >= 2 and word.split_[-2] == "degree":
            return " ".join([self._plnoun(word.first)] + word.split_[1:])

        with contextlib.suppress(ValueError):
            return self._handle_prepositional_phrase(
                word.lowered,
                functools.partial(self._plnoun, count=2),
                '-',
            )

        # HANDLE PRONOUNS

        for k, v in pl_pron_acc_keys_bysize.items():
            if word.lowered[-k:] in v:  # ends with accusative pronoun
                for pk, pv in pl_prep_bysize.items():
                    if word.lowered[:pk] in pv:  # starts with a prep
                        if word.lowered.split() == [
                            word.lowered[:pk],
                            word.lowered[-k:],
                        ]:
                            # only whitespace in between
                            return word.lowered[:-k] + pl_pron_acc[word.lowered[-k:]]

        try:
            return pl_pron_nom[word.lowered]
        except KeyError:
            pass

        try:
            return pl_pron_acc[word.lowered]
        except KeyError:
            pass

        # HANDLE ISOLATED IRREGULAR PLURALS

        if word.last in pl_sb_irregular_caps:
            llen = len(word.last)
            return f"{word[:-llen]}{pl_sb_irregular_caps[word.last]}"

        lowered_last = word.last.lower()
        if lowered_last in pl_sb_irregular:
            llen = len(lowered_last)
            return f"{word[:-llen]}{pl_sb_irregular[lowered_last]}"

        dash_split = word.lowered.split('-')
        if (" ".join(dash_split[-2:])).lower() in pl_sb_irregular_compound:
            llen = len(
                " ".join(dash_split[-2:])
            )  # TODO: what if 2 spaces between these words?
            return (
                f"{word[:-llen]}"
                f"{pl_sb_irregular_compound[(' '.join(dash_split[-2:])).lower()]}"
            )

        if word.lowered[-3:] == "quy":
            return f"{word[:-1]}ies"

        if word.lowered[-6:] == "person":
            if self.classical_dict["persons"]:
                return f"{word}s"
            else:
                return f"{word[:-4]}ople"

        # HANDLE FAMILIES OF IRREGULAR PLURALS

        if word.lowered[-3:] == "man":
            for k, v in pl_sb_U_man_mans_bysize.items():
                if word.lowered[-k:] in v:
                    return f"{word}s"
            for k, v in pl_sb_U_man_mans_caps_bysize.items():
                if word[-k:] in v:
                    return f"{word}s"
            return f"{word[:-3]}men"
        if word.lowered[-5:] == "mouse":
            return f"{word[:-5]}mice"
        if word.lowered[-5:] == "louse":
            v = pl_sb_U_louse_lice_bysize.get(len(word))
            if v and word.lowered in v:
                return f"{word[:-5]}lice"
            return f"{word}s"
        if word.lowered[-5:] == "goose":
            return f"{word[:-5]}geese"
        if word.lowered[-5:] == "tooth":
            return f"{word[:-5]}teeth"
        if word.lowered[-4:] == "foot":
            return f"{word[:-4]}feet"
        if word.lowered[-4:] == "taco":
            return f"{word[:-5]}tacos"

        if word.lowered == "die":
            return "dice"

        # HANDLE UNASSIMILATED IMPORTS

        if word.lowered[-4:] == "ceps":
            return word
        if word.lowered[-4:] == "zoon":
            return f"{word[:-2]}a"
        if word.lowered[-3:] in ("cis", "sis", "xis"):
            return f"{word[:-2]}es"

        for lastlet, d, numend, post in (
            ("h", pl_sb_U_ch_chs_bysize, None, "s"),
            ("x", pl_sb_U_ex_ices_bysize, -2, "ices"),
            ("x", pl_sb_U_ix_ices_bysize, -2, "ices"),
            ("m", pl_sb_U_um_a_bysize, -2, "a"),
            ("s", pl_sb_U_us_i_bysize, -2, "i"),
            ("n", pl_sb_U_on_a_bysize, -2, "a"),
            ("a", pl_sb_U_a_ae_bysize, None, "e"),
        ):
            if word.lowered[-1] == lastlet:  # this test to add speed
                for k, v in d.items():
                    if word.lowered[-k:] in v:
                        return word[:numend] + post

        # HANDLE INCOMPLETELY ASSIMILATED IMPORTS

        if self.classical_dict["ancient"]:
            if word.lowered[-4:] == "trix":
                return f"{word[:-1]}ces"
            if word.lowered[-3:] in ("eau", "ieu"):
                return f"{word}x"
            if word.lowered[-3:] in ("ynx", "inx", "anx") and len(word) > 4:
                return f"{word[:-1]}ges"

            for lastlet, d, numend, post in (
                ("n", pl_sb_C_en_ina_bysize, -2, "ina"),
                ("x", pl_sb_C_ex_ices_bysize, -2, "ices"),
                ("x", pl_sb_C_ix_ices_bysize, -2, "ices"),
                ("m", pl_sb_C_um_a_bysize, -2, "a"),
                ("s", pl_sb_C_us_i_bysize, -2, "i"),
                ("s", pl_sb_C_us_us_bysize, None, ""),
                ("a", pl_sb_C_a_ae_bysize, None, "e"),
                ("a", pl_sb_C_a_ata_bysize, None, "ta"),
                ("s", pl_sb_C_is_ides_bysize, -1, "des"),
                ("o", pl_sb_C_o_i_bysize, -1, "i"),
                ("n", pl_sb_C_on_a_bysize, -2, "a"),
            ):
                if word.lowered[-1] == lastlet:  # this test to add speed
                    for k, v in d.items():
                        if word.lowered[-k:] in v:
                            return word[:numend] + post

            for d, numend, post in (
                (pl_sb_C_i_bysize, None, "i"),
                (pl_sb_C_im_bysize, None, "im"),
            ):
                for k, v in d.items():
                    if word.lowered[-k:] in v:
                        return word[:numend] + post

        # HANDLE SINGULAR NOUNS ENDING IN ...s OR OTHER SILIBANTS

        if lowered_last in pl_sb_singular_s_complete:
            return f"{word}es"

        for k, v in pl_sb_singular_s_bysize.items():
            if word.lowered[-k:] in v:
                return f"{word}es"

        if word.lowered[-2:] == "es" and word[0] == word[0].upper():
            return f"{word}es"

        if word.lowered[-1] == "z":
            for k, v in pl_sb_z_zes_bysize.items():
                if word.lowered[-k:] in v:
                    return f"{word}es"

            if word.lowered[-2:-1] != "z":
                return f"{word}zes"

        if word.lowered[-2:] == "ze":
            for k, v in pl_sb_ze_zes_bysize.items():
                if word.lowered[-k:] in v:
                    return f"{word}s"

        if word.lowered[-2:] in ("ch", "sh", "zz", "ss") or word.lowered[-1] == "x":
            return f"{word}es"

        # HANDLE ...f -> ...ves

        if word.lowered[-3:] in ("elf", "alf", "olf"):
            return f"{word[:-1]}ves"
        if word.lowered[-3:] == "eaf" and word.lowered[-4:-3] != "d":
            return f"{word[:-1]}ves"
        if word.lowered[-4:] in ("nife", "life", "wife"):
            return f"{word[:-2]}ves"
        if word.lowered[-3:] == "arf":
            return f"{word[:-1]}ves"

        # HANDLE ...y

        if word.lowered[-1] == "y":
            if word.lowered[-2:-1] in "aeiou" or len(word) == 1:
                return f"{word}s"

            if self.classical_dict["names"]:
                if word.lowered[-1] == "y" and word[0] == word[0].upper():
                    return f"{word}s"

            return f"{word[:-1]}ies"

        # HANDLE ...o

        if lowered_last in pl_sb_U_o_os_complete:
            return f"{word}s"

        for k, v in pl_sb_U_o_os_bysize.items():
            if word.lowered[-k:] in v:
                return f"{word}s"

        if word.lowered[-2:] in ("ao", "eo", "io", "oo", "uo"):
            return f"{word}s"

        if word.lowered[-1] == "o":
            return f"{word}es"

        # OTHERWISE JUST ADD ...s

        return f"{word}s"

    @classmethod
    def _handle_prepositional_phrase(cls, phrase, transform, sep):
        """
        Given a word or phrase possibly separated by sep, parse out
        the prepositional phrase and apply the transform to the word
        preceding the prepositional phrase.

        Raise ValueError if the pivot is not found or if at least two
        separators are not found.

        >>> engine._handle_prepositional_phrase("man-of-war", str.upper, '-')
        'MAN-of-war'
        >>> engine._handle_prepositional_phrase("man of war", str.upper, ' ')
        'MAN of war'
        """
        parts = phrase.split(sep)
        if len(parts) < 3:
            raise ValueError("Cannot handle words with fewer than two separators")

        pivot = cls._find_pivot(parts, pl_prep_list_da)

        transformed = transform(parts[pivot - 1]) or parts[pivot - 1]
        return " ".join(
            parts[: pivot - 1] + [sep.join([transformed, parts[pivot], ''])]
        ) + " ".join(parts[(pivot + 1) :])

    def _handle_long_compounds(self, word: Words, count: int) -> Union[str, None]:
        """
        Handles the plural and singular for compound `Words` that
        have three or more words, based on the given count.

        >>> engine()._handle_long_compounds(Words("pair of scissors"), 2)
        'pairs of scissors'
        >>> engine()._handle_long_compounds(Words("men beyond hills"), 1)
        'man beyond hills'
        """
        inflection = self._sinoun if count == 1 else self._plnoun
        solutions = (  # type: ignore
            " ".join(
                itertools.chain(
                    leader,
                    [inflection(cand, count), prep],  # type: ignore
                    trailer,
                )
            )
            for leader, (cand, prep), trailer in windowed_complete(word.split_, 2)
            if prep in pl_prep_list_da  # type: ignore
        )
        return next(solutions, None)

    @staticmethod
    def _find_pivot(words, candidates):
        pivots = (
            index for index in range(1, len(words) - 1) if words[index] in candidates
        )
        try:
            return next(pivots)
        except StopIteration:
            raise ValueError("No pivot found") from None

    def _pl_special_verb(  # noqa: C901
        self, word: str, count: Optional[Union[str, int]] = None
    ) -> Union[str, bool]:
        if self.classical_dict["zero"] and str(count).lower() in pl_count_zero:
            return False
        count = self.get_count(count)

        if count == 1:
            return word

        # HANDLE USER-DEFINED VERBS

        value = self.ud_match(word, self.pl_v_user_defined)
        if value is not None:
            return value

        # HANDLE IRREGULAR PRESENT TENSE (SIMPLE AND COMPOUND)

        try:
            words = Words(word)
        except IndexError:
            return False  # word is ''

        if words.first in plverb_irregular_pres:
            return f"{plverb_irregular_pres[words.first]}{words[len(words.first) :]}"

        # HANDLE IRREGULAR FUTURE, PRETERITE AND PERFECT TENSES

        if words.first in plverb_irregular_non_pres:
            return word

        # HANDLE PRESENT NEGATIONS (SIMPLE AND COMPOUND)

        if words.first.endswith("n't") and words.first[:-3] in plverb_irregular_pres:
            return (
                f"{plverb_irregular_pres[words.first[:-3]]}n't"
                f"{words[len(words.first) :]}"
            )

        if words.first.endswith("n't"):
            return word

        # HANDLE SPECIAL CASES

        mo = PLVERB_SPECIAL_S_RE.search(word)
        if mo:
            return False
        if WHITESPACE.search(word):
            return False

        if words.lowered == "quizzes":
            return "quiz"

        # HANDLE STANDARD 3RD PERSON (CHOP THE ...(e)s OFF SINGLE WORDS)

        if (
            words.lowered[-4:] in ("ches", "shes", "zzes", "sses")
            or words.lowered[-3:] == "xes"
        ):
            return words[:-2]

        if words.lowered[-3:] == "ies" and len(words) > 3:
            return words.lowered[:-3] + "y"

        if (
            words.last.lower() in pl_v_oes_oe
            or words.lowered[-4:] in pl_v_oes_oe_endings_size4
            or words.lowered[-5:] in pl_v_oes_oe_endings_size5
        ):
            return words[:-1]

        if words.lowered.endswith("oes") and len(words) > 3:
            return words.lowered[:-2]

        mo = ENDS_WITH_S.search(words)
        if mo:
            return mo.group(1)

        # OTHERWISE, A REGULAR VERB (HANDLE ELSEWHERE)

        return False

    def _pl_general_verb(
        self, word: str, count: Optional[Union[str, int]] = None
    ) -> str:
        count = self.get_count(count)

        if count == 1:
            return word

        # HANDLE AMBIGUOUS PRESENT TENSES  (SIMPLE AND COMPOUND)

        mo = plverb_ambiguous_pres_keys.search(word)
        if mo:
            return f"{plverb_ambiguous_pres[mo.group(1).lower()]}{mo.group(2)}"

        # HANDLE AMBIGUOUS PRETERITE AND PERFECT TENSES

        mo = plverb_ambiguous_non_pres.search(word)
        if mo:
            return word

        # OTHERWISE, 1st OR 2ND PERSON IS UNINFLECTED

        return word

    def _pl_special_adjective(
        self, word: str, count: Optional[Union[str, int]] = None
    ) -> Union[str, bool]:
        count = self.get_count(count)

        if count == 1:
            return word

        # HANDLE USER-DEFINED ADJECTIVES

        value = self.ud_match(word, self.pl_adj_user_defined)
        if value is not None:
            return value

        # HANDLE KNOWN CASES

        mo = pl_adj_special_keys.search(word)
        if mo:
            return pl_adj_special[mo.group(1).lower()]

        # HANDLE POSSESSIVES

        mo = pl_adj_poss_keys.search(word)
        if mo:
            return pl_adj_poss[mo.group(1).lower()]

        mo = ENDS_WITH_APOSTROPHE_S.search(word)
        if mo:
            pl = self.plural_noun(mo.group(1))
            trailing_s = "" if pl[-1] == "s" else "s"
            return f"{pl}'{trailing_s}"

        # OTHERWISE, NO IDEA

        return False

    # @profile
    def _sinoun(  # noqa: C901
        self,
        word: str,
        count: Optional[Union[str, int]] = None,
        gender: Optional[str] = None,
    ) -> Union[str, bool]:
        count = self.get_count(count)

        # DEFAULT TO PLURAL

        if count == 2:
            return word

        # SET THE GENDER

        try:
            if gender is None:
                gender = self.thegender
            elif gender not in singular_pronoun_genders:
                raise BadGenderError
        except (TypeError, IndexError) as err:
            raise BadGenderError from err

        # HANDLE USER-DEFINED NOUNS

        value = self.ud_match(word, self.si_sb_user_defined)
        if value is not None:
            return value

        # HANDLE EMPTY WORD, SINGULAR COUNT AND UNINFLECTED PLURALS

        if word == "":
            return word

        if word in si_sb_ois_oi_case:
            return word[:-1]

        words = Words(word)

        if words.last.lower() in pl_sb_uninflected_complete:
            if len(words.split_) >= 3:
                return self._handle_long_compounds(words, count=1) or word
            return word

        if word in pl_sb_uninflected_caps:
            return word

        for k, v in pl_sb_uninflected_bysize.items():
            if words.lowered[-k:] in v:
                return word

        if self.classical_dict["herd"] and words.last.lower() in pl_sb_uninflected_herd:
            return word

        if words.last.lower() in pl_sb_C_us_us:
            return word if self.classical_dict["ancient"] else False

        # HANDLE COMPOUNDS ("Governor General", "mother-in-law", "aide-de-camp", ETC.)

        mo = PL_SB_POSTFIX_ADJ_STEMS_RE.search(word)
        if mo and mo.group(2) != "":
            return f"{self._sinoun(mo.group(1), 1, gender=gender)}{mo.group(2)}"

        with contextlib.suppress(ValueError):
            return self._handle_prepositional_phrase(
                words.lowered,
                functools.partial(self._sinoun, count=1, gender=gender),
                ' ',
            )

        with contextlib.suppress(ValueError):
            return self._handle_prepositional_phrase(
                words.lowered,
                functools.partial(self._sinoun, count=1, gender=gender),
                '-',
            )

        # HANDLE PRONOUNS

        for k, v in si_pron_acc_keys_bysize.items():
            if words.lowered[-k:] in v:  # ends with accusative pronoun
                for pk, pv in pl_prep_bysize.items():
                    if words.lowered[:pk] in pv:  # starts with a prep
                        if words.lowered.split() == [
                            words.lowered[:pk],
                            words.lowered[-k:],
                        ]:
                            # only whitespace in between
                            return words.lowered[:-k] + get_si_pron(
                                "acc", words.lowered[-k:], gender
                            )

        try:
            return get_si_pron("nom", words.lowered, gender)
        except KeyError:
            pass

        try:
            return get_si_pron("acc", words.lowered, gender)
        except KeyError:
            pass

        # HANDLE ISOLATED IRREGULAR PLURALS

        if words.last in si_sb_irregular_caps:
            llen = len(words.last)
            return f"{word[:-llen]}{si_sb_irregular_caps[words.last]}"

        if words.last.lower() in si_sb_irregular:
            llen = len(words.last.lower())
            return f"{word[:-llen]}{si_sb_irregular[words.last.lower()]}"

        dash_split = words.lowered.split("-")
        if (" ".join(dash_split[-2:])).lower() in si_sb_irregular_compound:
            llen = len(
                " ".join(dash_split[-2:])
            )  # TODO: what if 2 spaces between these words?
            return "{}{}".format(
                word[:-llen],
                si_sb_irregular_compound[(" ".join(dash_split[-2:])).lower()],
            )

        if words.lowered[-5:] == "quies":
            return word[:-3] + "y"

        if words.lowered[-7:] == "persons":
            return word[:-1]
        if words.lowered[-6:] == "people":
            return word[:-4] + "rson"

        # HANDLE FAMILIES OF IRREGULAR PLURALS

        if words.lowered[-4:] == "mans":
            for k, v in si_sb_U_man_mans_bysize.items():
                if words.lowered[-k:] in v:
                    return word[:-1]
            for k, v in si_sb_U_man_mans_caps_bysize.items():
                if word[-k:] in v:
                    return word[:-1]
        if words.lowered[-3:] == "men":
            return word[:-3] + "man"
        if words.lowered[-4:] == "mice":
            return word[:-4] + "mouse"
        if words.lowered[-4:] == "lice":
            v = si_sb_U_louse_lice_bysize.get(len(word))
            if v and words.lowered in v:
                return word[:-4] + "louse"
        if words.lowered[-5:] == "geese":
            return word[:-5] + "goose"
        if words.lowered[-5:] == "teeth":
            return word[:-5] + "tooth"
        if words.lowered[-4:] == "feet":
            return word[:-4] + "foot"

        if words.lowered == "dice":
            return "die"

        # HANDLE UNASSIMILATED IMPORTS

        if words.lowered[-4:] == "ceps":
            return word
        if words.lowered[-3:] == "zoa":
            return word[:-1] + "on"

        for lastlet, d, unass_numend, post in (
            ("s", si_sb_U_ch_chs_bysize, -1, ""),
            ("s", si_sb_U_ex_ices_bysize, -4, "ex"),
            ("s", si_sb_U_ix_ices_bysize, -4, "ix"),
            ("a", si_sb_U_um_a_bysize, -1, "um"),
            ("i", si_sb_U_us_i_bysize, -1, "us"),
            ("a", si_sb_U_on_a_bysize, -1, "on"),
            ("e", si_sb_U_a_ae_bysize, -1, ""),
        ):
            if words.lowered[-1] == lastlet:  # this test to add speed
                for k, v in d.items():
                    if words.lowered[-k:] in v:
                        return word[:unass_numend] + post

        # HANDLE INCOMPLETELY ASSIMILATED IMPORTS

        if self.classical_dict["ancient"]:
            if words.lowered[-6:] == "trices":
                return word[:-3] + "x"
            if words.lowered[-4:] in ("eaux", "ieux"):
                return word[:-1]
            if words.lowered[-5:] in ("ynges", "inges", "anges") and len(word) > 6:
                return word[:-3] + "x"

            for lastlet, d, class_numend, post in (
                ("a", si_sb_C_en_ina_bysize, -3, "en"),
                ("s", si_sb_C_ex_ices_bysize, -4, "ex"),
                ("s", si_sb_C_ix_ices_bysize, -4, "ix"),
                ("a", si_sb_C_um_a_bysize, -1, "um"),
                ("i", si_sb_C_us_i_bysize, -1, "us"),
                ("s", pl_sb_C_us_us_bysize, None, ""),
                ("e", si_sb_C_a_ae_bysize, -1, ""),
                ("a", si_sb_C_a_ata_bysize, -2, ""),
                ("s", si_sb_C_is_ides_bysize, -3, "s"),
                ("i", si_sb_C_o_i_bysize, -1, "o"),
                ("a", si_sb_C_on_a_bysize, -1, "on"),
                ("m", si_sb_C_im_bysize, -2, ""),
                ("i", si_sb_C_i_bysize, -1, ""),
            ):
                if words.lowered[-1] == lastlet:  # this test to add speed
                    for k, v in d.items():
                        if words.lowered[-k:] in v:
                            return word[:class_numend] + post

        # HANDLE PLURLS ENDING IN uses -> use

        if (
            words.lowered[-6:] == "houses"
            or word in si_sb_uses_use_case
            or words.last.lower() in si_sb_uses_use
        ):
            return word[:-1]

        # HANDLE PLURLS ENDING IN ies -> ie

        if word in si_sb_ies_ie_case or words.last.lower() in si_sb_ies_ie:
            return word[:-1]

        # HANDLE PLURLS ENDING IN oes -> oe

        if (
            words.lowered[-5:] == "shoes"
            or word in si_sb_oes_oe_case
            or words.last.lower() in si_sb_oes_oe
        ):
            return word[:-1]

        # HANDLE SINGULAR NOUNS ENDING IN ...s OR OTHER SILIBANTS

        if word in si_sb_sses_sse_case or words.last.lower() in si_sb_sses_sse:
            return word[:-1]

        if words.last.lower() in si_sb_singular_s_complete:
            return word[:-2]

        for k, v in si_sb_singular_s_bysize.items():
            if words.lowered[-k:] in v:
                return word[:-2]

        if words.lowered[-4:] == "eses" and word[0] == word[0].upper():
            return word[:-2]

        if words.last.lower() in si_sb_z_zes:
            return word[:-2]

        if words.last.lower() in si_sb_zzes_zz:
            return word[:-2]

        if words.lowered[-4:] == "zzes":
            return word[:-3]

        if word in si_sb_ches_che_case or words.last.lower() in si_sb_ches_che:
            return word[:-1]

        if words.lowered[-4:] in ("ches", "shes"):
            return word[:-2]

        if words.last.lower() in si_sb_xes_xe:
            return word[:-1]

        if words.lowered[-3:] == "xes":
            return word[:-2]

        # HANDLE ...f -> ...ves

        if word in si_sb_ves_ve_case or words.last.lower() in si_sb_ves_ve:
            return word[:-1]

        if words.lowered[-3:] == "ves":
            if words.lowered[-5:-3] in ("el", "al", "ol"):
                return word[:-3] + "f"
            if words.lowered[-5:-3] == "ea" and word[-6:-5] != "d":
                return word[:-3] + "f"
            if words.lowered[-5:-3] in ("ni", "li", "wi"):
                return word[:-3] + "fe"
            if words.lowered[-5:-3] == "ar":
                return word[:-3] + "f"

        # HANDLE ...y

        if words.lowered[-2:] == "ys":
            if len(words.lowered) > 2 and words.lowered[-3] in "aeiou":
                return word[:-1]

            if self.classical_dict["names"]:
                if words.lowered[-2:] == "ys" and word[0] == word[0].upper():
                    return word[:-1]

        if words.lowered[-3:] == "ies":
            return word[:-3] + "y"

        # HANDLE ...o

        if words.lowered[-2:] == "os":
            if words.last.lower() in si_sb_U_o_os_complete:
                return word[:-1]

            for k, v in si_sb_U_o_os_bysize.items():
                if words.lowered[-k:] in v:
                    return word[:-1]

            if words.lowered[-3:] in ("aos", "eos", "ios", "oos", "uos"):
                return word[:-1]

        if words.lowered[-3:] == "oes":
            return word[:-2]

        # UNASSIMILATED IMPORTS FINAL RULE

        if word in si_sb_es_is:
            return word[:-2] + "is"

        # OTHERWISE JUST REMOVE ...s

        if words.lowered[-1] == "s":
            return word[:-1]

        # COULD NOT FIND SINGULAR

        return False

    # ADJECTIVES

    @typechecked
    def a(self, text: Word, count: Optional[Union[int, str, Any]] = 1) -> str:
        """
        Return the appropriate indefinite article followed by text.

        The indefinite article is either 'a' or 'an'.

        If count is not one, then return count followed by text
        instead of 'a' or 'an'.

        Whitespace at the start and end is preserved.

        """
        mo = INDEFINITE_ARTICLE_TEST.search(text)
        if mo:
            word = mo.group(2)
            if not word:
                return text
            pre = mo.group(1)
            post = mo.group(3)
            result = self._indef_article(word, count)
            return f"{pre}{result}{post}"
        return ""

    an = a

    _indef_article_cases = (
        # HANDLE ORDINAL FORMS
        (A_ordinal_a, "a"),
        (A_ordinal_an, "an"),
        # HANDLE SPECIAL CASES
        (A_explicit_an, "an"),
        (SPECIAL_AN, "an"),
        (SPECIAL_A, "a"),
        # HANDLE ABBREVIATIONS
        (A_abbrev, "an"),
        (SPECIAL_ABBREV_AN, "an"),
        (SPECIAL_ABBREV_A, "a"),
        # HANDLE CONSONANTS
        (CONSONANTS, "a"),
        # HANDLE SPECIAL VOWEL-FORMS
        (ARTICLE_SPECIAL_EU, "a"),
        (ARTICLE_SPECIAL_ONCE, "a"),
        (ARTICLE_SPECIAL_ONETIME, "a"),
        (ARTICLE_SPECIAL_UNIT, "a"),
        (ARTICLE_SPECIAL_UBA, "a"),
        (ARTICLE_SPECIAL_UKR, "a"),
        (A_explicit_a, "a"),
        # HANDLE SPECIAL CAPITALS
        (SPECIAL_CAPITALS, "a"),
        # HANDLE VOWELS
        (VOWELS, "an"),
        # HANDLE y...
        # (BEFORE CERTAIN CONSONANTS IMPLIES (UNNATURALIZED) "i.." SOUND)
        (A_y_cons, "an"),
    )

    def _indef_article(self, word: str, count: Union[int, str, Any]) -> str:
        mycount = self.get_count(count)

        if mycount != 1:
            return f"{count} {word}"

        # HANDLE USER-DEFINED VARIANTS

        value = self.ud_match(word, self.A_a_user_defined)
        if value is not None:
            return f"{value} {word}"

        matches = (
            f'{article} {word}'
            for regexen, article in self._indef_article_cases
            if regexen.search(word)
        )

        # OTHERWISE, GUESS "a"
        fallback = f'a {word}'
        return next(matches, fallback)

    # 2. TRANSLATE ZERO-QUANTIFIED $word TO "no plural($word)"

    @typechecked
    def no(self, text: Word, count: Optional[Union[int, str]] = None) -> str:
        """
        If count is 0, no, zero or nil, return 'no' followed by the plural
        of text.

        If count is one of:
            1, a, an, one, each, every, this, that
            return count followed by text.

        Otherwise return count follow by the plural of text.

        In the return value count is always followed by a space.

        Whitespace at the start and end is preserved.

        """
        if count is None and self.persistent_count is not None:
            count = self.persistent_count

        if count is None:
            count = 0
        mo = PARTITION_WORD.search(text)
        if mo:
            pre = mo.group(1)
            word = mo.group(2)
            post = mo.group(3)
        else:
            pre = ""
            word = ""
            post = ""

        if str(count).lower() in pl_count_zero:
            count = 'no'
        return f"{pre}{count} {self.plural(word, count)}{post}"

    # PARTICIPLES

    @typechecked
    def present_participle(self, word: Word) -> str:
        """
        Return the present participle for word.

        word is the 3rd person singular verb.

        """
        plv = self.plural_verb(word, 2)
        ans = plv

        for regexen, repl in PRESENT_PARTICIPLE_REPLACEMENTS:
            ans, num = regexen.subn(repl, plv)
            if num:
                return f"{ans}ing"
        return f"{ans}ing"

    # NUMERICAL INFLECTIONS

    @typechecked
    def ordinal(self, num: Union[Number, Word]) -> str:
        """
        Return the ordinal of num.

        >>> ordinal = engine().ordinal
        >>> ordinal(1)
        '1st'
        >>> ordinal('one')
        'first'
        """
        if DIGIT.match(str(num)):
            if isinstance(num, (float, int)) and int(num) == num:
                n = int(num)
            else:
                if "." in str(num):
                    try:
                        # numbers after decimal,
                        # so only need last one for ordinal
                        n = int(str(num)[-1])

                    except ValueError:  # ends with '.', so need to use whole string
                        n = int(str(num)[:-1])
                else:
                    n = int(num)  # type: ignore
            try:
                post = nth[n % 100]
            except KeyError:
                post = nth[n % 10]
            return f"{num}{post}"
        else:
            return self._sub_ord(num)

    def millfn(self, ind: int = 0) -> str:
        if ind > len(mill) - 1:
            raise NumOutOfRangeError
        return mill[ind]

    def unitfn(self, units: int, mindex: int = 0) -> str:
        return f"{unit[units]}{self.millfn(mindex)}"

    def tenfn(self, tens, units, mindex=0) -> str:
        if tens != 1:
            tens_part = ten[tens]
            if tens and units:
                hyphen = "-"
            else:
                hyphen = ""
            unit_part = unit[units]
            mill_part = self.millfn(mindex)
            return f"{tens_part}{hyphen}{unit_part}{mill_part}"
        return f"{teen[units]}{mill[mindex]}"

    def hundfn(self, hundreds: int, tens: int, units: int, mindex: int) -> str:
        if hundreds:
            andword = f" {self._number_args['andword']} " if tens or units else ""
            # use unit not unitfn as simpler
            return (
                f"{unit[hundreds]} hundred{andword}"
                f"{self.tenfn(tens, units)}{self.millfn(mindex)}, "
            )
        if tens or units:
            return f"{self.tenfn(tens, units)}{self.millfn(mindex)}, "
        return ""

    def group1sub(self, mo: Match) -> str:
        units = int(mo.group(1))
        if units == 1:
            return f" {self._number_args['one']}, "
        elif units:
            return f"{unit[units]}, "
        else:
            return f" {self._number_args['zero']}, "

    def group1bsub(self, mo: Match) -> str:
        units = int(mo.group(1))
        if units:
            return f"{unit[units]}, "
        else:
            return f" {self._number_args['zero']}, "

    def group2sub(self, mo: Match) -> str:
        tens = int(mo.group(1))
        units = int(mo.group(2))
        if tens:
            return f"{self.tenfn(tens, units)}, "
        if units:
            return f" {self._number_args['zero']} {unit[units]}, "
        return f" {self._number_args['zero']} {self._number_args['zero']}, "

    def group3sub(self, mo: Match) -> str:
        hundreds = int(mo.group(1))
        tens = int(mo.group(2))
        units = int(mo.group(3))
        if hundreds == 1:
            hunword = f" {self._number_args['one']}"
        elif hundreds:
            hunword = str(unit[hundreds])
        else:
            hunword = f" {self._number_args['zero']}"
        if tens:
            tenword = self.tenfn(tens, units)
        elif units:
            tenword = f" {self._number_args['zero']} {unit[units]}"
        else:
            tenword = f" {self._number_args['zero']} {self._number_args['zero']}"
        return f"{hunword} {tenword}, "

    def hundsub(self, mo: Match) -> str:
        ret = self.hundfn(
            int(mo.group(1)), int(mo.group(2)), int(mo.group(3)), self.mill_count
        )
        self.mill_count += 1
        return ret

    def tensub(self, mo: Match) -> str:
        return f"{self.tenfn(int(mo.group(1)), int(mo.group(2)), self.mill_count)}, "

    def unitsub(self, mo: Match) -> str:
        return f"{self.unitfn(int(mo.group(1)), self.mill_count)}, "

    def enword(self, num: str, group: int) -> str:
        # import pdb
        # pdb.set_trace()

        if group == 1:
            num = DIGIT_GROUP.sub(self.group1sub, num)
        elif group == 2:
            num = TWO_DIGITS.sub(self.group2sub, num)
            num = DIGIT_GROUP.sub(self.group1bsub, num, 1)
        elif group == 3:
            num = THREE_DIGITS.sub(self.group3sub, num)
            num = TWO_DIGITS.sub(self.group2sub, num, 1)
            num = DIGIT_GROUP.sub(self.group1sub, num, 1)
        elif int(num) == 0:
            num = self._number_args["zero"]
        elif int(num) == 1:
            num = self._number_args["one"]
        else:
            num = num.lstrip().lstrip("0")
            self.mill_count = 0
            # surely there's a better way to do the next bit
            mo = THREE_DIGITS_WORD.search(num)
            while mo:
                num = THREE_DIGITS_WORD.sub(self.hundsub, num, 1)
                mo = THREE_DIGITS_WORD.search(num)
            num = TWO_DIGITS_WORD.sub(self.tensub, num, 1)
            num = ONE_DIGIT_WORD.sub(self.unitsub, num, 1)
        return num

    @staticmethod
    def _sub_ord(val):
        new = ordinal_suff.sub(lambda match: ordinal[match.group(1)], val)
        return new + "th" * (new == val)

    @classmethod
    def _chunk_num(cls, num, decimal, group):
        if decimal:
            max_split = -1 if group != 0 else 1
            chunks = num.split(".", max_split)
        else:
            chunks = [num]
        return cls._remove_last_blank(chunks)

    @staticmethod
    def _remove_last_blank(chunks):
        """
        Remove the last item from chunks if it's a blank string.

        Return the resultant chunks and whether the last item was removed.
        """
        removed = chunks[-1] == ""
        result = chunks[:-1] if removed else chunks
        return result, removed

    @staticmethod
    def _get_sign(num):
        return {'+': 'plus', '-': 'minus'}.get(num.lstrip()[0], '')

    @typechecked
    def number_to_words(  # noqa: C901
        self,
        num: Union[Number, Word],
        wantlist: bool = False,
        group: int = 0,
        comma: Union[Falsish, str] = ",",
        andword: str = "and",
        zero: str = "zero",
        one: str = "one",
        decimal: Union[Falsish, str] = "point",
        threshold: Optional[int] = None,
    ) -> Union[str, List[str]]:
        """
        Return a number in words.

        group = 1, 2 or 3 to group numbers before turning into words
        comma: define comma

        andword:
            word for 'and'. Can be set to ''.
            e.g. "one hundred and one" vs "one hundred one"

        zero: word for '0'
        one: word for '1'
        decimal: word for decimal point
        threshold: numbers above threshold not turned into words

        parameters not remembered from last call. Departure from Perl version.
        """
        self._number_args = {"andword": andword, "zero": zero, "one": one}
        num = str(num)

        # Handle "stylistic" conversions (up to a given threshold)...
        if threshold is not None and float(num) > threshold:
            spnum = num.split(".", 1)
            while comma:
                (spnum[0], n) = FOUR_DIGIT_COMMA.subn(r"\1,\2", spnum[0])
                if n == 0:
                    break
            try:
                return f"{spnum[0]}.{spnum[1]}"
            except IndexError:
                return str(spnum[0])

        if group < 0 or group > 3:
            raise BadChunkingOptionError

        sign = self._get_sign(num)

        if num in nth_suff:
            num = zero

        myord = num[-2:] in nth_suff
        if myord:
            num = num[:-2]

        chunks, finalpoint = self._chunk_num(num, decimal, group)

        loopstart = chunks[0] == ""
        first: bool | None = not loopstart

        def _handle_chunk(chunk):
            nonlocal first

            # remove all non numeric \D
            chunk = NON_DIGIT.sub("", chunk)
            if chunk == "":
                chunk = "0"

            if group == 0 and not first:
                chunk = self.enword(chunk, 1)
            else:
                chunk = self.enword(chunk, group)

            if chunk[-2:] == ", ":
                chunk = chunk[:-2]
            chunk = WHITESPACES_COMMA.sub(",", chunk)

            if group == 0 and first:
                chunk = COMMA_WORD.sub(f" {andword} \\1", chunk)
            chunk = WHITESPACES.sub(" ", chunk)
            # chunk = re.sub(r"(\A\s|\s\Z)", self.blankfn, chunk)
            chunk = chunk.strip()
            if first:
                first = None
            return chunk

        chunks[loopstart:] = map(_handle_chunk, chunks[loopstart:])

        numchunks = []
        if first != 0:
            numchunks = chunks[0].split(f"{comma} ")

        if myord and numchunks:
            numchunks[-1] = self._sub_ord(numchunks[-1])

        for chunk in chunks[1:]:
            numchunks.append(decimal)
            numchunks.extend(chunk.split(f"{comma} "))

        if finalpoint:
            numchunks.append(decimal)

        if wantlist:
            return [sign] * bool(sign) + numchunks

        signout = f"{sign} " if sign else ""
        valout = (
            ', '.join(numchunks)
            if group
            else ''.join(self._render(numchunks, decimal, comma))
        )
        return signout + valout

    @staticmethod
    def _render(chunks, decimal, comma):
        first_item = chunks.pop(0)
        yield first_item
        first = decimal is None or not first_item.endswith(decimal)
        for nc in chunks:
            if nc == decimal:
                first = False
            elif first:
                yield comma
            yield f" {nc}"

    @typechecked
    def join(
        self,
        words: Optional[Sequence[Word]],
        sep: Optional[str] = None,
        sep_spaced: bool = True,
        final_sep: Optional[str] = None,
        conj: str = "and",
        conj_spaced: bool = True,
    ) -> str:
        """
        Join words into a list.

        e.g. join(['ant', 'bee', 'fly']) returns 'ant, bee, and fly'

        options:
        conj: replacement for 'and'
        sep: separator. default ',', unless ',' is in the list then ';'
        final_sep: final separator. default ',', unless ',' is in the list then ';'
        conj_spaced: boolean. Should conj have spaces around it

        """
        if not words:
            return ""
        if len(words) == 1:
            return words[0]

        if conj_spaced:
            if conj == "":
                conj = " "
            else:
                conj = f" {conj} "

        if len(words) == 2:
            return f"{words[0]}{conj}{words[1]}"

        if sep is None:
            if "," in "".join(words):
                sep = ";"
            else:
                sep = ","
        if final_sep is None:
            final_sep = sep

        final_sep = f"{final_sep}{conj}"

        if sep_spaced:
            sep += " "

        return f"{sep.join(words[0:-1])}{final_sep}{words[-1]}"
