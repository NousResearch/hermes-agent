"""Friendly display names for delegated subagents (#118081).

The machine identity ``sa-{task_index}-{hex}`` stays the canonical routing key
for steering, stopping, ownership, logs and APIs. The display name assigned here
is presentation metadata only: given once at spawn, stable for the child's
lifetime, and unique among concurrently-live children so activity surfaces can
say "Steered Hypatia" instead of "Delegating steer sa-2-73b48efe". Surfaces that
lack a name (older events, pre-upgrade gateways) keep falling back to the raw
identifier.
"""

from typing import Iterable, Optional

# Curated, short, distinct pool in stable order (allocation prefers the
# earliest free name). Historical scientists and mathematicians across
# cultures; single-token names so they read cleanly next to a role label.
_FRIENDLY_NAMES: tuple = (
    "Hypatia", "Turing", "Lovelace", "Noether", "Ramanujan", "Curie",
    "AlKhwarizmi", "Fibonacci", "Alhazen", "Shen", "Aryabhata", "Omar",
    "Copernicus", "Brahe", "Kepler", "Galileo", "Huygens", "Newton",
    "Leibniz", "Euler", "Lagrange", "Laplace", "Gauss", "Cauchy",
    "Lobachevsky", "Boole", "Riemann", "Maxwell", "Kovalevskaya",
    "Boltzmann", "Cantor", "Tesla", "Planck", "Meitner", "Fermat",
    "Nightingale", "Somerville", "Lamarr", "Hopper", "Chien-Shiung",
    "Franklin", "Vera", "Rubin", "Jocelyn", "Katherine", "Dorothy",
    "Maryam", "Sofia", "Émilie", "Caroline", "Annie", "Henrietta",
)

# Names are unique only among concurrently-live children; reuse after a child
# exits is intended (the pool is small on purpose — recognition beats novelty
# for a small team of agents).


def assign_display_name(taken: Optional[Iterable] = None) -> str:
    """First pool name not in ``taken``, else ``"<name> 2"``-style suffixes.

    Deterministic and allocation-order-stable so a given spawn order yields
    the same names across runs — activity comparisons and tests stay
    reproducible.
    """
    busy = {t for t in (taken or ()) if t}
    for name in _FRIENDLY_NAMES:
        if name not in busy:
            return name
    round_ = 2
    while True:
        for name in _FRIENDLY_NAMES:
            candidate = f"{name} {round_}"
            if candidate not in busy:
                return candidate
        round_ += 1
