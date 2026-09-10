"""
Compatibility layer with Python 3.8/3.9
"""

from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:  # pragma: no cover
    # Prevent circular imports on runtime.
    from .. import Distribution, EntryPoint
else:
    Distribution = EntryPoint = Any


def normalized_name(dist: Distribution) -> Optional[str]:
    """
    Honor name normalization for distributions that don't provide ``_normalized_name``.
    """
    try:
        return dist._normalized_name
    except AttributeError:
        from .. import Prepared  # -> delay to prevent circular imports.

        return Prepared.normalize(getattr(dist, "name", None) or dist.metadata['Name'])


def ep_matches(ep: EntryPoint, **params) -> bool:
    """
    Workaround for ``EntryPoint`` objects without the ``matches`` method.
    """
    try:
        return ep.matches(**params)
    except AttributeError:
        from .. import EntryPoint  # -> delay to prevent circular imports.

        # Reconstruct the EntryPoint object to make sure it is compatible.
        return EntryPoint(ep.name, ep.value, ep.group).matches(**params)
