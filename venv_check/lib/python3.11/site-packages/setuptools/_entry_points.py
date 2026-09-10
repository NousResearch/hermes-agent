import functools
import itertools
import operator

from jaraco.functools import pass_none
from jaraco.text import yield_lines
from more_itertools import consume

from ._importlib import metadata
from ._itertools import ensure_unique
from .errors import OptionError


def ensure_valid(ep):
    """
    Exercise one of the dynamic properties to trigger
    the pattern match.
    """
    try:
        ep.extras
    except (AttributeError, AssertionError) as ex:
        # Why both? See https://github.com/python/importlib_metadata/issues/488
        msg = (
            f"Problems to parse {ep}.\nPlease ensure entry-point follows the spec: "
            "https://packaging.python.org/en/latest/specifications/entry-points/"
        )
        raise OptionError(msg) from ex


def load_group(value, group):
    """
    Given a value of an entry point or series of entry points,
    return each as an EntryPoint.
    """
    # normalize to a single sequence of lines
    lines = yield_lines(value)
    text = f'[{group}]\n' + '\n'.join(lines)
    return metadata.EntryPoints._from_text(text)


def by_group_and_name(ep):
    return ep.group, ep.name


def validate(eps: metadata.EntryPoints):
    """
    Ensure entry points are unique by group and name and validate each.
    """
    consume(map(ensure_valid, ensure_unique(eps, key=by_group_and_name)))
    return eps


@functools.singledispatch
def load(eps):
    """
    Given a Distribution.entry_points, produce EntryPoints.
    """
    groups = itertools.chain.from_iterable(
        load_group(value, group) for group, value in eps.items()
    )
    return validate(metadata.EntryPoints(groups))


@load.register(str)
def _(eps):
    r"""
    >>> ep, = load('[console_scripts]\nfoo=bar')
    >>> ep.group
    'console_scripts'
    >>> ep.name
    'foo'
    >>> ep.value
    'bar'
    """
    return validate(metadata.EntryPoints(metadata.EntryPoints._from_text(eps)))


load.register(type(None), lambda x: x)


@pass_none
def render(eps: metadata.EntryPoints):
    by_group = operator.attrgetter('group')
    groups = itertools.groupby(sorted(eps, key=by_group), by_group)

    return '\n'.join(f'[{group}]\n{render_items(items)}\n' for group, items in groups)


def render_items(eps):
    return '\n'.join(f'{ep.name} = {ep.value}' for ep in sorted(eps))
