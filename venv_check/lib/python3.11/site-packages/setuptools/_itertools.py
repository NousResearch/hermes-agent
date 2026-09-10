from more_itertools import consume  # noqa: F401


# copied from jaraco.itertools 6.1
def ensure_unique(iterable, key=lambda x: x):
    """
    Wrap an iterable to raise a ValueError if non-unique values are encountered.

    >>> list(ensure_unique('abc'))
    ['a', 'b', 'c']
    >>> consume(ensure_unique('abca'))
    Traceback (most recent call last):
    ...
    ValueError: Duplicate element 'a' encountered.
    """
    seen = set()
    seen_add = seen.add
    for element in iterable:
        k = key(element)
        if k in seen:
            raise ValueError(f"Duplicate element {element!r} encountered.")
        seen_add(k)
        yield element
