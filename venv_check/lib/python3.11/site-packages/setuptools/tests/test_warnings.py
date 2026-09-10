from inspect import cleandoc

import pytest

from setuptools.warnings import SetuptoolsDeprecationWarning, SetuptoolsWarning

_EXAMPLES = {
    "default": dict(
        args=("Hello {x}", "\n\t{target} {v:.1f}"),
        kwargs={"x": 5, "v": 3, "target": "World"},
        expected="""
    Hello 5
    !!

            ********************************************************************************
            World 3.0
            ********************************************************************************

    !!
    """,
    ),
    "futue_due_date": dict(
        args=("Summary", "Lorem ipsum"),
        kwargs={"due_date": (9999, 11, 22)},
        expected="""
    Summary
    !!

            ********************************************************************************
            Lorem ipsum

            By 9999-Nov-22, you need to update your project and remove deprecated calls
            or your builds will no longer be supported.
            ********************************************************************************

    !!
    """,
    ),
    "past_due_date_with_docs": dict(
        args=("Summary", "Lorem ipsum"),
        kwargs={"due_date": (2000, 11, 22), "see_docs": "some_page.html"},
        expected="""
    Summary
    !!

            ********************************************************************************
            Lorem ipsum

            This deprecation is overdue, please update your project and remove deprecated
            calls to avoid build errors in the future.

            See https://setuptools.pypa.io/en/latest/some_page.html for details.
            ********************************************************************************

    !!
    """,
    ),
}


@pytest.mark.parametrize("example_name", _EXAMPLES.keys())
def test_formatting(monkeypatch, example_name):
    """
    It should automatically handle indentation, interpolation and things like due date.
    """
    args = _EXAMPLES[example_name]["args"]
    kwargs = _EXAMPLES[example_name]["kwargs"]
    expected = _EXAMPLES[example_name]["expected"]

    monkeypatch.setenv("SETUPTOOLS_ENFORCE_DEPRECATION", "false")
    with pytest.warns(SetuptoolsWarning) as warn_info:
        SetuptoolsWarning.emit(*args, **kwargs)
    assert _get_message(warn_info) == cleandoc(expected)


def test_due_date_enforcement(monkeypatch):
    class _MyDeprecation(SetuptoolsDeprecationWarning):
        _SUMMARY = "Summary"
        _DETAILS = "Lorem ipsum"
        _DUE_DATE = (2000, 11, 22)
        _SEE_DOCS = "some_page.html"

    monkeypatch.setenv("SETUPTOOLS_ENFORCE_DEPRECATION", "true")
    with pytest.raises(SetuptoolsDeprecationWarning) as exc_info:
        _MyDeprecation.emit()

    expected = """
    Summary
    !!

            ********************************************************************************
            Lorem ipsum

            This deprecation is overdue, please update your project and remove deprecated
            calls to avoid build errors in the future.

            See https://setuptools.pypa.io/en/latest/some_page.html for details.
            ********************************************************************************

    !!
    """
    assert str(exc_info.value) == cleandoc(expected)


def _get_message(warn_info):
    return next(warn.message.args[0] for warn in warn_info)
