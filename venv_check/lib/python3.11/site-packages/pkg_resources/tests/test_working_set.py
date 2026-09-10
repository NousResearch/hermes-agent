import functools
import inspect
import re
import textwrap

import pytest

import pkg_resources

from .test_resources import Metadata


def strip_comments(s):
    return '\n'.join(
        line
        for line in s.split('\n')
        if line.strip() and not line.strip().startswith('#')
    )


def parse_distributions(s):
    """
    Parse a series of distribution specs of the form:
    {project_name}-{version}
       [optional, indented requirements specification]

    Example:

        foo-0.2
        bar-1.0
          foo>=3.0
          [feature]
          baz

    yield 2 distributions:
        - project_name=foo, version=0.2
        - project_name=bar, version=1.0,
          requires=['foo>=3.0', 'baz; extra=="feature"']
    """
    s = s.strip()
    for spec in re.split(r'\n(?=[^\s])', s):
        if not spec:
            continue
        fields = spec.split('\n', 1)
        assert 1 <= len(fields) <= 2
        name, version = fields.pop(0).rsplit('-', 1)
        if fields:
            requires = textwrap.dedent(fields.pop(0))
            metadata = Metadata(('requires.txt', requires))
        else:
            metadata = None
        dist = pkg_resources.Distribution(
            project_name=name, version=version, metadata=metadata
        )
        yield dist


class FakeInstaller:
    def __init__(self, installable_dists) -> None:
        self._installable_dists = installable_dists

    def __call__(self, req):
        return next(
            iter(filter(lambda dist: dist in req, self._installable_dists)), None
        )


def parametrize_test_working_set_resolve(*test_list):
    idlist = []
    argvalues = []
    for test in test_list:
        (
            name,
            installed_dists,
            installable_dists,
            requirements,
            expected1,
            expected2,
        ) = (
            strip_comments(s.lstrip())
            for s in textwrap.dedent(test).lstrip().split('\n\n', 5)
        )
        installed_dists = list(parse_distributions(installed_dists))
        installable_dists = list(parse_distributions(installable_dists))
        requirements = list(pkg_resources.parse_requirements(requirements))
        for id_, replace_conflicting, expected in (
            (name, False, expected1),
            (name + '_replace_conflicting', True, expected2),
        ):
            idlist.append(id_)
            expected = strip_comments(expected.strip())
            if re.match(r'\w+$', expected):
                expected = getattr(pkg_resources, expected)
                assert issubclass(expected, Exception)
            else:
                expected = list(parse_distributions(expected))
            argvalues.append(
                pytest.param(
                    installed_dists,
                    installable_dists,
                    requirements,
                    replace_conflicting,
                    expected,
                )
            )
    return pytest.mark.parametrize(
        (
            "installed_dists",
            "installable_dists",
            "requirements",
            "replace_conflicting",
            "resolved_dists_or_exception",
        ),
        argvalues,
        ids=idlist,
    )


@parametrize_test_working_set_resolve(
    """
    # id
    noop

    # installed

    # installable

    # wanted

    # resolved

    # resolved [replace conflicting]
    """,
    """
    # id
    already_installed

    # installed
    foo-3.0

    # installable

    # wanted
    foo>=2.1,!=3.1,<4

    # resolved
    foo-3.0

    # resolved [replace conflicting]
    foo-3.0
    """,
    """
    # id
    installable_not_installed

    # installed

    # installable
    foo-3.0
    foo-4.0

    # wanted
    foo>=2.1,!=3.1,<4

    # resolved
    foo-3.0

    # resolved [replace conflicting]
    foo-3.0
    """,
    """
    # id
    not_installable

    # installed

    # installable

    # wanted
    foo>=2.1,!=3.1,<4

    # resolved
    DistributionNotFound

    # resolved [replace conflicting]
    DistributionNotFound
    """,
    """
    # id
    no_matching_version

    # installed

    # installable
    foo-3.1

    # wanted
    foo>=2.1,!=3.1,<4

    # resolved
    DistributionNotFound

    # resolved [replace conflicting]
    DistributionNotFound
    """,
    """
    # id
    installable_with_installed_conflict

    # installed
    foo-3.1

    # installable
    foo-3.5

    # wanted
    foo>=2.1,!=3.1,<4

    # resolved
    VersionConflict

    # resolved [replace conflicting]
    foo-3.5
    """,
    """
    # id
    not_installable_with_installed_conflict

    # installed
    foo-3.1

    # installable

    # wanted
    foo>=2.1,!=3.1,<4

    # resolved
    VersionConflict

    # resolved [replace conflicting]
    DistributionNotFound
    """,
    """
    # id
    installed_with_installed_require

    # installed
    foo-3.9
    baz-0.1
        foo>=2.1,!=3.1,<4

    # installable

    # wanted
    baz

    # resolved
    foo-3.9
    baz-0.1

    # resolved [replace conflicting]
    foo-3.9
    baz-0.1
    """,
    """
    # id
    installed_with_conflicting_installed_require

    # installed
    foo-5
    baz-0.1
        foo>=2.1,!=3.1,<4

    # installable

    # wanted
    baz

    # resolved
    VersionConflict

    # resolved [replace conflicting]
    DistributionNotFound
    """,
    """
    # id
    installed_with_installable_conflicting_require

    # installed
    foo-5
    baz-0.1
        foo>=2.1,!=3.1,<4

    # installable
    foo-2.9

    # wanted
    baz

    # resolved
    VersionConflict

    # resolved [replace conflicting]
    baz-0.1
    foo-2.9
    """,
    """
    # id
    installed_with_installable_require

    # installed
    baz-0.1
        foo>=2.1,!=3.1,<4

    # installable
    foo-3.9

    # wanted
    baz

    # resolved
    foo-3.9
    baz-0.1

    # resolved [replace conflicting]
    foo-3.9
    baz-0.1
    """,
    """
    # id
    installable_with_installed_require

    # installed
    foo-3.9

    # installable
    baz-0.1
        foo>=2.1,!=3.1,<4

    # wanted
    baz

    # resolved
    foo-3.9
    baz-0.1

    # resolved [replace conflicting]
    foo-3.9
    baz-0.1
    """,
    """
    # id
    installable_with_installable_require

    # installed

    # installable
    foo-3.9
    baz-0.1
        foo>=2.1,!=3.1,<4

    # wanted
    baz

    # resolved
    foo-3.9
    baz-0.1

    # resolved [replace conflicting]
    foo-3.9
    baz-0.1
    """,
    """
    # id
    installable_with_conflicting_installable_require

    # installed
    foo-5

    # installable
    foo-2.9
    baz-0.1
        foo>=2.1,!=3.1,<4

    # wanted
    baz

    # resolved
    VersionConflict

    # resolved [replace conflicting]
    baz-0.1
    foo-2.9
    """,
    """
    # id
    conflicting_installables

    # installed

    # installable
    foo-2.9
    foo-5.0

    # wanted
    foo>=2.1,!=3.1,<4
    foo>=4

    # resolved
    VersionConflict

    # resolved [replace conflicting]
    VersionConflict
    """,
    """
    # id
    installables_with_conflicting_requires

    # installed

    # installable
    foo-2.9
        dep==1.0
    baz-5.0
        dep==2.0
    dep-1.0
    dep-2.0

    # wanted
    foo
    baz

    # resolved
    VersionConflict

    # resolved [replace conflicting]
    VersionConflict
    """,
    """
    # id
    installables_with_conflicting_nested_requires

    # installed

    # installable
    foo-2.9
        dep1
    dep1-1.0
        subdep<1.0
    baz-5.0
        dep2
    dep2-1.0
        subdep>1.0
    subdep-0.9
    subdep-1.1

    # wanted
    foo
    baz

    # resolved
    VersionConflict

    # resolved [replace conflicting]
    VersionConflict
    """,
    """
    # id
    wanted_normalized_name_installed_canonical

    # installed
    foo.bar-3.6

    # installable

    # wanted
    foo-bar==3.6

    # resolved
    foo.bar-3.6

    # resolved [replace conflicting]
    foo.bar-3.6
    """,
)
def test_working_set_resolve(
    installed_dists,
    installable_dists,
    requirements,
    replace_conflicting,
    resolved_dists_or_exception,
):
    ws = pkg_resources.WorkingSet([])
    list(map(ws.add, installed_dists))
    resolve_call = functools.partial(
        ws.resolve,
        requirements,
        installer=FakeInstaller(installable_dists),
        replace_conflicting=replace_conflicting,
    )
    if inspect.isclass(resolved_dists_or_exception):
        with pytest.raises(resolved_dists_or_exception):
            resolve_call()
    else:
        assert sorted(resolve_call()) == sorted(resolved_dists_or_exception)
