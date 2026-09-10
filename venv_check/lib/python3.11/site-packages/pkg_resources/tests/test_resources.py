import itertools
import os
import platform
import string
import sys

import pytest
from packaging.specifiers import SpecifierSet

import pkg_resources
from pkg_resources import (
    Distribution,
    EntryPoint,
    Requirement,
    VersionConflict,
    WorkingSet,
    parse_requirements,
    parse_version,
    safe_name,
    safe_version,
)


# from Python 3.6 docs. Available from itertools on Python 3.10
def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


class Metadata(pkg_resources.EmptyProvider):
    """Mock object to return metadata as if from an on-disk distribution"""

    def __init__(self, *pairs) -> None:
        self.metadata = dict(pairs)

    def has_metadata(self, name) -> bool:
        return name in self.metadata

    def get_metadata(self, name):
        return self.metadata[name]

    def get_metadata_lines(self, name):
        return pkg_resources.yield_lines(self.get_metadata(name))


dist_from_fn = pkg_resources.Distribution.from_filename


class TestDistro:
    def testCollection(self):
        # empty path should produce no distributions
        ad = pkg_resources.Environment([], platform=None, python=None)
        assert list(ad) == []
        assert ad['FooPkg'] == []
        ad.add(dist_from_fn("FooPkg-1.3_1.egg"))
        ad.add(dist_from_fn("FooPkg-1.4-py2.4-win32.egg"))
        ad.add(dist_from_fn("FooPkg-1.2-py2.4.egg"))

        # Name is in there now
        assert ad['FooPkg']
        # But only 1 package
        assert list(ad) == ['foopkg']

        # Distributions sort by version
        expected = ['1.4', '1.3-1', '1.2']
        assert [dist.version for dist in ad['FooPkg']] == expected

        # Removing a distribution leaves sequence alone
        ad.remove(ad['FooPkg'][1])
        assert [dist.version for dist in ad['FooPkg']] == ['1.4', '1.2']

        # And inserting adds them in order
        ad.add(dist_from_fn("FooPkg-1.9.egg"))
        assert [dist.version for dist in ad['FooPkg']] == ['1.9', '1.4', '1.2']

        ws = WorkingSet([])
        foo12 = dist_from_fn("FooPkg-1.2-py2.4.egg")
        foo14 = dist_from_fn("FooPkg-1.4-py2.4-win32.egg")
        (req,) = parse_requirements("FooPkg>=1.3")

        # Nominal case: no distros on path, should yield all applicable
        assert ad.best_match(req, ws).version == '1.9'
        # If a matching distro is already installed, should return only that
        ws.add(foo14)
        assert ad.best_match(req, ws).version == '1.4'

        # If the first matching distro is unsuitable, it's a version conflict
        ws = WorkingSet([])
        ws.add(foo12)
        ws.add(foo14)
        with pytest.raises(VersionConflict):
            ad.best_match(req, ws)

        # If more than one match on the path, the first one takes precedence
        ws = WorkingSet([])
        ws.add(foo14)
        ws.add(foo12)
        ws.add(foo14)
        assert ad.best_match(req, ws).version == '1.4'

    def checkFooPkg(self, d):
        assert d.project_name == "FooPkg"
        assert d.key == "foopkg"
        assert d.version == "1.3.post1"
        assert d.py_version == "2.4"
        assert d.platform == "win32"
        assert d.parsed_version == parse_version("1.3-1")

    def testDistroBasics(self):
        d = Distribution(
            "/some/path",
            project_name="FooPkg",
            version="1.3-1",
            py_version="2.4",
            platform="win32",
        )
        self.checkFooPkg(d)

        d = Distribution("/some/path")
        assert d.py_version == f'{sys.version_info.major}.{sys.version_info.minor}'
        assert d.platform is None

    def testDistroParse(self):
        d = dist_from_fn("FooPkg-1.3.post1-py2.4-win32.egg")
        self.checkFooPkg(d)
        d = dist_from_fn("FooPkg-1.3.post1-py2.4-win32.egg-info")
        self.checkFooPkg(d)

    def testDistroMetadata(self):
        d = Distribution(
            "/some/path",
            project_name="FooPkg",
            py_version="2.4",
            platform="win32",
            metadata=Metadata(('PKG-INFO', "Metadata-Version: 1.0\nVersion: 1.3-1\n")),
        )
        self.checkFooPkg(d)

    def distRequires(self, txt):
        return Distribution("/foo", metadata=Metadata(('depends.txt', txt)))

    def checkRequires(self, dist, txt, extras=()):
        assert list(dist.requires(extras)) == list(parse_requirements(txt))

    def testDistroDependsSimple(self):
        for v in "Twisted>=1.5", "Twisted>=1.5\nZConfig>=2.0":
            self.checkRequires(self.distRequires(v), v)

    needs_object_dir = pytest.mark.skipif(
        not hasattr(object, '__dir__'),
        reason='object.__dir__ necessary for self.__dir__ implementation',
    )

    def test_distribution_dir(self):
        d = pkg_resources.Distribution()
        dir(d)

    @needs_object_dir
    def test_distribution_dir_includes_provider_dir(self):
        d = pkg_resources.Distribution()
        before = d.__dir__()
        assert 'test_attr' not in before
        d._provider.test_attr = None
        after = d.__dir__()
        assert len(after) == len(before) + 1
        assert 'test_attr' in after

    @needs_object_dir
    def test_distribution_dir_ignores_provider_dir_leading_underscore(self):
        d = pkg_resources.Distribution()
        before = d.__dir__()
        assert '_test_attr' not in before
        d._provider._test_attr = None
        after = d.__dir__()
        assert len(after) == len(before)
        assert '_test_attr' not in after

    def testResolve(self):
        ad = pkg_resources.Environment([])
        ws = WorkingSet([])
        # Resolving no requirements -> nothing to install
        assert list(ws.resolve([], ad)) == []
        # Request something not in the collection -> DistributionNotFound
        with pytest.raises(pkg_resources.DistributionNotFound):
            ws.resolve(parse_requirements("Foo"), ad)

        Foo = Distribution.from_filename(
            "/foo_dir/Foo-1.2.egg",
            metadata=Metadata(('depends.txt', "[bar]\nBaz>=2.0")),
        )
        ad.add(Foo)
        ad.add(Distribution.from_filename("Foo-0.9.egg"))

        # Request thing(s) that are available -> list to activate
        for i in range(3):
            targets = list(ws.resolve(parse_requirements("Foo"), ad))
            assert targets == [Foo]
            list(map(ws.add, targets))
        with pytest.raises(VersionConflict):
            ws.resolve(parse_requirements("Foo==0.9"), ad)
        ws = WorkingSet([])  # reset

        # Request an extra that causes an unresolved dependency for "Baz"
        with pytest.raises(pkg_resources.DistributionNotFound):
            ws.resolve(parse_requirements("Foo[bar]"), ad)
        Baz = Distribution.from_filename(
            "/foo_dir/Baz-2.1.egg", metadata=Metadata(('depends.txt', "Foo"))
        )
        ad.add(Baz)

        # Activation list now includes resolved dependency
        assert list(ws.resolve(parse_requirements("Foo[bar]"), ad)) == [Foo, Baz]
        # Requests for conflicting versions produce VersionConflict
        with pytest.raises(VersionConflict) as vc:
            ws.resolve(parse_requirements("Foo==1.2\nFoo!=1.2"), ad)

        msg = 'Foo 0.9 is installed but Foo==1.2 is required'
        assert vc.value.report() == msg

    def test_environment_marker_evaluation_negative(self):
        """Environment markers are evaluated at resolution time."""
        ad = pkg_resources.Environment([])
        ws = WorkingSet([])
        res = ws.resolve(parse_requirements("Foo;python_version<'2'"), ad)
        assert list(res) == []

    def test_environment_marker_evaluation_positive(self):
        ad = pkg_resources.Environment([])
        ws = WorkingSet([])
        Foo = Distribution.from_filename("/foo_dir/Foo-1.2.dist-info")
        ad.add(Foo)
        res = ws.resolve(parse_requirements("Foo;python_version>='2'"), ad)
        assert list(res) == [Foo]

    def test_environment_marker_evaluation_called(self):
        """
        If one package foo requires bar without any extras,
        markers should pass for bar without extras.
        """
        (parent_req,) = parse_requirements("foo")
        (req,) = parse_requirements("bar;python_version>='2'")
        req_extras = pkg_resources._ReqExtras({req: parent_req.extras})
        assert req_extras.markers_pass(req)

        (parent_req,) = parse_requirements("foo[]")
        (req,) = parse_requirements("bar;python_version>='2'")
        req_extras = pkg_resources._ReqExtras({req: parent_req.extras})
        assert req_extras.markers_pass(req)

    def test_marker_evaluation_with_extras(self):
        """Extras are also evaluated as markers at resolution time."""
        ad = pkg_resources.Environment([])
        ws = WorkingSet([])
        Foo = Distribution.from_filename(
            "/foo_dir/Foo-1.2.dist-info",
            metadata=Metadata((
                "METADATA",
                "Provides-Extra: baz\nRequires-Dist: quux; extra=='baz'",
            )),
        )
        ad.add(Foo)
        assert list(ws.resolve(parse_requirements("Foo"), ad)) == [Foo]
        quux = Distribution.from_filename("/foo_dir/quux-1.0.dist-info")
        ad.add(quux)
        res = list(ws.resolve(parse_requirements("Foo[baz]"), ad))
        assert res == [Foo, quux]

    def test_marker_evaluation_with_extras_normlized(self):
        """Extras are also evaluated as markers at resolution time."""
        ad = pkg_resources.Environment([])
        ws = WorkingSet([])
        Foo = Distribution.from_filename(
            "/foo_dir/Foo-1.2.dist-info",
            metadata=Metadata((
                "METADATA",
                "Provides-Extra: baz-lightyear\n"
                "Requires-Dist: quux; extra=='baz-lightyear'",
            )),
        )
        ad.add(Foo)
        assert list(ws.resolve(parse_requirements("Foo"), ad)) == [Foo]
        quux = Distribution.from_filename("/foo_dir/quux-1.0.dist-info")
        ad.add(quux)
        res = list(ws.resolve(parse_requirements("Foo[baz-lightyear]"), ad))
        assert res == [Foo, quux]

    def test_marker_evaluation_with_multiple_extras(self):
        ad = pkg_resources.Environment([])
        ws = WorkingSet([])
        Foo = Distribution.from_filename(
            "/foo_dir/Foo-1.2.dist-info",
            metadata=Metadata((
                "METADATA",
                "Provides-Extra: baz\n"
                "Requires-Dist: quux; extra=='baz'\n"
                "Provides-Extra: bar\n"
                "Requires-Dist: fred; extra=='bar'\n",
            )),
        )
        ad.add(Foo)
        quux = Distribution.from_filename("/foo_dir/quux-1.0.dist-info")
        ad.add(quux)
        fred = Distribution.from_filename("/foo_dir/fred-0.1.dist-info")
        ad.add(fred)
        res = list(ws.resolve(parse_requirements("Foo[baz,bar]"), ad))
        assert sorted(res) == [fred, quux, Foo]

    def test_marker_evaluation_with_extras_loop(self):
        ad = pkg_resources.Environment([])
        ws = WorkingSet([])
        a = Distribution.from_filename(
            "/foo_dir/a-0.2.dist-info",
            metadata=Metadata(("METADATA", "Requires-Dist: c[a]")),
        )
        b = Distribution.from_filename(
            "/foo_dir/b-0.3.dist-info",
            metadata=Metadata(("METADATA", "Requires-Dist: c[b]")),
        )
        c = Distribution.from_filename(
            "/foo_dir/c-1.0.dist-info",
            metadata=Metadata((
                "METADATA",
                "Provides-Extra: a\n"
                "Requires-Dist: b;extra=='a'\n"
                "Provides-Extra: b\n"
                "Requires-Dist: foo;extra=='b'",
            )),
        )
        foo = Distribution.from_filename("/foo_dir/foo-0.1.dist-info")
        for dist in (a, b, c, foo):
            ad.add(dist)
        res = list(ws.resolve(parse_requirements("a"), ad))
        assert res == [a, c, b, foo]

    @pytest.mark.xfail(
        sys.version_info[:2] == (3, 12) and sys.version_info.releaselevel != 'final',
        reason="https://github.com/python/cpython/issues/103632",
    )
    def testDistroDependsOptions(self):
        d = self.distRequires(
            """
            Twisted>=1.5
            [docgen]
            ZConfig>=2.0
            docutils>=0.3
            [fastcgi]
            fcgiapp>=0.1"""
        )
        self.checkRequires(d, "Twisted>=1.5")
        self.checkRequires(
            d, "Twisted>=1.5 ZConfig>=2.0 docutils>=0.3".split(), ["docgen"]
        )
        self.checkRequires(d, "Twisted>=1.5 fcgiapp>=0.1".split(), ["fastcgi"])
        self.checkRequires(
            d,
            "Twisted>=1.5 ZConfig>=2.0 docutils>=0.3 fcgiapp>=0.1".split(),
            ["docgen", "fastcgi"],
        )
        self.checkRequires(
            d,
            "Twisted>=1.5 fcgiapp>=0.1 ZConfig>=2.0 docutils>=0.3".split(),
            ["fastcgi", "docgen"],
        )
        with pytest.raises(pkg_resources.UnknownExtra):
            d.requires(["foo"])


class TestWorkingSet:
    def test_find_conflicting(self):
        ws = WorkingSet([])
        Foo = Distribution.from_filename("/foo_dir/Foo-1.2.egg")
        ws.add(Foo)

        # create a requirement that conflicts with Foo 1.2
        req = next(parse_requirements("Foo<1.2"))

        with pytest.raises(VersionConflict) as vc:
            ws.find(req)

        msg = 'Foo 1.2 is installed but Foo<1.2 is required'
        assert vc.value.report() == msg

    def test_resolve_conflicts_with_prior(self):
        """
        A ContextualVersionConflict should be raised when a requirement
        conflicts with a prior requirement for a different package.
        """
        # Create installation where Foo depends on Baz 1.0 and Bar depends on
        # Baz 2.0.
        ws = WorkingSet([])
        md = Metadata(('depends.txt', "Baz==1.0"))
        Foo = Distribution.from_filename("/foo_dir/Foo-1.0.egg", metadata=md)
        ws.add(Foo)
        md = Metadata(('depends.txt', "Baz==2.0"))
        Bar = Distribution.from_filename("/foo_dir/Bar-1.0.egg", metadata=md)
        ws.add(Bar)
        Baz = Distribution.from_filename("/foo_dir/Baz-1.0.egg")
        ws.add(Baz)
        Baz = Distribution.from_filename("/foo_dir/Baz-2.0.egg")
        ws.add(Baz)

        with pytest.raises(VersionConflict) as vc:
            ws.resolve(parse_requirements("Foo\nBar\n"))

        msg = "Baz 1.0 is installed but Baz==2.0 is required by "
        msg += repr(set(['Bar']))
        assert vc.value.report() == msg


class TestEntryPoints:
    def assertfields(self, ep):
        assert ep.name == "foo"
        assert ep.module_name == "pkg_resources.tests.test_resources"
        assert ep.attrs == ("TestEntryPoints",)
        assert ep.extras == ("x",)
        assert ep.load() is TestEntryPoints
        expect = "foo = pkg_resources.tests.test_resources:TestEntryPoints [x]"
        assert str(ep) == expect

    def setup_method(self, method):
        self.dist = Distribution.from_filename(
            "FooPkg-1.2-py2.4.egg", metadata=Metadata(('requires.txt', '[x]'))
        )

    def testBasics(self):
        ep = EntryPoint(
            "foo",
            "pkg_resources.tests.test_resources",
            ["TestEntryPoints"],
            ["x"],
            self.dist,
        )
        self.assertfields(ep)

    def testParse(self):
        s = "foo = pkg_resources.tests.test_resources:TestEntryPoints [x]"
        ep = EntryPoint.parse(s, self.dist)
        self.assertfields(ep)

        ep = EntryPoint.parse("bar baz=  spammity[PING]")
        assert ep.name == "bar baz"
        assert ep.module_name == "spammity"
        assert ep.attrs == ()
        assert ep.extras == ("ping",)

        ep = EntryPoint.parse(" fizzly =  wocka:foo")
        assert ep.name == "fizzly"
        assert ep.module_name == "wocka"
        assert ep.attrs == ("foo",)
        assert ep.extras == ()

        # plus in the name
        spec = "html+mako = mako.ext.pygmentplugin:MakoHtmlLexer"
        ep = EntryPoint.parse(spec)
        assert ep.name == 'html+mako'

    reject_specs = "foo", "x=a:b:c", "q=x/na", "fez=pish:tush-z", "x=f[a]>2"

    @pytest.mark.parametrize("reject_spec", reject_specs)
    def test_reject_spec(self, reject_spec):
        with pytest.raises(ValueError):
            EntryPoint.parse(reject_spec)

    def test_printable_name(self):
        """
        Allow any printable character in the name.
        """
        # Create a name with all printable characters; strip the whitespace.
        name = string.printable.strip()
        spec = "{name} = module:attr".format(**locals())
        ep = EntryPoint.parse(spec)
        assert ep.name == name

    def checkSubMap(self, m):
        assert len(m) == len(self.submap_expect)
        for key, ep in self.submap_expect.items():
            assert m.get(key).name == ep.name
            assert m.get(key).module_name == ep.module_name
            assert sorted(m.get(key).attrs) == sorted(ep.attrs)
            assert sorted(m.get(key).extras) == sorted(ep.extras)

    submap_expect = dict(
        feature1=EntryPoint('feature1', 'somemodule', ['somefunction']),
        feature2=EntryPoint(
            'feature2', 'another.module', ['SomeClass'], ['extra1', 'extra2']
        ),
        feature3=EntryPoint('feature3', 'this.module', extras=['something']),
    )
    submap_str = """
            # define features for blah blah
            feature1 = somemodule:somefunction
            feature2 = another.module:SomeClass [extra1,extra2]
            feature3 = this.module [something]
    """

    def testParseList(self):
        self.checkSubMap(EntryPoint.parse_group("xyz", self.submap_str))
        with pytest.raises(ValueError):
            EntryPoint.parse_group("x a", "foo=bar")
        with pytest.raises(ValueError):
            EntryPoint.parse_group("x", ["foo=baz", "foo=bar"])

    def testParseMap(self):
        m = EntryPoint.parse_map({'xyz': self.submap_str})
        self.checkSubMap(m['xyz'])
        assert list(m.keys()) == ['xyz']
        m = EntryPoint.parse_map("[xyz]\n" + self.submap_str)
        self.checkSubMap(m['xyz'])
        assert list(m.keys()) == ['xyz']
        with pytest.raises(ValueError):
            EntryPoint.parse_map(["[xyz]", "[xyz]"])
        with pytest.raises(ValueError):
            EntryPoint.parse_map(self.submap_str)

    def testDeprecationWarnings(self):
        ep = EntryPoint(
            "foo", "pkg_resources.tests.test_resources", ["TestEntryPoints"], ["x"]
        )
        with pytest.warns(pkg_resources.PkgResourcesDeprecationWarning):
            ep.load(require=False)


class TestRequirements:
    def testBasics(self):
        r = Requirement.parse("Twisted>=1.2")
        assert str(r) == "Twisted>=1.2"
        assert repr(r) == "Requirement.parse('Twisted>=1.2')"
        assert r == Requirement("Twisted>=1.2")
        assert r == Requirement("twisTed>=1.2")
        assert r != Requirement("Twisted>=2.0")
        assert r != Requirement("Zope>=1.2")
        assert r != Requirement("Zope>=3.0")
        assert r != Requirement("Twisted[extras]>=1.2")

    def testOrdering(self):
        r1 = Requirement("Twisted==1.2c1,>=1.2")
        r2 = Requirement("Twisted>=1.2,==1.2c1")
        assert r1 == r2
        assert str(r1) == str(r2)
        assert str(r2) == "Twisted==1.2c1,>=1.2"
        assert Requirement("Twisted") != Requirement(
            "Twisted @ https://localhost/twisted.zip"
        )

    def testBasicContains(self):
        r = Requirement("Twisted>=1.2")
        foo_dist = Distribution.from_filename("FooPkg-1.3_1.egg")
        twist11 = Distribution.from_filename("Twisted-1.1.egg")
        twist12 = Distribution.from_filename("Twisted-1.2.egg")
        assert parse_version('1.2') in r
        assert parse_version('1.1') not in r
        assert '1.2' in r
        assert '1.1' not in r
        assert foo_dist not in r
        assert twist11 not in r
        assert twist12 in r

    def testOptionsAndHashing(self):
        r1 = Requirement.parse("Twisted[foo,bar]>=1.2")
        r2 = Requirement.parse("Twisted[bar,FOO]>=1.2")
        assert r1 == r2
        assert set(r1.extras) == set(("foo", "bar"))
        assert set(r2.extras) == set(("foo", "bar"))
        assert hash(r1) == hash(r2)
        assert hash(r1) == hash((
            "twisted",
            None,
            SpecifierSet(">=1.2"),
            frozenset(["foo", "bar"]),
            None,
        ))
        assert hash(
            Requirement.parse("Twisted @ https://localhost/twisted.zip")
        ) == hash((
            "twisted",
            "https://localhost/twisted.zip",
            SpecifierSet(),
            frozenset(),
            None,
        ))

    def testVersionEquality(self):
        r1 = Requirement.parse("foo==0.3a2")
        r2 = Requirement.parse("foo!=0.3a4")
        d = Distribution.from_filename

        assert d("foo-0.3a4.egg") not in r1
        assert d("foo-0.3a1.egg") not in r1
        assert d("foo-0.3a4.egg") not in r2

        assert d("foo-0.3a2.egg") in r1
        assert d("foo-0.3a2.egg") in r2
        assert d("foo-0.3a3.egg") in r2
        assert d("foo-0.3a5.egg") in r2

    def testSetuptoolsProjectName(self):
        """
        The setuptools project should implement the setuptools package.
        """

        assert Requirement.parse('setuptools').project_name == 'setuptools'
        # setuptools 0.7 and higher means setuptools.
        assert Requirement.parse('setuptools == 0.7').project_name == 'setuptools'
        assert Requirement.parse('setuptools == 0.7a1').project_name == 'setuptools'
        assert Requirement.parse('setuptools >= 0.7').project_name == 'setuptools'


class TestParsing:
    def testEmptyParse(self):
        assert list(parse_requirements('')) == []

    def testYielding(self):
        for inp, out in [
            ([], []),
            ('x', ['x']),
            ([[]], []),
            (' x\n y', ['x', 'y']),
            (['x\n\n', 'y'], ['x', 'y']),
        ]:
            assert list(pkg_resources.yield_lines(inp)) == out

    def testSplitting(self):
        sample = """
                    x
                    [Y]
                    z

                    a
                    [b ]
                    # foo
                    c
                    [ d]
                    [q]
                    v
                    """
        assert list(pkg_resources.split_sections(sample)) == [
            (None, ["x"]),
            ("Y", ["z", "a"]),
            ("b", ["c"]),
            ("d", []),
            ("q", ["v"]),
        ]
        with pytest.raises(ValueError):
            list(pkg_resources.split_sections("[foo"))

    def testSafeName(self):
        assert safe_name("adns-python") == "adns-python"
        assert safe_name("WSGI Utils") == "WSGI-Utils"
        assert safe_name("WSGI  Utils") == "WSGI-Utils"
        assert safe_name("Money$$$Maker") == "Money-Maker"
        assert safe_name("peak.web") != "peak-web"

    def testSafeVersion(self):
        assert safe_version("1.2-1") == "1.2.post1"
        assert safe_version("1.2 alpha") == "1.2.alpha"
        assert safe_version("2.3.4 20050521") == "2.3.4.20050521"
        assert safe_version("Money$$$Maker") == "Money-Maker"
        assert safe_version("peak.web") == "peak.web"

    def testSimpleRequirements(self):
        assert list(parse_requirements('Twis-Ted>=1.2-1')) == [
            Requirement('Twis-Ted>=1.2-1')
        ]
        assert list(parse_requirements('Twisted >=1.2, \\ # more\n<2.0')) == [
            Requirement('Twisted>=1.2,<2.0')
        ]
        assert Requirement.parse("FooBar==1.99a3") == Requirement("FooBar==1.99a3")
        with pytest.raises(ValueError):
            Requirement.parse(">=2.3")
        with pytest.raises(ValueError):
            Requirement.parse("x\\")
        with pytest.raises(ValueError):
            Requirement.parse("x==2 q")
        with pytest.raises(ValueError):
            Requirement.parse("X==1\nY==2")
        with pytest.raises(ValueError):
            Requirement.parse("#")

    def test_requirements_with_markers(self):
        assert Requirement.parse("foobar;os_name=='a'") == Requirement.parse(
            "foobar;os_name=='a'"
        )
        assert Requirement.parse(
            "name==1.1;python_version=='2.7'"
        ) != Requirement.parse("name==1.1;python_version=='3.6'")
        assert Requirement.parse(
            "name==1.0;python_version=='2.7'"
        ) != Requirement.parse("name==1.2;python_version=='2.7'")
        assert Requirement.parse(
            "name[foo]==1.0;python_version=='3.6'"
        ) != Requirement.parse("name[foo,bar]==1.0;python_version=='3.6'")

    def test_local_version(self):
        parse_requirements('foo==1.0+org1')

    def test_spaces_between_multiple_versions(self):
        parse_requirements('foo>=1.0, <3')
        parse_requirements('foo >= 1.0, < 3')

    @pytest.mark.parametrize(
        ("lower", "upper"),
        [
            ('1.2-rc1', '1.2rc1'),
            ('0.4', '0.4.0'),
            ('0.4.0.0', '0.4.0'),
            ('0.4.0-0', '0.4-0'),
            ('0post1', '0.0post1'),
            ('0pre1', '0.0c1'),
            ('0.0.0preview1', '0c1'),
            ('0.0c1', '0-rc1'),
            ('1.2a1', '1.2.a.1'),
            ('1.2.a', '1.2a'),
        ],
    )
    def testVersionEquality(self, lower, upper):
        assert parse_version(lower) == parse_version(upper)

    torture = """
        0.80.1-3 0.80.1-2 0.80.1-1 0.79.9999+0.80.0pre4-1
        0.79.9999+0.80.0pre2-3 0.79.9999+0.80.0pre2-2
        0.77.2-1 0.77.1-1 0.77.0-1
        """

    @pytest.mark.parametrize(
        ("lower", "upper"),
        [
            ('2.1', '2.1.1'),
            ('2a1', '2b0'),
            ('2a1', '2.1'),
            ('2.3a1', '2.3'),
            ('2.1-1', '2.1-2'),
            ('2.1-1', '2.1.1'),
            ('2.1', '2.1post4'),
            ('2.1a0-20040501', '2.1'),
            ('1.1', '02.1'),
            ('3.2', '3.2.post0'),
            ('3.2post1', '3.2post2'),
            ('0.4', '4.0'),
            ('0.0.4', '0.4.0'),
            ('0post1', '0.4post1'),
            ('2.1.0-rc1', '2.1.0'),
            ('2.1dev', '2.1a0'),
        ]
        + list(pairwise(reversed(torture.split()))),
    )
    def testVersionOrdering(self, lower, upper):
        assert parse_version(lower) < parse_version(upper)

    def testVersionHashable(self):
        """
        Ensure that our versions stay hashable even though we've subclassed
        them and added some shim code to them.
        """
        assert hash(parse_version("1.0")) == hash(parse_version("1.0"))


class TestNamespaces:
    ns_str = "__import__('pkg_resources').declare_namespace(__name__)\n"

    @pytest.fixture
    def symlinked_tmpdir(self, tmpdir):
        """
        Where available, return the tempdir as a symlink,
        which as revealed in #231 is more fragile than
        a natural tempdir.
        """
        if not hasattr(os, 'symlink'):
            yield str(tmpdir)
            return

        link_name = str(tmpdir) + '-linked'
        os.symlink(str(tmpdir), link_name)
        try:
            yield type(tmpdir)(link_name)
        finally:
            os.unlink(link_name)

    @pytest.fixture(autouse=True)
    def patched_path(self, tmpdir):
        """
        Patch sys.path to include the 'site-pkgs' dir. Also
        restore pkg_resources._namespace_packages to its
        former state.
        """
        saved_ns_pkgs = pkg_resources._namespace_packages.copy()
        saved_sys_path = sys.path[:]
        site_pkgs = tmpdir.mkdir('site-pkgs')
        sys.path.append(str(site_pkgs))
        try:
            yield
        finally:
            pkg_resources._namespace_packages = saved_ns_pkgs
            sys.path = saved_sys_path

    issue591 = pytest.mark.xfail(platform.system() == 'Windows', reason="#591")

    @issue591
    def test_two_levels_deep(self, symlinked_tmpdir):
        """
        Test nested namespace packages
        Create namespace packages in the following tree :
            site-packages-1/pkg1/pkg2
            site-packages-2/pkg1/pkg2
        Check both are in the _namespace_packages dict and that their __path__
        is correct
        """
        real_tmpdir = symlinked_tmpdir.realpath()
        tmpdir = symlinked_tmpdir
        sys.path.append(str(tmpdir / 'site-pkgs2'))
        site_dirs = tmpdir / 'site-pkgs', tmpdir / 'site-pkgs2'
        for site in site_dirs:
            pkg1 = site / 'pkg1'
            pkg2 = pkg1 / 'pkg2'
            pkg2.ensure_dir()
            (pkg1 / '__init__.py').write_text(self.ns_str, encoding='utf-8')
            (pkg2 / '__init__.py').write_text(self.ns_str, encoding='utf-8')
        with pytest.warns(DeprecationWarning, match="pkg_resources.declare_namespace"):
            import pkg1  # pyright: ignore[reportMissingImports] # Temporary package for test
        assert "pkg1" in pkg_resources._namespace_packages
        # attempt to import pkg2 from site-pkgs2
        with pytest.warns(DeprecationWarning, match="pkg_resources.declare_namespace"):
            import pkg1.pkg2  # pyright: ignore[reportMissingImports] # Temporary package for test
        # check the _namespace_packages dict
        assert "pkg1.pkg2" in pkg_resources._namespace_packages
        assert pkg_resources._namespace_packages["pkg1"] == ["pkg1.pkg2"]
        # check the __path__ attribute contains both paths
        expected = [
            str(real_tmpdir / "site-pkgs" / "pkg1" / "pkg2"),
            str(real_tmpdir / "site-pkgs2" / "pkg1" / "pkg2"),
        ]
        assert pkg1.pkg2.__path__ == expected

    @issue591
    def test_path_order(self, symlinked_tmpdir):
        """
        Test that if multiple versions of the same namespace package subpackage
        are on different sys.path entries, that only the one earliest on
        sys.path is imported, and that the namespace package's __path__ is in
        the correct order.

        Regression test for https://github.com/pypa/setuptools/issues/207
        """

        tmpdir = symlinked_tmpdir
        site_dirs = (
            tmpdir / "site-pkgs",
            tmpdir / "site-pkgs2",
            tmpdir / "site-pkgs3",
        )

        vers_str = "__version__ = %r"

        for number, site in enumerate(site_dirs, 1):
            if number > 1:
                sys.path.append(str(site))
            nspkg = site / 'nspkg'
            subpkg = nspkg / 'subpkg'
            subpkg.ensure_dir()
            (nspkg / '__init__.py').write_text(self.ns_str, encoding='utf-8')
            (subpkg / '__init__.py').write_text(vers_str % number, encoding='utf-8')

        with pytest.warns(DeprecationWarning, match="pkg_resources.declare_namespace"):
            import nspkg  # pyright: ignore[reportMissingImports] # Temporary package for test
            import nspkg.subpkg  # pyright: ignore[reportMissingImports] # Temporary package for test
        expected = [str(site.realpath() / 'nspkg') for site in site_dirs]
        assert nspkg.__path__ == expected
        assert nspkg.subpkg.__version__ == 1
