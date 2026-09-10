import re
import itertools
import textwrap
import functools

try:
    from importlib.resources import files  # type: ignore
except ImportError:  # pragma: nocover
    from importlib_resources import files  # type: ignore

from jaraco.functools import compose, method_cache
from jaraco.context import ExceptionTrap


def substitution(old, new):
    """
    Return a function that will perform a substitution on a string
    """
    return lambda s: s.replace(old, new)


def multi_substitution(*substitutions):
    """
    Take a sequence of pairs specifying substitutions, and create
    a function that performs those substitutions.

    >>> multi_substitution(('foo', 'bar'), ('bar', 'baz'))('foo')
    'baz'
    """
    substitutions = itertools.starmap(substitution, substitutions)
    # compose function applies last function first, so reverse the
    #  substitutions to get the expected order.
    substitutions = reversed(tuple(substitutions))
    return compose(*substitutions)


class FoldedCase(str):
    """
    A case insensitive string class; behaves just like str
    except compares equal when the only variation is case.

    >>> s = FoldedCase('hello world')

    >>> s == 'Hello World'
    True

    >>> 'Hello World' == s
    True

    >>> s != 'Hello World'
    False

    >>> s.index('O')
    4

    >>> s.split('O')
    ['hell', ' w', 'rld']

    >>> sorted(map(FoldedCase, ['GAMMA', 'alpha', 'Beta']))
    ['alpha', 'Beta', 'GAMMA']

    Sequence membership is straightforward.

    >>> "Hello World" in [s]
    True
    >>> s in ["Hello World"]
    True

    Allows testing for set inclusion, but candidate and elements
    must both be folded.

    >>> FoldedCase("Hello World") in {s}
    True
    >>> s in {FoldedCase("Hello World")}
    True

    String inclusion works as long as the FoldedCase object
    is on the right.

    >>> "hello" in FoldedCase("Hello World")
    True

    But not if the FoldedCase object is on the left:

    >>> FoldedCase('hello') in 'Hello World'
    False

    In that case, use ``in_``:

    >>> FoldedCase('hello').in_('Hello World')
    True

    >>> FoldedCase('hello') > FoldedCase('Hello')
    False

    >>> FoldedCase('ß') == FoldedCase('ss')
    True
    """

    def __lt__(self, other):
        return self.casefold() < other.casefold()

    def __gt__(self, other):
        return self.casefold() > other.casefold()

    def __eq__(self, other):
        return self.casefold() == other.casefold()

    def __ne__(self, other):
        return self.casefold() != other.casefold()

    def __hash__(self):
        return hash(self.casefold())

    def __contains__(self, other):
        return super().casefold().__contains__(other.casefold())

    def in_(self, other):
        "Does self appear in other?"
        return self in FoldedCase(other)

    # cache casefold since it's likely to be called frequently.
    @method_cache
    def casefold(self):
        return super().casefold()

    def index(self, sub):
        return self.casefold().index(sub.casefold())

    def split(self, splitter=' ', maxsplit=0):
        pattern = re.compile(re.escape(splitter), re.I)
        return pattern.split(self, maxsplit)


# Python 3.8 compatibility
_unicode_trap = ExceptionTrap(UnicodeDecodeError)


@_unicode_trap.passes
def is_decodable(value):
    r"""
    Return True if the supplied value is decodable (using the default
    encoding).

    >>> is_decodable(b'\xff')
    False
    >>> is_decodable(b'\x32')
    True
    """
    value.decode()


def is_binary(value):
    r"""
    Return True if the value appears to be binary (that is, it's a byte
    string and isn't decodable).

    >>> is_binary(b'\xff')
    True
    >>> is_binary('\xff')
    False
    """
    return isinstance(value, bytes) and not is_decodable(value)


def trim(s):
    r"""
    Trim something like a docstring to remove the whitespace that
    is common due to indentation and formatting.

    >>> trim("\n\tfoo = bar\n\t\tbar = baz\n")
    'foo = bar\n\tbar = baz'
    """
    return textwrap.dedent(s).strip()


def wrap(s):
    """
    Wrap lines of text, retaining existing newlines as
    paragraph markers.

    >>> print(wrap(lorem_ipsum))
    Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do
    eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad
    minim veniam, quis nostrud exercitation ullamco laboris nisi ut
    aliquip ex ea commodo consequat. Duis aute irure dolor in
    reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla
    pariatur. Excepteur sint occaecat cupidatat non proident, sunt in
    culpa qui officia deserunt mollit anim id est laborum.
    <BLANKLINE>
    Curabitur pretium tincidunt lacus. Nulla gravida orci a odio. Nullam
    varius, turpis et commodo pharetra, est eros bibendum elit, nec luctus
    magna felis sollicitudin mauris. Integer in mauris eu nibh euismod
    gravida. Duis ac tellus et risus vulputate vehicula. Donec lobortis
    risus a elit. Etiam tempor. Ut ullamcorper, ligula eu tempor congue,
    eros est euismod turpis, id tincidunt sapien risus a quam. Maecenas
    fermentum consequat mi. Donec fermentum. Pellentesque malesuada nulla
    a mi. Duis sapien sem, aliquet nec, commodo eget, consequat quis,
    neque. Aliquam faucibus, elit ut dictum aliquet, felis nisl adipiscing
    sapien, sed malesuada diam lacus eget erat. Cras mollis scelerisque
    nunc. Nullam arcu. Aliquam consequat. Curabitur augue lorem, dapibus
    quis, laoreet et, pretium ac, nisi. Aenean magna nisl, mollis quis,
    molestie eu, feugiat in, orci. In hac habitasse platea dictumst.
    """
    paragraphs = s.splitlines()
    wrapped = ('\n'.join(textwrap.wrap(para)) for para in paragraphs)
    return '\n\n'.join(wrapped)


def unwrap(s):
    r"""
    Given a multi-line string, return an unwrapped version.

    >>> wrapped = wrap(lorem_ipsum)
    >>> wrapped.count('\n')
    20
    >>> unwrapped = unwrap(wrapped)
    >>> unwrapped.count('\n')
    1
    >>> print(unwrapped)
    Lorem ipsum dolor sit amet, consectetur adipiscing ...
    Curabitur pretium tincidunt lacus. Nulla gravida orci ...

    """
    paragraphs = re.split(r'\n\n+', s)
    cleaned = (para.replace('\n', ' ') for para in paragraphs)
    return '\n'.join(cleaned)


lorem_ipsum: str = (
    files(__name__).joinpath('Lorem ipsum.txt').read_text(encoding='utf-8')
)


class Splitter:
    """object that will split a string with the given arguments for each call

    >>> s = Splitter(',')
    >>> s('hello, world, this is your, master calling')
    ['hello', ' world', ' this is your', ' master calling']
    """

    def __init__(self, *args):
        self.args = args

    def __call__(self, s):
        return s.split(*self.args)


def indent(string, prefix=' ' * 4):
    """
    >>> indent('foo')
    '    foo'
    """
    return prefix + string


class WordSet(tuple):
    """
    Given an identifier, return the words that identifier represents,
    whether in camel case, underscore-separated, etc.

    >>> WordSet.parse("camelCase")
    ('camel', 'Case')

    >>> WordSet.parse("under_sep")
    ('under', 'sep')

    Acronyms should be retained

    >>> WordSet.parse("firstSNL")
    ('first', 'SNL')

    >>> WordSet.parse("you_and_I")
    ('you', 'and', 'I')

    >>> WordSet.parse("A simple test")
    ('A', 'simple', 'test')

    Multiple caps should not interfere with the first cap of another word.

    >>> WordSet.parse("myABCClass")
    ('my', 'ABC', 'Class')

    The result is a WordSet, providing access to other forms.

    >>> WordSet.parse("myABCClass").underscore_separated()
    'my_ABC_Class'

    >>> WordSet.parse('a-command').camel_case()
    'ACommand'

    >>> WordSet.parse('someIdentifier').lowered().space_separated()
    'some identifier'

    Slices of the result should return another WordSet.

    >>> WordSet.parse('taken-out-of-context')[1:].underscore_separated()
    'out_of_context'

    >>> WordSet.from_class_name(WordSet()).lowered().space_separated()
    'word set'

    >>> example = WordSet.parse('figured it out')
    >>> example.headless_camel_case()
    'figuredItOut'
    >>> example.dash_separated()
    'figured-it-out'

    """

    _pattern = re.compile('([A-Z]?[a-z]+)|([A-Z]+(?![a-z]))')

    def capitalized(self):
        return WordSet(word.capitalize() for word in self)

    def lowered(self):
        return WordSet(word.lower() for word in self)

    def camel_case(self):
        return ''.join(self.capitalized())

    def headless_camel_case(self):
        words = iter(self)
        first = next(words).lower()
        new_words = itertools.chain((first,), WordSet(words).camel_case())
        return ''.join(new_words)

    def underscore_separated(self):
        return '_'.join(self)

    def dash_separated(self):
        return '-'.join(self)

    def space_separated(self):
        return ' '.join(self)

    def trim_right(self, item):
        """
        Remove the item from the end of the set.

        >>> WordSet.parse('foo bar').trim_right('foo')
        ('foo', 'bar')
        >>> WordSet.parse('foo bar').trim_right('bar')
        ('foo',)
        >>> WordSet.parse('').trim_right('bar')
        ()
        """
        return self[:-1] if self and self[-1] == item else self

    def trim_left(self, item):
        """
        Remove the item from the beginning of the set.

        >>> WordSet.parse('foo bar').trim_left('foo')
        ('bar',)
        >>> WordSet.parse('foo bar').trim_left('bar')
        ('foo', 'bar')
        >>> WordSet.parse('').trim_left('bar')
        ()
        """
        return self[1:] if self and self[0] == item else self

    def trim(self, item):
        """
        >>> WordSet.parse('foo bar').trim('foo')
        ('bar',)
        """
        return self.trim_left(item).trim_right(item)

    def __getitem__(self, item):
        result = super().__getitem__(item)
        if isinstance(item, slice):
            result = WordSet(result)
        return result

    @classmethod
    def parse(cls, identifier):
        matches = cls._pattern.finditer(identifier)
        return WordSet(match.group(0) for match in matches)

    @classmethod
    def from_class_name(cls, subject):
        return cls.parse(subject.__class__.__name__)


# for backward compatibility
words = WordSet.parse


def simple_html_strip(s):
    r"""
    Remove HTML from the string `s`.

    >>> str(simple_html_strip(''))
    ''

    >>> print(simple_html_strip('A <bold>stormy</bold> day in paradise'))
    A stormy day in paradise

    >>> print(simple_html_strip('Somebody <!-- do not --> tell the truth.'))
    Somebody  tell the truth.

    >>> print(simple_html_strip('What about<br/>\nmultiple lines?'))
    What about
    multiple lines?
    """
    html_stripper = re.compile('(<!--.*?-->)|(<[^>]*>)|([^<]+)', re.DOTALL)
    texts = (match.group(3) or '' for match in html_stripper.finditer(s))
    return ''.join(texts)


class SeparatedValues(str):
    """
    A string separated by a separator. Overrides __iter__ for getting
    the values.

    >>> list(SeparatedValues('a,b,c'))
    ['a', 'b', 'c']

    Whitespace is stripped and empty values are discarded.

    >>> list(SeparatedValues(' a,   b   , c,  '))
    ['a', 'b', 'c']
    """

    separator = ','

    def __iter__(self):
        parts = self.split(self.separator)
        return filter(None, (part.strip() for part in parts))


class Stripper:
    r"""
    Given a series of lines, find the common prefix and strip it from them.

    >>> lines = [
    ...     'abcdefg\n',
    ...     'abc\n',
    ...     'abcde\n',
    ... ]
    >>> res = Stripper.strip_prefix(lines)
    >>> res.prefix
    'abc'
    >>> list(res.lines)
    ['defg\n', '\n', 'de\n']

    If no prefix is common, nothing should be stripped.

    >>> lines = [
    ...     'abcd\n',
    ...     '1234\n',
    ... ]
    >>> res = Stripper.strip_prefix(lines)
    >>> res.prefix = ''
    >>> list(res.lines)
    ['abcd\n', '1234\n']
    """

    def __init__(self, prefix, lines):
        self.prefix = prefix
        self.lines = map(self, lines)

    @classmethod
    def strip_prefix(cls, lines):
        prefix_lines, lines = itertools.tee(lines)
        prefix = functools.reduce(cls.common_prefix, prefix_lines)
        return cls(prefix, lines)

    def __call__(self, line):
        if not self.prefix:
            return line
        null, prefix, rest = line.partition(self.prefix)
        return rest

    @staticmethod
    def common_prefix(s1, s2):
        """
        Return the common prefix of two lines.
        """
        index = min(len(s1), len(s2))
        while s1[:index] != s2[:index]:
            index -= 1
        return s1[:index]


def remove_prefix(text, prefix):
    """
    Remove the prefix from the text if it exists.

    >>> remove_prefix('underwhelming performance', 'underwhelming ')
    'performance'

    >>> remove_prefix('something special', 'sample')
    'something special'
    """
    null, prefix, rest = text.rpartition(prefix)
    return rest


def remove_suffix(text, suffix):
    """
    Remove the suffix from the text if it exists.

    >>> remove_suffix('name.git', '.git')
    'name'

    >>> remove_suffix('something special', 'sample')
    'something special'
    """
    rest, suffix, null = text.partition(suffix)
    return rest


def normalize_newlines(text):
    r"""
    Replace alternate newlines with the canonical newline.

    >>> normalize_newlines('Lorem Ipsum\u2029')
    'Lorem Ipsum\n'
    >>> normalize_newlines('Lorem Ipsum\r\n')
    'Lorem Ipsum\n'
    >>> normalize_newlines('Lorem Ipsum\x85')
    'Lorem Ipsum\n'
    """
    newlines = ['\r\n', '\r', '\n', '\u0085', '\u2028', '\u2029']
    pattern = '|'.join(newlines)
    return re.sub(pattern, '\n', text)


def _nonblank(str):
    return str and not str.startswith('#')


@functools.singledispatch
def yield_lines(iterable):
    r"""
    Yield valid lines of a string or iterable.

    >>> list(yield_lines(''))
    []
    >>> list(yield_lines(['foo', 'bar']))
    ['foo', 'bar']
    >>> list(yield_lines('foo\nbar'))
    ['foo', 'bar']
    >>> list(yield_lines('\nfoo\n#bar\nbaz #comment'))
    ['foo', 'baz #comment']
    >>> list(yield_lines(['foo\nbar', 'baz', 'bing\n\n\n']))
    ['foo', 'bar', 'baz', 'bing']
    """
    return itertools.chain.from_iterable(map(yield_lines, iterable))


@yield_lines.register(str)
def _(text):
    return filter(_nonblank, map(str.strip, text.splitlines()))


def drop_comment(line):
    """
    Drop comments.

    >>> drop_comment('foo # bar')
    'foo'

    A hash without a space may be in a URL.

    >>> drop_comment('http://example.com/foo#bar')
    'http://example.com/foo#bar'
    """
    return line.partition(' #')[0]


def join_continuation(lines):
    r"""
    Join lines continued by a trailing backslash.

    >>> list(join_continuation(['foo \\', 'bar', 'baz']))
    ['foobar', 'baz']
    >>> list(join_continuation(['foo \\', 'bar', 'baz']))
    ['foobar', 'baz']
    >>> list(join_continuation(['foo \\', 'bar \\', 'baz']))
    ['foobarbaz']

    Not sure why, but...
    The character preceding the backslash is also elided.

    >>> list(join_continuation(['goo\\', 'dly']))
    ['godly']

    A terrible idea, but...
    If no line is available to continue, suppress the lines.

    >>> list(join_continuation(['foo', 'bar\\', 'baz\\']))
    ['foo']
    """
    lines = iter(lines)
    for item in lines:
        while item.endswith('\\'):
            try:
                item = item[:-2].strip() + next(lines)
            except StopIteration:
                return
        yield item


def read_newlines(filename, limit=1024):
    r"""
    >>> tmp_path = getfixture('tmp_path')
    >>> filename = tmp_path / 'out.txt'
    >>> _ = filename.write_text('foo\n', newline='', encoding='utf-8')
    >>> read_newlines(filename)
    '\n'
    >>> _ = filename.write_text('foo\r\n', newline='', encoding='utf-8')
    >>> read_newlines(filename)
    '\r\n'
    >>> _ = filename.write_text('foo\r\nbar\nbing\r', newline='', encoding='utf-8')
    >>> read_newlines(filename)
    ('\r', '\n', '\r\n')
    """
    with open(filename, encoding='utf-8') as fp:
        fp.read(limit)
    return fp.newlines
