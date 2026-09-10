# Copyright 2014-2015 Nathan West
#
# This file is part of autocommand.
#
# autocommand is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# autocommand is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with autocommand.  If not, see <http://www.gnu.org/licenses/>.

import sys
from re import compile as compile_regex
from inspect import signature, getdoc, Parameter
from argparse import ArgumentParser
from contextlib import contextmanager
from functools import wraps
from io import IOBase
from autocommand.errors import AutocommandError


_empty = Parameter.empty


class AnnotationError(AutocommandError):
    '''Annotation error: annotation must be a string, type, or tuple of both'''


class PositionalArgError(AutocommandError):
    '''
    Postional Arg Error: autocommand can't handle postional-only parameters
    '''


class KWArgError(AutocommandError):
    '''kwarg Error: autocommand can't handle a **kwargs parameter'''


class DocstringError(AutocommandError):
    '''Docstring error'''


class TooManySplitsError(DocstringError):
    '''
    The docstring had too many ---- section splits. Currently we only support
    using up to a single split, to split the docstring into description and
    epilog parts.
    '''


def _get_type_description(annotation):
    '''
    Given an annotation, return the (type, description) for the parameter.
    If you provide an annotation that is somehow both a string and a callable,
    the behavior is undefined.
    '''
    if annotation is _empty:
        return None, None
    elif callable(annotation):
        return annotation, None
    elif isinstance(annotation, str):
        return None, annotation
    elif isinstance(annotation, tuple):
        try:
            arg1, arg2 = annotation
        except ValueError as e:
            raise AnnotationError(annotation) from e
        else:
            if callable(arg1) and isinstance(arg2, str):
                return arg1, arg2
            elif isinstance(arg1, str) and callable(arg2):
                return arg2, arg1

    raise AnnotationError(annotation)


def _add_arguments(param, parser, used_char_args, add_nos):
    '''
    Add the argument(s) to an ArgumentParser (using add_argument) for a given
    parameter. used_char_args is the set of -short options currently already in
    use, and is updated (if necessary) by this function. If add_nos is True,
    this will also add an inverse switch for all boolean options. For
    instance, for the boolean parameter "verbose", this will create --verbose
    and --no-verbose.
    '''

    # Impl note: This function is kept separate from make_parser because it's
    # already very long and I wanted to separate out as much as possible into
    # its own call scope, to prevent even the possibility of suble mutation
    # bugs.
    if param.kind is param.POSITIONAL_ONLY:
        raise PositionalArgError(param)
    elif param.kind is param.VAR_KEYWORD:
        raise KWArgError(param)

    # These are the kwargs for the add_argument function.
    arg_spec = {}
    is_option = False

    # Get the type and default from the annotation.
    arg_type, description = _get_type_description(param.annotation)

    # Get the default value
    default = param.default

    # If there is no explicit type, and the default is present and not None,
    # infer the type from the default.
    if arg_type is None and default not in {_empty, None}:
        arg_type = type(default)

    # Add default. The presence of a default means this is an option, not an
    # argument.
    if default is not _empty:
        arg_spec['default'] = default
        is_option = True

    # Add the type
    if arg_type is not None:
        # Special case for bool: make it just a --switch
        if arg_type is bool:
            if not default or default is _empty:
                arg_spec['action'] = 'store_true'
            else:
                arg_spec['action'] = 'store_false'

            # Switches are always options
            is_option = True

        # Special case for file types: make it a string type, for filename
        elif isinstance(default, IOBase):
            arg_spec['type'] = str

        # TODO: special case for list type.
        #   - How to specificy type of list members?
        #       - param: [int]
        #       - param: int =[]
        #   - action='append' vs nargs='*'

        else:
            arg_spec['type'] = arg_type

    # nargs: if the signature includes *args, collect them as trailing CLI
    # arguments in a list. *args can't have a default value, so it can never be
    # an option.
    if param.kind is param.VAR_POSITIONAL:
        # TODO: consider depluralizing metavar/name here.
        arg_spec['nargs'] = '*'

    # Add description.
    if description is not None:
        arg_spec['help'] = description

    # Get the --flags
    flags = []
    name = param.name

    if is_option:
        # Add the first letter as a -short option.
        for letter in name[0], name[0].swapcase():
            if letter not in used_char_args:
                used_char_args.add(letter)
                flags.append('-{}'.format(letter))
                break

        # If the parameter is a --long option, or is a -short option that
        # somehow failed to get a flag, add it.
        if len(name) > 1 or not flags:
            flags.append('--{}'.format(name))

        arg_spec['dest'] = name
    else:
        flags.append(name)

    parser.add_argument(*flags, **arg_spec)

    # Create the --no- version for boolean switches
    if add_nos and arg_type is bool:
        parser.add_argument(
            '--no-{}'.format(name),
            action='store_const',
            dest=name,
            const=default if default is not _empty else False)


def make_parser(func_sig, description, epilog, add_nos):
    '''
    Given the signature of a function, create an ArgumentParser
    '''
    parser = ArgumentParser(description=description, epilog=epilog)

    used_char_args = {'h'}

    # Arange the params so that single-character arguments are first. This
    # esnures they don't have to get --long versions. sorted is stable, so the
    # parameters will otherwise still be in relative order.
    params = sorted(
        func_sig.parameters.values(),
        key=lambda param: len(param.name) > 1)

    for param in params:
        _add_arguments(param, parser, used_char_args, add_nos)

    return parser


_DOCSTRING_SPLIT = compile_regex(r'\n\s*-{4,}\s*\n')


def parse_docstring(docstring):
    '''
    Given a docstring, parse it into a description and epilog part
    '''
    if docstring is None:
        return '', ''

    parts = _DOCSTRING_SPLIT.split(docstring)

    if len(parts) == 1:
        return docstring, ''
    elif len(parts) == 2:
        return parts[0], parts[1]
    else:
        raise TooManySplitsError()


def autoparse(
        func=None, *,
        description=None,
        epilog=None,
        add_nos=False,
        parser=None):
    '''
    This decorator converts a function that takes normal arguments into a
    function which takes a single optional argument, argv, parses it using an
    argparse.ArgumentParser, and calls the underlying function with the parsed
    arguments. If it is not given, sys.argv[1:] is used. This is so that the
    function can be used as a setuptools entry point, as well as a normal main
    function. sys.argv[1:] is not evaluated until the function is called, to
    allow injecting different arguments for testing.

    It uses the argument signature of the function to create an
    ArgumentParser. Parameters without defaults become positional parameters,
    while parameters *with* defaults become --options. Use annotations to set
    the type of the parameter.

    The `desctiption` and `epilog` parameters corrospond to the same respective
    argparse parameters. If no description is given, it defaults to the
    decorated functions's docstring, if present.

    If add_nos is True, every boolean option (that is, every parameter with a
    default of True/False or a type of bool) will have a --no- version created
    as well, which inverts the option. For instance, the --verbose option will
    have a --no-verbose counterpart. These are not mutually exclusive-
    whichever one appears last in the argument list will have precedence.

    If a parser is given, it is used instead of one generated from the function
    signature. In this case, no parser is created; instead, the given parser is
    used to parse the argv argument. The parser's results' argument names must
    match up with the parameter names of the decorated function.

    The decorated function is attached to the result as the `func` attribute,
    and the parser is attached as the `parser` attribute.
    '''

    # If @autoparse(...) is used instead of @autoparse
    if func is None:
        return lambda f: autoparse(
            f, description=description,
            epilog=epilog,
            add_nos=add_nos,
            parser=parser)

    func_sig = signature(func)

    docstr_description, docstr_epilog = parse_docstring(getdoc(func))

    if parser is None:
        parser = make_parser(
            func_sig,
            description or docstr_description,
            epilog or docstr_epilog,
            add_nos)

    @wraps(func)
    def autoparse_wrapper(argv=None):
        if argv is None:
            argv = sys.argv[1:]

        # Get empty argument binding, to fill with parsed arguments. This
        # object does all the heavy lifting of turning named arguments into
        # into correctly bound *args and **kwargs.
        parsed_args = func_sig.bind_partial()
        parsed_args.arguments.update(vars(parser.parse_args(argv)))

        return func(*parsed_args.args, **parsed_args.kwargs)

    # TODO: attach an updated __signature__ to autoparse_wrapper, just in case.

    # Attach the wrapped function and parser, and return the wrapper.
    autoparse_wrapper.func = func
    autoparse_wrapper.parser = parser
    return autoparse_wrapper


@contextmanager
def smart_open(filename_or_file, *args, **kwargs):
    '''
    This context manager allows you to open a filename, if you want to default
    some already-existing file object, like sys.stdout, which shouldn't be
    closed at the end of the context. If the filename argument is a str, bytes,
    or int, the file object is created via a call to open with the given *args
    and **kwargs, sent to the context, and closed at the end of the context,
    just like "with open(filename) as f:". If it isn't one of the openable
    types, the object simply sent to the context unchanged, and left unclosed
    at the end of the context. Example:

        def work_with_file(name=sys.stdout):
            with smart_open(name) as f:
                # Works correctly if name is a str filename or sys.stdout
                print("Some stuff", file=f)
                # If it was a filename, f is closed at the end here.
    '''
    if isinstance(filename_or_file, (str, bytes, int)):
        with open(filename_or_file, *args, **kwargs) as file:
            yield file
    else:
        yield filename_or_file
