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

from .autoparse import autoparse
from .automain import automain
try:
    from .autoasync import autoasync
except ImportError:  # pragma: no cover
    pass


def autocommand(
        module, *,
        description=None,
        epilog=None,
        add_nos=False,
        parser=None,
        loop=None,
        forever=False,
        pass_loop=False):

    if callable(module):
        raise TypeError('autocommand requires a module name argument')

    def autocommand_decorator(func):
        # Step 1: if requested, run it all in an asyncio event loop. autoasync
        # patches the __signature__ of the decorated function, so that in the
        # event that pass_loop is True, the `loop` parameter of the original
        # function will *not* be interpreted as a command-line argument by
        # autoparse
        if loop is not None or forever or pass_loop:
            func = autoasync(
                func,
                loop=None if loop is True else loop,
                pass_loop=pass_loop,
                forever=forever)

        # Step 2: create parser. We do this second so that the arguments are
        # parsed and passed *before* entering the asyncio event loop, if it
        # exists. This simplifies the stack trace and ensures errors are
        # reported earlier. It also ensures that errors raised during parsing &
        # passing are still raised if `forever` is True.
        func = autoparse(
            func,
            description=description,
            epilog=epilog,
            add_nos=add_nos,
            parser=parser)

        # Step 3: call the function automatically if __name__ == '__main__' (or
        # if True was provided)
        func = automain(module)(func)

        return func

    return autocommand_decorator
