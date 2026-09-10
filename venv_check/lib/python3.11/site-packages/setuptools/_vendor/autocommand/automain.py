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
from .errors import AutocommandError


class AutomainRequiresModuleError(AutocommandError, TypeError):
    pass


def automain(module, *, args=(), kwargs=None):
    '''
    This decorator automatically invokes a function if the module is being run
    as the "__main__" module. Optionally, provide args or kwargs with which to
    call the function. If `module` is "__main__", the function is called, and
    the program is `sys.exit`ed with the return value. You can also pass `True`
    to cause the function to be called unconditionally. If the function is not
    called, it is returned unchanged by the decorator.

    Usage:

    @automain(__name__)  # Pass __name__ to check __name__=="__main__"
    def main():
        ...

    If __name__ is "__main__" here, the main function is called, and then
    sys.exit called with the return value.
    '''

    # Check that @automain(...) was called, rather than @automain
    if callable(module):
        raise AutomainRequiresModuleError(module)

    if module == '__main__' or module is True:
        if kwargs is None:
            kwargs = {}

        # Use a function definition instead of a lambda for a neater traceback
        def automain_decorator(main):
            sys.exit(main(*args, **kwargs))

        return automain_decorator
    else:
        return lambda main: main
