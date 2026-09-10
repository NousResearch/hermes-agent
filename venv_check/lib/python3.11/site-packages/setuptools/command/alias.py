from setuptools.command.setopt import config_file, edit_config, option_base

from distutils.errors import DistutilsOptionError


def shquote(arg):
    """Quote an argument for later parsing by shlex.split()"""
    for c in '"', "'", "\\", "#":
        if c in arg:
            return repr(arg)
    if arg.split() != [arg]:
        return repr(arg)
    return arg


class alias(option_base):
    """Define a shortcut that invokes one or more commands"""

    description = "define a shortcut to invoke one or more commands"
    command_consumes_arguments = True

    user_options = [
        ('remove', 'r', 'remove (unset) the alias'),
    ] + option_base.user_options

    boolean_options = option_base.boolean_options + ['remove']

    def initialize_options(self):
        option_base.initialize_options(self)
        self.args = None
        self.remove = None

    def finalize_options(self) -> None:
        option_base.finalize_options(self)
        if self.remove and len(self.args) != 1:
            raise DistutilsOptionError(
                "Must specify exactly one argument (the alias name) when using --remove"
            )

    def run(self) -> None:
        aliases = self.distribution.get_option_dict('aliases')

        if not self.args:
            print("Command Aliases")
            print("---------------")
            for alias in aliases:
                print("setup.py alias", format_alias(alias, aliases))
            return

        elif len(self.args) == 1:
            (alias,) = self.args
            if self.remove:
                command = None
            elif alias in aliases:
                print("setup.py alias", format_alias(alias, aliases))
                return
            else:
                print(f"No alias definition found for {alias!r}")
                return
        else:
            alias = self.args[0]
            command = ' '.join(map(shquote, self.args[1:]))

        edit_config(self.filename, {'aliases': {alias: command}}, self.dry_run)


def format_alias(name, aliases):
    source, command = aliases[name]
    if source == config_file('global'):
        source = '--global-config '
    elif source == config_file('user'):
        source = '--user-config '
    elif source == config_file('local'):
        source = ''
    else:
        source = f'--filename={source!r}'
    return source + name + ' ' + command
