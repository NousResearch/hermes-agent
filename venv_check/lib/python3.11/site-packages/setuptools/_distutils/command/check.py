"""distutils.command.check

Implements the Distutils 'check' command.
"""

import contextlib
from typing import ClassVar

from ..core import Command
from ..errors import DistutilsSetupError

with contextlib.suppress(ImportError):
    import docutils.frontend
    import docutils.nodes
    import docutils.parsers.rst
    import docutils.utils

    class SilentReporter(docutils.utils.Reporter):
        def __init__(
            self,
            source,
            report_level,
            halt_level,
            stream=None,
            debug=False,
            encoding='ascii',
            error_handler='replace',
        ):
            self.messages = []
            super().__init__(
                source, report_level, halt_level, stream, debug, encoding, error_handler
            )

        def system_message(self, level, message, *children, **kwargs):
            self.messages.append((level, message, children, kwargs))
            return docutils.nodes.system_message(
                message, *children, level=level, type=self.levels[level], **kwargs
            )


class check(Command):
    """This command checks the meta-data of the package."""

    description = "perform some checks on the package"
    user_options: ClassVar[list[tuple[str, str, str]]] = [
        ('metadata', 'm', 'Verify meta-data'),
        (
            'restructuredtext',
            'r',
            'Checks if long string meta-data syntax are reStructuredText-compliant',
        ),
        ('strict', 's', 'Will exit with an error if a check fails'),
    ]

    boolean_options: ClassVar[list[str]] = ['metadata', 'restructuredtext', 'strict']

    def initialize_options(self):
        """Sets default values for options."""
        self.restructuredtext = False
        self.metadata = 1
        self.strict = False
        self._warnings = 0

    def finalize_options(self):
        pass

    def warn(self, msg):
        """Counts the number of warnings that occurs."""
        self._warnings += 1
        return Command.warn(self, msg)

    def run(self):
        """Runs the command."""
        # perform the various tests
        if self.metadata:
            self.check_metadata()
        if self.restructuredtext:
            if 'docutils' in globals():
                try:
                    self.check_restructuredtext()
                except TypeError as exc:
                    raise DistutilsSetupError(str(exc))
            elif self.strict:
                raise DistutilsSetupError('The docutils package is needed.')

        # let's raise an error in strict mode, if we have at least
        # one warning
        if self.strict and self._warnings > 0:
            raise DistutilsSetupError('Please correct your package.')

    def check_metadata(self):
        """Ensures that all required elements of meta-data are supplied.

        Required fields:
            name, version

        Warns if any are missing.
        """
        metadata = self.distribution.metadata

        missing = [
            attr for attr in ('name', 'version') if not getattr(metadata, attr, None)
        ]

        if missing:
            self.warn("missing required meta-data: {}".format(', '.join(missing)))

    def check_restructuredtext(self):
        """Checks if the long string fields are reST-compliant."""
        data = self.distribution.get_long_description()
        for warning in self._check_rst_data(data):
            line = warning[-1].get('line')
            if line is None:
                warning = warning[1]
            else:
                warning = f'{warning[1]} (line {line})'
            self.warn(warning)

    def _check_rst_data(self, data):
        """Returns warnings when the provided data doesn't compile."""
        # the include and csv_table directives need this to be a path
        source_path = self.distribution.script_name or 'setup.py'
        parser = docutils.parsers.rst.Parser()
        settings = docutils.frontend.OptionParser(
            components=(docutils.parsers.rst.Parser,)
        ).get_default_values()
        settings.tab_width = 4
        settings.pep_references = None
        settings.rfc_references = None
        reporter = SilentReporter(
            source_path,
            settings.report_level,
            settings.halt_level,
            stream=settings.warning_stream,
            debug=settings.debug,
            encoding=settings.error_encoding,
            error_handler=settings.error_encoding_error_handler,
        )

        document = docutils.nodes.document(settings, reporter, source=source_path)
        document.note_source(source_path, -1)
        try:
            parser.parse(data, document)
        except (AttributeError, TypeError) as e:
            reporter.messages.append((
                -1,
                f'Could not finish the parsing: {e}.',
                '',
                {},
            ))

        return reporter.messages
