from __future__ import annotations

from typing import Any

from typeguard._config import TypeCheckConfiguration, global_config


class TypeCheckMemo:
    """
    Contains information necessary for type checkers to do their work.

    .. attribute:: globals
       :type: dict[str, Any]

        Dictionary of global variables to use for resolving forward references.

    .. attribute:: locals
       :type: dict[str, Any]

        Dictionary of local variables to use for resolving forward references.

    .. attribute:: self_type
       :type: type | None

        When running type checks within an instance method or class method, this is the
        class object that the first argument (usually named ``self`` or ``cls``) refers
        to.

    .. attribute:: config
       :type: TypeCheckConfiguration

         Contains the configuration for a particular set of type checking operations.
    """

    __slots__ = "globals", "locals", "self_type", "config"

    def __init__(
        self,
        globals: dict[str, Any],
        locals: dict[str, Any],
        *,
        self_type: type | None = None,
        config: TypeCheckConfiguration = global_config,
    ):
        self.globals = globals
        self.locals = locals
        self.self_type = self_type
        self.config = config
