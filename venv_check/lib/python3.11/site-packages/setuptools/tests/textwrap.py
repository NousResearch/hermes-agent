import textwrap


def DALS(s):
    "dedent and left-strip"
    return textwrap.dedent(s).lstrip()
