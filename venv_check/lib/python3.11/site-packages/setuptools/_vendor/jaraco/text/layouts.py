qwerty = "-=qwertyuiop[]asdfghjkl;'zxcvbnm,./_+QWERTYUIOP{}ASDFGHJKL:\"ZXCVBNM<>?"
dvorak = "[]',.pyfgcrl/=aoeuidhtns-;qjkxbmwvz{}\"<>PYFGCRL?+AOEUIDHTNS_:QJKXBMWVZ"


to_dvorak = str.maketrans(qwerty, dvorak)
to_qwerty = str.maketrans(dvorak, qwerty)


def translate(input, translation):
    """
    >>> translate('dvorak', to_dvorak)
    'ekrpat'
    >>> translate('qwerty', to_qwerty)
    'x,dokt'
    """
    return input.translate(translation)


def _translate_stream(stream, translation):
    """
    >>> import io
    >>> _translate_stream(io.StringIO('foo'), to_dvorak)
    urr
    """
    print(translate(stream.read(), translation))
