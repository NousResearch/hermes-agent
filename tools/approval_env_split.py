"""GNU ``env -S`` / ``--split-string`` argv projection for approval detection (moved from approval_detection)."""


def _split_env_string(payload: str) -> list[str] | None:
    r"""Project GNU env -S literal argv, not POSIX shell words.

    Dynamic ${NAME} expansion is deliberately not evaluated: the execution
    backend's environment need not be this process's environment.
    """
    escapes = {"f": "\f", "n": "\n", "r": "\r", "t": "\t", "v": "\v",
               "#": "#", "$": "$", "\"": "\"", "'": "'", "\\": "\\"}
    args, word = [], []
    quote, started, index = None, False, 0
    while index < len(payload):
        char = payload[index]
        index += 1
        if char == "\\":
            if index == len(payload):
                return None
            escaped = payload[index]
            if quote == "'" and escaped not in ("'", "\\"):
                word.append(char)
                started = True
                continue
            index += 1
            if escaped == "c":
                if quote:
                    return None
                break
            if escaped == "_" and quote is None:
                if started:
                    args.append("".join(word))
                word, started = [], False
                continue
            if escaped not in escapes and escaped != "_":
                return None
            word.append(" " if escaped == "_" else escapes[escaped])
            started = True
            continue
        if char in ("'", '"') and (quote is None or char == quote):
            quote = char if quote is None else None
            started = True
            continue
        if quote is None and char in " \t\n\r\v\f":
            if started:
                args.append("".join(word))
            word, started = [], False
            continue
        if quote is None and char == "#" and not started:
            break
        if char == "$" and quote != "'":
            return None
        word.append(char)
        started = True
    if quote:
        return None
    if started:
        args.append("".join(word))
    return args
