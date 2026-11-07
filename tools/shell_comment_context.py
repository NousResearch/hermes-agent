"""Word-boundary state for the shared shell scanner; never executes expansions."""


class ShellCommentContext:
    """Remember lexical boundaries without changing the scanner's event stream.

    Parentheses close either a shell operator or a word component. Treating all
    closers alike either hides executable suffixes or scans genuine comments.
    """

    def __init__(self):
        self.word_start = True
        self.previous = ""
        self.parens: list[tuple[str, bool]] = []
        self.conditional = False
        self.regex_word = 0  # 1: waiting for =~ operand; 2: inside its word

    def starts_comment(self) -> bool:
        return self.word_start and not any(arithmetic for _, arithmetic in self.parens)

    def advance(self, text: str, kind: str, i: int, j: int, quote: str | None) -> None:
        if kind == "esc" and text[i:j] == "\\\n":
            return
        if kind == "comment":
            self.word_start, self.previous = True, ""
        elif kind != "char" or quote is not None:
            self.word_start, self.previous = False, ""
            if self.regex_word:
                self.regex_word = 2
        else:
            self._character(text[i])

    def _character(self, char: str) -> None:
        if self._regex_character(char):
            return
        if char == "(":
            word = self.previous if self.previous and self.previous in "$<>@?*+!=" else ""
            arithmetic = (self.previous == "(" and bool(self.parens)
                          and self.parens[-1][0] in ("", "$"))
            if arithmetic:
                outer_word, _ = self.parens[-1]
                self.parens[-1] = (outer_word, True)
            self.parens.append((word, arithmetic))
            self.word_start = True
        elif char == ")":
            word, _ = self.parens.pop() if self.parens else ("", False)
            self.word_start = not word
        else:
            self.word_start = char in " \t\n;&|<>"
        self.previous = char

    def _regex_character(self, char: str) -> bool:
        if self.previous == char == "[":
            self.conditional = True
        elif self.previous == char == "]":
            self.conditional, self.regex_word = False, 0
        if self.conditional and self.previous == "=" and char == "~":
            self.regex_word = 1
        elif char in " \t\n":
            if self.regex_word == 2:
                self.regex_word = 0
        elif self.regex_word:
            self.regex_word = 2
            self.word_start, self.previous = False, char
            return True
        return False


class ShellSubstitutionDepth:
    """Match a substitution closer without mistaking a case-pattern ``)`` for it."""

    def __init__(self):
        self.depth = 1
        self.cases: list[tuple[str, int]] = []
        self.word = ""
        self.command_start = True
        self.previous = ""

    def advance(self, text: str, kind: str, i: int, j: int, quote: str | None) -> bool:
        if kind == "esc" and text[i:j] == "\\\n":
            return False
        if kind == "comment":
            self._finish_word()
            return False
        if kind != "char" or quote is not None:
            self.word += "?"  # quoted/escaped keywords are ordinary words
            return False
        char = text[i]
        if char in " \t\n;&|()<>":
            self._finish_word()
            self._operator(char)
        else:
            self.word += char
        self.previous = char
        return self.depth == 0

    def _finish_word(self) -> None:
        if not self.word:
            return
        word, self.word = self.word, ""
        if word == "case" and self.command_start:
            self.cases.append(("subject", self.depth))
        elif self.cases:
            phase, depth = self.cases[-1]
            if phase == "subject":
                self.cases[-1] = ("in", depth)
            elif phase == "in" and word == "in":
                self.cases[-1] = ("pattern", depth)
            elif phase == "pattern" and word == "esac":
                self.cases.pop()
        self.command_start = self.command_start and word in {
            "then", "do", "else", "elif", "if", "while", "until", "!", "time",
        }

    def _operator(self, char: str) -> None:
        phase, depth = self.cases[-1] if self.cases else ("", -1)
        pattern = phase == "pattern" and depth == self.depth
        if char == "(":
            # A leading '(' on a case pattern is optional shell syntax.
            if not (pattern and self.previous in " \t\n;|"):
                self.depth += 1
            self.command_start = True
        elif char == ")":
            if pattern:
                self.cases[-1] = ("body", depth)
            else:
                self.depth -= 1
            self.command_start = True
        elif char in ";&" and self.previous == ";" and self.cases:
            self.cases[-1] = ("pattern", depth)
            self.command_start = True
        elif char in ";&|\n":
            self.command_start = True
