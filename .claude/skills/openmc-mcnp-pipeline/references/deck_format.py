"""MCNP 6.3 section 3.2.2: 128 columns after eight-column tab expansion.

Format generated decks (title first, no optional MESSAGE block). Never split a
data token. Comments may wrap as separate C cards; data uses five-space indents.
"""
import re
import textwrap

MAX_COLUMNS = 128


def overlong_lines(text):
    return [(i, len(line.expandtabs(8))) for i, line in enumerate(text.splitlines(), 1)
            if len(line.expandtabs(8)) > MAX_COLUMNS]


def comment_lines(text):
    return textwrap.wrap(text, width=MAX_COLUMNS, initial_indent="c ",
                         subsequent_indent="c ", break_long_words=True,
                         break_on_hyphens=False) or ["c "]


def format_deck(text):
    lines = text.splitlines()
    if not lines:
        return text
    out = []
    for number, raw in enumerate(lines, 1):
        line = raw.expandtabs(8).rstrip()
        if number == 1:
            # A title cannot continue as a data card. Preserve its remainder as comments.
            out.append(line[:MAX_COLUMNS])
            if len(line) > MAX_COLUMNS:
                out.extend(comment_lines("Title continued: " + line[MAX_COLUMNS:]))
            continue
        if len(line) <= MAX_COLUMNS:
            out.append(line)
            continue
        if re.match(r"^ {0,4}[cC](?:\s|$)", line) or line.lstrip().startswith("$"):
            out.extend(comment_lines(line.lstrip()[1:].lstrip()))
            continue
        data, sep, comment = line.partition("$")
        # Full-line comments may precede a continuation; they must not become data.
        if sep:
            out.extend(comment_lines(comment.strip()))
        data = data.rstrip()
        continued = data.endswith("&")
        if continued:
            data = data[:-1].rstrip()
        # Generated cards do not have quoted strings. Refuse unfamiliar syntax rather
        # than introduce a newline inside a quoted path/string.
        if "'" in data or '"' in data:
            raise ValueError(f"MCNP line {number}: cannot safely wrap a quoted data entry.")
        width = MAX_COLUMNS - (2 if continued else 0)
        current = "     " if data.startswith("     ") else ""
        chunks = []
        for token in data.split():
            gap = " " if current.strip() else ""
            if len(current) + len(gap) + len(token) > width:
                if current.strip():
                    chunks.append(current)
                current = "     "
                gap = ""
            if len(current) + len(token) > width:
                raise ValueError(f"MCNP line {number}: a data token cannot fit within {MAX_COLUMNS} columns.")
            current += gap + token
        if current.strip():
            chunks.append(current + (" &" if continued else ""))
        out.extend(chunks)
    return "\n".join(out) + ("\n" if text.endswith(("\n", "\r")) else "")
