"""Text wrapping shared by analytics figures."""

import textwrap


def wrap_text(text: str, width: int = 50) -> str:
    """Insert HTML line breaks at the requested wrapping width."""
    return "<br>".join(textwrap.wrap(text, width=width))
