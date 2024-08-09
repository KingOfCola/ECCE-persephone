import re

DEFAULT_SEP = ["^", r"[\.!\?]\s+"]


def capitalize(string: str, sep: list = None) -> str:
    """
    Capitalize the first letter of each sentence in a string.

    Parameters
    ----------
    string : str
        The string to capitalize.
    sep : list, optional
        A list of separators that define the sentence boundaries.
        If None, the default separators are used.
        Default is None.

    Returns
    -------
    str
        The capitalized string
    """
    if sep is None:
        sep = DEFAULT_SEP
    elif isinstance(sep, str):
        sep = [sep] + DEFAULT_SEP
    else:
        sep = sep + DEFAULT_SEP

    sep_pattern = "|".join(sep)
    pattern = f"({sep_pattern})(\w)"

    return re.sub(
        pattern, lambda x: f"{x.group(1)}{x.group(2).upper()}", string.lower()
    )
