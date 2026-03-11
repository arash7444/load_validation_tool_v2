import re
from typing import Iterable, Set

# Matches e.g. "Horizontal Wind Speed (m/s) at 79m" -> 79
_SPEED_RE = re.compile(r"Horizontal\s+Wind\s+Speed\s*\(m/s\)\s*at\s*(\d+)m", re.IGNORECASE)
# Matches e.g. "Wind Direction (deg) at 79m" -> 79
_DIR_RE   = re.compile(r"Wind\s+Direction\s*\(deg\)\s*at\s*(\d+)m", re.IGNORECASE)

def detect_heights(headers: Iterable[str]) -> Set[int]:
    """Return all heights found in wide-column headers (speed/direction)."""
    heights: Set[int] = set()
    for col in headers:
        col = col.strip()
        m1 = _SPEED_RE.fullmatch(col)
        m2 = _DIR_RE.fullmatch(col)
        if m1:
            heights.add(int(m1.group(1)))
        if m2:
            heights.add(int(m2.group(1)))
    return heights


def color_text(text, color=None, background=None, bold=False, underline=False):
    """
    Print colored and styled text using ANSI escape codes.
    
    Parameters
    ----------
    text : str
        The text to colorize
    color : str, optional
        One of ['black', 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white']
    background : str, optional
        Same color options as 'color'
    bold : bool, optional
        Make text bold
    underline : bool, optional
        Underline text

    """
    
    # Basic color maps
    colors = {
        "black": 30, "red": 31, "green": 32, "yellow": 33,
        "blue": 34, "magenta": 35, "cyan": 36, "white": 37
    }
    
    style = []
    
    # Bold and underline
    if bold:
        style.append("1")
    if underline:
        style.append("4")
    
    # Foreground color

    if color in colors:
        style.append(str(colors[color]))
    else:
        style.append(str(colors['black']))
    
    # Background color
    if background in colors:
        style.append(str(colors[background] + 10))
    
    # Combine style parts
    start = f"\033[{';'.join(style)}m" if style else ""
    end = "\033[0m"
    
    return f"{start}{text}{end}"




def NA_cols(df):
    """
    This function take a dataframe as input
    and returns a list of columns with missing values
    """
    missing_columns = df.isna().any()
    missing_columns = missing_columns[missing_columns].index.tolist()

    missing_columns_2 = df.columns[df.isna().any()]
    return missing_columns, missing_columns_2