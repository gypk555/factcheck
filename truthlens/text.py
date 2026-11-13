"""Text cleaning shared by training and the web app.

The same function MUST be used in both places, otherwise the model sees
different text at demo time than it saw while learning.

Besides normal cleanup, it removes "leaky" phrases found in WELFake that
reveal the label without saying anything about whether the story is true
(e.g. "(Reuters)" appears only in real articles, "Featured image via" only
in fake ones). If we left them in, the model would score very high on the
test set by spotting the source, and then fail on anything else.
"""

import re
import unicodedata

# Source/credit patterns that leak the label. Order matters a little:
# datelines first, then credits, then generic URL / handle cleanup.
_LEAK_PATTERNS = [
    # "WASHINGTON (Reuters) - " style datelines, and any leftover "(Reuters)".
    r"^[^\n]{0,80}?\(reuters\)\s*[-–—]?\s*",
    r"\(reuters\)",
    r"\breuters\b",
    # Agency boilerplate: sponsor notes, correction notices, graphic links.
    r"this article was funded in part by [^\n]{0,200}(production|creation)\.?",
    r"[^\n.]{0,80}independently created by [^\n.]{0,80}\.?",
    r"[^\n.]{0,60}no editorial involvement[^\n.]{0,80}\.?",
    r"\(?\s*(this )?story (corrects|refiles|has been refiled|has been corrected)[^\n)]{0,120}\)?",
    r"\brefiled to [^\n)]{0,80}",
    r"\S{0,100}tmsnrt\.rs\S{0,200}",
    # Embedded tweets: "- Name (@handle) October 27, 2017" (before @handles are removed).
    r"\(\s*@\w+\s*\)\s*[a-z]+\.? \d{1,2},? \d{4}",
    r"\b((19|20)\d{2})(?=[A-Z])",  # year glued to the next word: "2017This"
    # Bylines / outlet tags.
    r"\bfollow (him|her|[a-z]+ [a-z]+) on twitter\b[^\n]{0,40}",
    r"\b21\s?wire\b",
    r"\bzero ?hedge\b",
    r"image credit[^\n]{0,80}",
    # Photo credits typical of fake-news sites.
    r"featured image[^\n]{0,120}",
    r"photo by [^\n]{0,80}",
    r"getty images",
    r"image via [^\n]{0,80}",
    r"21st century wire[^\n]{0,40}",
    r"\bvia\s+(twitter|youtube|facebook|instagram)\b",
    # Bracket tags like [VIDEO], [IMAGES], (VIDEO).
    r"[\[(]\s*(video|videos|image|images|photo|photos|tweet|tweets|watch)\s*[\])]",
    # Outlet names, bylines and boilerplate that identify the source
    # (e.g. "... - The New York Times", "... - Breitbart", "Factbox:", "Read more: WND").
    r"\b(the )?new york times\b",
    r"\bnytimes\b",
    r"\bbreitbart( news)?\b",
    r"\bdelingpole\b",
    r"\bfactbox\b",
    r"\bread more\b\s*(at\b|:)?[^\n.!?]{0,40}",
    # URLs, e-mails, @handles, pic.twitter links (also when split by spaces).
    r"https?:\s*//\s*\S*",
    r"\bhttps?\b",
    r"www\.\S+",
    r"pic\s*\.?\s*twitter\s*\.?\s*com\s*/?\S*",
    r"\bpic\s*\.\s*twitter\b\s*\.?",  # scraped form without ".com"
    r"\bt\.co/\S*",
    # E-mails. Possessive quantifiers (++) keep this linear-time: the naive
    # \S+@\S+\.\S+ takes minutes on 20,000 "@" characters (ReDoS).
    r"(?<!\S)[^\s@]++@[^\s@.]++\.\S*+",
    r"@\w+",
    # Empty brackets left behind by the removals above.
    r"[(\[]\s*[)\]]",
]
_LEAK_RE = [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in _LEAK_PATTERNS]
_SPACE_RE = re.compile(r"[^\S\n]+")      # runs of spaces/tabs (not newlines)
_NEWLINES_RE = re.compile(r"\s*\n\s*(\n\s*)*")


def clean_text(text: str, max_chars: int | None = None) -> str:
    """Normalise a news headline/article into the form the model expects.

    max_chars caps the text AFTER Unicode normalisation, which can expand some
    characters up to 18x (e.g. U+FDFA). The web app sets it; training does not.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", str(text))
    if max_chars is not None:
        text = text[:max_chars]
    # Curly quotes / dashes -> plain ASCII so "’" and "'" are the same token.
    text = (
        text.replace("‘", "'").replace("’", "'")
        .replace("“", '"').replace("”", '"')
        .replace("–", "-").replace("—", "-")
    )
    for rx in _LEAK_RE:
        text = rx.sub(" ", text)
    text = _SPACE_RE.sub(" ", text)
    # Keep paragraph breaks (useful when showing the text back), at most one blank line.
    text = _NEWLINES_RE.sub(lambda m: "\n\n" if m.group(0).count("\n") > 1 else "\n", text)
    return text.strip()


def join_title_body(title: str | None, body: str | None) -> str:
    """Combine headline and article body the same way everywhere.

    Clean title and body SEPARATELY before joining: the dateline rule
    ("WASHINGTON (Reuters) - ") is anchored to the start of a line, and on a
    joined string it would also swallow the headline.
    """
    parts = [p.strip() for p in (title or "", body or "") if p and str(p).strip()]
    return " . ".join(parts)
