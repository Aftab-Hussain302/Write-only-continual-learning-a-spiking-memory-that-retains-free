"""Pre-submission scan of a PDF's text layer for what similarity checkers report as 'replaced characters' (homoglyphs): characters
from another script, or letter-like symbols, inside words written in Latin letters. An approximation of such checkers (their exact
rules are not public). Categories: (A) Greek or Cyrillic letters that look like Latin letters inside Latin words; (B) letter-like
symbols inside Latin words (micro sign, modifier accents, dotless i, combining marks); (C) digits 0 and 1 glued to Latin letters;
(D) any character outside Basic Latin, Latin-1 and the usual math and punctuation blocks."""
import re, unicodedata, collections


def _script(c):
    n = unicodedata.name(c, "")
    for s in ("LATIN", "GREEK", "CYRILLIC", "ARMENIAN", "HEBREW", "ARABIC", "CHEROKEE", "COPTIC"):
        if n.startswith(s): return s
    return "COMMON"


LETTERLIKE = set("µˆ˘˙ˇ˜ıȷ") | {chr(c) for c in range(0x0300, 0x0370)}
CONFUSABLE = set("ΑΒΕΖΗΙΚΜΝΟΡΤΥΧΣαγικνορτυχϲϳϴ" "АВЕКМНОРСТУХаеорсуххіјѕԁԛԝ")
OK_BLOCKS = [(0x20, 0x7E), (0xA0, 0xFF), (0x2010, 0x2027), (0x2030, 0x205E), (0x2190, 0x21FF), (0x2200, 0x22FF), (0x0391, 0x03C9)]


def scan(path):
    import pymupdf
    d = pymupdf.open(path); out = collections.defaultdict(list)
    for pno, page in enumerate(d, 1):
        for w in page.get_text("words"):
            t = w[4]
            if not re.search(r"[A-Za-z]", t): continue
            for c in t:
                sc = _script(c)
                if sc not in ("LATIN", "COMMON"): out["A" if c in CONFUSABLE else "a"].append((pno, c, t))
                elif c in LETTERLIKE: out["B"].append((pno, c, t))
            if re.search(r"[A-Za-z][01]|[01][A-Za-z]", t): out["C"].append((pno, "", t))
        for c in page.get_text():
            if not any(a <= ord(c) <= b for a, b in OK_BLOCKS) and c not in "\n\t": out["D"].append((pno, c, f"U+{ord(c):04X} {unicodedata.name(c, '?')}"))
    return out


def report(paths):
    lines = []
    for path in paths:
        o = scan(path)
        lines.append(f"{path}: A look-alike letters = {len(o['A'])}; B letter-like symbols = {len(o['B'])}; C digits 0/1 glued to letters = {len(o['C'])}; "
                     f"D unusual characters = {len(o['D'])}; (info: other Greek letters in Latin words = {len(o['a'])})")
        for k in "ABD":
            for pno, c, t in o[k][:25]: lines.append(f"   [{k}] p{pno}: {c!r} in {t!r}")
        if o["C"]: lines.append("   [C] e.g. " + ", ".join(repr(t) for _, _, t in o["C"][:12]))
    return "\n".join(lines) + "\n"
