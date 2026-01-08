# Pulisci_testo()

import re
import html

# Emoji (come la tua)
emoji_pattern = re.compile(
    "["
    "\U0001F000-\U0001FFFF"
    "\u2600-\u26FF"
    "\u2700-\u27BF"
    "]+",
    flags=re.UNICODE
)

# opzionale: regex per ripulire spazi/punteggiatura "strana"
MULTISPACE_RE = re.compile(r"\s+")
DATE_RE = re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b")

def pulisci_testo(t: str) -> str:
    """
    Pulizia unica coerente per MC + IronITA.
    Obiettivo: rendere l'input simile a quello visto dal modello in training.
    """
    t = str(t)

    # 0) decode HTML entities (&gt; &amp; ecc.)
    t = html.unescape(t)

    # 1) rimuovi link
    t = re.sub(r"http\S+|www\.\S+", " ", t)

    # 2) sostituisci @mention (IronITA) con token generico
    t = re.sub(r"@\w+", " account ", t)

    # 3) rimuovi parentesi quadre (IronITA)
    t = t.replace("[", " ").replace("]", " ")

    # 4) gestisci hashtag: togli solo il # ma tieni la parola
    #    "#labuonascuola" -> "labuonascuola"
    t = re.sub(r"#(\w+)", r"\1", t)

    # 5) rimuovi emoji
    t = emoji_pattern.sub("", t)

    # 6) normalizza apostrofi e lowercase
    t = t.replace("’", "'").replace("‘", "'")
    t = t.lower()

    # 7) rimuovi date tipo 19/01/2012
    t = DATE_RE.sub(" ", t)

    # 8) separa numeri e lettere attaccati: 5figli -> 5 figli
    t = re.sub(r"(\d)([a-zàèéìòù])", r"\1 \2", t, flags=re.IGNORECASE)
    t = re.sub(r"([a-zàèéìòù])(\d)", r"\1 \2", t, flags=re.IGNORECASE)

    # 9) rimuovi markup residuo tipo < > / e sequenze di trattini/frecce
    t = re.sub(r"[<>/]", " ", t)
    t = re.sub(r"-{2,}|\>{2,}", " ", t)

    # 10) rimuovi backslash e trattini singoli rimasti
    t = re.sub(r"[-\\]", " ", t)

    # 11) rimuovi replacement char
    t = t.replace("�", "")

    # 12) comprimi spazi e strip
    t = MULTISPACE_RE.sub(" ", t).strip()

    return t