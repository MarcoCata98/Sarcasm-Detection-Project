# Tokenize()

import re


APOSTROFE_PREFIX = {
    "l",      # l'energia
    "un",     # un'amica
    "d",      # d'annunzio
    "del", "dell",   # dell'abitazione
    "al", "all",     # all'ingresso
    "dal", "dall",   # dall'alba
    "nel", "nell",   # nell'armadio
    "sul", "sull",   # sull'acqua
    "col", "coll",   # coll'amico (più raro)
    "gl"             # gl'inviti (italiano più antiquato, ma per sicurezza)
}

def tokenize(text: str):
   
    text = str(text)

    # normalizza apostrofi “strani” in '
    text = text.replace("’", "'").replace("‘", "'")

    pattern = r"""
        \.\.\.+                   |  # tre o più puntini: "..." -> "..."
        \.\.                      |  # due puntini
        \.                        |  # singolo punto
        [a-zA-Zàèéìòù]+(?:'[a-zA-Zàèéìòù]+)? |  # parole con eventuale parte dopo apostrofo
        [0-9]+                    |  # numeri
        [!?;:,]                      # altra punteggiatura singola (no apostrofo)
    """

    raw_tokens = re.findall(pattern, text, flags=re.VERBOSE)

    tokens = []
    for tok in raw_tokens:
        if "'" in tok:
            left, right = tok.split("'", 1)
            # controlla se la parte prima dell'apostrofo è un articolo/preposizione da separare
            if left.lower() in APOSTROFE_PREFIX and right:
                # es: "dell'abitazione" -> ["dell", "'", "abitazione"]
                tokens.append(left)
                tokens.append("'")
                tokens.append(right)
            else:
                # tieni il token intero (es: "c'è", "perché", ecc.)
                tokens.append(tok)
        else:
            tokens.append(tok)

    return tokens

# "oggi ho perso il bus09, che bella idea è stata perdere l'autobus!"
# ['oggi', 'ho', 'perso', 'il', 'bus', '09', ',', 'che', 'bella', 'idea', 'è', 'stata', 'perdere', 'l', "'", 'autobus', '!']



#-----------------------------------------------

# + estraggo il vocabolario - tokens

# Token2id (devo recuperare il vocabolario e togliere le frequenze)

import pickle
from pathlib import Path

def carica_token2id(path_vocab):
    with open(path_vocab, "rb") as f:
        vocabolario = pickle.load(f)

    token2id = {tok: info["id"] for tok, info in vocabolario.items()}
    return token2id

#token2id = carica_token2id("../Datasets_puliti/vocabolario.pkl")