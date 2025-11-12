import pandas as pd
import os

# Use the script directory to build a path that works no matter the current working directory
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "data", "training_ironita2018.xlsx")

# Controllo esistenza file
if not os.path.exists(file_path):
    print("❌ File non trovato. Percorso cercato:")
    print(f"   {file_path}")
    print("✔️  Soluzioni possibili:\n  - Esegui lo script dalla cartella `src` (cd src; python test.py)\n  - Oppure usa un percorso assoluto al file Excel\n  - Oppure sposta il file in `src/data/`")
else:
    print(f"📂 Caricamento dataset da: {file_path}\n")
    # Legge il file Excel (la prima riga deve contenere i nomi delle colonne)
    df = pd.read_excel(file_path)
    # Mostra informazioni di base
    print("✅ Dataset caricato correttamente!\n")
    print("📊 Dimensioni:", df.shape)
    print("\n📄 Colonne trovate:", list(df.columns))
    # Mostra le prime righe per dare un’idea del contenuto
    print("\n🔍 Prime righe del dataset:")
    print(df.head(10))
