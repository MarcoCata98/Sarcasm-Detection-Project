import pandas as pd
import os

# Percorso del file Excel
file_path = "/Users/marcocata/Desktop/Progetto deep learning/Sarcasm-Detection-Project/src/trainingset/training_ironita2018.xlsx"

# Controllo esistenza file
if not os.path.exists(file_path):
    print("❌ File non trovato, controlla il percorso.")
else:
    print(f"📂 Caricamento dataset da: {file_path}\n")
    
    # Legge il file Excel (la prima riga deve contenere i nomi delle colonne)
    df = pd.read_excel(file_path)
    
    # Mostra informazioni di base
    print("✅ Dataset caricato correttamente!\n")
    print("📊 Dimensioni:", df.shape)
    print("\n📄 Colonne trovate:", list(df.columns))
    
    # Mostra le prime 5 righe per dare un’idea del contenuto
    print("\n🔍 Prime righe del dataset:")
    print(df.head(10))
