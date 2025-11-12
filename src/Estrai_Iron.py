import os
import pandas as pd
import numpy as np


def extract_xlsx_to_df(file_path):
    """
    Legge un file .xlsx e lo ritorna come pandas.DataFrame.

    Comportamento:
    - accetta percorsi assoluti o relativi (se relativi, sono risolti rispetto alla cartella del file `Estrai_Iron.py`)
    - stampa 'training...' e poi il DataFrame
    - in caso di file non trovato o errore di lettura, stampa un messaggio e ritorna None

    Args:
        file_path (str): percorso al file .xlsx

    Returns:
        pandas.DataFrame | None
    """
    # Risolvo percorsi relativi rispetto alla cartella dello script
    if not os.path.isabs(file_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, file_path)

    if not os.path.exists(file_path):
        print("❌ File non trovato:", file_path)
        return None

    try:
        # Legge il primo foglio del file Excel
        df = pd.read_excel(file_path)
        print("training...")
        # Stampa la tabella (DataFrame)
        print(df)
        return df
    except Exception as e:
        print("Errore leggendo il file:", str(e))
        return None


if __name__ == "__main__":
    # Demo: tenta di caricare data/training_ironita2018.xlsx dalla cartella src
    script_dir = os.path.dirname(os.path.abspath(__file__))
    demo_path = os.path.join(script_dir, "data", "training_ironita2018.xlsx")
    extract_xlsx_to_df(demo_path)
