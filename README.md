Sarcasm Detection Project

Questo repository contiene un progetto di deep learning per l'identificazione del sarcasmo in testo.

Scopo: costruire e addestrare una rete neurale che classifichi se una frase è sarcastica o meno. Il progetto è pensato anche per supportare persone nello spettro autistico che possono trovare difficile percepire segnali sociali come il sarcasmo.

Se vuoi che crei anche il repo remoto su GitHub, dimmi se vuoi che provi con la CLI `gh` o preferisci i comandi manuali.

## Installazione di PyTorch

PyTorch richiede wheel specifici per la CPU o per versioni diverse di CUDA. Per evitare installazioni errate su Windows, usa lo script PowerShell incluso che scarica il wheel corretto in base alla tua scelta.

Esempio di utilizzo raccomandato (in PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\install_pytorch.ps1
pip install -r requirements.txt
```

Lo script ti chiederà se installare la versione CPU oppure le wheel per CUDA (es. 11.8, 12.1) oppure un URL personalizzato.

Se preferisci installare manualmente, usa il selettore ufficiale:
https://pytorch.org/get-started/locally/
