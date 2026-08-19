# Bureau-Analyse-Terrestre-Transformers

Projet d'analyse des releves UFO geocodes. Les phases actuelles servent a reconstruire les premiers calculs statistiques et a formuler la future tache de prediction.

## Installation

Depuis Windows PowerShell :

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Execution

```powershell
python analyse.py
```

Le fichier de donnees est telecharge automatiquement au premier lancement depuis la source officielle. Il est conserve localement sous `releves_klaxo3.csv` et n'est pas versionne.

## Sorties

Le script genere :

- `outputs/phase0_top10_journees.csv`
- `outputs/phase0_volume_annuel.png`

Les resultats et l'interpretation des phases 0 et 1 sont documentes dans `RAPPORT.md`.
