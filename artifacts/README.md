# Artifacts directory

Place the 6 files exported from your notebook here. **Exact filenames matter** —
the backend loads them by name.

```
artifacts/
├── nbis_embedding_model.keras        # from Section 16 (save_artifacts)
├── nbis_faiss.index
├── nbis_index_mapping.pkl
├── nbis_threshold.json
├── nbis_preprocessing_config.json
└── nbis.db                           # from Section 10 (Egyptian Database)
```

## Where these come from

| File | Notebook cell |
|------|---------------|
| `nbis_embedding_model.keras` | Section 16 — `save_artifacts(...)` |
| `nbis_faiss.index`           | Section 16 — `save_artifacts(...)` |
| `nbis_index_mapping.pkl`     | Section 16 — `save_artifacts(...)` |
| `nbis_threshold.json`        | Section 16 — `save_artifacts(...)` |
| `nbis_preprocessing_config.json` | Section 16 — `save_artifacts(...)` |
| `nbis.db` | Section 10 — Egyptian DB cell (saves to `/content/nbis_database/nbis.db`) |

If you already downloaded them from Colab, copy them all into this folder.

## Override the DB path (optional)

By default the backend looks for `./artifacts/nbis.db`. Override with:

```bash
export NBIS_DB_PATH=/path/to/your/nbis.db
```

If the DB file is missing, the API will still run — `/identify` just returns
the biometric match without the joined child/parent/hospital record.

## Quick sanity check

From the project root:

```bash
python -c "
from pathlib import Path
files = ['nbis_embedding_model.keras', 'nbis_faiss.index',
         'nbis_index_mapping.pkl', 'nbis_threshold.json',
         'nbis_preprocessing_config.json', 'nbis.db']
for f in files:
    p = Path('artifacts') / f
    kb = p.stat().st_size // 1024 if p.exists() else 0
    mark = '✅' if p.exists() else '❌'
    print(f'{mark}  {f}  ({kb} KB)')
"
```
