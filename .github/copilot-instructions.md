# STIXBert — Copilot Instructions

## Project Overview
- **What**: Self-supervised Heterogeneous Graph Transformer (HGT) pre-trained on STIX 2.1 threat intelligence graphs.
- **Goal**: Learn structural representations of CTI data (ATT&CK techniques, malware, indicators, relationships) for downstream tasks like campaign clustering, ATT&CK classification, and link prediction.
- **Primary runtime**: Google Colab (A100 GPU). Data, checkpoints, and results stored on Google Drive.

## Repository
- **Remote**: `https://github.com/sd1977/STIXBert.git`
- **Branch**: `master`
- **Credentials**: Local-only (`credential.helper=store` in repo config). Never commit tokens.

## Project Structure
```
stixbert/
├── .env                  # HF_TOKEN, GITHUB_TOKEN (gitignored)
├── .github/
│   └── copilot-instructions.md
├── configs/
│   └── config.yaml       # Full config for src/ package version
├── docs/
│   ├── DEEP_LEARNING_TAXII_USECASES.md
│   └── STIXBERT_REVIEW.md
├── notebooks/
│   └── stixbert_pov.ipynb  # Self-contained Colab notebook (all code inline)
├── src/                    # Importable Python package (mirrors notebook logic)
│   ├── data/               # mitre_attack.py, misp_feeds.py, taxii_feeds.py
│   ├── graph/              # builder.py, features.py
│   ├── model/              # hgt.py (HGTConv-based encoder)
│   ├── training/           # pretrain.py, finetune.py
│   ├── evaluation/         # evaluate.py
│   └── utils/              # config.py, paths.py
├── requirements.txt
└── README.md
```

## Two Versions
1. **`notebooks/stixbert_pov.ipynb`** — Fully self-contained Colab notebook. All code is inline in cells. No `from src.*` imports. Uses an inline `CFG` dict instead of config.yaml. This is the primary deliverable.
2. **`src/` package** — Importable Python modules mirroring the notebook logic. For local development and IDE support. Uses `configs/config.yaml`.

Keep both in sync when making changes.

## Data Sources
| Source | URL | Notes |
|--------|-----|-------|
| MITRE ATT&CK | `raw.githubusercontent.com/mitre-attack/attack-stix-data/master/` | enterprise, mobile, ics bundles |
| ThreatFox | `https://threatfox.abuse.ch/export/json/recent/` | Public export endpoint. The POST API (`/api/v1/`) requires auth — do NOT use it. |
| DigitalSide | GitHub API → `davidonzo/Threat-Intel/contents/stix2` | ~1000 individual hash-named STIX bundles. List via GitHub API, download a batch. No `domain.json`/`ip.json`/`url.json` files. |

## Key Libraries
- PyTorch Geometric (`torch_geometric`) — HGTConv for heterogeneous graphs
- `sentence-transformers` (`all-MiniLM-L6-v2`) — text feature encoding
- `stix2` — STIX object creation
- `huggingface_hub` — model publishing (uses `HF_TOKEN` from `.env` or Colab Secrets)

## Environment
- macOS local dev, Python 3.9.6 system (services may need 3.11+)
- Colab target: Python 3.10+, CUDA/A100
- `.env` file holds `HF_TOKEN` and `GITHUB_TOKEN` — **never commit this file**

## Conventions
- Keep notebook cells self-contained — no cross-cell `from src.*` imports
- Data goes to Google Drive paths defined in `PATHS` dict (notebook) or `src/utils/paths.py` (package)
- Use `logging` module, not `print()`, for status messages in `src/`
- Notebook uses `print()` for user-visible output
