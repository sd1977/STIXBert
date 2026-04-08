# STIXBert: Self-Supervised STIX Graph Foundation Model

Pre-trained Heterogeneous Graph Transformer (HGT) on STIX 2.1 threat intelligence graphs for campaign clustering, predictive infrastructure detection, ATT&CK classification, and cross-feed deduplication.

## Project Structure

```
stixbert/
├── configs/            # Model and training configuration
│   └── default.py
├── data/               # Raw and processed data (gitignored)
│   └── raw/
├── docs/               # Research docs and reviews
│   ├── STIXBERT_REVIEW.md
│   └── DEEP_LEARNING_TAXII_USECASES.md
├── notebooks/          # Colab notebooks
│   └── stixbert_pov.ipynb
├── src/
│   ├── data/           # Data retrieval from MITRE, MISP, TAXII
│   │   ├── mitre_attack.py
│   │   ├── misp_feeds.py
│   │   └── taxii_feeds.py
│   ├── graph/          # Graph construction and feature encoding
│   │   ├── builder.py
│   │   └── features.py
│   ├── model/          # HGT encoder + pre-training heads
│   │   └── hgt.py
│   ├── training/       # Pre-training and fine-tuning loops
│   │   ├── pretrain.py
│   │   └── finetune.py
│   ├── evaluation/     # Metrics, plots, demo utilities
│   │   └── evaluate.py
│   └── utils/
└── README.md
```

## Quick Start (Colab)

1. Open `notebooks/stixbert_pov.ipynb` in Google Colab
2. Select **GPU runtime** (T4 for dev, A100 for full training)
3. Run all cells

## Data Sources

| Source | Type | Auth Required |
|--------|------|---------------|
| MITRE ATT&CK | STIX 2.1 bundles | No |
| ThreatFox (abuse.ch) | IOCs with malware family links | No |
| DigitalSide Threat-Intel | STIX 2.0 bundles | No |
| MISP/CIRCL OSINT | MISP events (→ STIX) | No |
| CISA AIS | STIX indicators | Registration |

## Requirements

```
torch>=2.0
torch-geometric>=2.4
stix2>=3.0
taxii2-client>=2.3
sentence-transformers>=2.2
scikit-learn>=1.3
umap-learn>=0.5
matplotlib>=3.7
```

## References

- Hu et al. (2020). *Heterogeneous Graph Transformer.* WWW 2020.
- Hou et al. (2022). *GraphMAE: Self-Supervised Masked Graph Autoencoders.* KDD 2022.
- OASIS (2021). *STIX Version 2.1.* OASIS Standard.
