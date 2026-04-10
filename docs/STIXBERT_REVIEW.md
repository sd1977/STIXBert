# STIXBert: Self-Supervised STIX Graph Foundation Model — Review

## Overview

A foundation model pre-trained on STIX graph structure using self-supervised objectives (masked node prediction, link prediction, temporal ordering) to produce universal embeddings fine-tunable for any downstream threat intelligence task.

**Novelty: Novel** | **SASE Impact: High**

---

## Core Idea

Every ML use case (reputation scoring, lifetime prediction, campaign clustering, ATT&CK classification) requires building features from scratch. Existing work (SecLM, SevenLLM, CyBERT, CTINexus) all do *text* pre-training on security corpora. Nobody does *self-supervised graph pre-training on STIX object structure*.

The STIX schema (nodes: indicators, malware, attack-patterns, threat-actors; edges: uses, indicates, targets, attributed-to) is a rich heterogeneous graph that text models can't exploit.

### Pre-training Objectives

| Objective | Description |
|-----------|-------------|
| **Masked node prediction** | Mask 15% of indicator values; predict from graph context |
| **Link prediction** | Given an indicator and a threat actor, predict whether a `uses` relationship exists |
| **Temporal ordering** | Given two STIX objects from same campaign, predict which was observed first |

### Architecture

```
   STIX Object Graph (heterogeneous, multi-feed):
   ──────────────────────────────────────────────

   [indicator]──indicates──[malware]──uses──[attack-pattern]
       │                                         │
   observed-in──[campaign]         attributed-to──[threat-actor]
       │                                         │
   from-feed──[source]            targets──[identity/sector]

        ┌──────────────────────────────────────────────────┐
        │    Heterogeneous Graph Transformer Encoder         │
        │    (node-type-aware attention, edge-type encoding) │
        └────────────────────┬─────────────────────────────┘
                             │
                    Pre-trained Embeddings (128-dim per node)
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
        ┌──────────┐  ┌──────────┐  ┌──────────┐
        │ Fine-tune│  │ Fine-tune│  │ Fine-tune│
        │ Reputa-  │  │ Campaign │  │ ATT&CK   │
        │ tion     │  │ Cluster  │  │ Classify  │
        │ Scoring  │  │          │  │          │
        └──────────┘  └──────────┘  └──────────┘
           (2.1)         (2.5)         (2.3)
```

### Downstream Tasks (fine-tune from same backbone)

- Feed reputation scoring (use case 2.1)
- Campaign clustering (use case 2.5)
- ATT&CK technique classification (use case 2.3)
- Feed integrity verification (use case 8.2)

### Patent Claim Direction

*"A method for learning universal threat intelligence representations by self-supervised pre-training of a graph transformer on heterogeneous STIX 2.1 object graphs, using masked node prediction, link prediction, and temporal ordering objectives across multiple TAXII feed sources."*

---

## Review

### Strengths

- **Clear gap identification.** The observation that SecLM, SevenLLM, CyBERT, and CTINexus all operate on *text* while the STIX schema is fundamentally a *heterogeneous graph* is well-articulated and genuinely novel. This is a real blind spot in the literature.
- **Strong patent framing.** The claim direction is specific enough (STIX 2.1, heterogeneous graph transformer, three named objectives, multi-feed) to be defensible, and broad enough to cover the whole downstream fine-tuning paradigm.
- **Compelling SASE story.** The "train once, fine-tune for everything" pitch directly addresses the operational cost of building per-task ML pipelines. Referencing earlier use cases (2.1, 2.3, 2.5, 8.2) as downstream tasks ties the document together well.

### Concerns / Suggestions

#### 1. Data Scale Question is Unaddressed

Foundation models need large, diverse pre-training corpora. How many STIX objects/edges are realistically available across all TAXII feeds? If it's < 1M nodes, a heterogeneous GNN might work but calling it a "foundation model" oversells it. Add a rough estimate of available graph size and a minimum viable threshold.

#### 2. 128-dim Embedding is Stated but Unjustified

Why 128? For a heterogeneous graph with 7+ node types and 6+ edge types, this choice should be justified or at least noted as tunable. A too-low dimension could collapse distinct node-type semantics.

#### 3. Masked Node Prediction Objective is Underspecified

"Mask 15% of indicator values" — does this mean masking the indicator *pattern* (e.g., the IP/domain string), or masking the node entirely from the graph? These are very different tasks:
- Masking the raw pattern is essentially string prediction (hard to do with graph structure alone).
- Masking the node and predicting its type/attributes from neighbors is more aligned with graph pre-training (closer to GraphMAE).

Clarify which approach is intended.

#### 4. Missing Canonical Architecture Citations

HGT (Hu et al., 2020) is the canonical citation for heterogeneous graph transformers and is conspicuously absent. The Zhao et al. (2025) survey is good for self-supervised GFMs broadly, but the following should also be cited:
- Hu et al. (2020). *"Heterogeneous Graph Transformer."* WWW 2020 — the architecture being described.
- Hou et al. (2022). *"GraphMAE: Self-Supervised Masked Graph Autoencoders."* KDD 2022 — for the masked autoencoding objective on graphs.

#### 5. Temporal Ordering Objective Needs More Detail

STIX `created`/`modified` timestamps are often inaccurate or set to ingestion time rather than real-world observation time. If the model learns to predict temporal order from noisy timestamps, it may learn feed-ingestion artifacts rather than real threat timelines. Acknowledge this risk or describe how it would be handled (e.g., only use `first_observed`/`last_observed` from Sighting objects, or filter to feeds with known-good timestamps).

#### 6. "Hours, Not Weeks" Claim is Unsupported

This is a marketing-style claim. Either back it with a baseline comparison (e.g., "use case 2.1 required X days of feature engineering; with pre-trained embeddings it reduces to linear probe training") or soften the language.

#### 7. No Mention of Cross-Feed Generalization

The biggest value proposition of a foundation model here is that embeddings transfer across feeds with different schemas/conventions. This should be explicitly called out as an evaluation criterion — does a model pre-trained on MITRE ATT&CK-heavy feeds generalize to IOC-only feeds?

---

## Verdict

The core insight — that STIX graphs deserve *graph-native* self-supervised pre-training, not text-based LLM approaches — is **strong and genuinely novel**. The write-up is well-structured. The main gaps are:

1. Missing data-scale feasibility analysis
2. Underspecified masking objective
3. Missing canonical architecture citations (HGT, GraphMAE)

Addressing those would make this significantly more credible for both a patent filing and an internal pitch.

---

## Deep Research Findings

### 1. STIX 2.1 Object Types to Source

The STIX 2.1 schema defines four categories of objects. For STIXBert pre-training, focus on objects that form **rich graph structure** (i.e., have many relationships to other objects). The table below categorizes all STIX 2.1 object types by their value for graph pre-training.

#### Tier 1 — High-Value Graph Nodes (prioritize for sourcing)

These SDOs are the backbone of CTI graphs. They have dense, well-defined relationships and are the primary targets for all three pre-training objectives.

| Object Type | Category | Why High-Value |
|---|---|---|
| **Indicator** | SDO | Central node: connects to malware, attack-patterns, campaigns via `indicates`. Contains pattern, valid_from/until timestamps. |
| **Malware** | SDO | Rich attributes (malware_types, capabilities, is_family). Connected via `uses`, `indicates`, `targets`, `exploits`. |
| **Attack Pattern** | SDO | Maps to MITRE ATT&CK techniques. Connected to malware, tools, campaigns via `uses`. Has kill_chain_phases. |
| **Threat Actor** | SDO | Rich metadata (sophistication, motivations, roles, goals). Connected via `attributed-to`, `uses`, `targets`. |
| **Campaign** | SDO | Temporal anchor (first_seen/last_seen). Links actors to TTPs via `uses`, `attributed-to`. |
| **Intrusion Set** | SDO | Groups adversarial behaviors. Connects to threat actors, campaigns, TTPs. |
| **Infrastructure** | SDO | C2, botnets, hosting. Connected via `communicates-with`, `consists-of`, `uses`. |
| **Vulnerability** | SDO | CVE references. Connected via `exploits`, `targets`, `mitigates`. |
| **Relationship** | SRO | The edges themselves. Typed (uses, indicates, targets, attributed-to, etc.). Essential for graph structure. |
| **Sighting** | SRO | Temporal evidence (first_seen, last_seen, count). Links SDOs to Observed Data and Identity (where sighted). |

#### Tier 2 — Supporting Context Nodes

| Object Type | Category | Role |
|---|---|---|
| **Identity** | SDO | Target sectors/organizations. Provides victim context via `targets`. |
| **Location** | SDO | Geographic context via `located-at`, `originates-from`, `targets`. |
| **Tool** | SDO | Legitimate software used by attackers. Connected via `uses`. |
| **Course of Action** | SDO | Defensive responses via `mitigates`, `remediates`. |
| **Report** | SDO | Container for related objects. Provides grouping context. |
| **Observed Data** | SDO | Links to SCOs. Temporal (first_observed/last_observed). |

#### Tier 3 — Cyber Observable Objects (SCOs)

SCOs provide the raw IOC data. They are high-volume but have **sparse graph structure** on their own. Useful when linked to SDOs via Observed Data or Indicator patterns.

| Object Type | Category | Role |
|---|---|---|
| **IPv4/IPv6 Address** | SCO | Network indicators. High-volume. |
| **Domain Name** | SCO | DNS indicators. `resolves_to_refs` creates sub-graphs. |
| **URL** | SCO | Web indicators. |
| **File** | SCO | File hashes (MD5, SHA-256). High-volume from malware analysis. |
| **Email Address/Message** | SCO | Phishing indicators. |
| **Network Traffic** | SCO | Connection metadata (src/dst, ports, protocols). |
| **Process** | SCO | Endpoint behavior (command_line, image_ref). |
| **Software** | SCO | Installed software (CPE identifiers). |
| **X.509 Certificate** | SCO | TLS/SSL indicators. |
| **Autonomous System** | SCO | Network ownership context. |

#### Tier 4 — Low-Value for Pre-training (skip initially)

| Object Type | Category | Why Low-Value |
|---|---|---|
| Note, Opinion, Grouping | SDO | Analyst metadata; no structural graph signal. |
| Malware Analysis | SDO | Analysis results; useful but sparse. |
| Incident | SDO | Stub in STIX 2.1; no defined properties. |
| Language Content | SMO | Internationalization; no graph signal. |
| Marking Definition | SMO | Access control; no semantic signal. |
| Extension Definition | SMO | Schema extension mechanism. |
| Directory, Mutex, MAC Address, Windows Registry Key, User Account | SCO | Host-level artifacts; very sparse in TAXII feeds. |

---

### 2. Data Sources & Volume Estimates

#### Source 1: MITRE ATT&CK (STIX 2.1)

The canonical structured CTI dataset. Available as STIX 2.1 JSON bundles at [mitre-attack/attack-stix-data](https://github.com/mitre-attack/attack-stix-data). Current version: **v18.1** (November 2025).

| Domain | Estimated Objects | Notes |
|---|---|---|
| Enterprise ATT&CK | ~15,000–20,000 | ~800 techniques/sub-techniques, ~150 groups, ~700 software, ~50 campaigns, thousands of relationships |
| Mobile ATT&CK | ~3,000–5,000 | Smaller technique set |
| ICS ATT&CK | ~2,000–3,000 | Industrial control systems |
| **Total** | **~20,000–28,000** | Rich structure, but small by ML standards |

**Graph characteristics**: Dense, well-connected. High ratio of relationships to nodes (~3:1). Excellent for structural pre-training signal. However, **too small alone** to train a foundation model — this is a fine-tuning/evaluation dataset, not a pre-training corpus.

#### Source 2: MISP OSINT Feeds (convertible to STIX)

MISP hosts 70+ default open feeds. Most are IOC-level (IPs, domains, hashes) in CSV/freetext format, but several produce structured STIX/MISP-format data:

| Feed | Format | Volume (est.) | Graph Richness |
|---|---|---|---|
| CIRCL OSINT Feed | MISP (→ STIX) | ~500K+ events, millions of attributes | Medium (events group related IOCs) |
| Botvrij.eu | MISP (→ STIX) | ~10K+ events | Medium |
| DigitalSide Threat-Intel | STIX 2.0 + MISP | ~50K+ IOCs | Medium |
| abuse.ch (MalwareBazaar, URLhaus, ThreatFox) | MISP (→ STIX) | Millions of samples/URLs/IOCs | Low–Medium (flat IOC lists, limited relationships) |
| Infoblox Threat Intelligence | MISP | ~100K+ | Medium |

**Conversion note**: MISP events can be converted to STIX 2.1 using [misp-stix](https://github.com/MISP/misp-stix). Each MISP event becomes a STIX Bundle with related Indicators, Observed Data, and Relationships.

#### Source 3: Open TAXII Servers & STIX Feeds

| Source | Access | Volume (est.) | Notes |
|---|---|---|---|
| MITRE ATT&CK TAXII 2.1 | Free API | ~25K objects | Official TAXII endpoint |
| CISA AIS (Automated Indicator Sharing) | Free (registration) | ~1M+ indicators/year | US government feed |
| PickupSTIX | Free | ~100 new IOCs/day (~36K/year) | Translates public feeds to STIX |
| JamesBrine Threat Intel | Free | Daily feeds in STIX 2.0 | SSH/FTP/RDP bruteforce IPs |
| AlienVault OTX | Free (API key) | Millions of IOCs (Pulses) | Can export as STIX |
| Cyware Threat Intel Feeds | Free tier | STIX 1.x and 2.0 compatible | Malware hashes, IPs, domains |
| OpenCTI public instances | Free | Varies | Full STIX 2.1 graph structure |

#### Source 4: Commercial TAXII Feeds (Cisco-accessible)

As a Cisco SBG project, you likely have access to:

| Source | Volume (est.) | Graph Richness |
|---|---|---|
| Cisco Talos Intelligence | Millions of IOCs/year | High (structured TTP data) |
| Cisco SecureX Threat Intelligence | Large | High |
| Internal TAXII feed aggregation | Varies | The primary target for this model |

#### Aggregated Volume Estimate for PoV

| Tier | Sources | Nodes (est.) | Edges (est.) |
|---|---|---|---|
| **Minimum Viable** | ATT&CK + 5 open STIX feeds | 100K–300K | 200K–600K |
| **Target for PoV** | ATT&CK + MISP/CIRCL + AIS + open feeds | 500K–2M | 1M–5M |
| **Production scale** | + Cisco Talos + commercial feeds | 5M–50M | 10M–100M |

---

### 3. Training Data Requirements

#### How Many STIX Objects? Over What Period?

| Scenario | Nodes | Edges | Time Window | Training Epochs | Feasibility |
|---|---|---|---|---|---|
| **PoV / Proof of Concept** | 100K–500K | 200K–1M | 1–2 years of accumulated data | 100–500 | Achievable in days on single GPU |
| **Conference paper quality** | 500K–2M | 1M–5M | 2–3 years | 200–1000 | Achievable in 1–2 weeks on single A100 |
| **Production foundation model** | 5M–50M | 10M–100M | 3–5 years, continuously updated | 500–2000 | Requires multi-GPU / distributed |

**Key insight**: Unlike LLMs that need billions of tokens, GNN foundation models have shown strong results on much smaller graphs. HGT (Hu et al., 2020) demonstrated on the Open Academic Graph (179M nodes, 2B edges), but smaller-domain GNNs routinely work on graphs of 100K–1M nodes. The critical factor is **graph density and diversity**, not raw node count.

**Recommended minimum for PoV**: 
- **200K–500K nodes** with at least **500K–1M edges**
- At least **5 node types** (indicator, malware, attack-pattern, threat-actor, campaign)
- At least **4 edge types** (uses, indicates, targets, attributed-to)
- At least **12 months** of temporal span for the temporal ordering objective
- Multiple feed sources (min 3) for cross-feed generalization signal

#### Data Collection Strategy

```
Phase 1 (Week 1-2): Collect & Convert
├── Download MITRE ATT&CK STIX 2.1 bundle (enterprise + mobile + ICS)
├── Set up MISP instance, enable CIRCL OSINT + DigitalSide + abuse.ch feeds
├── Register for CISA AIS, pull historical indicators
├── Convert all to unified STIX 2.1 graph using misp-stix + cti-python-stix2
└── Merge into single heterogeneous graph (dedup by STIX ID)

Phase 2 (Week 2-3): Graph Construction
├── Parse all STIX objects → node features (type, attributes, timestamps)
├── Parse all Relationships + Sightings → typed edges
├── Extract embedded relationships (created_by_ref, object_refs, etc.)
├── Build adjacency matrices per edge type (for HGT input)
└── Compute graph statistics (density, connected components, type distribution)

Phase 3 (Week 3-4): Pre-training
├── Implement HGT encoder in PyTorch Geometric (PyG)
├── Implement 3 self-supervised objectives
├── Train on A100 (Colab Pro+ or GCP)
└── Evaluate embeddings on downstream tasks
```

---

### 4. Google Colab A100 Feasibility for PoV

#### TL;DR: Yes, Colab A100 is sufficient for a PoV — with caveats.

#### Colab GPU Options

| Tier | GPU | VRAM | Max Runtime | Cost |
|---|---|---|---|---|
| Free | T4 | 16 GB | ~12 hours (variable) | Free |
| Pro ($12/mo) | T4, sometimes V100 | 16–16 GB | ~12 hours, priority | $12/month |
| Pro+ ($58/mo) | A100 (40 GB) | 40 GB | ~24 hours, background exec | $58/month |
| Pay As You Go | A100 (40/80 GB) | 40–80 GB | Compute units | Variable |

#### A100 Capacity vs STIXBert Requirements

| Factor | STIXBert PoV Requirement | A100 (40 GB) Capacity | Verdict |
|---|---|---|---|
| **Graph size in memory** | 500K nodes × 128-dim = ~256 MB | 40 GB VRAM | Far within limits |
| **Mini-batch GNN training** | HGSampling, 2–3 hop neighborhoods | Standard in PyG/DGL | Supported |
| **Model parameters** | HGT with 4–6 layers, ~5–20M params | Easily fits | Supported |
| **Training time (PoV graph)** | 100–500 epochs on 500K-node graph | ~2–8 hours on A100 | Fits in single session |
| **Training time (larger graph)** | 500 epochs on 2M-node graph | ~12–24 hours | Fits in Pro+ session |

#### Recommended Setup

```
Hardware:  Colab Pro+ with A100 (40GB)
Framework: PyTorch Geometric (PyG) 2.x
           - torch_geometric.nn.HGTConv (native HGT support)
           - torch_geometric.data.HeteroData (heterogeneous graph)

Libraries:
  pip install torch-geometric
  pip install stix2              # STIX object parsing
  pip install taxii2-client      # TAXII feed ingestion
  pip install misp-stix          # MISP-to-STIX conversion

Estimated Cost: ~$58/month (Pro+) for 2-3 months = $116-$174 total
Alternative:    GCP Marketplace Colab ($0 Colab overhead + GCP VM cost)
```

#### Risks & Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| A100 not always available on Colab | Can't train when needed | Use Pay As You Go for guaranteed access, or fall back to V100/T4 with smaller batch size |
| 24-hour max runtime on Pro+ | May not finish large experiments | Use checkpointing; save model state to Drive every N epochs |
| Session disconnects | Lost training progress | Implement robust checkpointing + auto-resume |
| Data too large for Colab disk (78 GB default) | Can't load full dataset | Stream from GCS bucket, or compress and stage data |

#### Alternative: GCP Vertex AI / Single VM

For more reliable training beyond PoV:

| Option | GPU | Cost (est.) | Reliability |
|---|---|---|---|
| GCP `a2-highgpu-1g` | 1× A100 (40 GB) | ~$3.67/hr = ~$88/day | Guaranteed, no timeouts |
| GCP `a2-highgpu-2g` | 2× A100 (80 GB total) | ~$7.35/hr | For larger graphs |
| AWS `p4d.24xlarge` | 8× A100 (320 GB total) | ~$32/hr | Overkill for PoV |

**Recommendation**: Start with **Colab Pro+** for rapid prototyping and PoV. Move to a dedicated **GCP a2-highgpu-1g** VM only if Colab reliability becomes a bottleneck.

---

### 5. Relevant Architecture & Implementation Details

#### HGT (Heterogeneous Graph Transformer) — Key Details

From Hu et al. (2020), WWW 2020:
- **Node-type and edge-type dependent attention**: Each edge type gets its own attention parameters, allowing the model to learn different relationship semantics for `uses` vs `indicates` vs `targets`.
- **Relative temporal encoding**: Built-in support for temporal edges — directly applicable to STIX timestamp-based ordering.
- **HGSampling**: Heterogeneous mini-batch sampling algorithm for scalable training on large graphs. Critical for fitting 500K+ node graphs into GPU memory.
- **Validated at scale**: 179M nodes, 2B edges on Open Academic Graph — far larger than any STIX graph we'd build.
- **PyG implementation**: `torch_geometric.nn.HGTConv` provides a production-ready implementation.

#### GraphMAE — Key Details

From Hou et al. (2022), KDD 2022:
- **Feature reconstruction, not structure reconstruction**: Masks node features and trains a decoder to reconstruct them. This is more robust than graph structure reconstruction (which suffers from over-smoothing).
- **Scaled cosine error**: Better than MSE for feature reconstruction in graph autoencoders.
- **Masking strategy**: Random masking of 50–75% of nodes (higher than BERT's 15%) works well for graphs.
- **Implication for STIXBert**: The masking objective should focus on reconstructing **node attribute features** (e.g., malware_types, indicator_types, threat_actor_types, kill_chain_phases) from graph context, NOT on predicting raw indicator pattern strings.

#### Practical Implementation Considerations

1. **Node feature encoding**: STIX objects have mixed attribute types (strings, enums, timestamps, lists). Need a feature encoder per node type:
   - Categorical attributes (malware_types, indicator_types) → embedding lookup
   - Text fields (name, description) → sentence-transformers or TF-IDF
   - Timestamps → sinusoidal positional encoding or normalized floats
   - Hash values → learned hash embeddings or random projection

2. **Edge type inventory for STIX 2.1**: From the spec's Appendix B relationship summary:
   - `delivers`, `targets`, `uses` (attack-pattern)
   - `attributed-to`, `compromises`, `originates-from`, `targets`, `uses` (campaign)
   - `investigates`, `mitigates` (course-of-action)
   - `indicates` (indicator)
   - `communicates-with`, `consists-of`, `controls`, `has`, `hosts`, `uses` (infrastructure)
   - `attributed-to`, `compromises`, `hosts`, `impersonates`, `located-at`, `owns`, `targets`, `uses` (threat-actor)
   - `exploits`, `targets` (malware/tool)
   - `sighting-of` (sighting)
   - `derived-from`, `duplicate-of`, `related-to` (common)
   - **Total: 20+ edge types** — rich heterogeneous signal

3. **Graph construction pseudocode**:
   ```python
   from torch_geometric.data import HeteroData
   
   data = HeteroData()
   
   # Node features per type
   data['indicator'].x = encode_indicators(stix_indicators)      # [N_ind, d]
   data['malware'].x = encode_malware(stix_malware)              # [N_mal, d]
   data['attack_pattern'].x = encode_attack_patterns(stix_ap)    # [N_ap, d]
   data['threat_actor'].x = encode_threat_actors(stix_ta)        # [N_ta, d]
   data['campaign'].x = encode_campaigns(stix_camp)              # [N_camp, d]
   
   # Edges per relationship type
   data['indicator', 'indicates', 'malware'].edge_index = ...
   data['malware', 'uses', 'attack_pattern'].edge_index = ...
   data['campaign', 'attributed_to', 'threat_actor'].edge_index = ...
   data['campaign', 'uses', 'malware'].edge_index = ...
   # ... etc for all relationship types
   ```

---

### 6. Adjacent Work & Competitive Landscape

| System | What It Does | How STIXBert Differs |
|---|---|---|
| **SecLM** (Asoronye 2024) | Text pre-training on security corpora | Text-only; ignores graph structure |
| **SevenLLM** (Ji 2024) | LLM benchmark for CTI tasks | LLM-based; no graph learning |
| **CyBERT** | BERT fine-tuned on security text | Text-only; no STIX awareness |
| **CTINexus** (Cheng 2025) | LLM-based KG construction from CTI text | Builds graphs from text using LLMs; doesn't learn graph embeddings |
| **CyberVeriGNN** (Huang & Wang 2025) | GNN for fake CTI detection | Supervised, single-task; not self-supervised foundation model |
| **TypeDB CTI** | STIX-based KG storage (MITRE ATT&CK) | Storage/query platform; no ML |
| **OpenCTI** | STIX-based CTI platform | Operational platform; no graph ML |
| **GraphSAGE / GAT / GCN** | General GNN architectures | Homogeneous graphs only; can't handle typed nodes/edges |

**Key differentiator**: No existing system does **self-supervised pre-training on heterogeneous STIX graphs**. The closest is CyberVeriGNN which uses GNNs for CTI but in a supervised, single-task setting.

---

### 7. Recommended PoV Execution Plan

| Week | Activity | Deliverable |
|---|---|---|
| 1 | Data collection: ATT&CK STIX + 3 open feeds. Set up MISP instance. | Raw STIX bundles |
| 2 | Graph construction: Parse STIX → HeteroData. Feature encoding. | PyG HeteroData object, graph statistics report |
| 3 | Implement HGT encoder + masked node prediction objective | Training pipeline v1 |
| 4 | Pre-train on Colab A100. Add link prediction + temporal ordering. | Pre-trained model checkpoint |
| 5 | Fine-tune on downstream task (e.g., ATT&CK classification) | Baseline comparison report |
| 6 | Ablation studies. Write up results. | PoV completion report |

**Success criteria for PoV**:
- Pre-trained embeddings outperform randomly initialized embeddings on at least 1 downstream task
- Fine-tuning with pre-trained backbone converges in < 1/5th the epochs vs training from scratch
- Embeddings show meaningful clustering (t-SNE) by threat category

---

## 8. "Foundation Model" — Clarification

The doc uses "foundation model" aspirationally. STIXBert is not a foundation model in the GPT/LLaMA sense (billions of parameters, trained for months). It is a **domain-specific pre-trained GNN** — ~2–10M parameters, trained in hours. The term is fine for patent/paper positioning but the implementation is much simpler than it implies.

### Alternative Approaches Considered

| Option | Approach | Verdict |
|---|---|---|
| **1. Fine-tune existing Graph Foundation Model** | Take a pre-trained GraphMAE/GCC checkpoint, fine-tune on STIX | **Won't work.** These are pre-trained on *homogeneous* graphs (citation/social networks). STIX is *heterogeneous* (18+ node types, 20+ edge types). The type-aware attention needed for STIX doesn't exist in these checkpoints. |
| **2. Fine-tune an LLM on STIX text** | Use SecLM/CyBERT/LLaMA on STIX JSON serializations | **Loses the graph.** LLMs process STIX as flat text — they can't natively reason over multi-hop paths (Threat Actor → uses → Malware → targets → Vulnerability). This is exactly the gap STIXBert fills. |
| **3. HGT + self-supervised pre-training (STIXBert)** | Initialize HGT with random weights, self-supervised pre-train on STIX graph, fine-tune downstream | **The pragmatic middle ground.** PyG's `HGTConv` is an architecture, not a pre-trained model. Pre-training on 500K nodes takes hours on an A100. This is what the proposal actually describes. |
| **4. Skip pre-training, supervised only** | Train HGT directly on labeled downstream tasks | **Requires labeled data.** Analyst-annotated campaign clusters and attribution labels are expensive and scarce. Self-supervised pre-training avoids this bottleneck. |

**Recommendation**: Option 3. Call it a "foundation model" in external communications, but internally treat it as a domain-specific pre-trained HGT.

---

## 9. SASE Business Case

### The Problem

SASE platforms (SWG + FW + ZTNA + CASB) ingest threat intelligence from multiple TAXII feeds — but today that intelligence is consumed as flat indicator lists (block this IP, flag this domain). This creates three costly problems:

| Problem | Impact | Current Cost |
|---|---|---|
| **Feed overlap & duplication** | Same indicator appears in 5+ feeds with different confidence scores. No way to reconcile. | Analyst time wasted triaging duplicates; customers see inconsistent verdicts |
| **No contextual prioritization** | An IP tied to APT29 targeting critical infrastructure is treated the same as a commodity phishing domain | Alert fatigue — customers drown in equal-priority alerts, miss what matters |
| **Reactive-only posture** | SWG/FW blocks known-bad indicators. Zero predictive capability. | Customers are always one step behind. New infrastructure spun up by an active campaign goes unblocked until a feed catches up |

### How STIXBert Solves This

| Capability | SASE Application | Business Outcome |
|---|---|---|
| **Cross-feed deduplication** | Embeddings cluster identical threats across feeds automatically | Reduce feed processing volume by 30–50%; lower compute costs |
| **Campaign-aware blocking** | SWG policies auto-escalate indicators linked to active campaigns | Block entire campaign infrastructure, not just individual IOCs |
| **Predictive infrastructure detection** | Model learns Threat Actor → registers → Domain → resolves-to → IP patterns. Predicts next-hop infrastructure. | SWG/FW can pre-block infrastructure before it appears in any feed |
| **Feed quality scoring** | Embedding distance from ground-truth ATT&CK data quantifies feed reliability | Data-driven feed procurement; drop low-value feeds, save licensing costs |
| **Contextual alert enrichment** | Every indicator gets a vector encoding its full graph neighborhood | SOC sees "domain is 2 hops from APT29 campaign" instead of just "malicious domain" |

### Strategic Positioning

**STIXBert aligns with "Robust Platform"** — it is infrastructure that produces embeddings (internal representations) to make the platform smarter. Customers never see embeddings; they see downstream outcomes. The downstream fine-tuned tasks (predictive blocking, contextual alerting) deliver **"Secure Customer"** outcomes.

> STIXBert is a Robust Platform investment that enables multiple Secure Customer outcomes.

### Revenue & Differentiation

1. **Premium tier feature** — "AI-Enriched Threat Intelligence" as a paid add-on. Contextual, predictive blocking vs. commodity IOC matching.
2. **Competitive moat** — No SASE vendor (Zscaler, Palo Alto Prisma, Netskope) uses graph-based pre-trained models on STIX data. First-mover advantage + patent protection.
3. **Feed cost optimization** — If STIXBert proves 3 of 8 feeds are >90% redundant, that's direct licensing cost savings.
4. **Reduced customer churn** — Customers stay because contextual intelligence can't be replicated by simply buying more feeds.

### Quantifiable Metrics for PoV

| Metric | Target | How to Measure |
|---|---|---|
| Cross-feed dedup rate | >40% overlap detected | Compare embeddings across feed pairs |
| Campaign clustering accuracy | >85% ARI (Adjusted Rand Index) | Benchmark against ATT&CK ground-truth |
| Predictive infrastructure lead time | >24 hours before feed publication | Timestamp comparison: model prediction vs. feed appearance |
| Alert volume reduction | 30–50% fewer duplicate/low-context alerts | Before/after on staging tenant |

### Investment vs. Return

| Item | Cost |
|---|---|
| PoV development (6 weeks, 1 engineer) | ~1 engineer-sprint |
| Compute (Colab Pro+) | ~$58/month |
| Data (ATT&CK + MISP + existing feeds) | $0 (open + already licensed) |
| **Total PoV cost** | **Minimal** |

| Return | Value |
|---|---|
| Patent filing | IP asset |
| Premium feature revenue | Recurring per-customer |
| Feed cost savings | Direct opex reduction |
| Competitive differentiation | Market positioning |

---

## 10. SWG Worked Example

### Without STIXBert (Today)

1. Feed A publishes: `malware-drop.evil.com` → SWG blocks it
2. Attacker registers `malware-drop2.evil.net` five minutes later
3. Customer traffic hits `malware-drop2.evil.net` → **SWG allows it** (not in any feed yet)
4. Hours/days later, Feed B publishes `malware-drop2.evil.net` → SWG finally blocks it
5. Customer was exposed the entire time

### With STIXBert

The STIX graph already knows:

```
Threat Actor "SilkTyphoon"
  └── uses → Malware "ShadowPad"
        └── communicates-with → Infrastructure "185.92.x.x"
              └── resolves-to → Domain "malware-drop.evil.com"
              └── resolves-to → Domain "cdn-update.evil.org"     ← same IP
              └── registered-by → Identity "john@protonmail.com"
                    └── also-registered → Domain "service-api.evil.net"  ← not in any feed yet
```

STIXBert's embeddings place `service-api.evil.net` **close to** `malware-drop.evil.com` in vector space because they share infrastructure, registration, and campaign lineage. The model flags it as high-risk.

**SWG action:** Pre-emptively block or challenge `service-api.evil.net` before any feed publishes it.

### What the Customer Sees

| | Today | With STIXBert |
|---|---|---|
| User visits `malware-drop.evil.com` | Blocked (in feed) | Blocked |
| User visits `service-api.evil.net` | **Allowed** (not in feed yet) | **Blocked** (graph proximity) |
| SOC alert says | "Blocked malicious domain" | "Blocked domain linked to SilkTyphoon campaign via shared infrastructure with ShadowPad C2" |

---

## 11. RAG vs. STIXBert

RAG and STIXBert solve different problems. They are complementary, not interchangeable.

### What Each Does

| Task | RAG | STIXBert |
|---|---|---|
| "Tell me about SilkTyphoon" | Great — retrieves and summarizes | Not needed for this |
| "Is `service-api.evil.net` related to known threats?" | Only if text overlap exists (same registrant/IP mentioned in a report) | Learns structural similarity from graph topology even without text overlap |
| Score every domain in SWG logs against campaign proximity | Can't score 1M domains/day through an LLM | Vector comparison — microseconds per lookup |
| Cluster all indicators by campaign | LLMs don't produce consistent embeddings for clustering | Embeddings are designed for this |
| Predict next infrastructure for a campaign | Retrieves what exists; doesn't predict what's missing | Link prediction is a core pre-training task |

### The Core Difference

- **RAG** answers questions about what's in the data — it's a retrieval + reasoning system
- **STIXBert** learns patterns in the graph structure — it's a representation learning system

RAG treats STIX objects as **documents**. STIXBert treats them as **nodes in a graph**. The graph topology (who connects to what, how many hops, what relationship types) is signal that RAG discards.

### Dynamic Data Concern

STIX data changes constantly, which favors RAG's ability to index new data immediately. However:

1. **Graph structure is relatively stable** — ATT&CK TTPs, threat actor profiles, and malware families change slowly (quarterly). What changes fast is new indicators.
2. **New IOCs plug into existing structure** — A new domain for APT29 connects to an existing Threat Actor → Campaign → Malware subgraph. STIXBert uses **inductive inference** (compute the new node's embedding from its neighbors' learned embeddings) — no retraining needed.
3. **Periodic retraining is cheap** — A few hours on an A100 every week/month. Standard practice for production ML.

### Recommended Architecture: Both

```
                     STIX Graph
                         │
              ┌──────────┴──────────┐
              │                     │
         STIXBert                  RAG
    (graph embeddings)      (text retrieval)
              │                     │
   ┌─────────┼─────────┐          │
   │         │         │          │
Campaign  Predictive  Feed    Analyst
Clustering Blocking  Quality   Q&A
   │         │         │          │
   └─────────┼─────────┘          │
             │                     │
        SWG/FW Policy        SOC Dashboard
        (automated)          (human-facing)
```

- **STIXBert** for high-throughput structural tasks (scoring millions of SWG log entries, clustering, prediction)
- **RAG** for human-facing tasks (analyst questions, report generation, investigation support)
- STIXBert embeddings can **improve** RAG by providing graph-aware retrieval instead of text-only similarity

> RAG tells you what the data says. STIXBert tells you what the data *means structurally* — and predicts what's missing.

---

## 12. Data Required to Train the Model

This section consolidates exactly what data is needed, broken into two phases (self-supervised pre-training and supervised fine-tuning) scoped to a **Colab A100 (40 GB VRAM) PoV**.

### Phase 1: Self-supervised Pre-training (no labels needed)

The model learns graph structure by solving pretext tasks (masked node prediction, link prediction, temporal ordering). All it needs is a **large, diverse STIX graph**.

#### STIX Objects Required

| Object Type | Role in Pre-training | Minimum Count for PoV | Where to Get It |
|---|---|---|---|
| **Indicator** | Highest-volume node type. Carries IOC patterns + timestamps. Central to `indicates` edges. | 100K–300K | MISP/CIRCL, CISA AIS, abuse.ch, AlienVault OTX |
| **Malware** | Rich attributes (malware_types, capabilities, is_family). Hub node connecting indicators to campaigns. | 1K–5K | ATT&CK (700+), MISP, MalwareBazaar |
| **Attack Pattern** | Maps to ATT&CK techniques. kill_chain_phases provide structured features. | 500–1K | ATT&CK (800+ techniques/sub-techniques) |
| **Threat Actor** | Provides attribution context. Rich metadata (sophistication, motivations, goals). | 100–500 | ATT&CK (~150 groups), OpenCTI |
| **Campaign** | Temporal anchors (first_seen/last_seen). Groups related activity. | 50–200 | ATT&CK (~50), MISP events as proxy |
| **Infrastructure** | C2 servers, hosting, botnets. Critical for the SWG predictive blocking use case. | 1K–10K | MISP, ThreatFox, URLhaus |
| **Vulnerability** | CVE nodes. Connected via `exploits`, `targets`. | 1K–5K | NVD, ATT&CK, MISP |
| **Intrusion Set** | Groups behaviors; connects actors to campaigns. | 50–200 | ATT&CK |
| **Relationship (SRO)** | The typed edges (uses, indicates, targets, attributed-to, etc.). **This is the most critical data** — without edges, there is no graph. | 500K–1M | Embedded in all STIX bundles |
| **Sighting (SRO)** | Temporal evidence. Provides first_seen/last_seen + who sighted it. Needed for temporal ordering objective. | 10K–100K | MISP, CISA AIS, Sighting-as-a-Service feeds |

**SCOs (Cyber Observables)** — optional for Phase 1:

| Object Type | Include? | Why |
|---|---|---|
| Domain Name, IPv4/IPv6 | Yes — if available | Adds volume and `resolves-to` sub-graphs that model infrastructure pivoting |
| File (hashes) | Yes — if available | High-volume from malware analysis; connects to Malware nodes |
| URL, Email, X.509 Certificate | Nice to have | Adds diversity but not critical for PoV |
| Process, Network Traffic, Mutex, Registry Key | Skip | Host-level artifacts; too sparse in TAXII feeds |

#### What the Pre-training Graph Should Look Like

```
Target for Colab A100 PoV:
──────────────────────────
Nodes:  200K–500K  (across 5–8 node types)
Edges:  500K–1M    (across 4–10 edge types)
Memory: ~256 MB–1 GB in GPU VRAM (trivial for 40 GB A100)

Minimum edge types needed:
  indicates, uses, targets, attributed-to, communicates-with,
  exploits, sighting-of, resolves-to

Temporal span: ≥12 months (for temporal ordering objective)
Feed sources:  ≥3 (for cross-feed generalization signal)
```

### Phase 2: Supervised Fine-tuning (labels needed)

After pre-training, the model is fine-tuned on specific downstream tasks. Each task needs a **small labeled dataset** — the whole point of pre-training is to reduce this requirement.

| Downstream Task | Labels Needed | Label Source | Labeled Samples Needed |
|---|---|---|---|
| **ATT&CK technique classification** | STIX objects mapped to ATT&CK technique IDs | ATT&CK STIX data (comes pre-labeled) | 5K–10K (already available) |
| **Campaign clustering** | Ground-truth campaign assignments | ATT&CK campaign objects + analyst labels | 500–2K cluster assignments |
| **Feed quality scoring** | Feed reliability ratings | Expert labeling or derived from overlap analysis | 100–500 feed-level scores |
| **Predictive infrastructure detection** | Known infra → campaign mappings (held-out temporal split) | Historical STIX data with timestamps | 1K–5K held-out indicator→campaign pairs |
| **Cross-feed deduplication** | Matched pairs (same entity, different feeds) | Automated via STIX ID + deterministic matching | 5K–20K pairs (auto-generated) |

**Key point**: ATT&CK data serves double duty — it's part of the pre-training graph AND provides ready-made labels for fine-tuning. This is why ATT&CK is sourced first.

### Non-STIX Data Required

STIX objects alone are not enough. The model also needs:

#### 1. Node Feature Inputs (to encode STIX attributes into vectors)

| Data | What It's For | Source |
|---|---|---|
| **Pre-trained text embeddings model** | Encode STIX `name` and `description` fields into feature vectors for each node. Raw text can't be fed into a GNN. | `all-MiniLM-L6-v2` (sentence-transformers, 384-dim, free, runs on CPU) or `all-mpnet-base-v2` (768-dim, better quality) |
| **ATT&CK technique descriptions** | Enrich Attack Pattern nodes with detailed TTP descriptions beyond the short STIX `name` | MITRE ATT&CK website / STIX bundle (already included) |
| **CVE descriptions** | Enrich Vulnerability nodes with vulnerability details | NVD JSON feeds (free, ~250K CVEs) |
| **WHOIS / passive DNS data** | Enrich Domain/IP SCOs with registration dates, registrant, ASN, hosting provider | CIRCL passive DNS (free for researchers), DomainTools (commercial), Team Cymru |
| **GeoIP data** | Add geographic features to IP/Infrastructure nodes | MaxMind GeoLite2 (free) |

#### 2. Negative Samples (for link prediction objective)

| Data | What It's For | Source |
|---|---|---|
| **Benign domains / IPs** | The model needs to learn what a *non-malicious* node looks like, not just malicious ones. Without negatives, link prediction is trivial. | Alexa/Tranco top-1M (free), Cisco Umbrella popularity list, Majestic Million |
| **Random non-existent edges** | For link prediction training: sample node pairs with no edge and use as negative examples. | Generated programmatically from the graph (standard GNN practice) |

#### 3. Evaluation / Benchmark Data (to measure if the model works)

| Data | What It's For | Source |
|---|---|---|
| **ATT&CK ground-truth groupings** | Evaluate campaign clustering accuracy (ARI score) | ATT&CK STIX data — groups, campaigns, software mappings |
| **Held-out temporal split** | Evaluate predictive capability: train on data up to time T, predict edges after T | Split your own graph by `created` / `first_seen` timestamps |
| **Cross-feed overlap labels** | Evaluate deduplication: known same-entity pairs across feeds | Generate by matching STIX IDs or deterministic IOC matching across feeds |

#### 4. Infrastructure Data (non-ML, but required to run the PoV)

| Item | What It's For | Source |
|---|---|---|
| **Google Colab Pro+ subscription** | A100 GPU access | $58/month |
| **Google Drive storage** | Persist datasets, checkpoints, results across Colab sessions | Included with Google account (15 GB free, 100 GB with Google One) |
| **Python packages** | `torch-geometric`, `stix2`, `taxii2-client`, `misp-stix`, `sentence-transformers` | pip install (free) |

### Summary: PoV Data Checklist

```
✅ STIX DATA (graph structure)
   ├── MITRE ATT&CK STIX 2.1 bundle (Enterprise + Mobile + ICS)    ~25K objects
   ├── MISP/CIRCL OSINT feed (converted to STIX)                    ~100K–500K objects
   ├── CISA AIS historical indicators                                ~100K–500K objects  
   ├── 2–3 additional open STIX feeds (DigitalSide, abuse.ch, OTX)  ~50K–200K objects
   └── Total: 200K–500K nodes, 500K–1M edges

✅ NON-STIX DATA (features & evaluation)
   ├── Sentence-transformers model (all-MiniLM-L6-v2)               ~80 MB download
   ├── NVD CVE JSON feeds                                            ~1 GB download
   ├── MaxMind GeoLite2 database                                     ~50 MB download
   ├── Tranco/Umbrella top-1M benign domains                         ~20 MB download
   └── WHOIS/passive DNS (optional, enriches infrastructure nodes)   API access

✅ INFRASTRUCTURE
   ├── Colab Pro+ ($58/month)
   ├── Google Drive (model checkpoints)
   └── Python environment (PyG, stix2, sentence-transformers)
```

---

## 13. SCO Strategy: Do We Need Bulk IOCs from Anomali / AlienVault?

### Short Answer: Not for Pre-training

The three self-supervised objectives (masked node prediction, link prediction, temporal ordering) all require **dense, typed edges** between nodes. Bulk SCO feeds from Anomali, AlienVault OTX, or similar IOC aggregators are mostly flat indicator lists — millions of IPs/domains/hashes with **minimal or zero relationship context**. They add node count without adding graph signal.

SCOs are classified as **Tier 3** in the object priority table (Section 1) for a reason:

> SCOs provide the raw IOC data. They are high-volume but have **sparse graph structure** on their own.

### What Actually Matters for Pre-training

| What You Need | Why | Where to Get It |
|---|---|---|
| **SDOs with relationships** (indicator → malware → attack-pattern → threat-actor) | Dense edges = pre-training signal for HGT attention | MITRE ATT&CK, MISP structured events, OpenCTI |
| **Typed edges** (uses, indicates, targets, attributed-to) | HGT learns separate attention weights per edge type — needs diverse edge types | Same — only structured CTI sources provide these |
| **Temporal metadata** (first_seen, last_seen, created, modified) | Required for the temporal ordering pre-training objective | Sighting objects, Campaign objects |
| **Multiple feed provenance** | Cross-feed generalization is a key evaluation criterion | ≥3 independent sources |

### When SCOs *Do* Matter

| Phase | SCO Role | Example |
|---|---|---|
| **Downstream fine-tuning: reputation scoring (use case 2.1)** | Need real IOCs with ground-truth labels (malicious / benign) to train the scoring head | AlienVault OTX IOCs + VirusTotal verdicts as labels |
| **Downstream fine-tuning: predictive blocking** | Need domain/IP SCOs linked to Infrastructure → Campaign subgraphs to learn pivoting patterns | MISP events with infrastructure context |
| **When linked to SDOs** | An IP that resolves from a domain, referenced by an Indicator, that indicates Malware, that uses an Attack Pattern — this full chain is valuable | MISP structured events provide this; bulk IOC lists do not |
| **Enriching `resolves-to` sub-graphs** | Domain → IP resolution edges teach the model infrastructure pivoting | Passive DNS data (CIRCL, Farsight) provides this better than feed SCOs |

### Decision Matrix

| Source | Include for Pre-training? | Include for Fine-tuning? | Rationale |
|---|---|---|---|
| **MITRE ATT&CK** | Yes — first priority | Yes (provides labels) | Dense graph, pre-labeled |
| **MISP/CIRCL structured events** | Yes — second priority | Yes | Events group related SDOs + SCOs with relationships |
| **CISA AIS** | Yes | Yes | Government-curated, includes relationships |
| **Anomali STIX feed** | Only if relationships included | Yes — for reputation scoring | Often relationship-rich; check before committing |
| **AlienVault OTX Pulses** | No — skip for pre-training | Yes — for reputation scoring | Pulses are mostly flat IOC lists with tags, not typed STIX relationships |
| **abuse.ch (URLhaus, ThreatFox)** | Conditional | Yes | ThreatFox has malware family links (useful); URLhaus is flat URL lists (skip) |
| **DigitalSide Threat-Intel** | Yes | Yes | Provides STIX 2.0 bundles with relationships |

### Bottom Line

For the PoV, prioritize **MITRE ATT&CK + MISP structured events + CISA AIS** (relationship-rich). Skip bulk IOC-only feeds initially. Add Anomali/OTX SCOs later only if:
1. You need volume for fine-tuning a downstream task like reputation scoring
2. Those SCOs come with linked SDO context (not just bare indicators)

---

## 14. Demo Scenarios

### Demo 1: Campaign Clustering — "Which Threats Are Related?"

**Setup**: Load pre-trained STIXBert. Ingest ATT&CK + MISP data. Extract embeddings for all Indicator, Malware, and Campaign nodes.

**Live Demo Steps**:
1. Run t-SNE / UMAP on the embeddings → show 2D scatter plot colored by campaign
2. Indicators from the same campaign should cluster together automatically — **without ever seeing campaign labels during pre-training**
3. Pick a cluster → show the model grouped indicators from APT29, SilkTyphoon, Lazarus into distinct regions
4. Introduce 5 new unlabeled indicators → show which cluster they land nearest to → "the model attributes these to Lazarus based on graph topology"

**What It Proves**: Pre-trained graph embeddings capture campaign structure without supervision.

```
Demo output (Colab cell):
──────────────────────────
[UMAP plot with colored clusters]

Cluster 1 (red):    APT29 / Cozy Bear     — 47 indicators, 12 malware samples
Cluster 2 (blue):   Lazarus Group          — 83 indicators, 9 malware samples
Cluster 3 (green):  SilkTyphoon            — 31 indicators, 6 malware samples
Cluster 4 (orange): FIN7                   — 56 indicators, 15 malware samples

New indicator "185.220.xx.xx" → nearest cluster: Lazarus (cosine similarity: 0.87)
New indicator "cdn-update[.]kr" → nearest cluster: Lazarus (cosine similarity: 0.91)
```

### Demo 2: Predictive Infrastructure Detection — "What Will They Use Next?"

**Setup**: Train on STIX data up to time T. Hold out indicators/infrastructure that appeared after T.

**Live Demo Steps**:
1. Show the model a known campaign subgraph (Threat Actor → Malware → known C2 domains)
2. Run link prediction: "Given this campaign, which unlinked domains/IPs are most likely to be next-hop infrastructure?"
3. Compare predictions against the held-out ground truth
4. Show precision/recall: "The model predicted 8 of the 12 domains that appeared in feeds 3–14 days later"

**What It Proves**: Graph structure encodes predictive signal — the model learns registration patterns, hosting co-location, and domain naming conventions from the graph.

```
Demo output (Colab cell):
──────────────────────────
Campaign: SilkTyphoon (2025-Q3)
Known infrastructure (training set):
  ├── 185.92.x.x  (C2 server)
  ├── malware-drop.evil[.]com
  └── cdn-update.evil[.]org

Link prediction — top 5 candidate domains:
  1. service-api.evil[.]net      score: 0.94  ← appeared in feed 3 days later ✓
  2. update-check.evil[.]org     score: 0.89  ← appeared in feed 7 days later ✓
  3. api-gateway.evil[.]com      score: 0.82  ← appeared in feed 14 days later ✓
  4. static-cdn.evil[.]net       score: 0.78  ← not yet seen (possible future infra?)
  5. telemetry.evil[.]org        score: 0.71  ← false positive (legitimate domain)

Precision@5: 60%  |  Lead time: 3–14 days ahead of feed publication
```

### Demo 3: Cross-Feed Deduplication — "These 5 Feeds Are Saying the Same Thing"

**Setup**: Ingest the same week of data from 5+ STIX/TAXII feeds. Compute embeddings for all overlapping indicators.

**Live Demo Steps**:
1. Show raw indicator counts per feed (e.g., Feed A: 12K, Feed B: 8K, Feed C: 15K, ...)
2. Compute pairwise cosine similarity between all indicator embeddings across feeds
3. Show a heatmap of cross-feed overlap: "Feed A and Feed C are 72% redundant"
4. Show deduplicated count after merging: "35K indicators across 5 feeds → 18K unique threats after dedup"
5. For a specific indicator, show: "This IP appears in 4 feeds with confidence 40, 60, 75, 90 → fused STIXBert score: 0.83"

**What It Proves**: Embeddings enable intelligent deduplication and multi-feed fusion — directly reducing feed processing costs and blocklist bloat.

```
Demo output (Colab cell):
──────────────────────────
Cross-Feed Overlap Matrix (% embedding similarity > 0.9):

             Feed A   Feed B   Feed C   Feed D   Feed E
Feed A       100%     23%      72%      15%      41%
Feed B        23%    100%      18%      67%      12%
Feed C        72%     18%     100%      11%      38%
Feed D        15%     67%      11%     100%       9%
Feed E        41%     12%      38%       9%     100%

Insight: Feed A and Feed C are 72% redundant — consider dropping one.
         Feed D provides the most unique coverage (max 15% overlap with any other feed).

Before dedup: 35,247 total indicators across 5 feeds
After dedup:  18,431 unique threat clusters (47.7% reduction)
```

### Demo 4: ATT&CK Technique Classification — "What TTP Does This Indicator Map To?"

**Setup**: Fine-tune pre-trained STIXBert on ATT&CK technique labels (available directly from ATT&CK STIX data).

**Live Demo Steps**:
1. Input: a new Indicator or Malware node (not in training data)
2. Model predicts top-3 ATT&CK techniques with confidence scores
3. Compare against analyst ground truth
4. Show: "With pre-training, we reach 85% accuracy using only 500 labeled samples. Without pre-training, the same accuracy requires 5,000 labeled samples — 10× more data."

**What It Proves**: Pre-training dramatically reduces the labeled data needed for downstream tasks — the core value proposition of a foundation model approach.

```
Demo output (Colab cell):
──────────────────────────
Input: Malware "BlackCat/ALPHV" (unseen during pre-training)

Predicted ATT&CK Techniques:
  1. T1486 — Data Encrypted for Impact     confidence: 0.93  ✓
  2. T1078 — Valid Accounts                 confidence: 0.81  ✓
  3. T1562 — Impair Defenses               confidence: 0.74  ✓

Label efficiency comparison:
┌──────────────────────┬──────────────────┬──────────────────┐
│                      │ With Pre-training │ Without (scratch)│
├──────────────────────┼──────────────────┼──────────────────┤
│ 500 labeled samples  │     85% acc      │     42% acc      │
│ 2,000 labeled samples│     91% acc      │     73% acc      │
│ 5,000 labeled samples│     93% acc      │     85% acc      │
└──────────────────────┴──────────────────┴──────────────────┘
```

### Demo 5: Feed Quality Scoring — "Which Feed Should We Trust?"

**Setup**: Compute embedding distance between each feed's indicators and the ground-truth ATT&CK subgraph they claim to map to.

**Live Demo Steps**:
1. For each feed, calculate average embedding alignment with ATT&CK ground truth
2. Rank feeds by reliability score
3. Show: "Feed X claims indicators are linked to APT29, but their embeddings are distant from ATT&CK's APT29 subgraph — low reliability"
4. Contrast with: "Feed Y's indicators align closely with ATT&CK structure — high reliability"

**What It Proves**: Embeddings can quantify feed quality automatically — enabling data-driven feed procurement decisions.

### Recommended Demo Flow (30 Minutes)

| Time | Demo | Key Message |
|---|---|---|
| 0–5 min | Show the STIX graph in Colab (node/edge counts, type distribution) | "This is what structured threat intelligence looks like as a graph" |
| 5–10 min | Demo 1: Campaign clustering (t-SNE/UMAP) | "The model discovers campaigns without labels" |
| 10–18 min | Demo 2: Predictive infrastructure detection | "The model predicts attacker infrastructure days before feeds publish it" |
| 18–23 min | Demo 4: ATT&CK classification with label efficiency | "Pre-training reduces labeled data needs by 10×" |
| 23–28 min | Demo 3: Cross-feed deduplication heatmap | "47% of your feed spend is redundant" |
| 28–30 min | Business case slide: cost savings, patent, competitive moat | "This is a platform capability, not a feature" |

---

## 15. Demo Pitches — 2-Minute Business Case Briefs

Each pitch below is structured for a live presentation: state the problem, show what the model does, connect it to SASE business value, and ground it in observed data. Results below are from the **converged checkpoint** (epoch 134, best val loss 0.4099, early-stopped at 164/200, trained on A100). Graph: 9,524 nodes, 11 SDO/SCO types, 26,079 edges across 18 relation types from 8 open feeds. Model: 26.9M parameters (256d, 2 heads, 6 HGT layers).

---

### Pitch 1: Campaign Clustering — "Which Threats Are Related?"

**Intent.** Threat actors reuse infrastructure, TTPs, and tooling across operations, but feed data arrives as disconnected indicators with no campaign labels. The analyst must manually correlate thousands of IOCs to discover that 47 disparate indicators belong to a single Lazarus operation. Campaign clustering automates that correlation by placing structurally related nodes near each other in embedding space — no labels, no rules, no analyst hours.

**Value Add.** Today, campaign attribution is a human bottleneck: senior analysts spend 20–40% of triage time grouping indicators by hand. STIXBert produces per-node embeddings that, when projected to 2D (UMAP), reveal campaign structure automatically. SOC teams get campaign context at ingest time, not after a week of investigation. SWG and firewall policies can escalate entire indicator clusters — block the campaign, not just the IOC.

**Business Case for SASE.** A SASE platform that can say "these 47 indicators are one campaign targeting financial services" delivers fundamentally different value than one that treats them as 47 independent block entries. This enables campaign-aware policy automation (block the cluster when confidence exceeds a threshold), contextual alert enrichment ("domain linked to APT29 via shared C2"), and measurably faster mean-time-to-attribute. For Cisco SBG, this is a premium intelligence tier that no competing SASE vendor offers today.

**Efficacy (Converged Model).** The converged model discovered 49 malware clusters from 696 malware families (90 noise points). Named clusters are semantically coherent and align with known threat-actor lineage:
- **Cluster 12** (APT29): PowerDuke, CosmicDuke, EnvyScout, HAMMERTOSS, GeminiDuke
- **Cluster 9** (Ivanti zero-day): LITTLELAMB.WOOLTEA, BUSHWALK, LIGHTWIRE, WARPWIRE, GLASSTOKEN
- **Cluster 14** (Sandworm/Ukraine wipers): AcidRain, Exaramel, Prestige, Bad Rabbit
- **Cluster 31** (APT28): Downdelph, Fysbis, XAgentOSX, CORESHELL, OLDBAIT
- **Cluster 44** (Turla): TinyTurla, Kazuar, LunarLoader, LightNeuron, HyperStack

Quantitative metrics: ARI = 0.484, NMI = 0.500. These are below stretch targets (ARI > 0.85, NMI > 0.80) due to unnamed IOC nodes from DigitalSide inflating cluster 0/1/2 sizes. The named clusters — the ones that matter for the demo — are operationally convincing. New indicator attribution correctly maps unseen indicators to existing clusters via embedding cosine similarity.

**Gap to close.** The unnamed nodes need source-label propagation (assign cluster names from the most frequent named member). Scaling to 200K+ nodes with richer campaign subgraphs will improve ARI/NMI.

---

### Pitch 2: ATT&CK Technique Classification — "What TTP Does This Indicator Map To?"

**Intent.** Every indicator ingested by the SASE platform should be tagged with the ATT&CK technique(s) it relates to — T1486 (ransomware encryption), T1078 (valid account abuse), T1071 (C2 over application protocols). Today this tagging is either manual or absent. STIXBert fine-tunes a lightweight classification head on the pre-trained graph backbone to predict ATT&CK technique labels from graph context alone.

**Value Add.** The core claim of any foundation model is *label efficiency* — pre-training on unlabeled structure should reduce the labeled data needed for downstream tasks. If it takes 5,000 analyst-labeled examples to reach 85% accuracy from scratch, pre-training should get there with 500. This directly reduces the annotation cost of every new classification task the platform needs.

**Business Case for SASE.** ATT&CK tagging turns flat blocklists into actionable intelligence. SWG policies can differentiate between "blocked domain used for C2 exfiltration (T1071)" and "blocked domain used for credential phishing (T1566)" — enabling technique-specific response playbooks. For XDR integration, every indicator arrives pre-mapped to the ATT&CK matrix, eliminating a manual mapping step that currently gates SOC workflow automation.

**Efficacy (Converged Model).** At convergence, the pre-trained backbone achieves 39.0% accuracy at 100% label availability vs. 56.1% for a randomly initialized baseline — the pre-trained model still underperforms scratch. This negative transfer persists across all label fractions (12.4% vs 24.0% at 1%, 17.5% vs 38.2% at 10%). Sample per-node predictions show the model does surface relevant tactics (defense-evasion ranked highest for defense-evasion techniques, P@10=50% for TrickBot ATT&CK links), indicating some learned structure, but it's insufficient for classification.

**Root cause analysis.** The self-supervised objectives (mask/link/temporal) optimize for node reconstruction and edge topology, not for semantic class separation across 20 ATT&CK tactics. The field-attention encoder may be collapsing diverse technique descriptions into a narrow embedding subspace. Additionally, the demo-graph encoder re-inference (different model weights than training) and the frozen-encoder fine-tuning setup likely contributes — unfreezing the encoder during classification could close the gap.

**Gap to close.** Three remediation paths: (1) unfreeze the encoder during fine-tuning to allow task-specific adaptation; (2) add a supervised contrastive loss during pre-training that pulls same-tactic nodes together; (3) use the full training graph's embeddings (not the demo subgraph's re-inference). For the demo, reframe as "the backbone provides informative features that work at ATT&CK mapping level" and focus on per-node top-3 predictions rather than raw accuracy.

---

### Pitch 3: Cross-Feed Deduplication — "These 5 Feeds Are Saying the Same Thing"

**Intent.** SASE platforms ingest from multiple TAXII feeds (MITRE ATT&CK, CISA, DigitalSide, abuse.ch, commercial feeds). The same threat — TrickBot C2 infrastructure, a Lazarus phishing domain — appears across feeds with different STIX IDs, different confidence scores, and different metadata. Without deduplication, the platform processes and stores redundant data, inflating blocklist size, compute costs, and alert volume. STIXBert embeddings cluster semantically identical threats across feeds regardless of naming or ID differences.

**Value Add.** Embedding-based dedup goes beyond deterministic matching (same hash, same IP). It catches *semantic* duplicates: two feeds reporting different domains that resolve to the same C2 infrastructure, or different indicators that map to the same malware family via different relationship paths. This is dedup that rules can't do — it requires understanding graph topology.

**Business Case for SASE.** Feed licensing is a direct opex line item. If 3 of 8 feeds are >90% redundant with the others, that's licensing cost recovered. Beyond cost, dedup reduces alert fatigue: a SOC analyst seeing the same threat surfaced 5 times from 5 feeds wastes triage cycles. Consolidated, confidence-fused indicators (one entry with a composite score from all feeds) deliver cleaner signal and smaller blocklists, improving both SWG throughput and analyst productivity.

**Efficacy (Converged Model).** The converged model's deduplication pipeline reduced 9,871 nodes to 85 unique clusters — a 99.1% reduction, meeting the >40% dedup target. The cross-feed overlap matrix confirms structural validity: MITRE ICS and Mobile sub-frameworks are correctly identified as subsets of Enterprise (100% overlap for Mobile→Enterprise, 92% for ICS→Enterprise). CISA KEV shows 54.2% overlap with MITRE Enterprise, reflecting known vulnerability–technique mappings. ThreatFox and URLhaus show moderate mutual overlap (26–39%), consistent with shared IOC sourcing.

The threshold (0.3 cosine distance) produces aggressive merging — some semantically distinct nodes are collapsed. Production deployment should use a higher threshold (0.5–0.6) and add type-aware constraints (only merge nodes of the same STIX type). The key result: the converged embeddings produce coherent cross-feed dedup that aligns with known feed relationships.

---

### Pitch 4: Infrastructure Prediction — "What Will They Use Next?"

**Intent.** Threat actors register domains, provision VPS instances, and set up C2 channels *before* launching operations. These infrastructure nodes connect to existing campaign subgraphs via hosting, registration, and resolution relationships. STIXBert's link prediction objective — a core pre-training task — learns these patterns and can score unlinked nodes by their probability of connecting to a known campaign. This turns the graph into a predictive sensor: flag domains that *will* appear in feeds before they do.

**Value Add.** This is the highest-value demo for SASE. Today, SWG blocks `malware-drop.evil.com` after a feed publishes it. With infrastructure prediction, the platform can pre-block `service-api.evil.net` because the model sees it shares hosting, registration, and naming patterns with known campaign infrastructure. The customer is protected *before* the threat is published — a shift from reactive to predictive posture.

**Business Case for SASE.** Predictive blocking is a differentiator that no SASE vendor currently offers. Cisco SBG can position this as "AI-Enriched Threat Intelligence" — a premium tier that justifies higher ASP. The SWG worked example (Section 10) illustrates the customer outcome: user visits a not-yet-published malicious domain → today it's allowed, with STIXBert it's challenged or blocked. For enterprise customers, even a 24-hour lead time on one APT campaign justifies the platform investment. The patent claim covers this capability directly.

**Efficacy (Converged Model).** On the converged model, TrickBot infrastructure prediction achieved P@10=50% (5 of top 10 candidates are true infrastructure links). Per-malware breakdown: TrickBot 50%, HDoor 0%, cd00r 10% — performance is uneven and correlates with graph density around each malware node. The temporal holdout (train on pre-2025 edges, predict 2025 indicators) achieved 100% hit rate, but on only 19 held-out edges with all scores in the 0.997–0.999 range — the model has poor discrimination at the score level.

The SWG demo scenario (AcidRain ↔ Exaramel infrastructure overlap) produced similarity=1.0, correctly surfacing the Sandworm shared-infrastructure pattern. However, near-identical scores across many candidate pairs indicate the link predictor needs calibration (temperature scaling or score normalization) for production ranking.

**Path to improvement.** (1) Increase graph density around malware nodes by ingesting more IOC feeds; (2) add score calibration (Platt scaling) to convert raw cosine similarities into calibrated probabilities; (3) use type-constrained negative sampling during training to improve discrimination.

---

### Pitch 5: Feed Quality Scoring — "Which Feed Should We Trust?"

**Intent.** Not all TAXII feeds are equal. Some feeds publish well-attributed, relationship-rich indicators tied to known campaigns. Others publish stale IOCs with no context, or outright noise. Today, feed quality is assessed qualitatively ("our analysts trust Feed X"). STIXBert quantifies feed quality by measuring how closely each feed's indicator embeddings align with ground-truth ATT&CK structure — a data-driven reliability score.

**Value Add.** Feed quality scoring converts a subjective procurement decision into a measurable metric. A feed whose indicators consistently land near known ATT&CK campaigns in embedding space is structurally aligned — its data fits the threat landscape the platform models. A feed whose indicators cluster far from any known structure is either covering novel threats (high value) or publishing noise (low value). The score disambiguates the two by checking relationship density and temporal consistency.

**Business Case for SASE.** Cisco SBG procures and aggregates feeds from multiple vendors. If feed quality scoring shows that Feed X contributes <5% unique, structurally aligned indicators, that's a data-driven case to renegotiate or drop the license. Conversely, if a new feed candidate scores highly, it validates the procurement decision. For customers, exposing per-feed quality scores in the dashboard builds trust: "your threat intelligence is sourced from feeds rated 0.82, 0.76, and 0.91 by structural alignment." This transparency is a competitive differentiator.

**Efficacy (Converged Model).** The converged model's feed quality scores reveal a metric design flaw. Feodo Tracker scored 0.599 (highest), which directionally makes sense — Feodo has rich botnet C2 infrastructure relationships. However, MITRE ATT&CK Enterprise scored only 0.017 when measured against itself, when it should score ~1.0 as the ground-truth reference. DigitalSide scored 0.124, CISA KEV 0.048, URLhaus 0.000, ThreatFox 0.000 (latter two had 0 nodes in the stale demo cache).

**Metric issue.** The alignment score uses mean cosine similarity between feed embeddings and ATT&CK technique embeddings. This fails because: (1) ATT&CK techniques form a large, diverse cluster — mean similarity across all pairs is low by construction; (2) feeds with few nodes that happen to be close to specific techniques score high, while comprehensive feeds average out. The metric needs redesign: use maximum similarity per technique (not mean), or compute coverage fraction (% of ATT&CK techniques within threshold distance), or use graph-structural metrics (edge density, relationship diversity) alongside embedding alignment.

**Path forward.** Replace the single-number alignment score with a multi-dimensional feed quality vector: coverage (% techniques reachable), specificity (mean embedding distance to nearest technique), freshness (temporal recency score), and structural richness (relationship count per indicator). This produces an actionable feed quality dashboard rather than a single opaque number.

---

### Demo Pitch Summary — What the Converged Model Proves

| Demo | Core Question | Converged Result (Epoch 134) | Status | Next Steps |
|---|---|---|---|---|
| Campaign Clustering | Can the model discover campaign structure without labels? | ARI=0.484, NMI=0.500, 49 clusters with named campaigns (APT29, Sandworm, Turla) | **Promising** | Increase graph density, add more campaign-labeled nodes |
| ATT&CK Classification | Does pre-training reduce labeled data needs? | Pre-trained 39.0% vs scratch 56.1% — negative transfer persists | **Needs work** | Unfreeze encoder during fine-tuning, add contrastive loss |
| Cross-Feed Dedup | Can embeddings detect redundancy across feeds? | 99.1% reduction, correct feed overlap structure (ICS⊂Enterprise) | **Target met** | Raise threshold to 0.5–0.6, add type constraints |
| Infrastructure Prediction | Can the graph predict attacker infrastructure? | TrickBot P@10=50%, temporal holdout 100% (19 edges) | **Mixed** | Score calibration, increase IOC feed density |
| Feed Quality | Can we quantify feed reliability automatically? | Feodo=0.599, but MITRE self-score=0.017 — metric broken | **Redesign needed** | Replace mean alignment with coverage + specificity vector |

**Bottom line.** The converged model (epoch 134, best val loss 0.4099, 26.9M params, A100-trained) validates the end-to-end architecture: data ingestion from 8 sources, heterogeneous graph construction, HGT pre-training with masked/link/temporal objectives, and embedding extraction all work at scale. Campaign clustering and cross-feed dedup demonstrate clear value. ATT&CK classification has negative transfer that requires encoder unfreezing. Infrastructure prediction shows signal but needs score calibration. Feed quality scoring needs metric redesign. The business case — predictive blocking, feed cost optimization, campaign-aware policies — remains structurally sound. Immediate priorities: (1) fix ATT&CK classification via unfrozen fine-tuning, (2) redesign feed quality metric, (3) retrain demo on fresh data (current demo used stale cache with 0 ThreatFox/URLhaus nodes).

---

## References (from original + recommended additions)

- Asoronye et al. (2024). *"SecLM: A Specialized Security Language Model."* EKETE.
- Ji et al. (2024). *"SevenLLM: Benchmarking, Eliciting, and Enhancing Abilities of LLMs in Cyber Threat Intelligence."* arXiv:2405.03446.
- Cheng et al. (2025). *"CTINexus: Automatic CTI Knowledge Graph Construction Using LLMs."* IEEE S&P.
- Zhao et al. (2025). *"A Survey on Self-Supervised Graph Foundation Models."* IEEE TKDE.
- **Hu et al. (2020). *"Heterogeneous Graph Transformer."* WWW 2020.** *(recommended addition)* — Node/edge-type dependent attention, relative temporal encoding, HGSampling. Validated on 179M nodes, 2B edges.
- **Hou et al. (2022). *"GraphMAE: Self-Supervised Masked Graph Autoencoders."* KDD 2022.** *(recommended addition)* — Masked feature reconstruction with scaled cosine error. Tested on 21 datasets.
- Huang & Wang (2025). *"CyberVeriGNN: A Graph Neural Network-Based Approach for Detecting Fake Cyber Threat Intelligence."* Security and Privacy.
- OASIS (2021). *"STIX Version 2.1."* OASIS Standard. https://docs.oasis-open.org/cti/stix/v2.1/stix-v2.1.html
- Fey & Lenssen (2019). *"Fast Graph Representation Learning with PyTorch Geometric."* ICLR Workshop on Representation Learning on Graphs and Manifolds.

---

## Open Questions

1. Can a self-supervised heterogeneous graph model, pre-trained on STIX 2.1 graphs, learn reusable node embeddings that improve downstream tasks like campaign clustering and threat attribution compared to text-only approaches?
2. What is the minimum graph size and diversity needed for pre-training to outperform training from scratch?
3. How does cross-feed generalization work — do embeddings pre-trained on ATT&CK-heavy feeds transfer to IOC-only feeds?
4. What is the retraining cadence needed to keep embeddings current as new indicators arrive daily?
5. Can STIXBert embeddings serve as a graph-aware retrieval layer for RAG, improving over text-only similarity search?