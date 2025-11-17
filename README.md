# picrust2-Tax4Fun2-benchmark



Reproducible benchmarking of 16S-based functional inference tools (**PICRUSt2** and **Tax4Fun2**) against shotgun metagenomics on the Mendes et al. (2017) rhizosphere soil dataset (PRJEB14409).

This repository contains all the code and scripts used in my master’s thesis:

> **“Inference of Microbial Genomic Data: A Benchmark of Existing Tools”**  
> University of Padova — Farhad Sadat

The pipeline goes from QIIME 2 preprocessing of 16S data, through PICRUSt2 and Tax4Fun2 functional predictions, to KO-level benchmarking against shotgun metagenomes and cross-method comparisons.

---

## 🔍 Overview

| Component              | Tool / Source                                                                 | Notes                                      |
|------------------------|-------------------------------------------------------------------------------|-------------------------------------------|
| 16S rRNA data          | ENA: [PRJEB14409](https://www.ebi.ac.uk/ena/browser/view/PRJEB14409)          | Mendes et al. 2017 – rhizosphere soils    |
| Shotgun KO table       | [ssun6/Inference_picrust](https://github.com/ssun6/Inference_picrust)        | `soil_LWM` subset only                    |
| 16S → function (tool 1)| **PICRUSt2**                                                                  | Phylogeny-based KO prediction             |
| 16S → function (tool 2)| **Tax4Fun2**                                                                  | Taxonomy-based KO prediction              |
| Benchmark reference    | [Sun et al., *Microbiome* (2020)](https://doi.org/10.1186/s40168-020-00815-y) | Mendes soil benchmark, extended here      |
| Metrics vs shotgun     | Spearman ρ, Jaccard similarity                                                | Per-sample KO profiles                    |
| Cross-method metrics   | Spearman ρ, Jaccard, KO-wise agreement                                        | PICRUSt2 vs Tax4Fun2 on same samples      |

---

## 🧪 What this repo does

1. **Preprocess Mendes 16S data in QIIME 2**  
   - Import demultiplexed reads  
   - Trim primer, denoise with DADA2 (single-end)  
   - Export ASV table and representative sequences

2. **Run PICRUSt2 on the ASV table**  
   - Place ASVs onto reference tree  
   - Predict KO abundances per sample  
   - Normalize to CPM (counts per million)

3. **Run Tax4Fun2 on the same 16S data**  
   - Build or use an existing SILVA-based reference  
   - Map ASVs to reference taxa  
   - Predict KO abundances per sample

4. **Align 16S-based predictions with shotgun KO tables**  
   - Match samples using ENA alias metadata  
   - Harmonize KO IDs and sample names  
   - Keep the Mendes soil LWM subset for benchmarking

5. **Benchmark and visualize**  
   - Compute Spearman correlations and Jaccard similarities  
     - **16S tool (PICRUSt2 or Tax4Fun2) vs shotgun** for each sample  
   - Summarize distributions per compartment (bulk, rhizosphere, root-associated)  
   - Compare PICRUSt2 and Tax4Fun2 directly (KO-wise and sample-wise)  
   - Generate scatter plots, boxplots and summary panels used in the thesis

---

## 📂 Key folders (orientation)

- `qiime2/` – QIIME 2 import, trimming, DADA2, and exports  
- `picrust2_se/` – PICRUSt2 single-end pipeline and outputs  
- `tax4fun2_out/` – Tax4Fun2 functional predictions and intermediate tables  
- `meta/` – Mendes metadata and ENA run ↔ sample alias mapping  
- `analysis/` – Python/R scripts for KO alignment, metrics and plots  
- `figures/` – Exported plots and panels used in the thesis

---

## ▶️ Example: Mendes soil benchmark vs shotgun (PICRUSt2)

PICRUSt2 vs shotgun KO benchmarking for the Mendes soil_LWM subset:

```bash
python analysis/benchmark_mendes_soil.py \
  --base . \
  --pred picrust2_se/KO_metagenome_out/pred_metagenome_unstrat_cpm.tsv \
  --alias-tsv meta/PRJEB14409_runs_with_alias.tsv \
  --shotgun-lwm-dir Inference_picrust/soil_LWM \
  --outdir results \
  --scatter-samples RTP1 BulkTP1 RAG1

