#!/usr/bin/env python3
import argparse
import os
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

def load_picrust(path: str) -> pd.DataFrame:
    """
    PICRUSt2 pred_metagenome_unstrat.tsv.gz
    Expects first column to be KO/function and subsequent columns = samples (ERR...).
    """
    df = pd.read_csv(path, sep="\t", compression="infer")
    # First column can be named 'function' or similar; force name
    first = df.columns[0]
    df[first] = df[first].astype(str)
    df[first] = (
        df[first]
        .str.replace(r"^ko:", "", regex=True, case=False)
        .str.replace(r"^KO:", "", regex=True, case=False)
    )
    df = df.rename(columns={first: "KO"}).set_index("KO")
    # numeric
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return df

def load_shotgun(path: str) -> pd.DataFrame:
    """
    Shotgun KO table (rows=KO, cols=samples), e.g. rhizo_wgs_p.txt
    Treat first column as index (KO).
    """
    df = pd.read_csv(path, sep="\t", engine="python", index_col=0)
    df.index = (
        df.index.astype(str)
        .str.replace(r"^ko:", "", regex=True, case=False)
        .str.replace(r"^KO:", "", regex=True, case=False)
    )
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return df

def load_map(path: str) -> pd.DataFrame:
    """
    CSV with columns: ERR,shotgun_id
    """
    m = pd.read_csv(path)
    if not {"ERR", "shotgun_id"}.issubset(set(m.columns)):
        raise ValueError("sample map must have columns: ERR,shotgun_id")
    # Drop duplicates and empty
    m = m.dropna(subset=["ERR", "shotgun_id"]).drop_duplicates()
    return m

def jaccard_presence(a: np.ndarray, b: np.ndarray, threshold: float = 0.0) -> float:
    """
    Jaccard on presence/absence with > threshold considered present.
    """
    a_bin = (a > threshold).astype(np.int8)
    b_bin = (b > threshold).astype(np.int8)
    inter = int((a_bin & b_bin).sum())
    union = int((a_bin | b_bin).sum())
    return inter / union if union > 0 else np.nan

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert each column to relative abundances (sum=1).
    """
    colsum = df.sum(axis=0)
    colsum[colsum == 0] = 1.0
    return df.div(colsum, axis=1)

def main():
    p = argparse.ArgumentParser(description="Compare PICRUSt2 vs shotgun KO per sample (column-wise).")
    p.add_argument("--picrust", required=True, help="pred_metagenome_unstrat.tsv.gz")
    p.add_argument("--shotgun", required=True, help="rhizo_wgs_p.txt")
    p.add_argument("--sample-map", required=True, help="CSV with columns: ERR,shotgun_id")
    p.add_argument("--outdir", required=True, help="Output directory")
    p.add_argument("--normalize", action="store_true",
                   help="Normalize both matrices to relative abundance per column before comparison")
    p.add_argument("--presence-threshold", type=float, default=0.0,
                   help="Threshold for presence/absence in Jaccard (default >0)")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Load
    pic = load_picrust(args.picrust)
    wgs = load_shotgun(args.shotgun)
    smap = load_map(args.sample_map)

    # Keep only samples present in both tables
    keep = smap[smap["ERR"].isin(pic.columns) & smap["shotgun_id"].isin(wgs.columns)].copy()
    if keep.empty:
        raise RuntimeError("No overlapping samples between PICRUSt and shotgun using the provided mapping.")

    # Subset and align columns using the mapping
    pic_sub = pic[keep["ERR"].tolist()].copy()
    pic_sub.columns = keep["shotgun_id"].tolist()  # rename to shotgun IDs for direct column matching

    common_samples = sorted(set(pic_sub.columns).intersection(set(wgs.columns)))
    if not common_samples:
        raise RuntimeError("No common sample names after renaming PICRUSt columns to shotgun IDs.")
    pic_sub = pic_sub[common_samples]
    wgs_sub = wgs[common_samples]

    # Optionally normalize to relative abundance
    if args.normalize:
        pic_sub = normalize_columns(pic_sub)
        wgs_sub = normalize_columns(wgs_sub)

    # Align rows (KOs)
    common_kos = sorted(set(pic_sub.index).intersection(set(wgs_sub.index)))
    if not common_kos:
        raise RuntimeError("No overlapping KO IDs between PICRUSt and shotgun.")
    pic_aln = pic_sub.loc[common_kos].copy()
    wgs_aln = wgs_sub.loc[common_kos].copy()

    # Compute per-sample metrics
    records = []
    for s in common_samples:
        x = pic_aln[s].to_numpy()
        y = wgs_aln[s].to_numpy()
        # Spearman
        rho, _ = spearmanr(x, y)
        # Jaccard (presence/absence)
        jac = jaccard_presence(x, y, threshold=args.presence_threshold)
        # Counts
        records.append({
            "sample": s,
            "spearman": rho,
            "jaccard": jac,
            "nonzero_picrust": int((x > args.presence_threshold).sum()),
            "nonzero_shotgun": int((y > args.presence_threshold).sum()),
            "n_kos_compared": len(common_kos)
        })
    metrics = pd.DataFrame.from_records(records).sort_values("sample")
    metrics_path = os.path.join(args.outdir, "colwise_metrics.csv")
    metrics.to_csv(metrics_path, index=False)

    # Simple histograms
    plt.figure()
    metrics["spearman"].dropna().hist(bins=15)
    plt.xlabel("Spearman correlation (per sample)")
    plt.ylabel("Count of samples")
    plt.title("PICRUSt2 vs Shotgun KO — Spearman (column-wise)")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "spearman_hist.png"), dpi=150)
    plt.close()

    plt.figure()
    metrics["jaccard"].dropna().hist(bins=15)
    plt.xlabel("Jaccard (presence/absence, per sample)")
    plt.ylabel("Count of samples")
    plt.title("PICRUSt2 vs Shotgun KO — Jaccard (column-wise)")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "jaccard_hist.png"), dpi=150)
    plt.close()

    # Console summary
    med_spear = metrics["spearman"].median()
    med_jacc = metrics["jaccard"].median()
    print(f"Samples compared: {len(common_samples)}")
    print(f"Common KOs: {len(common_kos)}")
    print(f"Median Spearman: {med_spear:.3f}")
    print(f"Median Jaccard: {med_jacc:.3f}")
    print(f"Saved: {metrics_path}")

if __name__ == "__main__":
    main()
