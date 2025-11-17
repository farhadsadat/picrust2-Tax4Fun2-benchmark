#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from pathlib import Path

def read_picrust(path: str) -> pd.DataFrame:
    # PICRUSt KO table (pred_metagenome_unstrat.tsv.gz)
    df = pd.read_csv(path, sep="\t", compression="infer")
    # first column should be function / KO id
    if df.columns[0].lower() not in ("function", "ko", "id"):
        raise ValueError(f"Unexpected first column name in PICRUSt file: {df.columns[0]}")
    df = df.set_index(df.columns[0])
    # Convert to numeric (coerce errors -> NaN -> 0)
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    # Normalize KO ids to plain Kxxxxx
    df.index = df.index.astype(str).str.replace("^ko:", "", regex=True).str.strip()
    return df

def read_shotgun(path: str) -> pd.DataFrame:
    # Shotgun KO table (rhizo_wgs_p.txt), tab-separated, quoted headers
    df = pd.read_csv(path, sep="\t", engine="python", index_col=0)
    # Strip quotes from column names if present
    df.columns = [str(c).replace('"', '').strip() for c in df.columns]
    # KO ids may be like ko:K00003 -> normalize to K00003
    df.index = df.index.astype(str).str.replace("^ko:", "", regex=True).str.strip()
    # Ensure numeric
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return df

def load_map(path: str) -> pd.DataFrame:
    # CSV with columns: ERR,shotgun_id
    m = pd.read_csv(path)
    needed = {"ERR", "shotgun_id"}
    if not needed.issubset({c.strip() for c in m.columns}):
        raise ValueError("sample_map must have columns: ERR, shotgun_id")
    m["ERR"] = m["ERR"].astype(str).str.strip()
    m["shotgun_id"] = m["shotgun_id"].astype(str).str.strip()
    # Drop duplicates and NA
    m = m.dropna().drop_duplicates(subset=["ERR", "shotgun_id"])
    return m

def col_align(pic: pd.DataFrame, wgs: pd.DataFrame, mapping: pd.DataFrame):
    # Keep only pairs that exist in both matrices
    pairs = mapping[
        mapping["ERR"].isin(pic.columns) & mapping["shotgun_id"].isin(wgs.columns)
    ].copy()

    # (Optional) If professor flagged a sample with zero reads after DADA2 (e.g., ERR1456820),
    # drop it here if present
    pairs = pairs[pairs["ERR"] != "ERR1456820"]

    if pairs.empty:
        raise ValueError("No overlapping samples between PICRUSt and WGS after mapping.")

    pic_aligned = pic[pairs["ERR"].tolist()].copy()
    wgs_aligned = wgs[pairs["shotgun_id"].tolist()].copy()
    # Rename WGS columns to the ERR ids so both share identical column names
    wgs_aligned.columns = pairs["ERR"].tolist()
    return pic_aligned, wgs_aligned

def to_relative(df: pd.DataFrame) -> pd.DataFrame:
    colsum = df.sum(axis=0)
    colsum[colsum == 0] = 1.0
    return df.div(colsum, axis=1)

def jaccard_row(x: np.ndarray, y: np.ndarray) -> float:
    xb = (x > 0)
    yb = (y > 0)
    inter = np.logical_and(xb, yb).sum()
    union = np.logical_or(xb, yb).sum()
    return float(inter) / float(union) if union > 0 else np.nan

def main():
    ap = argparse.ArgumentParser(description="Row-wise KO comparison (PICRUSt vs Shotgun)")
    ap.add_argument("--picrust", required=True, help="PICRUSt KO table (pred_metagenome_unstrat.tsv.gz)")
    ap.add_argument("--shotgun", required=True, help="Shotgun KO table (rhizo_wgs_p.txt)")
    ap.add_argument("--sample-map", required=True, help="CSV mapping with columns: ERR,shotgun_id")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--normalize", action="store_true", help="Column-wise relative normalization before stats")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    pic = read_picrust(args.picrust)
    wgs = read_shotgun(args.shotgun)
    mapping = load_map(args.sample_map)

    picA, wgsA = col_align(pic, wgs, mapping)

    if args.normalize:
        picA = to_relative(picA)
        wgsA = to_relative(wgsA)

    # Intersect KO ids
    common_kos = picA.index.intersection(wgsA.index)
    picA = picA.loc[common_kos].copy()
    wgsA = wgsA.loc[common_kos].copy()

    # Compute row-wise metrics
    rows = []
    for ko in common_kos:
        x = picA.loc[ko].to_numpy(dtype=float)
        y = wgsA.loc[ko].to_numpy(dtype=float)

        # Spearman
        rho = spearmanr(x, y).correlation
        # Jaccard on presence/absence
        jac = jaccard_row(x, y)

        # extra: intersection count, union count (presence/absence)
        xb = (x > 0); yb = (y > 0)
        inter = int(np.logical_and(xb, yb).sum())
        union = int(np.logical_or(xb, yb).sum())

        rows.append({"KO": ko, "spearman": rho, "jaccard": jac,
                     "intersect_nonzero": inter, "union_nonzero": union})

    res = pd.DataFrame(rows).set_index("KO")
    res.to_csv(outdir / "rowwise_metrics.csv")

    # Quick summary text
    med_spear = np.nanmedian(res["spearman"].values)
    med_jac = np.nanmedian(res["jaccard"].values)
    with open(outdir / "rowwise_summary.txt", "w") as f:
        f.write(f"Common KOs: {len(common_kos)}\n")
        f.write(f"Median Spearman (row-wise): {med_spear:.3f}\n")
        f.write(f"Median Jaccard (row-wise): {med_jac:.3f}\n")

    # Histograms
    plt.figure()
    res["spearman"].dropna().hist(bins=40)
    plt.xlabel("Row-wise Spearman")
    plt.ylabel("Count of KOs")
    plt.title("Row-wise Spearman (PICRUSt vs Shotgun)")
    plt.tight_layout()
    plt.savefig(outdir / "rowwise_spearman_hist.png", dpi=200)
    plt.close()

    plt.figure()
    res["jaccard"].dropna().hist(bins=40)
    plt.xlabel("Row-wise Jaccard (presence/absence)")
    plt.ylabel("Count of KOs")
    plt.title("Row-wise Jaccard (PICRUSt vs Shotgun)")
    plt.tight_layout()
    plt.savefig(outdir / "rowwise_jaccard_hist.png", dpi=200)
    plt.close()

    print(f"Common KOs: {len(common_kos)}")
    print(f"Median Spearman (row-wise): {med_spear:.3f}")
    print(f"Median Jaccard (row-wise): {med_jac:.3f}")
    print(f"Saved: {outdir/'rowwise_metrics.csv'}, {outdir/'rowwise_summary.txt'}")
    print(f"Saved plots: {outdir/'rowwise_spearman_hist.png'}, {outdir/'rowwise_jaccard_hist.png'}")

if __name__ == "__main__":
    main()
