"""
getExamples.py — Kepler KOI Manifest Generator

Queries the NASA Exoplanet Archive KOI cumulative table via TAP/ADQL and writes
koi_manifest.csv with labelled targets for the centroid-resolution degradation study.

Class definition:
  Positive (label=1): koi_disposition = CONFIRMED
  Negative (label=0): koi_disposition = FALSE POSITIVE (all subtypes, default)
                  OR: koi_disposition = FALSE POSITIVE AND koi_fpflag_co = 1 (--fp-subset centroid_offset)

Group identity: kepid — all KOIs from one KIC star must stay together in CV and test splits.

Usage:
    python getExamples.py [--fp-subset {all,centroid_offset}]
"""

import argparse
import io
import sys
import urllib.parse
import urllib.request

import pandas as pd

TAP_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"

ADQL_QUERY = """
SELECT kepid, kepoi_name, koi_disposition, koi_pdisposition,
       koi_fpflag_nt, koi_fpflag_ss, koi_fpflag_co, koi_fpflag_ec,
       koi_period, koi_time0bk, koi_duration
FROM cumulative
""".strip()

FALLBACK_TABLE = "q1_q17_dr25_koi"

REQUIRED_COLUMNS = {
    "kepid", "kepoi_name", "koi_disposition",
    "koi_fpflag_nt", "koi_fpflag_ss", "koi_fpflag_co", "koi_fpflag_ec",
    "koi_period", "koi_time0bk", "koi_duration",
}


def fetch_koi_table(table: str) -> pd.DataFrame:
    query = ADQL_QUERY.replace("cumulative", table)
    params = urllib.parse.urlencode({"query": query, "format": "csv"})
    url = f"{TAP_URL}?{params}"
    print(f"  Querying {TAP_URL} (table={table}) ...")
    with urllib.request.urlopen(url, timeout=120) as resp:
        return pd.read_csv(io.BytesIO(resp.read()))


def fp_subtype(row: pd.Series) -> str:
    """Encode which FP flags are set as a short string for subgroup analysis."""
    if row["label"] == 1:
        return "planet"
    parts = []
    if row.get("koi_fpflag_co", 0) == 1:
        parts.append("co")
    if row.get("koi_fpflag_ss", 0) == 1:
        parts.append("ss")
    if row.get("koi_fpflag_nt", 0) == 1:
        parts.append("nt")
    if row.get("koi_fpflag_ec", 0) == 1:
        parts.append("ec")
    return "+".join(parts) if parts else "other"


def main():
    parser = argparse.ArgumentParser(description="Build KOI manifest for degradation study")
    parser.add_argument(
        "--fp-subset",
        choices=["all", "centroid_offset"],
        default="all",
        help=(
            "all (default): all FALSE POSITIVE subtypes as negatives; "
            "centroid_offset: only FPs with koi_fpflag_co=1"
        ),
    )
    args = parser.parse_args()

    # Fetch from primary table; fall back on column-name mismatch or HTTP error
    df = None
    for table in ["cumulative", FALLBACK_TABLE]:
        try:
            df = fetch_koi_table(table)
            missing = REQUIRED_COLUMNS - set(df.columns)
            if missing:
                print(f"  Table '{table}' missing columns: {missing}. Trying fallback...")
                df = None
                continue
            print(f"  Fetched {len(df):,} rows from '{table}'.")
            break
        except Exception as exc:
            print(f"  Table '{table}' failed ({exc}). Trying fallback...")

    if df is None:
        print("ERROR: could not fetch KOI table from either source. Check network access.")
        sys.exit(1)

    # Drop rows with null or zero ephemerides — these cannot be phase-folded
    n_before = len(df)
    df = df.dropna(subset=["koi_period", "koi_time0bk", "koi_duration"])
    df = df[
        (df["koi_period"] > 0) &
        (df["koi_time0bk"] > 0) &
        (df["koi_duration"] > 0)
    ]
    print(f"  Dropped {n_before - len(df):,} rows with null/zero ephemerides.")

    # Fill FP flag NaNs with 0
    for col in ["koi_fpflag_nt", "koi_fpflag_ss", "koi_fpflag_co", "koi_fpflag_ec"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    # Build label column
    disp = df["koi_disposition"].str.strip().str.upper()
    confirmed_mask = disp == "CONFIRMED"
    fp_mask = disp == "FALSE POSITIVE"

    if args.fp_subset == "centroid_offset":
        # Option B: restrict negatives to centroid-offset FPs only
        neg_mask = fp_mask & (df["koi_fpflag_co"] == 1)
        print("\nDataset mode: Option B — centroid-offset FPs only (koi_fpflag_co=1).")
        print("NOTE: This focuses on the FP subtype centroid methods are designed to detect.")
        print("      AUC numbers may be optimistic; document this limitation in the paper.")
    else:
        # Option A (default): all FP subtypes
        neg_mask = fp_mask
        print("\nDataset mode: Option A — all FALSE POSITIVE subtypes as negatives.")
        print("  Subgroup performance (co/ss/nt/ec) will be reported separately.")

    positives = df[confirmed_mask].copy()
    negatives = df[neg_mask].copy()

    positives["label"] = 1
    negatives["label"] = 0

    manifest = pd.concat([positives, negatives], ignore_index=True)
    manifest["fp_subtype"] = manifest.apply(fp_subtype, axis=1)

    # Keep provenance columns; rename for clarity
    out = manifest[[
        "kepid", "kepoi_name", "label", "fp_subtype",
        "koi_disposition", "koi_pdisposition",
        "koi_fpflag_nt", "koi_fpflag_ss", "koi_fpflag_co", "koi_fpflag_ec",
        "koi_period", "koi_time0bk", "koi_duration",
    ]].copy()

    # koi_time0bk is BKJD (BJD − 2454833.0) — use directly; do not subtract 2457000
    out.to_csv("koi_manifest.csv", index=False)

    # Print summary
    n_pos = (out["label"] == 1).sum()
    n_neg = (out["label"] == 0).sum()
    n_kic = out["kepid"].nunique()
    print(f"\nWrote koi_manifest.csv — {len(out):,} total KOIs from {n_kic:,} unique KIC targets")
    print(f"  Positives (CONFIRMED):  {n_pos:,}")
    print(f"  Negatives (FP):         {n_neg:,}")

    if args.fp_subset == "all" and n_neg > 0:
        fp_df = out[out["label"] == 0]
        print("\n  FP subgroup breakdown:")
        for subtype, count in fp_df["fp_subtype"].value_counts().items():
            print(f"    {subtype}: {count:,}")

        n_co = int((out["koi_fpflag_co"] == 1).sum())
        n_ss = int((out["koi_fpflag_ss"] == 1).sum())
        n_no_centroid = int(fp_df[fp_df["koi_fpflag_co"] == 0].shape[0])
        print(f"\n  Of {n_neg:,} FP negatives:")
        print(f"    centroid-offset (koi_fpflag_co=1): {n_co:,}")
        print(f"    significant-secondary (koi_fpflag_ss=1, co=0): {n_ss - n_co if n_ss >= n_co else n_ss:,} (approx)")
        print(f"    without centroid offset flag: {n_no_centroid:,}")
        print(
            "\n  On-target EBs (koi_fpflag_ss=1, koi_fpflag_co=0) are INCLUDED as negatives "
            "(Option A). These have no centroid signature; subgroup reporting lets the paper "
            "discuss their effect on classification performance without baking the answer into "
            "the dataset. See CLAUDE.md for the Option A / Option B tradeoff."
        )


if __name__ == "__main__":
    main()
