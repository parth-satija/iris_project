"""
export_feature_correlations.py

analyze_safety_index.py prints its single-feature log-log correlations and throws them away. This saves them
(same loading code, same target: body_position_error_at_trigger) for figure 3 of plot_report_figures.py.

    python export_feature_correlations.py --csv-dir outputs/csv --tag all
    python export_feature_correlations.py --csv-dir outputs/csv_3body --tag 3body

Output: outputs/analysis/single_feature_corr_<tag>.csv  (feature, n, log_log_corr)
"""

from __future__ import annotations

import argparse
import os

import analyze_safety_index as asi


def main() -> None:
    root = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description="Save the single-feature log-log correlations for plotting.")
    ap.add_argument("--csv-dir", default=os.path.join(root, "outputs", "csv"))
    ap.add_argument("--tag", default="all", help="suffix of the output file (all = mixed bodies, 3body = 3-body only)")
    ap.add_argument("--min-rows", type=int, default=20)
    ap.add_argument("--out-dir", default=os.path.join(root, "outputs", "analysis"))
    args = ap.parse_args()

    df = asi.load_all_corrections(args.csv_dir, args.min_rows)
    corr = asi.log_log_correlations(df)
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, f"single_feature_corr_{args.tag}.csv")
    corr.to_csv(path, index=False)
    print(corr.to_string(index=False))
    print(f"\nSaved {path}  ({len(df)} correction rows)")


if __name__ == "__main__":
    main()
