"""Profile the ExtraSensory release without materializing all users at once."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from trustsense.schema import label_columns, modality_columns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw/extrasensory"),
        help="Directory containing per-user .csv.gz files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/dataset_profile"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = sorted(args.data_dir.glob("*.csv.gz"))
    if not files:
        raise FileNotFoundError(f"No .csv.gz user files found under {args.data_dir}")

    first_columns = pd.read_csv(files[0], nrows=0).columns.tolist()
    labels = label_columns(first_columns)
    modalities = modality_columns(first_columns)

    label_positive = pd.Series(0, index=labels, dtype="int64")
    label_observed = pd.Series(0, index=labels, dtype="int64")
    label_users_positive = pd.Series(0, index=labels, dtype="int64")
    modality_rows_available: dict[str, int] = defaultdict(int)
    modality_cell_observed: dict[str, int] = defaultdict(int)
    modality_cell_total: dict[str, int] = defaultdict(int)
    user_rows: list[dict[str, object]] = []
    total_rows = 0

    usecols = ["timestamp", *labels]
    for columns in modalities.values():
        usecols.extend(columns)
    usecols = list(dict.fromkeys(usecols))

    for path in files:
        frame = pd.read_csv(path, usecols=usecols, low_memory=False)
        n_rows = len(frame)
        total_rows += n_rows
        user_id = path.name.split(".features_labels")[0]

        user_record: dict[str, object] = {"user_id": user_id, "rows": n_rows}
        for label in labels:
            observed = frame[label].notna()
            positive = frame[label].eq(1)
            label_observed[label] += int(observed.sum())
            label_positive[label] += int(positive.sum())
            label_users_positive[label] += int(positive.any())

        for modality, columns in modalities.items():
            values = frame[columns]
            row_available = values.notna().any(axis=1)
            modality_rows_available[modality] += int(row_available.sum())
            modality_cell_observed[modality] += int(values.notna().sum().sum())
            modality_cell_total[modality] += int(values.size)
            user_record[f"{modality}_row_coverage"] = float(row_available.mean())

        user_rows.append(user_record)

    label_summary = pd.DataFrame(
        {
            "label": labels,
            "positive_examples": label_positive.values,
            "observed_examples": label_observed.values,
            "positive_users": label_users_positive.values,
        }
    )
    label_summary["prevalence_when_observed"] = (
        label_summary["positive_examples"] / label_summary["observed_examples"]
    )
    label_summary["observed_fraction"] = label_summary["observed_examples"] / total_rows
    label_summary = label_summary.sort_values(
        ["positive_users", "positive_examples"], ascending=False
    )

    modality_summary = pd.DataFrame(
        [
            {
                "modality": modality,
                "features": len(columns),
                "rows_available": modality_rows_available[modality],
                "row_coverage": modality_rows_available[modality] / total_rows,
                "cell_coverage": (
                    modality_cell_observed[modality] / modality_cell_total[modality]
                    if modality_cell_total[modality]
                    else np.nan
                ),
            }
            for modality, columns in modalities.items()
        ]
    ).sort_values("row_coverage", ascending=False)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    label_summary.to_csv(args.output_dir / "labels.csv", index=False)
    modality_summary.to_csv(args.output_dir / "modalities.csv", index=False)
    pd.DataFrame(user_rows).to_csv(args.output_dir / "users.csv", index=False)

    metadata = {
        "files": len(files),
        "rows": total_rows,
        "features": sum(len(columns) for columns in modalities.values()),
        "labels": len(labels),
        "unassigned_columns": sorted(
            set(first_columns)
            - {"timestamp", "label_source", *labels}
            - {column for columns in modalities.values() for column in columns}
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print(json.dumps(metadata, indent=2))
    print("\nModality coverage:")
    print(modality_summary.to_string(index=False))
    print("\nTop labels by number of participating users:")
    print(label_summary.head(20).to_string(index=False))


if __name__ == "__main__":
    main()

