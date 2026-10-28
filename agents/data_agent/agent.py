
"""First version of the MLOps Data Agent."""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def analyze_dataset(
    df: pd.DataFrame,
    required_columns: list[str] | None = None,
) -> dict:
    """Analyze missing values, duplicate rows and numeric outliers."""

    if df.empty:
        raise ValueError("The dataset is empty.")

    if required_columns:
        missing_columns = sorted(set(required_columns) - set(df.columns))
        if missing_columns:
            raise ValueError(
                f"Required columns are missing: {missing_columns}"
            )

    missing_values = {
        column: int(count)
        for column, count in df.isna().sum().items()
        if count > 0
    }

    numeric_outliers = {}

    for column in df.select_dtypes(include="number").columns:
        values = df[column].dropna()

        if len(values) < 4:
            continue

        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1

        # A constant column has no IQR-based outliers.
        if iqr == 0:
            continue

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        count = int(
            ((values < lower_bound) | (values > upper_bound)).sum()
        )

        if count:
            numeric_outliers[column] = {
                "count": count,
                "lower_bound": float(lower_bound),
                "upper_bound": float(upper_bound),
            }

    return {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "column_names": list(df.columns),
        "missing_values": missing_values,
        "duplicate_rows": int(df.duplicated().sum()),
        "numeric_outliers": numeric_outliers,
    }


def calculate_sha256(file_path: str | Path) -> str:
    """Calculate the SHA-256 fingerprint of a file."""

    digest = hashlib.sha256()

    with Path(file_path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def process_csv(
    input_path: str | Path,
    output_path: str | Path,
    required_columns: list[str] | None = None,
) -> dict:
    """Analyze a CSV, remove exact duplicates and save a report."""

    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)

    initial_report = analyze_dataset(
        df,
        required_columns=required_columns,
    )

    # Only remove exact duplicate rows in this first version.
    cleaned_df = df.drop_duplicates().copy()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(output_path, index=False)

    final_report = analyze_dataset(
        cleaned_df,
        required_columns=required_columns,
    )

    return {
        "status": "success",
        "input_file": str(input_path),
        "output_file": str(output_path),
        "initial_quality": initial_report,
        "cleaning": {
            "duplicates_removed": int(
                len(df) - len(cleaned_df)
            ),
            "rows_after_cleaning": int(len(cleaned_df)),
            "missing_values_after_cleaning": (
                final_report["missing_values"]
            ),
            "outliers_after_cleaning": (
                final_report["numeric_outliers"]
            ),
        },
        "sha256": calculate_sha256(output_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze and clean a CSV dataset."
    )
    parser.add_argument("input", help="Path to the raw CSV file")
    parser.add_argument("output", help="Path to the cleaned CSV file")
    args = parser.parse_args()

    report = process_csv(args.input, args.output)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
