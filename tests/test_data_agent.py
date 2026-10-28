
import pandas as pd
import pytest

from agents.data_agent.agent import (
    analyze_dataset,
    calculate_sha256,
    process_csv,
)


def test_analyze_dataset_detects_quality_issues():
    df = pd.DataFrame(
        {
            "customer": ["A", "B", None, "C", "D", "E", "B"],
            "spend": [10, 11, 12, 13, 14, 1000, 11],
        }
    )

    report = analyze_dataset(df)

    assert report["rows"] == 7
    assert report["duplicate_rows"] == 1
    assert report["missing_values"] == {"customer": 1}
    assert report["numeric_outliers"]["spend"]["count"] == 1


def test_required_columns_are_checked():
    df = pd.DataFrame({"customer": ["A"], "spend": [10]})

    with pytest.raises(ValueError, match="Required columns are missing"):
        analyze_dataset(df, required_columns=["customer", "age"])


def test_process_csv_removes_duplicates_and_saves_hash(tmp_path):
    input_path = tmp_path / "raw.csv"
    output_path = tmp_path / "processed" / "clean.csv"

    df = pd.DataFrame(
        {
            "customer": ["A", "B", "B", "C"],
            "spend": [10, 20, 20, 30],
        }
    )
    df.to_csv(input_path, index=False)

    report = process_csv(input_path, output_path)
    cleaned_df = pd.read_csv(output_path)

    assert report["status"] == "success"
    assert report["cleaning"]["duplicates_removed"] == 1
    assert len(cleaned_df) == 3
    assert len(report["sha256"]) == 64
    assert report["sha256"] == calculate_sha256(output_path)
    assert output_path.exists()


def test_missing_input_file_raises_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        process_csv(
            tmp_path / "does_not_exist.csv",
            tmp_path / "clean.csv",
        )
