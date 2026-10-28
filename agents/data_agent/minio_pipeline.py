
"""Run the Data Agent with datasets stored in MinIO."""

import json
import os
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from minio import Minio

from agents.data_agent.agent import process_csv

load_dotenv()


def create_minio_client() -> Minio:
    """Create a MinIO client using local environment variables."""

    endpoint = os.getenv("MINIO_ENDPOINT", "localhost:9000")
    access_key = os.getenv("MINIO_ACCESS_KEY")
    secret_key = os.getenv("MINIO_SECRET_KEY")
    secure = os.getenv("MINIO_SECURE", "false").lower() == "true"

    if not access_key or not secret_key:
        raise ValueError("MinIO credentials are missing from .env")

    return Minio(
        endpoint,
        access_key=access_key,
        secret_key=secret_key,
        secure=secure,
    )


def run_minio_pipeline(
    local_csv: str = "data/raw/customers.csv",
    object_name: str = "customers.csv",
) -> dict:
    """Upload, process and store a CSV and its quality report."""

    client = create_minio_client()

    raw_bucket = os.getenv("MINIO_RAW_BUCKET", "mlops-raw")
    processed_bucket = os.getenv(
        "MINIO_PROCESSED_BUCKET", "mlops-processed"
    )
    metadata_bucket = os.getenv(
        "MINIO_METADATA_BUCKET", "mlops-metadata"
    )

    source = Path(local_csv)
    if not source.is_file():
        raise FileNotFoundError(f"Dataset not found: {source}")

    # Check connectivity and ensure the buckets exist.
    for bucket in (raw_bucket, processed_bucket, metadata_bucket):
        if not client.bucket_exists(bucket):
            raise RuntimeError(
                f"Bucket '{bucket}' does not exist. Create it in MinIO first."
            )

    # 1. Upload the original dataset.
    client.fput_object(
        raw_bucket,
        object_name,
        str(source),
        content_type="text/csv",
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        downloaded_csv = temp_path / "input.csv"
        cleaned_csv = temp_path / "customers_clean.csv"
        report_path = temp_path / "quality_report.json"

        # 2. Download the dataset from MinIO.
        response = client.get_object(raw_bucket, object_name)
        try:
            with downloaded_csv.open("wb") as output:
                for chunk in response.stream(amt=1024 * 1024):
                    output.write(chunk)
        finally:
            response.close()
            response.release_conn()

        # 3. Run the existing Data Agent.
        report = process_csv(downloaded_csv, cleaned_csv)

        # 4. Save the quality report as JSON.
        report["source_object"] = f"{raw_bucket}/{object_name}"
        report["processed_object"] = (
            f"{processed_bucket}/customers_clean.csv"
        )
        report["metadata_object"] = (
            f"{metadata_bucket}/quality_report.json"
        )

        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        # 5. Upload the cleaned dataset and quality report.
        client.fput_object(
            processed_bucket,
            "customers_clean.csv",
            str(cleaned_csv),
            content_type="text/csv",
        )
        client.fput_object(
            metadata_bucket,
            "quality_report.json",
            str(report_path),
            content_type="application/json",
        )

    return {
        "status": "success",
        "raw_object": f"{raw_bucket}/{object_name}",
        "processed_object": (
            f"{processed_bucket}/customers_clean.csv"
        ),
        "metadata_object": (
            f"{metadata_bucket}/quality_report.json"
        ),
        "rows_after_cleaning": report["cleaning"]["rows_after_cleaning"],
        "duplicates_removed": report["cleaning"]["duplicates_removed"],
        "sha256": report["sha256"],
    }


if __name__ == "__main__":
    result = run_minio_pipeline()
    print(json.dumps(result, indent=2, ensure_ascii=False))
