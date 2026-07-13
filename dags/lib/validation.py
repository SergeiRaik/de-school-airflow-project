import re
import logging

import pandas as pd
from rdkit import Chem
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.exceptions import AirflowException
import pandera.pandas as pa
from pandera import Check

from .constants import BUCKET_NAME



def check_input_files(params):
    overwrite = params.get("overwrite", False)

    s3 = S3Hook(aws_conn_id="aws_s3")

    raw_files = s3.list_keys(
        bucket_name=BUCKET_NAME,
        prefix="raw/",
    ) or []

    scaffolds = {
        key.replace("raw/", "").replace("_scaffolds.csv", "")
        for key in raw_files
        if key.endswith("_scaffolds.csv")
    }

    rgroups = {
        key.replace("raw/", "").replace("_r_groups.csv", "")
        for key in raw_files
        if key.endswith("_r_groups.csv")
    }

    dataset_ids = scaffolds & rgroups

    logging.info("overwrite = %s", overwrite)
    logging.info("raw_files = %s", raw_files)
    logging.info("scaffolds = %s", scaffolds)
    logging.info("rgroups = %s", rgroups)
    logging.info("dataset_ids = %s", dataset_ids)

    datasets = []

    for dataset_id in sorted(dataset_ids):

        output_key = f"processed/{dataset_id}_molecules.csv"

        if (
            not overwrite
            and s3.check_for_key(
                key=output_key,
                bucket_name=BUCKET_NAME,
            )
        ):
            continue

        datasets.append(
            {
                "dataset_id": dataset_id,
                "scaffold_key": f"raw/{dataset_id}_scaffolds.csv",
                "rgroup_key": f"raw/{dataset_id}_r_groups.csv",
            }
        )

    if not datasets:
        raise AirflowException("No new datasets to process")

    return datasets

def normalize(smiles: str) -> str:
    # Replace "*" that is NOT already [*:number]
    return re.sub(r"(?<!\[)\*(?!:\d+\])", "[*:1]", smiles)


def validate_and_upload_file(s3, input_key):
    df = pd.read_csv(
        s3.get_key(key=input_key, bucket_name=BUCKET_NAME).get()["Body"]
    )

    if "smiles" not in df.columns:
        raise ValueError(f"{input_key}: missing required column 'smiles'")

    smiles = df["smiles"].dropna().astype(str)

    if smiles.empty:
        raise ValueError(f"{input_key}: file contains no SMILES")

    df["smiles"] = smiles.map(normalize)

    invalid = [
        s for s in df["smiles"]
        if Chem.MolFromSmiles(s) is None
    ]

    if invalid:
        raise ValueError(
            f"{input_key}: invalid SMILES found: {invalid[:5]}"
        )

    output_key = input_key.replace("raw/", "processed/", 1)

    s3.load_string(
        string_data=df.to_csv(index=False),
        key=output_key,
        bucket_name=BUCKET_NAME,
        replace=True,
    )

    return output_key


def validate_files(ti):
    datasets = ti.xcom_pull(task_ids="check_input_files")

    s3 = S3Hook(aws_conn_id="aws_s3")
    results = []

    for files in datasets:
        scaffold_key = validate_and_upload_file(
            s3,
            files["scaffold_key"],
        )

        rgroup_key = validate_and_upload_file(
            s3,
            files["rgroup_key"],
        )

        results.append({
            "dataset_id": files["dataset_id"],
            "scaffold_key": scaffold_key,
            "rgroups_key": rgroup_key,
        })

    return results


MOLECULE_SCHEMA = pa.DataFrameSchema(
    {
        "mw": pa.Column(float, Check.gt(0), nullable=False),
        "tpsa": pa.Column(float, Check.ge(0), nullable=False),
        "h_acceptors": pa.Column(int, Check.ge(0), nullable=False),
        "h_donors": pa.Column(int, Check.ge(0), nullable=False),
        "ring_count": pa.Column(int, Check.ge(0), nullable=False),
    },
    checks=Check(lambda df: ~df.isnull().values.any(), error="Dataset contains null values"),
    coerce=True,
    strict=False,
)


def validate_molecule_dataset(s3, key):
    df = pd.read_csv(
        s3.get_key(
            key=key,
            bucket_name=BUCKET_NAME,
        ).get()["Body"]
    )

    if df.isnull().values.any():
        raise ValueError(f"{key}: dataset contains null values")

    MOLECULE_SCHEMA.validate(df)

    return key


def validate_calculated_properties(ti):
    keys = ti.xcom_pull(task_ids="calculate_properties")

    s3 = S3Hook(aws_conn_id="aws_s3")

    for key in keys:
        validate_molecule_dataset(s3, key)

    return keys