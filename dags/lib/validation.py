import re

import pandas as pd
from rdkit import Chem
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.exceptions import AirflowException

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