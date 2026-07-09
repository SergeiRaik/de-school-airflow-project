import re

import pandas as pd
from rdkit import Chem
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.exceptions import AirflowException

from .constants import BUCKET_NAME



def check_input_files(params):
    dataset_id = params["dataset_id"]

    scaffold_key = f"raw/{dataset_id}_scaffolds.csv"
    rgroup_key = f"raw/{dataset_id}_r_groups.csv"

    s3 = S3Hook(aws_conn_id="aws_s3")

    missing = []

    if not s3.check_for_key(key=scaffold_key, bucket_name=BUCKET_NAME):
        missing.append(scaffold_key)

    if not s3.check_for_key(key=rgroup_key, bucket_name=BUCKET_NAME):
        missing.append(rgroup_key)

    if missing:
        raise AirflowException(
            f"Missing input file(s): {', '.join(missing)}"
        )

    return {
        "scaffold_key": scaffold_key,
        "rgroup_key": rgroup_key,
    }


def validate_files(ti):
    files = ti.xcom_pull(task_ids="check_input_files")

    scaffold_key = files["scaffold_key"]
    rgroup_key = files["rgroup_key"]

    s3 = S3Hook(aws_conn_id="aws_s3")

    output_keys = {}

    for input_key in (scaffold_key, rgroup_key):

        # Read CSV from S3
        df = pd.read_csv(
            s3.get_key(key=input_key, bucket_name=BUCKET_NAME).get()["Body"]
        )

        # 1. Check column exists
        if "smiles" not in df.columns:
            raise ValueError(f"{input_key}: missing required column 'smiles'")

        # 2. Check at least one value
        smiles = df["smiles"].dropna().astype(str)

        if smiles.empty:
            raise ValueError(f"{input_key}: file contains no SMILES")

        # 3. Normalize attachment points
        def normalize(smiles: str) -> str:
            # Replace "*" that is NOT already [*:number]
            return re.sub(r"(?<!\[)\*(?!:\d+\])", "[*:1]", smiles)

        df["smiles"] = smiles.map(normalize)

        # 4. Validate SMILES
        invalid = [
            s for s in df["smiles"]
            if Chem.MolFromSmiles(s) is None
        ]

        if invalid:
            raise ValueError(
                f"{input_key}: invalid SMILES found: {invalid[:5]}"
            )

        # 5. Upload to processed
        output_key = input_key.replace("raw/", "processed/", 1)

        s3.load_string(
            string_data=df.to_csv(index=False),
            key=output_key,
            bucket_name=BUCKET_NAME,
            replace=True,
        )

        if "scaffold" in output_key:
            output_keys["scaffold_key"] = output_key
        else:
            output_keys["rgroups_key"] = output_key

    return output_keys