from airflow.providers.standard.operators.empty import EmptyOperator
from airflow.providers.standard.operators.python import PythonOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.exceptions import AirflowException

from airflow.sdk import DAG, get_current_context

from rdkit import Chem
import pandas as pd
import io
import re

BUCKET_NAME = 'molecules'

def check_input_files():
    context = get_current_context()
    dataset_id = context["params"]["dataset_id"]

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


def validate_files():
    context = get_current_context()

    files = context["ti"].xcom_pull(task_ids="check_input_files")

    scaffold_key = files["scaffold_key"]
    rgroup_key = files["rgroup_key"]

    s3 = S3Hook(aws_conn_id="aws_s3")

    bronze_keys = {}

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

        # 5. Upload to bronze
        bronze_key = input_key.replace("raw/", "bronze/", 1)

        s3.load_string(
            string_data=df.to_csv(index=False),
            key=bronze_key,
            bucket_name=BUCKET_NAME,
            replace=True,
        )

        if "scaffold" in bronze_key:
            bronze_keys["scaffold_key"] = bronze_key
        else:
            bronze_keys["rgroups_key"] = bronze_key

    return bronze_keys


def normalize_attachment(smiles: str) -> str:
    """
    Convert '*' attachment points to RDKit atom map notation.
    """

    if "[*:1]" in smiles:
        return smiles

    return smiles.replace("*", "[*:1]")


def validate_smiles(smiles: str):
    normalized = normalize_attachment(smiles)

    mol = Chem.MolFromSmiles(normalized)

    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    return normalized



def do_nothing():
    pass

with DAG(
    dag_id='generate_molecules',
    params={
        'dataset_id': "id001",
    }
) as dag:
    start_op = EmptyOperator(
        task_id='start',
    )

    check_input_files_op = PythonOperator(
        task_id='check_input_files',
        python_callable=check_input_files,
    )


    validate_input_files_op = PythonOperator(
        task_id='validate_input_files',
        python_callable=validate_files,
    )

    generate_molecules_op = EmptyOperator(
        task_id='generate_molecules',
    )

    upload_output = EmptyOperator(
        task_id='upload_output',
    )

    finish_op = EmptyOperator(
        task_id='finish',
    )

    start_op >> check_input_files_op >> validate_input_files_op >> generate_molecules_op >> upload_output >> finish_op