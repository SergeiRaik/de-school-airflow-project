import io
import re

import numpy as np
import pandas as pd
from airflow.exceptions import AirflowException
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.providers.standard.operators.empty import EmptyOperator
from airflow.providers.standard.operators.python import PythonOperator
from airflow.sdk import DAG, get_current_context
from rdkit import Chem
from rdkit.DataStructs import ConvertToNumpyArray
from sklearn.cluster import KMeans

from lib.fingerprints import get_fingerprint_function
from lib.properties_calculation import PropertiesCalculator
from lib.validation import check_input_files, validate_files


BUCKET_NAME = 'molecules'

# def check_input_files():
#     context = get_current_context()
#     dataset_id = context["params"]["dataset_id"]

#     scaffold_key = f"raw/{dataset_id}_scaffolds.csv"
#     rgroup_key = f"raw/{dataset_id}_r_groups.csv"

#     s3 = S3Hook(aws_conn_id="aws_s3")

#     missing = []

#     if not s3.check_for_key(key=scaffold_key, bucket_name=BUCKET_NAME):
#         missing.append(scaffold_key)

#     if not s3.check_for_key(key=rgroup_key, bucket_name=BUCKET_NAME):
#         missing.append(rgroup_key)

#     if missing:
#         raise AirflowException(
#             f"Missing input file(s): {', '.join(missing)}"
#         )

#     return {
#         "scaffold_key": scaffold_key,
#         "rgroup_key": rgroup_key,
#     }


# def validate_files():
#     context = get_current_context()

#     files = context["ti"].xcom_pull(task_ids="check_input_files")

#     scaffold_key = files["scaffold_key"]
#     rgroup_key = files["rgroup_key"]

#     s3 = S3Hook(aws_conn_id="aws_s3")

#     bronze_keys = {}

#     for input_key in (scaffold_key, rgroup_key):

#         # Read CSV from S3
#         df = pd.read_csv(
#             s3.get_key(key=input_key, bucket_name=BUCKET_NAME).get()["Body"]
#         )

#         # 1. Check column exists
#         if "smiles" not in df.columns:
#             raise ValueError(f"{input_key}: missing required column 'smiles'")

#         # 2. Check at least one value
#         smiles = df["smiles"].dropna().astype(str)

#         if smiles.empty:
#             raise ValueError(f"{input_key}: file contains no SMILES")

#         # 3. Normalize attachment points
#         def normalize(smiles: str) -> str:
#             # Replace "*" that is NOT already [*:number]
#             return re.sub(r"(?<!\[)\*(?!:\d+\])", "[*:1]", smiles)

#         df["smiles"] = smiles.map(normalize)

#         # 4. Validate SMILES
#         invalid = [
#             s for s in df["smiles"]
#             if Chem.MolFromSmiles(s) is None
#         ]

#         if invalid:
#             raise ValueError(
#                 f"{input_key}: invalid SMILES found: {invalid[:5]}"
#             )

#         # 5. Upload to bronze
#         bronze_key = input_key.replace("raw/", "bronze/", 1)

#         s3.load_string(
#             string_data=df.to_csv(index=False),
#             key=bronze_key,
#             bucket_name=BUCKET_NAME,
#             replace=True,
#         )

#         if "scaffold" in bronze_key:
#             bronze_keys["scaffold_key"] = bronze_key
#         else:
#             bronze_keys["rgroups_key"] = bronze_key

#     return bronze_keys


def merge_molecule_blocks(scaffold, r_group):
   combined_smiles = scaffold + "." + r_group
   mol = Chem.MolFromSmiles(combined_smiles)
   prod = Chem.molzip(mol)
   return Chem.MolToSmiles(prod)


def generate_molecules():
    context = get_current_context()

    files = context["ti"].xcom_pull(task_ids="validate_input_files")

    scaffold_key = files["scaffold_key"]
    rgroup_key = files["rgroups_key"]

    s3 = S3Hook(aws_conn_id="aws_s3")

    # Read CSVs from S3
    scaffold_df = pd.read_csv(
        s3.get_key(key=scaffold_key, bucket_name=BUCKET_NAME).get()["Body"],
        skiprows=1,
        names=["scaffold"],
    )

    rgroup_df = pd.read_csv(
        s3.get_key(key=rgroup_key, bucket_name=BUCKET_NAME).get()["Body"],
        skiprows=1,
        names=["rgroup"],
    )

    molecules_df = scaffold_df.merge(rgroup_df, how="cross")

    molecules_df["molecule"] = molecules_df.apply(
        lambda row: merge_molecule_blocks(row['scaffold'], row['rgroup']), axis=1
    )

    # Upload to bronze
    dataset_id = context["params"]["dataset_id"]
    output_key = f"bronze/{dataset_id}_molecules.csv"
    s3.load_string(
        string_data=molecules_df.to_csv(index=False),
        key=output_key,
        bucket_name=BUCKET_NAME,
        replace=True,
    )

    return output_key


def calculate_properties():
    context = get_current_context()

    files = context["ti"].xcom_pull(task_ids="generate_molecules")

    molecules_key = files

    s3 = S3Hook(aws_conn_id="aws_s3")

    # Read CSV from S3
    molecules_df = pd.read_csv(
        s3.get_key(key=molecules_key, bucket_name=BUCKET_NAME).get()["Body"]
    )

    calculator = PropertiesCalculator()
    molecules_df["mol"] = molecules_df["molecule"].apply(Chem.MolFromSmiles)
    molecules_df = calculator.calculate_properties(molecules_df)

    fingerprint_fn = get_fingerprint_function("ECFP4")
    molecules_df["fingerprint"] = molecules_df["mol"].apply(fingerprint_fn)
    def fp_to_array(fp):
        arr = np.zeros((2048,), dtype=np.int8)
        ConvertToNumpyArray(fp, arr)
        return arr

    molecules_df["fingerprint_array"] = molecules_df["fingerprint"].apply(fp_to_array)
    X = np.array([
        fp_to_array(fp)
        for fp in molecules_df["fingerprint"]
    ])

    model = KMeans(
        n_clusters=10,
        random_state=42,
        n_init="auto",
    )

    molecules_df["cluster"] = model.fit_predict(X)

    # Upload to silver
    dataset_id = context["params"]["dataset_id"]
    output_key = f"silver/{dataset_id}_molecules_with_properties.csv"
    s3.load_string(
        string_data=molecules_df.to_csv(index=False),
        key=output_key,
        bucket_name=BUCKET_NAME,
        replace=True,
    )

    return output_key


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

    generate_molecules_op = PythonOperator(
        task_id='generate_molecules',
        python_callable=generate_molecules,
    )

    calculate_properties_op = PythonOperator(
        task_id='calculate_properties',
        python_callable=calculate_properties,
    )

    finish_op = EmptyOperator(
        task_id='finish',
    )

    start_op >> check_input_files_op >> validate_input_files_op >> generate_molecules_op >> calculate_properties_op >> finish_op