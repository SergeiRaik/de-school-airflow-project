from rdkit import Chem
from rdkit.DataStructs import ConvertToNumpyArray
from sklearn.cluster import KMeans
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
import pandas as pd
import numpy as np

from .properties_calculation import PropertiesCalculator
from .fingerprints import get_fingerprint_function
from .constants import BUCKET_NAME

def merge_molecule_blocks(scaffold, r_group):
   combined_smiles = scaffold + "." + r_group
   mol = Chem.MolFromSmiles(combined_smiles)
   prod = Chem.molzip(mol)
   return Chem.MolToSmiles(prod)


def generate_dataset(s3, files):
    scaffold_df = pd.read_csv(
        s3.get_key(
            key=files["scaffold_key"],
            bucket_name=BUCKET_NAME,
        ).get()["Body"],
        skiprows=1,
        names=["scaffold"],
    )

    rgroup_df = pd.read_csv(
        s3.get_key(
            key=files["rgroups_key"],
            bucket_name=BUCKET_NAME,
        ).get()["Body"],
        skiprows=1,
        names=["rgroup"],
    )

    molecules_df = scaffold_df.merge(rgroup_df, how="cross")

    molecules_df["molecule"] = molecules_df.apply(
        lambda row: merge_molecule_blocks(
            row["scaffold"],
            row["rgroup"],
        ),
        axis=1,
    )

    output_key = f'processed/{files["dataset_id"]}_molecules.csv'

    s3.load_string(
        string_data=molecules_df.to_csv(index=False),
        key=output_key,
        bucket_name=BUCKET_NAME,
        replace=True,
    )

    return {
        "dataset_id": files["dataset_id"],
        "molecules_key": output_key,
    }


def generate_molecules(ti):
    datasets = ti.xcom_pull(task_ids="validate_input_files")

    s3 = S3Hook(aws_conn_id="aws_s3")

    results = []

    for files in datasets:
        results.append(generate_dataset(s3, files))

    return results


def calculate_dataset_properties(s3, files):
    molecules_df = pd.read_csv(
        s3.get_key(
            key=files["molecules_key"],
            bucket_name=BUCKET_NAME,
        ).get()["Body"]
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

    X = np.array([fp_to_array(fp) for fp in molecules_df["fingerprint"]])

    model = KMeans(
        n_clusters=10,
        random_state=42,
        n_init="auto",
    )

    molecules_df["cluster"] = model.fit_predict(X)
    molecules_df = molecules_df.drop(columns=["mol", "fingerprint"])

    output_key = files["molecules_key"]

    s3.load_string(
        string_data=molecules_df.to_csv(index=False),
        key=output_key,
        bucket_name=BUCKET_NAME,
        replace=True,
    )

    return output_key


def calculate_properties(ti):
    datasets = ti.xcom_pull(task_ids="generate_molecules")

    s3 = S3Hook(aws_conn_id="aws_s3")

    results = []

    for files in datasets:
        results.append(calculate_dataset_properties(s3, files))

    return results