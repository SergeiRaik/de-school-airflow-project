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


def generate_molecules(ti, params):
    files = ti.xcom_pull(task_ids="validate_input_files")

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
    dataset_id = params["dataset_id"]
    output_key = f"bronze/{dataset_id}_molecules.csv"
    s3.load_string(
        string_data=molecules_df.to_csv(index=False),
        key=output_key,
        bucket_name=BUCKET_NAME,
        replace=True,
    )

    return output_key


def calculate_properties(ti, params):
    files = ti.xcom_pull(task_ids="generate_molecules")

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
    dataset_id = params["dataset_id"]
    output_key = f"silver/{dataset_id}_molecules_with_properties.csv"
    s3.load_string(
        string_data=molecules_df.to_csv(index=False),
        key=output_key,
        bucket_name=BUCKET_NAME,
        replace=True,
    )

    return output_key