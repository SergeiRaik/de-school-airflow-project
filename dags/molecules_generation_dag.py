from airflow.providers.standard.operators.empty import EmptyOperator
from airflow.providers.standard.operators.python import PythonOperator
from airflow.sdk import DAG

from lib.validation import check_input_files, validate_files
from lib.processing import generate_molecules, calculate_properties


with DAG(
    dag_id='generate_molecules',
    schedule="@weekly",
    params={
        'overwrite': False,
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