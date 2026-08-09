"""Tool functions for the NL2SQL and synthetic data agents."""

import enum

import json

import os

 

import numpy as np

import pandas as pd

from google.adk.tools import ToolContext

 

from .bigquery_utils import run_bigquery_validation

from .llm_utils import GeminiModel

from .prompts import DC_PROMPT_TEMPLATE, QP_PROMPT_TEMPLATE

from .sql_translator import SqlTranslator

 

BQ_PROJECT_ID = os.getenv("BQ_PROJECT_ID")

 

 

class GenerateSQLType(enum.Enum):

    DC = "dc"

    QP = "qp"

 

 

def parse_response(response: str) -> str:

    """Extracts SQL content from the model response."""

    query = response

    try:

        if "```sql" in response and "```" in response:

            query = response.split("```sql")[1].split("```")[0]

    except ValueError as e:

        print(f"Error in parsing response: {e}")

        query = response

    return query.strip()

 

 

def initial_bq_nl2sql(question: str, tool_context: ToolContext) -> str:

    """Generates an initial SQL query from a natural language question."""

    print("****** Running agent with ChaseSQL algorithm.")

    settings = tool_context.state["database_settings"]

    ddl_schema = settings["bq_ddl_schema"]

    project = settings["bq_project_id"]

    db = settings["bq_dataset_id"]

    transpile_to_bigquery = settings["transpile_to_bigquery"]

    process_input_errors = settings["process_input_errors"]

    process_tool_output_errors = settings["process_tool_output_errors"]

    number_of_candidates = settings["number_of_candidates"]

    model_name = settings["model"]

    temperature = settings["temperature"]

    generate_sql_type = settings["generate_sql_type"]

 

    if generate_sql_type == GenerateSQLType.DC.value:

        prompt = DC_PROMPT_TEMPLATE.format(SCHEMA=ddl_schema, QUESTION=question, BQ_PROJECT_ID=BQ_PROJECT_ID)

    elif generate_sql_type == GenerateSQLType.QP.value:

        prompt = QP_PROMPT_TEMPLATE.format(SCHEMA=ddl_schema, QUESTION=question, BQ_PROJECT_ID=BQ_PROJECT_ID)

    else:

        raise ValueError(f"Unsupported generate_sql_type: {generate_sql_type}")

 

    model = GeminiModel(model_name=model_name, temperature=temperature)

    requests = [prompt for _ in range(number_of_candidates)]

    responses = model.call_parallel(requests, parser_func=parse_response)

    responses = responses[0]

 

    if transpile_to_bigquery:

        translator = SqlTranslator(

            model=model, temperature=temperature,

            process_input_errors=process_input_errors,

            process_tool_output_errors=process_tool_output_errors,

        )

        responses = translator.translate(responses, ddl_schema=ddl_schema, db=db, catalog=project)

 

    return responses

 

 

def save_to_csv(tool_context: ToolContext) -> dict:

    """Saves query results to a CSV file in the Downloads folder."""

    df = tool_context.state.get("query_result")

    if df is None:

        return {"error": "No synthetic dataframe found to save."}

 

    df = pd.DataFrame(df)

    output_dir = r"C:\Users\SG0705226\Downloads"

    os.makedirs(output_dir, exist_ok=True)

    file_path = os.path.join(output_dir, "synthetic_output.csv")

 

    try:

        df.to_csv(file_path, index=False)

        return {"message": f"Data successfully saved to {file_path}"}

    except Exception as e:

        return {"error": f"Failed to write CSV: {str(e)}"}

 

 

def batch_inferencing(num_records: int) -> None:

    """Runs batch inference using Vertex AI and decodes predictions."""

    from google.cloud import aiplatform, bigquery, storage

    import psutil

    from scipy.stats import norm

 

    client = storage.Client(project="sab-dev-dap-aimlpipeline-4474")

    bucket = client.bucket("gan_test_1")

 

    # --- Generate and upload noise ---

    noise = np.random.normal(0, 1, (num_records, 128))

    instances = [{"keras_tensor": n} for n in noise.tolist()]

    json_lines = '\n'.join(json.dumps(r) for r in instances) + '\n'

    bucket.blob("output.jsonl").upload_from_string(json_lines, content_type='application/json')

    print("Finished writing noise file")

 

    # --- Vertex AI Batch Prediction ---

    aiplatform.init(project="sab-dev-dap-aimlpipeline-4474", location="us-central1")

    my_model = aiplatform.Model("6548118409475784704")

    batch_job = my_model.batch_predict(

        job_display_name="test_job",

        gcs_source="gs://gan_test_1/output.jsonl",

        gcs_destination_prefix="gs://gan_test_1/batch_output1/",

        instances_format="jsonl",

        machine_type="n1-standard-2",

        starting_replica_count=1,

        max_replica_count=1,

        sync=True,

    )

    batch_job.wait()

    print(batch_job.display_name, batch_job.resource_name, batch_job.state)

 

    # --- Download predictions ---

    predictions_list = []

    for blob in client.list_blobs("gan_test_1", prefix="batch_output1/"):

        if blob.name.endswith("/"):

            continue

        for line in blob.download_as_text().splitlines():

            if not line.strip():

                continue

            try:

                record = json.loads(line)

                if "prediction" in record:

                    predictions_list.append(record["prediction"])

            except json.JSONDecodeError as e:

                print(f"Skipping bad line: {e}")

 

    pd.DataFrame(predictions_list).to_csv("predictions.csv", index=False)

 

    # --- Fetch original data for transformer fitting ---

    bq_client = bigquery.Client()

    query = """

    SELECT pnr_locator_id, pnr_locator_id_hash, pnr_sequence, pnr_create_ts, from_ts, transmission_ts,

           pnr_create_dt, t2.number_in_party, number_of_infant, source_system_id, load_ts,

           number_of_air_segment, ingest_ts, tty_airline_cd, oac_accounting_cd, pcc, year, month, day,

           t1.*, t3.*, t4.* EXCEPT (ancillary_travel_portion), t5.*

    FROM `sab-dev-datahub-g3-5746.retailing_intelligence_iq.riq_pnr_masked` AS t2,

    UNNEST(passenger) AS t1,

    UNNEST(air_segment) AS t3,

    UNNEST(ancillary_services) AS t4,

    UNNEST(t4.ancillary_travel_portion) AS t5

    WHERE t2.year='2026' LIMIT 30000

    """

    original_data = bq_client.query(query).to_dataframe()

    original_data.drop(columns=['segment_booked_time', 'service_start_time', 'service_end_time'], inplace=True, errors='ignore')

 

    date_features = [

        'service_end_dt', 'service_start_dt', 'departure_ts', 'departure_datetime', 'arrival_ts',

        'segment_booked_ts', 'segment_booked_dt', 'departure_at_airport_ts', 'arrival_datetime',

        'purchase_ts', 'purchase_datetime', 'tvp_departure_dt', 'departure_dt', 'ingest_ts',

        'load_ts', 'pnr_create_dt', 'transmission_ts', 'from_ts', 'pnr_create_ts',

    ]

    high_null = [col for col in original_data.columns if original_data[col].isnull().mean() > 0.7]

    categorical_features = [col for col in original_data.columns if col not in date_features and col not in high_null]

    datetime_features = [col for col in date_features if col in original_data.columns and col not in high_null]

 

    # --- Transformers (defined inline for self-containment) ---

    from .transformers import CategoricalTransformer, DatetimeTransformer, decode_synthetic

 

    dt_transformers = {col: DatetimeTransformer() for col in datetime_features}

    for col, tr in dt_transformers.items():

        tr.fit(original_data[col])

 

    cat_transformers = {col: CategoricalTransformer() for col in categorical_features}

    for col, tr in cat_transformers.items():

        tr.fit(original_data[col])

 

    df_decoded = decode_synthetic(predictions_list, len(categorical_features), cat_transformers, categorical_features, dt_transformers, datetime_features)

    df_decoded.to_csv("decoded_predictions.csv", index=False)

 

    bucket.blob("synthetic_data.csv").upload_from_filename("decoded_predictions.csv")

    print("Synthetic data uploaded to GCS.")
