"""BigQuery client, schema, and validation utilities."""

import datetime

import logging

import os

 

from google.adk.tools import ToolContext

from google.cloud import bigquery

 

from .constants import chase_sql_constants_dict

 

MAX_NUM_ROWS = 1000

database_settings = None

 

 

def get_bq_client() -> bigquery.Client:

    """Get BigQuery client."""

    return bigquery.Client(project=os.getenv("BQ_PROJECT_ID"))

 

 

def get_bigquery_schema(dataset_id: str, client=None, project_id: str = None) -> str:

    """Retrieves schema and generates DDL with example values for a BigQuery dataset."""

    if client is None:

        client = bigquery.Client(project=project_id)

 

    dataset_ref = bigquery.DatasetReference(project_id, dataset_id)

    ddl_statements = ""

 

    for table in client.list_tables(dataset_ref):

        table_ref = dataset_ref.table(table.table_id)

        table_obj = client.get_table(table_ref)

 

        if table_obj.table_type != "TABLE":

            continue

 

        ddl_statement = f"CREATE OR REPLACE TABLE `{table_ref}` (\n"

        for field in table_obj.schema:

            ddl_statement += f"  `{field.name}` {field.field_type}"

            if field.mode == "REPEATED":

                ddl_statement += " ARRAY"

            if field.description:

                ddl_statement += f" COMMENT '{field.description}'"

            ddl_statement += ",\n"

        ddl_statement = ddl_statement[:-2] + "\n);\n\n"

 

        rows = client.list_rows(table_ref, max_results=5).to_dataframe()

        if not rows.empty:

            ddl_statement += f"-- Example values for table `{table_ref}`:\n"

            for _, row in rows.iterrows():

                ddl_statement += f"INSERT INTO `{table_ref}` VALUES\n"

                example_row_str = "("

                for value in row.values:

                    if isinstance(value, str):

                        example_row_str += f"'{value}',"

                    elif value is None:

                        example_row_str += "NULL,"

                    else:

                        example_row_str += f"{value},"

                example_row_str = example_row_str[:-1] + ");\n\n"

                ddl_statement += example_row_str

 

        ddl_statements += ddl_statement

 

    return ddl_statements

 

 

def get_bq_database_settings() -> dict:

    """Get (or build) database settings."""

    global database_settings

    if database_settings is None:

        database_settings = update_database_settings()

    return database_settings

 

 

def update_database_settings() -> dict:

    """Update database settings from environment and BigQuery schema."""

    global database_settings

    ddl_schema = get_bigquery_schema(

        os.getenv("BQ_DATASET_ID"),

        client=get_bq_client(),

        project_id=os.getenv("BQ_PROJECT_ID"),

    )

    database_settings = {

        "bq_project_id": os.getenv("BQ_PROJECT_ID"),

        "bq_dataset_id": os.getenv("BQ_DATASET_ID"),

        "bq_ddl_schema": ddl_schema,

        **chase_sql_constants_dict,

    }

    return database_settings

 

 

def run_bigquery_validation(sql_string: str, tool_context: ToolContext) -> dict:

    """Validates BigQuery SQL syntax and executes the query."""

 

    def cleanup_sql(sql: str) -> str:

        sql = sql.replace('\\"', '"')

        sql = sql.replace("\\\n", "\n")

        sql = sql.replace("\\'", "'")

        sql = sql.replace("\\n", "\n")

        if "limit" not in sql.lower():

            sql = sql + " limit " + str(MAX_NUM_ROWS)

        return sql

 

    logging.info("Validating SQL: %s", sql_string)

    sql_string = cleanup_sql(sql_string)

    logging.info("Validating SQL (after cleanup): %s", sql_string)

 

    final_result = {"query_result": None, "error_message": None}

 

    try:

        query_job = get_bq_client().query(sql_string)

        results = query_job.result()

        if results.schema:

            rows = [

                {

                    key: (value if not isinstance(value, datetime.date) else value.strftime("%Y-%m-%d"))

                    for key, value in row.items()

                }

                for row in results

            ][:MAX_NUM_ROWS]

            final_result["query_result"] = rows

            tool_context.state["query_result"] = rows

        else:

            final_result["error_message"] = "Valid SQL. Query executed successfully (no results)."

    except Exception as e:

        final_result["error_message"] = f"Invalid SQL: {e}"

 

    print("\n run_bigquery_validation final_result: \n", final_result)

    return final_result
