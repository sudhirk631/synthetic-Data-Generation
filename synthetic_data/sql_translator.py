"""Translator from SQLite to BigQuery using SQLGlot."""

import re

from typing import Any, Final

 

import regex

import sqlglot

import sqlglot.optimizer

 

from .llm_utils import GeminiModel

from .prompts import CORRECTION_PROMPT_TEMPLATE_V1_0

 

ColumnSchemaType = tuple[str, str]

AllColumnsSchemaType = list[ColumnSchemaType]

TableSchemaType = tuple[str, AllColumnsSchemaType]

DDLSchemaType = list[TableSchemaType]

SQLGlotColumnsDictType = dict[str, str]

SQLGlotSchemaType = dict[str, Any]

BirdSampleType = dict[str, Any]

 

 

def _isinstance_list_of_str_tuples_lists(obj: Any) -> bool:

    return (

        isinstance(obj, list)

        and all([isinstance(v, (tuple, list)) for v in obj])

        and all([isinstance(v[0], str) and isinstance(v[1], str) for v in obj])

    )

 

 

def _isinstance_ddl_schema_type(obj: Any) -> bool:

    return (

        isinstance(obj, list)

        and all([isinstance(v, (tuple, list)) for v in obj])

        and all([isinstance(v[0], str) and isinstance(v[1], list) for v in obj])

        and all([_isinstance_list_of_str_tuples_lists(v[1]) for v in obj])

    )

 

 

def _isinstance_sqlglot_schema_type(obj: Any) -> bool:

    return (

        isinstance(obj, dict)

        and all([isinstance(v, dict) for v in obj.values()])

        and all([isinstance(c, str) for d in obj.values() for c, _ in d.items()])

        and all([isinstance(t, str) for d in obj.values() for _, t in d.items()])

    )

 

 

def _isinstance_bird_sample_type(obj: Any) -> bool:

    return isinstance(obj, dict) and not _isinstance_sqlglot_schema_type(obj)

 

 

class SqlTranslator:

    """Translator from SQLite to BigQuery."""

 

    INPUT_DIALECT: Final[str] = "sqlite"

    OUTPUT_DIALECT: Final[str] = "bigquery"

 

    def __init__(

            self,

            model: str | GeminiModel = "gemini-2.0-flash-001",

            temperature: float = 0.5,

            process_input_errors: bool = False,

            process_tool_output_errors: bool = False,

    ):

        self._process_input_errors = process_input_errors

        self._process_tool_output_errors = process_tool_output_errors

        self._input_errors = None

        self._tool_output_errors = None

        self._temperature = temperature

        if isinstance(model, str):

            self._model = GeminiModel(model_name=model, temperature=self._temperature)

        else:

            self._model = model

 

    @classmethod

    def _parse_response(cls, text: str) -> str | None:

        pattern = r"```sql(.*?)```"

        match = re.search(pattern, text, re.DOTALL)

        if match:

            return match.group(1).strip()

        return None

 

    @classmethod

    def _apply_heuristics(cls, sql_query: str) -> str:

        if "''" in sql_query:

            sql_query = sql_query.replace("''", "\\'")

        return sql_query

 

    @classmethod

    def _extract_schema_from_ddl_statement(cls, ddl_statement: str) -> TableSchemaType:

        splitter_pattern = (

            r"^\s*CREATE\s+(?:OR\s+REPLACE\s+)?TABLE\s+"

            r"(?:`)?(?P<table_name>[\w\d\-\_\.]+)(?:`)?\s*"

            r"\((?P<all_columns>.*)\);$"

        )

        split_match = regex.search(splitter_pattern, ddl_statement, flags=re.DOTALL | re.VERBOSE | re.MULTILINE)

        if not split_match:

            return None, None

 

        table_name = split_match.group("table_name")

        all_columns = split_match.group("all_columns").strip()

        if not table_name or not all_columns:

            return None, None

 

        column_pattern = (

            r"\s*--.*(*SKIP)(*FAIL)"

            r"|\s*INSERT\s+INTO.*(*SKIP)(*FAIL)"

            r"|\s*\(.*(*SKIP)(*FAIL)"

            r"|\s*(?:`)?\s*(?P<column_name>\w+)(?:`)?\s+(?P<column_type>\w+).*"

        )

        columns = regex.findall(column_pattern, all_columns, flags=re.VERBOSE)

        return table_name, columns

 

    @classmethod

    def extract_schema_from_ddls(cls, ddls: str) -> DDLSchemaType:

        ddl_statements = ddls.split(";\n")

        ddl_statements = [ddl.strip() for ddl in ddl_statements if ddl.strip()]

        schema = []

        for ddl_statement in ddl_statements:

            if ddl_statement:

                ddl_statement = ddl_statement.strip() + ";"

                table_name, columns = cls._extract_schema_from_ddl_statement(ddl_statement)

                if table_name and columns:

                    schema.append((table_name, columns))

        return schema

 

    @classmethod

    def _get_schema_from_bird_sample(cls, sample: BirdSampleType) -> dict[str, dict[str, str]]:

        col_types_map = {

            "text": "TEXT", "number": "FLOAT", "date": "DATE",

            "datetime": "DATETIME", "time": "TIME", "timestamp": "TIMESTAMP", "bool": "BOOL",

        }

        tables = sample["db_table_names"]

        table_ids = sample["db_column_names"]["table_id"][1:]

        column_names = sample["db_column_names"]["column_name"][1:]

        column_types = [col_types_map[t] for t in sample["db_column_types"][1:]]

        cols_and_types = list(zip(column_names, column_types))

        tables_to_columns: dict[str, dict[str, str]] = {}

        for id_pos, table_id in enumerate(table_ids):

            if tables[table_id] in tables_to_columns:

                tables_to_columns[tables[table_id]].update(dict([cols_and_types[id_pos]]))

            else:

                tables_to_columns[tables[table_id]] = dict([cols_and_types[id_pos]])

        return tables_to_columns

 

    @classmethod

    def _get_table_parts(cls, table_name: str) -> tuple[str | None, str | None, str]:

        table_parts = table_name.split(".")

        if len(table_parts) == 3:

            return table_parts

        elif len(table_parts) == 2:

            return None, *table_parts

        elif len(table_parts) == 1:

            return None, None, *table_parts

        else:

            raise ValueError(f"Invalid table name: {table_name}")

 

    @classmethod

    def format_schema(cls, schema: DDLSchemaType) -> SQLGlotSchemaType:

        schema_dict = {}

        catalog, db = None, None

        for table_name, columns in schema:

            catalog, db, table_name = cls._get_table_parts(table_name)

            schema_dict[table_name] = {col: typ for col, typ in columns}

        if db:

            schema_dict = {db: schema_dict}

        if catalog:

            schema_dict = {catalog: schema_dict}

        return schema_dict

 

    @classmethod

    def rewrite_schema_for_sqlglot(cls, schema) -> SQLGlotSchemaType:

        schema_dict = None

        if schema:

            if isinstance(schema, str):

                schema = cls.extract_schema_from_ddls(schema)

                schema_dict = cls.format_schema(schema)

            elif _isinstance_sqlglot_schema_type(schema):

                schema_dict = schema

            elif _isinstance_bird_sample_type(schema):

                schema_dict = cls._get_schema_from_bird_sample(schema)

            elif _isinstance_ddl_schema_type(schema):

                schema_dict = cls.format_schema(schema)

            else:

                raise TypeError(f"Unsupported schema type: {type(schema)}")

        return schema_dict

 

    @classmethod

    def _check_for_errors(cls, sql_query, sql_dialect, db=None, catalog=None, schema_dict=None):

        try:

            sql_query_ast = sqlglot.parse_one(

                sql=sql_query, read=sql_dialect.lower(),

                error_level=sqlglot.ErrorLevel.IMMEDIATE,

            )

            for table in sql_query_ast.find_all(sqlglot.exp.Table):

                table.set("catalog", sqlglot.exp.Identifier(this=catalog, quoted=True))

                table.set("db", sqlglot.exp.Identifier(this=db, quoted=True))

            sql_query_ast = sqlglot.optimizer.optimize(

                sql_query_ast, dialect=sql_dialect.lower(),

                schema=schema_dict, db=db, catalog=catalog,

                error_level=sqlglot.ErrorLevel.IMMEDIATE,

            )

            sql_query = sql_query_ast.sql(sql_dialect.lower())

        except sqlglot.errors.SqlglotError as e:

            return str(e), sql_query

        return None, sql_query

 

    def _fix_errors(self, sql_query, sql_dialect, apply_heuristics, db=None, catalog=None, ddl_schema=None, number_of_candidates=1):

        if apply_heuristics:

            sql_query = self._apply_heuristics(sql_query)

        schema_dict = self.rewrite_schema_for_sqlglot(ddl_schema)

        errors, sql_query = self._check_for_errors(

            sql_query=sql_query, sql_dialect=self.OUTPUT_DIALECT,

            db=db, catalog=catalog, schema_dict=schema_dict,

        )

        responses = sql_query

        if errors:

            print("Processing input errors")

            schema_insert = f"\nThe database schema is:\n{schema_dict}\n" if schema_dict else "\n"

            prompt = CORRECTION_PROMPT_TEMPLATE_V1_0.format(

                sql_dialect=sql_dialect.lower(), errors=errors,

                sql_query=sql_query, schema_insert=schema_insert,

            )

            requests = [prompt for _ in range(number_of_candidates)]

            responses = self._model.call_parallel(requests, parser_func=self._parse_response)

            if responses:

                responses = [r for r in responses if r is not None]

                if responses:

                    responses = responses[0]

        return responses

 

    def translate(self, sql_query, db=None, catalog=None, ddl_schema=None) -> str:

        print("****** sql_query at translator entry:", sql_query)

        if self._process_input_errors:

            sql_query = self._fix_errors(

                sql_query, db=db, catalog=catalog,

                sql_dialect=self.OUTPUT_DIALECT, ddl_schema=ddl_schema, apply_heuristics=True,

            )

        print("****** sql_query after fix_errors:", sql_query)

        sql_query = sqlglot.transpile(

            sql=sql_query, read=self.INPUT_DIALECT, write=self.OUTPUT_DIALECT,

            error_level=sqlglot.ErrorLevel.IMMEDIATE,

        )[0]

        print("****** sql_query after transpile:", sql_query)

        if self._tool_output_errors:

            sql_query = self._fix_errors(

                sql_query, db=db, catalog=catalog,

                sql_dialect=self.OUTPUT_DIALECT, ddl_schema=ddl_schema, apply_heuristics=True,

            )

        sql_query = sql_query.strip().replace('"', "`")

        sql_query = self._apply_heuristics(sql_query)

        return sql_query
