"""Constants used by the ChaseSQL algorithm."""

import os

from typing import Any

 

import immutabledict

 

chase_sql_constants_dict: immutabledict.immutabledict[str, Any] = immutabledict.immutabledict(

    {

        "transpile_to_bigquery": True,

        "process_input_errors": True,

        "process_tool_output_errors": True,

        "number_of_candidates": 1,

        "model": os.getenv("CHASE_NL2SQL_MODEL"),

        "temperature": 0.5,

        "generate_sql_type": "dc",

    }

)
