"""Agent definitions and setup callbacks."""

import os

from datetime import date

 

from google.adk.agents import Agent

from google.adk.agents.callback_context import CallbackContext

from google.adk.tools import ToolContext

from google.adk.tools.agent_tool import AgentTool

from google.genai import types

 

from .bigquery_utils import get_bq_database_settings, run_bigquery_validation

from .prompts import return_instructions_bigquery, return_instructions_root

from .tools import initial_bq_nl2sql, save_to_csv, batch_inferencing

 

date_today = date.today()

 

 

def setup_before_database_agent_call(callback_context: CallbackContext) -> None:

    """Setup callback for the database sub-agent."""

    if "database_settings" not in callback_context.state:

        callback_context.state["database_settings"] = get_bq_database_settings()

 

 

database_agent = Agent(

    model=os.getenv("BIGQUERY_AGENT_MODEL"),

    name="database_agent",

    instruction=return_instructions_bigquery(),

    tools=[initial_bq_nl2sql, run_bigquery_validation],

    before_agent_callback=setup_before_database_agent_call,

    generate_content_config=types.GenerateContentConfig(temperature=0.01),

)

 

 

async def call_db_agent(question: str, tool_context: ToolContext):

    """Tool to call the database (nl2sql) sub-agent."""

    print(f"\n call_db_agent.use_database: {tool_context.state['all_db_settings']['use_database']}")

    agent_tool = AgentTool(agent=database_agent)

    db_agent_output = await agent_tool.run_async(args={"request": question}, tool_context=tool_context)

    tool_context.state["db_agent_output"] = db_agent_output

    return db_agent_output

 

 

def setup_before_root_agent_call(callback_context: CallbackContext) -> None:

    """Setup callback for the root agent."""

    if "database_settings" not in callback_context.state:

        db_settings = {"use_database": "BigQuery"}

        callback_context.state["all_db_settings"] = db_settings

 

    if callback_context.state["all_db_settings"]["use_database"] == "BigQuery":

        callback_context.state["database_settings"] = get_bq_database_settings()

        schema = callback_context.state["database_settings"]["bq_ddl_schema"]

        callback_context._invocation_context.agent.instruction = (

            return_instructions_root()

            + f"""

 

    --------- The BigQuery schema of the relevant data with a few sample rows. ---------

    {schema}

 

    """

        )

 

 

root_agent = Agent(

    model=os.getenv("ROOT_AGENT_MODEL"),

    name="db_ds_multiagent",

    instruction=return_instructions_root(),

    global_instruction=f"""

        You are a Data Science and Data Analytics Multi Agent System.

        Todays date: {date_today}

    """,

    tools=[save_to_csv, initial_bq_nl2sql, run_bigquery_validation, batch_inferencing],

    before_agent_callback=setup_before_root_agent_call,

    generate_content_config=types.GenerateContentConfig(temperature=0.01),

)
