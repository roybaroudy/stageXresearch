import os
import re
import dotenv
from typing_extensions import TypedDict
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from langgraph.graph import StateGraph, START, END
import google.generativeai as genai

dotenv.load_dotenv()
API_KEY = os.getenv("API_KEY")

genai.configure(api_key=API_KEY)
gemini = genai.GenerativeModel("gemini-2.0-flash")

engine = create_engine("postgresql+psycopg2://postgres:admin@localhost/uae_real_estate_db")

class AgentState(TypedDict):
    input: str
    prompt: str
    sql: str
    result: str
    is_similar_query: bool

def prompt_classifier(state: AgentState) -> AgentState:
    user_input = state["input"].lower()
    state["is_similar_query"] = "similar" in user_input or "like" in user_input
    return state

def query_modifier(state: AgentState) -> AgentState:
    if not state["is_similar_query"]:
        state["prompt"] = f"""
        You are a text to SQL agent. Given a question, generate a SQL query.

        Schema:
        apartments("id" int, "title" text, "display_address" text, "bathrooms" bigint, 
                   "bedrooms" bigint, "added_on" text, "type" text, "rera" bigint, 
                   "price" int, "property_type" text)

        Question: {state['input']}
        """
        return state

    input_text = state["input"]
    match_id = re.search(r"\b(?:id|listing)\s*(\d+)", input_text)
    match_title = re.search(r"(?:called|named|like)\s+['\"](.+?)['\"]", input_text)

    query = None
    if match_id:
        query = f"SELECT * FROM apartments WHERE id = {match_id.group(1)}"
    elif match_title:
        query = f"SELECT * FROM apartments WHERE title ILIKE '%{match_title.group(1)}%'"

    if query:
        try:
            with engine.connect() as conn:
                result = conn.execute(text(query))
                apt = result.mappings().first()

            if apt:
                state["prompt"] = f"""
                Generate a SQL query to find apartments similar to one with:
                - Bedrooms: {apt['bedrooms']}
                - Bathrooms: {apt['bathrooms']}
                - Price: around {apt['price']}
                - Location: around '{apt['display_address']}'

                the id of the reference apartment is {apt['id']} so dont include it.

                Use the following schema:
                apartments("id" int, "title" text, "display_address" text, "bathrooms" bigint, 
                           "bedrooms" bigint, "added_on" text, "type" text, "rera" bigint, 
                           "price" int, "property_type" text)
                """
            else:
                state["prompt"] = "ERROR: No matching apartment found."
        except Exception as e:
            state["prompt"] = f"ERROR: Failed to fetch apartment: {e}"
    else:
        state["prompt"] = "ERROR: Could not extract reference apartment."

    return state

def generate_sql(state: AgentState) -> AgentState:
    if state["prompt"].startswith("ERROR"):
        state["sql"] = ""
        state["result"] = state["prompt"]
        return state

    try:
        response = gemini.generate_content(state["prompt"])
        sql_raw = response.text.strip()

        match = re.search(r"```(?:sql)?\s*(.*?)\s*```", sql_raw, re.DOTALL)
        state["sql"] = match.group(1).strip() if match else sql_raw
    except Exception as e:
        state["sql"] = ""
        state["result"] = f"Gemini Error: {e}"

    return state

def execute_sql(state: AgentState) -> AgentState:
    if not state["sql"]:
        return state

    try:
        with engine.connect() as conn:
            result = conn.execute(text(state["sql"]))
            rows = [dict(row._mapping) for row in result]
            state["result"] = str(rows)
    except SQLAlchemyError as e:
        state["result"] = f"SQL Error: {str(e)}"

    return state

builder = StateGraph(AgentState)
builder.add_node("Classify", prompt_classifier)
builder.add_node("ModifyPrompt", query_modifier)
builder.add_node("LLMtoSQL", generate_sql)
builder.add_node("ExecuteSQL", execute_sql)

builder.set_entry_point("Classify")
builder.add_edge("Classify", "ModifyPrompt")
builder.add_edge("ModifyPrompt", "LLMtoSQL")
builder.add_edge("LLMtoSQL", "ExecuteSQL")
builder.add_edge("ExecuteSQL", END)

agent_graph = builder.compile()

if __name__ == "__main__":
    print("Real Estate SQL Agent")
    print("Ask about listings, or about characteristics. Type 'exit' to quit.\n")

    while True:
        query = input("Enter your question: ")

        if query.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        state = {
            "input": query,
            "prompt": "",
            "sql": "",
            "result": "",
            "is_similar_query": False
        }

        output = agent_graph.invoke(state)

        print("\nSQL Generated:\n", output["sql"])
        print("\nQuery Result:\n", output["result"], "\n")
