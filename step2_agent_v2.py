"""
Step 2: Conversational SQL Agent using LangGraph
867 Sales Data - Jan 2022 to July 2024

Architecture (single agent, 5-node sequential graph with retry loop):

  [Node 1] get_schema     → fetch table schema + sample values from SQLite
      ↓
  [Node 2] generate_sql   → LLM generates SQL from question + schema
      ↓
  [Node 3] validate_sql   → blocks INSERT/UPDATE/DELETE/DROP (SELECT only)
      ↓
  [Node 4] execute_sql    → runs query, captures rows or error
      ↓ error? ──────────────────────► [Node 2] retry (max 2x)
      ↓ success
  [Node 5] format_answer  → LLM converts raw rows → natural language answer

Run:
  export OPENAI_API_KEY=sk-...
  python step2_agent_v2.py
"""

import sqlite3
import re
from typing import TypedDict, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

# ── CONFIG ────────────────────────────────────────────────────────────────────
DB_PATH   = "sales867.db"
MODEL     = "gpt-4o"
MAX_RETRY = 2

# ── STATE ─────────────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    question:     str
    schema:       str
    sql:          str
    sql_result:   str
    final_answer: str
    error:        Optional[str]
    retry_count:  int

# ── LLM ───────────────────────────────────────────────────────────────────────
llm = ChatOpenAI(model=MODEL, temperature=0)

# ── HELPERS ───────────────────────────────────────────────────────────────────
def get_db_schema(db_path: str) -> str:
    conn = sqlite3.connect(db_path)
    cur  = conn.cursor()

    cur.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='sales'")
    ddl = cur.fetchone()[0]

    # Sample values for key categorical columns
    key_cols = [
        "product", "transaction_type", "item_dose",
        "receiver_class_of_trade", "distributor_class_of_trade",
        "receiver_channel", "receiver_state", "distributor_state",
        "year", "month_name", "quarter",
    ]
    samples = []
    for col in key_cols:
        try:
            cur.execute(f"SELECT DISTINCT {col} FROM sales WHERE {col} IS NOT NULL LIMIT 8")
            vals = [str(r[0]) for r in cur.fetchall()]
            samples.append(f"  -- {col}: {', '.join(vals)}")
        except Exception:
            pass

    conn.close()
    return ddl + "\n\n-- Sample values for key columns:\n" + "\n".join(samples)


def run_query(db_path: str, sql: str):
    try:
        conn = sqlite3.connect(db_path)
        cur  = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description] if cur.description else []
        conn.close()
        return cols, rows, None
    except Exception as e:
        return [], [], str(e)

# ── NODES ─────────────────────────────────────────────────────────────────────
def node_get_schema(state: AgentState) -> AgentState:
    state["schema"]      = get_db_schema(DB_PATH)
    state["retry_count"] = state.get("retry_count", 0)
    return state


def node_generate_sql(state: AgentState) -> AgentState:
    error_context = ""
    if state.get("error"):
        error_context = f"""
The previous SQL failed with this error:
{state['error']}

Previously generated SQL:
{state.get('sql', '')}

Please fix the SQL based on the error above.
"""

    system = SystemMessage(content="""You are an expert SQL analyst for a pharmaceutical sales database (SQLite).
Generate a single SQLite SELECT query to answer the user's question.

IMPORTANT RULES:
- Output ONLY the raw SQL query — no markdown, no backticks, no explanation.
- Use only SELECT. Never use INSERT, UPDATE, DELETE, DROP, ALTER, or CREATE.
- Always filter to transaction_type = 'Sales' unless the user asks about returns or transfers.

KEY COLUMNS in the `sales` table:
  day_date               — transaction date (YYYY-MM-DD)
  year, month, quarter   — derived from day_date
  month_name             — e.g. 'January', 'February'
  week                   — ISO week number

  product                — 'AKYNZEO Oral' or 'AKYNZEO IV'
  item_dose              — full dose description
  item_pack, item_strength

  current_value          — dollar value of the transaction
  extended_unit          — unit quantity (use this for unit counts)
  pack_unit              — same as extended_unit (units shipped)

  transaction_type       — 'Sales', 'Returns', 'Transfers'

  distributor            — distributor company name
  distributor_state      — distributor state (2-letter abbreviation)
  distributor_class_of_trade
  distributor_channel

  receiver               — end customer/facility name
  receiver_state         — receiver state (2-letter abbreviation)
  receiver_city
  receiver_zip
  receiver_class_of_trade — e.g. 'Oncology Clinic', 'DSH/PHS Hospital', etc.
  receiver_channel
  receiver_senior_parent
  receiver_trade_partner

  contract_number        — contract used (nullable)
  ndc_no                 — NDC product code
  invoice_number

TIPS:
- For "total sales" use SUM(current_value) for revenue or SUM(extended_unit) for units
- For "territory" or "region" questions, use receiver_state
- For top N queries, add ORDER BY ... DESC LIMIT N
- Round monetary values: ROUND(SUM(current_value), 2)
- Date filtering: WHERE year = 2024 or WHERE day_date BETWEEN 'YYYY-MM-DD' AND 'YYYY-MM-DD'
""")

    human = HumanMessage(content=f"""
Database Schema:
{state['schema']}

{error_context}
User Question: {state['question']}

SQL Query:""")

    response = llm.invoke([system, human])
    raw_sql  = response.content.strip()
    raw_sql  = re.sub(r"```sql|```", "", raw_sql).strip()

    state["sql"]   = raw_sql
    state["error"] = None
    return state


def node_validate_sql(state: AgentState) -> AgentState:
    sql_upper = state["sql"].upper().strip()
    forbidden = ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "TRUNCATE"]
    for kw in forbidden:
        if re.search(rf"\b{kw}\b", sql_upper):
            state["error"]      = f"Blocked: SQL contains forbidden keyword '{kw}'. Only SELECT is allowed."
            state["sql_result"] = state["error"]
            return state
    return state


def node_execute_sql(state: AgentState) -> AgentState:
    if state.get("error"):
        return state

    cols, rows, error = run_query(DB_PATH, state["sql"])

    if error:
        state["error"]      = error
        state["sql_result"] = f"Error: {error}"
    else:
        if rows:
            header    = " | ".join(cols)
            divider   = "-" * len(header)
            row_lines = [" | ".join(str(v) for v in row) for row in rows[:50]]
            state["sql_result"] = f"{header}\n{divider}\n" + "\n".join(row_lines)
            if len(rows) > 50:
                state["sql_result"] += f"\n... ({len(rows) - 50} more rows)"
        else:
            state["sql_result"] = "No results found."
        state["error"] = None

    return state


def node_format_answer(state: AgentState) -> AgentState:
    system = SystemMessage(content="""You are a concise pharmaceutical sales analyst.
Convert the SQL query result into a clear, specific natural language answer.
- Include actual numbers, state names, product names, and time periods from the result.
- For dollar values, format with $ and commas (e.g. $1,234,567.89).
- For units, use plain numbers with commas.
- If no results, say so and suggest why.
- Do NOT mention SQL, tables, or technical details.""")

    human = HumanMessage(content=f"""
User asked: {state['question']}

SQL used: {state['sql']}

Result:
{state['sql_result']}

Answer:""")

    response = llm.invoke([system, human])
    state["final_answer"] = response.content.strip()
    return state

# ── ROUTING ───────────────────────────────────────────────────────────────────
def should_retry(state: AgentState) -> str:
    if state.get("error") and state.get("retry_count", 0) < MAX_RETRY:
        state["retry_count"] = state.get("retry_count", 0) + 1
        return "retry"
    return "format"

# ── BUILD GRAPH ───────────────────────────────────────────────────────────────
def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("get_schema",    node_get_schema)
    graph.add_node("generate_sql",  node_generate_sql)
    graph.add_node("validate_sql",  node_validate_sql)
    graph.add_node("execute_sql",   node_execute_sql)
    graph.add_node("format_answer", node_format_answer)

    graph.set_entry_point("get_schema")
    graph.add_edge("get_schema",    "generate_sql")
    graph.add_edge("generate_sql",  "validate_sql")
    graph.add_edge("validate_sql",  "execute_sql")
    graph.add_conditional_edges(
        "execute_sql",
        should_retry,
        {"retry": "generate_sql", "format": "format_answer"},
    )
    graph.add_edge("format_answer", END)

    return graph.compile()

# ── CHAT LOOP ─────────────────────────────────────────────────────────────────
def chat():
    print("=" * 65)
    print("  867 Sales Conversational AI  |  Jan 2022 – Jul 2024")
    print("  Ask about Akynzeo sales, revenue, territories, customers")
    print("  Type 'quit' to exit")
    print("=" * 65)

    agent = build_graph()

    while True:
        question = input("\nYou: ").strip()
        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if not question:
            continue

        initial_state: AgentState = {
            "question":     question,
            "schema":       "",
            "sql":          "",
            "sql_result":   "",
            "final_answer": "",
            "error":        None,
            "retry_count":  0,
        }

        result = agent.invoke(initial_state)

        print(f"\nAgent: {result['final_answer']}")
        print(f"\n[SQL Used]\n{result['sql']}")


if __name__ == "__main__":
    chat()
