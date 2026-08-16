# 867 Sales Conversational SQL Agent

A conversational AI agent that answers plain-English questions about pharmaceutical sales data by generating and running SQL — built with **LangGraph**, served as a **FastAPI** REST API, and optimized with caching.

Ask *"What was total revenue in 2024?"* and get back a SQL-backed answer, not a hallucination.

---

## What it does

```
Question:  "total revenue in 2024"
    ↓
Agent:     writes SQL → validates it → runs it against the database → formats the answer
    ↓
Answer:    "Total revenue in 2024 was $27,814,963.53"
           (+ the exact SQL it used, for transparency)
```

The agent turns natural language into validated SQL over a real sales dataset (57,751 rows of pharmaceutical 867 sales data), runs it, and returns a clear answer plus the query it used.

---

## Architecture

A **5-node LangGraph agent** with a retry loop:

```
get_schema → generate_sql → validate_sql → execute_sql → format_answer
                  ↑                              │
                  └────────── retry on error ────┘  (max 2 retries)
```

- **get_schema** — reads the database structure so the model knows the columns
- **generate_sql** — GPT-4o writes SQL from the question (grounded in the schema)
- **validate_sql** — blocks anything that isn't a safe SELECT (no writes/deletes)
- **execute_sql** — runs the query against SQLite
- **format_answer** — turns the raw result into a natural-language answer
- **retry loop** — if SQL fails, loops back to regenerate (up to 2 times)

---

## Key features

- **Natural-language to SQL** over real pharmaceutical sales data
- **Safety validation** — only read-only SELECT queries are allowed to execute
- **Self-correction** — a retry loop regenerates SQL when a query fails
- **Deployed as a REST API** (FastAPI) with an interactive `/docs` page
- **Caching** — repeat questions return instantly with no LLM call, cutting cost and latency
- **Context engineering** — sends the compact schema (~1,185 tokens), never the raw 57K rows

---

## Tech stack

`LangGraph` · `LangChain` · `OpenAI GPT-4o` · `FastAPI` · `SQLite` · `pandas` · `Pydantic`

---

## Project structure

| File | Role |
|---|---|
| `step1_ingest_v2.py` | Loads the source Excel into a SQLite database |
| `step2_agent_v2.py` | The 5-node LangGraph agent (core logic) |
| `main.py` | FastAPI wrapper — serves the agent as a REST API, with caching |
| `step3_evals.py` | Evaluation harness — 15 tests across 6 categories |
| `requirements.txt` | Dependencies |

---

## Run it

```bash
# 1. Install
pip install -r requirements.txt

# 2. Set your OpenAI key
export OPENAI_API_KEY=sk-your-key

# 3. Build the database from the source data
python step1_ingest_v2.py

# 4. Serve the API
uvicorn main:app --reload
```

Then open **http://localhost:8000/docs** for an interactive page to ask questions.

Or call it directly:
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "total revenue in 2024"}'
```

```json
{
  "question": "total revenue in 2024",
  "answer": "The total revenue from sales in 2024 is $27,814,963.53.",
  "sql": "SELECT ROUND(SUM(current_value), 2) FROM sales WHERE transaction_type = 'Sales' AND year = 2024",
  "cached": false
}
```

---

## System-design notes

This project demonstrates several production considerations beyond a working prototype:

- **Caching** — a repeat question is served from memory, skipping the LLM entirely (lower cost, instant response). Safe here because the data is historical and fixed, so cached answers never go stale.
- **Cost control** — the agent injects only the schema, not the data, keeping each prompt small (~1,185 tokens vs. ~10M for the raw rows).
- **Safety** — SQL validation blocks any non-SELECT statement from running.
- **Observability** — `/cache/stats` exposes what's cached; the `sql` field in every response makes the agent's reasoning transparent and auditable.

---

## Evaluation

`step3_evals.py` runs 15 test questions across 6 categories (aggregations, rankings, filters, product-specific queries, date logic, and edge cases), checking each answer against a known-correct value — the same measure-don't-eyeball discipline used to verify the agent works reliably, not just occasionally.
