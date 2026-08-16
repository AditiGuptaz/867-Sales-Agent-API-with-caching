"""
═══════════════════════════════════════════════════════════════════════════
  867 SALES AGENT — FASTAPI + CACHING  (System Design, Step 2)
═══════════════════════════════════════════════════════════════════════════

  WHAT CHANGED FROM STEP 1:
  We added a CACHE. Before calling the (slow, costly) LLM agent, we check:
  "have we answered this exact question before?" If yes, return the saved
  answer instantly — no LLM call, no cost, no wait.

  WHY THIS MATTERS (the system-design lesson):
  Every agent call costs money (~$0.02) and takes ~8 seconds. In production
  with thousands of users, the SAME questions get asked over and over
  ("2024 revenue?", "top customers?"). Answering those from cache instead
  of re-running the LLM:
     - CUTS COST     (no LLM call for a repeat question)
     - CUTS LATENCY  (instant instead of ~8 seconds)
  This is the single highest-value system-design optimization.

  THE TRADE-OFF (there's always one):
     - Cached answers can go STALE if the underlying data changes.
     - For this 867 data (historical, fixed), that's fine — the answer to
       "2024 revenue" never changes. For live-changing data you'd add a
       time limit (TTL) so the cache expires.

  HOW TO RUN: same as before —
    uvicorn main:app --reload
    then watch the response: repeat questions come back instantly + marked cached.
═══════════════════════════════════════════════════════════════════════════
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from step2_agent_v2 import build_graph, AgentState


app = FastAPI(
    title="867 Sales Agent API (with caching)",
    description="Ask plain-English questions about Akynzeo sales data; get SQL-backed answers. Now with caching.",
    version="2.0.0",
)

agent = build_graph()


# ═══════════════════════════════════════════════════════════════════════════
# THE CACHE — this is the whole new concept
# ═══════════════════════════════════════════════════════════════════════════
# A cache is just a place to store answers we've already computed, so we can
# reuse them instead of recomputing. The simplest possible cache is a Python
# dictionary: { question -> answer }.
#
# In production you'd use Redis (a fast shared cache that survives restarts
# and works across multiple servers). But a dict teaches the exact same idea
# and works perfectly for one server. Start simple.
#
# Key insight: the cache lives in memory and persists BETWEEN requests
# because it's a module-level variable — the same dict every request sees.

CACHE: dict[str, dict] = {}   # question (lowercased) -> the full answer dict


def normalize(question: str) -> str:
    """
    Turn a question into a consistent cache KEY.
    Why: "2024 Revenue?" and "  2024 revenue? " should be treated as the SAME
    question, or the cache would miss obvious repeats. We lowercase and strip
    whitespace so trivial differences don't cause a cache miss.
    """
    return question.strip().lower()


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    question: str
    answer: str
    sql: str
    cached: bool   # NEW: tells the caller whether this came from cache


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    """Send a plain-English question, get back the answer, SQL, and whether it was cached."""

    key = normalize(request.question)

    # ── STEP 1: CHECK THE CACHE FIRST (before any expensive work) ────────────
    # This is the heart of caching: look before you leap. If we've seen this
    # question, return the stored answer immediately — no LLM call at all.
    if key in CACHE:
        saved = CACHE[key]
        return AskResponse(
            question=request.question,
            answer=saved["answer"],
            sql=saved["sql"],
            cached=True,          # ← flag it so you can SEE the cache working
        )

    # ── STEP 2: CACHE MISS — do the real (expensive) work ────────────────────
    # Only reached if the question is NEW. Run the agent as normal.
    initial_state: AgentState = {
        "question":     request.question,
        "schema":       "",
        "sql":          "",
        "sql_result":   "",
        "final_answer": "",
        "error":        None,
        "retry_count":  0,
    }

    try:
        result = agent.invoke(initial_state)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {e}")

    # ── STEP 3: SAVE TO CACHE before returning ───────────────────────────────
    # Store this answer so the NEXT time someone asks the same question, we hit
    # STEP 1 and skip all of this. This is what makes the cache fill up over time.
    CACHE[key] = {
        "answer": result["final_answer"],
        "sql":    result["sql"],
    }

    return AskResponse(
        question=request.question,
        answer=result["final_answer"],
        sql=result["sql"],
        cached=False,             # ← this one was freshly computed
    )


# ═══════════════════════════════════════════════════════════════════════════
# CACHE MANAGEMENT ENDPOINTS — useful for monitoring & control
# ═══════════════════════════════════════════════════════════════════════════
# These show a second system-design idea: OBSERVABILITY. You want to SEE what
# the cache is doing (how many entries, what's stored) and be able to clear it.

@app.get("/cache/stats")
def cache_stats():
    """See how many questions are cached — basic observability."""
    return {
        "cached_questions": len(CACHE),
        "questions": list(CACHE.keys()),
    }


@app.post("/cache/clear")
def cache_clear():
    """
    Empty the cache. You'd call this if the underlying data changed and the
    cached answers might be stale. (For fixed historical 867 data, rarely needed.)
    """
    count = len(CACHE)
    CACHE.clear()
    return {"cleared": count, "message": "Cache emptied."}


@app.get("/")
def health():
    return {"status": "ok", "service": "867 Sales Agent API", "cache_size": len(CACHE)}
