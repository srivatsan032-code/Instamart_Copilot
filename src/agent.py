"""Part D tasks 5-8: tools, ReAct loop, safe stopping, checkpoint/resume."""
import json, os, sqlite3, sys, time
from datetime import datetime, timezone
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "src"))
import math
import mock_llm
import rag

DB_PATH = os.path.join(ROOT, "data", "instamart.db")
LEDGER_PATH = os.path.join(ROOT, "outputs", "refund_ledger.jsonl")     # demos override this module variable
STATE_DIR = os.path.join(ROOT, "state")

class SimulatedCrash(RuntimeError):
    pass

# ------------------------------------------------------------------ tools
def get_order(order_id):
    con = sqlite3.connect(DB_PATH)
    try:
        row = con.execute("SELECT order_id, status, order_amount, delivery_minutes FROM orders WHERE order_id = ?",
                          (order_id,)).fetchone()
    finally:
        con.close()
    if row is None:
        return {"error": "order not found"}
    dm = None if row[3] is None or (isinstance(row[3], float) and math.isnan(row[3])) else row[3]
    return {"order_id": row[0], "status": row[1], "order_amount": float(row[2]), "delivery_minutes": dm}

def search_policy(query):
    return {"results": rag.retrieve(query, 3)}

def calculate_refund(order_amount, delivery_minutes):
    """Delivered in more than 45 minutes -> 100% refund; otherwise 0. (Status is checked by the agent flow.)"""
    if delivery_minutes is not None and delivery_minutes > 45:
        return {"refund_inr": round(float(order_amount), 2)}
    return {"refund_inr": 0}

def _ledger(tool, order_id, refund_inr):
    os.makedirs(os.path.dirname(LEDGER_PATH), exist_ok=True)
    with open(LEDGER_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps({"timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                            "tool": tool, "order_id": order_id, "refund_inr": refund_inr}) + "\n")
    return {"refund_inr": refund_inr}

def issue_refund(order_id, refund_inr):
    return _ledger("issue_refund", order_id, refund_inr)

def escalate_to_human(order_id, refund_inr):
    return _ledger("escalate_to_human", order_id, refund_inr)

def _spec(name, desc, props, required):
    return {"name": name, "description": desc,
            "parameters": {"type": "object", "properties": props, "required": required}}

TOOLS_SPEC = [
    _spec("get_order", "Look up an order by id in the Instamart database.",
          {"order_id": {"type": "string", "description": "Order id such as ORD-100123"}}, ["order_id"]),
    _spec("search_policy", "Search the policy documents and return the three best matching chunks.",
          {"query": {"type": "string", "description": "Free-text policy query"}}, ["query"]),
    _spec("calculate_refund", "Compute the refund in INR for a completed order from its amount and delivery time.",
          {"order_amount": {"type": "number"}, "delivery_minutes": {"type": "number"}}, ["order_amount", "delivery_minutes"]),
    _spec("issue_refund", "Issue an automatic refund (up to INR 1000) and record it in the ledger.",
          {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}}, ["order_id", "refund_inr"]),
    _spec("escalate_to_human", "Escalate a refund above INR 1000 to a human support lead and record it.",
          {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}}, ["order_id", "refund_inr"]),
]
TOOL_MAP = {"get_order": get_order, "search_policy": search_policy, "calculate_refund": calculate_refund,
            "issue_refund": issue_refund, "escalate_to_human": escalate_to_human}

SYSTEM_AGENT = ("You are a refund AGENT for Instamart support. Use the tools get_order, search_policy, "
                "calculate_refund, issue_refund and escalate_to_human. Finish with one line of the form "
                "STATUS: explanation where STATUS is REFUND, ESCALATE, DENIED or NOT_FOUND.")

# ------------------------------------------------------------------ state
def _state_path(run_id):
    return os.path.join(STATE_DIR, f"{run_id}.json")

def _save(state):
    os.makedirs(STATE_DIR, exist_ok=True)
    tmp = _state_path(state["run_id"]) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=1, default=str)
    os.replace(tmp, _state_path(state["run_id"]))

def _fallback(state, status, reason):
    done = [s["tool"] for s in state["steps"]]
    summary = (f"Handing over to a human agent ({reason}). Request: {state['user_message']!r}. "
               f"Tools completed so far: {done if done else 'none'}.")
    return _result(state, status, summary)

def _result(state, status, final_answer):
    refund = 0.0
    for s in state["steps"]:
        if s["tool"] in ("issue_refund", "escalate_to_human") and "refund_inr" in s["observation"]:
            refund = float(s["observation"]["refund_inr"])
    return {"run_id": state["run_id"], "scenario_id": state["scenario_id"], "status": status,
            "final_answer": final_answer, "steps": state["steps"], "llm_calls": state["llm_calls"],
            "tokens_used": state["tokens_used"], "refund_inr": refund,
            "tools_executed": list(state["tools_executed"])}

# ------------------------------------------------------------------ the loop
def _loop(state, max_iterations, token_limit, time_limit_seconds, crash_after_tools):
    t0 = time.monotonic()
    while True:
        if time.monotonic() - t0 >= time_limit_seconds:                      # (c) wall-clock limit
            return _fallback(state, "FALLBACK_TIMEOUT", "time limit reached")
        if state["llm_calls"] >= max_iterations:                             # (a) iteration limit
            return _fallback(state, "FALLBACK_MAX_ITERATIONS", f"no final answer after {max_iterations} model calls")
        resp = mock_llm.chat(state["messages"], tools=TOOLS_SPEC)
        state["llm_calls"] += 1
        state["tokens_used"] += resp.usage.total_tokens
        if state["tokens_used"] > token_limit:                               # (b) token limit; tool NOT executed
            return _fallback(state, "FALLBACK_TOKEN_LIMIT", f"token limit {token_limit} exceeded")
        msg = resp.choices[0].message
        if not msg.tool_calls:
            final = msg.content or ""
            status = final.split(":", 1)[0].strip() if ":" in final else "UNKNOWN"
            return _result(state, status, final)
        for tc in msg.tool_calls:
            state["messages"].append({"role": "assistant", "content": None,
                                      "tool_calls": [{"id": tc.id, "name": tc.name, "arguments": tc.arguments}]})
            fn = TOOL_MAP.get(tc.name)
            try:
                obs = fn(**tc.arguments) if fn else {"error": f"unknown tool {tc.name}"}
            except Exception as e:                                           # a tool error is an observation, not a crash
                obs = {"error": f"{type(e).__name__}: {e}"}
            state["messages"].append({"role": "tool", "tool_call_id": tc.id, "name": tc.name,
                                      "content": json.dumps(obs)})
            state["steps"].append({"tool": tc.name, "arguments": tc.arguments, "observation": obs})
            state["tools_executed"].append(tc.name)
            state["step"] += 1
            _save(state)                                                      # checkpoint after every observation
            if crash_after_tools is not None and len(state["tools_executed"]) == crash_after_tools:
                raise SimulatedCrash(f"simulated crash after tool #{crash_after_tools} ({tc.name})")

def run_agent(user_message, scenario_id, max_iterations=6, token_limit=6000, time_limit_seconds=10,
              run_id=None, crash_after_tools=None):
    state = {"run_id": run_id or f"run_{scenario_id}", "scenario_id": scenario_id, "user_message": user_message,
             "messages": [{"role": "system", "content": SYSTEM_AGENT}, {"role": "user", "content": user_message}],
             "tokens_used": 0, "step": 0, "llm_calls": 0, "tools_executed": [], "steps": []}
    return _loop(state, max_iterations, token_limit, time_limit_seconds, crash_after_tools)

def resume_agent(run_id, max_iterations=6, token_limit=6000, time_limit_seconds=10):
    with open(_state_path(run_id), encoding="utf-8") as f:
        state = json.load(f)
    return _loop(state, max_iterations, token_limit, time_limit_seconds, None)
