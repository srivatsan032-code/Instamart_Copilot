"""Part D: RAG evaluation, seven agent scenarios, stop-safely demos, crash-and-resume.  python src/part_d_run.py"""
import glob, json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "src"))
import pandas as pd
import agent, rag
os.makedirs("outputs", exist_ok=True); os.makedirs("state", exist_ok=True)

# ---------------------------------------------------------------- RAG
print("=== Chunking")
total = 0
for p in sorted(glob.glob("data/policies/*.txt")):
    n = len(rag.chunk_words(open(p, encoding="utf-8").read(), 40, 10)); total += n
    print(f"{os.path.basename(p)}: {n} chunks")
print("total chunks:", total)
print("top-3 sources for 'refund policy late delivery':", [h["source"] for h in rag.retrieve("refund policy late delivery")])
for ov in (10, 0):
    idx = rag.build_index(40, ov)
    res, fails = rag.evaluate(idx)
    print(f"\n=== Evaluation, chunk=40 overlap={ov} ({len(idx['chunks'])} chunks)")
    print(res)
    for q, why, txt in fails:
        print("  FAIL:", q, "|", why, "|", txt.replace("\n", " "))

# ---------------------------------------------------------------- seven scenarios
for path in ["outputs/agent_runs.jsonl", "outputs/refund_ledger.jsonl", "outputs/agent_demos.json",
             "outputs/demo_refund_ledger.jsonl"] + glob.glob("state/*.json"):
    if os.path.exists(path):
        os.remove(path)
sc = pd.read_csv("data/raw/agent_scenarios.csv")
rows = []
print("\n=== Agent scenarios")
with open("outputs/agent_runs.jsonl", "a", encoding="utf-8") as out:
    for s in sc.itertuples(index=False):
        r = agent.run_agent(s.message, s.scenario_id)
        out.write(json.dumps({"scenario_id": r["scenario_id"], "status": r["status"], "final_answer": r["final_answer"],
                              "llm_calls": r["llm_calls"], "tokens_used": r["tokens_used"],
                              "refund_inr": r["refund_inr"],
                              "steps": [{"tool": x["tool"], "arguments": x["arguments"], "observation": x["observation"]}
                                        for x in r["steps"]]}) + "\n")
        expected = s.expected_outcome
        ok = (r["status"] == expected or (expected == "FALLBACK" and r["status"] == "FALLBACK_MAX_ITERATIONS")) \
            and abs(r["refund_inr"] - float(s.expected_refund_inr)) <= 0.01
        rows.append((s.scenario_id, r["status"], r["refund_inr"], r["llm_calls"], r["tokens_used"], expected, ok))
tbl = pd.DataFrame(rows, columns=["scenario", "status", "refund_inr", "llm_calls", "tokens", "expected", "matches"])
print(tbl.to_string(index=False))
assert tbl.matches.all()
s1_msg = sc.loc[sc.scenario_id == "S1", "message"].iloc[0]
demos = {}

# ---------------------------------------------------------------- stop-safely demos
print("\n=== Stop-safely demos")
r = agent.run_agent(s1_msg, "S1", token_limit=200, run_id="demo_S1_token200")
print("S1 token_limit=200 ->", r["status"], "| llm_calls", r["llm_calls"], "| tokens", r["tokens_used"], "| tools run:", r["tools_executed"])
print("   handoff:", r["final_answer"]); demos["token_limit_200"] = {k: r[k] for k in ("status", "llm_calls", "tokens_used", "tools_executed", "final_answer")}
r = agent.run_agent(s1_msg, "S1", time_limit_seconds=0, run_id="demo_S1_timeout0")
print("S1 time_limit_seconds=0 ->", r["status"], "| llm_calls", r["llm_calls"], "| tools run:", r["tools_executed"])
print("   handoff:", r["final_answer"]); demos["time_limit_0"] = {k: r[k] for k in ("status", "llm_calls", "tokens_used", "tools_executed", "final_answer")}
s7 = sc.loc[sc.scenario_id == "S7"].iloc[0]
r = agent.run_agent(s7.message, "S7", run_id="demo_S7_default_limits")
print("S7 default limits ->", r["status"], "| llm_calls", r["llm_calls"], "| tokens", r["tokens_used"])

# ---------------------------------------------------------------- crash and resume
print("\n=== Crash and resume (S1)")
agent.LEDGER_PATH = os.path.join(ROOT, "outputs", "demo_refund_ledger.jsonl")
try:
    agent.run_agent(s1_msg, "S1", run_id="demo_S1_crash", crash_after_tools=2)
except agent.SimulatedCrash as e:
    print("crashed:", e)
ck = json.load(open("state/demo_S1_crash.json"))
print("checkpoint tools_executed:", ck["tools_executed"], "| step:", ck["step"], "| tokens_used:", ck["tokens_used"],
      "| messages:", len(ck["messages"]))
r = agent.resume_agent("demo_S1_crash")
print("resumed ->", r["status"], r["refund_inr"], "| tools executed across crash+resume:", r["tools_executed"])
assert r["tools_executed"] == ["get_order", "search_policy", "calculate_refund", "issue_refund"]
assert r["status"] == "REFUND" and abs(r["refund_inr"] - 128.25) < 0.01
print("demo ledger lines:", open("outputs/demo_refund_ledger.jsonl").read().count("\n"))
demos["crash_resume"] = {k: r[k] for k in ("status", "refund_inr", "tools_executed", "llm_calls", "tokens_used")}
json.dump(demos, open("outputs/agent_demos.json", "w"), indent=1)
print("\nledger lines (7 scenarios):", open("outputs/refund_ledger.jsonl").read().count("\n"))
print("state files:", sorted(os.path.basename(p) for p in glob.glob("state/*.json")))
