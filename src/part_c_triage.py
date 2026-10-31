"""Part C: support-ticket triage with guardrails, redaction, retries, validation, budget and routing.
Run after Part B:  python src/part_c_triage.py"""
import json, os, random, re, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, ROOT)
import pandas as pd
import mock_llm
from mock_llm import est_tokens, TransientLLMError

MAX_INPUT_TOKENS = 400
MAX_RETRIES, BASE_DELAY = 3, 0.05
CATEGORIES = ["billing", "delivery", "technical", "fraud"]

# ------------------------------------------------------------------ redaction (order matters)
PATTERNS = [("email", re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[EMAIL]"),
            ("phone", re.compile(r"\b[6-9]\d{9}\b"), "[PHONE]"),
            ("order_id", re.compile(r"ORD-\d{6}"), "[ORDER_ID]")]

def redact(text, counter=None):
    for name, pat, ph in PATTERNS:
        text, n = pat.subn(ph, text)
        if counter is not None:
            counter[name] = counter.get(name, 0) + n
    return text

def pii_found(text):
    return sum(len(p.findall(text)) for _, p, _ in PATTERNS)

# ------------------------------------------------------------------ prompts
ZERO_SHOT = ('You CLASSIFY customer support tickets into exactly one category. '
             'Allowed categories: billing, delivery, technical, fraud. '
             'Reply with JSON only, in the form {"category": "...", "reason": "..."} and nothing else.')
FEW_SHOT = ZERO_SHOT + '''
EXAMPLES:
Ticket: I was charged twice for order [ORDER_ID]. Please refund the extra payment. Contact [EMAIL].
Answer: {"category": "billing", "reason": "double charge on an order"}
Ticket: My order [ORDER_ID] has not arrived and the rider is not picking up calls.
Answer: {"category": "delivery", "reason": "order not received"}
Ticket: The app crashes every time I open the cart and shows an error.
Answer: {"category": "technical", "reason": "app crash"}
Ticket: There is an unauthorized transaction on my card. I did not place it. This is fraud.
Answer: {"category": "fraud", "reason": "unauthorized transaction"}'''
assert "CLASSIFY" in ZERO_SHOT and "EXAMPLES:" not in ZERO_SHOT
assert "CLASSIFY" in FEW_SHOT and "EXAMPLES:" in FEW_SHOT and '"category": "fraud"' in FEW_SHOT

# ------------------------------------------------------------------ resilient call
RETRY_LOG = []
def call_with_retry(messages, ticket_id):
    for attempt in range(MAX_RETRIES + 1):
        try:
            return mock_llm.chat(messages)
        except TransientLLMError:
            if attempt == MAX_RETRIES:
                raise
            wait = BASE_DELAY * 2 ** attempt * random.uniform(0.9, 1.15)
            line = f"retry ticket={ticket_id} retry_number={attempt} wait={wait:.4f}s"
            RETRY_LOG.append((ticket_id, attempt, wait, line)); print(line)
            time.sleep(wait)

# ------------------------------------------------------------------ output validation
def validate(content):
    """Return (status, category). Never guess."""
    try:
        obj = json.loads(content)
        if isinstance(obj, dict) and isinstance(obj.get("category"), str) and obj["category"] in CATEGORIES:
            return "classified_json", obj["category"]
        return "flagged_invalid_output", ""
    except (json.JSONDecodeError, TypeError):
        words = set(re.findall(r"[a-z]+", (content or "").lower()))
        found = [c for c in CATEGORIES if c in words]
        if len(found) == 1:
            return "classified_keyword", found[0]
        return "flagged_invalid_output", ""

# ------------------------------------------------------------------ pipeline
def load_tickets():
    return pd.read_csv("data/raw/tickets.csv", keep_default_na=False, dtype=str)

def run_pipeline(tickets, system_prompt, token_budget, verbose_label=""):
    mock_llm.reset(); random.seed(42); RETRY_LOG.clear()
    running, rows, sent = 0, [], []
    errors_recovered = 0
    for t in tickets.itertuples(index=False):
        text, status, cat, used = t.text, "", "", 0
        if not text.strip():
            status = "skipped_empty"
        elif est_tokens(text) > MAX_INPUT_TOKENS:
            status = "flagged_long"
        elif running >= token_budget:
            status = "budget_exhausted"
        else:
            red = redact(text)
            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": red}]
            assert messages[1]["content"] == red and pii_found(red) == 0 and pii_found(system_prompt) == 0
            sent.append(messages)
            try:
                resp = call_with_retry(messages, t.ticket_id)
            except TransientLLMError:
                resp = None
            if resp is None:
                status = "flagged_invalid_output"      # never expected with this mock; fail safe to humans
            else:
                used = resp.usage.total_tokens; running += used
                status, cat = validate(resp.choices[0].message.content)
        rows.append(dict(ticket_id=t.ticket_id, customer_id=t.customer_id, status=status, category=cat,
                         tokens_used=used, true_category=t.true_category))
    df = pd.DataFrame(rows)
    # audit: no PII in anything ever sent to the model
    assert all(pii_found(m["content"]) == 0 for msgs in sent for m in msgs)
    return df, len(sent), running

def route(df, seg):
    seg = seg.set_index("customer_id")
    out = df.copy()
    out["segment_name"] = out.customer_id.map(seg["segment_name"]).fillna("")
    prem = out.customer_id.map(seg["is_premium"]).fillna(False).astype(bool)
    human = out.status.isin(["skipped_empty", "flagged_long", "flagged_invalid_output", "budget_exhausted"])
    out["route"] = "support_queue"; out["priority"] = "NORMAL"
    fraud = (~human) & (out.category == "fraud")
    pdk = (~human) & (~fraud) & prem
    out.loc[pdk, ["route", "priority"]] = ["premium_desk", "HIGH"]
    out.loc[fraud, ["route", "priority"]] = ["security_team", "HIGH"]
    out.loc[human, ["route", "priority"]] = ["human_review", "NORMAL"]
    return out[["ticket_id", "customer_id", "status", "category", "segment_name", "priority", "route", "tokens_used"]]

def summarize(label, df):
    clf = df[df.status.isin(["classified_json", "classified_keyword"])]
    y = pd.read_csv("data/raw/tickets.csv", keep_default_na=False, dtype=str).set_index("ticket_id")["true_category"]
    correct = int((clf.category == clf.ticket_id.map(y)).sum())
    tot = int(df.tokens_used.sum())
    print(f"\n[{label}] statuses: {df.status.value_counts().to_dict()}")
    print(f"[{label}] classified={len(clf)} correct={correct} accuracy={correct/len(clf)*100:.1f}% "
          f"total_tokens={tot} tokens_per_classified={tot/len(clf):.1f}")
    return clf, correct, tot

if __name__ == "__main__":
    tickets = load_tickets()
    seg = pd.read_csv("outputs/customer_segments.csv")

    # ---- guardrail + redaction audit
    print("=== Guardrails and redaction")
    empty = (tickets.text.str.strip() == "").sum()
    long_ = tickets.text.map(lambda s: bool(s.strip()) and est_tokens(s) > MAX_INPUT_TOKENS)
    print(f"empty tickets: {empty} | long tickets: {int(long_.sum())} | est tokens of long: {sorted(set(tickets.text[long_].map(est_tokens)))}")
    counter = {}
    red_all = tickets.text.map(lambda s: redact(s, counter))
    print("substitutions:", counter, "| total:", sum(counter.values()))
    print("PII found when re-applying patterns to redacted texts:", int(red_all.map(pii_found).sum()))
    print("zero-shot prompt est tokens:", est_tokens(ZERO_SHOT), "| few-shot prompt est tokens:", est_tokens(FEW_SHOT))

    # ---- full runs, no budget
    print("\n=== Zero-shot full run")
    zs, n_calls_zs, _ = run_pipeline(tickets, ZERO_SHOT, token_budget=10**9)
    zs_log = list(RETRY_LOG)
    print(f"TransientLLMError retries logged: {len(zs_log)} | model calls with a ticket: {n_calls_zs}")
    waits = [w for _, n, w, _ in zs_log]
    print(f"retry numbers seen: {sorted(set(n for _, n, _, _ in zs_log))} | wait range: {min(waits):.4f}-{max(waits):.4f}s")
    assert all(0.045 <= w <= 0.0575 for w in waits) and len(zs_log) == 19
    clf_z, ok_z, tok_z = summarize("zero-shot", zs)
    print("\nZero-shot confusion table (rows = true, cols = predicted), classified tickets only:")
    print(pd.crosstab(clf_z.true_category, clf_z.category).to_string())
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/retry_log.txt", "w") as f:
        f.write("\n".join(l for *_, l in zs_log) + "\n")

    print("\n=== Few-shot full run")
    fs, _, _ = run_pipeline(tickets, FEW_SHOT, token_budget=10**9)
    print(f"retries logged: {len(RETRY_LOG)}")
    clf_f, ok_f, tok_f = summarize("few-shot", fs)
    print(f"extra tokens per classified ticket (few-shot - zero-shot): {tok_f/len(clf_f) - tok_z/len(clf_z):.1f}")

    # ---- budget demo (zero-shot, budget 6000)
    print("\n=== Budget demo: zero-shot, TOKEN_BUDGET = 6000")
    bd, _, bd_tokens = run_pipeline(tickets, ZERO_SHOT, token_budget=6000)
    print("tokens consumed:", bd_tokens, "| status counts:", bd.status.value_counts().to_dict())
    n_ex = int((bd.status == "budget_exhausted").sum())
    print("budget_exhausted tickets:", n_ex); assert n_ex >= 1
    route(bd, seg).to_csv("outputs/triage_budget_demo.csv", index=False)

    # ---- final routed output (few-shot, budget 100000)
    print("\n=== Final run: few-shot, TOKEN_BUDGET = 100000")
    fin, _, fin_tokens = run_pipeline(tickets, FEW_SHOT, token_budget=100000)
    routed = route(fin, seg)
    routed.to_csv("outputs/triage_results.csv", index=False)
    print("rows:", len(routed), "| tokens:", fin_tokens)
    print("status counts:", routed.status.value_counts().to_dict())
    print("route x priority:\n", routed.groupby(["route", "priority"]).size().to_string())
    print("customers with no segment row:", int((routed.segment_name == "").sum()))
