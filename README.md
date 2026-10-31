# Instamart Customer Intelligence & Support Copilot

All data is synthetic and fictional. All money is in INR. No paid service, API key or network call is used: every model call goes through the supplied offline `mock_llm.py`.

## Overview

The Copilot is one repository in four connected Parts. Each Part consumes the files produced by the previous one.

| Part | What it does | Main files |
|---|---|---|
| A | Audits and cleans the raw exports, stores them in SQLite, answers seven SQL questions, cross-checks in pandas | `src/part_a_clean.py`, `src/validate.py`, `src/part_a_sql.py`, `sql/queries.sql` |
| B | RFM segmentation (K-means) and a lapse-prediction model | `src/part_b_segments.py`, `src/part_b_lapse.py` |
| C | Support-ticket triage: guardrails, PII redaction, retries, output validation, token budget, routing by segment | `src/part_c_triage.py` |
| D | Policy RAG assistant and a tool-using refund agent with safe stopping and crash recovery | `src/rag.py`, `src/agent.py`, `src/part_d_run.py` |

`generate_data.py` and `mock_llm.py` are the supplied files and are committed unmodified. Everything else is my own work. Allowed libraries only: standard library, pandas, numpy, scikit-learn, joblib.

Data flow: `data/raw` → (A) `data/clean` + `data/instamart.db` → (B) `outputs/customer_segments.csv` → (C) routing; `data/instamart.db` + `data/policies` → (D).

## Reproduce

Run from the repository root, in this order (Python 3.11 or newer; tested on 3.12). Every random seed is fixed (generator seed 42, `random_state=42` in all sklearn calls, `random.seed(42)` for retry jitter).

1. `git clone <repo-url> && cd <repo>` (use branch `main`)
2. `python -m venv .venv`
3. `source .venv/bin/activate` (Windows: `.venv\Scripts\activate`)
4. `pip install -r requirements.txt`
5. `python generate_data.py` — last line must be `generated: customers=1200 products=40 orders_raw=13084 tickets=164`
6. `python src/part_a_clean.py` — audit, clean, write `data/clean/*.csv` and `data/instamart.db`
7. `python src/validate.py` — prints `ALL CHECKPOINTS PASSED`
8. `python src/part_a_sql.py` — runs Q1–Q7, pandas cross-check, EDA
9. `python src/part_b_segments.py` — RFM, K-means, `outputs/customer_segments.csv`
10. `python src/part_b_lapse.py` — lapse model, `outputs/lapse_model.joblib`, `outputs/retention_list.csv`
11. `python src/part_c_triage.py` — `outputs/triage_results.csv`, `outputs/triage_budget_demo.csv`
12. `python src/part_d_run.py` — RAG evaluation, `outputs/agent_runs.jsonl`, `outputs/refund_ledger.jsonl`, `state/`

Steps 9 and 10 must run in that order (10 reads the segment file), and step 11 reads the segment file too.

## Part A

### Audit of the raw data

| Table | Shape | Columns with missing values (count, %) |
|---|---|---|
| customers.csv | 1200 × 5 | age 61 (5.08%), gender 30 (2.50%); customer_id, city, signup_date 0 |
| products.csv | 40 × 4 | none |
| orders.csv | 13084 × 9 | order_amount 53 (0.41%), delivery_minutes 2541 (19.42%); all other columns 0 |

Data types as loaded by pandas: customers — customer_id/city/signup_date/gender are strings, age float64. products — price_inr int64, the rest strings. orders — quantity and discount_pct int64, delivery_minutes float64, order_amount a string column (because it contains `error`), the rest strings.

Note on order_amount: pandas reads the literal text `N/A` as missing by default, so the 53 "missing" amounts are the `N/A` entries; the other 67 unparseable entries are the text `error`. Together they give the 120 non-numeric entries.

| Audit figure | Value |
|---|---|
| raw orders.csv rows | 13,084 |
| exact duplicate rows | 119 |
| rows with status pending | 656 |
| distinct order_id | 12,309 |
| non-numeric order_amount | 120 |
| negative delivery_minutes | 106 |
| blank delivery_minutes | 2,541 |
| customers: blank age | 61 |
| customers: age outside 10–100 | 6 |
| customers: distinct raw city spellings | 24 |
| customers: distinct non-blank raw gender spellings | 10 |
| customers: blank gender | 30 |

All twelve match the reference values.

### Cleaning rules and what each changed

Customers (1,200 rows):

| Rule | Effect |
|---|---|
| (a) city: strip spaces, title case | 24 spellings → 6 cities; 476 rows had their city text changed |
| (b) gender → male / female / unknown | 920 spellings normalised, 30 blanks → `unknown` |
| (c) ages outside 10–100 → missing, then city median via `groupby().transform("median")` | 6 impossible ages removed; 67 ages filled in total (61 blank + 6 removed) |
| (d) signup_date → datetime | all 1,200 values converted |

Result: Pune 227, Delhi 210, Bengaluru 197, Chennai 191, Hyderabad 191, Mumbai 184; female 610, male 560, unknown 30; 0 missing ages, all ages between 18 and 60.

Orders (13,084 → 12,309 rows), applied in this order:

| # | Rule | Effect |
|---|---|---|
| 1 | order_ts → datetime | all 13,084 values converted |
| 2 | remove exact duplicate rows | 119 rows removed |
| 3 | per order_id keep the record with the latest order_ts | 656 earlier `pending` records removed |
| 4 | order_amount → numeric (errors → missing), fill from `round(price_inr × quantity × (1 − discount_pct/100), 2)` | 112 amounts filled (120 raw entries, 8 of which sat on rows already removed by rules 2–3) |
| 5a | negative delivery_minutes → missing | 105 values set to missing (106 raw, 1 on a removed duplicate) |
| 5b | completed orders: fill missing delivery_minutes with the median of the product category among completed orders (`groupby().transform`) | 774 values filled; cancelled orders deliberately left missing because they were never delivered |

Result: 12,309 rows (one per order_id); 11,110 completed and 1,199 cancelled; no pending rows; 0 missing order_amount; 0 missing delivery_minutes among completed orders and 1,199 among cancelled; completed revenue = INR 2,998,668.80.

### Store and validate

The clean tables are written to `data/clean/{customers,products,orders}.csv` and to `data/instamart.db` (tables `customers`, `products`, `orders`; `order_ts` stored as text `YYYY-MM-DD HH:MM:SS`). `src/validate.py` asserts 22 checkpoints (all audit figures above, clean counts, revenue total, database tables and row count) and prints `ALL CHECKPOINTS PASSED`.

### SQL analysis (`sql/queries.sql`, run with `pd.read_sql_query`)

| Query | Technique | Result |
|---|---|---|
| Q1 | WHERE then HAVING | 30 customers |
| Q2 | three-table join | 30 rows; revenues sum to INR 2,998,668.80; largest cell Bengaluru / Electronics = INR 233,909.45 |
| Q3 | scalar subquery | 11 products priced above the catalogue average |
| Q4 | LEFT JOIN | 25 customers with no completed order |
| Q5 | two CTEs | 550 customers above the average spend of INR 2,552.06 |
| Q6 | RANK / DENSE_RANK / ROW_NUMBER per city | top per city: Bengaluru C0082 (12,856.20), Chennai C0930 (10,151.40), Delhi C0291 (13,742.60), Hyderabad C0743 (12,730.05), Mumbai C0693 (14,766.55), Pune C0426 (10,377.10); 8 rows have RANK different from ROW_NUMBER |
| Q7 | running total with `SUM() OVER (ORDER BY day)` | 183 daily rows; last cumulative value INR 2,998,668.80 |

Why ties cause the Q6 difference: customers with equal total spend share the same RANK (and DENSE_RANK), but ROW_NUMBER must give every row a distinct number, so after a tie the two sequences diverge.

### pandas cross-check

`pd.pivot_table(..., margins=True)` gives a grand total of INR 2,998,668.80, equal to the SQL total of Q2.

| City | Sum (INR) | Mean (INR) | Count |
|---|---|---|---|
| Bengaluru | 523,643.40 | 276.47 | 1,894 |
| Chennai | 463,629.15 | 272.72 | 1,700 |
| Delhi | 502,829.80 | 243.62 | 2,064 |
| Hyderabad | 528,589.90 | 284.95 | 1,855 |
| Mumbai | 479,058.50 | 297.00 | 1,613 |
| Pune | 500,918.05 | 252.48 | 1,984 |

### Exploratory statistics of completed order_amount

| Statistic | Value |
|---|---|
| mean / median | 269.91 / 135.00 |
| skewness | 4.61 |
| Q1 / Q3 | 64.80 / 297.00 |
| 1.5×IQR fences | lower −283.50, upper 645.30 |
| outliers below / above | 0 / 897 |

Correlation matrix (completed orders):

| | order_amount | discount_pct | quantity | delivery_minutes |
|---|---|---|---|---|
| order_amount | 1.000 | −0.095 | 0.087 | −0.003 |
| discount_pct | −0.095 | 1.000 | −0.000 | −0.010 |
| quantity | 0.087 | −0.000 | 1.000 | −0.013 |
| delivery_minutes | −0.003 | −0.010 | −0.013 | 1.000 |

**Answers (126 words).**
*What is the data about?* 11,110 completed quick-commerce orders in six cities over six months, one product line per order with quantity, discount and delivery time.
*Distributions?* order_amount is strongly right-skewed: mean 269.91 is double the median 135.00, skewness 4.61, middle half of orders between 64.80 and 297.00.
*Outliers or anomalies?* 897 orders lie above the 645.30 fence and none below; 635 of the 897 are Electronics, so they look like genuine big-ticket purchases and were kept. About 9% of completed orders took more than 45 minutes (some delivery times were imputed, see rule 5b).
*Relationships?* All correlations are weak. The largest in absolute value is order_amount with discount_pct (−0.095), just ahead of order_amount with quantity (+0.087). A correlation of this size does not establish causation.

## Part B

Only completed orders are used for customer behaviour unless stated.

### RFM table (snapshot 2025-10-01 00:00:00)

1,175 customers. Recency: median 19, min 0, max 180 days. Frequency: mean 9.46, max 33. Monetary: mean 2,552.06, max 14,766.55 INR.

### K-means (StandardScaler, `n_init=10`, `random_state=42`)

| K | Inertia | Silhouette |
|---|---|---|
| 2 | 1820.8 | 0.465 |
| 3 | 1268.2 | 0.445 |
| 4 | 743.1 | 0.526 |
| 5 | 617.4 | 0.480 |
| 6 | 503.2 | 0.482 |
| 7 | 408.5 | 0.416 |
| 8 | 366.5 | 0.412 |

**Choice: K = 4.** The inertia drops steeply up to K=4 (1268 → 743) and much more slowly afterwards (743 → 617 → 503), so the elbow is at 4. K=4 also has the best silhouette (0.526) and gives four segments the business can act on (a premium group, a loyal group, a casual group and a dormant group). K=5 adds a split with no clearly different action.

Why K=8 always has lower inertia than K=2: inertia is the sum of squared distances to the nearest cluster centre, and every extra cluster can only keep it the same or reduce it (with one cluster per customer it would be zero). A falling inertia therefore says nothing by itself about the best K; the elbow, the silhouette and interpretability decide.

### Segment profile, names and actions (cluster numbers are only labels with no ordering)

| segment_id | Name | Size | Mean recency (days) | Mean frequency | Mean monetary (INR) | Recommended action |
|---|---|---|---|---|---|---|
| 1 | **Premium High-Spenders** (is_premium = true) | 138 | 24.12 | 6.77 | 7,129.70 | VIP early access to electronics launches and a dedicated premium support desk |
| 2 | Loyal Regulars | 419 | 8.77 | 19.40 | 3,410.02 | Reward streaks with a loyalty/subscription offer and basket-building bundles |
| 0 | Casual Occasional Shoppers | 436 | 28.86 | 3.77 | 1,057.62 | Nudge to a second weekly order with category coupons |
| 3 | Dormant / At-Risk | 182 | 109.04 | 2.20 | 685.99 | Win-back offer with free delivery; stop discount spend after two attempts |

Names are assigned by rule from the profile (highest monetary → Premium, highest recency → Dormant, highest frequency among the rest → Loyal), not from the arbitrary cluster number. `outputs/customer_segments.csv` has columns `customer_id, recency, frequency, monetary, segment_id, segment_name, is_premium`; `is_premium` is true for exactly the 138 customers of the highest-mean-monetary segment.

PCA on the scaled RFM matrix: explained variance ratio 0.634 and 0.218, total 0.852.

### Lapse problem

Cutoff 2025-09-01. Cohort = customers with at least one completed order before the cutoff. Label `lapsed` = 1 if there is no completed order from 2025-09-01 to 2025-09-30.

| Item | Value |
|---|---|
| cohort | 1,156 customers |
| lapsed | 443 (38.32%) |
| split (`test_size=0.2`, `stratify=y`, `random_state=42`) | 924 train / 232 test |
| train class counts | 570 not lapsed / 354 lapsed |
| baseline "always not lapsed", test accuracy | 61.6% |
| baseline lapsed-class F1 | 0.00 |

Features (11 raw columns): recency (days from cutoff to last completed order), frequency, monetary, average discount_pct, cancellation rate (cancelled ÷ all final-status orders before the cutoff), average delivery_minutes, favourite product category, city, age, gender, tenure in days (cutoff − signup_date).

**Leakage controls, stated explicitly:**
- No order dated 2025-09-01 or later entered any feature (the code asserts the latest order used for features is before the cutoff; the latest one is 2025-08-31 22:35:45). Orders from September are used only to build the label.
- The segment labels from Part B task 3 were **not** used as features.
- No encoder, scaler or resampler saw test rows: all preprocessing sits inside a scikit-learn `Pipeline` / `ColumnTransformer`, so it is fitted on training rows only (inside each cross-validation fold during tuning), and the oversampler touches the training set only.

### Preprocessing

Numeric features are standardised; `city`, `gender` and `fav_category` are one-hot encoded with `OneHotEncoder(drop="first")` (k−1 columns each), giving 19 encoded columns. No feature column contains a missing value (asserted in code, 0 missing before and after encoding).

Learned mean of frequency in the scaler (train rows) = **8.1147**, versus the full-cohort mean **8.0285**. They differ because the scaler is fitted only on the 924 training rows, and the 232 test rows, which it must not see, have a slightly different average.

### Tuning and evaluation

| Model | Grid | Best parameters | Best 5-fold CV F1 |
|---|---|---|---|
| LogisticRegression(max_iter=1000) | C ∈ {0.01, 0.1, 1, 10, 100} | C = 100 | 0.651 |
| RandomForest(n_estimators=100, random_state=42) | max_depth ∈ {3, 5, 8} × min_samples_leaf ∈ {5, 10, 20} (45 fits) | max_depth = 8, min_samples_leaf = 10 | 0.681 |

Untouched test set (lapsed = positive class; confusion matrix rows are actual, columns predicted: [[TN, FP], [FN, TP]]):

| Model | Confusion matrix | Precision | Recall | F1 |
|---|---|---|---|---|
| Always "not lapsed" (baseline) | [[143, 0], [89, 0]] | 0.000 | 0.000 | 0.000 |
| Logistic regression (C=100) | [[111, 32], [31, 58]] | 0.644 | 0.652 | 0.648 |
| Random forest (depth 8, leaf 10) | [[113, 30], [31, 58]] | 0.659 | 0.652 | **0.655** |

The logistic regression chose C=100 here, where the brief's reference shows C=1; the brief allows differences because feature sets may differ, and both values give a test F1 of the stated order.

**Selected model: random forest.** It has the best test F1 (0.655, above the 0.55 requirement), the best precision and the same recall as the logistic regression. Missing a customer who is about to lapse loses all of their future revenue, while wrongly sending a retention offer to a loyal customer costs only one discount or message, so recall matters more than precision here. Both models catch 58 of 89 lapsers, so the forest wins on the extra precision, and if the business wants to catch even more lapsers it can lower the decision threshold (see the balancing experiment).

### Class-balance experiment (training set only)

Lapsed rows were duplicated with replacement (`random_state=42`) until both classes had 570 rows (all original training rows kept), the best logistic regression was refitted, and evaluated on the original test set.

| Logistic regression | Confusion matrix | Precision | Recall | F1 |
|---|---|---|---|---|
| un-balanced | [[111, 32], [31, 58]] | 0.644 | 0.652 | 0.648 |
| over-sampled | [[90, 53], [13, 76]] | 0.589 | 0.854 | 0.697 |

Recall moved most (+0.20), while precision fell slightly (0.644 → 0.589) and did not rise. For the retention team this means over-sampling makes the model flag many more customers (false alarms rise from 32 to 53) but misses far fewer true lapsers (31 → 13). It is the right trade when an unneeded offer is cheap and a lost customer is expensive.

### Interpretation

Top five random-forest features (importance): frequency 0.280, monetary 0.271, recency 0.115, avg_delivery 0.061, cancel_rate 0.052. Business reading: how often and how much a customer has been buying dominates; how recently they bought comes next; slower deliveries and cancellations carry some extra signal.

Logistic regression coefficients (numeric features are standardised, so size is comparable):

| Feature | Coefficient | Reading |
|---|---|---|
| frequency | −1.698 | more orders → much less likely to lapse |
| recency | +0.040 | a longer gap since the last order slightly raises lapse risk (small, because frequency already captures most of it) |
| monetary | +0.285 | for the same number of orders, higher spend goes with slightly higher lapse risk; frequency and monetary are correlated (0.41), so read this sign with care |

Persistence: the selected pipeline is saved with `joblib.dump` to `outputs/lapse_model.joblib`; reloading with `joblib.load` reproduces the original test predictions and probabilities exactly (checked in code).

### Retention list (`outputs/retention_list.csv`)

Among the 40 test-set customers in the Premium segment, the 10 with the highest predicted lapse probability were written to the file (columns `customer_id, lapse_probability, actually_lapsed`). **1 of the 10 truly lapsed.** The top probabilities range from 0.452 to 0.491. The Premium flag comes from the whole-window RFM, which includes the September window that defines the lapse label, so it is a post-hoc whole-window label: this retention list is illustrative and not a leakage-free evaluation.

## Part C

### Input guardrails

4 empty tickets get `skipped_empty`, and 4 long tickets (each estimated at 734 tokens, above `MAX_INPUT_TOKENS = 400`) get `flagged_long`. Neither kind is ever sent to the model.

### Redaction

Applied in this order: email → `[EMAIL]`, phone → `[PHONE]`, order id → `[ORDER_ID]`, using exactly the patterns in the brief. The user message sent to the model is exactly the redacted text.

| Type | Substitutions |
|---|---|
| emails | 55 |
| phone numbers | 91 |
| order ids | 132 |
| **total** | **278** |

Re-applying the three patterns to the redacted texts finds 0. The pipeline also asserts, before every call, that neither the system prompt nor the user message contains any of the three patterns.

### Prompts

Both prompts contain `CLASSIFY`, the four categories (billing, delivery, technical, fraud) and a JSON-only reply format `{"category": "...", "reason": "..."}`. The zero-shot prompt (38 estimated tokens) has no `EXAMPLES:` line; the few-shot prompt (155 estimated tokens) has an `EXAMPLES:` line followed by four labelled examples, including a fraud example.

### Resilient calls

Every call goes through a retry wrapper: at most 3 retries, wait = `BASE_DELAY × 2**n × random.uniform(0.9, 1.15)` with `BASE_DELAY = 0.05`, one log line per retry (ticket id, retry number, wait), and `mock_llm.reset()` before each run. A full run over the 156 eligible tickets hits **19** `TransientLLMError`s, all recovered on the first retry (every retry is retry number 0; waits ranged from 0.0453 to 0.0562 s, inside 0.045–0.0575). No ticket ended with a permanent error. The retry log of the zero-shot run is in `outputs/retry_log.txt`.

### Output validation

Valid JSON with an allowed category → `classified_json`. Otherwise, lowercase words are extracted with `re.findall(r"[a-z]+", text.lower())` and the reply is accepted only if exactly one allowed category word is present → `classified_keyword`. Anything else, including valid JSON with a category outside the list (for example `shipping`), → `flagged_invalid_output` and human review; the pipeline never guesses.

Status counts for one full run (identical for zero-shot and few-shot): **135 classified_json, 12 classified_keyword, 9 flagged_invalid_output**, plus 4 skipped_empty and 4 flagged_long (164 tickets).

### Zero-shot versus few-shot (no token budget)

| Run | Tickets classified | Correct | Accuracy | Total tokens | Tokens per classified ticket |
|---|---|---|---|---|---|
| zero-shot | 147 | 126 | 85.7% | 10,336 | 70.3 |
| few-shot | 147 | 147 | 100.0% | 28,588 | 194.5 |

Zero-shot confusion table (rows = true category, columns = predicted; classified tickets only):

| true \ predicted | billing | delivery | fraud | technical |
|---|---|---|---|---|
| billing | 38 | 0 | 0 | 0 |
| delivery | 0 | 36 | 0 | 0 |
| fraud | 11 | 0 | 23 | 0 |
| technical | 10 | 0 | 0 | 29 |

**When the few-shot tokens are worth paying for.** Few-shot costs 124.2 more tokens per classified ticket (194.5 versus 70.3, about 2.8 times, or 18,252 extra tokens over the 156 tickets) and in return removes all 21 errors, raising accuracy from 85.7% to 100%. That is worth it here because all 21 zero-shot errors are fraud tickets sent to billing (11) or technical tickets sent to billing (10), and a misrouted fraud ticket misses the 2-hour security deadline. For low-stakes bulk tagging where a wrong label is cheap, the zero-shot prompt would be enough. Few-shot did not change the 9 invalid outputs, which come from the model's formatting behaviour, not from the prompt. This comparison runs against the deterministic mock, so with a real model the gap would be less clear-cut.

### Token budget, routing and outputs

**Budget demo:** zero-shot run with `TOKEN_BUDGET = 6000` → `outputs/triage_budget_demo.csv`. The run consumed 6,036 tokens, then stopped calling the model: **65 tickets ended `budget_exhausted`** (and went to human review); 79 classified_json, 7 classified_keyword and 5 flagged_invalid_output were completed first. Budget used: 6000 (no change needed).

**Final output:** `outputs/triage_results.csv` (164 rows, columns `ticket_id, customer_id, status, category, segment_name, priority, route, tokens_used`) from the few-shot run with `TOKEN_BUDGET = 100000` (28,588 tokens). Routing reads `outputs/customer_segments.csv`: statuses skipped_empty / flagged_long / flagged_invalid_output / budget_exhausted → `human_review`, NORMAL; else fraud → `security_team`, HIGH; else premium customer → `premium_desk`, HIGH; else `support_queue`, NORMAL. A customer with no segment row would be treated as not premium (this did not occur: every ticket customer had a segment row).

| Route | Priority | Tickets |
|---|---|---|
| security_team | HIGH | 34 |
| premium_desk | HIGH | 11 |
| support_queue | NORMAL | 102 |
| human_review | NORMAL | 17 (4 empty, 4 long, 9 invalid output) |

**Rule-validated versus human:** 147 tickets (135 JSON + 12 keyword-extracted) passed the validation rules and were routed automatically; 17 tickets went to human review and received no category. Category is left blank for those 17.

## Part D

### Chunking (40 words, overlap 10)

15 chunks in total: refund 4, delivery 3, escalation 3, payment 3, privacy 2. The source file name is stored with every chunk.

### Retrieval

`TfidfVectorizer()` is fitted on the chunk texts, a query is transformed with the same fitted vectorizer, and chunks are ranked by cosine similarity; `retrieve(query, k=3)` returns dictionaries with `source`, `score`, `text`. For `refund policy late delivery` all three top chunks come from `refund_policy.txt`.

TF (term frequency) measures how often a word appears in a chunk, and IDF (inverse document frequency) down-weights words that appear in many chunks so rare, informative words count more. A limitation is that TF-IDF matches words, not meaning, so a question phrased as "money back" would not match a chunk that says "refund".

### Augment and answer

`answer(question)` retrieves three chunks and sends two messages to `mock_llm.chat`: a system message containing `POLICY ASSISTANT` (answer only from the context, otherwise say the information could not be found) and a user message of the form `CONTEXT:` newline, the three chunks each prefixed with `[<source file>]` on its own line, newline `QUESTION:` newline, the question. It returns the model text and the sources.

### Retrieval and grounding evaluation (`policy_eval.csv`)

| Setting | Chunks | Answerable: expected doc in top 3 | Answerable: expected fact in answer | Unanswerable: fallback "I could not find…" |
|---|---|---|---|---|
| chunk 40, overlap 10 | 15 | 12 / 12 | 12 / 12 | 2 / 2 |
| chunk 40, overlap 0 | 13 | 12 / 12 | 10 / 12 | 2 / 2 |

With overlap 0, two questions break. "What delivery time makes a completed order eligible for a full refund?" fails because the key sentence is cut in two: the first chunk ends with "…delivered in more than 45 minutes, the customer" and the second starts "receives a full refund of the…", so the answer comes from the half without the fact. "How soon is money returned when a payment fails but is deducted?" fails the same way: one chunk ends "…returned automatically" and the next starts "within 48 hours", so the answer misses "48 hours". The 10-word overlap keeps such sentences whole in at least one chunk.

### Tools

Five tools with a name, a one-sentence description and a JSON-style parameter spec in `TOOLS_SPEC`, and a `TOOL_MAP` from name to function: `get_order` (reads `data/instamart.db`), `search_policy` (the `retrieve` function above, under key `results`), `calculate_refund` (completed order delivered in more than 45 minutes → 100% of the amount, otherwise 0), `issue_refund` and `escalate_to_human` (each appends one JSON line with timestamp and tool name to `outputs/refund_ledger.jsonl`). The agent code looks up the tool name in `TOOL_MAP` and calls the function with the arguments dictionary the model gave. Only completed orders can be refunded and refunds above INR 1000 are escalated, not issued.

### ReAct agent: seven scenarios

`run_agent(user_message, scenario_id, max_iterations=6, token_limit=6000, time_limit_seconds=10)` loops: call the model; if there is no tool call the content is the final answer, otherwise run the tool, append the assistant and tool messages, write a checkpoint, and repeat. Every run is appended to `outputs/agent_runs.jsonl` with all steps.

| Scenario | Status | Refund (INR) | Model calls | Tokens used |
|---|---|---|---|---|
| S1 | REFUND | 128.25 | 5 | 1,057 |
| S2 | REFUND | 135.00 | 5 | 1,057 |
| S3 | ESCALATE | 1,999.00 | 5 | 1,044 |
| S4 | DENIED | 0.00 | 4 | 733 |
| S5 | DENIED | 0.00 | 3 | 433 |
| S6 | NOT_FOUND | 0.00 | 2 | 136 |
| S7 | FALLBACK_MAX_ITERATIONS | 0.00 | 6 | 3,192 |

All refund amounts equal `expected_refund_inr` within 0.01. S7 uses all six model calls under the default `token_limit=6000`. The ledger `outputs/refund_ledger.jsonl` holds 3 lines (two issued refunds for S1 and S2, one escalation for S3).

### Stop safely

| Demonstration | Status | Detail |
|---|---|---|
| S1 with `token_limit=200` | **FALLBACK_TOKEN_LIMIT** | stopped after the 3rd model call (450 tokens used); `get_order` and `search_policy` had run, the requested `calculate_refund` was not executed |
| S1 with `time_limit_seconds=0` | **FALLBACK_TIMEOUT** | stopped before the first model call (0 calls, no tools run) |

Every fallback returns the steps completed so far and a short human-handoff summary in `final_answer`. Limits are checked in this order: timeout at the top of each iteration (`time.monotonic()`), then the iteration cap, then the token limit right after each model call. Results of these demonstrations are in `outputs/agent_demos.json`.

### Crash and resume

After every tool observation the agent writes `state/<run_id>.json` (run id, scenario id, full message list, tokens used, step number, tools already executed, steps). For S1 a simulated crash was raised right after the second tool execution (after its checkpoint was saved); `resume_agent("demo_S1_crash")` then loaded the checkpoint and continued. Tools executed across the crash and the resume, in order:

1. `get_order`
2. `search_policy`  (crash here, checkpoint saved)
3. `calculate_refund`  (after resume)
4. `issue_refund`  (after resume)

Four executions, none repeated. Final status REFUND, INR 128.25. The demo writes to a separate ledger (`outputs/demo_refund_ledger.jsonl`, 1 line) so the main ledger stays clean.

### ReAct versus plan-and-execute

My ReAct agent needed 5 model calls for S1, S2 and S3, 4 for S4, 3 for S5 and 2 for S6. Because each call re-sends the whole history, S1 cost 1,057 tokens, mostly from re-reading the policy search results. A plan-and-execute design asks the model for the full step list once, runs the tools without the model, and calls it again only on error. For S1 that would be about 2 model calls and a few hundred tokens (my estimate; the mock can only behave as ReAct, so I could not measure it). With an unexpected observation such as NOT_FOUND (S6) or a cancelled order (S5), ReAct reacts immediately and stops after 2 or 3 calls, whereas a fixed plan would carry on with steps that no longer make sense until the replanning rule triggers, so it needs extra checks. I would choose plan-and-execute for this refund workflow, because the steps are nearly always the same and cheap to predict, with a rule that replans on any error or unexpected status. I would keep ReAct for unusual requests.

## Limitations and Next Steps

- **Premium flag is post-hoc.** It comes from whole-window RFM that includes September, the window that defines the lapse label, so the retention list is illustrative and not a leakage-free evaluation (only 1 of its 10 customers truly lapsed). A proper evaluation would compute segments only from orders before the cutoff.
- **Synthetic data and a mock model.** All results, including the 100% few-shot accuracy and the agent behaviour, come from synthetic data and a deterministic keyword mock; real models and real tickets would be noisier.
- **One time split.** The lapse model is evaluated on a single random stratified split of 232 customers; a time-based backtest across several cutoffs and calibrated probabilities would be more reliable.
- **Imputation.** Missing delivery times of completed orders were filled with category medians, which makes some late deliveries (and refunds) invisible or invented; the 7-day refund window is assumed, not checked.
- **TF-IDF retrieval** matches words, not meaning; embeddings or query rewriting would handle paraphrases, and the fixed 40/10 chunking should respect sentence boundaries.
- **Next steps:** run the ReAct and plan-and-execute designs side by side on the same scenarios, add human-approval and audit tooling around the ledger, and schedule the pipeline with data-quality checks on every new export.
