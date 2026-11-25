# /// script
# requires-python = ">=3.10"
# dependencies = ["anthropic==2.14.0"]
# ///
"""Anyfin case night, Track A: label bank transactions.

Usage:
  uv run classify.py data/validation.csv            -> writes predictions.csv
  uv run classify.py --eval                         -> leave-one-customer-out on training set
Env: ANTHROPIC_API_KEY (or .env file), MODEL (default claude-opus-5-5), OUT (output path).
Exit code 1 if any rows could not be classified because the API failed.
"""
import csv, json, os, re, sys, hashlib, threading, time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import date

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN = os.path.join(HERE, "data", "transactions.csv")
CACHE = os.path.join(HERE, ".cache", "llm.json")


def load_env():
    p = os.path.join(HERE, ".env")
    if os.path.exists(p):
        for line in open(p):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


load_env()  # before reading MODEL, so MODEL can be set in .env
MODEL = os.environ.get("MODEL", "claude-opus-5-5")
RETRY_SLEEP = 2.0

CATS = """income.employment: salary and wages from an employer
income.public: benefits/allowances from public bodies (Försäkringskassan, Pensionsmyndigheten, CSN payouts)
income.sales: selling own things on a marketplace (Tradera, Blocket, Vinted, Sellpy)
income.capital: interest and dividends
spend.housing: rent, electricity, internet and phone, moving costs
spend.groceries: supermarkets and food shops
spend.dining: restaurants, cafés, fast food, food delivery
spend.transport: public transport, taxis, ride-hailing, scooters, fuel, parking, bridge tolls
spend.shopping: clothes, beauty, home/furniture, electronics, DIY, online stores
spend.leisure: streaming, games, gym, cinema, events, flights, hotels, theme parks
spend.health: pharmacies, doctors, dentists
spend.family: childcare and school fees
spend.gambling: betting, casinos, lotteries (deposits and winnings)
credit.loan: repaying a loan, credit card bill, or pay-later invoice (Klarna/Qliro/Walley invoices, bolån, CSN repayment)
credit.collections: paying a debt collector or Kronofogden
transfer.own: between the customer's own everyday accounts (incl. credit card account)
transfer.savings: into or out of the customer's own savings
transfer.person: to/from a private person with no purpose stated
transfer.cash: cash withdrawals"""
VALID = {l.split(":")[0] for l in CATS.splitlines()} | {"unknown"}

RULES = """- Label by the kind of business paid, not the occasion. Groceries on holiday are groceries.
- A refund or payout takes the category of what it reverses (returned parcel = spend.shopping, casino win = spend.gambling).
- Card purchases through a payment provider (KLARNA*, ZETTLE_*, SUMUP *, IZ *, PAYPAL *, TRUSTLY*, GOOGLE *) take the category of the shop behind it.
- Paying a Klarna/Qliro/Walley invoice or a loan is credit.loan.
- Swish to a company takes the company's category. Swish to/from a private person takes the category its message names (e.g. "hyra" = spend.housing); no message = transfer.person.
- Money in of 10k+ SEK that recurs monthly from an employer, a municipality (kommun), a region or another organisation is income.employment."""


def read(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def amt(r):
    return float(r["amount_sek"])


CURRENCY = re.compile(r"\s+(EUR|DKK|NOK|USD|GBP)\s+[\d.,]+$")
SEB_DATE = re.compile(r"\s*/\d{2}-\d{2}-\d{2}$")


def norm(text):
    t = text.upper()
    t = re.sub(r"^(KORTKÖP|KORTKOP)\s+\d{6}\s+", "", t)
    t = re.sub(r"^(AUTOGIRO|AG|BETALNING BG|BG|BETALNING)\s+", "", t)
    for _ in range(2):  # either suffix can come last
        t = SEB_DATE.sub("", CURRENCY.sub("", t))
    t = re.sub(r"\b\d{4,}\b", "", t)
    return re.sub(r"\s+", " ", t).strip()


SAVE_NAME = re.compile(r"SPAR|BUFFERT|RÄNTE|RANTE|RESERV|KAPITAL", re.I)
TRANSFER_TXT = re.compile(r"ÖVERF|OVERF|TRANSFER", re.I)
CARD_TXT = re.compile(r"KREDITKORT|\bKORT\b|MASTERCARD|\bVISA\b|AMEX|EUROCARD", re.I)
NOT_TRANSFER = re.compile(r"^(KORTKÖP|KORTKOP|SWISH)|\b(UTTAG|BANKOMAT|ATM)\b", re.I)


def own_names(rows):
    """customer -> {ACCOUNT NAME: account id}; names shorter than 4 letters are too ambiguous to match in text."""
    out = defaultdict(dict)
    for r in rows:
        if len(r["account_name"]) >= 4:
            out[r["customer"]][r["account_name"].upper()] = r["account"]
    return out


def mentioned_account(r, names):
    """Another of this customer's accounts named in the text, or None."""
    up = r["text"].upper()
    for name, acc in names.get(r["customer"], {}).items():
        if acc != r["account"] and name in up:
            return acc
    return None


def transferish(r, names):
    """Text that looks like a move between accounts or a card-bill payment (never a purchase, Swish or ATM)."""
    t = r["text"]
    if NOT_TRANSFER.search(t):
        return False
    return bool(TRANSFER_TXT.search(t) or CARD_TXT.search(t) or re.search(r"INBETALNING", t, re.I)
                or mentioned_account(r, names))


def savings_accounts(rows):
    """Savings by type or name; by behaviour only with real history (>= 4 rows, >= 90% transfers or interest)."""
    by = defaultdict(list)
    for r in rows:
        by[r["account"]].append(r)
    out = set()
    for acc, rs in by.items():
        r0 = rs[0]
        if r0["account_type"] == "savings" or SAVE_NAME.search(r0["account_name"]):
            out.add(acc)
        elif r0["account_type"] != "credit_card" and len(rs) >= 4:
            tr = sum(bool(TRANSFER_TXT.search(r["text"])) or norm(r["text"]) in ("RÄNTA", "RANTA") for r in rs)
            if tr / len(rs) >= 0.9:
                out.add(acc)
    return out


def pair_transfers(rows):
    """Match own-account moves: same customer, both rows transfer-like text, different accounts,
    opposite equal amount, within 5 days. Ambiguous matches (two equally close candidates) are skipped."""
    res = {}
    names = own_names(rows)
    by_c = defaultdict(list)
    for r in rows:
        by_c[r["customer"]].append(r)
    for c, rs in by_c.items():
        sav = savings_accounts(rs)
        types = {r["account"]: r["account_type"] for r in rs}
        cand = [r for r in rs if transferish(r, names)]
        used = set()

        def options(x):
            """Unused rows that could be x's other half, closest first."""
            dx = date.fromisoformat(x["date"])
            out = []
            for y in cand:
                if y["id"] in used or y["account"] == x["account"] or abs(amt(y) + amt(x)) > 0.005:
                    continue
                gap = abs((date.fromisoformat(y["date"]) - dx).days)
                if gap <= 5:
                    out.append((gap, y))
            return sorted(out, key=lambda o: o[0])

        def unique_nearest(opts):
            if not opts or (len(opts) > 1 and opts[1][0] == opts[0][0]):
                return None
            return opts[0][1]

        for a in cand:
            if a["id"] in used or amt(a) >= 0:
                continue
            b = unique_nearest(options(a))
            # Pair only mutual, unique nearest matches; anything ambiguous goes to the other rules.
            if b is None or unique_nearest(options(b)) is not a:
                continue
            used |= {a["id"], b["id"]}
            if a["account"] in sav or b["account"] in sav:
                res[a["id"]] = res[b["id"]] = "transfer.savings"
            elif types[b["account"]] == "credit_card":
                res[a["id"]], res[b["id"]] = "credit.loan", "transfer.own"
            elif types[a["account"]] == "credit_card":
                res[a["id"]], res[b["id"]] = "transfer.own", "transfer.own"
            else:
                res[a["id"]] = res[b["id"]] = "transfer.own"
    return res


SWISH = re.compile(r"^SWISH(?:\s+(?:BETALNING|INBETALNING|TILL|FRÅN|FRAN))?\s+(.*)$", re.I)


def swish_parts(text):
    m = SWISH.match(text.strip())
    if not m:
        return None
    toks = m.group(1).split()
    if len(toks) >= 2 and len(toks[1]) == 1 and toks[1].isalpha():
        return ("person", " ".join(toks[2:]))
    return ("company", m.group(1))


# Swish messages whose category is unambiguous (rule 6 on the case page). Anything else goes to the model.
SWISH_MSG = {"HYRA": "spend.housing", "EL": "spend.housing", "ELRÄKNING": "spend.housing",
             "MATKASSE": "spend.groceries", "PIZZA": "spend.dining", "LUNCH": "spend.dining",
             "MIDDAG": "spend.dining", "FIKA": "spend.dining", "BILJETTER": "spend.leisure",
             "BILJETT": "spend.leisure", "KONSERT": "spend.leisure", "BIO": "spend.leisure"}


def rule(r, sav, names):
    t, u, a, typ = r["text"], r["text"].upper(), amt(r), r["account_type"]
    n = norm(t)
    sw = swish_parts(t)
    if sw:
        if sw[0] == "person":
            msg = sw[1].upper().strip()
            return SWISH_MSG.get(msg) if msg else "transfer.person"
        return None  # company -> LLM
    if re.search(r"\b(UTTAG|BANKOMAT|ATM)\b", u):
        return "transfer.cash"
    if a > 0 and re.match(r"^LÖN\b|^LON\b", n):
        return "income.employment"
    if a > 0 and re.match(r"^(RÄNTA|RANTA|UTDELNING)\b", n):
        return "income.capital"
    if a > 0 and re.search(r"FÖRSÄKRINGSKASSAN|FORSAKRINGSKASSAN|PENSIONSMYNDIGHETEN", u):
        return "income.public"
    if re.search(r"INTRUM|KRONOFOGD|LOWELL|INKASSO|ALEKTUM|SVEA EKONOMI", u):
        return "credit.collections"
    if a < 0 and re.match(r"^(KLARNA AB|KLARNA BANK|QLIRO|WALLEY)", n):
        return "credit.loan"
    if typ == "credit_card" and a > 0 and re.match(r"^(INBETALNING|BETALNING)", n):
        return "transfer.own"
    if typ != "credit_card" and a < 0 and CARD_TXT.search(n) and not re.match(r"(?i)kortköp", t):
        return "credit.loan"
    if TRANSFER_TXT.search(u):
        # Only call it an own-account move with evidence of ownership; otherwise let the model decide.
        other = mentioned_account(r, names)
        if r["account"] in sav or (other and other in sav):
            return "transfer.savings"
        if other:
            return "transfer.own"
        if SAVE_NAME.search(u):
            return "transfer.savings"
    return None


# ---------------- LLM ----------------
_lock = threading.Lock()
USAGE = []
FAILED_ROWS = []
PRICE = {"claude-opus-5-5": (4.0, 20.0), "claude-sonnet-5-5": (2.0, 10.0), "claude-haiku-5-5": (0.10, 0.50)}
STATS = Counter()


def cost_usd():
    pin, pout = PRICE.get(MODEL, (4.0, 20.0))
    tok_in = sum(u.input_tokens + 1.25 * (u.cache_creation_input_tokens or 0)
                 + 0.1 * (u.cache_read_input_tokens or 0) for u in USAGE)
    return (tok_in * pin + sum(u.output_tokens for u in USAGE) * pout) / 1e6


def load_cache():
    try:
        return json.load(open(CACHE))
    except Exception:
        return {}


def save_cache(c):
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    json.dump(c, open(CACHE, "w"), ensure_ascii=False)


def size(a):
    a = abs(a)
    return "<100" if a < 100 else "100-999" if a < 1000 else "1k-9.9k" if a < 10000 else "10k+"


def recurring(rows):
    """(customer, merchant key, direction) seen in 3+ different months."""
    months = defaultdict(set)
    for r in rows:
        months[(r["customer"], norm(r["text"]), amt(r) > 0)].add(r["date"][:7])
    return {k for k, m in months.items() if len(m) >= 3}


def item_of(r, rec):
    """What the model sees. Private persons' names are replaced, so they never leave the machine."""
    sw = swish_parts(r["text"])
    if sw and sw[0] == "person":
        text = f"SWISH PRIVATE PERSON {sw[1].upper()}".strip()
    elif sw:
        text = f"SWISH {sw[1].upper()}"
    else:
        text = norm(r["text"])
    return {"text": text, "dir": "in" if amt(r) > 0 else "out", "acct": r["account_type"], "size": size(amt(r)),
            "monthly": (r["customer"], norm(r["text"]), amt(r) > 0) in rec}


def key_of(it):
    return hashlib.sha1(json.dumps(it, sort_keys=True).encode()).hexdigest()[:16]


def render(it):
    return f"{it['text']} | {it['dir']} {it['size']} SEK | {it['acct']}" + (" | monthly" if it["monthly"] else "")


def build_system(examples):
    ex = "\n".join(f"{render(e)} -> {c}" for e, c in examples)
    return f"""You label Swedish bank transactions with exactly one category.

Categories:
{CATS}

Rules:
{RULES}

Scoring: +1 correct, -1 wrong, 0 for "unknown". Only give a category if you are more likely right than wrong; otherwise answer "unknown".
Use your knowledge of Swedish and international merchants. Store/city suffixes (GBG, STHLM, MALMÖ, branch names) are not part of the business type.
Each line is: text | money in/out and size | account type | "monthly" if it recurs in 3+ months.

Labelled examples:
{ex}"""


def classify_batch(client, system, batch):
    lines = "\n".join(f"{i}. {render(it)}" for i, it in enumerate(batch))
    schema = {
        "type": "object",
        "properties": {"items": {"type": "array", "items": {
            "type": "object",
            "properties": {"i": {"type": "integer"}, "category": {"type": "string", "enum": sorted(VALID)},
                           "confidence": {"type": "string", "enum": ["high", "medium", "low"]}},
            "required": ["i", "category", "confidence"], "additionalProperties": False}}},
        "required": ["items"], "additionalProperties": False,
    }
    resp = client.messages.create(
        model=MODEL, max_tokens=16000,
        system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        output_config={"effort": "low", "format": {"type": "json_schema", "schema": schema}},
        messages=[{"role": "user", "content": f"Label each transaction:\n{lines}"}],
    )
    with _lock:
        USAGE.append(resp.usage)
    if resp.stop_reason in ("refusal", "max_tokens"):
        raise RuntimeError(f"stop_reason={resp.stop_reason}")
    text = next(b.text for b in resp.content if b.type == "text")
    out = {}
    for x in json.loads(text)["items"]:
        if 0 <= x["i"] < len(batch):
            c = x["category"] if x["confidence"] != "low" else "unknown"
            out[x["i"]] = c if c in VALID else "unknown"
    return out


def llm_label(items, examples):
    """Returns key -> category, or key -> None where the API failed (not the same as a deliberate "unknown")."""
    import anthropic
    client = anthropic.Anthropic(max_retries=6)
    cache = load_cache()
    system = build_system(examples)
    sk = hashlib.sha1((MODEL + system).encode()).hexdigest()[:8]
    todo = {}
    for it in items:
        k = sk + key_of(it)
        if k not in cache:
            todo[k] = it
    keys = list(todo)
    STATS["unique_items"] += len({key_of(it) for it in items})
    STATS["cache_hits"] += len({key_of(it) for it in items}) - len(keys)
    batches = [keys[i:i + 60] for i in range(0, len(keys), 60)]
    failed = set()

    def run(bkeys):
        """Retry a batch up to 3 times; items missing from a response count as failed and are retried too."""
        remaining = list(bkeys)
        for attempt in range(3):
            try:
                res = classify_batch(client, system, [todo[k] for k in remaining])
            except (anthropic.AuthenticationError, anthropic.PermissionDeniedError,
                    anthropic.NotFoundError, anthropic.BadRequestError):
                raise  # configuration problem: stop the whole run
            except Exception as e:
                print(f"batch failed ({type(e).__name__}: {e}); attempt {attempt + 1}/3", file=sys.stderr)
                time.sleep(RETRY_SLEEP * (attempt + 1))
                continue
            with _lock:
                for i, k in enumerate(remaining):
                    if i in res:
                        cache[k] = res[i]
            missing = [k for i, k in enumerate(remaining) if i not in res]
            if not missing:
                return
            print(f"response missing {len(missing)} of {len(remaining)} items; attempt {attempt + 1}/3", file=sys.stderr)
            remaining = missing
        with _lock:
            failed.update(remaining)

    if batches:
        print(f"LLM: {len(keys)} unique items in {len(batches)} batches ({MODEL})", file=sys.stderr)
        with ThreadPoolExecutor(8) as ex:
            list(ex.map(run, batches))
        save_cache(cache)
    out = {}
    for it in items:
        k = sk + key_of(it)
        out[key_of(it)] = None if k in failed else cache.get(k, "unknown")
    return out


def predict(rows, train_rows):
    names = own_names(rows)
    pairs = pair_transfers(rows)
    sav_by_c = {}
    by_c = defaultdict(list)
    for r in rows:
        by_c[r["customer"]].append(r)
    for c, rs in by_c.items():
        sav_by_c[c] = savings_accounts(rs)
    pred, need = {}, []
    for r in rows:
        c = pairs.get(r["id"]) or rule(r, sav_by_c[r["customer"]], names)
        if c:
            pred[r["id"]] = c
        else:
            need.append(r)
    STATS["rows"] += len(rows)
    STATS["rule_rows"] += len(rows) - len(need)
    # few-shot examples: training rows that rules don't handle, deduped
    seen, examples = set(), []
    tp = pair_transfers(train_rows)
    tnames = own_names(train_rows)
    trec = recurring(train_rows)
    tsav = defaultdict(set)
    for c in {r["customer"] for r in train_rows}:
        tsav[c] = savings_accounts([r for r in train_rows if r["customer"] == c])
    for r in train_rows:
        if r["id"] in tp or rule(r, tsav[r["customer"]], tnames):
            continue
        it = item_of(r, trec)
        k = key_of(it)
        if k not in seen:
            seen.add(k)
            examples.append((it, r["category"]))
    rec = recurring(rows)
    labels = llm_label([item_of(r, rec) for r in need], examples) if need else {}
    for r in need:
        c = labels.get(key_of(item_of(r, rec)), "unknown")
        if c is None:
            FAILED_ROWS.append(r["id"])
            c = "unknown"
        pred[r["id"]] = c
    return pred


def score(pred, rows):
    right = sum(pred.get(r["id"]) == r["category"] for r in rows)
    lab = sum(pred.get(r["id"], "unknown") != "unknown" for r in rows)
    n = len(rows)
    return {"score": 100 * (right - (lab - right)) / n, "acc": right / n,
            "prec": right / lab if lab else 0.0, "cov": lab / n}


def fmt(s):
    return f"score {s['score']:5.1f}  acc {s['acc']:.1%}  prec {s['prec']:.1%}  cov {s['cov']:.1%}"


def main():
    train = read(TRAIN)
    if "--eval" in sys.argv:
        allp, unseen = {}, []
        for c in sorted({r["customer"] for r in train}):
            test = [r for r in train if r["customer"] == c]
            rest = [r for r in train if r["customer"] != c]
            p = predict([{k: v for k, v in r.items() if k != "category"} for r in test], rest)
            print(f"{c}: {fmt(score(p, test))}")
            allp.update(p)
            known = {norm(r["text"]) for r in rest}
            unseen += [r for r in test if norm(r["text"]) not in known]
        print(f"ALL: {fmt(score(allp, train))}")
        print(f"unseen merchant keys ({len(unseen)} rows): {fmt(score(allp, unseen))}")
        print(f"rules labelled {STATS['rule_rows'] / STATS['rows']:.1%} of rows; "
              f"{len(USAGE)} model calls, ~${cost_usd():.3f}; cache hits {STATS['cache_hits']}/{STATS['unique_items']}")
        wrong = [(r["text"], r["category"], allp[r["id"]]) for r in train if allp[r["id"]] not in (r["category"], "unknown")]
        for w in wrong[:60]:
            print("  WRONG", w)
        return 1 if FAILED_ROWS else 0
    t0 = time.time()
    rows = read(sys.argv[1])
    p = predict(rows, train)
    out = os.environ.get("OUT", "predictions.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "category"])
        for r in rows:
            w.writerow([r["id"], p.get(r["id"], "unknown")])
    unknown = sum(p.get(r["id"], "unknown") == "unknown" for r in rows)
    print(f"wrote {out} ({len(rows)} rows, {unknown} unknown) in {time.time() - t0:.0f}s; "
          f"rules {STATS['rule_rows']}/{len(rows)} rows; model: {STATS['unique_items']} unique items, "
          f"{STATS['cache_hits']} from cache, {len(USAGE)} calls, ~${cost_usd():.3f}", file=sys.stderr)
    if FAILED_ROWS:
        print(f"ERROR: {len(FAILED_ROWS)} rows could not be classified because the API failed; "
              f"they are 'unknown' in {out}. Re-run to retry them (successful answers are cached).", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
