"""Reproducible, offline stress corpus. Labels are authored assumptions, not ground truth.

python3 synthetic.py                 regenerate JSON corpus and rules report
uv run --with anthropic==2.14.0 synthetic.py --llm   evaluate full pipeline (uses API/cache)
"""
import argparse
from collections import defaultdict
from datetime import date, timedelta
import json
from pathlib import Path
import random

import classify as C

ROOT = Path(__file__).resolve().parent
DEST = ROOT / "data" / "synthetic.json"

# Explicit business types from the case taxonomy; no validation answers or model labels.
MERCHANTS = {
    "spend.groceries": ["ICA KVANTUM", "LIDL", "COOP FORUM"],
    "spend.dining": ["ESPRESSO HOUSE", "MAX HAMBURGARE", "PIZZERIA VESUVIO"],
    "spend.transport": ["SL BILJETT", "PREEM", "EASYPARK"],
    "spend.shopping": ["IKEA", "ELGIGANTEN", "ZALANDO"],
    "spend.leisure": ["SPOTIFY", "SATS", "FILMSTADEN"],
    "spend.health": ["APOTEKET", "FOLKTANDVARDEN"],
    "spend.housing": ["TELIA SVERIGE AB", "VATTENFALL"],
    "spend.gambling": ["SVENSKA SPEL", "BETSSON"],
}
FORMATS = ["{m}", "Kortköp {stamp} {m}", "KORTKOP {stamp} {m}",
           "{m} /26-10-22", "Kortköp {stamp} {m} EUR 21.98",
           "KLARNA*{m}", "PAYPAL *{m}", "ZETTLE_{m}", "SUMUP *{m}", "IZ *{m}"]


def generate(seed=20261125, repeats=12):
    rng = random.Random(seed)
    records = []
    serial = 0

    def group(scenario, specs, rationale):
        nonlocal serial
        serial += 1
        customer = f"syn-{serial:05d}"
        for spec in specs:
            account, name, typ, offset, amount, text, expected = spec
            records.append({"scenario": scenario, "group": customer, "rationale": rationale,
                            "expected": expected, "row": {
                "id": f"s{len(records)+1:06d}", "customer": customer, "bank": "Synthetic",
                "account": f"{customer}-{account}", "account_name": name, "account_type": typ,
                "date": (date(2026, 5, 10) + timedelta(days=offset)).isoformat(),
                "amount_sek": f"{amount:.2f}", "text": text}})

    def one(text, expected, amount=-399, account="main", name="Vardag", typ="checking", offset=0):
        return (account, name, typ, offset, amount, text, expected)

    for category, merchants in MERCHANTS.items():
        for merchant in merchants:
            for fmt in FORMATS:
                for sign in (-1, 1):
                    text = fmt.format(m=merchant, stamp="260509")
                    group("merchant_refund" if sign > 0 else "merchant_purchase",
                          [one(text, category, sign * rng.randint(10_00, 4500_00) / 100)],
                          "Business category survives bank/provider format and refund direction.")

    cases = [
        ("salary", [one("LÖN EXEMPEL AB", "income.employment", 32000)], "Explicit salary."),
        ("benefit", [one("FÖRSÄKRINGSKASSAN", "income.public", 1250)], "Explicit public benefit."),
        ("capital", [one("RÄNTA", "income.capital", 12.45)], "Interest income."),
        ("dividend", [one("UTDELNING", "income.capital", 450)], "Dividend income."),
        ("sale", [one("TRADERA UTBETALNING", "income.sales", 780)], "Explicit marketplace sale payout."),
        ("school", [one("Autogiro FORSKOLEAVGIFT", "spend.family", -1500)], "Explicit childcare fee."),
        ("invoice", [one("QLIRO AB", "credit.loan")], "Provider invoice, rather than a card purchase."),
        ("mortgage", [one("Autogiro BOLÅN", "credit.loan", -8500)], "Mortgage repayment."),
        ("collections", [one("BG INTRUM SVERIGE AB", "credit.collections")], "Collector bill."),
        ("cash", [one("UTTAG BANKOMAT", "transfer.cash")], "Cash withdrawal."),
        ("swish_person", [one("Swish betalning ANNA B", "transfer.person")], "Private person without message."),
        ("swish_rent", [one("Swish ANNA B hyra", "spend.housing")], "Purpose determines private Swish category."),
        ("swish_food", [one("Swish ANNA B lunch", "spend.dining", 125)], "Reimbursement keeps purpose category."),
        ("swish_company", [one("Swish ICA KVANTUM", "spend.groceries")], "Swish company retains business category."),
        ("card_purchase_trap", [one("KORTKOP 260509 VISA SHOP", "spend.shopping")], "Card purchase must not become a card bill."),
        ("card_merchant_trap", [one("Kortköp 260509 AMEX CAFE", "spend.dining")], "Card-related substring inside merchant purchase."),
        ("collector_merchant_trap", [one("Kortköp 260509 LOWELL CAFE", "spend.dining")], "Collector name inside a clearly marked cafe purchase."),
        ("cash_merchant_trap", [one("Kortköp 260509 ATM CAFE", "spend.dining")], "ATM token inside a merchant purchase is not a withdrawal."),
        ("provider_invoice_trap", [one("Kortköp 260509 QLIRO SHOP", "spend.shopping")], "Purchase marker takes precedence over provider invoice heuristic."),
        ("provider_collection_trap", [one("PAYPAL *LOWELL CAFE", "spend.dining")], "Provider routes a merchant purchase."),
        ("unknown_reference", [one("REF 84739201", "unknown")], "No evidence supports a category; abstain."),
        ("unknown_transfer", [one("Överföring 12345678", "unknown")], "No ownership or recipient evidence; abstain."),
        ("unknown_swish_message", [one("Swish ANNA B tack", "unknown")], "Message supplies no financial purpose."),
    ]
    for _ in range(repeats):
        for scenario, specs, reason in cases:
            # Vary case/spacing independently of labels, retain dates/amount semantics.
            specs = [(*s[:5], (s[5].lower() if rng.random() < .5 else s[5]), s[6]) for s in specs]
            group(scenario, specs, reason)
        for gap in (0, 5, 6):
            paired = gap <= 5
            group(f"transfer_gap_{gap}", [
                one("Överföring 11111111", "transfer.own" if paired else "unknown", -673.21),
                one("Överföring 22222222", "transfer.own" if paired else "unknown", 673.21,
                    "other", "Extra", offset=gap)], "Pair only opposite amounts on own accounts within five days.")
        group("savings_pair", [
            one("Överföring Buffert", "transfer.savings", -3250),
            one("Överföring Vardag", "transfer.savings", 3250, "save", "Buffert")],
            "Saving account purpose overrides checking type.")
        group("card_bill_pair", [one("SEB KORT BG", "credit.loan", -4246.02),
              one("INBETALNING", "transfer.own", 4246.02, "card", "Kreditkort", "credit_card", 4)],
              "Bill payment and card-side receipt have different labels.")
        group("ambiguous_pair", [one("Överföring 11111111", "unknown", -500),
              one("Överföring 22222222", "unknown", 500, "b", "Extra", offset=-1),
              one("Överföring 33333333", "unknown", 500, "d", "Tredje", offset=1)],
              "Equally close counterpart candidates do not prove ownership of a specific move.")
        group("unrelated_equal_amounts", [one("Autogiro TELIA SVERIGE AB", "spend.housing", -399),
              one("Swish ANNA B", "transfer.person", 399, "other", "Extra")],
              "Equal and opposite amounts do not establish an own-account transfer.")
    rng.shuffle(records)
    return records


def rules_predictions(rows):
    pairs, names = C.pair_transfers(rows), C.own_names(rows)
    customers = defaultdict(list)
    for row in rows:
        customers[row["customer"]].append(row)
    savings = {c: C.savings_accounts(rs) for c, rs in customers.items()}
    return {r["id"]: pairs.get(r["id"]) or C.rule(r, savings[r["customer"]], names) or "unknown"
            for r in rows}


def metrics(records, predictions):
    # Unknown is an abstention even where the authored expectation is unknown.
    n = len(records)
    correct = sum(predictions[r["row"]["id"]] == r["expected"] != "unknown" for r in records)
    labelled = sum(predictions[r["row"]["id"]] != "unknown" for r in records)
    return {"rows": n, "correct": correct, "wrong": labelled-correct, "unknown": n-labelled,
            "score": round(100*(2*correct-labelled)/n, 2),
            "precision": round(correct/labelled, 4) if labelled else None,
            "coverage": round(labelled/n, 4),
            "expectation_match": sum(predictions[r["row"]["id"]] == r["expected"] for r in records)}


def evaluate(records, predictions):
    by = defaultdict(list)
    for r in records:
        by[r["scenario"]].append(r)
    return {"overall": metrics(records, predictions),
            "by_scenario": {k: metrics(rs, predictions) for k, rs in sorted(by.items())},
            "errors": [{"id": r["row"]["id"], "scenario": r["scenario"], "text": r["row"]["text"],
                        "expected": r["expected"], "predicted": predictions[r["row"]["id"]]}
                       for r in records if predictions[r["row"]["id"]] not in (r["expected"], "unknown")]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=20261125)
    p.add_argument("--repeats", type=int, default=12)
    p.add_argument("--llm", action="store_true", help="Full classifier: uses API key and may incur cost")
    args = p.parse_args()
    if args.repeats < 1:
        p.error("--repeats must be at least 1")
    records = generate(args.seed, args.repeats)
    payload = {"version": 1, "seed": args.seed, "repeats": args.repeats,
               "source": "https://anyfin.github.io/kth-case-night/", "records": records}
    DEST.write_text(json.dumps(payload, ensure_ascii=False, indent=2)+"\n")
    rows = [r["row"] for r in records]  # expectations and rationales never enter predict()
    pred = C.predict(rows, C.read(C.TRAIN)) if args.llm else rules_predictions(rows)
    report = {"mode": "full_pipeline" if args.llm else "rules_only", "seed": args.seed,
              "synthetic_only": True, **evaluate(records, pred)}
    if C.FAILED_ROWS:
        report["api_failed_rows"] = list(C.FAILED_ROWS)
    if args.llm:
        report["model"] = C.MODEL
        report["model_calls"] = len(C.USAGE)
        report["estimated_cost_usd"] = round(C.cost_usd(), 4)
        report["cache_hits"] = C.STATS["cache_hits"]
    out = ROOT / "data" / ("synthetic-llm-report.json" if args.llm else "synthetic-rules-report.json")
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps(report["overall"], indent=2))
    print(f"Corpus: {DEST}\nReport: {out}")
    return 1 if report["overall"]["wrong"] or C.FAILED_ROWS else 0


if __name__ == "__main__":
    raise SystemExit(main())
